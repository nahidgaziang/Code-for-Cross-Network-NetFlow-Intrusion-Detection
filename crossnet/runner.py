"""
Resumable experiment runner for the leave-one-testbed-out benchmark.

A job is (release, method). Each job tunes, selects and evaluates, and writes its own files under
results/benchmark/ (override with NF_RESULTS):
  tune_{release}__{method}.csv, selected_{release}__{method}.json, eval_{release}__{method}.csv,
  scores_{release}/{target}__{method}__{seed}.npz
Several worker processes can run at once: a worker claims a job by atomically creating
locks/{release}__{method}.lock. Finished rows are skipped on restart.

Tuning: each method's fixed candidate list (12 configurations) is scored by leave-one-source-out
validation on the fold's source testbeds only: train on two sources and use the third as a
pseudo-target with its own buffer, window and evaluation split, then rotate. The configuration with
the highest mean validation AUROC is selected (tie-break PR-AUC). No target flow is used. Seed 0.

Evaluation: the selected configuration is frozen and run on ten evaluation seeds. Probabilistic
methods are scored under the fixed, em, win_mad and win_fpr rules; anomaly scorers under win_mad and
win_fpr. All scores are saved, so further decision rules can be evaluated offline.

  python -m crossnet.runner --jobs nfv2:ERM-CB nfv3:KNN-win ...  [--budget 12 --seeds ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import data as data_mod
from .data import FoldConfig, MethodInput, TESTBEDS, build_fold
from .decision import decide
from .methods import METHODS, run_method, sample_configs
from .metrics import evaluate

TUNE_SEED = 0
EVAL_SEEDS = [42, 1337, 2025, 2026, 2027, 7, 11, 101, 2024, 31337]
OUT = os.environ.get("NF_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results", "benchmark"))



def done_keys(path, cols):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(map(tuple, df[cols].astype(str).values))


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def rules_for(method):
    return ("fixed", "em", "win_mad", "win_fpr") if METHODS[method][4] else ("win_mad", "win_fpr")


def loso_inputs(inp, rng, n_buf=1500, n_win=500, win_attack_frac=0.05):
    """Pseudo-target splits built from SOURCE testbeds only (unchanged since G0)."""
    X = np.concatenate([inp.X_tr, inp.X_va]); y = np.concatenate([inp.y_tr, inp.y_va])
    d = np.concatenate([inp.d_tr, inp.d_va]); is_tr = np.r_[np.ones(len(inp.y_tr), bool), np.zeros(len(inp.y_va), bool)]
    for h in np.unique(d):
        idx = rng.permutation(np.where(d == h)[0])
        buf, rest = idx[:n_buf], idx[n_buf:]
        n_att = int(round(win_attack_frac * n_win))
        win = np.concatenate([rest[y[rest] == 0][:n_win - n_att], rest[y[rest] == 1][:n_att]])
        ev = np.setdiff1d(rest, win)
        keep_tr, keep_va = is_tr & (d != h), ~is_tr & (d != h)
        yield int(h), MethodInput(X_tr=X[keep_tr], y_tr=y[keep_tr], d_tr=d[keep_tr],
                                  X_va=X[keep_va], y_va=y[keep_va], d_va=d[keep_va],
                                  X_buf=X[buf], X_win=X[win], X_eval=X[ev]), y[ev]


def tune(family, method, targets, budget, fold_kw):
    path = os.path.join(OUT, f"tune_{family}__{method}.csv")
    done = done_keys(path, ["target", "config_id", "held_out"])
    for target in targets:
        inp, _, _ = build_fold(FoldConfig(family=family, target=target, seed=TUNE_SEED, **fold_kw))
        splits = list(loso_inputs(inp, np.random.default_rng(TUNE_SEED)))
        for cid, hp in enumerate(sample_configs(method, budget, seed=TUNE_SEED)):
            for h, sinp, y_ev in splits:
                if (target, str(cid), str(h)) in done:
                    continue
                t0 = time.time()
                p, _ = run_method(method, sinp, hp, seed=TUNE_SEED)
                f1 = np.nan
                if METHODS[method][4]:
                    yhat, _ = decide("em", p["val"], sinp.y_va, p["eval"], p["buf"])
                    f1 = evaluate(y_ev, p["eval"], yhat)["MacroF1"]
                m = evaluate(y_ev, p["eval"], np.zeros(len(y_ev), np.int8))
                append(path, {"family": family, "target": target, "method": method, "config_id": cid,
                              "held_out": h, "AUROC": m["AUROC"], "PRAUC": m["PRAUC"], "MacroF1_em": f1,
                              "seconds": round(time.time() - t0, 1), "hp": json.dumps(hp)})
        print(f"[tune] {family} {method} {target} done", flush=True)
    return select(family, method)


def select(family, method):
    df = pd.read_csv(os.path.join(OUT, f"tune_{family}__{method}.csv"))
    agg = df.groupby(["target", "config_id", "hp"])[["AUROC", "PRAUC"]].mean().reset_index()
    best = agg.sort_values(["AUROC", "PRAUC", "config_id"], ascending=[False, False, True]).groupby("target").head(1)
    sel = {r.target: json.loads(r.hp) for r in best.itertuples()}
    json.dump(sel, open(os.path.join(OUT, f"selected_{family}__{method}.json"), "w"), indent=1)
    return sel


def evaluate_stage(family, method, targets, seeds, fold_kw):
    sel = json.load(open(os.path.join(OUT, f"selected_{family}__{method}.json")))
    path = os.path.join(OUT, f"eval_{family}__{method}.csv")
    rules = rules_for(method)
    done = done_keys(path, ["target", "seed", "rule"])
    stats_dir = os.path.join(data_mod.POOL_DIR, family)
    sdir = os.path.join(OUT, f"scores_{family}")
    os.makedirs(sdir, exist_ok=True)
    for seed in seeds:
        for target in targets:
            if (target, str(seed), rules[0]) in done:
                continue
            names = {int(k): v for k, v in json.load(open(os.path.join(stats_dir, f"{target}_stats.json")))
                     ["attack_code_names"].items()}
            inp, truth, _ = build_fold(FoldConfig(family=family, target=target, seed=seed, **fold_kw), names)
            t0 = time.time()
            hp = sel[target]
            p, info = run_method(method, inp, hp, seed=seed)
            secs = round(time.time() - t0, 1)
            np.savez_compressed(os.path.join(sdir, f"{target}__{method}__{seed}.npz"),
                                p_val=np.asarray(p["val"], np.float32), y_val=inp.y_va.astype(np.int8),
                                p_buf=np.asarray(p["buf"], np.float32), p_win=np.asarray(p["win"], np.float32),
                                p_eval=np.asarray(p["eval"], np.float32), y_eval=truth.y_eval.astype(np.int8),
                                attack_eval=truth.attack_eval, win_is_attack=truth.win_is_attack.astype(np.int8))
            src_f1 = (evaluate(inp.y_va, p["val"], (p["val"] >= 0.5).astype(np.int8))["MacroF1"]
                      if METHODS[method][4] else np.nan)
            msg = []
            for rule in rules:
                yhat, dinfo = decide(rule, p["val"], inp.y_va, p["eval"], p["buf"], p["win"], METHODS[method][4])
                m = evaluate(truth.y_eval, p["eval"], yhat, truth.attack_eval, truth.attack_names)
                recalls = {k[7:-1]: round(v, 6) for k, v in m.items() if k.startswith("recall[")}
                m = {k: v for k, v in m.items() if not k.startswith("recall[")}
                m["recall_by_attack"] = json.dumps(recalls)
                msg.append(f"{rule} {m['MacroF1']:.4f}")
                append(path, {"family": family, "target": target, "method": method, "setting": METHODS[method][3],
                              "seed": seed, "rule": rule, **m, "SrcVal_MacroF1": src_f1, "seconds": secs,
                              "hp": json.dumps(hp), "info": json.dumps(info),
                              "decision": json.dumps({k: round(v, 4) for k, v in dinfo.items()})})
            print(f"[eval] {family} seed {seed} {target} {method}: AUROC {m['AUROC']:.4f} | MacroF1 "
                  f"{', '.join(msg)} ({secs:.0f}s)", flush=True)


def claim(job):
    lock_dir = os.path.join(OUT, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    try:
        fd = os.open(os.path.join(lock_dir, job.replace(":", "__") + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True, help="family:method entries, claimed in order")
    ap.add_argument("--stage", default="all", choices=["tune", "eval", "all"])
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    ap.add_argument("--n_src", type=int, default=100_000)
    ap.add_argument("--n_eval", type=int, default=100_000)
    ap.add_argument("--no_lock", action="store_true", help="run every listed job without claiming (tests)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fold_kw = {"n_src": a.n_src, "n_eval": a.n_eval}
    current_family = None
    for job in a.jobs:
        family, method = job.split(":")
        assert family in TESTBEDS and method in METHODS, job
        if not a.no_lock and not claim(job):
            continue
        if family != current_family:  # keep at most one family's pools in memory
            data_mod.load_pool.cache_clear()
            current_family = family
        targets = [t for t in a.targets if t in TESTBEDS[family]] if a.targets else TESTBEDS[family]
        t0 = time.time()
        print(f"==== JOB START {job} pid {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        try:
            if a.stage in ("tune", "all"):
                tune(family, method, targets, a.budget, fold_kw)
            if a.stage in ("eval", "all"):
                evaluate_stage(family, method, targets, a.seeds, fold_kw)
            print(f"==== JOB DONE {job} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
        except Exception:  # log and continue with the next job; a restart resumes this one
            import traceback
            traceback.print_exc()
            print(f"==== JOB FAILED {job} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
    print(f"==== WORKER DONE pid {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
