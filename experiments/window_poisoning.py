"""
Targeted poisoning of the S2 reference window.

An attacker controls a fraction p in {0, 1, 2, 5, 10, 20}% of the window (N = 500) and fills it with
flows of one attack family (the target's most frequent family), aiming to blind the detector to that
family. Folds are the benchmark folds, so the evaluation stream is identical to the benchmark. The
benign part comes from the benign pool and the injected part from the uniform pool, both outside the
evaluation and buffer sets and nested across p.
Defences: naive (window as given) and trimmed (remove the 10% of window flows with the highest
leave-one-out window score, then rerun the detector).
Per run (KNN-win, IF-win): AUROC, targeted-family and other-family AUROC against target benign flows,
recall and benign FPR at the declared operating point (win_mad, Z = 3), and targeted-family recall at a
matched benign FPR of 1% and 5%. All evaluation and window scores are saved.
  python experiments/window_poisoning.py run|report
"""
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.data import POOL_DIR, TESTBEDS, FoldConfig, build_fold, load_pool  # noqa: E402
from crossnet.decision import window_mad_decisions  # noqa: E402
from crossnet.methods import run_method  # noqa: E402
from crossnet.runner import EVAL_SEEDS, append, done_keys  # noqa: E402

BENCH = os.path.join(HERE, "..", "results", "benchmark")
RES = os.environ.get("POISON_RES", os.path.join(HERE, "..", "results", "poisoning"))
if os.environ.get("POISON_SEEDS"):
    EVAL_SEEDS = [int(s) for s in os.environ["POISON_SEEDS"].split(",")]
if os.environ.get("POISON_TARGETS"):
    _keep = os.environ["POISON_TARGETS"].split(",")
    TESTBEDS = {f: [t for t in ts if any(k in t for k in _keep)] for f, ts in TESTBEDS.items()}
METHODS = ("KNN-win", "IF-win")
LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
N_WIN, TRIM, Z = 500, 0.10, 3.0


def selected(fam, method):
    return json.load(open(os.path.join(BENCH, f"selected_{fam}__{method}.json")))


def top_family(fam, target):
    st = json.load(open(os.path.join(POOL_DIR, fam, f"{target}_stats.json")))
    names = {int(k): v for k, v in st["attack_code_names"].items()}
    counts = {k: v for k, v in st["attack_counts"].items() if k.lower() != "benign"}
    top = max(counts, key=counts.get)
    return [c for c, n in names.items() if n == top][0], top


def auroc(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def tpr_at_fpr(neg, pos, fpr):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    return float(np.mean(pos > np.quantile(neg, 1 - fpr)))


def metrics(s, s_win, y, is_tgt, is_oth):
    ben = s[y == 0]
    yhat, _ = window_mad_decisions(s, s_win, probabilistic=False, z=Z)
    return {"AUROC": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
            "AUROC_target_family": auroc(ben, s[is_tgt]), "AUROC_other_families": auroc(ben, s[is_oth]),
            "TPR_target_z3": float(yhat[is_tgt].mean()) if is_tgt.any() else np.nan,
            "TPR_other_z3": float(yhat[is_oth].mean()) if is_oth.any() else np.nan,
            "FPR_benign_z3": float(yhat[y == 0].mean()),
            "MacroF1_z3": float(f1_score(y, yhat, average="macro", zero_division=0)),
            "TPR_target_at_FPR1": tpr_at_fpr(ben, s[is_tgt], 0.01),
            "TPR_target_at_FPR5": tpr_at_fpr(ben, s[is_tgt], 0.05)}


def run():
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "poisoning_runs.csv")
    done = done_keys(path, ["family", "target", "seed", "p_poison", "method", "defense"])
    for fam in ("nfv2", "nfv3"):
        load_pool.cache_clear()
        sdir = os.path.join(RES, f"scores_{fam}")
        os.makedirs(sdir, exist_ok=True)
        for seed in EVAL_SEEDS:
            for target in TESTBEDS[fam]:
                keys = [(fam, target, str(seed), str(p), m, d) for p in LEVELS for m in METHODS for d in ("naive", "trimmed")]
                if all(k in done for k in keys):
                    continue
                code, fname = top_family(fam, target)
                inp, truth, meta = build_fold(FoldConfig(family=fam, target=target, seed=seed))
                y, att = truth.y_eval.astype(int), truth.attack_eval
                z1 = np.load(os.path.join(BENCH, f"scores_{fam}", f"{target}__KNN-win__{seed}.npz"))
                assert np.array_equal(z1["y_eval"].astype(int), y), "fold differs from the benchmark"
                is_tgt, is_oth = (y == 1) & (att == code), (y == 1) & (att != code)
                np.savez_compressed(os.path.join(sdir, f"{target}__{seed}__labels.npz"), y_eval=y.astype(np.int8),
                                    attack_eval=att, target_code=code)
                tu, tb = load_pool(fam, target, "uniform"), load_pool(fam, target, "benign")
                used = np.union1d(meta["rows_eval"], meta["rows_buf"])
                rng = np.random.default_rng([seed, TESTBEDS[fam].index(target), 4242])
                ben = rng.permutation(np.where(~np.isin(tb["row"], used))[0])
                poi = rng.permutation(np.where((tu["attack"] == code) & ~np.isin(tu["row"], used))[0])
                for p in LEVELS:
                    n = int(round(p * N_WIN))
                    assert len(poi) >= n and len(ben) >= N_WIN - n
                    X_win = np.concatenate([tb["X"][ben[:N_WIN - n]], tu["X"][poi[:n]]])
                    is_poison = np.r_[np.zeros(N_WIN - n, bool), np.ones(n, bool)]
                    order = rng.permutation(N_WIN)
                    X_win, is_poison = X_win[order], is_poison[order]
                    for method in METHODS:
                        hp = selected(fam, method)[target]
                        t0 = time.time()
                        pn, _ = run_method(method, replace(inp, X_win=X_win), hp, seed=seed)
                        keep = np.argsort(pn["win"], kind="stable")[: N_WIN - int(round(TRIM * N_WIN))]
                        pt, _ = run_method(method, replace(inp, X_win=X_win[keep]), hp, seed=seed)
                        purged = int(is_poison.sum() - is_poison[keep].sum())
                        for dname, pr in (("naive", pn), ("trimmed", pt)):
                            if (fam, target, str(seed), str(p), method, dname) in done:
                                continue
                            np.savez_compressed(os.path.join(sdir, f"{target}__{seed}__{p}__{method}__{dname}.npz"),
                                                p_eval=np.asarray(pr["eval"], np.float32), p_win=np.asarray(pr["win"], np.float32))
                            append(path, {"family": fam, "target": target, "seed": seed, "p_poison": p, "n_poison": n,
                                          "target_family": fname, "method": method, "defense": dname,
                                          "purged_poison": purged if dname == "trimmed" else 0,
                                          **metrics(np.asarray(pr["eval"], np.float64), np.asarray(pr["win"], np.float64),
                                                    y, is_tgt, is_oth),
                                          "seconds": round(time.time() - t0, 1)})
                print(f"[poisoning] {fam} {target} seed {seed} done", flush=True)


def run_report():
    from scipy.stats import wilcoxon
    d = pd.read_csv(os.path.join(RES, "poisoning_runs.csv"))
    L = ["WINDOW POISONING (benchmark folds; declared operating point win_mad Z = 3)", ""]
    cols = ["AUROC", "AUROC_target_family", "AUROC_other_families", "TPR_target_z3", "FPR_benign_z3",
            "TPR_target_at_FPR1", "TPR_target_at_FPR5"]
    for (fam, m), g in d.groupby(["family", "method"]):
        L.append(f"--- {fam} {m}")
        L.append(g.groupby(["defense", "p_poison"])[cols].mean().round(4).to_string())
    for fam in ("nfv2", "nfv3"):
        k = d[(d.family == fam) & (d.method == "KNN-win")]
        piv = k.pivot_table(index=["target", "seed"], columns=["defense", "p_poison"], values="AUROC_target_family")
        drop = piv[("naive", 0.0)] - piv[("naive", 0.1)]
        rec = piv[("trimmed", 0.1)] - piv[("naive", 0.1)]
        oth = k.pivot_table(index=["target", "seed"], columns=["defense", "p_poison"], values="AUROC_other_families")
        odrop = (oth[("naive", 0.0)] - oth[("naive", 0.1)]).dropna()
        ok4 = drop.mean() >= 0.10 and wilcoxon(drop).pvalue < 0.05 and abs(odrop.mean()) < 0.05
        L.append(f"\nH-P4 {fam}: targeted-family AUROC drop at 10% = {100 * drop.mean():.2f} pp (p {wilcoxon(drop).pvalue:.2g}, "
                 f"n={len(drop)}); other-family change {100 * -odrop.mean():+.2f} pp -> {'HOLDS' if ok4 else 'FAILS'}")
        share = rec.mean() / drop.mean()
        L.append(f"H-P5 {fam}: trimming recovers {100 * rec.mean():.2f} pp = {100 * share:.0f}% of the drop "
                 f"(p {wilcoxon(rec).pvalue:.2g}) -> {'HOLDS' if share >= 0.5 else 'FAILS'}")
    text = "\n".join(L)
    open(os.path.join(RES, "report.txt"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    {"run": run, "report": run_report}[sys.argv[1]]()
