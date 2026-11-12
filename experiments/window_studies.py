"""
Stress tests for the window detectors (setting S2). Window detectors need no source training, so
these runs are cheap. Hyperparameters are the ones selected in the benchmark (leave-one-source-out
on the sources; no target information).

  e5t     NF-v3 temporal window: the window is drawn from the first 20% (or 50%) of the target
          timeline (benign flows plus 5% attack flows, all with start time <= cut, disjoint from the
          evaluation and buffer flows) and evaluated on benchmark evaluation flows that start after
          the cut. Comparators on the same late flows: the benchmark's random-time window and every
          S0/S1 method (from saved benchmark scores).
  e9      window size {50, 100, 200, 500, 1000, 2000} at 5% contamination, and contamination
          {0, 5, 10, 20, 30}% at size 500; both releases, four targets, ten seeds. Folds are built
          with the benchmark's fold function, so evaluation streams are identical to the benchmark.
  report  hypothesis tests and tables -> results/{e5t,e9}/report.txt
  python experiments/window_studies.py e5t|e9|report
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from dataclasses import replace
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.data import TESTBEDS, FoldConfig, build_fold, load_pool  # noqa: E402
from crossnet.decision import decide  # noqa: E402
from crossnet.methods import METHODS, run_method  # noqa: E402
from crossnet.metrics import evaluate  # noqa: E402
from crossnet.runner import EVAL_SEEDS, append, done_keys  # noqa: E402

BENCH = os.path.join(HERE, "..", "results", "benchmark")  # benchmark outputs (read-only)
# smoke-test overrides: WS_RES (output root), WS_SEEDS ("42,1337"), WS_TARGETS (substring filter, "UNSW,ToN")
RES = os.environ.get("WS_RES", os.path.join(HERE, "..", "results"))
if os.environ.get("WS_SEEDS"):
    EVAL_SEEDS = [int(s) for s in os.environ["WS_SEEDS"].split(",")]
if os.environ.get("WS_TARGETS"):
    _keep = os.environ["WS_TARGETS"].split(",")
    TESTBEDS = {f: [t for t in ts if any(k in t for k in _keep)] for f, ts in TESTBEDS.items()}
WIN_METHODS = ("KNN-win", "IF-win")
E5T_CUTS = (0.2, 0.5)
E9_GRID = [(n, 0.05) for n in (50, 100, 200, 500, 1000, 2000)] + [(500, c) for c in (0.0, 0.10, 0.20, 0.30)]


def selected(fam, method):
    return json.load(open(os.path.join(BENCH, f"selected_{fam}__{method}.json")))


def metrics(y, s, p_win, probabilistic=False):
    out = {"AUROC": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan}
    for rule in ("win_mad", "win_fpr"):
        yhat, _ = decide(rule, None, None, s, None, p_win, probabilistic)
        out[f"MacroF1_{rule}"] = evaluate(y, s, yhat)["MacroF1"]
    m = evaluate(y, s, np.zeros(len(y), np.int8))
    out["TPR_at_FPR0.01"] = m["TPR_at_FPR0.01"]
    return out


def eval_t_start(fam, target, rows_eval):
    tu = load_pool(fam, target, "uniform")
    idx = pd.Index(tu["row"]).get_indexer(rows_eval)
    assert (idx >= 0).all(), "eval rows not found in pool"
    return tu["t_start"][idx].astype(np.float64)


# ------------------------------------------------------------------------------------------- E5-T
def run_e5t():
    out = os.path.join(RES, "e5t"); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "e5t_runs.csv")
    done = done_keys(path, ["target", "seed", "cut", "method", "window"])
    fam = "nfv3"
    for seed in EVAL_SEEDS:
        for target in TESTBEDS[fam]:
            if all((target, str(seed), str(c), m, w) in done for c in E5T_CUTS for m in WIN_METHODS
                   for w in ("early", "random")):
                continue
            inp, truth, meta = build_fold(FoldConfig(family=fam, target=target, seed=seed))
            y_all = truth.y_eval.astype(int)
            t_eval = eval_t_start(fam, target, meta["rows_eval"])
            tu, tb = load_pool(fam, target, "uniform"), load_pool(fam, target, "benign")
            used = np.union1d(meta["rows_eval"], meta["rows_buf"])
            for cut_q in E5T_CUTS:
                cut = float(np.quantile(tu["t_start"].astype(np.float64), cut_q))
                late = t_eval > cut
                y = y_all[late]
                rng = np.random.default_rng([seed, TESTBEDS[fam].index(target), 5, int(cut_q * 100)])
                ben = np.where((tb["t_start"] <= cut) & ~np.isin(tb["row"], used))[0]
                att = np.where((tu["y"] == 1) & (tu["t_start"] <= cut) & ~np.isin(tu["row"], used))[0]
                n_att = min(int(round(0.05 * 500)), len(att))
                pb, pa = rng.choice(ben, 500 - n_att, replace=False), rng.choice(att, n_att, replace=False)
                X_win = np.concatenate([tb["X"][pb], tu["X"][pa]])[rng.permutation(500)]
                for method in WIN_METHODS:
                    hp = selected(fam, method)[target]
                    t0 = time.time()
                    p, _ = run_method(method, replace(inp, X_win=X_win), hp, seed=seed)
                    base = {"family": fam, "target": target, "seed": seed, "cut": cut_q, "method": method,
                            "n_late": int(late.sum()), "n_late_benign": int((y == 0).sum()),
                            "late_attack_share": float(y.mean()), "win_attacks": n_att}
                    append(path, {**base, "window": "early", **metrics(y, p["eval"][late], p["win"]),
                                  "seconds": round(time.time() - t0, 1)})
                    z = np.load(os.path.join(BENCH, f"scores_{fam}", f"{target}__{method}__{seed}.npz"))
                    assert np.array_equal(z["y_eval"].astype(int), y_all), "benchmark evaluation stream mismatch"
                    append(path, {**base, "window": "random",
                                  **metrics(y, z["p_eval"][late], z["p_win"]), "seconds": 0})
            print(f"[e5t] {target} seed {seed} done", flush=True)
    # S0/S1 comparators on the same late flows (saved benchmark scores; no new training)
    cpath = os.path.join(out, "e5t_s01_late.csv")
    rows = []
    for seed in EVAL_SEEDS:
        for target in TESTBEDS[fam]:
            inp, truth, meta = build_fold(FoldConfig(family=fam, target=target, seed=seed))
            t_eval = eval_t_start(fam, target, meta["rows_eval"])
            tu = load_pool(fam, target, "uniform")
            for cut_q in E5T_CUTS:
                late = t_eval > float(np.quantile(tu["t_start"].astype(np.float64), cut_q))
                for method in METHODS:
                    if METHODS[method][3] not in ("S0", "S1"):
                        continue
                    z = np.load(os.path.join(BENCH, f"scores_{fam}", f"{target}__{method}__{seed}.npz"))
                    y = z["y_eval"].astype(int)[late]
                    rows.append({"family": fam, "target": target, "seed": seed, "cut": cut_q, "method": method,
                                 "AUROC": roc_auc_score(y, z["p_eval"][late]) if 0 < y.sum() < len(y) else np.nan})
    pd.DataFrame(rows).to_csv(cpath, index=False)
    print("[e5t] comparators written", flush=True)


# ------------------------------------------------------------------------------------------- E9
def run_e9():
    out = os.path.join(RES, "e9"); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "e9_runs.csv")
    done = done_keys(path, ["family", "target", "seed", "n_win", "contamination", "method"])
    for fam in ("nfv2", "nfv3"):
        load_pool.cache_clear()
        for seed in EVAL_SEEDS:
            for target in TESTBEDS[fam]:
                for n_win, c in E9_GRID:
                    keys = [(fam, target, str(seed), str(n_win), str(c), m) for m in WIN_METHODS]
                    if all(k in done for k in keys):
                        continue
                    inp, truth, _ = build_fold(FoldConfig(family=fam, target=target, seed=seed,
                                                          n_win=n_win, win_attack_frac=c))
                    y = truth.y_eval.astype(int)
                    for method in WIN_METHODS:
                        t0 = time.time()
                        p, _ = run_method(method, inp, selected(fam, method)[target], seed=seed)
                        r = {"family": fam, "target": target, "seed": seed, "n_win": n_win, "contamination": c,
                             "method": method, "win_attacks": int(truth.win_is_attack.sum()),
                             **metrics(y, p["eval"], p["win"]), "seconds": round(time.time() - t0, 1),
                             "E1_AUROC_diff": np.nan}  # fixed column set (CSV schema)
                        if (n_win, c) == (500, 0.05):  # regression: must equal the benchmark
                            z = np.load(os.path.join(BENCH, f"scores_{fam}", f"{target}__{method}__{seed}.npz"))
                            r["E1_AUROC_diff"] = abs(r["AUROC"] - roc_auc_score(z["y_eval"], z["p_eval"]))
                        append(path, r)
                print(f"[e9] {fam} {target} seed {seed} done", flush=True)


# ------------------------------------------------------------------------------------------- report
def run_report():
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(20261103)

    def wil(d):
        d = np.asarray(d, float)
        return 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)

    def boot(df, col):
        piv = df.pivot_table(index="seed", columns="target", values=col).values
        idx = rng.integers(0, len(piv), size=(10000, len(piv)))
        m = np.nanmean(piv[idx], axis=(1, 2))
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    L = []
    e5 = os.path.join(RES, "e5t", "e5t_runs.csv")
    if os.path.exists(e5):
        d = pd.read_csv(e5); c = pd.read_csv(os.path.join(RES, "e5t", "e5t_s01_late.csv"))
        L += ["E5-T TEMPORAL WINDOW (NF-v3): window from the first CUT of the timeline, metrics on late flows", ""]
        for cut in E5T_CUTS:
            dc, cc = d[d.cut == cut], c[c.cut == cut]
            L.append(f"--- cut = {cut:.0%} {'(PRIMARY)' if cut == 0.2 else '(sensitivity)'}")
            L.append(dc.groupby("target")[["n_late", "n_late_benign", "late_attack_share", "win_attacks"]]
                     .first().to_string())
            tab = dc.pivot_table(index="target", columns=["method", "window"], values="AUROC").round(3)
            s01 = cc.groupby(["target", "method"]).AUROC.mean().unstack()
            tab[("best S0/S1", s01.mean().idxmax())] = s01[s01.mean().idxmax()].round(3)
            L.append(tab.to_string())
            keep = dc.groupby("target").n_late_benign.min()
            ok = keep[keep >= 100].index.tolist()
            L.append(f"targets in the primary test (>= 100 late benign flows): {ok}")
            s01k = cc[cc.target.isin(ok)]
            best = s01k.groupby("method").AUROC.mean().idxmax()
            knn = dc[(dc.method == "KNN-win") & (dc.window == "early") & dc.target.isin(ok)]
            m = knn[["target", "seed", "AUROC"]].merge(s01k[s01k.method == best][["target", "seed", "AUROC"]],
                                                       on=["target", "seed"], suffixes=("_knn", "_best"))
            diff = m.AUROC_knn - m.AUROC_best
            verdict = diff.mean() >= 0.10 and wil(diff) < 0.05
            L.append(f"H-T1{' (primary)' if cut == 0.2 else ' (sensitivity)'}: KNN-win early "
                     f"{m.AUROC_knn.mean():.4f} vs best S0/S1 {best} {m.AUROC_best.mean():.4f}: "
                     f"d {diff.mean():+.4f}, p {wil(diff):.2g}, n={len(diff)} -> {'HOLDS' if verdict else 'FAILS'}")
            for meth in WIN_METHODS:
                e = dc[(dc.method == meth)].pivot_table(index=["target", "seed"], columns="window", values="AUROC")
                e = e[e.index.get_level_values(0).isin(ok)]
                dd = (e.early - e.random).reset_index().rename(columns={0: "d"})
                dd["d"] = e.early.values - e.random.values
                lo, hi = boot(dd, "d")
                L.append(f"H-T2 {meth}: early - random window AUROC {dd.d.mean():+.4f} "
                         f"[95% CI {lo:+.4f}, {hi:+.4f}], p {wil(dd.d):.2g}")
            L.append("")
    e9 = os.path.join(RES, "e9", "e9_runs.csv")
    if os.path.exists(e9):
        d = pd.read_csv(e9)
        main = pd.read_csv(os.path.join(BENCH, "analysis", "main_auroc.csv"))
        L += ["E9 WINDOW CURVES (mean AUROC over 4 targets x 10 seeds)", ""]
        if "E1_AUROC_diff" in d:
            L.append(f"regression vs benchmark at (500, 5%): max |AUROC diff| = {d.E1_AUROC_diff.max():.2e}")
        for name, sub, col in (("size (5% contamination)", d[d.contamination == 0.05], "n_win"),
                               ("contamination (size 500)", d[d.n_win == 500], "contamination")):
            L.append(f"--- by {name}: AUROC")
            L.append(sub.groupby(["family", "method", col]).AUROC.mean().unstack(col).round(3).to_string())
            L.append(f"--- by {name}: Macro-F1 under win_mad")
            L.append(sub.groupby(["family", "method", col]).MacroF1_win_mad.mean().unstack(col).round(3).to_string())
        bench = []
        for fam in ("nfv2", "nfv3"):
            m = main[(main.family == fam) & main.setting.isin(["S0", "S1"])].set_index("method").AUROC
            best = m.idxmax()
            ev = pd.read_csv(os.path.join(BENCH, f"eval_{fam}__{best}.csv")).drop_duplicates(["target", "seed"])
            for hname, (n, c) in (("H-W1", (100, 0.05)), ("H-W2", (500, 0.20))):
                k = d[(d.family == fam) & (d.method == "KNN-win") & (d.n_win == n) & (d.contamination == c)]
                mm = k[["target", "seed", "AUROC"]].merge(ev[["target", "seed", "AUROC"]], on=["target", "seed"],
                                                          suffixes=("_knn", "_best"))
                diff = mm.AUROC_knn - mm.AUROC_best
                bench.append((hname, fam, n, c, best, mm.AUROC_knn.mean(), mm.AUROC_best.mean(), diff.mean(),
                              wil(diff), len(diff), diff.mean() >= 0.10 and wil(diff) < 0.05))
        L.append("")
        for h in ("H-W1", "H-W2"):
            rows = [b for b in bench if b[0] == h]
            for b in rows:
                L.append(f"{h} {b[1]} (size {b[2]}, contamination {b[3]:.0%}): KNN-win {b[5]:.4f} vs best "
                         f"benchmark S0/S1 {b[4]} {b[6]:.4f}: d {b[7]:+.4f}, p {b[8]:.2g}, n={b[9]}")
            L.append(f"   => {h} {'HOLDS' if all(b[10] for b in rows) and len(rows) == 2 else 'FAILS'}")
    for sub in ("e5t", "e9"):
        if os.path.isdir(os.path.join(RES, sub)):
            open(os.path.join(RES, sub, "report.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"e5t": run_e5t, "e9": run_e9, "report": run_report}[sys.argv[1]]()
