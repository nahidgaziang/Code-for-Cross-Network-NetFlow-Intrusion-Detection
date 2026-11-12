"""
Identifier-free features and the target-label budget.

  e4      KNN-win, IF-win, ERM-CB, XGB, RF and BRN on the benchmark folds with identifier-like
          columns removed (L4_SRC_PORT, L4_DST_PORT, DNS_QUERY_ID, ICMP_TYPE, ICMP_IPV4_TYPE,
          FTP_COMMAND_RET_CODE); benchmark-selected hyperparameters.
  e3      k in {10, 30, 100, 300, 1000} labelled target flows, drawn uniformly (natural prior),
          nested across k and disjoint from the evaluation, buffer and window flows.
          XGB-T: target labels only. XGB-ST: source flows plus target flows, target total weight equal
          to source total weight, class-balanced, early stopping on source validation.
  report  -> results/{e4,e3}/report.txt
  python experiments/feature_label_studies.py e4|e3|report
"""
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.data import TESTBEDS, FoldConfig, build_fold, load_pool  # noqa: E402
from crossnet.methods import run_method  # noqa: E402
from crossnet.preprocess import GlobalScaler, log_features  # noqa: E402
from crossnet.runner import EVAL_SEEDS, append, done_keys  # noqa: E402

BENCH = os.path.join(HERE, "..", "results", "benchmark")
RES = os.environ.get("FL_RES", os.path.join(HERE, "..", "results"))
if os.environ.get("FL_SEEDS"):
    EVAL_SEEDS = [int(s) for s in os.environ["FL_SEEDS"].split(",")]
if os.environ.get("FL_TARGETS"):
    _keep = os.environ["FL_TARGETS"].split(",")
    TESTBEDS = {f: [t for t in ts if any(k in t for k in _keep)] for f, ts in TESTBEDS.items()}

ID_COLS = ("L4_SRC_PORT", "L4_DST_PORT", "DNS_QUERY_ID", "ICMP_TYPE", "ICMP_IPV4_TYPE", "FTP_COMMAND_RET_CODE")
E4_METHODS = ("ERM-CB", "XGB", "RF", "BRN", "KNN-win", "IF-win")
E3_K = (10, 30, 100, 300, 1000)


def selected(fam, method):
    return json.load(open(os.path.join(BENCH, f"selected_{fam}__{method}.json")))


def auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def benchmark_labels(fam, target, method, seed):
    return np.load(os.path.join(BENCH, f"scores_{fam}", f"{target}__{method}__{seed}.npz"))["y_eval"].astype(int)


# ------------------------------------------------------------------------------------------- E4
def run_e4():
    out = os.path.join(RES, "e4"); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "e4_runs.csv")
    done = done_keys(path, ["family", "target", "seed", "method"])
    for fam in ("nfv2", "nfv3"):
        load_pool.cache_clear()
        for seed in EVAL_SEEDS:
            for target in TESTBEDS[fam]:
                todo = [m for m in E4_METHODS if (fam, target, str(seed), m) not in done]
                if not todo:
                    continue
                inp, truth, _ = build_fold(FoldConfig(family=fam, target=target, seed=seed))
                y = truth.y_eval.astype(int)
                keep = [i for i, f in enumerate(inp.features) if f not in ID_COLS]
                assert len(inp.features) - len(keep) == len(ID_COLS), "identifier columns not all found"
                sub = replace(inp, **{k: getattr(inp, k)[:, keep] for k in ("X_tr", "X_va", "X_buf", "X_win", "X_eval")},
                              features=[inp.features[i] for i in keep])
                for method in todo:
                    assert np.array_equal(benchmark_labels(fam, target, method, seed), y), "fold differs from E1"
                    t0 = time.time()
                    p, _ = run_method(method, sub, selected(fam, method)[target], seed=seed)
                    append(path, {"family": fam, "target": target, "seed": seed, "method": method,
                                  "n_features": len(keep), "AUROC": auroc(y, p["eval"]),
                                  "seconds": round(time.time() - t0, 1)})
                print(f"[e4] {fam} {target} seed {seed} done", flush=True)


# ------------------------------------------------------------------------------------------- E3
def run_e3():
    out = os.path.join(RES, "e3"); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "e3_runs.csv")
    done = done_keys(path, ["family", "target", "seed", "k", "method"])
    for fam in ("nfv2", "nfv3"):
        load_pool.cache_clear()
        for seed in EVAL_SEEDS:
            for target in TESTBEDS[fam]:
                if all((fam, target, str(seed), str(k), m) in done for k in E3_K for m in ("XGB-T", "XGB-ST")):
                    continue
                inp, truth, meta = build_fold(FoldConfig(family=fam, target=target, seed=seed))
                y = truth.y_eval.astype(int)
                assert np.array_equal(benchmark_labels(fam, target, "XGB", seed), y), "fold differs from E1"
                tu = load_pool(fam, target, "uniform")
                used = np.concatenate([meta["rows_eval"], meta["rows_buf"], meta["rows_win"]])
                free = np.where(~np.isin(tu["row"], used))[0]
                order = np.random.default_rng([seed, TESTBEDS[fam].index(target), 3]).permutation(free)
                sc = GlobalScaler().fit(log_features(inp.X_tr))
                tr, va, ev = (sc.transform(log_features(a)) for a in (inp.X_tr, inp.X_va, inp.X_eval))
                hp = selected(fam, "XGB")[target]
                for k in E3_K:
                    idx = order[:k]
                    Xl, yl = sc.transform(log_features(tu["X"][idx])), tu["y"][idx].astype(int)
                    base = {"family": fam, "target": target, "seed": seed, "k": k, "k_attacks": int(yl.sum())}
                    for method in ("XGB-T", "XGB-ST"):
                        if (fam, target, str(seed), str(k), method) in done:
                            continue
                        t0 = time.time()
                        if method == "XGB-T":
                            if 0 < yl.sum() < k:
                                n1 = yl.sum()
                                m = xgb.XGBClassifier(**hp, tree_method="hist", device="cuda", random_state=seed,
                                                      scale_pos_weight=(k - n1) / n1, verbosity=0)
                                m.fit(Xl, yl)
                                s, flag = m.predict_proba(ev)[:, 1], ""
                            else:
                                s, flag = np.full(len(ev), 0.5, np.float32), "single_class"
                        else:
                            Xc, yc = np.concatenate([tr, Xl]), np.concatenate([inp.y_tr, yl])
                            dom = np.r_[np.ones(len(tr)), np.full(k, len(tr) / k)]
                            cls = np.where(yc == 1, len(yc) / (2 * max(yc.sum(), 1)), len(yc) / (2 * max((yc == 0).sum(), 1)))
                            m = xgb.XGBClassifier(**hp, tree_method="hist", device="cuda", random_state=seed,
                                                  eval_metric="logloss", early_stopping_rounds=30, verbosity=0)
                            m.fit(Xc, yc, sample_weight=dom * cls, eval_set=[(va, inp.y_va)], verbose=False)
                            s, flag = m.predict_proba(ev)[:, 1], ""
                        append(path, {**base, "method": method, "AUROC": auroc(y, s), "flag": flag,
                                      "seconds": round(time.time() - t0, 1)})
                print(f"[e3] {fam} {target} seed {seed} done", flush=True)


# ------------------------------------------------------------------------------------------- report
def run_report():
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(20261104)

    def wil(d):
        d = np.asarray(d, float)
        return 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)

    def benchmark_auroc(fam, method):
        d = pd.read_csv(os.path.join(BENCH, f"eval_{fam}__{method}.csv")).drop_duplicates(["target", "seed"])
        return d[["target", "seed", "AUROC"]]

    L = []
    p4 = os.path.join(RES, "e4", "e4_runs.csv")
    if os.path.exists(p4):
        d = pd.read_csv(p4)
        L += ["E4 IDENTIFIER-FREE FEATURES (mean AUROC over 4 targets x seeds)", ""]
        full = pd.concat([benchmark_auroc(f, m).assign(family=f, method=m) for f in ("nfv2", "nfv3") for m in E4_METHODS])
        mm = d.merge(full, on=["family", "target", "seed", "method"], suffixes=("_idfree", "_full"))
        L.append(mm.groupby(["family", "method"])[["AUROC_full", "AUROC_idfree"]].mean().round(3).to_string())
        L.append("\nH-I2 (descriptive): AUROC change id-free - full, paired, seed-bootstrap 95% CI")
        for (fam, m), g in mm.groupby(["family", "method"]):
            g = g.assign(d=g.AUROC_idfree - g.AUROC_full)
            piv = g.pivot_table(index="seed", columns="target", values="d").values
            boots = np.nanmean(piv[rng.integers(0, len(piv), (10000, len(piv)))], axis=(1, 2))
            L.append(f"   {fam} {m:8s} {g.d.mean():+.4f} [{np.percentile(boots, 2.5):+.4f}, "
                     f"{np.percentile(boots, 97.5):+.4f}] p {wil(g.d):.2g}")
        L.append("\nH-I1: id-free KNN-win vs best id-free source-trained method (ERM-CB, XGB, RF)")
        verdict = []
        for fam in ("nfv2", "nfv3"):
            g = d[d.family == fam]
            src = g[g.method.isin(["ERM-CB", "XGB", "RF"])].groupby("method").AUROC.mean()
            best = src.idxmax()
            x = g[g.method == "KNN-win"].merge(g[g.method == best], on=["target", "seed"])
            diff = x.AUROC_x - x.AUROC_y
            ok = diff.mean() >= 0.10 and wil(diff) < 0.05
            verdict.append(ok)
            L.append(f"   {fam}: KNN-win {x.AUROC_x.mean():.4f} vs {best} {x.AUROC_y.mean():.4f}: d {diff.mean():+.4f}, "
                     f"p {wil(diff):.2g}, n={len(diff)}")
        L.append(f"   => H-I1 {'HOLDS' if all(verdict) else 'FAILS'}\n")
    p3 = os.path.join(RES, "e3", "e3_runs.csv")
    if os.path.exists(p3):
        d = pd.read_csv(p3)
        L += ["E3 LABEL BUDGET (mean AUROC over 4 targets x seeds; KNN-win = benchmark 500-flow unlabelled window)", ""]
        L.append(d.pivot_table(index=["family", "method"], columns="k", values="AUROC").round(3).to_string())
        L.append("single-class draws (XGB-T constant score): " +
                 d[d.flag == "single_class"].groupby(["family", "k"]).size().to_string().replace("\n", "; "))
        verdict = []
        for fam in ("nfv2", "nfv3"):
            knn = benchmark_auroc(fam, "KNN-win")
            knn = knn[knn.seed.isin(d.seed.unique()) & knn.target.isin(d[d.family == fam].target.unique())]
            ref = knn.AUROC.mean()
            best_k = d[d.family == fam].pivot_table(index=["target", "seed"], columns=["k", "method"],
                                                     values="AUROC")
            means = {k: max(best_k[(k, m)].mean() for m in ("XGB-T", "XGB-ST")) for k in E3_K}
            rate = next((k for k in E3_K if means[k] >= ref), "> 1000")
            L.append(f"H-L1 {fam}: KNN-win reference {ref:.4f}; best S3 mean by k "
                     f"{ {k: round(v, 3) for k, v in means.items()} } -> exchange rate k* = {rate}")
            k100 = best_k[[(100, "XGB-T"), (100, "XGB-ST")]]
            mbest = "XGB-T" if k100[(100, "XGB-T")].mean() >= k100[(100, "XGB-ST")].mean() else "XGB-ST"
            x = k100[(100, mbest)].rename("s3").reset_index().merge(knn, on=["target", "seed"])
            diff = x.s3 - x.AUROC
            ok = diff.mean() < 0 and wil(diff) < 0.05
            verdict.append(ok)
            L.append(f"H-L2 {fam}: k=100 {mbest} {x.s3.mean():.4f} vs KNN-win {x.AUROC.mean():.4f}: d {diff.mean():+.4f}, "
                     f"p {wil(diff):.2g}, n={len(diff)}")
        L.append(f"   => H-L2 {'HOLDS' if all(verdict) else 'FAILS'}")
    for sub in ("e4", "e3"):
        if os.path.isdir(os.path.join(RES, sub)):
            open(os.path.join(RES, sub, "report.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"e4": run_e4, "e3": run_e3, "report": run_report}[sys.argv[1]]()
