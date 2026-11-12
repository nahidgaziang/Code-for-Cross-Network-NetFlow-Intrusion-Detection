"""
Exact-duplicate check for the window detectors: reports the share of evaluation flows whose feature
vector equals a window flow, and the AUROC of saved scores on the de-duplicated evaluation flows.
  python analysis/duplicate_check.py [n_seeds]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.data import TESTBEDS, FoldConfig, build_fold  # noqa: E402
from crossnet.runner import EVAL_SEEDS  # noqa: E402

RES = os.path.join(HERE, "..", "results", "benchmark")
OUT = os.path.join(RES, "analysis")
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
METHODS = ("KNN-win", "IF-win", "BRN", "ERM-CB")
os.makedirs(OUT, exist_ok=True)


def row_keys(X):
    X = np.ascontiguousarray(X)
    return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()


rows = []
for fam in ("nfv2", "nfv3"):
    for target in TESTBEDS[fam]:
        for seed in EVAL_SEEDS[:N_SEEDS]:
            inp, truth, _ = build_fold(FoldConfig(family=fam, target=target, seed=seed))
            y = truth.y_eval.astype(int)
            dup = np.isin(row_keys(inp.X_eval), row_keys(inp.X_win))
            r = {"family": fam, "target": target.replace("NF-", ""), "seed": seed,
                 "dup_share_benign": dup[y == 0].mean(), "dup_share_attack": dup[y == 1].mean(),
                 "n_dedup": int((~dup).sum()), "attack_share_dedup": y[~dup].mean()}
            for m in METHODS:
                z = np.load(os.path.join(RES, f"scores_{fam}", f"{target}__{m}__{seed}.npz"))
                assert np.array_equal(z["y_eval"].astype(int), y), f"fold mismatch {fam} {target} {seed} {m}"
                s = z["p_eval"].astype(np.float64)
                r[f"AUROC_all[{m}]"] = roc_auc_score(y, s)
                yd = y[~dup]
                r[f"AUROC_dedup[{m}]"] = roc_auc_score(yd, s[~dup]) if 0 < yd.sum() < len(yd) else np.nan
            rows.append(r)
            print(f"{fam} {target} seed {seed}: dup benign {r['dup_share_benign']:.2f} attack "
                  f"{r['dup_share_attack']:.2f} | KNN-win AUROC all {r['AUROC_all[KNN-win]']:.3f} "
                  f"dedup {r['AUROC_dedup[KNN-win]']:.3f}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "window_duplicates.csv"), index=False)
cols = ["dup_share_benign", "dup_share_attack", "attack_share_dedup"] + \
       [f"AUROC_{k}[{m}]" for m in METHODS for k in ("all", "dedup")]
rep = ["Exact-duplicate check: eval flows whose feature vector equals a window flow (mean over seeds)",
       d.groupby(["family", "target"])[cols].mean().round(3).T.to_string(), "",
       "Mean over targets and seeds:", d.groupby("family")[cols].mean().round(3).T.to_string()]
open(os.path.join(OUT, "window_duplicates_report.txt"), "w", encoding="utf-8").write("\n".join(rep))
print("\n".join(rep))
