"""
From the saved benchmark scores: AUROC, standardised partial AUROC for FPR <= 1% and TPR at 1% and 0.1% FPR
for every run of every method; and two fixed fusion rules (average rank percentile of KNN-win with RF, and
with the release's best S0/S1 method).
  python analysis/low_fpr_and_fusion.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.methods import METHODS  # noqa: E402
from crossnet.runner import EVAL_SEEDS  # noqa: E402
from crossnet.data import TESTBEDS  # noqa: E402

B = os.path.join(HERE, "..", "results", "benchmark")
OUT = os.path.join(B, "analysis")
RELEASE_BEST = {"nfv2": "IRM", "nfv3": "RF"}


def metrics(y, s):
    ben, att = s[y == 0], s[y == 1]
    return {"AUROC": float(roc_auc_score(y, s)), "pAUC01": float(roc_auc_score(y, s, max_fpr=0.01)),
            "TPR_at_FPR0.01": float(np.mean(att > np.quantile(ben, 0.99))),
            "TPR_at_FPR0.001": float(np.mean(att > np.quantile(ben, 0.999)))}


def load(fam, target, method, seed):
    with np.load(os.path.join(B, f"scores_{fam}", f"{target}__{method}__{seed}.npz")) as z:
        return z["y_eval"].astype(int), z["p_eval"].astype(np.float64)


def main():
    low, fus = [], []
    for fam in ("nfv2", "nfv3"):
        for target in TESTBEDS[fam]:
            for seed in EVAL_SEEDS:
                sc = {}
                for m in METHODS:
                    y, s = load(fam, target, m, seed)
                    sc[m] = s
                    low.append({"family": fam, "target": target, "seed": seed, "method": m, **metrics(y, s)})
                rk = {m: rankdata(sc[m]) / len(y) for m in ("KNN-win", "RF", RELEASE_BEST[fam])}
                for rule, other in (("KNN+RF", "RF"), ("KNN+best", RELEASE_BEST[fam])):
                    fus.append({"family": fam, "target": target, "seed": seed, "rule": rule, "partner": other,
                                **metrics(y, (rk["KNN-win"] + rk[other]) / 2)})
            print(f"{fam} {target} done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(low).to_csv(os.path.join(OUT, "low_fpr.csv"), index=False)
    pd.DataFrame(fus).to_csv(os.path.join(OUT, "fusion.csv"), index=False)


if __name__ == "__main__":
    main()
