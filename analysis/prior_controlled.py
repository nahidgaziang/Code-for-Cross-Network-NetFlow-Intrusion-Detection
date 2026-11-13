"""
Prior-controlled split of the threshold-irrecoverable Macro-F1 loss (saved benchmark scores; no training).

The best achievable Macro-F1 depends on the class prior even for an unchanged ROC curve. For every run of
the 16 probabilistic methods, the source-validation scores are evaluated at the target's class mix
(classes reweighted to the target prior). The gap between source Macro-F1 and the target's best-threshold
Macro-F1 then splits into a prior effect and a ranking effect at the target prior.
  python analysis/prior_controlled.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.methods import METHODS  # noqa: E402

RES = os.path.join(HERE, "..", "results", "benchmark")
OUT = os.path.join(RES, "analysis")


def best_macro_f1(y, s, w=None):
    """Exact maximum (weighted) Macro-F1 over all thresholds; cuts only between distinct scores."""
    w = np.ones(len(y)) if w is None else w
    order = np.argsort(-s, kind="stable")
    ys, ss, ws = y[order].astype(np.int64), s[order], w[order]
    P, N = float((ws * ys).sum()), float((ws * (1 - ys)).sum())
    tp = np.concatenate([[0.0], np.cumsum(ws * ys)])
    fp = np.concatenate([[0.0], np.cumsum(ws * (1 - ys))])
    fn, tn = P - tp, N - fp
    f1_att = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-300), 0.0)
    f1_ben = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1e-300), 0.0)
    valid = np.r_[True, ss[1:] != ss[:-1], True]
    return float((0.5 * (f1_att + f1_ben))[valid].max())


def main():
    rows = []
    for fam in ("nfv2", "nfv3"):
        for method, spec in METHODS.items():
            if not spec[4]:
                continue
            ev = pd.read_csv(os.path.join(RES, f"eval_{fam}__{method}.csv"))
            src = ev[ev.rule == "em"].set_index(["target", "seed"]).SrcVal_MacroF1
            for f in sorted(glob.glob(os.path.join(RES, f"scores_{fam}", f"*__{method}__*.npz"))):
                target, _, seed = os.path.basename(f)[:-4].split("__")
                with np.load(f) as z:
                    pv, yv = z["p_val"].astype(np.float64), z["y_val"].astype(int)
                    pe, ye = z["p_eval"].astype(np.float64), z["y_eval"].astype(int)
                pi_s, pi_t = yv.mean(), ye.mean()
                w = np.where(yv == 1, pi_t / pi_s, (1 - pi_t) / (1 - pi_s))
                rows.append({"family": fam, "target": target, "method": method, "seed": int(seed),
                             "prior_src": pi_s, "prior_tgt": pi_t,
                             "F_S": float(src.loc[(target, int(seed))]), "F_S_best": best_macro_f1(yv, pv),
                             "F_S_prior": best_macro_f1(yv, pv, w), "F_T_best": best_macro_f1(ye, pe),
                             "AUROC_src": float(roc_auc_score(yv, pv)), "AUROC_tgt": float(roc_auc_score(ye, pe))})
            print(f"{fam} {method}: done", flush=True)
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "prior_controlled.csv"), index=False)
    L = []
    for fam, g in d.groupby("family"):
        loss = (g.F_S - g.F_T_best).mean()
        prior = (g.F_S_best - g.F_S_prior).mean()
        rank = (g.F_S_prior - g.F_T_best).mean()
        L.append(f"{fam} ({len(g)} runs): F_S {g.F_S.mean():.3f}, F_S_best {g.F_S_best.mean():.3f}, "
                 f"F_S_prior {g.F_S_prior.mean():.3f}, F_T_best {g.F_T_best.mean():.3f}")
        L.append(f"   threshold-irrecoverable loss {100 * loss:.1f} pp = prior effect {100 * prior:.1f} pp "
                 f"+ ranking effect {100 * rank:.1f} pp (+ source threshold slack {100 * (g.F_S - g.F_S_best).mean():.1f} pp)")
        L.append(f"   AUROC source validation {g.AUROC_src.mean():.3f} -> target {g.AUROC_tgt.mean():.3f}")
        L.append("   by target: " + g.groupby("target").apply(
            lambda h: f"prior {100 * (h.F_S_best - h.F_S_prior).mean():.1f} / rank {100 * (h.F_S_prior - h.F_T_best).mean():.1f}"
        ).to_string().replace("\n", "; "))
    text = "\n".join(L)
    open(os.path.join(OUT, "prior_controlled_report.txt"), "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
