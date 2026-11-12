"""
Decision-layer analysis from saved scores.

For each saved evaluation run, applies the fixed and em rules, BBSE, and two diagnostic upper bounds
that use target labels (the true target prior with the em threshold, and the best possible threshold
for target Macro-F1). The share of the source-to-target Macro-F1 loss that the best threshold
recovers separates ranking failure from threshold failure.
  python analysis/decision_layer.py [results_dir]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.decision import bbse_prior, em_decisions  # noqa: E402
from crossnet.methods import METHODS  # noqa: E402
from crossnet.metrics import evaluate  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "benchmark")
OUT = os.path.join(RES, "analysis")
RULES = ["fixed", "em", "em_noclip", "bbse", "em_oracle_prior", "oracle_threshold"]


def best_macro_f1(y, s):
    """Exact maximum Macro-F1 over all thresholds (cuts only between distinct scores)."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order].astype(np.int64), s[order]
    n, P = len(ys), int(ys.sum())
    N = n - P
    tp = np.concatenate([[0], np.cumsum(ys)])
    fp = np.arange(n + 1) - tp
    fn, tn = P - tp, N - fp
    f1_att = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    f1_ben = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
    valid = np.r_[True, ss[1:] != ss[:-1], True]
    return float((0.5 * (f1_att + f1_ben))[valid].max())


def rules(z):
    pv, yv, pb, pe, ye = z["p_val"], z["y_val"], z["p_buf"], z["p_eval"], z["y_eval"]
    out = {"fixed": evaluate(ye, pe, (pe >= 0.5).astype(np.int8))["MacroF1"]}
    for name, kw in (("em", {}), ("em_noclip", {"clip": (0.001, 0.999)}),
                     ("bbse", {"prior_fn": bbse_prior}), ("em_oracle_prior", {"prior": float(ye.mean())})):
        yhat, info = em_decisions(pv, yv, pe, pb, **kw)
        out[name] = evaluate(ye, pe, yhat)["MacroF1"]
        out[f"prior_{name}"] = info["pi_t"]
    out["oracle_threshold"] = best_macro_f1(ye, pe)
    out["prior_true"] = float(ye.mean())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for fam in ("nfv2", "nfv3"):
        for method, spec in METHODS.items():
            if not spec[4]:
                continue
            ev = pd.read_csv(os.path.join(RES, f"eval_{fam}__{method}.csv"))
            em_rows = ev[ev.rule == "em"].set_index(["target", "seed"])
            for f in sorted(glob.glob(os.path.join(RES, f"scores_{fam}", f"*__{method}__*.npz"))):
                target, _, seed = os.path.basename(f)[:-4].split("__")
                with np.load(f) as z:
                    r = {"family": fam, "target": target, "method": method, "seed": int(seed), **rules(z)}
                saved = em_rows.loc[(target, int(seed))]
                assert abs(saved.MacroF1 - r["em"]) < 1e-12, f"em rule does not reproduce the saved run: {f}"
                r["src_val_macro_f1"] = float(saved.SrcVal_MacroF1)
                rows.append(r)
            print(f"{fam} {method}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "decision_layer.csv"), index=False)

    lines = []
    for fam, d in df.groupby("family"):
        lines.append(f"\n===== {fam} ({len(d)} runs, {d.method.nunique()} methods) =====")
        lines.append("Mean Macro-F1 by method and rule:\n" +
                     d.groupby("method")[["src_val_macro_f1"] + RULES].mean().round(4).to_string())
        loss = d.src_val_macro_f1 - d.em
        rec_thr = (d.oracle_threshold - d.em).mean() / loss.mean()
        rec_pri = (d.em_oracle_prior - d.em).mean() / loss.mean()
        lines.append(f"\nMean loss in-distribution -> target under em: {100 * loss.mean():.2f} pp")
        lines.append(f"   recovered by the true target prior: {100 * rec_pri:.1f}%")
        lines.append(f"   recovered by the best possible threshold: {100 * rec_thr:.1f}%")
        lines.append(f"   remaining after the best threshold (ranking-limited): "
                     f"{100 * (d.src_val_macro_f1 - d.oracle_threshold).mean():.2f} pp")
        per_m = d.groupby("method").apply(lambda g: (g.oracle_threshold - g.em).mean() / (g.src_val_macro_f1 - g.em).mean())
        lines.append("   best-threshold recovery share by method: " +
                     ", ".join(f"{k} {100 * v:.0f}%" for k, v in per_m.sort_values().items()))
        lines.append(f"em_oracle_prior - em: {100 * (d.em_oracle_prior - d.em).mean():+.2f} pp, "
                     f"Wilcoxon p={wilcoxon(d.em_oracle_prior, d.em).pvalue:.2g}")
        lines.append(f"bbse - em: {100 * (d.bbse - d.em).mean():+.2f} pp, Wilcoxon p={wilcoxon(d.bbse, d.em).pvalue:.2g}")
    text = "\n".join(lines)
    open(os.path.join(OUT, "decision_layer_report.txt"), "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
