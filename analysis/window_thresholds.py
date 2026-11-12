"""
Threshold rules for the window detectors from saved scores: win_mad over Z in {1, 1.5, 2, 2.5, 3},
win_fpr over alpha in {0.01, 0.05, 0.10, 0.20}, and the best possible threshold (uses target labels;
diagnostic upper bound). Z = 3 and alpha = 0.01 are the declared operating points; the other values
are a sensitivity analysis and must not be read as tuned rules.
  python analysis/window_thresholds.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "results", "benchmark")
OUT_DIR = os.path.join(BENCH, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def oracle_macro_f1(y, s):
    order = np.argsort(-s, kind="stable")
    ys = y[order].astype(np.int64)
    n, P = len(ys), int(ys.sum())
    N = n - P
    tp = np.concatenate([[0], np.cumsum(ys)])
    fp = np.arange(n + 1) - tp
    fn, tn = P - tp, N - fp
    f1_att = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    f1_ben = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
    macro = 0.5 * (f1_att + f1_ben)
    best_idx = int(np.argmax(macro)) if macro.size > 0 else 0
    best_macro = float(macro[best_idx]) if macro.size > 0 else 0.0
    tau = float(s[order[max(0, best_idx - 1)]]) if best_idx > 0 and best_idx <= len(order) else float(np.median(s))
    return best_macro, tau


def eval_rule(y_true, scores, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    f1_att = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_ben = 2 * tn / max(2 * tn + fp + fn, 1)
    macro = 0.5 * (f1_att + f1_ben)
    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)
    return float(macro), float(fpr), float(tpr)


def main():
    rows = []
    z_sweep = [1.0, 1.5, 2.0, 2.5, 3.0]
    alpha_sweep = [0.01, 0.05, 0.10, 0.20]

    for fam in ("nfv2", "nfv3"):
        p = os.path.join(BENCH, f"scores_{fam}")
        for method in ("KNN-win", "IF-win"):
            files = sorted(glob.glob(os.path.join(p, f"*{method}*.npz")))
            for f in files:
                base = os.path.basename(f)[:-4]
                target, m, seed_s = base.split("__")
                seed = int(seed_s)
                with np.load(f) as z:
                    pe = z["p_eval"]
                    pw = z["p_win"]
                    ye = z["y_eval"]

                # 1. Oracle
                f1_oracle, tau_oracle = oracle_macro_f1(ye, pe)
                rows.append({"family": fam, "target": target, "method": method, "seed": seed,
                             "rule": "oracle_threshold", "param": "exact", "tau": tau_oracle,
                             "MacroF1": f1_oracle, "FPR": np.nan, "TPR": np.nan})

                # 2. win_mad sweep
                med = float(np.median(pw))
                mad = 1.4826 * float(np.median(np.abs(pw - med)))
                for z_val in z_sweep:
                    tau = med + z_val * max(mad, 1e-6)
                    yp = (pe > tau).astype(np.int8)
                    f1, fpr, tpr = eval_rule(ye, pe, yp)
                    rows.append({"family": fam, "target": target, "method": method, "seed": seed,
                                 "rule": "win_mad", "param": f"z={z_val}", "tau": tau,
                                 "MacroF1": f1, "FPR": fpr, "TPR": tpr})

                # 3. win_fpr sweep
                for a_val in alpha_sweep:
                    tau = float(np.quantile(pw, 1.0 - a_val))
                    yp = (pe > tau).astype(np.int8)
                    f1, fpr, tpr = eval_rule(ye, pe, yp)
                    rows.append({"family": fam, "target": target, "method": method, "seed": seed,
                                 "rule": "win_fpr", "param": f"alpha={a_val}", "tau": tau,
                                 "MacroF1": f1, "FPR": fpr, "TPR": tpr})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "s2_decisions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}", flush=True)

    # Generate formal report
    lines = [
        "=" * 80,
        "WINDOW-DETECTOR THRESHOLD RULES",
        "Threshold calibration for windowed detectors (KNN-win, IF-win)",
        "=" * 80,
        ""
    ]

    for fam in ("nfv2", "nfv3"):
        lines.append(f"\n--- {fam.upper()} RESULTS (Mean over 4 targets x 10 seeds) ---")
        sub = df[df.family == fam]
        pivot = sub.pivot_table(index=["method", "rule", "param"],
                                values=["MacroF1", "FPR", "TPR"],
                                aggfunc="mean").round(4)
        lines.append(pivot.to_string())

        # Target breakdown for KNN-win under z=1.5 and z=3.0 vs oracle
        lines.append(f"\nTarget-by-Target Macro-F1 Breakdown for {fam.upper()} KNN-win:")
        knn_sub = sub[(sub.method == "KNN-win") & sub.rule.isin(["win_mad", "oracle_threshold"])]
        knn_pivot = knn_sub.pivot_table(index="target", columns=["rule", "param"], values="MacroF1").round(4)
        lines.append(knn_pivot.to_string())

        # Statistical comparison: z=1.5 vs standard z=3.0
        z15 = sub[(sub.method == "KNN-win") & (sub.param == "z=1.5")].set_index(["target", "seed"])["MacroF1"]
        z30 = sub[(sub.method == "KNN-win") & (sub.param == "z=3.0")].set_index(["target", "seed"])["MacroF1"]
        diff = z15 - z30
        w_stat, w_p = wilcoxon(z15, z30)
        lines.append(f"\nKNN-win z=1.5 vs z=3.0: mean diff = {diff.mean() * 100:+.2f} pp, Wilcoxon p = {w_p:.2g} (n={len(diff)})")

    rep_path = os.path.join(OUT_DIR, "s2_decisions_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {rep_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
