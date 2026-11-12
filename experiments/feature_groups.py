"""
Feature-group ablation of target-benign false alarms.

XGBoost is retrained on the benchmark folds without one feature group at a time (TTL: MIN_TTL,
MAX_TTL, DNS_TTL_ANSWER; TCP window: TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT; timing: durations and, in
NF-v3, inter-arrival statistics; all artefacts: TTL + TCP window + ports). Reported: target AUROC and
the target-benign false-positive rate at a threshold calibrated to 1% FPR on source validation
benign flows.
  python experiments/feature_groups.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.data import build_fold, FoldConfig, TESTBEDS
from crossnet.preprocess import GlobalScaler, log_features
from crossnet.runner import EVAL_SEEDS, append, done_keys

RES = os.path.join(HERE, "..", "results", "e7")
os.makedirs(RES, exist_ok=True)

GROUPS = {
    "all_features": [],
    "drop_ttl": ["MIN_TTL", "MAX_TTL", "DNS_TTL_ANSWER"],
    "drop_tcp_win": ["TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"],
    "drop_timing": ["FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
                    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
                    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV"],
    "drop_artifacts": ["MIN_TTL", "MAX_TTL", "DNS_TTL_ANSWER",
                       "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
                       "L4_SRC_PORT", "L4_DST_PORT", "DNS_QUERY_ID", "ICMP_TYPE", "ICMP_IPV4_TYPE", "FTP_COMMAND_RET_CODE"]
}


def run_e7():
    csv_path = os.path.join(RES, "e7_runs.csv")
    done = done_keys(csv_path, ["family", "target", "seed", "ablation"])

    for fam in ("nfv2", "nfv3"):
        for target in TESTBEDS[fam]:
            for seed in EVAL_SEEDS:
                todo = [g for g in GROUPS if (fam, target, str(seed), g) not in done]
                if not todo:
                    continue
                inp, truth, _ = build_fold(FoldConfig(family=fam, target=target, seed=seed))
                y_tr = inp.y_tr.astype(int)
                y_va = inp.y_va.astype(int)
                y_ev = truth.y_eval.astype(int)

                for g in todo:
                    drop_cols = [c for c in GROUPS[g] if c in inp.features]
                    keep_idx = [i for i, f in enumerate(inp.features) if f not in drop_cols]

                    X_tr_sub = inp.X_tr[:, keep_idx]
                    X_va_sub = inp.X_va[:, keep_idx]
                    X_ev_sub = inp.X_eval[:, keep_idx]

                    sc = GlobalScaler().fit(log_features(X_tr_sub))
                    tr_s = sc.transform(log_features(X_tr_sub))
                    va_s = sc.transform(log_features(X_va_sub))
                    ev_s = sc.transform(log_features(X_ev_sub))

                    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                            tree_method="hist", device="cuda", random_state=seed)
                    clf.fit(tr_s, y_tr)

                    p_va = clf.predict_proba(va_s)[:, 1]
                    p_ev = clf.predict_proba(ev_s)[:, 1]

                    # Source-calibrated 1% FPR threshold
                    va_ben = p_va[y_va == 0]
                    tau_1pct = float(np.quantile(va_ben, 0.99)) if len(va_ben) > 0 else 0.5

                    # Target benign FPR at source-calibrated threshold
                    tgt_ben = p_ev[y_ev == 0]
                    fpr_at_src1 = float(np.mean(tgt_ben > tau_1pct)) if len(tgt_ben) > 0 else 0.0
                    auroc = float(roc_auc_score(y_ev, p_ev))

                    append(csv_path, {
                        "family": fam, "target": target, "seed": seed, "ablation": g,
                        "n_features": len(keep_idx), "AUROC": auroc, "tgt_benign_FPR": fpr_at_src1,
                        "tau_src1": tau_1pct
                    })
            print(f"[e7] {fam} {target} done", flush=True)

    # Report generation
    df = pd.read_csv(csv_path)
    lines = [
        "=" * 80,
        "EXPERIMENT E7: FEATURE-GROUP ABLATION",
        "Target AUROC and target-benign false alarms without each feature group (association only)",
        "=" * 80,
        ""
    ]

    for fam in ("nfv2", "nfv3"):
        lines.append(f"\n--- {fam.upper()} (Mean over 4 targets x 10 seeds) ---")
        sub = df[df.family == fam]
        piv = sub.pivot_table(index="ablation", values=["AUROC", "tgt_benign_FPR"], aggfunc="mean").round(4)
        lines.append(piv.to_string())

        lines.append(f"\nTarget Breakdown for {fam.upper()} tgt_benign_FPR:")
        t_piv = sub.pivot_table(index="target", columns="ablation", values="tgt_benign_FPR").round(4)
        lines.append(t_piv.to_string())

    rep_path = os.path.join(RES, "report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {rep_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    run_e7()

