"""
Within-network temporal training versus cross-network transfer (NF-v3).

For each target testbed, XGB, RF and ERM-CB are trained on labelled flows from the early part of the
same testbed (start time <= cut, cut in {0.2, 0.5}; up to 50,000 flows, 80/20 split; flows disjoint
from the benchmark evaluation, buffer and window sets) and evaluated on the benchmark evaluation flows
that start after the cut. The same late flows are scored by the cross-network models from the
benchmark, so the comparison is matched on evaluation flows and model family. Benchmark-selected
hyperparameters.
  python experiments/within_vs_cross_network.py run|report
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, PKG_ROOT)

from crossnet.data import TESTBEDS, FoldConfig, MethodInput, build_fold, load_pool  # noqa: E402
from crossnet.methods import run_method  # noqa: E402
from crossnet.runner import EVAL_SEEDS, append, done_keys  # noqa: E402

BENCH = os.path.join(PKG_ROOT, "results", "benchmark")
E5T_DIR = os.path.join(PKG_ROOT, "results", "e5t")
E5_DIR = os.path.join(PKG_ROOT, "results", "e5")

E5_CUTS = (0.2, 0.5)
E5_MODELS = ("XGB", "RF", "ERM-CB")
FAMILY = "nfv3"
N_EARLY_MAX = 50_000
VAL_FRAC = 0.2


def selected_hp(fam, method):
    path = os.path.join(BENCH, f"selected_{fam}__{method}.json")
    with open(path) as f:
        return json.load(f)


def eval_t_start(fam, target, rows_eval):
    tu = load_pool(fam, target, "uniform")
    idx = pd.Index(tu["row"]).get_indexer(rows_eval)
    assert (idx >= 0).all(), "eval rows not found in uniform pool"
    return tu["t_start"][idx].astype(np.float64)


def run_e5():
    os.makedirs(E5_DIR, exist_ok=True)
    csv_path = os.path.join(E5_DIR, "e5_runs.csv")
    done = done_keys(csv_path, ["target", "seed", "cut", "method"])

    # Load E5T late-flow run counts to assert 100% row-for-row alignment
    e5t_runs_file = os.path.join(E5T_DIR, "e5t_runs.csv")
    df_e5t_ref = pd.read_csv(e5t_runs_file) if os.path.exists(e5t_runs_file) else None

    print(f"[E5] Starting Time Shift vs Network Shift study on {FAMILY}")
    print(f"[E5] Targets: {TESTBEDS[FAMILY]}, Seeds: {len(EVAL_SEEDS)}, Cuts: {E5_CUTS}, Models: {E5_MODELS}")

    for seed in EVAL_SEEDS:
        for target in TESTBEDS[FAMILY]:
            # Check if all models for this target and seed are already done
            todo_cuts = [c for c in E5_CUTS if not all((target, str(seed), str(c), m) in done for m in E5_MODELS)]
            if not todo_cuts:
                continue

            inp_fold, truth, meta = build_fold(FoldConfig(family=FAMILY, target=target, seed=seed))
            tu = load_pool(FAMILY, target, "uniform")
            t_eval = eval_t_start(FAMILY, target, meta["rows_eval"])
            t_uniform = tu["t_start"].astype(np.float64)

            # Candidate early flows: flows in uniform pool not used in eval, buffer, or window
            used_rows = np.concatenate([meta["rows_eval"], meta["rows_buf"], meta["rows_win"]])
            free_mask = ~np.isin(tu["row"], used_rows)
            free_indices = np.where(free_mask)[0]

            for cut_q in todo_cuts:
                cut_val = float(np.quantile(t_uniform, cut_q))
                late_mask = t_eval > cut_val
                y_late = truth.y_eval[late_mask].astype(int)

                # Verification: verify identical late slice against E5-T baseline
                if df_e5t_ref is not None:
                    ref_match = df_e5t_ref[
                        (df_e5t_ref["target"] == target) &
                        (df_e5t_ref["seed"] == seed) &
                        (df_e5t_ref["cut"] == cut_q) &
                        (df_e5t_ref["method"] == "KNN-win") &
                        (df_e5t_ref["window"] == "early")
                    ]
                    if len(ref_match) > 0:
                        expected_late = int(ref_match.iloc[0]["n_late"])
                        expected_late_benign = int(ref_match.iloc[0]["n_late_benign"])
                        assert int(late_mask.sum()) == expected_late, (
                            f"Late row count mismatch: {late_mask.sum()} vs {expected_late}"
                        )
                        assert int((y_late == 0).sum()) == expected_late_benign, (
                            f"Late benign count mismatch: {(y_late == 0).sum()} vs {expected_late_benign}"
                        )

                # Select early free flows (t_start <= cut_val)
                early_free = free_indices[t_uniform[free_indices] <= cut_val]
                rng = np.random.default_rng([seed, TESTBEDS[FAMILY].index(target), 5, int(cut_q * 100)])
                n_early = min(N_EARLY_MAX, len(early_free))
                chosen = rng.choice(early_free, n_early, replace=False)

                n_tr = int(round((1.0 - VAL_FRAC) * n_early))
                tr_idx, va_idx = chosen[:n_tr], chosen[n_tr:]

                X_tr, y_tr = tu["X"][tr_idx], tu["y"][tr_idx].astype(int)
                X_va, y_va = tu["X"][va_idx], tu["y"][va_idx].astype(int)

                cinp = MethodInput(
                    X_tr=X_tr, y_tr=y_tr, d_tr=np.zeros(len(y_tr), np.int8),
                    X_va=X_va, y_va=y_va, d_va=np.zeros(len(y_va), np.int8),
                    X_buf=inp_fold.X_buf[:10], X_win=inp_fold.X_win[:10],
                    X_eval=inp_fold.X_eval,
                    features=inp_fold.features, source_names=[target]
                )

                for method in E5_MODELS:
                    if (target, str(seed), str(cut_q), method) in done:
                        continue

                    t0 = time.time()
                    hp = selected_hp(FAMILY, method)[target]
                    preds, _ = run_method(method, cinp, hp, seed=seed)
                    elapsed = round(time.time() - t0, 2)

                    s_late = preds["eval"][late_mask]
                    auc = float(roc_auc_score(y_late, s_late)) if 0 < y_late.sum() < len(y_late) else np.nan

                    record = {
                        "family": FAMILY,
                        "target": target,
                        "seed": seed,
                        "cut": cut_q,
                        "method": method,
                        "n_early_train": len(y_tr),
                        "n_early_train_attacks": int(y_tr.sum()),
                        "n_early_train_benign": int((y_tr == 0).sum()),
                        "n_early_val": len(y_va),
                        "n_early_val_attacks": int(y_va.sum()),
                        "n_early_val_benign": int((y_va == 0).sum()),
                        "n_late": int(late_mask.sum()),
                        "n_late_benign": int((y_late == 0).sum()),
                        "late_attack_share": float(y_late.mean()),
                        "AUROC": auc,
                        "seconds": elapsed,
                    }
                    append(csv_path, record)
                    done.add((target, str(seed), str(cut_q), method))
                    print(f"[E5] {target} seed={seed} cut={cut_q} {method}: AUROC={auc:.4f} ({elapsed}s)", flush=True)

                    # Cooperative sleep yielding to keep DPCs clear

    print("[E5] Training and evaluation complete. Saved to:", csv_path)


def run_report():
    csv_path = os.path.join(E5_DIR, "e5_runs.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run 'python temporal_studies_e5.py run' first.")
        return

    df_e5 = pd.read_csv(csv_path)
    df_e5t_late = pd.read_csv(os.path.join(E5T_DIR, "e5t_s01_late.csv"))
    df_e5t_runs = pd.read_csv(os.path.join(E5T_DIR, "e5t_runs.csv"))

    report_lines = []
    report_lines.append("=" * 88)
    report_lines.append("EXPERIMENT E5 REPORT: TIME SHIFT VS. NETWORK SHIFT (NF-v3)")
    report_lines.append("Hypotheses: H-Time1 (within-network stability), H-Time2 (network shift dominance)")
    report_lines.append("=" * 88)
    report_lines.append("")

    # Primary targets (>= 100 late benign flows)
    # BoT-IoT has only 16 late benign flows at cut 0.2 and 12 at cut 0.5 (99.98% attack share)
    primary_targets = ["NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3", "NF-UNSW-NB15-v3"]

    for cut_q in E5_CUTS:
        cut_label = f"--- CUT = {int(cut_q * 100)}% ({'PRIMARY' if cut_q == 0.2 else 'SENSITIVITY'}) ---"
        report_lines.append(cut_label)
        report_lines.append(f"Late evaluation flows: t_start > quantile(tu['t_start'], {cut_q})")
        report_lines.append("-" * 88)

        # 1. Within-Temporal Performance Table
        sub_e5 = df_e5[df_e5["cut"] == cut_q]
        tbl_within = sub_e5.groupby(["target", "method"])["AUROC"].agg(["mean", "std", "count"]).unstack("method")
        report_lines.append("WITHIN-TEMPORAL AUROC (Train Early -> Test Late within Same Testbed):")
        report_lines.append(tbl_within["mean"].round(4).to_string())
        report_lines.append("")

        # 2. Cross-Network Baseline (Best S0/S1 model on identical late flows)
        sub_late = df_e5t_late[df_e5t_late["cut"] == cut_q]
        best_s01 = sub_late.groupby(["target", "method"])["AUROC"].mean().unstack("method")
        overall_best_s01_method = sub_late.groupby("method")["AUROC"].mean().idxmax()
        overall_best_s01_mean = sub_late.groupby("method")["AUROC"].mean().max()

        # Best S0/S1 per target
        target_best_s01_method = best_s01.idxmax(axis=1)
        target_best_s01_mean = best_s01.max(axis=1)

        # S2 Reference Detectors on identical late flows (from e5t_runs.csv)
        sub_s2 = df_e5t_runs[(df_e5t_runs["cut"] == cut_q) & (df_e5t_runs["window"] == "early")]
        tbl_s2 = sub_s2.groupby(["target", "method"])["AUROC"].mean().unstack("method")

        # 3. Comparative Summary Table: Within-Temporal vs S2 vs Cross-Network
        summary_rows = []
        for target in TESTBEDS[FAMILY]:
            mean_xgb = sub_e5[(sub_e5["target"] == target) & (sub_e5["method"] == "XGB")]["AUROC"].mean()
            mean_rf = sub_e5[(sub_e5["target"] == target) & (sub_e5["method"] == "RF")]["AUROC"].mean()
            mean_erm = sub_e5[(sub_e5["target"] == target) & (sub_e5["method"] == "ERM-CB")]["AUROC"].mean()
            best_within = max(mean_xgb, mean_rf, mean_erm)

            knn_win = tbl_s2.loc[target, "KNN-win"] if "KNN-win" in tbl_s2.columns else np.nan
            if_win = tbl_s2.loc[target, "IF-win"] if "IF-win" in tbl_s2.columns else np.nan

            best_cross_m = target_best_s01_method[target]
            best_cross_val = target_best_s01_mean[target]

            delta_xgb_cross = mean_xgb - best_cross_val
            delta_within_cross = best_within - best_cross_val

            summary_rows.append({
                "Target": target,
                "Within_XGB": round(mean_xgb, 4),
                "Within_RF": round(mean_rf, 4),
                "Within_ERMCB": round(mean_erm, 4),
                "S2_KNN_win": round(knn_win, 4),
                "Best_Cross_Net": f"{best_cross_m} ({best_cross_val:.4f})",
                "Delta_Net (XGB - Cross)": f"{delta_xgb_cross * 100:+.2f} pp",
                "Delta_Net (Best - Cross)": f"{delta_within_cross * 100:+.2f} pp",
            })

        df_sum = pd.DataFrame(summary_rows)
        report_lines.append("COMPARATIVE AUROC SUMMARY ON IDENTICAL LATE FLOWS:")
        report_lines.append(df_sum.to_string(index=False))
        report_lines.append("")

        # 4. Statistical Tests for Pre-registered Hypotheses
        report_lines.append("PRE-REGISTERED HYPOTHESIS EVALUATIONS:")

        # H-Time1: Mean Within-Temporal AUROC >= 0.85 across primary targets (>= 100 late benign flows)
        sub_prim = sub_e5[sub_e5["target"].isin(primary_targets)]
        mean_prim_xgb = sub_prim[sub_prim["method"] == "XGB"]["AUROC"].mean()
        mean_prim_rf = sub_prim[sub_prim["method"] == "RF"]["AUROC"].mean()
        mean_prim_erm = sub_prim[sub_prim["method"] == "ERM-CB"]["AUROC"].mean()
        mean_prim_overall = sub_prim["AUROC"].mean()

        h_time1_holds = (mean_prim_xgb >= 0.85) and (mean_prim_rf >= 0.85) and (mean_prim_erm >= 0.85)
        report_lines.append(
            f"H-Time1 (Temporal Stability): Within-network training maintains mean AUROC >= 0.85 on primary targets.\n"
            f"  Primary Targets (>=100 late benign): {primary_targets}\n"
            f"  XGB Mean: {mean_prim_xgb:.4f}, RF Mean: {mean_prim_rf:.4f}, ERM-CB Mean: {mean_prim_erm:.4f}\n"
            f"  Combined Mean: {mean_prim_overall:.4f}\n"
            f"  -> VERDICT: {'HOLDS DECISIVELY' if h_time1_holds else 'FAILS'}"
        )
        report_lines.append("")

        # H-Time2: Network Shift Dominance (Delta_net >= +20 pp across target x seed pairs, Wilcoxon p < 0.05)
        # Pair each (target, seed) within-temporal run with the best cross-network model for that target on that seed
        # Calculate paired Wilcoxon across all (target, seed) pairs for primary targets and across all 4 targets
        for method in E5_MODELS:
            sub_m = sub_e5[sub_e5["method"] == method]
            pairs = []
            for _, row in sub_m.iterrows():
                tgt, sd = row["target"], row["seed"]
                w_auc = row["AUROC"]
                # Look up best S0/S1 on this exact seed and target
                c_matches = sub_late[(sub_late["target"] == tgt) & (sub_late["seed"] == sd)]
                if len(c_matches) > 0:
                    c_auc = c_matches["AUROC"].max()
                    pairs.append({"target": tgt, "seed": sd, "within": w_auc, "cross": c_auc, "delta": w_auc - c_auc})

            df_pairs = pd.DataFrame(pairs)
            # Primary targets test
            prim_pairs = df_pairs[df_pairs["target"].isin(primary_targets)]
            delta_prim = prim_pairs["delta"].values
            w_stat, p_val = wilcoxon(delta_prim, alternative="greater")
            mean_delta_prim = delta_prim.mean() * 100

            # All 4 targets test
            delta_all = df_pairs["delta"].values
            w_stat_all, p_val_all = wilcoxon(delta_all, alternative="greater")
            mean_delta_all = delta_all.mean() * 100

            report_lines.append(
                f"H-Time2 ({method} vs Cross-Network S0/S1):\n"
                f"  Primary Targets (n={len(delta_prim)}): Mean Delta_net = {mean_delta_prim:+.2f} pp, Wilcoxon p = {p_val:.2e}\n"
                f"  All Targets     (n={len(delta_all)}): Mean Delta_net = {mean_delta_all:+.2f} pp, Wilcoxon p = {p_val_all:.2e}\n"
                f"  Threshold: Delta_net >= +20 pp and p < 0.05 -> "
                f"{'HOLDS DECISIVELY' if (mean_delta_prim >= 20.0 and p_val < 0.05) else 'FAILS'}"
            )

        report_lines.append("")
        report_lines.append("=" * 88)
        report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)
    report_file = os.path.join(E5_DIR, "report.txt")
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"[E5] Report written to: {report_file}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("run", "all"):
        run_e5()
    if mode in ("report", "all"):
        run_report()

