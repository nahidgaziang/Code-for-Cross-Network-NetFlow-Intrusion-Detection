"""
Single-source pairwise transfer and a source-side divergence audit.

For each release, every directed pair (source -> target, 12 pairs) plus the four in-distribution
cases is evaluated with XGB, RF and ERM-CB over five seeds. Transfer loss = in-distribution AUROC
minus transfer AUROC. Benign-traffic divergence between source and target (mean PSI, mean normalised
Wasserstein distance, mean absolute standardised mean difference) is correlated with the loss
(Spearman, bootstrap CI) to test whether the loss can be predicted before deployment.
  python experiments/pairwise_transfer.py run|report
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, wasserstein_distance
from sklearn.metrics import roc_auc_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, PKG_ROOT)

from crossnet.data import TESTBEDS, FoldConfig, MethodInput, build_fold, load_pool  # noqa: E402
from crossnet.methods import run_method  # noqa: E402
from crossnet.preprocess import log_features, GlobalScaler  # noqa: E402
from crossnet.runner import append, done_keys  # noqa: E402

BENCH = os.path.join(PKG_ROOT, "results", "benchmark")
E6_DIR = os.path.join(PKG_ROOT, "results", "e6")

PAIRWISE_SEEDS = [42, 1337, 2025, 2026, 2027]
E6_MODELS = ("XGB", "RF", "ERM-CB")
FAMILIES = ("nfv2", "nfv3")
N_SRC_TRAIN = 50_000
VAL_FRAC = 0.2


def selected_hp(fam, method):
    path = os.path.join(BENCH, f"selected_{fam}__{method}.json")
    with open(path) as f:
        return json.load(f)


def compute_psi(source_col: np.ndarray, target_col: np.ndarray, bins: int = 10) -> float:
    try:
        quantiles = np.linspace(0, 100, bins + 1)
        cuts = np.percentile(source_col, quantiles)
        cuts[0], cuts[-1] = -np.inf, np.inf
        cuts = np.unique(cuts)
        if len(cuts) < 3:
            return 0.0
        src_counts, _ = np.histogram(source_col, bins=cuts)
        tgt_counts, _ = np.histogram(target_col, bins=cuts)
        src_pct = np.clip(src_counts / max(len(source_col), 1), 1e-4, 1.0)
        tgt_pct = np.clip(tgt_counts / max(len(target_col), 1), 1e-4, 1.0)
        return float(np.sum((tgt_pct - src_pct) * np.log(tgt_pct / src_pct)))
    except Exception:
        return 0.0


def compute_pair_divergence(fam: str, src_testbed: str, tgt_testbed: str, sample_size: int = 10_000) -> dict:
    src_pool = load_pool(fam, src_testbed, "benign")
    tgt_pool = load_pool(fam, tgt_testbed, "benign")

    Xs = log_features(src_pool["X"][:sample_size])
    Xt = log_features(tgt_pool["X"][:sample_size])

    n_feats = Xs.shape[1]
    psis, w_dists, z_diffs = [], [], []

    sc = GlobalScaler().fit(Xs)
    Xs_norm = sc.transform(Xs)
    Xt_norm = sc.transform(Xt)

    for f in range(n_feats):
        psis.append(compute_psi(Xs[:, f], Xt[:, f]))
        w_dists.append(wasserstein_distance(Xs_norm[:, f], Xt_norm[:, f]))
        z_diffs.append(abs(float(np.mean(Xt_norm[:, f]) - np.mean(Xs_norm[:, f]))))

    return {
        "Mean_PSI": float(np.mean(psis)),
        "Mean_Wasserstein": float(np.mean(w_dists)),
        "Mean_Abs_Z_Diff": float(np.mean(z_diffs)),
    }


def compute_audit_table():
    os.makedirs(E6_DIR, exist_ok=True)
    audit_file = os.path.join(E6_DIR, "e6_audit_divergence.csv")
    rows = []
    for fam in FAMILIES:
        testbeds = TESTBEDS[fam]
        for src in testbeds:
            for tgt in testbeds:
                if src == tgt:
                    rows.append({
                        "family": fam, "source": src, "target": tgt,
                        "Mean_PSI": 0.0, "Mean_Wasserstein": 0.0, "Mean_Abs_Z_Diff": 0.0
                    })
                else:
                    div = compute_pair_divergence(fam, src, tgt)
                    rows.append({"family": fam, "source": src, "target": tgt, **div})
    df_audit = pd.DataFrame(rows)
    df_audit.to_csv(audit_file, index=False)
    print(f"[E6 Audit] Computed benign feature divergence for all 32 pairs -> {audit_file}")
    return df_audit


def run_e6():
    os.makedirs(E6_DIR, exist_ok=True)
    csv_path = os.path.join(E6_DIR, "e6_pairwise_runs.csv")
    done = done_keys(csv_path, ["family", "source", "target", "seed", "method"])

    # Ensure audit divergence table exists
    audit_file = os.path.join(E6_DIR, "e6_audit_divergence.csv")
    if not os.path.exists(audit_file):
        compute_audit_table()

    print("[E6] Starting Pairwise Transfer & Predictive Audit Benchmark")
    print(f"[E6] Families: {FAMILIES}, Seeds: {PAIRWISE_SEEDS}, Models: {E6_MODELS}")

    for fam in FAMILIES:
        testbeds = TESTBEDS[fam]
        for seed in PAIRWISE_SEEDS:
            # Build target evaluation test sets for all 4 testbeds using official folds
            target_eval_streams = {}
            for t_name in testbeds:
                inp_f, truth_f, _ = build_fold(FoldConfig(family=fam, target=t_name, seed=seed))
                target_eval_streams[t_name] = (inp_f.X_eval, truth_f.y_eval.astype(int), inp_f.features)

            for src in testbeds:
                # Check if all pairs from this source are already done
                all_done = True
                for tgt in testbeds:
                    for m in E6_MODELS:
                        if (fam, src, tgt, str(seed), m) not in done:
                            all_done = False
                            break
                    if not all_done:
                        break
                if all_done:
                    continue

                # Sample training flows from source uniform pool
                src_pool = load_pool(fam, src, "uniform")
                rng = np.random.default_rng([seed, testbeds.index(src), 6])
                n_total = min(N_SRC_TRAIN, len(src_pool["y"]))
                perm = rng.permutation(len(src_pool["y"]))
                src_idx = perm[:n_total]

                # Disjoint in-distribution evaluation split from source
                id_eval_idx = perm[n_total: n_total + 20_000] if len(perm) >= n_total + 20_000 else perm[:10_000]
                X_id_eval = src_pool["X"][id_eval_idx]
                y_id_eval = src_pool["y"][id_eval_idx].astype(int)

                n_tr = int(round((1.0 - VAL_FRAC) * n_total))
                tr_idx, va_idx = src_idx[:n_tr], src_idx[n_tr:]

                X_tr, y_tr = src_pool["X"][tr_idx], src_pool["y"][tr_idx].astype(int)
                X_va, y_va = src_pool["X"][va_idx], src_pool["y"][va_idx].astype(int)

                for tgt in testbeds:
                    X_tgt_eval, y_tgt_eval, feat_names = (
                        (X_id_eval, y_id_eval, [str(f) for f in src_pool["features"]])
                        if src == tgt
                        else target_eval_streams[tgt]
                    )

                    cinp = MethodInput(
                        X_tr=X_tr, y_tr=y_tr, d_tr=np.zeros(len(y_tr), np.int8),
                        X_va=X_va, y_va=y_va, d_va=np.zeros(len(y_va), np.int8),
                        X_buf=X_tgt_eval[:10], X_win=X_tgt_eval[:10],
                        X_eval=X_tgt_eval,
                        features=feat_names, source_names=[src]
                    )

                    for method in E6_MODELS:
                        if (fam, src, tgt, str(seed), method) in done:
                            continue

                        t0 = time.time()
                        hp = selected_hp(fam, method)[src]  # source-tuned hyperparameters
                        preds, _ = run_method(method, cinp, hp, seed=seed)
                        elapsed = round(time.time() - t0, 2)

                        s_eval = preds["eval"]
                        auc = float(roc_auc_score(y_tgt_eval, s_eval)) if 0 < y_tgt_eval.sum() < len(y_tgt_eval) else np.nan

                        # Class-balanced operational threshold on source validation
                        s_va = preds["val"]
                        thresh = float(np.median(s_va)) if len(s_va) > 0 else 0.5
                        y_pred = (s_eval >= thresh).astype(int)
                        f1 = float(f1_score(y_tgt_eval, y_pred, average="macro"))

                        record = {
                            "family": fam,
                            "source": src,
                            "target": tgt,
                            "seed": seed,
                            "method": method,
                            "is_transfer": int(src != tgt),
                            "n_train": len(y_tr),
                            "n_eval": len(y_tgt_eval),
                            "AUROC": auc,
                            "MacroF1": f1,
                            "seconds": elapsed,
                        }
                        append(csv_path, record)
                        done.add((fam, src, tgt, str(seed), method))
                        print(f"[E6] {fam} {src} -> {tgt} seed={seed} {method}: AUROC={auc:.4f}, F1={f1:.4f} ({elapsed}s)", flush=True)


    print("[E6] Pairwise benchmark finished ->", csv_path)


def run_report():
    csv_path = os.path.join(E6_DIR, "e6_pairwise_runs.csv")
    audit_file = os.path.join(E6_DIR, "e6_audit_divergence.csv")

    if not os.path.exists(csv_path):
        print("Run 'python pairwise_audit_e6.py run' first.")
        return

    df_runs = pd.read_csv(csv_path)
    df_audit = pd.read_csv(audit_file) if os.path.exists(audit_file) else compute_audit_table()

    report_lines = []
    report_lines.append("=" * 88)
    report_lines.append("EXPERIMENT E6 REPORT: PAIRWISE TRANSFERS & PREDICTIVE AUDIT (AUDIT GATE)")
    report_lines.append("=" * 88)
    report_lines.append("")

    # 1. Pairwise Transfer Matrices (Mean AUROC and Macro-F1 across 5 seeds)
    for fam in FAMILIES:
        sub_fam = df_runs[df_runs["family"] == fam]
        report_lines.append(f"--- PAIRWISE TRANSFER MATRICES: {fam.upper()} (Source Row -> Target Column) ---")
        for method in E6_MODELS:
            sub_m = sub_fam[sub_fam["method"] == method]
            tbl_auc = sub_m.groupby(["source", "target"])["AUROC"].mean().unstack("target")
            tbl_f1 = sub_m.groupby(["source", "target"])["MacroF1"].mean().unstack("target")

            report_lines.append(f"\n[{fam.upper()} - {method}] Mean AUROC across 5 seeds:")
            report_lines.append(tbl_auc.round(4).to_string())
            report_lines.append(f"\n[{fam.upper()} - {method}] Mean Macro-F1 across 5 seeds:")
            report_lines.append(tbl_f1.round(4).to_string())
        report_lines.append("\n" + "-" * 88 + "\n")

    # 2. Compute Transfer Losses and Correlate with Benign Audit Divergence
    # For each pair (fam, src, tgt, method), compute:
    # Transfer_Loss_AUROC = AUROC(src -> src) - AUROC(src -> tgt)
    # Transfer_Loss_MacroF1 = MacroF1(src -> src) - MacroF1(src -> tgt)
    pair_records = []
    for (fam, method), grp in df_runs.groupby(["family", "method"]):
        id_means = grp[grp["source"] == grp["target"]].groupby("source")[["AUROC", "MacroF1"]].mean()
        transfers = grp[grp["source"] != grp["target"]].groupby(["source", "target"])[["AUROC", "MacroF1"]].mean().reset_index()

        for _, row in transfers.iterrows():
            s, t = row["source"], row["target"]
            id_auc = id_means.loc[s, "AUROC"]
            id_f1 = id_means.loc[s, "MacroF1"]
            loss_auc = id_auc - row["AUROC"]
            loss_f1 = id_f1 - row["MacroF1"]

            # Merge audit divergence metrics
            audit_match = df_audit[(df_audit["family"] == fam) & (df_audit["source"] == s) & (df_audit["target"] == t)]
            if len(audit_match) > 0:
                rec = audit_match.iloc[0]
                pair_records.append({
                    "family": fam, "method": method, "source": s, "target": t,
                    "In_Dist_AUROC": id_auc, "Transfer_AUROC": row["AUROC"], "Loss_AUROC": loss_auc,
                    "In_Dist_MacroF1": id_f1, "Transfer_MacroF1": row["MacroF1"], "Loss_MacroF1": loss_f1,
                    "Mean_PSI": rec["Mean_PSI"], "Mean_Wasserstein": rec["Mean_Wasserstein"],
                    "Mean_Abs_Z_Diff": rec["Mean_Abs_Z_Diff"],
                })

    df_pairs = pd.DataFrame(pair_records)
    pairs_csv = os.path.join(E6_DIR, "e6_transfer_losses.csv")
    df_pairs.to_csv(pairs_csv, index=False)

    report_lines.append("AUDIT GATE HYPOTHESIS EVALUATION: PREDICTIVE SOURCE-ONLY AUDIT")
    report_lines.append("Hypothesis: Feature divergence (benign source vs target) correlates with Transfer Loss (Spearman rho >= 0.50).")
    report_lines.append("Success Criterion: Spearman rho >= 0.50 with 95% bootstrap CI strictly excluding 0 across 24 directed pairs.")
    report_lines.append("-" * 88)

    rng = np.random.default_rng(20261105)

    def bootstrap_ci(x, y, n_boot=10_000, alpha=0.05):
        rhos = []
        n = len(x)
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            r, _ = spearmanr(x[idx], y[idx])
            if np.isfinite(r):
                rhos.append(r)
        lo = np.percentile(rhos, 100 * (alpha / 2))
        hi = np.percentile(rhos, 100 * (1 - alpha / 2))
        return lo, hi

    g1_overall_verdict = False
    for method in E6_MODELS:
        sub_m = df_pairs[df_pairs["method"] == method]
        report_lines.append(f"\n>>> METHOD: {method} (Total directed transfer pairs n = {len(sub_m)})")

        for metric_name, loss_col in [("AUROC Loss", "Loss_AUROC"), ("Macro-F1 Loss", "Loss_MacroF1")]:
            report_lines.append(f"  Target Loss Metric: {metric_name}")
            for div_name in ["Mean_PSI", "Mean_Wasserstein", "Mean_Abs_Z_Diff"]:
                x = sub_m[div_name].values
                y = sub_m[loss_col].values

                rho, p_spear = spearmanr(x, y)
                r_pears, p_pears = pearsonr(x, y)
                ci_lo, ci_hi = bootstrap_ci(x, y)

                passes_g1 = (rho >= 0.50) and (ci_lo > 0.0)
                if passes_g1:
                    g1_overall_verdict = True

                verdict_str = "PASSES G1 (Predictive)" if passes_g1 else "FAILS G1 (Descriptive Only)"
                report_lines.append(
                    f"    vs {div_name:16s}: Spearman rho = {rho:+.4f} (p={p_spear:.2e}, 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]) | "
                    f"Pearson r = {r_pears:+.4f} -> {verdict_str}"
                )

    report_lines.append("\n" + "=" * 88)
    report_lines.append("OVERALL AUDIT GATE VERDICT & OPERATIONAL IMPLICATIONS:")
    if g1_overall_verdict:
        report_lines.append(
            "AUDIT GATE PASSES: source-only benign feature divergence predicts the transfer loss on these pairs."
        )
    else:
        report_lines.append(
            "AUDIT GATE FAILS: Benign feature divergence alone does not achieve rho >= 0.50 with a strictly positive CI across all pairs.\n"
            "The audit is descriptive only: source-target divergence does not predict which transfers fail."
        )
    report_lines.append("=" * 88)

    report_text = "\n".join(report_lines)
    print(report_text)
    report_file = os.path.join(E6_DIR, "report.txt")
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"\n[E6] Complete report written to: {report_file}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("run", "all"):
        run_e6()
    if mode in ("report", "all"):
        run_report()

