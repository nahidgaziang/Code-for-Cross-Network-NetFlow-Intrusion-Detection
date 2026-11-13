"""
Results ledger for the paper. Every number reported in the paper is recomputed here from the run-level
result files (one row per target x seed x method) and written to paper/results_ledger.csv, with one
LaTeX macro per value in paper/generated/results.tex. The paper's tables are written to
paper/generated/tables/. Seed-bootstrap intervals use a fixed generator seed, so the output is
deterministic.
  python paper/build_ledger.py
Environment: NF_RESULTS_DIR (default: results_archive/), NF_POOL_DIR (default: results_archive/pools/).
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
R = os.environ.get("NF_RESULTS_DIR", os.path.join(ROOT, "results_archive"))
B = os.path.join(R, "benchmark")
POOLS = os.environ.get("NF_POOL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_archive", "pools"))
OUT = HERE
GEN = os.path.join(OUT, "generated")
os.makedirs(os.path.join(GEN, "tables"), exist_ok=True)
sys.path.insert(0, ROOT)
from crossnet.methods import METHODS  # noqa: E402

FAMS = {"nfv2": "Vtwo", "nfv3": "Vthree"}
TGT = {"NF-BoT-IoT": "Bot", "NF-CSE-CIC-IDS2018": "Cse", "NF-ToN-IoT": "Ton", "NF-UNSW-NB15": "Unsw"}
DISPLAY = {"VREx": "V-REx"}   # display names for the paper
S01 = [m for m in METHODS if METHODS[m][3] in ("S0", "S1")]
BOOT_SEED, N_BOOT = 20261108, 10000
ledger = []


def mname(m):
    return re.sub(r"[^A-Za-z]", "", DISPLAY.get(m, m).title().replace("-", ""))


def tname(t):
    return TGT[t.rsplit("-", 1)[0]]


def put(key, value, dec, kind, source, note=""):
    assert re.fullmatch(r"[A-Za-z]+", key), key
    assert value is not None and np.isfinite(value), (key, value)
    if kind == "int":
        assert abs(value - round(value)) < 1e-9, (key, value)
    ledger.append({"key": key, "value": float(value), "decimals": dec, "kind": kind, "source": source, "note": note})


def wil(d):
    d = np.asarray(d, float)
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)


def boot_ci(piv):
    """Seed bootstrap of the mean over targets: piv is seeds x targets."""
    assert not piv.isna().any().any(), "incomplete seeds x targets table"
    v = piv.values.mean(1)
    rng = np.random.default_rng(BOOT_SEED)
    m = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_ci_long(df, col):
    """Seed-bootstrap CI of a column given in long form (target, seed, col)."""
    return boot_ci(df.pivot_table(index="seed", columns="target", values=col))


# ---------------------------------------------------------------- study constants
put("NTestbeds", 4, 0, "int", "design")
put("NReleases", 2, 0, "int", "design")
put("NTargets", 8, 0, "int", "design")
put("NMethods", len(METHODS), 0, "int", "crossnet/methods.py")
put("NSZeroSOne", len(S01), 0, "int", "crossnet/methods.py")
put("NSeeds", 10, 0, "int", "runner EVAL_SEEDS")
put("NTuningConfigs", 12, 0, "int", "runner --budget")
put("NEvalFlows", 100000, 0, "int", "data.FoldConfig n_eval")
put("NSourceFlows", 100000, 0, "int", "data.FoldConfig n_src")
put("NBuffer", 1500, 0, "int", "data.FoldConfig n_buf")
put("NWindow", 500, 0, "int", "data.FoldConfig n_win")
put("WindowAttackPct", 5, 0, "int", "data.FoldConfig win_attack_frac")
put("NFeatVtwo", 41, 0, "int", "pool stats")
put("NFeatVthree", 49, 0, "int", "pool stats")
put("NRunsPerRelease", 40, 0, "int", "4 targets x 10 seeds")
put("NWindowBenign", 475, 0, "int", "data.FoldConfig")
put("NWindowAttack", 25, 0, "int", "data.FoldConfig")
put("NTrainPerSource", 80000, 0, "int", "80/20 split of n_src")
put("NValPerSource", 20000, 0, "int", "80/20 split of n_src")
put("NBoot", N_BOOT, 0, "int", "seed bootstrap resamples")
put("NPoolUniform", 500000, 0, "int", "build_pools.py")
put("NPoolBenign", 200000, 0, "int", "build_pools.py")
for fam, F in FAMS.items():
    s = json.load(open(os.path.join(POOLS, fam, f"NF-BoT-IoT-{'v2' if fam == 'nfv2' else 'v3'}_stats.json")))
    put(f"NPoolBenignBot{F}", s["pool_sizes"]["benign"], 0, "int", "pool stats")
put("NBenchRuns", 2 * 18 * 40, 0, "int", "2 releases x 18 methods x 40")
put("NPoisonRuns", 1920, 0, "int", "poisoning_runs.csv")

# ---------------------------------------------------------------- testbeds
for fam, F in FAMS.items():
    for t in TGT:
        s = json.load(open(os.path.join(POOLS, fam, f"{t}-{'v2' if fam == 'nfv2' else 'v3'}_stats.json")))
        T = TGT[t]
        put(f"Corpus{F}{T}", s["rows_total"], 0, "int", "pool stats")
        put(f"BenignPct{F}{T}", 100 * s["count"]["benign"] / s["rows_total"], 2, "percent", "pool stats")

def table_testbeds():
    names = {"Bot": "BoT-IoT", "Cse": "CSE-CIC-IDS2018", "Ton": "ToN-IoT", "Unsw": "UNSW-NB15"}
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{NF-v2} & \multicolumn{3}{c}{NF-v3} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Testbed & Flows & Benign\,\% & Classes & Flows & Benign\,\% & Classes \\", r"\midrule"]
    for t, T in sorted(TGT.items(), key=lambda kv: names[kv[1]]):
        cells = []
        for fam in FAMS:
            s = json.load(open(os.path.join(POOLS, fam, f"{t}-{'v2' if fam == 'nfv2' else 'v3'}_stats.json")))
            n_cls = sum(1 for k in s["attack_counts"] if k.lower() != "benign")
            cells += [f"{s['rows_total'] / 1e6:.1f}", f"{100 * s['count']['benign'] / s['rows_total']:.1f}", str(n_cls)]
        L.append(f"{names[T]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tables", "tab_testbeds.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


table_testbeds()

# ---------------------------------------------------------------- E1 benchmark
E1 = {}
for fam in FAMS:
    for m in METHODS:
        d = pd.read_csv(os.path.join(B, f"eval_{fam}__{m}.csv")).drop_duplicates(["target", "seed"])
        assert d.groupby(["target", "seed"]).ngroups == 40
        E1[(fam, m)] = d[["target", "seed", "AUROC", "TPR_at_FPR0.01", "TPR_at_FPR0.001"]].copy()

rows = []
for fam, F in FAMS.items():
    means = {}
    for m in METHODS:
        d = E1[(fam, m)]
        piv = d.pivot_table(index="seed", columns="target", values="AUROC")
        lo, hi = boot_ci(piv)
        means[m] = d.AUROC.mean()
        M = mname(m)
        put(f"Auc{F}{M}", means[m], 3, "plain", f"eval_{fam}__{m}.csv")
        put(f"Auc{F}{M}Lo", lo, 3, "plain", "seed bootstrap")
        put(f"Auc{F}{M}Hi", hi, 3, "plain", "seed bootstrap")
        r = {"family": fam, "method": DISPLAY.get(m, m), "setting": METHODS[m][3], "mean": means[m], "lo": lo, "hi": hi}
        for t, g in d.groupby("target"):
            put(f"Auc{F}{M}{tname(t)}", g.AUROC.mean(), 3, "plain", f"eval_{fam}__{m}.csv")
            r[tname(t)] = g.AUROC.mean()
        rows.append(r)
    s01 = {m: means[m] for m in S01}
    best = max(s01, key=s01.get)
    put(f"MinAucSZeroSOne{F}", min(s01.values()), 3, "plain", "E1")
    put(f"MaxAucSZeroSOne{F}", max(s01.values()), 3, "plain", "E1", f"best = {best}")
    for m in ("KNN-win", "IF-win"):
        x = E1[(fam, m)].merge(E1[(fam, best)], on=["target", "seed"])
        diff = x.AUROC_x - x.AUROC_y
        put(f"Gain{F}{mname(m)}OverBest", 100 * diff.mean(), 1, "pp", "E1", f"vs {best}")
        put(f"P{F}{mname(m)}OverBest", wil(diff), 1, "pvalue", "Wilcoxon over 40 pairs")
    x = E1[(fam, "KNN-win")].merge(E1[(fam, "ERM-CB")], on=["target", "seed"])
    put(f"Gain{F}KnnwinOverErmcb", 100 * (x.AUROC_x - x.AUROC_y).mean(), 1, "pp", "E1")
    # per-target: S0/S1 methods below 0.5, KNN-win beats every S0/S1 method
    for t in E1[(fam, "KNN-win")].target.unique():
        tm = {m: E1[(fam, m)][E1[(fam, m)].target == t].AUROC.mean() for m in S01}
        put(f"NBelowHalf{F}{tname(t)}", sum(v < 0.5 for v in tm.values()), 0, "int", "E1")
best_rows = pd.DataFrame(rows)
wins = 0
for fam in FAMS:
    for t in E1[(fam, "KNN-win")].target.unique():
        k = E1[(fam, "KNN-win")][E1[(fam, "KNN-win")].target == t].AUROC.mean()
        wins += all(k > E1[(fam, m)][E1[(fam, m)].target == t].AUROC.mean() for m in S01)
put("NTargetsKnnBeatsAllSZeroSOne", wins, 0, "int", "E1 per-target means")
cap_both = 0
for t in TGT:
    ok = []
    for fam in FAMS:
        tt = f"{t}-{'v2' if fam == 'nfv2' else 'v3'}"
        k = E1[(fam, "KNN-win")][E1[(fam, "KNN-win")].target == tt].AUROC.mean()
        ok.append(all(k > E1[(fam, m)][E1[(fam, m)].target == tt].AUROC.mean() for m in S01))
    cap_both += all(ok)
put("NCapturesKnnBeatsAllBoth", cap_both, 0, "int", "E1", "captures where KNN-win beats every S0/S1 method in both releases")

# network-level evidence: per-target gains of KNN-win over the best S0/S1 method of each release,
# leave-one-target-out means, and a sign test over the 8 targets (the unit is the network, not the seed)
from scipy.stats import binomtest  # noqa: E402
pos, gains_all = 0, []
for fam, F in FAMS.items():
    best = max(S01, key=lambda m: E1[(fam, m)].AUROC.mean())
    x = E1[(fam, "KNN-win")].merge(E1[(fam, best)], on=["target", "seed"])
    per_t = (100 * (x.AUROC_x - x.AUROC_y)).groupby(x.target).mean()
    gains_all += list(per_t.values)
    pos += int((per_t > 0).sum())
    put(f"TgtGainMin{F}", per_t.min(), 1, "pp", "E1", f"KNN-win - {best}, per target")
    put(f"TgtGainMax{F}", per_t.max(), 1, "pp", "E1", f"KNN-win - {best}, per target")
    loo = [per_t.drop(t).mean() for t in per_t.index]
    put(f"LooGainMin{F}", min(loo), 1, "pp", "E1", "mean gain with one target left out")
    put(f"LooGainMax{F}", max(loo), 1, "pp", "E1", "mean gain with one target left out")
put("NTargetsPosGain", pos, 0, "int", "E1", "of 8 targets, vs the release's best S0/S1 method")
put("SignTestP", binomtest(wins, 8, 0.5, alternative="greater").pvalue, 1, "pvalue", "one-sided sign test over 8 targets",
    "KNN-win beats every S0/S1 method")

# H-B1: no S0/S1 method gains >= 5 pp mean AUROC over ERM-CB in both releases
gain_cb = {}
for fam, F in FAMS.items():
    for m in S01:
        x = E1[(fam, m)].merge(E1[(fam, "ERM-CB")], on=["target", "seed"])
        gain_cb[(fam, m)] = 100 * (x.AUROC_x - x.AUROC_y).mean()
        put(f"GainOverErmCb{F}{mname(m)}", gain_cb[(fam, m)], 1, "pp", "E1")
    put(f"MaxGainOverErmCb{F}", max(gain_cb[(fam, m)] for m in S01 if m != "ERM-CB"), 1, "pp", "E1")
put("NGainFiveBothReleases", sum(all(gain_cb[(f, m)] >= 5 for f in FAMS) for m in S01 if m != "ERM-CB"), 0, "int", "H-B1")
# H-B2: marginal alignment vs ERM-CB on the extreme-prior target BoT-IoT
for fam, F in FAMS.items():
    for m in ("CORAL", "DANN"):
        a = E1[(fam, m)][E1[(fam, m)].target.str.startswith("NF-BoT")].AUROC.mean()
        b = E1[(fam, "ERM-CB")][E1[(fam, "ERM-CB")].target.str.startswith("NF-BoT")].AUROC.mean()
        put(f"BotGap{F}{mname(m)}", 100 * (a - b), 1, "pp", "E1", "method - ERM-CB on BoT-IoT")
put("AlignBotMaxLoss", -min(r["value"] for r in ledger if r["key"].startswith("BotGap")), 1, "pp", "E1",
    "largest loss of CORAL/DANN vs ERM-CB on BoT-IoT, both releases")

# Table: main benchmark
def fmt(v):
    return f"{v:.3f}"


def table_main():
    """Main benchmark table, NF-v2 and NF-v3 side by side."""
    order = [m for s in ("S0", "S1", "S2") for m in METHODS if METHODS[m][3] == s]
    L = [r"\begin{tabular}{@{}ll@{\hspace{6pt}}lcccc@{\hspace{10pt}}lcccc@{}}", r"\toprule",
         r" & & \multicolumn{5}{c}{NF-v2} & \multicolumn{5}{c}{NF-v3} \\",
         r"\cmidrule(lr){3-7}\cmidrule(l){8-12}",
         r"Setting & Method & Mean [95\% CI] & BoT & CIC & ToN & UNSW & Mean [95\% CI] & BoT & CIC & ToN & UNSW \\",
         r"\midrule"]

    def cells(fam, m):
        sub = best_rows[best_rows.family == fam].set_index("method")
        r = sub.loc[DISPLAY.get(m, m)]
        out = []
        mean = fmt(r["mean"])
        if r["mean"] == sub["mean"].max():
            mean = rf"\textbf{{{mean}}}"
        out.append(f"{mean} [{fmt(r.lo)}, {fmt(r.hi)}]")
        for T in ("Bot", "Cse", "Ton", "Unsw"):
            v = r[T]
            s_ = fmt(v)
            if v == sub[T].max():
                s_ = rf"\textbf{{{s_}}}"
            elif v < 0.5:
                s_ = rf"\underline{{{s_}}}"
            out.append(s_)
        return out

    for i, m in enumerate(order):
        new_group = i == 0 or METHODS[order[i - 1]][3] != METHODS[m][3]
        if new_group and i:
            L.append(r"\addlinespace[2pt]")
        setting = METHODS[m][3] if new_group else ""
        L.append(f"{setting} & {DISPLAY.get(m, m)} & " + " & ".join(cells("nfv2", m) + cells("nfv3", m)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tables", "tab_main_auroc.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

table_main()

# ---------------------------------------------------------------- decision layer (all probabilistic methods)
dl = pd.read_csv(os.path.join(B, "analysis", "decision_layer.csv"))
for fam, F in FAMS.items():
    d = dl[dl.family == fam]
    loss = (d.src_val_macro_f1 - d.em).mean()
    put(f"DlLoss{F}", 100 * loss, 1, "pp", "decision_layer.csv")
    put(f"DlSrcFOne{F}", d.src_val_macro_f1.mean(), 3, "plain", "decision_layer.csv")
    put(f"DlEmFOne{F}", d.em.mean(), 3, "plain", "decision_layer.csv")
    put(f"DlOracleFOne{F}", d.oracle_threshold.mean(), 3, "plain", "decision_layer.csv")
    put(f"DlRecThr{F}", 100 * (d.oracle_threshold - d.em).mean() / loss, 1, "percent", "decision_layer.csv")
    put(f"DlRecPrior{F}", 100 * (d.em_oracle_prior - d.em).mean() / loss, 1, "percent", "decision_layer.csv")
    put(f"DlRankLimited{F}", 100 * (d.src_val_macro_f1 - d.oracle_threshold).mean(), 1, "pp", "decision_layer.csv")
    put(f"DlBbseMinusEm{F}", 100 * (d.bbse - d.em).mean(), 1, "pp", "decision_layer.csv")
# prior-controlled split of the threshold-irrecoverable loss (prior_controlled_decomposition.py)
pc = pd.read_csv(os.path.join(B, "analysis", "prior_controlled.csv"))
assert len(pc) == len(dl)
for fam, F in FAMS.items():
    g = pc[pc.family == fam]
    put(f"PcLoss{F}", 100 * (g.F_S - g.F_T_best).mean(), 1, "pp", "prior_controlled.csv")
    put(f"PcPrior{F}", 100 * (g.F_S_best - g.F_S_prior).mean(), 1, "pp", "prior_controlled.csv")
    put(f"PcRank{F}", 100 * (g.F_S_prior - g.F_T_best).mean(), 1, "pp", "prior_controlled.csv")
    put(f"PcRankShare{F}", 100 * (g.F_S_prior - g.F_T_best).mean() / (g.F_S - g.F_T_best).mean(), 0, "percent", "prior_controlled.csv")
    b = g[g.target.str.startswith("NF-BoT")]
    put(f"PcPriorBot{F}", 100 * (b.F_S_best - b.F_S_prior).mean(), 1, "pp", "prior_controlled.csv")
    put(f"PcRankBot{F}", 100 * (b.F_S_prior - b.F_T_best).mean(), 1, "pp", "prior_controlled.csv")
    put(f"PcAucSrc{F}", g.AUROC_src.mean(), 3, "plain", "prior_controlled.csv")
    put(f"PcAucTgt{F}", g.AUROC_tgt.mean(), 3, "plain", "prior_controlled.csv")
put("DlNRuns", len(dl), 0, "int", "decision_layer.csv")
put("DlNMethods", dl.method.nunique(), 0, "int", "decision_layer.csv")

# ---------------------------------------------------------------- E2 failure decomposition
fd = pd.read_csv(os.path.join(B, "failure_decomposition.csv"))
fd["setting"] = fd.method.map(lambda m: METHODS[m][3])
for fam, F in FAMS.items():
    g = fd[(fd.family == fam) & fd.setting.isin(["S0", "S1"])].groupby("method").tgt_benignFPR_at_srcFPR1.mean()
    put(f"TgtBenignFprMin{F}", 100 * g.min(), 0, "percent", "failure_decomposition.csv")
    put(f"TgtBenignFprMax{F}", 100 * g.max(), 0, "percent", "failure_decomposition.csv")
    s = fd[(fd.family == fam) & fd.setting.isin(["S0", "S1"])]
    put(f"SrcTprAtOne{F}", 100 * s.src_TPR_at_srcFPR1.mean(), 0, "percent", "failure_decomposition.csv")
    put(f"TgtTprAtOne{F}", 100 * s.tgt_TPR_at_srcFPR1.mean(), 0, "percent", "failure_decomposition.csv")
    put(f"AucSeen{F}", s.AUROC_seen.mean(), 3, "plain", "failure_decomposition.csv")
    put(f"AucNovel{F}", s.AUROC_novel.mean(), 3, "plain", "failure_decomposition.csv")
    put(f"SeenSharePctMin{F}", 100 * s.groupby("target").share_attacks_seen.mean().min(), 0, "percent", "failure_decomposition.csv")
    w_ = fd[(fd.family == fam) & fd.method.isin(["KNN-win", "IF-win"])]
    put(f"AucSeenWin{F}", w_.AUROC_seen.mean(), 3, "plain", "failure_decomposition.csv")
assert set(fd.method) == set(METHODS), "failure decomposition must cover all methods"
# H-F1: seen-family AUROC < 0.75 on >= 2 of 4 targets, in both releases
pt = fd.groupby(["family", "method", "target"]).AUROC_seen.mean().reset_index()
low = pt.assign(low=pt.AUROC_seen < 0.75).groupby(["family", "method"]).low.sum().unstack(0)
hf1 = (low >= 2).all(axis=1)
put("HFOneNHold", hf1[[m for m in S01]].sum(), 0, "int", "failure_decomposition.csv", "of 14 S0/S1 methods")
put("HFOneNHoldWin", hf1[["KNN-win", "IF-win"]].sum(), 0, "int", "failure_decomposition.csv")
put("HFOneMaxLowWin", low.loc[["KNN-win", "IF-win"]].values.max(), 0, "int", "failure_decomposition.csv")

# ---------------------------------------------------------------- operating points of the window detectors
for fam, F in FAMS.items():
    for m in ("KNN-win", "IF-win"):
        full = pd.read_csv(os.path.join(B, f"eval_{fam}__{m}.csv"))
        M = mname(m)
        for rule, R_ in (("win_mad", "Mad"), ("win_fpr", "Q")):
            g = full[full.rule == rule]
            put(f"Op{F}{M}{R_}FOne", g.MacroF1.mean(), 3, "plain", f"eval_{fam}__{m}.csv")
            put(f"Op{F}{M}{R_}Fpr", 100 * g.FPR_op.mean(), 1, "percent", f"eval_{fam}__{m}.csv")
            put(f"Op{F}{M}{R_}Tpr", 100 * g.TPR_op.mean(), 1, "percent", f"eval_{fam}__{m}.csv")
        put(f"Op{F}{M}TprAtOne", 100 * E1[(fam, m)]["TPR_at_FPR0.01"].mean(), 1, "percent", "E1")
        put(f"Op{F}{M}TprAtTenth", 100 * E1[(fam, m)]["TPR_at_FPR0.001"].mean(), 1, "percent", "E1")
        put(f"Op{F}{M}PrAuc", full[full.rule == "win_mad"].PRAUC.mean(), 3, "plain", f"eval_{fam}__{m}.csv")


def table_operating():
    """Operating points of the window detectors, per target (mean over 10 seeds)."""
    names = {"Bot": "BoT-IoT", "Cse": "CIC-IDS2018", "Ton": "ToN-IoT", "Unsw": "UNSW-NB15"}
    L = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
         r" & & & & \multicolumn{3}{c}{MAD rule ($Z=3$)} & \multicolumn{2}{c}{99th-pct.\ rule} & TPR at \\",
         r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
         r"Release & Target & AUROC & PR-AUC & FPR & TPR & Prec. & FPR & TPR & 1\% FPR \\", r"\midrule"]
    prior = dl.groupby(["target", "seed"]).prior_true.first()
    for fam, F in FAMS.items():
        full = pd.read_csv(os.path.join(B, f"eval_{fam}__KNN-win.csv"))
        rel = "NF-v2" if fam == "nfv2" else "NF-v3"
        rows = []
        for t, g in sorted(full.groupby("target"), key=lambda kv: names[tname(kv[0])]):
            md, q = g[g.rule == "win_mad"], g[g.rule == "win_fpr"]
            pi = np.array([prior.loc[(t, sd)] for sd in md.seed])
            prec = md.TPR_op.values * pi / np.maximum(md.TPR_op.values * pi + md.FPR_op.values * (1 - pi), 1e-12)
            rows.append([names[tname(t)], md.AUROC.mean(), md.PRAUC.mean(), 100 * md.FPR_op.mean(), 100 * md.TPR_op.mean(),
                         100 * prec.mean(), 100 * q.FPR_op.mean(), 100 * q.TPR_op.mean(), 100 * md["TPR_at_FPR0.01"].mean()])
        put(f"Op{F}KnnWinMadPrecMin", min(r[5] for r in rows), 1, "percent", f"eval_{fam}__KNN-win.csv", "per-target minimum")
        put(f"Op{F}KnnWinMadPrecMax", max(r[5] for r in rows), 1, "percent", f"eval_{fam}__KNN-win.csv", "per-target maximum")
        put(f"Op{F}KnnWinPrAucMin", min(r[2] for r in rows), 3, "plain", f"eval_{fam}__KNN-win.csv", "per-target minimum")
        put(f"Op{F}KnnWinPrAucMax", max(r[2] for r in rows), 3, "plain", f"eval_{fam}__KNN-win.csv", "per-target maximum")
        rows.append(["Mean"] + list(np.mean([r[1:] for r in rows], axis=0)))
        for i, r in enumerate(rows):
            lead = rel if i == 0 else ""
            if r[0] == "Mean":
                L.append(r"\cmidrule(l){2-10}")
            L.append(f"{lead} & {r[0]} & {r[1]:.3f} & {r[2]:.3f} & {r[3]:.1f} & {r[4]:.1f} & {r[5]:.1f} & {r[6]:.1f} & {r[7]:.1f} & {r[8]:.1f}" + r" \\")
        L.append(r"\midrule" if fam == "nfv2" else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(GEN, "tables", "tab_operating.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


table_operating()

# ---------------------------------------------------------------- E5-T temporal window (NF-v3)
t5 = pd.read_csv(os.path.join(R, "e5t", "e5t_runs.csv"))
c5 = pd.read_csv(os.path.join(R, "e5t", "e5t_s01_late.csv"))
prim = ["NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3", "NF-UNSW-NB15-v3"]
for cut, C in ((0.2, "Twenty"), (0.5, "Fifty")):
    k = t5[(t5.cut == cut) & (t5.method == "KNN-win")]
    i_ = t5[(t5.cut == cut) & (t5.method == "IF-win")]
    for name, g in (("Knnwin", k), ("Ifwin", i_)):
        e = g[g.window == "early"]
        put(f"EtEarly{name}{C}Prim", e[e.target.isin(prim)].AUROC.mean(), 3, "plain", "e5t_runs.csv")
        put(f"EtEarly{name}{C}All", e.AUROC.mean(), 3, "plain", "e5t_runs.csv")
        pv = g.pivot_table(index=["target", "seed"], columns="window", values="AUROC")
        pv = pv[pv.index.get_level_values(0).isin(prim)]
        put(f"EtCost{name}{C}", 100 * (pv.random - pv.early).mean(), 1, "pp", "e5t_runs.csv", "random - early, primary targets")
        cst = (100 * (pv.random - pv.early)).rename("d").reset_index()
        lo, hi = boot_ci_long(cst, "d")
        put(f"EtCost{name}{C}Lo", lo, 1, "pp", "seed bootstrap")
        put(f"EtCost{name}{C}Hi", hi, 1, "pp", "seed bootstrap")
    cc = c5[(c5.cut == cut) & c5.target.isin(prim)]
    best = cc.groupby("method").AUROC.mean().idxmax()
    x = k[(k.window == "early") & k.target.isin(prim)].merge(cc[cc.method == best], on=["target", "seed"])
    put(f"EtBestSZeroSOne{C}", cc[cc.method == best].AUROC.mean(), 3, "plain", "e5t_s01_late.csv", f"best = {best}")
    put(f"EtGain{C}", 100 * (x.AUROC_x - x.AUROC_y).mean(), 1, "pp", "e5t", f"KNN early vs {best}")
    put(f"EtP{C}", wil(x.AUROC_x - x.AUROC_y), 1, "pvalue", "Wilcoxon, n=30")
    for t in sorted(k.target.unique()):
        kt = k[k.target == t]
        put(f"Et{tname(t)}EarlyKnn{C}", kt[kt.window == "early"].AUROC.mean(), 3, "plain", "e5t_runs.csv")
        put(f"Et{tname(t)}RandomKnn{C}", kt[kt.window == "random"].AUROC.mean(), 3, "plain", "e5t_runs.csv")
        put(f"Et{tname(t)}BestSrc{C}", c5[(c5.cut == cut) & (c5.target == t)].groupby("method").AUROC.mean().max(),
            3, "plain", "e5t_s01_late.csv", "best S0/S1 method chosen per target with hindsight")
    bot = k[k.target == "NF-BoT-IoT-v3"]
    put(f"EtBotLateBenign{C}", bot.n_late_benign.max(), 0, "int", "e5t_runs.csv", "max over seeds")

# ---------------------------------------------------------------- E9 window size and contamination
e9 = pd.read_csv(os.path.join(R, "e9", "e9_runs.csv")).drop_duplicates(
    ["family", "target", "seed", "n_win", "contamination", "method"])
assert len(e9) == 1600
for fam, F in FAMS.items():
    for m in ("KNN-win", "IF-win"):
        g = e9[(e9.family == fam) & (e9.method == m)]
        M = mname(m)
        for n in (50, 100, 200, 500, 1000, 2000):
            put(f"Ws{F}{M}N{['Fifty','Hundred','TwoHundred','FiveHundred','Thousand','TwoThousand'][[50,100,200,500,1000,2000].index(n)]}",
                g[(g.n_win == n) & (g.contamination == 0.05)].AUROC.mean(), 3, "plain", "e9_runs.csv")
        for c, C in ((0.0, "Zero"), (0.10, "Ten"), (0.20, "Twenty"), (0.30, "Thirty")):
            put(f"Wc{F}{M}C{C}", g[(g.n_win == 500) & (g.contamination == c)].AUROC.mean(), 3, "plain", "e9_runs.csv")
        # consistency with E1 at the benchmark configuration
        e1_ = g[(g.n_win == 500) & (g.contamination == 0.05)].merge(E1[(fam, m)], on=["target", "seed"])
        assert len(e1_) == 40 and np.allclose(e1_.AUROC_x, e1_.AUROC_y, atol=1e-9)
    # H-W1 (n = 100, 5%) and H-W2 (n = 500, 20%): KNN-win minus the best benchmark S0/S1 method (paired)
    best = max(S01, key=lambda m: E1[(fam, m)].AUROC.mean())
    g = e9[(e9.family == fam) & (e9.method == "KNN-win")]
    for (n, c), H in (((100, 0.05), "WOne"), ((500, 0.20), "WTwo")):
        x = g[(g.n_win == n) & (g.contamination == c)].merge(E1[(fam, best)], on=["target", "seed"])
        assert len(x) == 40
        put(f"H{H}Gain{F}", 100 * (x.AUROC_x - x.AUROC_y).mean(), 1, "pp", "e9 vs E1", f"vs {best}")
        put(f"H{H}P{F}", wil(x.AUROC_x - x.AUROC_y), 1, "pvalue", "Wilcoxon, n=40")

# ---------------------------------------------------------------- E4 identifier-free
e4 = pd.read_csv(os.path.join(R, "e4", "e4_runs.csv"))
for fam, F in FAMS.items():
    for m in ("KNN-win", "IF-win", "ERM-CB", "XGB", "RF", "BRN"):
        a = e4[(e4.family == fam) & (e4.method == m)][["target", "seed", "AUROC"]].merge(E1[(fam, m)], on=["target", "seed"])
        M = mname(m)
        assert len(a) == 40
        put(f"Idf{F}{M}", a.AUROC_x.mean(), 3, "plain", "e4_runs.csv")
        put(f"Idf{F}{M}Delta", 100 * (a.AUROC_x - a.AUROC_y).mean(), 1, "pp", "e4 vs E1")
        lo, hi = boot_ci_long(a.assign(d=100 * (a.AUROC_x - a.AUROC_y)), "d")
        put(f"Idf{F}{M}DeltaLo", lo, 1, "pp", "seed bootstrap")
        put(f"Idf{F}{M}DeltaHi", hi, 1, "pp", "seed bootstrap")
    # H-I1: identifier-free KNN-win vs the best identifier-free source-trained method
    g4 = e4[e4.family == fam]
    best = max(("ERM-CB", "XGB", "RF"), key=lambda m: g4[g4.method == m].AUROC.mean())
    x = g4[g4.method == "KNN-win"].merge(g4[g4.method == best], on=["target", "seed"])
    put(f"HIOneGain{F}", 100 * (x.AUROC_x - x.AUROC_y).mean(), 1, "pp", "e4_runs.csv", f"vs {best}")
    put(f"HIOneP{F}", wil(x.AUROC_x - x.AUROC_y), 1, "pvalue", "Wilcoxon, n=40")
put("IdfNFeatVtwo", e4[e4.family == "nfv2"].n_features.max(), 0, "int", "e4_runs.csv")
put("IdfNFeatVthree", e4[e4.family == "nfv3"].n_features.max(), 0, "int", "e4_runs.csv")

# ---------------------------------------------------------------- E3 label budget
e3 = pd.read_csv(os.path.join(R, "e3", "e3_runs.csv"))
for fam, F in FAMS.items():
    for m, M in (("XGB-ST", "St"), ("XGB-T", "T")):
        for k, K in ((10, "Ten"), (30, "Thirty"), (100, "Hundred"), (300, "ThreeHundred"), (1000, "Thousand")):
            put(f"Lb{F}{M}{K}", e3[(e3.family == fam) & (e3.method == m) & (e3.k == k)].AUROC.mean(), 3, "plain", "e3_runs.csv")
    put(f"LbSingleClassTen{F}", ((e3.family == fam) & (e3.method == "XGB-T") & (e3.k == 10) & (e3.flag == "single_class")).sum(),
        0, "int", "e3_runs.csv")
    # H-L1: smallest k at which the better S3 variant reaches KNN-win's mean AUROC
    knn = E1[(fam, "KNN-win")].AUROC.mean()
    g3 = e3[e3.family == fam]
    best_k = g3.groupby(["k", "method"]).AUROC.mean().unstack().max(axis=1)
    reach = [k for k in sorted(best_k.index) if best_k[k] >= knn]
    put(f"LbExchange{F}", reach[0] if reach else np.inf, 0, "int", "e3 vs E1")
    # H-L2: at k = 100, the better S3 variant vs KNN-win (paired)
    bm = g3[g3.k == 100].groupby("method").AUROC.mean().idxmax()
    x = g3[(g3.k == 100) & (g3.method == bm)].merge(E1[(fam, "KNN-win")], on=["target", "seed"])
    assert len(x) == 40
    put(f"HLTwoGain{F}", 100 * (x.AUROC_x - x.AUROC_y).mean(), 1, "pp", "e3 vs E1", f"{bm} - KNN-win")
    put(f"HLTwoP{F}", wil(x.AUROC_x - x.AUROC_y), 1, "pvalue", "Wilcoxon, n=40")

# ---------------------------------------------------------------- E5 within vs cross network (NF-v3)
# Same exclusion rule as H-T1: targets whose late stream has < 100 benign flows (BoT-IoT-v3: 14-17)
# are excluded from the summary and reported separately.
e5 = pd.read_csv(os.path.join(R, "e5", "e5_runs.csv"))
assert (e5[e5.target == "NF-BoT-IoT-v3"].n_late_benign < 100).all()
assert (e5[e5.target.isin(prim)].n_late_benign >= 100).all()
for cut, C in ((0.2, "Twenty"), (0.5, "Fifty")):
    wins = 0
    for m in ("XGB", "RF", "ERM-CB"):
        w = e5[(e5.cut == cut) & (e5.method == m)][["target", "seed", "AUROC"]]
        x = c5[(c5.cut == cut) & (c5.method == m)][["target", "seed", "AUROC"]]
        mm = w.merge(x, on=["target", "seed"], suffixes=("_w", "_x"))
        assert len(mm) == 40
        M = mname(m)
        p_ = mm[mm.target.isin(prim)]
        put(f"Wn{C}{M}Within", p_.AUROC_w.mean(), 3, "plain", "e5_runs.csv", "3 targets")
        put(f"Wn{C}{M}Cross", p_.AUROC_x.mean(), 3, "plain", "e5t_s01_late.csv", "3 targets")
        put(f"Wn{C}{M}Diff", 100 * (p_.AUROC_w - p_.AUROC_x).mean(), 1, "pp", "e5", "3 targets")
        put(f"Wn{C}{M}P", wil(p_.AUROC_w - p_.AUROC_x), 1, "pvalue", "Wilcoxon, n=30")
        for t, g in mm.groupby("target"):
            put(f"Wn{C}{M}{tname(t)}Within", g.AUROC_w.mean(), 3, "plain", "e5_runs.csv")
            put(f"Wn{C}{M}{tname(t)}Cross", g.AUROC_x.mean(), 3, "plain", "e5t_s01_late.csv")
            wins += (t in prim) and g.AUROC_w.mean() > g.AUROC_x.mean()
    put(f"Wn{C}NWithinWins", wins, 0, "int", "e5", "of 9 target x model cells (3 targets)")
put("WnBotLateBenignMin", e5[e5.target == "NF-BoT-IoT-v3"].n_late_benign.min(), 0, "int", "e5_runs.csv")
put("WnBotLateBenignMax", e5[e5.target == "NF-BoT-IoT-v3"].n_late_benign.max(), 0, "int", "e5_runs.csv")

# ---------------------------------------------------------------- E6 audit
tl = pd.read_csv(os.path.join(R, "e6", "e6_transfer_losses.csv"))
put("AuditNPairs", len(tl[tl.method == "XGB"]), 0, "int", "e6_transfer_losses.csv")
for m in ("XGB", "RF", "ERM-CB"):
    x = tl[tl.method == m]
    for col, Cn in (("Mean_PSI", "Psi"), ("Mean_Wasserstein", "Wass"), ("Mean_Abs_Z_Diff", "Z")):
        put(f"Audit{mname(m)}{Cn}", spearmanr(x.Loss_AUROC, x[col]).correlation, 2, "plain", "e6_transfer_losses.csv")
rhos = [spearmanr(tl[tl.method == m].Loss_AUROC, tl[tl.method == m][c]).correlation
        for m in ("XGB", "RF", "ERM-CB") for c in ("Mean_PSI", "Mean_Wasserstein", "Mean_Abs_Z_Diff")]
rng_a = np.random.default_rng(BOOT_SEED)
n_excl, ci_lo, ci_hi = 0, [], []
for m in ("XGB", "RF", "ERM-CB"):
    x = tl[tl.method == m].reset_index(drop=True)
    for c in ("Mean_PSI", "Mean_Wasserstein", "Mean_Abs_Z_Diff"):
        bs = []
        for _ in range(2000):
            idx = rng_a.integers(0, len(x), len(x))
            r_ = spearmanr(x.Loss_AUROC.values[idx], x[c].values[idx]).correlation
            if np.isfinite(r_):
                bs.append(r_)
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        ci_lo.append(lo_); ci_hi.append(hi_)
        n_excl += int(lo_ > 0 or hi_ < 0)
put("AuditNCiExcludeZero", n_excl, 0, "int", "e6_transfer_losses.csv", "of 9 correlations; bootstrap over pairs")
put("AuditCiLoMin", min(ci_lo), 2, "plain", "bootstrap over pairs")
put("AuditCiHiMax", max(ci_hi), 2, "plain", "bootstrap over pairs")
put("AuditRhoMin", min(rhos), 2, "plain", "e6_transfer_losses.csv", "3 models x 3 divergences")
put("AuditRhoMax", max(rhos), 2, "plain", "e6_transfer_losses.csv", "3 models x 3 divergences")
pr = pd.read_csv(os.path.join(R, "e6", "e6_pairwise_runs.csv"))
put("AuditNSeeds", pr.seed.nunique(), 0, "int", "e6_pairwise_runs.csv")
put("AuditNEval", pr.n_eval.max(), 0, "int", "e6_pairwise_runs.csv")
put("AuditNTrain", pr.n_train.max(), 0, "int", "e6_pairwise_runs.csv")

# ---------------------------------------------------------------- exact-duplicate check (window vs evaluation stream)
dup = pd.read_csv(os.path.join(B, "analysis", "window_duplicates.csv"))
put("DupNSeeds", dup.seed.nunique(), 0, "int", "window_duplicates.csv")
put("DupBenignMaxPct", 100 * dup.groupby(["family", "target"]).dup_share_benign.mean().max(), 1, "percent", "window_duplicates.csv")
for fam, F in FAMS.items():
    g = dup[dup.family == fam]
    put(f"DupKnnDelta{F}", 100 * (g["AUROC_dedup[KNN-win]"] - g["AUROC_all[KNN-win]"]).mean(), 1, "pp", "window_duplicates.csv")
    put(f"DupKnnMaxAbsDelta{F}", 100 * (g["AUROC_dedup[KNN-win]"] - g["AUROC_all[KNN-win]"]).abs().groupby(g.target).mean().max(),
        1, "pp", "window_duplicates.csv")
put("PairInDist", pr[pr.is_transfer == False].AUROC.mean(), 3, "plain", "e6_pairwise_runs.csv")  # noqa: E712
put("PairTransfer", pr[pr.is_transfer == True].AUROC.mean(), 3, "plain", "e6_pairwise_runs.csv")  # noqa: E712

# ---------------------------------------------------------------- E7 feature groups
e7 = pd.read_csv(os.path.join(R, "e7", "e7_runs.csv"))
for fam, F in FAMS.items():
    for ab, A in (("all_features", "All"), ("drop_ttl", "Ttl"), ("drop_tcp_win", "Tcp"), ("drop_timing", "Timing"),
                  ("drop_artifacts", "Artifacts")):
        g = e7[(e7.family == fam) & (e7.ablation == ab)]
        assert len(g) == 40
        put(f"Fg{F}{A}Auc", g.AUROC.mean(), 3, "plain", "e7_runs.csv")
        put(f"Fg{F}{A}Fpr", 100 * g.tgt_benign_FPR.mean(), 1, "percent", "e7_runs.csv")
        put(f"Fg{F}{A}NFeat", g.n_features.iloc[0], 0, "int", "e7_runs.csv")


def table_feature_groups():
    labels = [("all_features", "All features"), ("drop_ttl", "$-$ TTL fields"), ("drop_tcp_win", "$-$ TCP window fields"),
              ("drop_timing", "$-$ timing fields"), ("drop_artifacts", "$-$ TTL, TCP window, identifiers")]
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{NF-v2} & \multicolumn{3}{c}{NF-v3} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Features & $d$ & AUROC & Benign FPR (\%) & $d$ & AUROC & Benign FPR (\%) \\", r"\midrule"]
    for ab, lab in labels:
        cells = []
        for fam in FAMS:
            g = e7[(e7.family == fam) & (e7.ablation == ab)]
            cells += [str(int(g.n_features.iloc[0])), f"{g.AUROC.mean():.3f}", f"{100 * g.tgt_benign_FPR.mean():.1f}"]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tables", "tab_feature_groups.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


table_feature_groups()

# ---------------------------------------------------------------- P2 poisoning
p2 = pd.read_csv(os.path.join(R, "poisoning", "poisoning_runs.csv"))
assert len(p2) == 1920 and not p2.duplicated(["family", "target", "seed", "p_poison", "method", "defense"]).any()
for fam, F in FAMS.items():
    for m in ("KNN-win", "IF-win"):
        g = p2[(p2.family == fam) & (p2.method == m)]
        M = mname(m)
        for p, P in ((0.0, "Zero"), (0.05, "Five"), (0.10, "Ten"), (0.20, "Twenty")):
            for d, D in (("naive", "Naive"), ("trimmed", "Trim")):
                put(f"Po{F}{M}{D}{P}", g[(g.p_poison == p) & (g.defense == d)].AUROC_target_family.mean(), 3, "plain", "poisoning_runs.csv")
        piv = g.pivot_table(index=["target", "seed"], columns=["defense", "p_poison"], values="AUROC_target_family")
        drop = piv[("naive", 0.0)] - piv[("naive", 0.1)]
        rec = piv[("trimmed", 0.1)] - piv[("naive", 0.1)]
        put(f"Po{F}{M}DropTen", 100 * drop.mean(), 1, "pp", "poisoning_runs.csv")
        put(f"Po{F}{M}DropTenP", wil(drop), 1, "pvalue", "Wilcoxon, n=40")
        put(f"Po{F}{M}RecShare", 100 * rec.mean() / drop.mean(), 0, "percent", "poisoning_runs.csv")
        oth = g.pivot_table(index=["target", "seed"], columns=["defense", "p_poison"], values="AUROC_other_families")
        od = (oth[("naive", 0.0)] - oth[("naive", 0.1)]).dropna()
        put(f"Po{F}{M}OtherDropTen", 100 * od.mean(), 1, "pp", "poisoning_runs.csv")
        put(f"Po{F}{M}OtherN", len(od), 0, "int", "poisoning_runs.csv", "target x seed pairs with other families")
        tp = g.pivot_table(index=["target", "seed"], columns=["defense", "p_poison"], values="TPR_target_at_FPR1")
        for p, P in ((0.0, "Zero"), (0.10, "Ten")):
            put(f"Po{F}{M}TprOne{P}", 100 * tp[("naive", p)].mean(), 1, "percent", "poisoning_runs.csv", "target family, 1% FPR")
        n0 = g[(g.p_poison == 0.10)]
        put(f"Po{F}{M}FprNaiveTen", 100 * n0[n0.defense == "naive"].FPR_benign_z3.mean(), 1, "percent", "poisoning_runs.csv")
        put(f"Po{F}{M}FprTrimTen", 100 * n0[n0.defense == "trimmed"].FPR_benign_z3.mean(), 1, "percent", "poisoning_runs.csv")

# ---------------------------------------------------------------- U: windows collected without label selection
U_PATH = os.path.join(R, "unselected", "u_runs.csv")
u = pd.read_csv(U_PATH) if os.path.exists(U_PATH) else pd.DataFrame()
ukey = ["family", "target", "seed", "window", "method", "defense"]
U_DONE = len(u) == 2 * 4 * 10 * 2 * 2 + 4 * 10 * 2 * 2 and not u.duplicated(ukey).any()
if not U_DONE:
    print(f"note: experiment U incomplete ({len(u)} of 480 rows); its values are skipped")
    u = pd.DataFrame(columns=ukey + ["win_attack_share", "AUROC"])
if U_DONE:
    uk = u[(u.method == "KNN-win") & (u.window == "rand")]
    n_beat, rows_u = 0, []
    names_u = {"Bot": "BoT-IoT", "Cse": "CIC-IDS2018", "Ton": "ToN-IoT", "Unsw": "UNSW-NB15"}
    for fam, F in FAMS.items():
        best = max(S01, key=lambda m: E1[(fam, m)].AUROC.mean())
        for t in sorted(uk[uk.family == fam].target.unique()):
            T = tname(t)
            g = uk[(uk.family == fam) & (uk.target == t)]
            nv, tr = g[g.defense == "naive"], g[g.defense == "trimmed"]
            src = E1[(fam, best)][E1[(fam, best)].target == t].AUROC.mean()
            sel = E1[(fam, "KNN-win")][E1[(fam, "KNN-win")].target == t].AUROC.mean()
            put(f"U{F}{T}Share", 100 * nv.win_attack_share.mean(), 1, "percent", "u_runs.csv")
            put(f"U{F}{T}Auc", nv.AUROC.mean(), 3, "plain", "u_runs.csv")
            put(f"U{F}{T}AucTrim", tr.AUROC.mean(), 3, "plain", "u_runs.csv")
            put(f"U{F}{T}BestSrc", src, 3, "plain", "E1", f"{best} on this target")
            n_beat += nv.AUROC.mean() > src
            rows_u.append((F, T, 100 * nv.win_attack_share.mean(), sel, nv.AUROC.mean(), tr.AUROC.mean(), src))
        h = uk[(uk.family == fam) & (uk.defense == "naive") & uk.target.str.contains("UNSW|CSE")]
        put(f"UBenignMajAuc{F}", h.AUROC.mean(), 3, "plain", "u_runs.csv", "UNSW-NB15 and CSE-CIC-IDS2018")
    put("UNTargetsBeatBest", n_beat, 0, "int", "u_runs.csv", "of 8 targets, unselected window vs release-best S0/S1")
    hu1 = all(r[4] > r[6] for r in rows_u if r[1] in ("Unsw", "Cse"))
    put("HUOneHolds", int(hu1), 0, "int", "H-U1: 1 = holds")
    # U-early (NF-v3): unselected window from the first 20% of the timeline, late flows
    ue = u[(u.window == "early") & (u.method == "KNN-win") & (u.defense == "naive")]
    for t in sorted(ue.target.unique()):
        g = ue[ue.target == t]
        put(f"UEarly{tname(t)}Share", 100 * g.win_attack_share.mean(), 1, "percent", "u_runs.csv")
        put(f"UEarly{tname(t)}Auc", g.AUROC.mean(), 3, "plain", "u_runs.csv")
    cc = c5[(c5.cut == 0.2) & c5.target.isin(prim)]
    best_late = cc.groupby("method").AUROC.mean().idxmax()
    x = ue[ue.target.isin(prim)][["target", "seed", "AUROC"]].merge(cc[cc.method == best_late], on=["target", "seed"])
    assert len(x) == 30
    put("UEarlyPrim", x.AUROC_x.mean(), 3, "plain", "u_runs.csv", "3 interpretable NF-v3 targets")
    put("UEarlyGain", 100 * (x.AUROC_x - x.AUROC_y).mean(), 1, "pp", "u_runs.csv", f"vs {best_late} on late flows")
    put("UEarlyP", wil(x.AUROC_x - x.AUROC_y), 1, "pvalue", "Wilcoxon, n=30")


    def table_unselected():
        L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r" & & Window & \multicolumn{3}{c}{KNN-win AUROC} & Best \\", r"\cmidrule(lr){4-6}",
             r"Release & Target & attacks (\%) & selected & unselected & trimmed & S0/S1 \\", r"\midrule"]
        for i, (F, T, sh, sel, un, tr, src) in enumerate(rows_u):
            lead = ("NF-v2" if F == "Vtwo" else "NF-v3") if i % 4 == 0 else ""
            cell = (lambda v: rf"\textbf{{{v:.3f}}}" if v > src else f"{v:.3f}")
            L.append(f"{lead} & {names_u[T]} & {sh:.1f} & {sel:.3f} & {cell(un)} & {cell(tr)} & {src:.3f}" + r" \\")
            if i == 3:
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(GEN, "tables", "tab_unselected.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


    table_unselected()

# ---------------------------------------------------------------- second review: analyses of saved scores (post hoc)
lf = pd.read_csv(os.path.join(B, "analysis", "low_fpr.csv"))
assert len(lf) == 2 * 18 * 40
for fam, F in FAMS.items():
    g = lf[lf.family == fam].groupby("method")[["pAUC01", "TPR_at_FPR0.01", "TPR_at_FPR0.001"]].mean()
    put(f"PaucKnn{F}", g.loc["KNN-win", "pAUC01"], 3, "plain", "low_fpr.csv")
    put(f"PaucMin{F}", g.pAUC01.min(), 3, "plain", "low_fpr.csv", "all 18 methods")
    put(f"PaucMax{F}", g.pAUC01.max(), 3, "plain", "low_fpr.csv", f"max: {g.pAUC01.idxmax()}")
    put(f"TprOneMax{F}", 100 * g["TPR_at_FPR0.01"].max(), 1, "percent", "low_fpr.csv", f"max: {g['TPR_at_FPR0.01'].idxmax()}")
    put(f"NTprOneAboveKnn{F}", int((g["TPR_at_FPR0.01"] > g.loc["KNN-win", "TPR_at_FPR0.01"]).sum()), 0, "int", "low_fpr.csv")
    for m in ("IW-DANN", "CORAL", "BRN-XGB", "IRM"):
        put(f"TprOne{F}{mname(m)}", 100 * g.loc[m, "TPR_at_FPR0.01"], 1, "percent", "low_fpr.csv")


def table_lowfpr():
    """TPR at 1% FPR and standardised pAUC (FPR <= 1%) of all methods, both releases."""
    order = [m for s in ("S0", "S1", "S2") for m in METHODS if METHODS[m][3] == s]
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{NF-v2} & \multicolumn{3}{c}{NF-v3} \\", r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"Setting & Method & AUROC & pAUC$_{1\%}$ & TPR$_{1\%}$ & AUROC & pAUC$_{1\%}$ & TPR$_{1\%}$ \\", r"\midrule"]
    G = {fam: lf[lf.family == fam].groupby("method")[["AUROC", "pAUC01", "TPR_at_FPR0.01"]].mean() for fam in FAMS}
    best = {fam: G[fam].max() for fam in FAMS}
    for i, m in enumerate(order):
        new = i == 0 or METHODS[order[i - 1]][3] != METHODS[m][3]
        if new and i:
            L.append(r"\addlinespace[2pt]")
        cells = []
        for fam in FAMS:
            for c, f in (("AUROC", "{:.3f}"), ("pAUC01", "{:.3f}"), ("TPR_at_FPR0.01", "{:.1f}")):
                v = G[fam].loc[m, c] * (100 if c.startswith("TPR") else 1)
                s_ = f.format(v)
                cells.append(rf"\textbf{{{s_}}}" if G[fam].loc[m, c] == best[fam][c] else s_)
        L.append(f"{METHODS[m][3] if new else ''} & {DISPLAY.get(m, m)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tables", "tab_lowfpr.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


table_lowfpr()

fu = pd.read_csv(os.path.join(B, "analysis", "fusion.csv"))
for fam, F in FAMS.items():
    g = fu[(fu.family == fam) & (fu.rule == "KNN+best")]
    put(f"FusAuc{F}", g.AUROC.mean(), 3, "plain", "fusion.csv", f"KNN-win + {g.partner.iloc[0]}")
    put(f"FusTprOne{F}", 100 * g["TPR_at_FPR0.01"].mean(), 1, "percent", "fusion.csv")
    put(f"FusPauc{F}", g.pAUC01.mean(), 3, "plain", "fusion.csv")
    put(f"FusAucDrop{F}", 100 * (E1[(fam, "KNN-win")].AUROC.mean() - g.AUROC.mean()), 1, "pp", "fusion.csv")
    k2 = lf[(lf.family == fam) & (lf.method == "KNN-win")]
    hc3 = (g["TPR_at_FPR0.01"].mean() > k2["TPR_at_FPR0.01"].mean()) and (k2.AUROC.mean() - g.AUROC.mean() <= 0.02)
    put(f"HCThree{F}", int(hc3), 0, "int", "H-C3: 1 = holds")

# label budget by target: attacks among the k labels, and XGB-ST runs whose labels hold no attack
st3 = e3[e3.method == "XGB-ST"]
for fam, F in FAMS.items():
    for t in sorted(st3[st3.family == fam].target.unique()):
        g = st3[(st3.family == fam) & (st3.target == t) & (st3.k == 30)]
        put(f"LbAtt{F}{tname(t)}Thirty", g.k_attacks.mean(), 1, "plain", "e3_runs.csv", "attacks among 30 labels")
        knn_t = E1[(fam, "KNN-win")][E1[(fam, "KNN-win")].target == t].AUROC.mean()
        bk = st3[(st3.family == fam) & (st3.target == t)].groupby("k").AUROC.mean()
        reach = [k for k in sorted(bk.index) if bk[k] >= knn_t]
        put(f"LbExch{F}{tname(t)}", reach[0] if reach else 2000, 0, "int", "e3 vs E1", "2000 = not reached by k = 1000")
z0 = st3[(st3.k_attacks == 0) & st3.target.str.contains("UNSW")]
put("LbZeroAttUnswN", len(z0), 0, "int", "e3_runs.csv", "XGB-ST runs on UNSW with no attack among the labels")
put("LbZeroAttUnswMin", z0.AUROC.min(), 3, "plain", "e3_runs.csv")
put("LbZeroAttUnswMax", z0.AUROC.max(), 3, "plain", "e3_runs.csv")
put("LbZeroAttUnswMean", z0.AUROC.mean(), 3, "plain", "e3_runs.csv")
z1 = st3[(st3.k_attacks == 0) & st3.target.str.contains("CSE")]
put("LbZeroAttCseMean", z1.AUROC.mean(), 3, "plain", "e3_runs.csv")
_ub = {(fam, m): E1[(fam, m)][E1[(fam, m)].target.str.contains("UNSW")].AUROC.mean() for fam in FAMS for m in S01}
put("UnswBestSrcAny", max(_ub.values()), 3, "plain", "E1",
    f"best S0/S1 method on UNSW-NB15 in either release, chosen with hindsight: {max(_ub, key=_ub.get)}")

# without BoT-IoT; margins on the near-tie targets; source vs target attack share
for fam, F in FAMS.items():
    nb = lambda m: E1[(fam, m)][~E1[(fam, m)].target.str.startswith("NF-BoT")].AUROC.mean()  # noqa: E731
    s01nb = {m: nb(m) for m in S01}
    put(f"NoBotKnn{F}", nb("KNN-win"), 3, "plain", "E1", "3 targets, BoT-IoT excluded")
    put(f"NoBotBestSrc{F}", max(s01nb.values()), 3, "plain", "E1", f"best: {max(s01nb, key=s01nb.get)}")
    put(f"NoBotGain{F}", 100 * (nb("KNN-win") - max(s01nb.values())), 1, "pp", "E1")
    bot = {m: E1[(fam, m)][E1[(fam, m)].target.str.startswith("NF-BoT")].AUROC.mean() for m in S01}
    put(f"BotMargin{F}", 100 * (E1[(fam, "KNN-win")][E1[(fam, "KNN-win")].target.str.startswith("NF-BoT")].AUROC.mean()
                                - max(bot.values())), 1, "pp", "E1", f"vs per-target best {max(bot, key=bot.get)}")
    suf = "v2" if fam == "nfv2" else "v3"
    share = {t: 1 - json.load(open(os.path.join(POOLS, fam, f"{t}-{suf}_stats.json")))["count"]["benign"]
             / json.load(open(os.path.join(POOLS, fam, f"{t}-{suf}_stats.json")))["rows_total"] for t in TGT}
    for t, T in TGT.items():
        put(f"SrcAttShare{F}{T}", 100 * np.mean([share[s] for s in TGT if s != t]), 1, "percent", "pool stats",
            "mean attack share of the three sources")
        put(f"TgtAttShare{F}{T}", 100 * share[t], 1, "percent", "pool stats")
    for m in ("CORAL", "DANN"):
        u_ = {mm: E1[(fam, mm)][E1[(fam, mm)].target.str.startswith("NF-UNSW")].AUROC.mean() for mm in (m, "ERM-CB")}
        put(f"UnswGap{F}{mname(m)}", 100 * (u_[m] - u_["ERM-CB"]), 1, "pp", "E1", "method - ERM-CB on UNSW-NB15")

# precision of KNN-win's MAD rule at realistic attack prevalence (from its realised TPR and FPR)
for fam, F in FAMS.items():
    md = pd.read_csv(os.path.join(B, f"eval_{fam}__KNN-win.csv"))
    md = md[md.rule == "win_mad"]
    for pi, P in ((0.01, "One"), (0.001, "Tenth")):
        prec = md.TPR_op * pi / (md.TPR_op * pi + md.FPR_op * (1 - pi))
        put(f"PrecPrev{P}{F}", 100 * prec.mean(), 1, "percent", f"eval_{fam}__KNN-win.csv", f"MAD rule at prevalence {pi}")

# ---------------------------------------------------------------- second review: controls (H-C1, H-C2, H-C4, C2)
CT = os.path.join(R, "controls")
oc = pd.read_csv(os.path.join(CT, "oneclass.csv"))
xw = pd.read_csv(os.path.join(CT, "xgbwin.csv"))
nd = pd.read_csv(os.path.join(CT, "neardup.csv"))
pk = pd.read_csv(os.path.join(CT, "poisonk.csv"))
assert len(oc) == 320 and len(xw) == 80 and len(nd) == 80 and len(pk) == 960
ctl_rows, cap_win = [], {}
for fam, F in FAMS.items():
    kw = E1[(fam, "KNN-win")].groupby("target").AUROC.mean()
    ks = oc[(oc.family == fam) & (oc.method == "KNN-src") & (oc.n_ref == 500)].groupby("target").AUROC.mean()
    put(f"OcKnnSrc{F}", ks.mean(), 3, "plain", "oneclass.csv", "KNN on 500 source benign flows")
    put(f"OcKnnSrcBig{F}", oc[(oc.family == fam) & (oc.method == "KNN-src") & (oc.n_ref == 20000)].AUROC.mean(), 3, "plain", "oneclass.csv")
    put(f"OcIfSrc{F}", oc[(oc.family == fam) & (oc.method == "IF-src") & (oc.n_ref == 500)].AUROC.mean(), 3, "plain", "oneclass.csv")
    put(f"OcGap{F}", 100 * (kw.mean() - ks.mean()), 1, "pp", "oneclass.csv", "KNN-win - KNN-src(500)")
    x = E1[(fam, "KNN-win")].merge(oc[(oc.family == fam) & (oc.method == "KNN-src") & (oc.n_ref == 500)], on=["target", "seed"])
    put(f"OcP{F}", wil(x.AUROC_x - x.AUROC_y), 1, "pvalue", "Wilcoxon, n=40")
    for t in kw.index:
        cap_win.setdefault(tname(t), []).append(kw[t] - ks[t] >= 0.10)
        put(f"OcKnnSrc{F}{tname(t)}", ks[t], 3, "plain", "oneclass.csv")
    xs = xw[xw.family == fam].groupby("target").AUROC.mean()
    put(f"SwAuc{F}", xs.mean(), 3, "plain", "xgbwin.csv", "XGB on sources + window labelled benign")
    put(f"SwTprOne{F}", 100 * xw[xw.family == fam]["TPR_at_FPR0.01"].mean(), 1, "percent", "xgbwin.csv")
    best = max(S01, key=lambda m: E1[(fam, m)].AUROC.mean())
    bs = E1[(fam, best)].groupby("target").AUROC.mean()
    for t in sorted(kw.index, key=lambda t: {"Bot": 0, "Cse": 1, "Ton": 2, "Unsw": 3}[tname(t)]):
        ctl_rows.append((F, tname(t), kw[t], ks[t], oc[(oc.family == fam) & (oc.method == "IF-src") & (oc.n_ref == 500)
                                                     & (oc.target == t)].AUROC.mean(), xs[t], bs[t], best))
    g = nd[nd.family == fam]
    put(f"NdRemovedBenign{F}", 100 * g.removed_benign_p5.mean(), 1, "percent", "neardup.csv", "eps = 5th percentile")
    put(f"NdDrop{F}", 100 * (g.AUROC_all - g.AUROC_p5).mean(), 1, "pp", "neardup.csv")
    put(f"NdDropMax{F}", 100 * (g.AUROC_all - g.AUROC_p5).groupby(g.target).mean().max(), 1, "pp", "neardup.csv", "worst target")
    q = pk[pk.family == fam].pivot_table(index="p_poison", columns="k", values="AUROC_target_family")
    for k_, K_ in ((20, "Twenty"), (100, "Hundred"), (200, "TwoHundred")):
        put(f"PkClean{F}K{K_}", q.loc[0.0, k_], 3, "plain", "poisonk.csv")
        put(f"PkPois{F}K{K_}", q.loc[0.1, k_], 3, "plain", "poisonk.csv")
put("OcNCapturesBoth", sum(all(v) for v in cap_win.values()), 0, "int", "oneclass.csv",
    "captures where KNN-win beats KNN-src by >= 10 pp in both releases")
put("HCOneHolds", int(all(r["value"] >= 10 for r in ledger if r["key"] in ("OcGapVtwo", "OcGapVthree"))
                      and sum(all(v) for v in cap_win.values()) >= 3), 0, "int", "H-C1: 1 = holds")
put("HCTwoHolds", int(all(r["value"] < 5 for r in ledger if r["key"] in ("NdDropVtwo", "NdDropVthree"))), 0, "int", "H-C2")
put("HCFourHolds", int(all(xw[xw.family == f].AUROC.mean() < E1[(f, "KNN-win")].AUROC.mean() for f in FAMS)), 0, "int", "H-C4")


def table_controls():
    names = {"Bot": "BoT-IoT", "Cse": "CIC-IDS2018", "Ton": "ToN-IoT", "Unsw": "UNSW-NB15"}
    L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{KNN reference} & IF ref. & XGB + & Best \\", r"\cmidrule(lr){3-4}",
         r"Release & Target & target window & source benign & source benign & window & S0/S1 \\", r"\midrule"]
    for i, (F, T, kw_, ks_, is_, xs_, bs_, best) in enumerate(ctl_rows):
        lead = ("NF-v2" if F == "Vtwo" else "NF-v3") if i % 4 == 0 else ""
        vals = [kw_, ks_, is_, xs_, bs_]
        mx = max(vals)
        L.append(f"{lead} & {names[T]} & " + " & ".join(rf"\textbf{{{v:.3f}}}" if v == mx else f"{v:.3f}" for v in vals) + r" \\")
        if i == 3:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tables", "tab_controls.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


table_controls()

# poisoning mechanism: tuned k vs poisoned flows; poisoned flows removed by trimming
kk = [v["k"] for fam in FAMS for v in json.load(open(os.path.join(B, f"selected_{fam}__KNN-win.json"))).values()]
put("KnnKMin", min(kk), 0, "int", "selected_*__KNN-win.json")
put("KnnKMax", max(kk), 0, "int", "selected_*__KNN-win.json")
tr10 = p2[(p2.defense == "trimmed") & (p2.method == "KNN-win") & (p2.p_poison == 0.10)]
put("TrimPurged", tr10.purged_poison.mean(), 1, "plain", "poisoning_runs.csv", "poisoned flows removed by trimming, of 50")
put("NPoisonTen", 50, 0, "int", "10% of 500")

# ---------------------------------------------------------------- Holm correction over the declared paired tests
def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
    out = np.empty_like(adj)
    out[o] = adj
    return out


declared =["PVtwoKnnWinOverBest", "PVthreeKnnWinOverBest", "EtPTwenty", "HWOnePVtwo", "HWOnePVthree",
            "HWTwoPVtwo", "HWTwoPVthree", "HIOnePVtwo", "HIOnePVthree", "HLTwoPVtwo", "HLTwoPVthree",
            "PoVtwoKnnWinDropTenP", "PoVthreeKnnWinDropTenP"]
pv = {r["key"]: r["value"] for r in ledger if r["key"] in declared}
assert len(pv) == len(declared)
adj = dict(zip(declared, holm([pv[k] for k in declared])))
put("HolmNTests", len(declared), 0, "int", "declared paired tests")
sig = [k for k in declared if pv[k] < 0.05]
put("HolmMaxAdjSig", max(adj[k] for k in sig), 1, "pvalue", "max Holm-adjusted p among raw p < 0.05")
put("HolmNSig", sum(adj[k] < 0.05 for k in declared), 0, "int", "Holm-adjusted p < 0.05")

# ---------------------------------------------------------------- write
L = pd.DataFrame(ledger)
assert not L.key.duplicated().any(), L[L.key.duplicated()].key.tolist()
L.to_csv(os.path.join(OUT, "results_ledger.csv"), index=False)


def tex_value(r):
    v = r.value
    if r.kind == "pvalue":
        if v < 1e-3:
            e = int(np.floor(np.log10(v)))
            return rf"\ensuremath{{{v / 10 ** e:.1f}\times10^{{{e}}}}}"
        return f"{v:.3f}" if v < 0.01 else f"{v:.2f}"
    if r.kind == "int":
        return f"{int(round(v)):,}".replace(",", r"\,")
    s = f"{v:.{int(r.decimals)}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


with open(os.path.join(GEN, "results.tex"), "w", encoding="utf-8") as f:
    f.write("% Generated by paper/analysis/build_ledger.py from paper/results_ledger.csv. Do not edit.\n")
    for r in L.itertuples():
        f.write(rf"\newcommand{{\res{r.key}}}{{{tex_value(r)}}}" + "\n")
print(f"ledger: {len(L)} values -> results_ledger.csv, generated/results.tex, tables/tab_main_auroc.tex")
