"""
Main benchmark tables and hypothesis tests from the evaluation files.

AUROC on the target evaluation stream is the primary metric (rule-independent, one value per target
and seed). Paired Wilcoxon signed-rank tests over the 40 (target x seed) pairs per release, Holm
correction within each hypothesis family, seed-bootstrap 95% CIs (resampling the ten seeds); a claim
is reported as holding only if it holds in both releases.
  python analysis/benchmark_tests.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.methods import METHODS  # noqa: E402

RES = os.path.join(HERE, "..", "results", "benchmark")
OUT = os.path.join(RES, "analysis")
POOLS = os.environ.get("NF_POOL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pools"))
FAMS = ("nfv2", "nfv3")
REF = "ERM-CB"
rng = np.random.default_rng(20261103)
os.makedirs(OUT, exist_ok=True)
lines = []


def say(s=""):
    lines.append(s); print(s, flush=True)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def wil(d):
    d = np.asarray(d, float)
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)


def boot_ci(df, col="AUROC"):
    """Seed bootstrap of the mean over targets x seeds: resample seeds with replacement."""
    piv = df.pivot_table(index="seed", columns="target", values=col)
    m = piv.values; idx = rng.integers(0, len(m), size=(10000, len(m)))
    means = m[idx].mean(axis=(1, 2))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- load
rows = []
for fam in FAMS:
    for method in METHODS:
        ev = pd.read_csv(os.path.join(RES, f"eval_{fam}__{method}.csv"))
        ev["family"] = fam
        rows.append(ev)
EV = pd.concat(rows, ignore_index=True)
EV["target_short"] = EV.target.str.replace("NF-", "").str.replace("-v2", "").str.replace("-v3", "")
AU = EV.drop_duplicates(["family", "target", "method", "seed"])[
    ["family", "target", "target_short", "method", "setting", "seed", "AUROC"]].copy()
benign = {}
for fam in FAMS:
    for t in AU[AU.family == fam].target.unique():
        st = json.load(open(os.path.join(POOLS, fam, f"{t}_stats.json")))
        benign[(fam, t)] = 1 - st["count"]["attack"] / st["rows_total"]


def paired(fam, a, b, target=None, col="AUROC", df=None):
    df = AU if df is None else df
    x = df[(df.family == fam) & (df.method == a)]; y = df[(df.family == fam) & (df.method == b)]
    if target:
        x, y = x[x.target_short == target], y[y.target_short == target]
    m = x.merge(y, on=["target", "seed"], suffixes=("_a", "_b"))
    d = m[f"{col}_a"] - m[f"{col}_b"]
    return float(d.mean()), wil(d), len(d)


# ---------------------------------------------------------------- main tables
say("BENCHMARK - main results (AUROC on the unseen target; mean over 4 targets x 10 seeds)")
main = []
for (fam, method), d in AU.groupby(["family", "method"]):
    lo, hi = boot_ci(d)
    main.append({"family": fam, "method": method, "setting": METHODS[method][3], "AUROC": d.AUROC.mean(),
                 "CI_lo": lo, "CI_hi": hi, "AUROC_sd_over_seeds": d.groupby("seed").AUROC.mean().std()})
MAIN = pd.DataFrame(main)
MAIN.to_csv(os.path.join(OUT, "main_auroc.csv"), index=False)
wide = MAIN.pivot(index=["setting", "method"], columns="family", values="AUROC")
for fam in FAMS:
    wide[f"{fam} 95% CI"] = [f"[{r.CI_lo:.3f}, {r.CI_hi:.3f}]" for r in
                             MAIN[MAIN.family == fam].set_index(["setting", "method"]).loc[wide.index].itertuples()]
say(wide.round(3).sort_index().to_string()); say()

BYT = AU.groupby(["family", "target_short", "method"]).AUROC.mean().unstack("method").round(3)
BYT.to_csv(os.path.join(OUT, "main_by_target.csv"))
say("AUROC by target (mean over 10 seeds)"); say(BYT.T.to_string()); say()

f1 = EV[((EV.setting.isin(["S0", "S1"])) & (EV.rule == "em")) | ((EV.setting == "S2") & (EV.rule == "win_mad"))]
F1 = f1.groupby(["family", "method", "rule"]).MacroF1.mean().unstack("family").round(3)
F1.to_csv(os.path.join(OUT, "main_macro_f1.csv"))
say("Macro-F1 (secondary): em rule for S0/S1, win_mad rule for S2"); say(F1.to_string()); say()

# ---------------------------------------------------------------- hypotheses
H = []
say("=" * 100); say("HYPOTHESES (declared in the analysis plan)"); say("=" * 100)

# H-B1
s01 = [m for m in METHODS if METHODS[m][3] in ("S0", "S1") and m != REF]
say(f"\nH-B1: no S0/S1 method improves mean AUROC over {REF} by >= 5 pp in both families")
improves_both = []
tab = {}
for fam in FAMS:
    res = [paired(fam, m, REF) for m in s01]
    adj = holm([r[1] for r in res])
    tab[fam] = {m: (r[0], a) for m, r, a in zip(s01, res, adj)}
for m in s01:
    ok = [tab[f][m][0] >= 0.05 and tab[f][m][1] < 0.05 for f in FAMS]
    say(f"   {m:9s} dAUROC v2 {tab['nfv2'][m][0]:+.4f} (pHolm {tab['nfv2'][m][1]:.2g})  "
        f"v3 {tab['nfv3'][m][0]:+.4f} (pHolm {tab['nfv3'][m][1]:.2g})  >=5pp&sig both: {all(ok)}")
    if all(ok):
        improves_both.append(m)
    for f in FAMS:
        H.append({"hypothesis": "H-B1", "family": f, "comparison": f"{m} - {REF}", "mean_diff": tab[f][m][0],
                  "p_holm": tab[f][m][1]})
say(f"   => H-B1 {'HOLDS' if not improves_both else 'FAILS (' + ', '.join(improves_both) + ')'}")

# H-B2
say(f"\nH-B2: CORAL and DANN do worse than {REF} on the BoT-IoT target (AUROC), both families")
verdict = []
for fam in FAMS:
    res = [paired(fam, m, REF, target="BoT-IoT") for m in ("CORAL", "DANN")]
    adj = holm([r[1] for r in res])
    for m, r, a in zip(("CORAL", "DANN"), res, adj):
        worse = r[0] < 0 and a < 0.05
        verdict.append(worse)
        say(f"   {fam} {m:6s} dAUROC on BoT-IoT {r[0]:+.4f} (n={r[2]}, pHolm {a:.2g}) worse&sig: {worse}")
        H.append({"hypothesis": "H-B2", "family": fam, "comparison": f"{m} - {REF} @BoT-IoT",
                  "mean_diff": r[0], "p_holm": a})
say(f"   => H-B2 {'HOLDS' if all(verdict) else 'PARTIAL' if any(verdict) else 'FAILS'} "
    f"({sum(verdict)}/4 method x family cells worse & significant)")

# H-B3
say("\nH-B3: the best S2 method beats the best S0/S1 method on mean AUROC, both families")
v3 = []
for fam in FAMS:
    mm = MAIN[MAIN.family == fam].set_index("method")
    b2 = mm[mm.setting == "S2"].AUROC.idxmax(); b01 = mm[mm.setting.isin(["S0", "S1"])].AUROC.idxmax()
    d, p, n = paired(fam, b2, b01)
    v3.append(d > 0 and p < 0.05)
    say(f"   {fam}: best S2 {b2} {mm.AUROC[b2]:.4f} vs best S0/S1 {b01} {mm.AUROC[b01]:.4f}: "
        f"d {d:+.4f}, p {p:.2g} (n={n})")
    H.append({"hypothesis": "H-B3", "family": fam, "comparison": f"{b2} - {b01}", "mean_diff": d, "p_holm": p})
say(f"   => H-B3 {'HOLDS' if all(v3) else 'FAILS'}")

# H-B4
say("\nH-B4: BRN > DomNorm on AUROC (benign vs mixture reference), both families")
v4 = []
for fam in FAMS:
    d, p, n = paired(fam, "BRN", "DomNorm")
    v4.append(d > 0 and p < 0.05)
    say(f"   {fam}: BRN - DomNorm {d:+.4f}, p {p:.2g} (n={n})")
    H.append({"hypothesis": "H-B4", "family": fam, "comparison": "BRN - DomNorm", "mean_diff": d, "p_holm": p})
say(f"   => H-B4 {'HOLDS' if all(v4) else 'FAILS'}")

# H-B5
say(f"\nH-B5 (published CORAL result): (a) CORAL - {REF} >= 5 pp AUROC on UNSW-NB15, both families;"
    " (b) descriptive: gain vs target benign share")
v5 = []
gains = []
for fam in FAMS:
    d, p, n = paired(fam, "CORAL", REF, target="UNSW-NB15")
    v5.append(d >= 0.05 and p < 0.05)
    say(f"   (a) {fam}: CORAL - {REF} on UNSW-NB15 {d:+.4f}, p {p:.2g} (n={n})")
    H.append({"hypothesis": "H-B5a", "family": fam, "comparison": f"CORAL - {REF} @UNSW-NB15",
              "mean_diff": d, "p_holm": p})
    for t in AU[AU.family == fam].target.unique():
        ts = t.replace("NF-", "").replace("-v2", "").replace("-v3", "")
        gains.append((fam, ts, benign[(fam, t)], paired(fam, "CORAL", REF, target=ts)[0]))
say(f"   => H-B5(a) {'HOLDS' if all(v5) else 'FAILS'}")
g = pd.DataFrame(gains, columns=["family", "target", "benign_share", "CORAL_gain"])
rho = spearmanr(g.benign_share, g.CORAL_gain).correlation
say("   (b) per-target CORAL gain vs benign share:\n" + g.round(4).to_string(index=False))
say(f"       Spearman rho = {rho:+.3f} (n = 8, descriptive only)")
H.append({"hypothesis": "H-B5b", "family": "both", "comparison": "Spearman(benign share, CORAL gain)",
          "mean_diff": rho, "p_holm": np.nan})

# H-E8: HYB from saved scores
say("\nH-E8: HYB (mean of window-ECDF-normalised BRN and KNN-win scores) >= max(BRN, KNN-win) on mean AUROC")
hyb = []
for fam in FAMS:
    for t in AU[AU.family == fam].target.unique():
        for s in sorted(AU.seed.unique()):
            z = {m: np.load(os.path.join(RES, f"scores_{fam}", f"{t}__{m}__{s}.npz")) for m in ("BRN", "KNN-win")}
            y = z["BRN"]["y_eval"].astype(int)
            assert np.array_equal(y, z["KNN-win"]["y_eval"]), "eval streams differ between methods"
            norm = [np.searchsorted(np.sort(z[m]["p_win"].astype(np.float64)), z[m]["p_eval"].astype(np.float64),
                                    side="right") / len(z[m]["p_win"]) for m in ("BRN", "KNN-win")]
            hyb.append({"family": fam, "target": t, "seed": s, "method": "HYB",
                        "AUROC": roc_auc_score(y, (norm[0] + norm[1]) / 2)})
HYB = pd.DataFrame(hyb)
HYB.to_csv(os.path.join(OUT, "hyb_auroc.csv"), index=False)
v8 = []
for fam in FAMS:
    h = HYB[HYB.family == fam].AUROC.mean()
    mb = {m: AU[(AU.family == fam) & (AU.method == m)].AUROC.mean() for m in ("BRN", "KNN-win")}
    best = max(mb, key=mb.get)
    hd = HYB[HYB.family == fam][["target", "seed", "AUROC"]].merge(
        AU[(AU.family == fam) & (AU.method == best)][["target", "seed", "AUROC"]], on=["target", "seed"])
    d = hd.AUROC_x - hd.AUROC_y
    v8.append(h >= mb[best])
    say(f"   {fam}: HYB {h:.4f} vs BRN {mb['BRN']:.4f}, KNN-win {mb['KNN-win']:.4f}; HYB - {best} "
        f"{d.mean():+.4f}, p {wil(d):.2g}")
    H.append({"hypothesis": "H-E8", "family": fam, "comparison": f"HYB - {best}", "mean_diff": d.mean(),
              "p_holm": wil(d)})
say(f"   => H-E8 {'HOLDS' if all(v8) else 'FAILS'} (as declared: mean comparison)")

pd.DataFrame(H).to_csv(os.path.join(OUT, "hypotheses.csv"), index=False)
open(os.path.join(OUT, "hypotheses_report.txt"), "w", encoding="utf-8").write("\n".join(lines))
