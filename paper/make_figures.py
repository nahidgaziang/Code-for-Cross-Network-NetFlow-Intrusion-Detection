"""
Figures of the paper (Figs 2-7), generated from the same run-level result files as the ledger.
Vector PDFs are written to paper/figures/; --png also writes previews.
  python paper/make_figures.py [--png]
Encoding: release -> colour (Okabe-Ito blue / vermilion), method -> marker shape; uncertainty is a 95%
seed-bootstrap interval of the mean over targets.
Environment: NF_RESULTS_DIR (default: results_archive/).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
R = os.environ.get("NF_RESULTS_DIR", os.path.join(ROOT, "results_archive"))
B = os.path.join(R, "benchmark")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
from crossnet.methods import METHODS  # noqa: E402

PNG = "--png" in sys.argv
COLW, TEXTW = 3.49, 7.22                     # elsarticle 5p column and text width, inches
COL = {"nfv2": "#0072B2", "nfv3": "#D55E00"}  # validated: CVD dE 21.9, contrast >= 3:1 on white
WIN = "#009E73"                               # emphasis colour for the reference-window setting (S2)
INK, MUTED, FAINT, GRID = "#1a1a1a", "#6e6e6e", "#b8b8b8", "#e8e8e8"
REL = {"nfv2": "NF-v2", "nfv3": "NF-v3"}
TB = {"NF-BoT-IoT": "BoT-IoT", "NF-CSE-CIC-IDS2018": "CIC-IDS2018", "NF-ToN-IoT": "ToN-IoT", "NF-UNSW-NB15": "UNSW-NB15"}
TB_MARK = {"BoT-IoT": "o", "CIC-IDS2018": "s", "ToN-IoT": "^", "UNSW-NB15": "D"}
MMARK = {"KNN-win": "^", "IF-win": "D"}
DISPLAY = {"VREx": "V-REx"}   # display names for the paper
S01 = [m for m in METHODS if METHODS[m][3] in ("S0", "S1")]
BOOT_SEED, N_BOOT = 20261108, 10000

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.6, "lines.linewidth": 1.2,
    "lines.markersize": 4, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def tb(target):
    return TB[target.rsplit("-", 1)[0]] if target.startswith("NF-") else TB["NF-" + target.rsplit("-", 1)[0]]


def boot(df, col):
    """Mean over targets and seeds, with a seed-bootstrap 95% CI. df has columns target, seed, col."""
    piv = df.pivot_table(index="seed", columns="target", values=col)
    assert not piv.isna().any().any()
    v = piv.values.mean(1)
    rng = np.random.default_rng(BOOT_SEED)
    m = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    if PNG:
        fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", name)


def panel_label(ax, s, x=-0.02):
    ax.text(x, 1.02, s, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


# ---------------------------------------------------------------- data
E1 = {}
for fam in REL:
    for m in METHODS:
        d = pd.read_csv(os.path.join(B, f"eval_{fam}__{m}.csv")).drop_duplicates(["target", "seed"])
        assert d.groupby(["target", "seed"]).ngroups == 40
        E1[(fam, m)] = d[["target", "seed", "AUROC"]].copy()
best_s01 = {fam: max(S01, key=lambda m: E1[(fam, m)].AUROC.mean()) for fam in REL}


def ref_line(ax, fam, label=True, x=None):
    v = E1[(fam, best_s01[fam])].AUROC.mean()
    ax.axhline(v, color=COL[fam], lw=0.6, alpha=0.55, zorder=1)
    return v


# ---------------------------------------------------------------- Fig. 2: main benchmark
def fig_main():
    order = [m for s in ("S0", "S1", "S2") for m in METHODS if METHODS[m][3] == s]
    ypos = {}
    y = 0
    for i, m in enumerate(order):
        if i and METHODS[m][3] != METHODS[order[i - 1]][3]:
            y += 0.7
        ypos[m] = y
        y += 1
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 3.05), sharey=True)
    for ax, fam in zip(axes, REL):
        ax.axvline(0.5, color=MUTED, lw=0.6, zorder=1)
        for s in ("S0", "S1", "S2"):
            ms = [m for m in order if METHODS[m][3] == s]
            if s == "S1":
                ax.axhspan(ypos[ms[0]] - 0.6, ypos[ms[-1]] + 0.6, color="#f4f4f4", zorder=0, lw=0)
        for m in order:
            d = E1[(fam, m)]
            for t, g in d.groupby("target"):
                ax.plot(g.AUROC.mean(), ypos[m], marker=TB_MARK[tb(t)], ms=3.2, mfc="white", mec=FAINT,
                        mew=0.7, ls="none", zorder=2)
            mu, lo, hi = boot(d, "AUROC")
            c = WIN if METHODS[m][3] == "S2" else INK
            ax.plot([lo, hi], [ypos[m]] * 2, color=c, lw=1.4, solid_capstyle="round", zorder=3)
            ax.plot(mu, ypos[m], "o", ms=4.2, color=c, mec="white", mew=0.6, zorder=4)
        ax.set_xlim(0.05, 1.0)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_xlabel("AUROC on the unseen target testbed")
        ax.set_title(REL[fam], loc="left", fontweight="bold")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks([ypos[m] for m in order])
    axes[0].set_yticklabels([DISPLAY.get(m, m) for m in order])
    axes[0].invert_yaxis()
    for s, lab in (("S0", "S0: source only"), ("S1", "S1: unlabelled buffer"), ("S2", "S2: window")):
        ms = [m for m in order if METHODS[m][3] == s]
        y0, y1 = ypos[ms[0]] - 0.35, ypos[ms[-1]] + 0.35
        axes[0].annotate("", xy=(-0.205, y0), xytext=(-0.205, y1), xycoords=("axes fraction", "data"),
                         arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.8), annotation_clip=False)
        axes[0].annotate(lab, xy=(-0.225, (y0 + y1) / 2), xycoords=("axes fraction", "data"), rotation=90,
                         ha="right", va="center", fontsize=7, color=MUTED, annotation_clip=False)
    hs = [Line2D([], [], marker=TB_MARK[k], ls="none", mfc="white", mec=FAINT, mew=0.7, ms=3.6, label=k) for k in TB_MARK]
    hs += [Line2D([], [], marker="o", color=INK, mec="white", lw=1.4, ms=4.2, label="mean over targets, 95% CI"),
           Line2D([], [], marker="o", color=WIN, mec="white", lw=1.4, ms=4.2, label="mean, reference-window setting")]
    fig.legend(handles=hs, loc="upper center", ncol=6, bbox_to_anchor=(0.55, 1.05), columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig_main_auroc")


# ---------------------------------------------------------------- Fig. 3: ranking vs decision; benign side
def fig_decision():
    dl = pd.read_csv(os.path.join(B, "analysis", "decision_layer.csv"))
    fd = pd.read_csv(os.path.join(B, "failure_decomposition.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.15), gridspec_kw={"width_ratios": [1.25, 1.1, 1.0]})
    # (a) Macro-F1 by decision rule
    ax = axes[0]
    rules = [("src_val_macro_f1", "Source validation\n(in-distribution)"), ("fixed", "Fixed threshold 0.5"),
             ("em", "EM prior correction"), ("bbse", "BBSE prior correction"),
             ("em_oracle_prior", "True target prior*"), ("oracle_threshold", "Best target threshold*")]
    for j, fam in enumerate(REL):
        d = dl[dl.family == fam]
        for i, (c, lab) in enumerate(rules):
            v = d.groupby(["target", "seed"])[c].mean().reset_index()
            mu, lo, hi = boot(v, c)
            yy = i + (-0.13 if j == 0 else 0.13)
            ax.plot([lo, hi], [yy] * 2, color=COL[fam], lw=1.2)
            ax.plot(mu, yy, "o" if j == 0 else "s", color=COL[fam], ms=3.8, mec="white", mew=0.5)
    d = dl[dl.family == "nfv2"]
    em_, thr_, src_ = d.em.mean(), d.oracle_threshold.mean(), d.src_val_macro_f1.mean()
    for a_, b_, lab in ((em_, thr_, "threshold"), (thr_, src_, "irrecoverable")):
        ax.annotate("", xy=(a_, 6.0), xytext=(b_, 6.0), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7,
                                                                          shrinkA=0, shrinkB=0))
        ax.text((a_ + b_) / 2, 6.15, lab, ha="center", va="top", fontsize=6.5, color=MUTED)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([r[1] for r in rules])
    ax.set_ylim(6.55, -0.5)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Macro-F1 on the target")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", color=COL["nfv2"], mec="white", label="NF-v2"),
                       Line2D([], [], marker="s", color=COL["nfv3"], mec="white", label="NF-v3")],
              loc="center right", handletextpad=0.3)
    panel_label(ax, "a", x=-0.62)
    # (b) target-benign FPR at the source-calibrated 1%-FPR threshold
    ax = axes[1]
    s = fd[fd.method.isin(S01)]
    pts = s.groupby(["family", "target", "method"]).tgt_benignFPR_at_srcFPR1.mean().reset_index()
    xt = []
    for j, fam in enumerate(REL):
        for i, t in enumerate(["BoT-IoT", "CIC-IDS2018", "ToN-IoT", "UNSW-NB15"]):
            x0 = i + 4.6 * j
            g = pts[(pts.family == fam) & (pts.target.map(tb) == t)]
            rng = np.random.default_rng(i + 10 * j)
            ax.plot(x0 + rng.uniform(-0.22, 0.22, len(g)), 100 * g.tgt_benignFPR_at_srcFPR1, "o", ms=2.6,
                    mfc="white", mec=COL[fam], mew=0.6, ls="none", zorder=3)
            med = 100 * g.tgt_benignFPR_at_srcFPR1.median()
            ax.plot([x0 - 0.3, x0 + 0.3], [med] * 2, color=COL[fam], lw=1.4, zorder=4)
            xt.append((x0, t))
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.set_xticks([x for x, _ in xt])
    ax.set_xticklabels([t for _, t in xt], rotation=90)
    ax.set_ylabel("Target benign flows flagged (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 8.5)
    ax.grid(axis="x", visible=False)
    for j, fam in enumerate(REL):
        ax.text(1.5 + 4.6 * j, 103, REL[fam], color=COL[fam], ha="center", va="bottom", fontweight="bold")
    panel_label(ax, "b", x=-0.2)
    # (c) AUROC on attack families present in the sources
    ax = axes[2]
    rows = []
    for fam in REL:
        for t in ["BoT-IoT", "CIC-IDS2018", "ToN-IoT", "UNSW-NB15"]:
            g = fd[(fd.family == fam) & (fd.target.map(tb) == t)]
            src = g[g.method.isin(S01)].groupby("method").AUROC_seen.mean()
            win = g[g.method.isin(["KNN-win", "IF-win"])].groupby("method").AUROC_seen.mean()
            rows.append((fam, t, src.min(), src.median(), src.max(), win.mean(), g.share_attacks_seen.mean()))
    for i, (fam, t, lo, med, hi, w, share) in enumerate(rows):
        yy = i + (0.6 if fam == "nfv3" else 0)
        ax.plot([lo, hi], [yy] * 2, color=FAINT, lw=1.6, solid_capstyle="round")
        ax.plot(med, yy, "o", color=COL[fam], ms=3.8, mec="white", mew=0.5)
        ax.plot(w, yy, "^", color=WIN, ms=4.2, mec="white", mew=0.5)
    ax.set_yticks([i + (0.6 if r[0] == "nfv3" else 0) for i, r in enumerate(rows)])
    ax.set_yticklabels([f"{r[1]} ({REL[r[0]][3:]})" for r in rows])
    ax.invert_yaxis()
    ax.axvline(0.5, color=MUTED, lw=0.6)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("AUROC, attack families seen in training")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=FAINT, lw=1.6, label="source-side range"),
                       Line2D([], [], marker="o", color=MUTED, ls="none", mec="white", label="source-side median"),
                       Line2D([], [], marker="^", color=WIN, ls="none", mec="white", label="window detectors")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, handletextpad=0.3, borderaxespad=0.1)
    panel_label(ax, "c", x=-0.42)
    fig.subplots_adjust(wspace=0.75)
    save(fig, "fig_decision")


# ---------------------------------------------------------------- Fig. 4: network shift vs time shift (NF-v3)
def fig_network_time():
    e5 = pd.read_csv(os.path.join(R, "e5", "e5_runs.csv"))
    c5 = pd.read_csv(os.path.join(R, "e5t", "e5t_s01_late.csv"))
    t5 = pd.read_csv(os.path.join(R, "e5t", "e5t_runs.csv"))
    prim = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, cut, lab in ((axes[0], 0.2, "a"), (axes[1], 0.5, "b")):
        yl = []
        i = 0
        for t in prim:
            for m in ("XGB", "RF", "ERM-CB"):
                w = e5[(e5.cut == cut) & (e5.method == m) & (e5.target == t)].AUROC.mean()
                x = c5[(c5.cut == cut) & (c5.method == m) & (c5.target == t)].AUROC.mean()
                ax.plot([x, w], [i, i], color=FAINT, lw=1.4, solid_capstyle="round", zorder=1)
                ax.plot(w, i, "o", color=INK, ms=4, mec="white", mew=0.5, zorder=3)
                ax.plot(x, i, "o", mfc="white", mec=INK, ms=3.6, mew=0.8, zorder=3)
                yl.append(f"{tb(t)} {m}")
                i += 1
            i += 0.5
        ax.set_yticks([j + 0.5 * (j // 3) for j in range(9)])
        ax.set_yticklabels(yl)
        ax.invert_yaxis()
        ax.set_xlim(0.2, 1.02)
        ax.axvline(0.5, color=MUTED, lw=0.6)
        ax.set_xlabel("AUROC on late target flows")
        ax.set_title(f"Training data: first {int(cut * 100)}%", loc="left", fontsize=7)
        ax.grid(axis="y", visible=False)
        panel_label(ax, lab, x=-0.55 if lab == "a" else -0.04)
    axes[1].set_yticklabels([])
    axes[0].legend(handles=[Line2D([], [], marker="o", color=INK, ls="none", mec="white", label="within network (target's early flows)"),
                            Line2D([], [], marker="o", mfc="white", mec=INK, ls="none", label="cross network (other testbeds)")],
                   loc="lower left", bbox_to_anchor=(-0.55, 1.08), ncol=2, handletextpad=0.2, columnspacing=1.0)
    # (c) early vs random reference window, first 20% of the timeline
    ax = axes[2]
    tg = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3"]
    for i, t in enumerate(tg):
        g = t5[(t5.cut == 0.2) & (t5.method == "KNN-win") & (t5.target == t)]
        e, r = g[g.window == "early"].AUROC.mean(), g[g.window == "random"].AUROC.mean()
        bst = c5[(c5.cut == 0.2) & (c5.target == t)].groupby("method").AUROC.mean().max()
        ax.plot([bst], [i], "o", mfc="white", mec=INK, ms=3.6, mew=0.8, zorder=3)
        ax.plot([r], [i], "^", color=FAINT, ms=4.4, mec="white", mew=0.5, zorder=3)
        ax.plot([e], [i], "^", color=WIN, ms=4.4, mec="white", mew=0.5, zorder=4)
    ax.set_yticks(range(4))
    ax.set_yticklabels([tb(t) + ("$^\\dagger$" if "BoT" in t else "") for t in tg])
    ax.invert_yaxis()
    ax.set_xlim(0.1, 1.02)
    ax.axvline(0.5, color=MUTED, lw=0.6)
    ax.set_xlabel("AUROC on late target flows")
    ax.set_title("Reference window: first 20%", loc="left", fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="^", color=WIN, ls="none", mec="white", ms=4.4, label="KNN-win, early window"),
                       Line2D([], [], marker="^", color=FAINT, ls="none", mec="white", ms=4.4, label="KNN-win, random-time window"),
                       Line2D([], [], marker="o", mfc="white", mec=INK, ls="none", ms=3.6, label="best source-side (per target)")],
                   loc="lower left", bbox_to_anchor=(-0.05, 1.08), ncol=1, handletextpad=0.2, borderaxespad=0)
    panel_label(ax, "c", x=-0.3)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig_network_time")


# ---------------------------------------------------------------- Fig. 5: reference-window size, contamination, identifiers
def fig_window():
    e9 = pd.read_csv(os.path.join(R, "e9", "e9_runs.csv")).drop_duplicates(
        ["family", "target", "seed", "n_win", "contamination", "method"])
    e4 = pd.read_csv(os.path.join(R, "e4", "e4_runs.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    specs = [(axes[0], "n_win", [50, 100, 200, 500, 1000, 2000], ("contamination", 0.05)),
             (axes[1], "contamination", [0.0, 0.05, 0.10, 0.20, 0.30], ("n_win", 500))]
    for ax, var, xs, (fix, fv) in specs:
        for fam in REL:
            for m in ("KNN-win", "IF-win"):
                g = e9[(e9.family == fam) & (e9.method == m) & (e9[fix] == fv)]
                stats = [boot(g[g[var] == x], "AUROC") for x in xs]
                xv = np.array(xs) * (100 if var == "contamination" else 1)
                mu = np.array([s[0] for s in stats])
                lo = np.array([s[1] for s in stats])
                hi = np.array([s[2] for s in stats])
                knn = m == "KNN-win"
                ax.fill_between(xv, lo, hi, color=COL[fam], alpha=0.12 if knn else 0.0, lw=0)
                ax.plot(xv, mu, marker=MMARK[m], color=COL[fam], lw=1.3 if knn else 0.7,
                        mfc=COL[fam] if knn else "white", mec=COL[fam] if not knn else "white", mew=0.6,
                        ms=4.8 if knn else 3.0, zorder=4 if knn else 3)
            ref_line(ax, fam)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUROC")
    axes[0].set_xscale("log")
    axes[0].set_xticks([50, 100, 200, 500, 1000, 2000])
    axes[0].set_xticklabels(["50", "100", "200", "500", "1k", "2k"])
    axes[0].minorticks_off()
    axes[0].set_xlabel("Reference-window size (flows)")
    axes[0].set_title("5% attack contamination", loc="left", fontsize=7)
    axes[1].set_xticks([0, 5, 10, 20, 30])
    axes[1].set_xlabel("Attack contamination of the window (%)")
    axes[1].set_title("500-flow window", loc="left", fontsize=7)
    for fam, va, dy in (("nfv2", "bottom", 0.006), ("nfv3", "top", -0.006)):
        axes[0].text(2000, E1[(fam, best_s01[fam])].AUROC.mean() + dy, f"best source-side, {REL[fam]}",
                     fontsize=6.5, color=COL[fam], ha="right", va=va)
    panel_label(axes[0], "a", x=-0.22)
    panel_label(axes[1], "b", x=-0.22)
    hs = [Line2D([], [], color=COL["nfv2"], lw=1.2, label="NF-v2"), Line2D([], [], color=COL["nfv3"], lw=1.2, label="NF-v3"),
          Line2D([], [], marker="^", color=MUTED, mec="white", lw=1.3, ms=4.8, label="KNN-win (band: 95% CI)"),
          Line2D([], [], marker="D", mfc="white", mec=MUTED, color=MUTED, lw=0.7, ms=3.0, label="IF-win")]
    axes[0].legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.22, 1.1), ncol=4, handletextpad=0.3, columnspacing=1.0)
    # (c) identifier-free features: change in AUROC
    ax = axes[2]
    ms_ = ["KNN-win", "IF-win", "ERM-CB", "XGB", "RF", "BRN"]
    for j, fam in enumerate(REL):
        for i, m in enumerate(ms_):
            a = e4[(e4.family == fam) & (e4.method == m)][["target", "seed", "AUROC"]].merge(E1[(fam, m)], on=["target", "seed"])
            a["d"] = 100 * (a.AUROC_x - a.AUROC_y)
            mu, lo, hi = boot(a, "d")
            yy = i + (-0.14 if j == 0 else 0.14)
            ax.plot([lo, hi], [yy] * 2, color=COL[fam], lw=1.2)
            ax.plot(mu, yy, "o" if j == 0 else "s", color=COL[fam], ms=3.6, mec="white", mew=0.5)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(ms_)))
    ax.set_yticklabels(ms_)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC change without identifiers (pp)")
    ax.set_title("Identifier fields removed", loc="left", fontsize=7)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "c", x=-0.3)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_window")


# ---------------------------------------------------------------- Fig. 6: poisoned reference window
def fig_poison():
    p2 = pd.read_csv(os.path.join(R, "poisoning", "poisoning_runs.csv"))
    ps = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    fig, axes = plt.subplots(2, 2, figsize=(COLW, 2.9), sharex=True, sharey=True)
    for r_, m in enumerate(("KNN-win", "IF-win")):
        for c_, fam in enumerate(REL):
            ax = axes[r_, c_]
            g = p2[(p2.family == fam) & (p2.method == m)]
            for col, d, sty in (("AUROC_target_family", "naive", "tn"), ("AUROC_target_family", "trimmed", "tt"),
                                ("AUROC_other_families", "naive", "on")):
                st = []
                for p in ps:
                    h = g[(g.p_poison == p) & (g.defense == d)][["target", "seed", col]].dropna()
                    h = h[h.groupby("target").seed.transform("count") == 10]
                    st.append(boot(h, col))
                mu = np.array([s[0] for s in st])
                x = 100 * np.array(ps)
                if sty == "tn":
                    ax.fill_between(x, [s[1] for s in st], [s[2] for s in st], color=COL[fam], alpha=0.14, lw=0)
                    ax.plot(x, mu, marker=MMARK[m], color=COL[fam], mec="white", mew=0.5, ms=4)
                elif sty == "tt":
                    ax.plot(x, mu, marker=MMARK[m], color=COL[fam], lw=0.8, mfc="white", mec=COL[fam], mew=0.7, ms=3.4)
                else:
                    ax.plot(x, mu, marker="o", color=FAINT, lw=0.9, ms=2.8, mec="white", mew=0.4)
            ax.axhline(0.5, color=MUTED, lw=0.6)
            ax.set_ylim(0.35, 1.0)
            ax.set_xticks([0, 5, 10, 20])
            if r_ == 0:
                ax.set_title(REL[fam], loc="left", fontweight="bold", color=COL[fam])
            if c_ == 0:
                ax.set_ylabel(f"{m}\nAUROC")
            if r_ == 1:
                ax.set_xlabel("Poisoned share of window (%)")
    hs = [Line2D([], [], color=INK, marker="o", mec="white", ms=3.6, label="targeted family"),
          Line2D([], [], color=INK, lw=0.8, marker="o", mfc="white", mec=INK, ms=3.2, label="targeted, trimmed"),
          Line2D([], [], color=FAINT, marker="o", mec="white", ms=2.8, label="other families")]
    fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.04), handletextpad=0.3, columnspacing=0.9)
    fig.subplots_adjust(hspace=0.18, wspace=0.12)
    save(fig, "fig_poison")


# ---------------------------------------------------------------- Fig. 7: label budget
def fig_labels():
    e3 = pd.read_csv(os.path.join(R, "e3", "e3_runs.csv"))
    ks = [10, 30, 100, 300, 1000]
    # same size and two-row structure as Fig. 6, which it sits beside in the paper
    fig, axes = plt.subplots(2, 1, figsize=(COLW, 2.9), sharex=True, sharey=True)
    for r_, (ax, fam) in enumerate(zip(axes, REL)):
        _, klo, khi = boot(E1[(fam, "KNN-win")], "AUROC")
        kmu = E1[(fam, "KNN-win")].AUROC.mean()
        ax.axhspan(klo, khi, color=WIN, alpha=0.18, lw=0)
        ax.axhline(kmu, color=WIN, lw=1.0)
        ax.axhline(E1[(fam, best_s01[fam])].AUROC.mean(), color=MUTED, lw=0.7)
        for m, filled in (("XGB-ST", True), ("XGB-T", False)):
            st = [boot(e3[(e3.family == fam) & (e3.method == m) & (e3.k == k)], "AUROC") for k in ks]
            ax.errorbar(ks, [s[0] for s in st], yerr=[[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]],
                        color=COL[fam], marker="o", ms=3.6 if filled else 3.2, mfc=COL[fam] if filled else "white",
                        mec=COL[fam] if not filled else "white", mew=0.7, lw=1.2 if filled else 0.8, capsize=0, elinewidth=0.7)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel(f"{REL[fam]}\nAUROC")
        ax.yaxis.label.set_color(COL[fam])
        if r_ == 1:
            ax.set_xlabel("Labelled target flows k")
    axes[0].text(11, E1[("nfv2", best_s01["nfv2"])].AUROC.mean() - 0.015, "best source-side", fontsize=6.5, color=MUTED, va="top")
    hs = [Line2D([], [], color=INK, marker="o", mec="white", label="XGB-ST (source + k)"),
          Line2D([], [], color=INK, lw=0.8, marker="o", mfc="white", mec=INK, label="XGB-T (k only)"),
          Line2D([], [], color=WIN, lw=4, alpha=0.4, label="KNN-win (500 flows)")]
    fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.04), handletextpad=0.3, columnspacing=0.8)
    fig.subplots_adjust(hspace=0.18)
    save(fig, "fig_labels")


if __name__ == "__main__":
    fig_main()
    fig_decision()
    fig_network_time()
    fig_window()
    fig_poison()
    fig_labels()
