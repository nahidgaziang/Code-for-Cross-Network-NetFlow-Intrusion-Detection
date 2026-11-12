"""
Failure decomposition from saved scores.

Is the cross-network loss caused by novel attacks (target attack families absent from the sources)
or by the network (benign traffic and known attacks look different)? For every saved run:
  1. attack families are harmonised across testbeds; a target family is "seen" if some source
     testbed has it at >= 0.1% of its flows;
  2. AUROC of target benign versus seen-family attacks, and versus novel-family attacks;
  3. at a threshold calibrated to 1% FPR on source validation benign flows: realised target
     benign FPR and target attack TPR.
  python analysis/failure_decomposition.py [results_dir]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "benchmark")
POOLS = os.environ.get("NF_POOL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pools"))
TESTBEDS = {"nfv2": ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"],
            "nfv3": ["NF-UNSW-NB15-v3", "NF-BoT-IoT-v3", "NF-ToN-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]}
SEEN_MIN_SHARE = 0.001


def family(name):
    n = name.lower()
    if n == "benign":
        return "Benign"
    if n.startswith("ddos"):
        return "DDoS"
    if n.startswith("dos"):
        return "DoS"
    if n in ("reconnaissance", "scanning"):
        return "Scan"
    if "xss" in n or "injection" in n:
        return "WebInjection"
    if "brute" in n or n == "password":
        return "BruteForce"
    return {"infilteration": "Infiltration"}.get(n, name.capitalize())


def testbed_families(fam):
    out = {}
    for tb in TESTBEDS[fam]:
        s = json.load(open(os.path.join(POOLS, fam, f"{tb}_stats.json")))
        shares = {}
        for k, v in s["attack_counts"].items():
            shares[family(k)] = shares.get(family(k), 0) + v / s["rows_total"]
        out[tb] = {"names": {int(k): v for k, v in s["attack_code_names"].items()}, "shares": shares}
    return out


def auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


rows = []
for fam in TESTBEDS:
    info = testbed_families(fam)
    for f in sorted(glob.glob(os.path.join(RES, f"scores_{fam}", "*.npz"))):
        target, method, seed = os.path.basename(f)[:-4].split("__")
        z = np.load(f)
        y, s, att = z["y_eval"].astype(int), z["p_eval"].astype(np.float64), z["attack_eval"]
        names = info[target]["names"]
        fams = np.array([family(names[int(a)]) for a in att])
        src_fams = set()
        for tb in TESTBEDS[fam]:
            if tb != target:
                src_fams |= {k for k, v in info[tb]["shares"].items() if v >= SEEN_MIN_SHARE and k != "Benign"}
        seen = (y == 1) & np.isin(fams, list(src_fams))
        novel = (y == 1) & ~np.isin(fams, list(src_fams))
        ben = y == 0
        r = {"family": fam, "target": target.replace("NF-", ""), "method": method, "seed": int(seed),
             "share_attacks_seen": float(seen.sum() / max(y.sum(), 1)),
             "AUROC_all": auroc(y, s),
             "AUROC_seen": auroc(np.r_[np.zeros(ben.sum()), np.ones(seen.sum())], np.r_[s[ben], s[seen]]) if seen.any() else np.nan,
             "AUROC_novel": auroc(np.r_[np.zeros(ben.sum()), np.ones(novel.sum())], np.r_[s[ben], s[novel]]) if novel.any() else np.nan}
        pv, yv = z["p_val"].astype(np.float64), z["y_val"].astype(int)
        thr = float(np.quantile(pv[yv == 0], 0.99))
        r.update({"src_TPR_at_srcFPR1": float(np.mean(pv[yv == 1] > thr)),
                  "tgt_benignFPR_at_srcFPR1": float(np.mean(s[ben] > thr)),
                  "tgt_TPR_at_srcFPR1": float(np.mean(s[y == 1] > thr))})
        for fm in sorted(set(fams[y == 1])):
            m = (y == 1) & (fams == fm)
            if m.sum() >= 30:
                r[f"AUROC[{fm}{'' if fm in src_fams else '*'}]"] = auroc(
                    np.r_[np.zeros(ben.sum()), np.ones(m.sum())], np.r_[s[ben], s[m]])
        rows.append(r)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(RES, "failure_decomposition.csv"), index=False)
lines = ["Failure decomposition (exploratory). '*' after a family = novel (absent from all sources).",
         f"Seen family = present in >= {SEEN_MIN_SHARE:.1%} of some source testbed's flows.\n"]
core = ["share_attacks_seen", "AUROC_all", "AUROC_seen", "AUROC_novel",
        "src_TPR_at_srcFPR1", "tgt_benignFPR_at_srcFPR1", "tgt_TPR_at_srcFPR1"]
for fam, g in d.groupby("family"):
    lines.append(f"===== {fam}: mean over methods and seeds, by target")
    lines.append(g.groupby("target")[core].mean().round(3).to_string())
    lines.append(f"\n{fam}: by method (mean over targets and seeds)")
    lines.append(g.groupby("method")[core].mean().round(3).to_string())
    fcols = [c for c in g.columns if c.startswith("AUROC[")]
    lines.append(f"\n{fam}: per attack family AUROC vs target benign (mean over methods and seeds)")
    lines.append(g.groupby("target")[fcols].mean().round(3).dropna(axis=1, how="all").to_string())
    lines.append("")
open(os.path.join(RES, "failure_decomposition_report.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
