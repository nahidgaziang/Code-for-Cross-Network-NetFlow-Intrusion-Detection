"""
Integrity checks for a completed benchmark (results/benchmark): every job complete; every run
evaluated under all its decision rules exactly once; selections reproduced from the tuning files;
score files present, finite and correctly shaped; and every metric recomputed from the saved scores
equal to the evaluation files.
  python tests/verify_benchmark.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.decision import decide  # noqa: E402
from crossnet.methods import METHODS  # noqa: E402
from crossnet.metrics import evaluate  # noqa: E402
from crossnet.runner import EVAL_SEEDS, rules_for  # noqa: E402

RES = os.path.join(HERE, "..", "results", "benchmark")
POOLS = os.environ.get("NF_POOL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pools"))
FAMS = ("nfv2", "nfv3")
problems = []
TOL = 1e-6


def check(cond, msg):
    if not cond:
        problems.append(msg)


for fam in FAMS:
    prior = {}
    for f in glob.glob(os.path.join(POOLS, fam, "*_stats.json")):
        st = json.load(open(f))
        prior[os.path.basename(f)[:-11]] = st["count"]["attack"] / st["rows_total"]
    max_dev = {}
    for method in METHODS:
        tag = f"{fam}:{method}"
        tpath = os.path.join(RES, f"tune_{fam}__{method}.csv")
        epath = os.path.join(RES, f"eval_{fam}__{method}.csv")
        spath = os.path.join(RES, f"selected_{fam}__{method}.json")
        if not all(map(os.path.exists, (tpath, epath, spath))):
            problems.append(f"{tag}: missing tune/eval/selected file"); continue
        tu = pd.read_csv(tpath)
        check(len(tu) == 144 and not tu.duplicated(["target", "config_id", "held_out"]).any(),
              f"{tag}: tuning has {len(tu)} rows / duplicates")
        agg = tu.groupby(["target", "config_id", "hp"])[["AUROC", "PRAUC"]].mean().reset_index()
        best = agg.sort_values(["AUROC", "PRAUC", "config_id"], ascending=[False, False, True]).groupby("target").head(1)
        resel = {r.target: json.loads(r.hp) for r in best.itertuples()}
        sel = json.load(open(spath))
        check(resel == sel, f"{tag}: selected json differs from re-selection")
        ev = pd.read_csv(epath)
        rules = set(rules_for(method))
        g = ev.groupby(["target", "seed"])
        check(g.ngroups == 40, f"{tag}: {g.ngroups} runs, expected 40")
        check(set(ev.seed.unique()) == set(EVAL_SEEDS), f"{tag}: seed set differs")
        check(not ev.duplicated(["target", "seed", "rule"]).any(), f"{tag}: duplicate (target, seed, rule) rows")
        for (t, s), d in g:
            check(set(d.rule) == rules, f"{tag} {t} seed {s}: rules {sorted(d.rule)} != {sorted(rules)}")
        for r in ev.itertuples():
            if json.loads(r.hp) != sel[r.target]:
                problems.append(f"{tag} {r.target} seed {r.seed}: eval hp != selected"); break
        dev = 0.0
        for (t, s), d in g:
            f = os.path.join(RES, f"scores_{fam}", f"{t}__{method}__{s}.npz")
            if not os.path.exists(f):
                problems.append(f"{tag}: missing score file {os.path.basename(f)}"); continue
            z = np.load(f)
            # use the saved dtype (float32) as the run did: the EM rule is numerically fragile, and a

            arrs = {k: z[k] for k in ("p_val", "p_buf", "p_win", "p_eval")}
            if not all(np.isfinite(a).all() for a in arrs.values()):
                problems.append(f"{f}: non-finite scores")
            if METHODS[method][4] and any(a.min() < 0 or a.max() > 1 for a in arrs.values()):
                problems.append(f"{f}: probabilities outside [0,1]")
            y = z["y_eval"].astype(int)
            check(len(arrs["p_buf"]) == 1500 and len(arrs["p_win"]) == 500 and len(y) == len(arrs["p_eval"])
                  and set(np.unique(y)) <= {0, 1}, f"{f}: bad shapes or labels")
            check(abs(y.mean() - prior[t]) < 0.01, f"{f}: eval attack share {y.mean():.3f} vs {prior[t]:.3f}")
            check(abs(z["win_is_attack"].mean() - 0.05) < 1e-9, f"{f}: window attack share {z['win_is_attack'].mean()}")
            for rule in rules:
                yhat, _ = decide(rule, arrs["p_val"], z["y_val"], arrs["p_eval"], arrs["p_buf"], arrs["p_win"],
                                 METHODS[method][4])
                m = evaluate(y, arrs["p_eval"], yhat)
                row = d[d.rule == rule].iloc[0]
                for col in ("MacroF1", "AUROC", "PRAUC", "TPR_at_FPR0.01"):
                    dev = max(dev, abs(m[col] - row[col]))
        max_dev[method] = dev
        n_files = len(glob.glob(os.path.join(RES, f"scores_{fam}", f"*__{method}__*.npz")))
        check(n_files == 40, f"{tag}: {n_files} score files, expected 40")
    print(f"{fam}: max |metric recomputed from saved scores - CSV| by method:")
    print("   " + ", ".join(f"{k} {v:.1e}" for k, v in max_dev.items()))
    for k, v in max_dev.items():
        if v > TOL:
            problems.append(f"{fam}:{k}: recomputation deviates by {v:.2e} (> {TOL})")

print("\nRESULT:", "ALL CHECKS PASS" if not problems else f"PROBLEMS FOUND ({len(problems)})")
for p in problems[:60]:
    print("  -", p)
