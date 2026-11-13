"""
Reference windows collected without label selection.

The benchmark window has a label-controlled composition (benign-pool flows plus a fixed attack share).
Here the window is drawn without labels:
  rand   500 flows drawn uniformly from the target's uniform pool (natural class mix), outside the
         evaluation and buffer rows; scored on the full benchmark evaluation stream (both releases);
  early  NF-v3 only: 500 flows drawn uniformly from target flows that start before the 20% time cut;
         scored on the late evaluation flows.
Each window is also run trimmed: the 20% of window flows with the highest leave-one-out window score are
dropped and the detector is refitted. KNN-win and IF-win use the benchmark-selected hyperparameters. The
realised window attack share is recorded.
  python experiments/unselected_window.py run|report
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from crossnet.data import POOL_DIR, TESTBEDS, FoldConfig, build_fold, load_pool  # noqa: E402
from crossnet.decision import window_mad_decisions  # noqa: E402
from crossnet.methods import _knn_scores  # noqa: E402
from crossnet.preprocess import log_features, window_zscore  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402


def gpu_keepalive(period=0.1):
    """Keeps the laptop GPU out of its idle power states while the experiment runs. This workload uses
    the GPU in short bursts separated by seconds of CPU work; on the development laptop those repeated
    idle/wake transitions made the GPU drop off the bus (cudaErrorUnknown). A tiny kernel every 100 ms
    keeps the GPU awake. It touches no data of the experiment, so the results are unaffected."""
    import threading
    import torch
    if not torch.cuda.is_available():
        return None
    stop = threading.Event()
    x = torch.ones(64, 64, device="cuda")

    def loop():
        while not stop.is_set():
            (x @ x).sum().item()
            stop.wait(period)

    threading.Thread(target=loop, daemon=True).start()
    return stop


def score_window(method, X_eval, X_win, hp, seed):
    """Evaluation and window scores of a window detector, computed exactly as run_method does
    (same transforms, same learners, same seeds) but without the source-validation and buffer scores,
    which this experiment does not use. Returns ({"eval": ..., "win": ...}, {})."""
    Le, Lw = log_features(X_eval), log_features(X_win)
    if method == "KNN-win":                                  # prep "window": z-score by the window
        ev, win = window_zscore(Le, Lw), window_zscore(Lw, Lw)
        for name, a in (("window", win), ("eval", ev)):
            if not np.isfinite(a).all():
                raise ValueError(f"{method}: non-finite values in {name} features after preprocessing")
        return {"eval": _knn_scores(ev, win, hp["k"]), "win": _knn_scores(win, win, hp["k"], exclude_self=True)}, {}
    if method == "IF-win":                                   # prep "raw": log features
        m = IsolationForest(n_estimators=hp["n_estimators"], max_samples=min(hp["max_samples"], len(Lw)),
                            max_features=hp["max_features"], random_state=seed % (2 ** 31), n_jobs=1).fit(Lw)
        return {"eval": (-m.score_samples(Le)).astype(np.float32), "win": (-m.score_samples(Lw)).astype(np.float32)}, {}
    raise ValueError(method)
from crossnet.runner import EVAL_SEEDS, append, done_keys  # noqa: E402

BENCH = os.path.join(HERE, "..", "results", "benchmark")
RES = os.environ.get("U_RES", os.path.join(HERE, "..", "results", "unselected"))
METHODS = ("KNN-win", "IF-win")
N_WIN, TRIM, CUT_Q = 500, 0.20, 0.2


def selected(fam, method):
    return json.load(open(os.path.join(BENCH, f"selected_{fam}__{method}.json")))


def metrics(y, s, s_win):
    yhat, _ = window_mad_decisions(s, s_win, probabilistic=False, z=3.0)
    ben, att = s[y == 0], s[y == 1]
    return {"AUROC": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
            "FPR_mad": float(yhat[y == 0].mean()) if (y == 0).any() else np.nan,
            "TPR_mad": float(yhat[y == 1].mean()) if (y == 1).any() else np.nan,
            "TPR_at_FPR0.01": float(np.mean(att > np.quantile(ben, 0.99))) if len(ben) and len(att) else np.nan}


def windows(fam, target, seed, meta):
    """Yields (window name, window X, window labels, evaluation mask or None)."""
    tu = load_pool(fam, target, "uniform")
    used = np.union1d(meta["rows_eval"], meta["rows_buf"])
    free = ~np.isin(tu["row"], used)
    rng = np.random.default_rng([seed, TESTBEDS[fam].index(target), 7171])
    idx = rng.choice(np.where(free)[0], N_WIN, replace=False)
    yield "rand", tu["X"][idx], tu["y"][idx].astype(int), None
    if fam == "nfv3":
        t = tu["t_start"].astype(np.float64)
        cut = float(np.quantile(t, CUT_Q))
        idx = rng.choice(np.where(free & (t <= cut))[0], N_WIN, replace=False)
        ev = pd.Index(tu["row"]).get_indexer(meta["rows_eval"])
        assert (ev >= 0).all()
        yield "early", tu["X"][idx], tu["y"][idx].astype(int), t[ev] > cut


def run():
    keep = gpu_keepalive()
    print(f"[U] GPU keep-alive {'on' if keep else 'off (no CUDA device)'}", flush=True)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "u_runs.csv")
    done = done_keys(path, ["family", "target", "seed", "window", "method", "defense"])
    for fam in ("nfv2", "nfv3"):
        load_pool.cache_clear()
        for seed in EVAL_SEEDS:
            for target in TESTBEDS[fam]:
                wins = ("rand", "early") if fam == "nfv3" else ("rand",)
                if all((fam, target, str(seed), w, m, d) in done for w in wins for m in METHODS
                       for d in ("naive", "trimmed")):
                    continue
                inp, truth, meta = build_fold(FoldConfig(family=fam, target=target, seed=seed))
                y_all = truth.y_eval.astype(int)
                z = np.load(os.path.join(BENCH, f"scores_{fam}", f"{target}__KNN-win__{seed}.npz"))
                assert np.array_equal(z["y_eval"].astype(int), y_all), "fold differs from the benchmark"
                for wname, X_win, y_win, mask in windows(fam, target, seed, meta):
                    y = y_all if mask is None else y_all[mask]
                    for method in METHODS:
                        if all((fam, target, str(seed), wname, method, dd) in done for dd in ("naive", "trimmed")):
                            continue
                        hp = selected(fam, method)[target]
                        t0 = time.time()
                        pn, _ = score_window(method, inp.X_eval, X_win, hp, seed)
                        keep = np.argsort(pn["win"], kind="stable")[: N_WIN - int(round(TRIM * N_WIN))]
                        pt, _ = score_window(method, inp.X_eval, X_win[keep], hp, seed)
                        for dname, pr, yw in (("naive", pn, y_win), ("trimmed", pt, y_win[keep])):
                            s = np.asarray(pr["eval"], np.float64)
                            s = s if mask is None else s[mask]
                            append(path, {"family": fam, "target": target, "seed": seed, "window": wname,
                                          "method": method, "defense": dname, "n_win": len(yw),
                                          "win_attack_share": float(yw.mean()), "n_eval": len(y),
                                          "n_eval_benign": int((y == 0).sum()),
                                          **metrics(y, s, np.asarray(pr["win"], np.float64)),
                                          "seconds": round(time.time() - t0, 1)})
                print(f"[U] {fam} {target} seed {seed} done", flush=True)


def run_report():
    d = pd.read_csv(os.path.join(RES, "u_runs.csv"))
    L = ["EXPERIMENT U: UNSELECTED REFERENCE WINDOWS (mean over 10 seeds)", ""]
    L.append(d.groupby(["family", "window", "target", "method", "defense"])[["win_attack_share", "AUROC", "FPR_mad", "TPR_mad"]]
             .mean().round(3).to_string())
    text = "\n".join(L)
    open(os.path.join(RES, "report.txt"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    {"run": run, "report": run_report}[sys.argv[1]]()
