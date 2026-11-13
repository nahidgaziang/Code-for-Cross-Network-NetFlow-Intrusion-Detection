"""
Control experiments on the benchmark folds (10 seeds, both releases, benchmark-selected hyperparameters):
  oneclass  KNN and isolation forest fitted on source benign training flows (500 and 20,000 flows) instead of
            the target window; separates the target information from the one-class method.
  neardup   KNN-win's AUROC after removing evaluation flows within eps of a window flow (window-standardised
            space; eps = 0 and the 1st and 5th percentiles of the window's nearest-neighbour distances).
  xgbwin    XGBoost on the source data plus the window flows labelled benign (weighted as in XGB-ST).
  poisonk   KNN-win on the clean and 10%-poisoned windows of the poisoning study with k in {5, ..., 200}.
  python experiments/controls.py oneclass|neardup|xgbwin|poisonk|all
"""
import json
import os
import sys
import time

import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from crossnet.data import POOL_DIR, TESTBEDS, FoldConfig, build_fold, load_pool  # noqa: E402
from crossnet.engine import DEVICE, to_dev  # noqa: E402
from crossnet.methods import _knn_scores  # noqa: E402
from crossnet.preprocess import GlobalScaler, log_features, window_zscore  # noqa: E402
from crossnet.runner import EVAL_SEEDS, append, done_keys  # noqa: E402
from unselected_window import gpu_keepalive  # noqa: E402

BENCH = os.path.join(HERE, "..", "results", "benchmark")
RES = os.environ.get("RC_RES", os.path.join(HERE, "..", "results", "controls"))
N_WIN, POISON_P, KS = 500, 0.10, (5, 10, 20, 50, 100, 200)


def selected(fam, method):
    return json.load(open(os.path.join(BENCH, f"selected_{fam}__{method}.json")))


def auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def tpr_at(y, s, fpr):
    ben, att = s[y == 0], s[y == 1]
    return float(np.mean(att > np.quantile(ben, 1 - fpr))) if len(ben) and len(att) else np.nan


def knn_ref(X_eval, X_ref, k):
    """KNN scores of X_eval against reference X_ref, standardised by the reference (as KNN-win)."""
    Le, Lr = log_features(X_eval), log_features(X_ref)
    return _knn_scores(window_zscore(Le, Lr), window_zscore(Lr, Lr), k, chunk=2048 if len(X_ref) > 5000 else 8192)


def if_ref(X_eval, X_ref, hp, seed):
    Le, Lr = log_features(X_eval), log_features(X_ref)
    m = IsolationForest(n_estimators=hp["n_estimators"], max_samples=min(hp["max_samples"], len(Lr)),
                        max_features=hp["max_features"], random_state=seed % (2 ** 31), n_jobs=1).fit(Lr)
    return (-m.score_samples(Le)).astype(np.float32)


def folds(keys_done, keyfn):
    for fam in ("nfv2", "nfv3"):
        load_pool.cache_clear()
        for seed in EVAL_SEEDS:
            for target in TESTBEDS[fam]:
                if keyfn(fam, target, seed) <= keys_done:
                    continue
                inp, truth, meta = build_fold(FoldConfig(family=fam, target=target, seed=seed))
                z = np.load(os.path.join(BENCH, f"scores_{fam}", f"{target}__KNN-win__{seed}.npz"))
                assert np.array_equal(z["y_eval"].astype(int), truth.y_eval.astype(int)), "fold differs from the benchmark"
                yield fam, target, seed, inp, truth, meta, z


def run_oneclass():
    path = os.path.join(RES, "oneclass.csv")
    done = done_keys(path, ["family", "target", "seed", "method", "n_ref"])
    keys = lambda f, t, s: {(f, t, str(s), m, str(n)) for m in ("KNN-src", "IF-src") for n in (500, 20000)}  # noqa: E731
    for fam, target, seed, inp, truth, meta, _ in folds(done, keys):
        y = truth.y_eval.astype(int)
        ben = np.where(inp.y_tr == 0)[0]
        rng = np.random.default_rng([seed, TESTBEDS[fam].index(target), 8181])
        order = rng.permutation(ben)
        for n in (500, 20000):
            ref = inp.X_tr[order[:min(n, len(order))]]
            for m in ("KNN-src", "IF-src"):
                if (fam, target, str(seed), m, str(n)) in done:
                    continue
                t0 = time.time()
                s = (knn_ref(inp.X_eval, ref, selected(fam, "KNN-win")[target]["k"]) if m == "KNN-src"
                     else if_ref(inp.X_eval, ref, selected(fam, "IF-win")[target], seed))
                append(path, {"family": fam, "target": target, "seed": seed, "method": m, "n_ref": n,
                              "n_ref_used": len(ref), "AUROC": auroc(y, s), "TPR_at_FPR0.01": tpr_at(y, s, 0.01),
                              "TPR_at_FPR0.001": tpr_at(y, s, 0.001), "seconds": round(time.time() - t0, 1)})
        print(f"[oneclass] {fam} {target} seed {seed} done", flush=True)


def run_neardup():
    path = os.path.join(RES, "neardup.csv")
    done = done_keys(path, ["family", "target", "seed"])
    for fam, target, seed, inp, truth, meta, z in folds(done, lambda f, t, s: {(f, t, str(s))}):
        y, s = truth.y_eval.astype(int), z["p_eval"].astype(np.float64)
        Lw = log_features(inp.X_win)
        W = to_dev(window_zscore(Lw, Lw))
        Q = window_zscore(log_features(inp.X_eval), Lw)
        dmin = torch.cat([torch.cdist(to_dev(Q[i:i + 8192]), W).min(1).values for i in range(0, len(Q), 8192)]).cpu().numpy()
        dw = torch.cdist(W, W)
        dw.fill_diagonal_(float("inf"))
        loo = dw.min(1).values.cpu().numpy()
        r = {"family": fam, "target": target, "seed": seed, "AUROC_all": auroc(y, s)}
        for name, eps in (("exact", 0.0), ("p1", float(np.percentile(loo, 1))), ("p5", float(np.percentile(loo, 5)))):
            keep = dmin > eps
            r.update({f"eps_{name}": eps, f"removed_benign_{name}": float((~keep & (y == 0)).sum() / max((y == 0).sum(), 1)),
                      f"removed_attack_{name}": float((~keep & (y == 1)).sum() / max((y == 1).sum(), 1)),
                      f"AUROC_{name}": auroc(y[keep], s[keep])})
        append(path, r)
        print(f"[neardup] {fam} {target} seed {seed} done", flush=True)


def run_xgbwin():
    path = os.path.join(RES, "xgbwin.csv")
    done = done_keys(path, ["family", "target", "seed"])
    for fam, target, seed, inp, truth, meta, _ in folds(done, lambda f, t, s: {(f, t, str(s))}):
        y = truth.y_eval.astype(int)
        t0 = time.time()
        sc = GlobalScaler().fit(log_features(inp.X_tr))
        tr, va, ev, win = (sc.transform(log_features(a)) for a in (inp.X_tr, inp.X_va, inp.X_eval, inp.X_win))
        Xc, yc = np.concatenate([tr, win]), np.concatenate([inp.y_tr.astype(int), np.zeros(len(win), int)])
        dom = np.r_[np.ones(len(tr)), np.full(len(win), len(tr) / len(win))]
        cls = np.where(yc == 1, len(yc) / (2 * max(yc.sum(), 1)), len(yc) / (2 * max((yc == 0).sum(), 1)))
        m = xgb.XGBClassifier(**selected(fam, "XGB")[target], tree_method="hist", device="cuda" if DEVICE.type == "cuda" else "cpu",
                              random_state=seed, eval_metric="logloss", early_stopping_rounds=30, verbosity=0)
        m.fit(Xc, yc, sample_weight=dom * cls, eval_set=[(va, inp.y_va.astype(int))], verbose=False)
        s = m.predict_proba(ev)[:, 1]
        append(path, {"family": fam, "target": target, "seed": seed, "method": "XGB-SW", "AUROC": auroc(y, s),
                      "TPR_at_FPR0.01": tpr_at(y, s, 0.01), "TPR_at_FPR0.001": tpr_at(y, s, 0.001),
                      "best_iteration": int(m.best_iteration), "seconds": round(time.time() - t0, 1)})
        print(f"[xgbwin] {fam} {target} seed {seed} done", flush=True)


def run_poisonk():
    from window_poisoning import top_family
    path = os.path.join(RES, "poisonk.csv")
    done = done_keys(path, ["family", "target", "seed", "p_poison", "k"])
    keys = lambda f, t, s: {(f, t, str(s), str(p), str(k)) for p in (0.0, POISON_P) for k in KS}  # noqa: E731
    for fam, target, seed, inp, truth, meta, _ in folds(done, keys):
        y, att = truth.y_eval.astype(int), truth.attack_eval
        code, fname = top_family(fam, target)
        is_tgt, is_oth = (y == 1) & (att == code), (y == 1) & (att != code)
        tu, tb = load_pool(fam, target, "uniform"), load_pool(fam, target, "benign")
        used = np.union1d(meta["rows_eval"], meta["rows_buf"])
        rng = np.random.default_rng([seed, TESTBEDS[fam].index(target), 4242])     # same draws as the poisoning study
        ben = rng.permutation(np.where(~np.isin(tb["row"], used))[0])
        poi = rng.permutation(np.where((tu["attack"] == code) & ~np.isin(tu["row"], used))[0])
        Le = log_features(inp.X_eval)
        # replicate the poisoning study's window row order exactly: it drew one shuffle per level, in this order
        orders = {p: rng.permutation(N_WIN) for p in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)}
        for p in (0.0, POISON_P):
            n = int(round(p * N_WIN))
            Lw = log_features(np.concatenate([tb["X"][ben[:N_WIN - n]], tu["X"][poi[:n]]])[orders[p]])
            Q, W = window_zscore(Le, Lw), window_zscore(Lw, Lw)
            for k in KS:
                if (fam, target, str(seed), str(p), str(k)) in done:
                    continue
                s = _knn_scores(Q, W, k).astype(np.float64)
                b = s[y == 0]
                append(path, {"family": fam, "target": target, "seed": seed, "p_poison": p, "k": k, "target_family": fname,
                              "AUROC_target_family": auroc(np.r_[np.zeros(len(b)), np.ones(is_tgt.sum())], np.r_[b, s[is_tgt]]),
                              "AUROC_other_families": auroc(np.r_[np.zeros(len(b)), np.ones(is_oth.sum())], np.r_[b, s[is_oth]])
                              if is_oth.any() else np.nan})
        print(f"[poisonk] {fam} {target} seed {seed} done", flush=True)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    keep = gpu_keepalive()
    print(f"[controls] device {DEVICE}; GPU keep-alive {'on' if keep else 'off'}", flush=True)
    jobs = {"oneclass": run_oneclass, "neardup": run_neardup, "xgbwin": run_xgbwin, "poisonk": run_poisonk}
    for name in (jobs if sys.argv[1] == "all" else [sys.argv[1]]):
        jobs[name]()
