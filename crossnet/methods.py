"""
Method registry with fixed hyperparameter search spaces.

A method = (feature preparation, learner, class weighting, information setting, probabilistic).
Methods that share a learner share its backbone search space and its backbone draws (common random
numbers), so no method gets a better-tuned backbone than its counterpart. Method-specific
hyperparameters come from a separate stream keyed by the method name.

Information settings (what a method may use about the target network):
  S0  labelled source testbeds only        ERM, ERM-CB, XGB, RF, IRM, VREx, GroupDRO
  S1  + unlabelled target buffer           CORAL, DANN, IW-DANN, DANN-R, AdaBN, TENT, DomNorm
  S2  + short, mostly benign target window BRN, BRN-XGB, IF-win, KNN-win

  ERM       global scaling + MLP, plain cross-entropy
  ERM-CB    global scaling + MLP, class-balanced cross-entropy (reference model)
  XGB       global scaling + XGBoost, class-balanced
  RF        global scaling + random forest, class-balanced
  IRM / VREx / GroupDRO       ERM-CB + domain-generalisation objective over the source testbeds
  CORAL / DANN / IW-DANN / DANN-R   ERM-CB + alignment to the target buffer
  AdaBN / TENT                ERM-CB + test-time adaptation on the target buffer
  DomNorm   per-domain mixture normalisation + MLP (control for BRN)
  BRN       benign-referenced normalisation + MLP
  BRN-XGB   benign-referenced normalisation + XGBoost
  IF-win    isolation forest fitted on the target window only (no source training)
  KNN-win   mean distance to the k nearest window flows, window-standardised (no source training)
"""
import math
import zlib

import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from . import losses
from .engine import TabularResNet, fit, predict_proba, seed_all, to_dev
from .preprocess import GlobalScaler, brn_transform, domnorm_transform, log_features, window_zscore


def _mlp_space(r):
    return {"lr": float(10 ** r.uniform(-3.5, -2.5)), "weight_decay": float(10 ** r.uniform(-5, -3)),
            "dropout": float(r.choice([0.0, 0.1, 0.2])), "width": int(r.choice([64, 96, 128, 256])),
            "n_blocks": int(r.choice([1, 2])), "batch_size": int(r.choice([256, 1024]))}


def _xgb_space(r):
    return {"n_estimators": int(r.choice([200, 400])), "max_depth": int(r.choice([4, 6, 8])),
            "learning_rate": float(10 ** r.uniform(-1.5, -0.5)), "subsample": float(r.choice([0.7, 1.0])),
            "colsample_bytree": float(r.choice([0.7, 1.0])), "min_child_weight": int(r.choice([1, 5]))}


def _rf_space(r):
    return {"n_estimators": int(r.choice([100, 300])), "max_depth": [12, 20, None][int(r.integers(3))],
            "min_samples_leaf": int(r.choice([1, 5])), "max_features": ["sqrt", 0.5][int(r.integers(2))]}


def _iforest_space(r):
    return {"n_estimators": int(r.choice([100, 300])), "max_samples": int(r.choice([128, 256, 512])),
            "max_features": float(r.choice([0.5, 1.0]))}


def _knn_space(r):
    return {"k": int(r.choice([1, 3, 5, 10, 20, 50]))}


EXTRA = {  # method-specific hyperparameters
    "CORAL": lambda r: {"lam": float(10 ** r.uniform(-1, 1))},
    "DANN": lambda r: {"lam": float(10 ** r.uniform(-2, 0))},
    "IW-DANN": lambda r: {"lam": float(10 ** r.uniform(-2, 0))},
    "DANN-R": lambda r: {"lam_adv": float(10 ** r.uniform(-2, 0)), "lam_recon": float(10 ** r.uniform(-3, -1))},
    "IRM": lambda r: {"lam": float(10 ** r.uniform(-1, 3))},
    "VREx": lambda r: {"lam": float(10 ** r.uniform(-1, 3))},
    "GroupDRO": lambda r: {"eta": float(10 ** r.uniform(-3, -1))},
    "TENT": lambda r: {"tent_lr": float(10 ** r.uniform(-4, -2)), "tent_steps": int(r.choice([1, 5, 10]))},
}

# name: (prep, learner, class_balanced, setting, probabilistic)
METHODS = {
    "ERM": ("global", "mlp", False, "S0", True),
    "ERM-CB": ("global", "mlp", True, "S0", True),
    "XGB": ("global", "xgb", True, "S0", True),
    "RF": ("global", "rf", True, "S0", True),
    "IRM": ("global", "mlp", True, "S0", True),
    "VREx": ("global", "mlp", True, "S0", True),
    "GroupDRO": ("global", "mlp", True, "S0", True),
    "CORAL": ("global", "mlp", True, "S1", True),
    "DANN": ("global", "mlp", True, "S1", True),
    "IW-DANN": ("global", "mlp", True, "S1", True),
    "DANN-R": ("global", "mlp", True, "S1", True),
    "AdaBN": ("global", "mlp", True, "S1", True),
    "TENT": ("global", "mlp", True, "S1", True),
    "DomNorm": ("domnorm", "mlp", True, "S1", True),
    "BRN": ("brn", "mlp", True, "S2", True),
    "BRN-XGB": ("brn", "xgb", True, "S2", True),
    "IF-win": ("raw", "iforest", False, "S2", False),
    "KNN-win": ("window", "knn", False, "S2", False),
}
SPACES = {"mlp": _mlp_space, "xgb": _xgb_space, "rf": _rf_space, "iforest": _iforest_space, "knn": _knn_space}
RF_JOBS = 4  # CPU threads for Random Forest (leaves cores for the parallel GPU workers)


# DANN-R was developed under the name DI-NIDS; its hyperparameter stream keeps that key so the
# configuration draws (and therefore all reported results) are unchanged by the rename.
RNG_KEY = {"DANN-R": "DI-NIDS"}


def sample_configs(method, budget, seed):
    """Fixed candidate list per method. Backbone draws depend only on (learner, budget, seed),
    so methods sharing a learner get identical backbones; extras come from a method-keyed stream."""
    learner = METHODS[method][1]
    r = np.random.default_rng(seed)
    r_extra = np.random.default_rng([seed, zlib.crc32(RNG_KEY.get(method, method).encode())])
    out = []
    for _ in range(budget):
        hp = SPACES[learner](r)
        if method in EXTRA:
            hp.update(EXTRA[method](r_extra))
        out.append(hp)
    return out


def _prepare(prep, inp, L):
    """Returns transformed (tr, va, buf, win, eval)."""
    if prep in ("global", "raw"):
        if prep == "raw":
            return L["X_tr"], L["X_va"], L["X_buf"], L["X_win"], L["X_eval"]
        sc = GlobalScaler().fit(L["X_tr"])
        return tuple(sc.transform(L[k]) for k in ("X_tr", "X_va", "X_buf", "X_win", "X_eval"))
    if prep == "brn":
        tr, va, buf, ev = brn_transform(L["X_tr"], inp.y_tr, inp.d_tr, L["X_va"], inp.d_va,
                                        L["X_buf"], L["X_win"], L["X_eval"])
        return tr, va, buf, window_zscore(L["X_win"], L["X_win"]), ev
    if prep == "domnorm":
        return domnorm_transform(L["X_tr"], inp.d_tr, L["X_va"], inp.d_va, L["X_buf"], L["X_win"], L["X_eval"])
    if prep == "window":
        return tuple(window_zscore(L[k], L["X_win"]) for k in ("X_tr", "X_va", "X_buf", "X_win", "X_eval"))
    raise ValueError(prep)


def _knn_scores(Q, W, k, exclude_self=False, chunk=8192):
    """Mean Euclidean distance to the k nearest window flows (GPU). exclude_self: Q is W (leave-one-out)."""
    Wt = to_dev(W)
    k = min(k, len(W) - (1 if exclude_self else 0))
    out = []
    for i in range(0, len(Q), chunk):
        d = torch.cdist(to_dev(Q[i:i + chunk]), Wt)
        if exclude_self:
            d[torch.arange(len(d)), torch.arange(i, i + len(d))] = float("inf")
        out.append(d.topk(k, dim=1, largest=False).values.mean(1))
    return torch.cat(out).cpu().numpy().astype(np.float32)


def _mlp_run(method, inp, hp, seed, tr, va, buf, win, ev):
    seed_all(seed)  # weight initialization must be seeded too
    net = TabularResNet(tr.shape[1], h=hp["width"], n_blocks=hp["n_blocks"], dropout=hp["dropout"])
    model, loss_fn, target_X = net, None, None
    if method == "CORAL":
        loss_fn, target_X = losses.coral_loss(hp["lam"]), buf
    elif method == "DANN":
        model, loss_fn, target_X = losses.WithDiscriminator(net, hp["width"]), losses.dann_loss(hp["lam"]), buf
    elif method == "IW-DANN":
        p_src_att = float(inp.y_tr.mean()) if len(inp.y_tr) > 0 else 0.5
        model, loss_fn, target_X = losses.WithDiscriminator(net, hp["width"]), losses.iw_dann_loss(hp["lam"], p_src_att), buf
    elif method == "DANN-R":
        model, loss_fn, target_X = losses.WithDiscriminatorAndDecoder(net, hp["width"], tr.shape[1]), losses.dann_recon_loss(hp["lam_adv"], hp["lam_recon"]), buf
    elif method == "IRM":
        loss_fn = losses.irm_loss(hp["lam"], inp.d_tr)
    elif method == "VREx":
        loss_fn = losses.vrex_loss(hp["lam"], inp.d_tr)
    elif method == "GroupDRO":
        loss_fn = losses.groupdro_loss(hp["eta"], inp.d_tr)
    info = fit(model, tr, inp.y_tr, va, inp.y_va, class_balanced=METHODS[method][2], seed=seed,
               loss_fn=loss_fn, target_X=target_X,
               lr=hp["lr"] * math.sqrt(hp["batch_size"] / 256), weight_decay=hp["weight_decay"],
               batch_size=hp["batch_size"])
    p = {"val": predict_proba(model, va)}  # source-validation scores before any target adaptation
    if method == "AdaBN":
        losses.adabn(model, buf)
    elif method == "TENT":
        losses.tent(model, buf, hp["tent_lr"], hp["tent_steps"])
    p.update({k: predict_proba(model, a) for k, a in (("buf", buf), ("win", win), ("eval", ev))})
    return p, info


def run_method(method, inp, hp, seed):
    """Returns attack scores on the source validation split, target buffer, target window and
    target eval stream (probabilities for probabilistic methods, anomaly scores otherwise)."""
    prep, learner, balanced, _, _ = METHODS[method]
    L = {k: log_features(getattr(inp, k)) for k in ("X_tr", "X_va", "X_buf", "X_win", "X_eval")}
    tr, va, buf, win, ev = _prepare(prep, inp, L)
    for name, arr in (("train", tr), ("val", va), ("buffer", buf), ("window", win), ("eval", ev)):
        if not np.isfinite(arr).all():  # fail loudly: one inf poisons every score
            raise ValueError(f"{method}: non-finite values in {name} features after preprocessing")
    arrays = (("val", va), ("buf", buf), ("win", win), ("eval", ev))
    if learner == "mlp":
        return _mlp_run(method, inp, hp, seed, tr, va, buf, win, ev)
    if learner == "xgb":
        n1 = int(inp.y_tr.sum())
        model = xgb.XGBClassifier(**hp, tree_method="hist", device="cuda", random_state=seed % (2 ** 31),
                                  scale_pos_weight=(len(inp.y_tr) - n1) / max(n1, 1) if balanced else 1.0,
                                  eval_metric="logloss", early_stopping_rounds=30, verbosity=0)
        model.fit(tr, inp.y_tr, eval_set=[(va, inp.y_va)], verbose=False)
        return {k: model.predict_proba(a)[:, 1] for k, a in arrays}, {"best_iteration": int(model.best_iteration)}
    if learner == "rf":
        model = RandomForestClassifier(**hp, class_weight="balanced" if balanced else None,
                                       n_jobs=RF_JOBS, random_state=seed % (2 ** 31))
        model.fit(tr, inp.y_tr)
        return {k: model.predict_proba(a)[:, 1].astype(np.float32) for k, a in arrays}, {}
    if learner == "iforest":
        model = IsolationForest(n_estimators=hp["n_estimators"], max_samples=min(hp["max_samples"], len(win)),
                                max_features=hp["max_features"], random_state=seed % (2 ** 31), n_jobs=1)
        model.fit(win)
        return {k: (-model.score_samples(a)).astype(np.float32) for k, a in arrays}, {}
    if learner == "knn":
        p = {k: _knn_scores(a, win, hp["k"]) for k, a in (("val", va), ("buf", buf), ("eval", ev))}
        p["win"] = _knn_scores(win, win, hp["k"], exclude_self=True)
        return p, {}
    raise ValueError(learner)
