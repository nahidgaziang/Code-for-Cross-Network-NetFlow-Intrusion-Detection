"""
Leakage-safe fold construction.

A fold is (release, target testbed, seed). The three remaining testbeds are the sources. Everything
a method may see is in `MethodInput`; target labels live only in `EvalTruth`, which the runner passes
to the metrics and never to a method.

Sources: `n_src` flows per source testbed from its uniform pool, split 80/20 into training and
validation, stratified by testbed.
Target: three disjoint sets drawn from the target's pools (inductive protocol):
  eval    evaluation stream (labels hidden from methods)
  buffer  unlabelled target flows with the natural class prior (setting S1)
  window  short, mostly benign target reference: benign flows from the benign pool plus a fixed
          fraction of attack flows (setting S2; labels hidden)
All draws come from numpy.random.default_rng([seed, target index]), so a fold is a pure function
of its configuration.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import os

import numpy as np

POOL_DIR = os.environ.get("NF_POOL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pools"))
TESTBEDS = {
    "nfv2": ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"],
    "nfv3": ["NF-UNSW-NB15-v3", "NF-BoT-IoT-v3", "NF-ToN-IoT-v3", "NF-CSE-CIC-IDS2018-v3"],
}


@dataclass
class FoldConfig:
    family: str = "nfv2"
    target: str = "NF-CSE-CIC-IDS2018-v2"
    seed: int = 42
    n_src: int = 100_000      # flows per source testbed
    n_eval: int = 100_000     # target evaluation flows
    n_buf: int = 1_500        # unlabeled target buffer
    n_win: int = 500          # target window size
    win_attack_frac: float = 0.05
    val_frac: float = 0.2


@dataclass
class MethodInput:
    """Everything a method is allowed to see."""
    X_tr: np.ndarray
    y_tr: np.ndarray
    d_tr: np.ndarray          # source-testbed id per training flow
    X_va: np.ndarray
    y_va: np.ndarray
    d_va: np.ndarray
    X_buf: np.ndarray         # unlabeled
    X_win: np.ndarray         # unlabeled (mostly benign by construction)
    X_eval: np.ndarray        # unlabeled at method time
    features: list = field(default_factory=list)
    source_names: list = field(default_factory=list)


@dataclass
class EvalTruth:
    """Only the runner/metrics see this."""
    y_eval: np.ndarray
    attack_eval: np.ndarray
    attack_names: dict
    win_is_attack: np.ndarray


@lru_cache(maxsize=16)
def load_pool(family, testbed, kind):
    with np.load(os.path.join(POOL_DIR, family, f"{testbed}_{kind}.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _take(pool, idx):
    return {k: (v[idx] if k != "features" else v) for k, v in pool.items()}


def build_fold(cfg: FoldConfig, attack_names: dict | None = None):
    rng = np.random.default_rng([cfg.seed, TESTBEDS[cfg.family].index(cfg.target)])
    sources = [t for t in TESTBEDS[cfg.family] if t != cfg.target]

    # Sources: n_src per testbed, then a testbed-stratified 80/20 split.
    Xs, ys, ds, tr_mask = [], [], [], []
    for i, tb in enumerate(sources):
        pool = load_pool(cfg.family, tb, "uniform")
        idx = rng.choice(len(pool["y"]), size=min(cfg.n_src, len(pool["y"])), replace=False)
        Xs.append(pool["X"][idx]); ys.append(pool["y"][idx]); ds.append(np.full(len(idx), i, np.int8))
        m = np.zeros(len(idx), bool)
        m[rng.permutation(len(idx))[: int(round((1 - cfg.val_frac) * len(idx)))]] = True
        tr_mask.append(m)
    Xs, ys, ds, tr_mask = map(np.concatenate, (Xs, ys, ds, tr_mask))

    # Target: disjoint eval / buffer from the uniform pool; window from the benign pool.
    tu = load_pool(cfg.family, cfg.target, "uniform")
    tb_pool = load_pool(cfg.family, cfg.target, "benign")
    perm = rng.permutation(len(tu["y"]))
    n_eval = min(cfg.n_eval, len(perm) - cfg.n_buf - cfg.n_win)
    ev, buf, rest = perm[:n_eval], perm[n_eval:n_eval + cfg.n_buf], perm[n_eval + cfg.n_buf:]
    used_rows = set(tu["row"][ev]) | set(tu["row"][buf])
    ben_ok = np.array([r not in used_rows for r in tb_pool["row"]])
    ben_idx = rng.permutation(np.where(ben_ok)[0])
    n_att = int(round(cfg.win_attack_frac * cfg.n_win))
    att_idx = rest[tu["y"][rest] == 1][:n_att]
    n_ben = min(cfg.n_win - len(att_idx), len(ben_idx))
    X_win = np.concatenate([tb_pool["X"][ben_idx[:n_ben]], tu["X"][att_idx]])
    win_is_attack = np.concatenate([np.zeros(n_ben, np.int8), np.ones(len(att_idx), np.int8)])
    order = rng.permutation(len(X_win))

    inp = MethodInput(
        X_tr=Xs[tr_mask], y_tr=ys[tr_mask], d_tr=ds[tr_mask],
        X_va=Xs[~tr_mask], y_va=ys[~tr_mask], d_va=ds[~tr_mask],
        X_buf=tu["X"][buf], X_win=X_win[order], X_eval=tu["X"][ev],
        features=[str(f) for f in tu["features"]], source_names=sources)
    truth = EvalTruth(y_eval=tu["y"][ev], attack_eval=tu["attack"][ev],
                      attack_names=attack_names or {}, win_is_attack=win_is_attack[order])
    meta = {"rows_eval": tu["row"][ev], "rows_buf": tu["row"][buf],
            "rows_win": np.concatenate([tb_pool["row"][ben_idx[:n_ben]], tu["row"][att_idx]])[order]}
    return inp, truth, meta
