"""
Feature transforms shared by all methods.

  log_features      ln(1 + max(0, x)) for heavy-tailed counters
  GlobalScaler      z-score fitted on the pooled source training flows
  brn_transform     benign-referenced normalisation (BRN): each source testbed is z-scored with
                    its own benign training flows, the target with its window
  domnorm_transform mixture-referenced control for BRN: sources z-scored with all their training
                    flows, the target with its unlabelled buffer
  window_zscore     z-score with statistics of the target window (one-class window detectors)
"""
import numpy as np

FLOOR, CLIP = 0.1, 5.0


def log_features(X):
    return np.log1p(np.maximum(X, 0.0)).astype(np.float32)


class GlobalScaler:
    def fit(self, X):
        self.mu = X.mean(0, keepdims=True)
        self.sd = X.std(0, keepdims=True) + 1e-6
        return self

    def transform(self, X):
        return np.clip((X - self.mu) / self.sd, -CLIP, CLIP).astype(np.float32)


def _zscore(X, ref):
    mu, sd = ref.mean(0, keepdims=True), np.maximum(ref.std(0, keepdims=True), FLOOR)
    return np.clip((X - mu) / sd, -CLIP, CLIP).astype(np.float32)


def brn_transform(Ltr, ytr, dtr, Lva, dva, Lbuf, Lwin, Leval):
    """Inputs are log-features. Returns transformed (tr, va, buf, eval)."""
    tr, va = np.empty_like(Ltr), np.empty_like(Lva)
    for d in np.unique(dtr):
        ref = Ltr[(dtr == d) & (ytr == 0)]
        tr[dtr == d] = _zscore(Ltr[dtr == d], ref)
        va[dva == d] = _zscore(Lva[dva == d], ref)
    return tr, va, _zscore(Lbuf, Lwin), _zscore(Leval, Lwin)


def window_zscore(L, Lwin):
    """Target arrays expressed relative to the benign window (BRN's target side)."""
    return _zscore(L, Lwin)


def domnorm_transform(Ltr, dtr, Lva, dva, Lbuf, Lwin, Leval):
    """Mixture-referenced per-domain normalization (the control for BRN):
    each source testbed is z-scored with ALL its training flows (benign + attack), the target
    with its unlabeled buffer (natural class prior). Differs from BRN only in the reference
    population, so BRN - DomNorm isolates the benign reference (prior-invariance claim).
    Returns transformed (tr, va, buf, win, eval)."""
    tr, va = np.empty_like(Ltr), np.empty_like(Lva)
    for d in np.unique(dtr):
        ref = Ltr[dtr == d]
        tr[dtr == d] = _zscore(Ltr[dtr == d], ref)
        va[dva == d] = _zscore(Lva[dva == d], ref)
    return tr, va, _zscore(Lbuf, Lbuf), _zscore(Lwin, Lbuf), _zscore(Leval, Lbuf)
