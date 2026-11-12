"""
Decision rules that turn scores into alerts. Every method is scored under the same rules.

  fixed    threshold 0.5 on the predicted attack probability
  em       label-free prior correction: temperature scaling on the source validation split,
           damped Saerens EM for the target prior on the unlabelled buffer (prior clipped to
           [0.04, 0.96]), and the threshold that maximises expected Macro-F1 on the buffer
  win_mad  median + Z * 1.4826 * MAD of the window scores (log-odds scale for probabilities);
           Z = 3 is the declared operating point
  win_fpr  the (1 - alpha) quantile of the window scores, alpha = 0.01
  bbse     black-box shift estimation of the target prior (Lipton et al., 2018), diagnostic
  oracle_threshold_decisions  best threshold for target Macro-F1; uses target labels, diagnostic only
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPS = 1e-5
WIN_ALPHA = 0.01
WIN_Z = 3.0


def em_decisions(p_val, y_val, p_target, p_buffer, clip=(0.04, 0.96), prior=None, prior_fn=None):
    """Defaults reproduce the benchmark EM rule exactly. Diagnostic options:
    clip      bounds on the EM prior estimate (default [0.04, 0.96])
    prior     use this target attack prior instead of EM (e.g. the true prior, oracle diagnostic)
    prior_fn  callable(buf_p_temperature_scaled, val_p_temperature_scaled) -> prior (e.g. BBSE)"""
    val_s = np.clip(p_val, EPS, 1 - EPS)
    val_logits = np.log(val_s / (1 - val_s))

    def nll(t):
        T = max(float(t[0]), 0.2)
        p = expit(val_logits / T)
        return -np.mean(y_val * np.log(p + EPS) + (1 - y_val) * np.log(1 - p + EPS))

    T = max(float(minimize(nll, x0=[1.2], bounds=[(0.4, 5.0)], method="L-BFGS-B").x[0]), 1.0)

    def temp(p):
        p = np.clip(p, EPS, 1 - EPS)
        return expit(np.log(p / (1 - p)) / T)

    buf_p, tgt_p = temp(p_buffer), temp(p_target)
    pi_s = float(np.clip(np.mean(y_val == 1), 0.05, 0.95))
    pi_t = pi_s
    if prior is not None:
        pi_t = float(np.clip(prior, 1e-4, 1 - 1e-4))
    elif prior_fn is not None:
        pi_t = float(np.clip(prior_fn(buf_p, temp(p_val), y_val), 1e-4, 1 - 1e-4))
    else:
        for _ in range(25):
            r1, r0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
            q = (r1 * buf_p) / (r1 * buf_p + r0 * (1 - buf_p) + 1e-8)
            pi_t = 0.65 * pi_t + 0.35 * float(np.clip(np.mean(q), clip[0], clip[1]))
    r1, r0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
    buf_post = (r1 * buf_p) / (r1 * buf_p + r0 * (1 - buf_p) + 1e-8)
    tgt_post = (r1 * tgt_p) / (r1 * tgt_p + r0 * (1 - tgt_p) + 1e-8)

    best_tau, best = 0.5, -1.0
    for tau in np.linspace(0.18, 0.82, 33):
        pos = buf_post >= tau
        tp1, fp1, fn1 = buf_post[pos].sum(), (1 - buf_post[pos]).sum(), buf_post[~pos].sum()
        tp0, fp0, fn0 = (1 - buf_post[~pos]).sum(), buf_post[~pos].sum(), (1 - buf_post[pos]).sum()
        f = 0.5 * (2 * tp1 / (2 * tp1 + fp1 + fn1 + 1e-8) + 2 * tp0 / (2 * tp0 + fp0 + fn0 + 1e-8))
        if f > best:
            best, best_tau = f, float(tau)
    return (tgt_post >= best_tau).astype(np.int8), {"T": T, "pi_t": pi_t, "tau": best_tau}


def bbse_prior(buf_p, val_p, y_val, thr=0.5):
    """Black-box shift estimation (Lipton et al., ICML 2018), binary case:
    q = (mu - FPR) / (TPR - FPR), with TPR/FPR of the hard classifier on source validation."""
    yh = val_p >= thr
    tpr, fpr = float(np.mean(yh[y_val == 1])), float(np.mean(yh[y_val == 0]))
    mu = float(np.mean(buf_p >= thr))
    return (mu - fpr) / max(tpr - fpr, 1e-6)


def oracle_threshold_decisions(y_true, p_target):
    """Diagnostic upper bound only (uses target labels): threshold maximizing target Macro-F1."""
    from sklearn.metrics import f1_score
    grid = np.unique(np.quantile(p_target, np.linspace(0, 1, 201)))
    best = max(grid, key=lambda t: f1_score(y_true, p_target >= t, average="macro", zero_division=0))
    return (p_target >= best).astype(np.int8), {"tau": float(best)}


def window_fpr_decisions(p_target, p_win, alpha=WIN_ALPHA):
    tau = float(np.quantile(p_win, 1 - alpha))
    return (p_target > tau).astype(np.int8), {"tau": tau}


def window_mad_decisions(p_target, p_win, probabilistic=True, z=WIN_Z):
    def f(p):
        p = np.asarray(p, np.float64)
        if not probabilistic:
            return p
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    w = f(p_win)
    med = float(np.median(w))
    mad = 1.4826 * float(np.median(np.abs(w - med)))
    tau = med + z * max(mad, 1e-6)
    return (f(p_target) > tau).astype(np.int8), {"tau": tau, "mad": mad}


def decide(rule, p_val, y_val, p_target, p_buffer, p_win=None, probabilistic=True):
    if rule == "win_mad":
        return window_mad_decisions(p_target, p_win, probabilistic)
    if rule == "win_fpr":
        return window_fpr_decisions(p_target, p_win)
    if rule == "fixed":
        return (p_target >= 0.5).astype(np.int8), {}
    if rule == "em":
        return em_decisions(p_val, y_val, p_target, p_buffer)
    raise ValueError(rule)
