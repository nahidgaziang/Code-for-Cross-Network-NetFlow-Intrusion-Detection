"""
Evaluation metrics (scikit-learn implementations where they exist): AUROC, PR-AUC (average
precision), Macro-F1, FPR at 95% recall, TPR at fixed false-positive budgets (1% and 0.1%), the
realised FPR/TPR of a decision rule, and recall per attack family. Only the runner calls these with
target labels.
"""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

BUDGETS = (0.01, 0.001)


def tpr_at_fpr(y, s, fpr_budget):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_budget
    return float(tpr[ok].max()) if ok.any() else 0.0


def fpr_at_tpr(y, s, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr >= tpr_target
    return float(fpr[ok].min()) if ok.any() else 1.0


def evaluate(y, scores, yhat, attack=None, attack_names=None):
    """y: true labels; scores: attack scores; yhat: method's binary decisions."""
    both = len(np.unique(y)) == 2
    out = {
        "MacroF1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "PRAUC": float(average_precision_score(y, scores)) if both else np.nan,

        "AUROC": float(roc_auc_score(y, scores)) if both else np.nan,
        "FPR_at_TPR95": fpr_at_tpr(y, scores) if both else np.nan,
        "FPR_op": float(np.mean(yhat[y == 0] == 1)) if (y == 0).any() else np.nan,
        "TPR_op": float(np.mean(yhat[y == 1] == 1)) if (y == 1).any() else np.nan,
    }
    for b in BUDGETS:
        out[f"TPR_at_FPR{b:g}"] = tpr_at_fpr(y, scores, b) if both else np.nan
    if attack is not None and attack_names:
        for code in np.unique(attack[y == 1]):
            m = (attack == code) & (y == 1)
            out[f"recall[{attack_names.get(int(code), code)}]"] = float(np.mean(yhat[m] == 1))
    return out
