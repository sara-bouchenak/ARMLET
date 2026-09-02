import numpy as np
import torch


def membership_scores(attack_output: torch.Tensor) -> torch.Tensor:
    if attack_output.ndim == 1:
        return torch.sigmoid(attack_output)
    if attack_output.shape[-1] == 1:
        return torch.sigmoid(attack_output.squeeze(-1))
    return torch.softmax(attack_output, dim=-1)[:, 1]


def roc_curve_points(y_true, y_score):
    order = np.argsort(-np.asarray(y_score))
    y_true = np.asarray(y_true).astype(int)[order]
    y_score = np.asarray(y_score)[order]

    positives = max(1, int(y_true.sum()))
    negatives = max(1, int((1 - y_true).sum()))
    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    thresholds = np.r_[np.inf, y_score[threshold_idxs], -np.inf]
    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def tpr_at_fpr(y_true, y_score, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    valid = fpr <= float(max_fpr)
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def best_balanced_attack_success_rate(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    return float(np.max(1 - (fpr + (1 - tpr)) / 2))
