import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


def precision_at_k(y_true, y_score, top_fraction=0.10):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = max(1, int(len(y_score) * top_fraction))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].mean())


def lift_at_k(y_true, y_score, top_fraction=0.10):
    base_rate = np.mean(y_true)
    return 0.0 if base_rate == 0 else precision_at_k(y_true, y_score, top_fraction) / base_rate


def choose_threshold_for_f1(y_true, y_score):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (np.asarray(y_score) >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = t, score
    return best_t


def compute_metrics(y_true, y_score, top_fraction=0.10, threshold=None):
    if threshold is None:
        threshold = choose_threshold_for_f1(y_true, y_score)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision_at_k": precision_at_k(y_true, y_score, top_fraction),
        "lift_at_k": lift_at_k(y_true, y_score, top_fraction),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "threshold": float(threshold),
    }
