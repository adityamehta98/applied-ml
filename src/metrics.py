import numpy as np


def confusion_counts(y_true, y_pred, positive=1):
    """Return (tp, fp, tn, fn) for a binary problem, treating `positive` as the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_pos_true = y_true == positive
    is_pos_pred = y_pred == positive
    tp = int(np.sum(is_pos_true & is_pos_pred))
    fp = int(np.sum(~is_pos_true & is_pos_pred))
    tn = int(np.sum(~is_pos_true & ~is_pos_pred))
    fn = int(np.sum(is_pos_true & ~is_pos_pred))
    return tp, fp, tn, fn


def precision(y_true, y_pred, positive=1):
    """TP / (TP + FP): of everything flagged positive, how much was right. 0.0 if nothing was flagged."""
    tp, fp, _, _ = confusion_counts(y_true, y_pred, positive)
    denom = tp + fp
    return tp / denom if denom else 0.0


def recall(y_true, y_pred, positive=1):
    """TP / (TP + FN): of everything truly positive, how much was caught. 0.0 if there are no positives."""
    tp, _, _, fn = confusion_counts(y_true, y_pred, positive)
    denom = tp + fn
    return tp / denom if denom else 0.0


def f1(y_true, y_pred, positive=1):
    """Harmonic mean of precision and recall. 0.0 when both are 0."""
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    denom = p + r
    return 2 * p * r / denom if denom else 0.0