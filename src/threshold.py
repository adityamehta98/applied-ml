import numpy as np

from src.metrics import precision, recall


def sweep(y_true, y_score, thresholds):
    """For each threshold t, decide positive when score >= t; return (t, precision, recall) rows."""
    y_score = np.asarray(y_score)
    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        rows.append((float(t), precision(y_true, pred), recall(y_true, pred)))
    return rows


def threshold_for_recall(y_true, y_score, target_recall, positive=1):
    """Highest threshold whose recall is at least target_recall.

    Raising the threshold flags fewer rows, so recall can only fall as the threshold
    rises. Among the thresholds that still reach the target, the largest one flags
    the fewest rows and so keeps precision highest. Returns None if the target is
    unreachable.
    """
    y_score = np.asarray(y_score)
    chosen = None
    for t in np.unique(y_score):  # ascending: recall is non-increasing as t grows
        pred = (y_score >= t).astype(int)
        if recall(y_true, pred, positive) >= target_recall:
            chosen = float(t)
    return chosen