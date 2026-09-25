import numpy as np

from src.metrics import recall
from src.threshold import sweep, threshold_for_recall


def test_recall_falls_as_threshold_rises():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.6])
    recalls = [r for _, _, r in sweep(y_true, y_score, [0.0, 0.5, 0.9])]
    assert recalls == sorted(recalls, reverse=True)  # non-increasing


def test_threshold_hits_the_target_recall():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.55, 0.3, 0.45, 0.7, 0.9])
    t = threshold_for_recall(y_true, y_score, 0.75)
    pred = (y_score >= t).astype(int)
    assert recall(y_true, pred) >= 0.75


def test_picks_the_highest_threshold_that_still_reaches_the_target():
    # recall 1.0 needs t <= 0.30; the largest such distinct score is 0.30
    y_true = np.array([0, 1, 1, 1])
    y_score = np.array([0.05, 0.30, 0.60, 0.90])
    assert threshold_for_recall(y_true, y_score, 1.0) == 0.30


def test_unreachable_target_returns_none():
    y_true = np.array([1, 1, 1, 0])
    y_score = np.array([0.1, 0.1, 0.1, 0.1])  # only threshold 0.1 flags all -> recall 1.0 at most
    assert threshold_for_recall(y_true, y_score, 1.01) is None  # >100% recall is impossible