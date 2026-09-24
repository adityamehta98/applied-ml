import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)

from src.metrics import confusion_counts, f1, precision, recall


def random_labels(seed, n=500):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n), rng.integers(0, 2, n)


def test_confusion_counts_match_sklearn():
    y_true, y_pred = random_labels(0)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, y_pred).ravel()
    assert (tp, fp, tn, fn) == (tp_s, fp_s, tn_s, fn_s)


def test_matches_sklearn_on_random_labels():
    for seed in range(5):
        y_true, y_pred = random_labels(seed)
        assert precision(y_true, y_pred) == precision_score(y_true, y_pred, zero_division=0)
        assert recall(y_true, y_pred) == recall_score(y_true, y_pred, zero_division=0)
        assert abs(f1(y_true, y_pred) - f1_score(y_true, y_pred, zero_division=0)) < 1e-12


def test_perfect_predictions():
    y = np.array([0, 1, 1, 0, 1])
    assert precision(y, y) == 1.0 and recall(y, y) == 1.0 and f1(y, y) == 1.0


def test_no_positive_predictions_gives_zero_not_crash():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.zeros(5, dtype=int)
    assert precision(y_true, y_pred) == 0.0  # 0/0 -> 0.0, not a ZeroDivisionError
    assert recall(y_true, y_pred) == 0.0
    assert f1(y_true, y_pred) == 0.0


def test_no_real_positives_gives_zero_recall():
    y_true = np.zeros(5, dtype=int)
    y_pred = np.array([0, 1, 0, 1, 0])
    assert recall(y_true, y_pred) == 0.0
    assert precision(y_true, y_pred) == 0.0


def test_positive_label_can_be_the_rare_class():
    y_true = np.array([1, 1, 1, 0, 1, 1])  # class 0 is the rare "positive"
    y_pred = np.array([1, 1, 0, 0, 1, 1])
    assert recall(y_true, y_pred, positive=0) == 1.0  # the one 0 was caught