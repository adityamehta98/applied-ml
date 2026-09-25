import numpy as np

from src.dataset import load_split
from src.overfitting import depth_curve, gap, sweet_spot

DEPTHS = [1, 2, 3, 4, 5, 7, 10, 15, 20]


def test_sweet_spot_picks_the_best_validation_depth():
    val = np.array([0.91, 0.93, 0.95, 0.94, 0.90])  # peaks at index 2
    assert sweet_spot([1, 2, 3, 4, 5], val) == 3


def test_training_score_rises_with_depth_but_validation_peaks_then_falls():
    X_train, _, y_train, _ = load_split()
    train, val = depth_curve(X_train, y_train, DEPTHS)
    assert train[-1] >= train[0]                 # deeper trees fit training data better
    assert train[-1] > 0.99                       # a deep tree nearly memorizes the training set
    best = sweet_spot(DEPTHS, val)
    assert 2 <= best <= 7                          # the best depth is moderate, not the deepest
    assert val[DEPTHS.index(best)] > val[-1]       # the deepest tree is worse on validation


def test_gap_widens_as_the_tree_overfits():
    X_train, _, y_train, _ = load_split()
    train, val = depth_curve(X_train, y_train, DEPTHS)
    g = gap(train, val)
    assert g[-1] > g[0]  # train-minus-val gap is larger at max depth than at depth 1
