import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.regression_metrics import mae, r2, rmse


def random_pair(seed, n=300):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(50, 10, n)
    y_pred = y_true + rng.normal(0, 5, n)  # noisy but correlated predictions
    return y_true, y_pred


def test_matches_sklearn():
    for seed in range(5):
        y_true, y_pred = random_pair(seed)
        assert abs(mae(y_true, y_pred) - mean_absolute_error(y_true, y_pred)) < 1e-9
        assert abs(rmse(y_true, y_pred) - mean_squared_error(y_true, y_pred) ** 0.5) < 1e-9
        assert abs(r2(y_true, y_pred) - r2_score(y_true, y_pred)) < 1e-9


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0 and rmse(y, y) == 0.0 and r2(y, y) == 1.0


def test_rmse_punishes_a_big_miss_harder_than_mae():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    small = np.array([1.0, 1.0, 1.0, 1.0])   # four misses of 1
    one_big = np.array([0.0, 0.0, 0.0, 4.0])  # one miss of 4, same total absolute error
    assert mae(y_true, small) == mae(y_true, one_big)   # MAE cannot tell them apart
    assert rmse(y_true, one_big) > rmse(y_true, small)  # RMSE flags the big miss


def test_constant_target_returns_zero_not_crash():
    y_true = np.array([7.0, 7.0, 7.0])
    y_pred = np.array([7.0, 8.0, 6.0])
    assert r2(y_true, y_pred) == 0.0  # ss_tot == 0 handled, no ZeroDivisionError
