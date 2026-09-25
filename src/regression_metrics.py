"""Regression metrics from scratch: MAE, RMSE, and R2."""
import numpy as np


def mae(y_true, y_pred):
    """Mean absolute error: the average size of a miss, in the target's own units."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error: like MAE but squares each miss first, so big misses hurt more."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    """R squared: the fraction of the target's variance the model explains. 1.0 is perfect, 0.0 ties the mean."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0  # the target never varies, so "variance explained" is undefined; report 0.0
    return float(1 - ss_res / ss_tot)
