"""Overfitting made visible: a validation curve over a decision tree's max_depth."""
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier


def depth_curve(X, y, depths, cv=5, seed=42):
    """Return (train_means, val_means): mean F1 at each max_depth, on train folds and val folds."""
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(random_state=seed), X, y,
        param_name="max_depth", param_range=list(depths),
        cv=cv, scoring="f1", n_jobs=-1)
    return train_scores.mean(axis=1), val_scores.mean(axis=1)


def sweet_spot(depths, val_means):
    """The depth with the highest cross-validated score: the edge of the overfitting cliff."""
    return list(depths)[int(np.argmax(val_means))]


def gap(train_means, val_means):
    """Train score minus validation score at each depth. A widening gap is overfitting."""
    return np.asarray(train_means) - np.asarray(val_means)
