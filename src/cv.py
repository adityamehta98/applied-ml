"""K-fold cross-validation written from scratch (no sklearn splitters)."""
import numpy as np
from sklearn.base import clone


def kfold_indices(n_samples, n_splits=5, shuffle=True, seed=42):
    """Yield (train_idx, val_idx) for each of n_splits folds.

    Every row lands in exactly one validation fold. Fold sizes differ by at most one.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot be larger than the number of rows")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    start = 0
    for size in fold_sizes:
        stop = start + size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        yield train_idx, val_idx
        start = stop


def cross_val_scores(model, X, y, splits, metric):
    """Fit a fresh copy of `model` on each training fold and score it on the validation fold."""
    scores = []
    for train_idx, val_idx in splits:
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = fold_model.predict(X.iloc[val_idx])
        scores.append(metric(y.iloc[val_idx], preds))
    return np.array(scores)
