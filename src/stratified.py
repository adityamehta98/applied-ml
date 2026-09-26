"""Stratified k-fold written from scratch: every fold keeps the overall class mix."""
import numpy as np


def stratified_kfold_indices(y, n_splits=5, shuffle=True, seed=42):
    """Yield (train_idx, val_idx) so each fold has the same class proportions as y."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if counts.min() < n_splits:
        raise ValueError(
            f"the smallest class has {counts.min()} rows, fewer than n_splits={n_splits}")

    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    dealt = 0
    for cls in classes:
        rows = np.flatnonzero(y == cls)
        if shuffle:
            rng.shuffle(rows)
        fold_of[rows] = (np.arange(len(rows)) + dealt) % n_splits
        dealt += len(rows)

    positions = np.arange(len(y))
    for k in range(n_splits):
        yield positions[fold_of != k], positions[fold_of == k]
