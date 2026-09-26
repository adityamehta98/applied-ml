"""Grid search vs randomized search over the Day 3 Pipeline, judged by cross-validation only."""
import time

import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.pipeline import build_model


def grid_search(X, y, cv=5, seed=42):
    """Try every C in a fixed list. Exhaustive but the cost is len(grid) x folds fits."""
    model = build_model(X.columns.tolist(), [])
    grid = {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    search = GridSearchCV(model, grid, cv=cv, scoring="f1", n_jobs=-1)
    t = time.perf_counter()
    search.fit(X, y)
    return search, time.perf_counter() - t


def random_search(X, y, n_iter=7, cv=5, seed=42):
    """Sample C from a log-uniform range. Same fit budget, but it can land between grid points."""
    model = build_model(X.columns.tolist(), [])
    dist = {"model__C": loguniform(1e-2, 1e1)}
    search = RandomizedSearchCV(model, dist, n_iter=n_iter, cv=cv,
                                scoring="f1", random_state=seed, n_jobs=-1)
    t = time.perf_counter()
    search.fit(X, y)
    return search, time.perf_counter() - t


if __name__ == "__main__":
    from sklearn.metrics import f1_score

    from src.dataset import load_split

    X_train, X_test, y_train, y_test = load_split()  # test set sealed until the very end
    gs, gt = grid_search(X_train, y_train)
    rs, rt = random_search(X_train, y_train)
    print(f"grid   : best CV F1 {gs.best_score_:.3f}  C={gs.best_params_['model__C']:.3g}  ({gt:.1f}s)")
    print(f"random : best CV F1 {rs.best_score_:.3f}  C={rs.best_params_['model__C']:.3g}  ({rt:.1f}s)")
    winner = gs if gs.best_score_ >= rs.best_score_ else rs
    test_f1 = f1_score(y_test, winner.predict(X_test))
    print(f"confirmed once on the sealed test set: F1 {test_f1:.3f}")
