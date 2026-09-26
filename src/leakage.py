"""A leak you can measure: feature selection before vs inside cross-validation."""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def noise_data(n_rows=200, n_cols=5000, seed=0):
    """Pure noise: random features and random labels. No honest model beats 50%."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_cols))
    y = rng.integers(0, 2, size=n_rows)
    return X, y


def leaky_score(X, y, k=20, seed=0):
    """WRONG: pick features using every row, then cross-validate."""
    X_selected = SelectKBest(f_classif, k=k).fit_transform(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_score(LogisticRegression(max_iter=1000), X_selected, y, cv=cv).mean()


def honest_score(X, y, k=20, seed=0):
    """RIGHT: selection lives inside the Pipeline, so it is refitted on each training fold."""
    pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=k)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_score(pipe, X, y, cv=cv).mean()


if __name__ == "__main__":
    X, y = noise_data()
    print(f"Leaky  (select on all rows, then CV): accuracy {leaky_score(X, y):.2f}")
    print(f"Honest (select inside each fold):     accuracy {honest_score(X, y):.2f}")
