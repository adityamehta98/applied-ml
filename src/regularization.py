"""L1 vs L2 regularization: how the penalty and its strength change a linear model."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logreg(X, y, penalty, C, seed=42):
    """A scaled logistic regression. liblinear is the solver that supports both L1 and L2."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty=penalty, C=C, solver="liblinear",
                           max_iter=5000, random_state=seed))
    return model.fit(X, y)


def nonzero_coefs(model):
    """How many feature weights survived. L1 drives some to exactly 0; L2 does not."""
    return int(np.sum(model[-1].coef_[0] != 0))


def compare(X, y, penalties=("l1", "l2"), cs=(0.01, 0.1, 1.0), seed=42):
    """Return a row per (penalty, C): non-zero coefficient count and cross-validated F1."""
    rows = []
    for penalty in penalties:
        for c in cs:
            model = fit_logreg(X, y, penalty, c, seed)
            cv_f1 = cross_val_score(model, X, y, cv=5, scoring="f1").mean()
            rows.append({"penalty": penalty, "C": c,
                         "nonzero": nonzero_coefs(model), "cv_f1": round(cv_f1, 3)})
    return rows


if __name__ == "__main__":
    from src.dataset import load_split

    X_train, _, y_train, _ = load_split()
    total = X_train.shape[1]
    print(f"{'penalty':8} {'C':>6} {'nonzero':>12} {'cv_f1':>7}")
    print("-" * 36)
    for r in compare(X_train, y_train):
        print(f"{r['penalty']:8} {r['C']:>6} {str(r['nonzero']) + '/' + str(total):>12} {r['cv_f1']:>7}")
