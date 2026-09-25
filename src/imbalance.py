"""Why accuracy lies on imbalanced data, and how class_weight plus a tuned threshold fix it.

Breast cancer is too separable to show this: a plain model already catches nearly every
case. So, as with the Day 3 leakage demo, we use a controlled synthetic dataset: 10 percent
positives and a deliberately hard signal, where ignoring the rare class really does pay off
in accuracy. That is exactly when class_weight matters.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.metrics import precision, recall
from src.threshold import threshold_for_recall


def imbalanced_split(seed=42):
    """A hard, 10-percent-positive dataset, split stratified so the test set stays imbalanced."""
    X, y = make_classification(
        n_samples=4000, n_features=20, n_informative=6, n_redundant=2,
        weights=[0.9, 0.1], class_sep=0.6, flip_y=0.01, random_state=seed)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def fit_proba(class_weight, seed=42):
    """Train logistic regression and return (y_test, probability of the positive class)."""
    X_train, X_test, y_train, y_test = imbalanced_split(seed)
    model = LogisticRegression(max_iter=5000, class_weight=class_weight)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return y_test, proba


def row(name, y_test, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "setup": name,
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "recall": round(recall(y_test, pred), 3),
        "precision": round(precision(y_test, pred), 3),
    }


def before_after(seed=42):
    plain_y, plain_p = fit_proba(None, seed)
    bal_y, bal_p = fit_proba("balanced", seed)
    tuned_t = threshold_for_recall(bal_y, bal_p, 0.90)
    return [
        row("plain logistic regression", plain_y, plain_p, 0.5),
        row("class_weight='balanced'", bal_y, bal_p, 0.5),
        row("balanced + threshold for recall 0.90", bal_y, bal_p, tuned_t),
    ]


if __name__ == "__main__":
    head = f"{'setup':40} {'accuracy':>9} {'recall':>7} {'precision':>10}"
    print(head)
    print("-" * len(head))
    for r in before_after():
        print(f"{r['setup']:40} {r['accuracy']:>9} {r['recall']:>7} {r['precision']:>10}")
