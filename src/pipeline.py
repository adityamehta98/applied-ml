"""The full model: preprocessing and a classifier in one leak-free Pipeline."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


def build_model(numeric, categorical):
    """Preprocess, then classify. Fit it on training data; everything inside is fitted together."""
    return Pipeline([
        ("pre", build_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=5000)),
    ])
