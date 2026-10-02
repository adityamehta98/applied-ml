"""A leak-free churn Pipeline and a stratified cross-validation comparison of baselines."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import TARGET, feature_columns


def build_preprocessor(numeric, categorical):
    """Impute + scale numerics, impute + one-hot categoricals. Same pattern as applied-ml."""
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def build_pipeline(numeric, categorical, estimator):
    """Preprocessing and an estimator in one object, so every CV fold is refit leak-free."""
    return Pipeline([
        ("pre", build_preprocessor(numeric, categorical)),
        ("model", estimator),
    ])


def candidate_models():
    """The baselines every project should beat before anything fancy: a dummy, then two real models."""
    return {
        "dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "gradient boosting": HistGradientBoostingClassifier(random_state=42),
    }


def compare_models(df, seed=42):
    """Cross-validate each candidate with PR-AUC (average precision), the right metric for churn."""
    numeric, categorical = feature_columns(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    for name, estimator in candidate_models().items():
        pipe = build_pipeline(numeric, categorical, estimator)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
        rows.append({"model": name, "pr_auc": round(scores.mean(), 3),
                     "std": round(scores.std(), 3)})
    return rows
