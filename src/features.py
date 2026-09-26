"""Preprocessing: turn a raw table into numbers a model can train on."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric, categorical):
    """Impute and scale numeric columns; impute and one-hot encode categorical ones.

    numeric, categorical: lists of column names.
    Returns an unfitted ColumnTransformer. Fit it on training data only.
    """
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, categorical),
    ])
