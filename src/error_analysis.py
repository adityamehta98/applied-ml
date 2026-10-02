"""Look at the mistakes, not just the average score: churn recall by Contract and tenure."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import TARGET, feature_columns
from src.model import build_pipeline, candidate_models

TENURE_BINS = [-1, 12, 24, 48, 10_000]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]


def tenure_bucket(tenure):
    """Group tenure (months) into four readable bands."""
    return pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS)


def fit_and_predict(df, model_name="gradient boosting", seed=42):
    """Train the chosen model on a stratified split and attach predictions to the held-out rows."""
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    numeric, categorical = feature_columns(df)
    model = build_pipeline(numeric, categorical, candidate_models()[model_name])
    model.fit(X_tr, y_tr)
    out = X_te.copy()
    out[TARGET] = y_te.to_numpy()
    out["pred"] = model.predict(X_te)
    return out


def error_by(df_pred, column):
    """Per segment: how many rows, the real churn rate, and recall on the churners there."""
    rows = []
    for key, grp in df_pred.groupby(column, observed=True):
        churners = grp[grp[TARGET] == 1]
        recall = (churners["pred"] == 1).mean() if len(churners) else None
        rows.append({
            "segment": str(key),
            "n": len(grp),
            "churn_rate": round(float(grp[TARGET].mean()), 3),
            "recall": round(float(recall), 3) if recall is not None else None,
        })
    return rows


def error_by_tenure(df_pred):
    """error_by on a derived tenure-band column."""
    df_pred = df_pred.copy()
    df_pred["tenure_band"] = tenure_bucket(df_pred["tenure"])
    return error_by(df_pred, "tenure_band")
