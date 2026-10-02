from src.data import TARGET, clean, feature_columns
from src.model import build_pipeline, candidate_models, compare_models


def test_pipeline_fits_and_predicts_on_raw_categoricals(raw_df):
    df = clean(raw_df)
    numeric, categorical = feature_columns(df)
    pipe = build_pipeline(numeric, categorical, candidate_models()["logistic regression"])
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds) <= {0, 1}


def test_real_models_beat_the_dummy_baseline(raw_df):
    rows = compare_models(clean(raw_df))
    by_name = {r["model"]: r["pr_auc"] for r in rows}
    dummy = by_name["dummy (most frequent)"]
    assert by_name["logistic regression"] > dummy
    assert by_name["gradient boosting"] > dummy


def test_dummy_pr_auc_is_about_the_base_rate(raw_df):
    df = clean(raw_df)
    rows = compare_models(df)
    dummy = next(r for r in rows if r["model"].startswith("dummy"))
    assert abs(dummy["pr_auc"] - df[TARGET].mean()) < 0.05  # PR-AUC of a no-skill model ~ positive rate
