from src.data import clean
from src.error_analysis import (error_by, error_by_tenure, fit_and_predict,
                                tenure_bucket)


def test_tenure_bucket_labels():
    assert str(tenure_bucket([0])[0]) == "0-12"
    assert str(tenure_bucket([60])[0]) == "49+"
    assert str(tenure_bucket([12])[0]) == "0-12"    # 12 is the top of the first band
    assert str(tenure_bucket([13])[0]) == "13-24"


def test_segment_counts_add_up_to_the_test_set(raw_df):
    pred = fit_and_predict(clean(raw_df))
    rows = error_by(pred, "Contract")
    assert sum(r["n"] for r in rows) == len(pred)


def test_month_to_month_churns_more_than_two_year(raw_df):
    pred = fit_and_predict(clean(raw_df))
    by = {r["segment"]: r["churn_rate"] for r in error_by(pred, "Contract")}
    assert by["Month-to-month"] > by["Two year"]  # the signal we built in shows up in the segments


def test_error_by_tenure_returns_a_row_per_band(raw_df):
    pred = fit_and_predict(clean(raw_df))
    rows = error_by_tenure(pred)
    segments = {r["segment"] for r in rows}
    assert segments <= {"0-12", "13-24", "25-48", "49+"}
    assert sum(r["n"] for r in rows) == len(pred)
