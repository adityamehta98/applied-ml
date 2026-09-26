import numpy as np
import pandas as pd
import pytest

from src.features import build_preprocessor

TRAIN = pd.DataFrame({
    "tenure": [1.0, 5.0, np.nan, 12.0],
    "charges": [20.0, 35.0, 50.0, np.nan],
    "contract": ["monthly", "yearly", np.nan, "monthly"],
})
NUMERIC, CATEGORICAL = ["tenure", "charges"], ["contract"]


def fitted():
    return build_preprocessor(NUMERIC, CATEGORICAL).fit(TRAIN)


def test_output_has_no_missing_values():
    out = fitted().transform(TRAIN)
    assert not np.isnan(out).any()


def test_expected_columns():
    # 2 scaled numerics + 2 missing indicators (both had a blank) + 2 categories
    out = fitted().transform(TRAIN)
    assert out.shape == (4, 6)


def test_median_is_learned_from_training_rows():
    imputer = fitted().named_transformers_["num"].named_steps["impute"]
    assert imputer.statistics_[0] == pytest.approx(5.0)  # median of 1, 5, 12


def test_unseen_category_does_not_crash():
    live = pd.DataFrame({"tenure": [3.0], "charges": [40.0], "contract": ["two-year"]})
    onehot = fitted().transform(live)[0, -2:]
    assert (onehot == 0).all()  # unknown category -> all zeros, no error


def test_scaled_column_is_centred_on_training_data():
    out = fitted().transform(TRAIN)
    assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-9)
