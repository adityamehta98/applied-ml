import numpy as np
import pandas as pd
import torch

from src.data import TARGET, TensorPairs, clean, feature_columns, make_loader
from src.eda import summarize

def test_clean_drops_customer_id(raw_df):
    assert "customerID" not in clean(raw_df).columns


def test_total_charges_becomes_numeric_with_blanks_as_nan(raw_df):
    out = clean(raw_df)
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    # the brand-new customers (tenure 0) had a blank string; it is NaN now, not a crash
    assert out["TotalCharges"].isna().sum() == (raw_df["tenure"] == 0).sum()

def test_churn_becomes_zero_one(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert out[TARGET].dtype == np.int64 or out[TARGET].dtype == int


def test_feature_columns_split_numeric_and_categorical(raw_df):
    out = clean(raw_df)
    numeric, categorical = feature_columns(out)
    assert "tenure" in numeric and "MonthlyCharges" in numeric
    assert "Contract" in categorical and "PaymentMethod" in categorical
    assert TARGET not in numeric and TARGET not in categorical


def test_summarize_reports_the_key_facts(raw_df):
    facts = summarize(clean(raw_df))
    assert facts["rows"] == len(raw_df)
    assert 0.1 < facts["churn_rate"] < 0.5      # imbalanced but not extreme
    assert facts["n_categorical"] > facts["n_numeric"]

def test_dataset_length():
    ds = TensorPairs(torch.randn(50, 4), torch.zeros(50))
    assert len(ds) == 50


def test_getitem_returns_one_row():
    ds = TensorPairs(torch.randn(50, 4), torch.arange(50))
    x0, y0 = ds[7]
    assert x0.shape == (4,)
    assert y0.item() == 7


def test_loader_batch_shapes():
    loader = make_loader(torch.randn(130, 4), torch.zeros(130), batch_size=64)
    batches = list(loader)
    assert len(batches) == 3                      # 64 + 64 + 2
    xb, yb = batches[0]
    assert xb.shape == (64, 4)
    assert yb.shape == (64,)
    assert batches[-1][0].shape == (2, 4)         # the remainder batch


def test_last_batch_is_the_remainder():
    loader = make_loader(torch.randn(130, 4), torch.zeros(130), batch_size=64, shuffle=False)
    total = sum(len(xb) for xb, _ in loader)
    assert total == 130
