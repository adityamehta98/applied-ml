import warnings

import numpy as np

from src.dataset import load_split
from src.regularization import compare, fit_logreg, nonzero_coefs

# penalty= is being deprecated in very recent sklearn in favour of l1_ratio; the classic
# API is still correct and portable, so we silence that one FutureWarning here.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def test_l1_zeroes_coefficients_at_strong_regularization():
    X_train, _, y_train, _ = load_split()
    total = X_train.shape[1]
    l1 = nonzero_coefs(fit_logreg(X_train, y_train, "l1", C=0.01))
    l2 = nonzero_coefs(fit_logreg(X_train, y_train, "l2", C=0.01))
    assert l1 < total          # L1 drops features
    assert l2 == total         # L2 keeps them all, just small


def test_l1_gets_sparser_as_C_shrinks():
    X_train, _, y_train, _ = load_split()
    strong = nonzero_coefs(fit_logreg(X_train, y_train, "l1", C=0.01))
    weak = nonzero_coefs(fit_logreg(X_train, y_train, "l1", C=1.0))
    assert strong < weak       # smaller C means stronger penalty means fewer surviving features


def test_compare_table_has_a_row_per_combo():
    X_train, _, y_train, _ = load_split()
    rows = compare(X_train, y_train)
    assert len(rows) == 6      # 2 penalties x 3 C values
    for r in rows:
        assert 0 <= r["nonzero"] <= X_train.shape[1]
        assert 0.8 < r["cv_f1"] <= 1.0
