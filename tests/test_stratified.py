import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from src.cv import kfold_indices
from src.stratified import stratified_kfold_indices

SORTED_LABELS = np.array([0] * 90 + [1] * 10)  # a file sorted by label, 10% positives


def positives_per_fold(splits, y):
    return [int(y[val].sum()) for _, val in splits]


def test_plain_kfold_breaks_on_sorted_labels():
    counts = positives_per_fold(kfold_indices(100, 5, shuffle=False), SORTED_LABELS)
    assert counts == [0, 0, 0, 0, 10]


def test_stratified_gives_every_fold_the_same_share():
    counts = positives_per_fold(stratified_kfold_indices(SORTED_LABELS, 5), SORTED_LABELS)
    assert counts == [2, 2, 2, 2, 2]


def test_every_row_validated_exactly_once():
    y = load_breast_cancer().target
    val = np.concatenate([v for _, v in stratified_kfold_indices(y, 5)])
    assert sorted(val.tolist()) == list(range(len(y)))


def test_class_ratio_matches_overall_in_every_fold():
    y = load_breast_cancer().target
    overall = y.mean()
    for _, val in stratified_kfold_indices(y, 5):
        assert abs(y[val].mean() - overall) < 0.01


def test_fold_sizes_differ_by_at_most_one():
    y = load_breast_cancer().target
    sizes = [len(v) for _, v in stratified_kfold_indices(y, 5)]
    assert max(sizes) - min(sizes) <= 1


def test_rejects_class_smaller_than_k():
    with pytest.raises(ValueError):
        list(stratified_kfold_indices([0, 0, 0, 0, 0, 1, 1], 5))
