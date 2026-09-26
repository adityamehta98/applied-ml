import numpy as np
import pytest
from sklearn.model_selection import KFold

from src.cv import kfold_indices


def test_every_row_validated_exactly_once():
    val = np.concatenate([v for _, v in kfold_indices(103, 5)])
    assert sorted(val.tolist()) == list(range(103))


def test_train_and_validation_never_overlap():
    for train, val in kfold_indices(50, 5):
        assert set(train).isdisjoint(val)
        assert len(train) + len(val) == 50


def test_fold_sizes_match_sklearn():
    ours = [len(v) for _, v in kfold_indices(103, 5)]
    theirs = [len(v) for _, v in KFold(5).split(np.zeros(103))]
    assert ours == theirs == [21, 21, 21, 20, 20]


def test_without_shuffle_matches_sklearn_exactly():
    ours = [v.tolist() for _, v in kfold_indices(20, 4, shuffle=False)]
    theirs = [v.tolist() for _, v in KFold(4, shuffle=False).split(np.zeros(20))]
    assert ours == theirs


def test_same_seed_same_folds():
    a = [v.tolist() for _, v in kfold_indices(40, 4, seed=7)]
    b = [v.tolist() for _, v in kfold_indices(40, 4, seed=7)]
    assert a == b


@pytest.mark.parametrize("k", [1, 11])
def test_rejects_bad_k(k):
    with pytest.raises(ValueError):
        list(kfold_indices(10, k))
