import json

import numpy as np

from src.dataset import load_split
from src.persist import load_model, train_and_save


def test_saved_and_reloaded_model_predicts_identically(tmp_path):
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "metadata.json"
    original = train_and_save(str(model_path), str(meta_path))
    reloaded = load_model(str(model_path))

    _, X_test, _, _ = load_split()
    assert np.array_equal(original.predict(X_test), reloaded.predict(X_test))
    assert np.allclose(original.predict_proba(X_test), reloaded.predict_proba(X_test))


def test_metadata_records_the_versions_needed_to_reproduce(tmp_path):
    meta_path = tmp_path / "metadata.json"
    train_and_save(str(tmp_path / "model.joblib"), str(meta_path))
    meta = json.loads(meta_path.read_text())
    for key in ("seed", "sklearn", "numpy", "python", "test_f1"):
        assert key in meta
    assert meta["seed"] == 42


def test_same_seed_reproduces_the_same_score(tmp_path):
    a = train_and_save(str(tmp_path / "a.joblib"), str(tmp_path / "a.json"), seed=1)
    b = train_and_save(str(tmp_path / "b.joblib"), str(tmp_path / "b.json"), seed=1)
    _, X_test, _, _ = load_split(seed=1)
    assert np.array_equal(a.predict(X_test), b.predict(X_test))
