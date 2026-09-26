"""Reproducibility: save the whole fitted Pipeline and the versions it was built with."""
import json
import platform

import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score

from src.dataset import load_split
from src.pipeline import build_model


def train_and_save(model_path, meta_path, seed=42):
    """Fit the Pipeline, save it, and write the versions needed to reproduce it later."""
    X_train, X_test, y_train, y_test = load_split(seed=seed)
    model = build_model(X_train.columns.tolist(), [])
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    meta = {
        "seed": seed,
        "test_f1": round(f1_score(y_test, model.predict(X_test)), 4),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return model


def load_model(model_path):
    """Load the fitted Pipeline back. It carries its preprocessing, so it predicts on raw rows."""
    return joblib.load(model_path)


if __name__ == "__main__":
    import os

    os.makedirs("models", exist_ok=True)
    train_and_save("models/model.joblib", "models/metadata.json")
    reloaded = load_model("models/model.joblib")
    X_train, X_test, _, _ = load_split()
    print("saved models/model.joblib and models/metadata.json")
    print("reloaded model predicts on raw rows:", reloaded.predict(X_test[:3]).tolist())
