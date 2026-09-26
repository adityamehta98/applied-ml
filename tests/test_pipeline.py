from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.dataset import load_split
from src.leakage import honest_score, leaky_score, noise_data
from src.pipeline import build_model


def test_pipeline_scores_well_on_real_data():
    X_train, _, y_train, _ = load_split()
    model = build_model(X_train.columns.tolist(), [])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    assert cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean() > 0.9


def test_leak_inflates_score_on_pure_noise():
    X, y = noise_data()
    assert leaky_score(X, y) > 0.7
    assert honest_score(X, y) < 0.65
