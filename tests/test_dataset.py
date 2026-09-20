from src.dataset import load_split


def test_split_adds_up():
    X_tr, X_te, y_tr, y_te = load_split(test_size=0.2)
    assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te)
    total = len(X_tr) + len(X_te)
    assert abs(len(X_te) - 0.2 * total) < 1


def test_same_seed_same_split():
    a = load_split(seed=1)[0].index.tolist()
    b = load_split(seed=1)[0].index.tolist()
    assert a == b