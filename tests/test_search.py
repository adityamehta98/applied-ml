from sklearn.metrics import f1_score

from src.dataset import load_split
from src.search import grid_search, random_search


def test_grid_search_finds_a_good_C():
    X_train, _, y_train, _ = load_split()
    search, seconds = grid_search(X_train, y_train)
    assert "model__C" in search.best_params_        # it tuned the pipeline step, not a bare model
    assert search.best_score_ > 0.95
    assert seconds >= 0


def test_random_search_is_competitive_with_fewer_fits():
    X_train, _, y_train, _ = load_split()
    rs, _ = random_search(X_train, y_train, n_iter=7)
    assert rs.best_score_ > 0.95                     # sampling lands on a near-best C too


def test_the_test_set_is_only_touched_after_tuning():
    # tuning uses only the training data; the test score is read once, at the end
    X_train, X_test, y_train, y_test = load_split()
    search, _ = grid_search(X_train, y_train)
    test_f1 = f1_score(y_test, search.predict(X_test))
    assert test_f1 > 0.9
