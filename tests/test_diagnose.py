from src.diagnose import diagnose


def label(train, val):
    return diagnose(train, val)["label"]


def test_healthy_curve():
    assert label([1.0, 0.6, 0.4, 0.3], [1.0, 0.65, 0.5, 0.36]) == "healthy"


def test_overfitting_curve():
    assert label([0.9, 0.5, 0.3, 0.2], [0.9, 0.6, 0.7, 0.9]) == "overfitting"


def test_underfitting_curve():
    assert label([1.9, 1.9, 1.88, 1.9], [1.9, 1.9, 1.9, 1.9]) == "underfitting"


def test_unstable_curve():
    assert label([1.0, 0.5, 1.2, 0.4, 0.3], [1.0, 0.6, 1.0, 0.5, 0.4]) == "unstable"


def test_nan_is_diverged():
    assert label([1.0, float("nan"), float("nan")], [1.0, float("nan"), float("nan")]) == "diverged"


def test_short_run_is_not_judged():
    assert label([1.0, 0.9], [1.0, 0.95]) == "too_short"


def test_overfitting_reports_the_best_epoch_and_an_action():
    result = diagnose([0.9, 0.5, 0.3, 0.2], [0.9, 0.6, 0.7, 0.9])
    assert result["best_epoch"] == 1
    assert "Stop" in result["action"]
