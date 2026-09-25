from src.imbalance import before_after, fit_proba
from src.metrics import recall


def test_accuracy_looks_high_but_recall_is_poor_without_weighting():
    y, p = fit_proba(None)
    pred = (p >= 0.5).astype(int)
    assert (pred == y).mean() > 0.90          # accuracy looks great
    assert recall(y, pred) < 0.5              # yet it misses most of the rare class


def test_class_weight_lifts_recall_a_lot():
    plain_y, plain_p = fit_proba(None)
    bal_y, bal_p = fit_proba("balanced")
    plain_recall = recall(plain_y, (plain_p >= 0.5).astype(int))
    bal_recall = recall(bal_y, (bal_p >= 0.5).astype(int))
    assert bal_recall > plain_recall + 0.3    # balancing roughly doubles recall here


def test_before_after_table_shape_and_trend():
    rows = before_after()
    assert [r["setup"] for r in rows][0].startswith("plain")
    # accuracy falls but recall rises as we move down the table
    assert rows[0]["accuracy"] > rows[2]["accuracy"]
    assert rows[2]["recall"] >= 0.90          # the tuned threshold hits its target
