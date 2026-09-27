import pytest

from src.linreg import fit_manual, fit_with_module, make_line


def test_manual_fit_recovers_w_and_b():
    x, y = make_line(w=3.0, b=2.0)
    w, b = fit_manual(x, y)
    assert w == pytest.approx(3.0, abs=0.1)
    assert b == pytest.approx(2.0, abs=0.1)


def test_module_fit_recovers_w_and_b():
    x, y = make_line(w=3.0, b=2.0)
    w, b = fit_with_module(x, y)
    assert w == pytest.approx(3.0, abs=0.1)
    assert b == pytest.approx(2.0, abs=0.1)


def test_manual_and_module_agree():
    x, y = make_line(w=3.0, b=2.0)
    wm, bm = fit_manual(x, y)
    wt, bt = fit_with_module(x, y)
    assert wm == pytest.approx(wt, abs=0.1)
    assert bm == pytest.approx(bt, abs=0.1)
