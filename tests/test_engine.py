"""Cycle 1 tests: the forward pass and the computation graph (no backprop yet)."""
from micrograd.engine import Value


def test_add_and_mul_values():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0


def test_children_are_recorded():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    assert c._prev == {a, b}
    assert c._op == "*"


def test_python_numbers_are_wrapped():
    a = Value(2.0)
    c = a + 1                       # 1 is not a Value; __add__ wraps it
    assert c.data == 3.0


def test_expression_builds_a_graph():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c                   # d = 2*-3 + 10 = 4
    assert d.data == 4.0
    assert d._op == "+"
