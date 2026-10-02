"""A tiny autograd engine, built up in two cycles. Cycle 1: the forward graph."""


class Value:
    """A single scalar that remembers how it was produced (its children and op),
    so that later it can backpropagate gradients through that graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None      # filled in when we add backprop (cycle 2)
        self._prev = set(_children)        # the Values that produced this one
        self._op = _op                     # what operation made it (for debugging)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
