"""Gradient descent by hand, then the same fit with nn.Linear + optim.SGD."""
import torch
import torch.nn as nn


def make_line(n=200, w=3.0, b=2.0, noise=0.1, seed=0):
    """y = w*x + b plus a little noise. The truth we try to recover."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 10 - 5           # x in [-5, 5]
    y = w * x + b + noise * torch.randn(n, 1, generator=g)
    return x, y


def fit_manual(x, y, lr=0.03, steps=400):
    """Fit y = w*x + b with raw tensors: no optimizer, no nn.Module."""
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(steps):
        pred = x * w + b
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    return w.item(), b.item()


def fit_with_module(x, y, lr=0.03, steps=400):
    """The same fit with the framework: nn.Linear and optim.SGD."""
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model.weight.item(), model.bias.item()
