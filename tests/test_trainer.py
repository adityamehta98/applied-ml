import torch
import torch.nn as nn

from src.trainer import build_mlp, evaluate, train_step


def test_mlp_output_shape():
    model = build_mlp()
    out = model(torch.randn(16, 1, 28, 28))     # a batch of fake images
    assert out.shape == (16, 10)


def test_one_step_lowers_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    model = build_mlp(in_features=20, hidden=32, classes=3)
    xb = torch.randn(64, 20)
    yb = torch.randint(0, 3, (64,))
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    loss_fn = nn.CrossEntropyLoss()

    before = loss_fn(model(xb), yb).item()
    for _ in range(20):
        train_step(model, xb, yb, opt, loss_fn)
    after = loss_fn(model(xb), yb).item()
    assert after < before


def test_evaluate_reports_perfect_accuracy_when_it_should():
    model = build_mlp(in_features=4, hidden=8, classes=2)
    xb = torch.randn(10, 4)
    yb = model(xb).argmax(dim=1)                 # labels the model already agrees with
    _, acc = evaluate(model, xb, yb, nn.CrossEntropyLoss())
    assert acc == 1.0
