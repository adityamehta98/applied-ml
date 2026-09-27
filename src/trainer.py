"""The training loop as reusable pieces, plus a FashionMNIST script in main()."""
import torch
import torch.nn as nn


def build_mlp(in_features=784, hidden=128, classes=10):
    """A plain multi-layer perceptron: flatten, linear, ReLU, linear."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, classes),
    )


def train_step(model, xb, yb, opt, loss_fn):
    """One optimization step. Returns the loss before the update, as a float."""
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, xb, yb, loss_fn):
    """Loss and accuracy in eval mode, without building the gradient graph."""
    model.eval()
    logits = model(xb)
    loss = loss_fn(logits, yb).item()
    acc = (logits.argmax(dim=1) == yb).float().mean().item()
    return loss, acc


def main():  # pragma: no cover - downloads data, run by hand not in tests
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tf = transforms.ToTensor()
    train = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    tl, vl = DataLoader(train, 64, shuffle=True), DataLoader(test, 256)

    model = build_mlp()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(5):
        for xb, yb in tl:
            train_step(model, xb, yb, opt, loss_fn)
        xb, yb = next(iter(vl))
        print("epoch", epoch, "val", evaluate(model, xb, yb, loss_fn))


if __name__ == "__main__":
    main()
