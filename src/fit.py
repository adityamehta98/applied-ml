"""A training run as one function: same config in, same history out."""
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.trainer import build_mlp, evaluate, train_step


def set_seed(seed):
    """Fix the random number generators so a run can be repeated exactly."""
    random.seed(seed)
    torch.manual_seed(seed)


def fit(cfg, x_train, y_train, x_val, y_val, classes):
    """Train an MLP for cfg.epochs. Returns (model, history)."""
    set_seed(cfg.seed)
    model = build_mlp(in_features=x_train.shape[1], hidden=cfg.hidden, classes=classes)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for _ in range(cfg.epochs):
        losses = [train_step(model, xb, yb, opt, loss_fn) for xb, yb in loader]
        history["train_loss"].append(sum(losses) / len(losses))
        val_loss, val_acc = evaluate(model, x_val, y_val, loss_fn)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    return model, history
