"""Tensor drills: the four moves you repeat in every PyTorch model."""
import torch


def standardize(x):
    """Center each column at 0 and scale to unit std, using broadcasting."""
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)


def add_bias_column(x):
    """Prepend a column of ones so a linear layer's bias is just another weight."""
    ones = torch.ones(x.shape[0], 1)
    return torch.cat([ones, x], dim=1)


def batched_matmul(a, b):
    """Multiply a stack of matrices: (B, n, k) @ (B, k, m) -> (B, n, m)."""
    return torch.bmm(a, b)


def to_device(x, device="cpu"):
    """Move a tensor to a device. On a machine with a GPU, device='cuda'."""
    return x.to(device)
