"""A custom Dataset and a DataLoader: how PyTorch feeds data in batches."""
import torch
from torch.utils.data import DataLoader, Dataset


class TensorPairs(Dataset):
    """Wrap features X and labels y so a DataLoader can batch and shuffle them."""

    def __init__(self, X, y):
        assert len(X) == len(y), "X and y must have the same number of rows"
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loader(X, y, batch_size=64, shuffle=True):
    """Return a DataLoader that yields (batch_X, batch_y) tuples."""
    return DataLoader(TensorPairs(X, y), batch_size=batch_size, shuffle=shuffle)
