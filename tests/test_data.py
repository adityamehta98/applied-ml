import torch

from src.data import TensorPairs, make_loader


def test_dataset_length():
    ds = TensorPairs(torch.randn(50, 4), torch.zeros(50))
    assert len(ds) == 50


def test_getitem_returns_one_row():
    ds = TensorPairs(torch.randn(50, 4), torch.arange(50))
    x0, y0 = ds[7]
    assert x0.shape == (4,)
    assert y0.item() == 7


def test_loader_batch_shapes():
    loader = make_loader(torch.randn(130, 4), torch.zeros(130), batch_size=64)
    batches = list(loader)
    assert len(batches) == 3                      # 64 + 64 + 2
    xb, yb = batches[0]
    assert xb.shape == (64, 4)
    assert yb.shape == (64,)
    assert batches[-1][0].shape == (2, 4)         # the remainder batch


def test_last_batch_is_the_remainder():
    loader = make_loader(torch.randn(130, 4), torch.zeros(130), batch_size=64, shuffle=False)
    total = sum(len(xb) for xb, _ in loader)
    assert total == 130
