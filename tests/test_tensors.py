import torch

from src.tensors import add_bias_column, batched_matmul, standardize, to_device


def test_broadcasting_standardizes_each_column():
    x = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    out = standardize(x)
    assert torch.allclose(out.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert out.shape == (3, 2)


def test_matmul_shape_rule():
    a = torch.randn(4, 3)
    b = torch.randn(3, 2)
    assert (a @ b).shape == (4, 2)


def test_matmul_shape_mismatch_raises():
    a = torch.randn(4, 3)
    b = torch.randn(2, 5)
    try:
        _ = a @ b
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_add_bias_column():
    x = torch.randn(5, 3)
    out = add_bias_column(x)
    assert out.shape == (5, 4)
    assert torch.all(out[:, 0] == 1.0)


def test_batched_matmul():
    a = torch.randn(8, 4, 3)
    b = torch.randn(8, 3, 2)
    assert batched_matmul(a, b).shape == (8, 4, 2)


def test_to_device_cpu_roundtrip():
    x = torch.randn(2, 2)
    assert to_device(x, "cpu").device.type == "cpu"
