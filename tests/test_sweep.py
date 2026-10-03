import csv

import torch

from src.config import TrainConfig
from src.fit import fit
from src.sweep import grid, run_sweep, save_csv


def test_grid_expands_every_combination():
    configs = grid(TrainConfig(), lr=[1e-3, 1e-2], hidden=[16, 32, 64])
    assert len(configs) == 6
    assert {c.hidden for c in configs} == {16, 32, 64}


def test_best_config_is_first_and_chosen_by_validation_loss():
    def fake_run(cfg):
        # pretend a middling learning rate is best; training loss is misleadingly low for the biggest lr
        val = abs(cfg.lr - 1e-2)
        return {"train_loss": [0.01], "val_loss": [val]}

    rows = run_sweep(grid(TrainConfig(), lr=[1e-4, 1e-2, 1e-1]), fake_run)
    assert rows[0]["lr"] == 1e-2
    assert rows[0]["val_loss"] <= rows[1]["val_loss"] <= rows[2]["val_loss"]


def test_rows_record_how_long_each_run_took():
    rows = run_sweep(grid(TrainConfig(), lr=[1e-3]), lambda cfg: {"train_loss": [1.0], "val_loss": [1.0]})
    assert rows[0]["seconds"] >= 0


def test_a_real_tiny_sweep_saves_a_csv(tmp_path):
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (80,), generator=g)
    x = torch.randn(80, 4, generator=g) + y.unsqueeze(1) * 2.0

    def run(cfg):
        return fit(cfg, x, y, x, y, classes=2)[1]

    rows = run_sweep(grid(TrainConfig(epochs=2), lr=[1e-3, 1e-2]), run)
    path = tmp_path / "sweep.csv"
    save_csv(rows, str(path))
    with open(path, encoding="utf-8") as fh:
        saved = list(csv.DictReader(fh))
    assert len(saved) == 2
    assert "val_loss" in saved[0]
