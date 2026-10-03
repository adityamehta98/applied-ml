from src.config import TrainConfig
from src.tracking import best_run, load_runs, log_run

GOOD = {"train_loss": [1.0, 0.6, 0.4, 0.3], "val_loss": [1.0, 0.65, 0.5, 0.36]}
BAD = {"train_loss": [0.9, 0.5, 0.3, 0.2], "val_loss": [0.9, 0.6, 0.7, 0.9]}


def test_runs_are_appended_not_overwritten(tmp_path):
    path = str(tmp_path / "runs.jsonl")
    log_run(path, TrainConfig(lr=1e-3), GOOD, note="baseline")
    log_run(path, TrainConfig(lr=1e-1), BAD, note="too aggressive")
    assert len(load_runs(path)) == 2


def test_each_run_keeps_its_config_note_and_diagnosis(tmp_path):
    path = str(tmp_path / "runs.jsonl")
    log_run(path, TrainConfig(lr=1e-1), BAD, note="too aggressive")
    run = load_runs(path)[0]
    assert run["config"]["lr"] == 1e-1
    assert run["note"] == "too aggressive"
    assert run["diagnosis"] == "overfitting"


def test_best_run_is_the_lowest_final_validation_loss(tmp_path):
    path = str(tmp_path / "runs.jsonl")
    log_run(path, TrainConfig(lr=1e-3), GOOD)
    log_run(path, TrainConfig(lr=1e-1), BAD)
    assert best_run(path)["config"]["lr"] == 1e-3
