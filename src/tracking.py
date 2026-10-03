"""An append-only run log: one JSON line per run, kept even when the run was bad."""
import json
import time

from src.diagnose import diagnose


def log_run(path, cfg, history, note=""):
    """Append one run (config, final metrics, diagnosis, note) to a JSON Lines file."""
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg.to_dict(),
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": min(history["val_loss"]),
        "diagnosis": diagnose(history["train_loss"], history["val_loss"])["label"],
        "note": note,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load_runs(path):
    """Read every logged run back as a list of dicts."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def best_run(path):
    """The run with the lowest final validation loss, or None if the log is empty."""
    runs = load_runs(path)
    return min(runs, key=lambda r: r["final_val_loss"]) if runs else None
