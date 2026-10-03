"""Try a grid of settings, time each run, and rank the results."""
import csv
import itertools
import time
from dataclasses import replace


def grid(base, **options):
    """Every combination of the option lists, each applied to a copy of base."""
    names = list(options)
    return [replace(base, **dict(zip(names, combo)))
            for combo in itertools.product(*options.values())]


def run_sweep(configs, run_fn):
    """run_fn(cfg) returns a history dict. Rows come back best (lowest val loss) first."""
    rows = []
    for cfg in configs:
        start = time.perf_counter()
        hist = run_fn(cfg)
        seconds = time.perf_counter() - start
        val = hist["val_loss"]
        rows.append({
            **cfg.to_dict(),
            "val_loss": val[-1],
            "best_val_loss": min(val),
            "train_loss": hist["train_loss"][-1],
            "seconds": round(seconds, 3),
        })
    rows.sort(key=lambda r: r["val_loss"])
    return rows


def save_csv(rows, path):
    """Write the ranked table so the evidence behind a choice is kept."""
    if not rows:
        raise ValueError("nothing to save")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
