"""Table viewer demo with made-up data.

Run with::

    uv run --extra tables python -m xpm_mlboard.tables.demo [--port 8765] [DIR]

Writes fake experiment outputs (an ``eval`` table per run, growing ``train``
logs, an experiment-level ``summary``) and starts the viewer on them. While it
runs, the data keeps changing so that live updates can be checked: training
logs grow, a pending run produces its results, and a new column appears.
"""

import argparse
import json
import random
import tempfile
import threading
import time
from pathlib import Path

import polars as pl

from . import write_entry
from .server import main as serve

MODELS = ["alpha", "beta"]
LEARNING_RATES = [1e-3, 1e-4]
SEEDS = [1, 2]
DATASETS = ["dev", "test"]
METRICS = ["ndcg@10", "mrr"]


def run_name(lr: float, model: str, seed: int) -> str:
    return f"lr={lr}_model={model}_seed={seed}"


def write_eval(path: Path, rng: random.Random, timing: bool = False):
    rows = [
        {"dataset": dataset, "metric": metric, "value": rng.random()}
        for dataset in DATASETS
        for metric in METRICS
    ]
    if timing:
        for row in rows:
            row["seconds"] = rng.uniform(10, 100)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def train_line(step: int, lr: float, rng: random.Random) -> str:
    loss = 1 / (1 + step * lr * 10 * rng.uniform(0.8, 1.2))
    return json.dumps({"step": step, "loss": loss}) + "\n"


def register(root: Path, name: str, path: Path, run=None, tags=None):
    write_entry(
        root,
        {"name": name, "path": str(path), "format": None, "run": run, "tags": tags},
    )


def create(directory: Path, rng: random.Random):
    """Write the fake experiment and register its tables under
    ``directory/tables``; returns the paths that the live updates change."""
    root = directory / "tables"
    jobs = directory / "jobs"
    train_logs = []
    for model in MODELS:
        for lr in LEARNING_RATES:
            for seed in SEEDS:
                run = run_name(lr, model, seed)
                tags = {"lr": lr, "model": model, "seed": seed}
                write_eval(jobs / run / "eval.parquet", rng)
                train = jobs / run / "train.jsonl"
                with train.open("wt") as fp:
                    for step in range(0, 500, 50):
                        fp.write(train_line(step, lr, rng))
                train_logs.append((train, lr))
                register(root, "eval", jobs / run / "eval.parquet", run, tags)
                register(root, "train", train, run, tags)

    # A run whose results do not exist yet (shown as pending)
    pending_tags = {"lr": 1e-2, "model": "gamma", "seed": 1}
    pending = jobs / run_name(1e-2, "gamma", 1) / "eval.parquet"
    register(root, "eval", pending, run_name(1e-2, "gamma", 1), pending_tags)

    summary = directory / "summary.csv"
    summary.write_text("model,parameters\nalpha,110\nbeta,340\ngamma,780\n")
    register(root, "summary", summary)
    return root, train_logs, pending


def live_updates(train_logs, pending: Path, rng: random.Random, interval: float):
    """Grow the training logs; later, finish the pending run and add a
    ``seconds`` column to the results of one run (new column in ``eval``)."""
    step = 500
    for iteration in range(1, 1000):
        time.sleep(interval)
        for path, lr in train_logs:
            with path.open("at") as fp:
                fp.write(train_line(step, lr, rng))
        step += 50
        if iteration == 5:
            write_eval(pending, rng)
        if iteration == 10:
            first = train_logs[0][0].parent / "eval.parquet"
            write_eval(first, rng, timing=True)


def main():
    parser = argparse.ArgumentParser(description="xpm-mlboard table viewer demo")
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    directory = args.directory or Path(tempfile.mkdtemp(prefix="xpm-tables-demo-"))
    if directory.exists() and any(directory.iterdir()):
        parser.error(f"{directory} is not empty")
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Demo data in {directory}")

    rng = random.Random(args.seed)
    root, train_logs, pending = create(directory, rng)
    threading.Thread(
        target=live_updates,
        args=(train_logs, pending, rng, args.interval),
        daemon=True,
    ).start()
    serve(["--root", str(root), "--port", str(args.port)])


if __name__ == "__main__":
    main()
