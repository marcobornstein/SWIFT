#!/usr/bin/env python3
"""Plot loss against wall-clock time, the way the paper's figures are drawn.

Reads the run directories produced by ``swift-train`` and expects the layout
``scripts/run_experiment.sh`` creates::

    outputs/<experiment>/<algorithm>/seed<N>/metrics-rank<K>.csv

Each curve averages over clients, then shows a bootstrapped confidence band
across seeds.

    python tools/plot.py outputs/baseline-ring --metric test_loss
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Display names and a stable colour per algorithm, so a figure drawn from a
# subset of the runs still matches one drawn from all of them.
ALGORITHMS = {
    "d-sgd": ("D-SGD", "tab:red"),
    "ld-sgd": ("LD-SGD", "tab:blue"),
    "pa-sgd": ("PA-SGD", "tab:green"),
    "swift": ("SWIFT", "tab:purple"),
    "swift-2sgd": ("SWIFT (2-SGD)", "tab:orange"),
}
METRIC_LABELS = {
    "train_loss": "Training Loss",
    "test_loss": "Test Loss",
    "train_acc": "Training Accuracy (%)",
    "comm_time": "Communication Time (s)",
    "epoch_time": "Epoch Time (s)",
}


def read_run(run_dir: Path) -> dict[str, np.ndarray]:
    """Stack one run's per-client CSVs into ``{column: (epochs, clients)}``."""
    per_client: list[dict[str, list[float]]] = []
    for path in sorted(run_dir.glob("metrics-rank*.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            per_client.append(
                {key: [float(row[key]) for row in rows] for key in rows[0]}
            )
    if not per_client:
        raise FileNotFoundError(f"no metrics-rank*.csv under {run_dir}")

    epochs = min(len(client["epoch"]) for client in per_client)
    return {
        key: np.stack([client[key][:epochs] for client in per_client], axis=1)
        for key in per_client[0]
    }


def bootstrap_interval(
    curves: np.ndarray, samples: int = 1000, low: float = 1.0, high: float = 99.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and percentile band over seeds, resampled at each epoch."""
    rng = rng or np.random.default_rng(0)
    num_seeds = curves.shape[0]
    draws = rng.integers(0, num_seeds, size=(samples, num_seeds))
    resampled = curves[draws].mean(axis=1)  # (samples, epochs)
    return (
        curves.mean(axis=0),
        np.percentile(resampled, low, axis=0),
        np.percentile(resampled, high, axis=0),
    )


def collect(
    algorithm_dir: Path, metric: str, time_from: str = "epoch_time"
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-seed curves of ``metric`` and of cumulative minutes.

    ``time_from`` picks what the x-axis accumulates. The paper's loss-versus-time
    figures use epoch time (computation plus communication, the algorithm's own
    cost); the straggler figure uses wall time, which also contains the delay.
    """
    values, times = [], []
    for seed_dir in sorted(p for p in algorithm_dir.iterdir() if p.is_dir()):
        try:
            data = read_run(seed_dir)
        except FileNotFoundError:
            continue
        # Average over clients, then accumulate time the way the paper does.
        values.append(data[metric].mean(axis=1))
        times.append(np.cumsum(data[time_from].mean(axis=1) / 60.0))
    if not values:
        return None

    epochs = min(len(v) for v in values)
    return (
        np.stack([v[:epochs] for v in values]),
        np.stack([t[:epochs] for t in times]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path,
                        help="directory holding one sub-directory per algorithm")
    parser.add_argument("--metric", default="test_loss", choices=sorted(METRIC_LABELS))
    parser.add_argument("--x", default="time", choices=["time", "epoch"],
                        help="plot against elapsed minutes or epoch index")
    parser.add_argument("--time-from", default="epoch_time",
                        choices=["epoch_time", "wall_time"],
                        help="epoch_time is the algorithm's own cost (the paper's "
                             "loss-vs-time figures); wall_time also includes any "
                             "straggler delay (the heterogeneity figure)")
    parser.add_argument("--output", type=Path, help="save here instead of showing")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    if not args.experiment.is_dir():
        parser.error(f"{args.experiment} is not a directory")

    figure, axes = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for key, (label, color) in ALGORITHMS.items():
        algorithm_dir = args.experiment / key
        if not algorithm_dir.is_dir():
            continue
        result = collect(algorithm_dir, args.metric, args.time_from)
        if result is None:
            continue
        values, times = result

        mean, low, high = bootstrap_interval(values)
        x = times.mean(axis=0) if args.x == "time" else np.arange(values.shape[1])
        axes.plot(x, mean, label=label, color=color, linewidth=2)
        axes.fill_between(x, low, high, color=color, alpha=0.2, linewidth=0)
        plotted += 1

    if not plotted:
        parser.error(f"no algorithm runs found under {args.experiment}")

    label = "Wall-clock" if args.time_from == "wall_time" else "Epoch"
    axes.set_xlabel(f"{label} Time (minutes)" if args.x == "time" else "Epoch")
    axes.set_ylabel(METRIC_LABELS[args.metric])
    axes.set_title(args.title or args.experiment.name)
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()

    if args.output:
        figure.savefig(args.output, dpi=200)
        print(f"wrote {args.output}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
