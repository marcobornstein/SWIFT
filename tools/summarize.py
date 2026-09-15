#!/usr/bin/env python3
"""Summarise an experiment as the table the paper reports.

Prints mean epoch time, mean communication time, final losses and consensus
accuracy per algorithm, averaged over clients and seeds, with the percentage
change against a baseline the way Table 3 does.

    python tools/summarize.py outputs/baseline-ring
    python tools/summarize.py outputs/baseline-ring --baseline d-sgd --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

# Display order and names, matching tools/plot.py.
ALGORITHMS = {
    "swift": "SWIFT",
    "swift-2sgd": "SWIFT (2-SGD)",
    "d-sgd": "D-SGD",
    "pa-sgd": "PA-SGD",
    "ld-sgd": "LD-SGD",
}


def read_seed(seed_dir: Path) -> dict[str, float] | None:
    """Average one run's per-client summaries into a single row."""
    summaries = []
    for path in sorted(seed_dir.glob("summary-rank*.json")):
        summaries.append(json.loads(path.read_text()))
    if not summaries:
        return None

    def mean(key: str) -> float:
        values = [s[key] for s in summaries if key in s]
        return statistics.fmean(values) if values else float("nan")

    return {
        "epoch_time": mean("mean_epoch_time"),
        "comm_time": mean("mean_comm_time"),
        "sync_time": mean("mean_sync_time"),
        "total_time": mean("total_wall_time"),
        "wall_per_epoch": mean("total_wall_time") / max(mean("epochs"), 1),
        "train_loss": mean("final_train_loss"),
        "test_loss": mean("final_test_loss"),
        "consensus_accuracy": mean("consensus_accuracy"),
        "clients": len(summaries),
    }


def collect(experiment: Path) -> dict[str, dict[str, float]]:
    """One row per algorithm, averaged over whatever seeds are present."""
    rows: dict[str, dict[str, float]] = {}
    for key in ALGORITHMS:
        algorithm_dir = experiment / key
        if not algorithm_dir.is_dir():
            continue
        seeds = [
            row
            for path in sorted(p for p in algorithm_dir.iterdir() if p.is_dir())
            if (row := read_seed(path)) is not None
        ]
        if not seeds:
            continue
        rows[key] = {
            field: statistics.fmean(seed[field] for seed in seeds)
            for field in seeds[0]
        }
        rows[key]["seeds"] = len(seeds)
    return rows


def _change(value: float, reference: float) -> str:
    """Percentage change, or a ratio once a percentage stops being readable.

    Communication times differ by orders of magnitude between shared memory and
    a real network, and "+78047%" carries less than "783x".
    """
    if not reference:
        return "-"
    ratio = value / reference
    if ratio > 10 or (ratio < 0.1 and ratio > 0):
        return f"{ratio:.3g}x"
    return f"{100.0 * (value - reference) / reference:+.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", type=Path,
                        help="directory holding one sub-directory per algorithm")
    parser.add_argument("--baseline", default="d-sgd",
                        help="algorithm the percentage changes are relative to")
    parser.add_argument("--csv", type=Path, help="also write the table here")
    args = parser.parse_args()

    if not args.experiment.is_dir():
        parser.error(f"{args.experiment} is not a directory")
    rows = collect(args.experiment)
    if not rows:
        parser.error(f"no algorithm runs found under {args.experiment}")

    reference = rows.get(args.baseline)
    clients = int(next(iter(rows.values()))["clients"])
    seeds = int(next(iter(rows.values()))["seeds"])
    print(f"\n{args.experiment.name}: {clients} clients, {seeds} seed(s) per algorithm")
    header = (
        f"{'Algorithm':<16}{'Epoch (s)':>11}{'change':>10}"
        f"{'Comm (s)':>11}{'change':>10}{'Sync (s)':>10}"
        f"{'Wall (s)':>10}{'Test loss':>11}{'Consensus %':>13}"
    )
    print(header)
    print("-" * len(header))

    for key, row in rows.items():
        if reference:
            epoch_change = _change(row["epoch_time"], reference["epoch_time"])
            comm_change = _change(row["comm_time"], reference["comm_time"])
        else:
            epoch_change = comm_change = "-"
        print(
            f"{ALGORITHMS[key]:<16}{row['epoch_time']:>11.3f}{epoch_change:>10}"
            f"{row['comm_time']:>11.3f}{comm_change:>10}"
            f"{row['sync_time']:>10.3f}{row['wall_per_epoch']:>10.3f}"
            f"{row['test_loss']:>11.4f}{row['consensus_accuracy']:>13.2f}"
        )
    print(f"\nchange is relative to {ALGORITHMS.get(args.baseline, args.baseline)}.")
    print(
        "epoch = comp + comm, which excludes the wait for the slowest client "
        "(shown as sync);\nwall is the honest per-epoch comparison."
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["algorithm", *sorted(next(iter(rows.values())))])
            for key, row in rows.items():
                writer.writerow([key, *(row[f] for f in sorted(row))])
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
