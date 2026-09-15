"""Per-client metric recording.

One CSV per rank plus a JSON summary, written under ``config.run_dir``. The CSV
columns are what the paper's figures are drawn from: loss against wall-clock
time, with communication time broken out.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config


@dataclass
class EpochRecord:
    epoch: int
    lr: float
    train_loss: float
    train_acc: float
    test_loss: float
    comp_time: float
    """Seconds of local computation, excluding communication and metric bookkeeping."""
    comm_time: float
    """Seconds inside the communicator."""
    sync_time: float
    """Seconds waiting for the slowest client before a synchronous exchange.

    Zero for SWIFT, which never waits. Excluded from ``epoch_time`` so that
    figure stays comparable with the paper's tables, which account for it the
    same way -- but it is real time, and ``wall_time`` contains it.
    """
    epoch_time: float
    """comp_time + comm_time: the algorithm's own cost for this epoch."""
    wall_time: float
    """Elapsed wall-clock for the epoch, excluding evaluation."""


class Recorder:
    """Collects per-epoch metrics for one client and writes them out."""

    def __init__(self, config: Config, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.records: list[EpochRecord] = []
        self.run_dir = config.run_dir

    def add(self, record: EpochRecord) -> None:
        self.records.append(record)

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / f"metrics-rank{self.rank}.csv"

    def save(self, summary: dict[str, object] | None = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

        with self.metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[f for f in EpochRecord.__annotations__]
            )
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

        payload: dict[str, object] = {"rank": self.rank, "epochs": len(self.records)}
        if self.records:
            payload |= {
                "total_comp_time": sum(r.comp_time for r in self.records),
                "total_comm_time": sum(r.comm_time for r in self.records),
                "total_sync_time": sum(r.sync_time for r in self.records),
                "mean_sync_time": sum(r.sync_time for r in self.records)
                / len(self.records),
                "total_wall_time": sum(r.wall_time for r in self.records),
                "mean_epoch_time": sum(r.epoch_time for r in self.records)
                / len(self.records),
                "mean_comm_time": sum(r.comm_time for r in self.records)
                / len(self.records),
                "final_train_loss": self.records[-1].train_loss,
                "final_test_loss": self.records[-1].test_loss,
            }
        if summary:
            payload |= summary

        path = self.run_dir / f"summary-rank{self.rank}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
