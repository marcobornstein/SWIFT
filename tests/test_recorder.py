"""Recorder output: the CSV schema the plotting tool reads back."""

from __future__ import annotations

import csv
import json

import pytest

from swift.config import Config
from swift.recorder import EpochRecord, Recorder

EXPECTED_COLUMNS = [
    "epoch", "lr", "train_loss", "train_acc", "test_loss",
    "comp_time", "comm_time", "sync_time", "epoch_time", "wall_time",
]


def _record(epoch: int) -> EpochRecord:
    return EpochRecord(
        epoch=epoch, lr=0.1, train_loss=2.0 - epoch, train_acc=10.0 * epoch,
        test_loss=2.5 - epoch, comp_time=1.0, comm_time=0.25, sync_time=0.5,
        epoch_time=1.25, wall_time=1.5,
    )


def test_metrics_round_trip(tmp_path) -> None:
    config = Config(name="run", output_dir=tmp_path, epochs=2)
    recorder = Recorder(config, rank=3)
    for epoch in range(2):
        recorder.add(_record(epoch))
    recorder.save({"consensus_accuracy": 42.0})

    with recorder.metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == EXPECTED_COLUMNS
    assert [int(row["epoch"]) for row in rows] == [0, 1]
    assert float(rows[1]["train_loss"]) == 1.0


def test_summary_reports_totals(tmp_path) -> None:
    config = Config(name="run", output_dir=tmp_path)
    recorder = Recorder(config, rank=0)
    for epoch in range(4):
        recorder.add(_record(epoch))
    recorder.save({"consensus_accuracy": 42.0})

    summary = json.loads((config.run_dir / "summary-rank0.json").read_text())
    assert summary["epochs"] == 4
    assert summary["total_comm_time"] == 1.0
    assert summary["total_sync_time"] == 2.0
    assert summary["mean_epoch_time"] == 1.25
    assert summary["consensus_accuracy"] == 42.0


def test_nested_run_names_create_directories(tmp_path) -> None:
    """run_experiment.sh names runs <experiment>/<algorithm>/seed<N>."""
    config = Config(name="baseline/swift/seed7", output_dir=tmp_path)
    recorder = Recorder(config, rank=0)
    recorder.add(_record(0))
    recorder.save()
    assert (tmp_path / "baseline" / "swift" / "seed7" / "metrics-rank0.csv").exists()


def test_plot_tool_reads_the_recorder_output(tmp_path) -> None:
    import sys
    sys.path.insert(0, str(tmp_path.parent))
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    import plot

    config = Config(name="exp/swift/seed1", output_dir=tmp_path)
    for rank in range(3):
        recorder = Recorder(config, rank=rank)
        for epoch in range(5):
            recorder.add(_record(epoch))
        recorder.save()

    data = plot.read_run(config.run_dir)
    assert data["test_loss"].shape == (5, 3)

    collected = plot.collect(tmp_path / "exp" / "swift", "test_loss")
    assert collected is not None
    values, times = collected
    assert values.shape == (1, 5)
    # epoch_time is 1.25s per epoch by default, accumulated into minutes.
    assert times[0][-1] == pytest.approx(5 * 1.25 / 60)

    # wall_time (1.5s per epoch) is what the straggler figure accumulates.
    _, wall = plot.collect(tmp_path / "exp" / "swift", "test_loss", "wall_time")
    assert wall[0][-1] == pytest.approx(5 * 1.5 / 60)
