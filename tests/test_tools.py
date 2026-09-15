"""The reporting tools, exercised on synthetic runs.

They are the last step of a reproduction, so a crash there costs whatever the
run cost. Both are driven here against the schema Recorder actually writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import summarize


def _experiment(root: Path) -> Path:
    """Two seeds x four clients for three algorithms."""
    experiment = root / "exp"
    for algorithm, epoch, comm, accuracy in (
        ("swift", 1.0, 0.08, 91.0),
        ("d-sgd", 1.5, 0.62, 90.0),
        ("ld-sgd", 1.3, 0.42, 90.5),
    ):
        for seed in (1, 2):
            run = experiment / algorithm / f"seed{seed}"
            run.mkdir(parents=True)
            for rank in range(4):
                (run / f"summary-rank{rank}.json").write_text(
                    json.dumps({
                        "rank": rank,
                        "mean_epoch_time": epoch,
                        "mean_comm_time": comm,
                        "mean_sync_time": 0.0 if algorithm == "swift" else 0.4,
                        "total_wall_time": epoch * 10,
                        "epochs": 10,
                        "final_train_loss": 0.5,
                        "final_test_loss": 0.6,
                        "consensus_accuracy": accuracy,
                    })
                )
    return experiment


def test_collect_averages_over_clients_and_seeds(tmp_path: Path) -> None:
    rows = summarize.collect(_experiment(tmp_path))
    assert set(rows) == {"swift", "d-sgd", "ld-sgd"}
    assert rows["swift"]["epoch_time"] == pytest.approx(1.0)
    assert rows["swift"]["comm_time"] == pytest.approx(0.08)
    assert rows["swift"]["clients"] == 4
    assert rows["swift"]["seeds"] == 2
    # SWIFT never waits for a neighbour; the synchronous baselines do.
    assert rows["swift"]["sync_time"] == pytest.approx(0.0)
    assert rows["d-sgd"]["sync_time"] > 0
    assert rows["swift"]["wall_per_epoch"] == pytest.approx(1.0)


def test_percentage_change_matches_the_paper_convention() -> None:
    """Table 3 reports the change against D-SGD, negative being an improvement."""
    assert summarize._change(1.0, 1.5) == "-33.33%"
    assert summarize._change(1.5, 1.5) == "+0.00%"
    assert summarize._change(1.0, 0.0) == "-"


def test_large_changes_are_shown_as_a_ratio() -> None:
    """Communication times differ by orders of magnitude between transports,
    where a percentage stops conveying anything."""
    assert summarize._change(42.3, 0.054) == "783x"
    assert summarize._change(0.054, 42.3) == "0.00128x"
    # Ordinary differences stay as percentages.
    assert summarize._change(1.2, 1.0) == "+20.00%"


def test_summarize_runs_end_to_end(tmp_path: Path, capsys, monkeypatch) -> None:
    experiment = _experiment(tmp_path)
    out = tmp_path / "table.csv"
    monkeypatch.setattr(sys, "argv", ["summarize", str(experiment), "--csv", str(out)])
    assert summarize.main() == 0

    printed = capsys.readouterr().out
    assert "SWIFT" in printed and "D-SGD" in printed
    assert "-33.33%" in printed
    assert out.exists()
    assert "swift" in out.read_text()


def test_missing_experiment_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["summarize", str(tmp_path / "nope")])
    with pytest.raises(SystemExit):
        summarize.main()


def test_empty_experiment_is_rejected(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(sys, "argv", ["summarize", str(tmp_path / "empty")])
    with pytest.raises(SystemExit):
        summarize.main()
