"""scripts/run_experiment.sh, driven against stub executables.

A sweep is long enough that it will be interrupted, and a shell script is easy
to break silently, so the behaviours that matter -- resume, and not abandoning
the rest of the sweep when one run fails -- are exercised here without running
any training.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SCRIPT = REPO / "scripts" / "run_experiment.sh"

MPIRUN_STUB = """#!/usr/bin/env bash
shift 2          # drop "-np N"
exec "$@"
"""

# Writes the summary file the script looks for when deciding to skip a run,
# and fails for whichever algorithm FAIL_ON names.
TRAIN_STUB = """#!/usr/bin/env bash
name=""; out=""; algorithm=""; i1=""; i2=""; steps=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) name=$2; shift 2;;
    --output-dir) out=$2; shift 2;;
    --algorithm) algorithm=$2; shift 2;;
    --i1) i1=$2; shift 2;;
    --i2) i2=$2; shift 2;;
    --local-steps) steps=$2; shift 2;;
    *) shift;;
  esac
done
if [ -n "${FAIL_ON:-}" ]; then
  case "$name" in *"$FAIL_ON"*) exit 3;; esac
fi
mkdir -p "$out/$name"
# Record what was actually passed, so the test can check the flags reached the
# command rather than only naming the directory.
printf '{"algorithm": "%s", "i1": "%s", "i2": "%s", "local_steps": "%s"}\n' \
  "$algorithm" "$i1" "$i2" "$steps" > "$out/$name/summary-rank0.json"
"""


@pytest.fixture
def sweep(tmp_path: Path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in (("mpirun", MPIRUN_STUB), ("swift-train", TRAIN_STUB)):
        path = binaries / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def run(seeds: str = "1 2", fail_on: str = "") -> subprocess.CompletedProcess:
        env = os.environ | {
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "SEEDS": seeds,
            "OUTPUT_DIR": str(tmp_path / "out"),
            "FAIL_ON": fail_on,
        }
        return subprocess.run(
            ["bash", str(SCRIPT), str(REPO / "configs" / "baseline-ring.yaml"), "4"],
            capture_output=True, text=True, env=env, timeout=120,
        )

    return run, tmp_path / "out"


def _completed(out: Path) -> int:
    return len(list(out.rglob("summary-rank0.json")))


def test_a_sweep_runs_every_algorithm_and_seed(sweep) -> None:
    run, out = sweep
    result = run(seeds="1 2")
    assert result.returncode == 0, result.stderr
    assert _completed(out) == 10, "5 algorithms x 2 seeds"


def test_finished_runs_are_skipped_on_a_second_pass(sweep) -> None:
    """An interrupted sweep has to resume rather than start over."""
    run, _ = sweep
    run(seeds="1 2")
    result = run(seeds="1 2")
    assert result.returncode == 0
    assert "skipped 10 run(s)" in result.stdout


def test_one_failure_does_not_abandon_the_sweep(sweep) -> None:
    run, out = sweep
    result = run(seeds="1 2", fail_on="pa-sgd")

    assert result.returncode == 1, "a failed run must be reported in the exit code"
    assert _completed(out) == 8, "the other four algorithms should still have run"
    assert "pa-sgd" in result.stderr


def test_each_run_is_launched_with_its_own_algorithm(sweep) -> None:
    """A loop variable that names the output directory but never reaches the
    command would run the same algorithm five times under five names."""
    import json

    run, out = sweep
    assert run(seeds="1").returncode == 0

    passed = {
        path.parent.parent.name: json.loads(path.read_text())
        for path in out.rglob("summary-rank0.json")
    }
    assert passed["swift"]["algorithm"] == "swift"
    assert passed["swift"]["local_steps"] == "1"
    assert passed["swift-2sgd"]["algorithm"] == "swift"
    assert passed["swift-2sgd"]["local_steps"] == "2"
    assert passed["d-sgd"]["algorithm"] == "d-sgd"
    # PA-SGD and LD-SGD collapse into D-SGD without these.
    assert passed["pa-sgd"]["algorithm"] == "pa-sgd"
    assert passed["pa-sgd"]["i1"] == "1"
    assert passed["ld-sgd"]["algorithm"] == "ld-sgd"
    assert (passed["ld-sgd"]["i1"], passed["ld-sgd"]["i2"]) == ("1", "2")


def test_a_missing_config_fails_immediately(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path / "absent.yaml"), "4"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1
    assert "no such config" in result.stderr
