"""The README is the entry point for anyone reproducing the paper, so the
claims it makes about the code are checked here rather than by hand."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from swift.config import Config
from swift.recorder import EpochRecord

REPO = Path(__file__).parent.parent
README = (REPO / "README.md").read_text()


def test_defaults_reproduce_the_baseline_config() -> None:
    """The README says a bare `swift-train` runs the baseline experiment."""
    defaults = Config()
    baseline = Config.from_yaml(REPO / "configs" / "baseline-ring.yaml")
    differing = {
        field
        for field in Config.field_names()
        if field not in {"name", "notes"}
        and getattr(defaults, field) != getattr(baseline, field)
    }
    assert not differing, f"defaults drifted from baseline-ring.yaml: {differing}"


@pytest.mark.parametrize("field", ["algorithm", "topology"])
def test_every_choice_is_documented(field: str) -> None:
    for value in get_args(get_type_hints(Config)[field]):
        assert f"`{value}`" in README, f"{field} {value!r} is undocumented"


def test_every_metrics_column_is_documented() -> None:
    for column in EpochRecord.__annotations__:
        assert column in README, f"CSV column {column!r} is undocumented"


def test_referenced_files_exist() -> None:
    patterns = [
        r"\(([a-zA-Z0-9_./-]+\.(?:py|yaml|md|sbatch|sh))\)",       # markdown links
        r"`(tools/[a-z_]+\.py|scripts/[a-z_.]+|configs/[a-z-]+\.yaml|tests/[a-z_]+\.py)`",
    ]
    referenced = {m for pattern in patterns for m in re.findall(pattern, README)}
    assert referenced, "no file references found; did the README change shape?"
    missing = sorted(path for path in referenced if not (REPO / path).exists())
    assert not missing, f"README references files that do not exist: {missing}"


def test_every_config_is_referenced() -> None:
    for path in sorted((REPO / "configs").glob("*.yaml")):
        assert path.name in README, f"{path.name} is not mentioned in the README"


def test_shipped_configs_use_paper_hyperparameters() -> None:
    """Table 8: momentum 0.9 and weight decay 1e-4 throughout."""
    for path in sorted((REPO / "configs").glob("*.yaml")):
        config = Config.from_yaml(path)
        assert config.momentum == pytest.approx(0.9), path.name
        assert config.weight_decay == pytest.approx(1e-4), path.name
        assert config.resnet_depth in {18, 50}, path.name
