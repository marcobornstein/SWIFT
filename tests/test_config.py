"""Config defaults, validation and the learning-rate schedules from Table 8."""

from __future__ import annotations

import pytest

from swift.cli import parse_config
from swift.config import Config


def test_multistep_matches_the_baseline_row() -> None:
    """Table 8 row 1: 0.1, dropping by 1/10 after epochs 81 and 122."""
    config = Config(lr=0.1, lr_schedule="multistep", lr_gamma=0.1,
                    lr_milestones=[81, 122])
    assert config.lr_at(0) == pytest.approx(0.1)
    assert config.lr_at(81) == pytest.approx(0.1)
    assert config.lr_at(82) == pytest.approx(0.01)
    assert config.lr_at(122) == pytest.approx(0.01)
    assert config.lr_at(123) == pytest.approx(0.001)


def test_step_matches_the_non_iid_row() -> None:
    """Table 8 row 2: 0.8, halving every 10 epochs from epoch 200."""
    config = Config(lr=0.8, lr_schedule="step", lr_gamma=0.5,
                    lr_decay_start=200, lr_decay_every=10)
    assert config.lr_at(199) == pytest.approx(0.8)
    assert config.lr_at(209) == pytest.approx(0.8)
    assert config.lr_at(210) == pytest.approx(0.4)
    assert config.lr_at(220) == pytest.approx(0.2)


def test_constant_schedule_never_decays() -> None:
    config = Config(lr=0.5, lr_schedule="constant")
    assert config.lr_at(0) == config.lr_at(999) == pytest.approx(0.5)


def test_weights_default_per_algorithm() -> None:
    assert Config(algorithm="swift").weights == "ccs"
    assert Config(algorithm="d-sgd").weights == "neighborhood-uniform"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"non_iid": 1.5}, "non_iid"),
        ({"local_steps": 0}, "local_steps"),
        ({"i2": 0}, "i2"),
        ({"slowdown": 0.5}, "slowdown"),
        ({"algorithm": "swift", "weights": "global-uniform"}, "SWIFT requires"),
        ({"algorithm": "pa-sgd", "i1": 0}, "i1 >= 1"),
        ({"algorithm": "ld-sgd", "i1": 0}, "i1 >= 1"),
        ({"algorithm": "ld-sgd", "i1": 1, "i2": 1}, "i2 >= 2"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"eval_subset": -1}, "eval_subset"),
        ({"clusters": 0}, "clusters"),
        ({"lr_decay_every": 0}, "lr_decay_every"),
        ({"resnet_depth": 19}, "resnet_depth"),
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


def test_yaml_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("epochs: 3\nnot_a_real_key: 7\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_yaml(path)


def test_resnet_depths_agree_with_the_model() -> None:
    """config.py duplicates the list to keep torch off the --help path."""
    from swift.config import RESNET_DEPTHS
    from swift.model import _CONFIGS

    assert set(RESNET_DEPTHS) == set(_CONFIGS)


def test_shipped_configs_load() -> None:
    from pathlib import Path

    configs = sorted(Path(__file__).parent.parent.glob("configs/*.yaml"))
    assert configs, "no configs found"
    for path in configs:
        assert Config.from_yaml(path).epochs > 0


def test_cli_overrides_beat_the_config_file(tmp_path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("epochs: 300\nlr: 0.8\n")
    config, _ = parse_config(["--config", str(path), "--epochs", "5"])
    assert (config.epochs, config.lr) == (5, 0.8)


def test_algorithm_override_resolves_weights_against_the_override(tmp_path) -> None:
    """--algorithm on top of a swift config must not inherit CCS weights.

    scripts/run_experiment.sh runs every baseline this way, and CCS weights
    would silently change what the baselines compute.
    """
    path = tmp_path / "run.yaml"
    path.write_text("algorithm: swift\ntopology: ring\n")
    for algorithm in ("d-sgd", "pa-sgd", "ld-sgd"):
        config, _ = parse_config(
            ["--config", str(path), "--algorithm", algorithm, "--i1", "1", "--i2", "2"]
        )
        assert config.weights == "neighborhood-uniform", algorithm

    config, _ = parse_config(["--config", str(path)])
    assert config.weights == "ccs"


def test_explicit_weights_survive_an_algorithm_override(tmp_path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("algorithm: d-sgd\nweights: global-uniform\n")
    config, _ = parse_config(
        ["--config", str(path), "--algorithm", "pa-sgd", "--i1", "1"]
    )
    assert config.weights == "global-uniform"


def test_legacy_flags_still_work() -> None:
    config, _ = parse_config(
        ["--comm_style", "pd-sgd", "--resSize", "50", "--bs", "64", "--i1", "1",
         "--sgd_steps", "2", "--degree_noniid", "0.7", "--randomSeed", "1337"]
    )
    assert config.algorithm == "pa-sgd"
    assert config.resnet_depth == 50
    assert config.batch_size == 64
    assert config.local_steps == 2
    assert config.non_iid == pytest.approx(0.7)
    assert config.seed == 1337
