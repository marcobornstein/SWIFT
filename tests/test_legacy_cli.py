"""Pre-2.0 command lines must still run.

The command lines below are taken verbatim from the 1.0 Scripts/ directory (and
its test-script.sh), one per experiment and algorithm. They are the reason the
translation layer exists, so they are the thing worth testing: two 1.0 flags
gate other flags rather than carrying a value, and a plain rename table misses
both.
"""

from __future__ import annotations

import pytest

from swift.cli import parse_config

# Verbatim from the 1.0 scripts, minus the leading "python Train.py".
LEGACY_COMMANDS = [
    # test-script.sh
    "--name swift-test --graph ring --customLR 1 --sgd_steps 1 --weight_type swift"
    " --momentum 0.9 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 10"
    " --wb 1 --description SWIFT-test --randomSeed 3782 --datasetRoot Data"
    " --outputFolder Output",
    # 10-ROC-NonIID-0.9-Scripts/dsgd-noniid.sh
    "--graph clique-ring --num_clusters 3 --name dsgd-noniid-0.9-test1-10W"
    " --comm_style d-sgd --momentum 0.9 --lr 0.8 --degree_noniid 0.9 --noniid 1"
    " --resSize 18 --bs 32 --epoch 300 --description DSGD-paper --randomSeed 1000"
    " --datasetRoot ./data --outputFolder Output",
    # 10-ROC-NonIID-Scripts/ldsgd-noniid.sh
    "--graph clique-ring --num_clusters 3 --name ldsgd-noniid-test1-10W"
    " --comm_style ld-sgd --momentum 0.9 --lr 0.8 --degree_noniid 0.5 --noniid 1"
    " --i1 1 --i2 2 --resSize 18 --bs 32 --epoch 300 --description LDSGD-paper"
    " --randomSeed 115 --datasetRoot ./data --outputFolder Output",
    # 16-Ring-IID-Scripts/pdsgdiid.sh
    "--graph ring --num_clusters 3 --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1"
    " --customLR 1 --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200"
    " --description PDSGD --randomSeed 2828 --datasetRoot ./data --outputFolder Output",
    # 16-Ring-IID-Scripts/swift-iid1nomem.sh
    "--graph ring --sgd_steps 1 --customLR 1 --weight_type swift --momentum 0.9"
    " --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 200"
    " --memory_efficient 1 --wb 0 --description SWIFT --randomSeed 1333"
    " --datasetRoot ./data --outputFolder Output",
    # 16-ROC-IID-Scripts/dsgdiid-test.sh
    "--graph clique-ring --num_clusters 4 --comm_style d-sgd --lr 0.1 --momentum 0.9"
    " --degree_noniid 0 --noniid 0 --resSize 50 --bs 64 --epoch 200"
    " --description DSGD --randomSeed 1000 --datasetRoot ./data --outputFolder Output",
    # Slowdown-16-Ring/swift-slow.sh
    "--graph ring --slowdown 4 --sgd_steps 1 --weight_type swift --momentum 0.9"
    " --degree_noniid 0 --noniid 0 --resSize 18 --bs 32 --epoch 100 --wb 1"
    " --description SWIFT --randomSeed 1333 --datasetRoot ./data --outputFolder Output",
]


@pytest.mark.parametrize("command", LEGACY_COMMANDS)
def test_legacy_command_lines_still_parse(command: str) -> None:
    config, _ = parse_config(command.split())
    assert config.epochs > 0
    assert config.resnet_depth in {18, 50}


def test_noniid_zero_overrides_the_degree() -> None:
    """1.0 gated the degree on --noniid; at 0 the partition was IID."""
    config, _ = parse_config(["--noniid", "0", "--degree_noniid", "0.7"])
    assert config.non_iid == 0.0

    config, _ = parse_config(["--noniid", "1", "--degree_noniid", "0.7"])
    assert config.non_iid == pytest.approx(0.7)


def test_custom_lr_selects_the_schedule() -> None:
    """Table 8: customLR 1 is the multistep row, 0 the every-N-epochs row."""
    config, _ = parse_config(["--customLR", "1", "--noniid", "0"])
    assert config.lr_schedule == "multistep"
    assert config.lr_gamma == pytest.approx(0.1)
    assert config.lr_milestones == [81, 122]

    config, _ = parse_config(["--customLR", "0", "--noniid", "0"])
    assert config.lr_schedule == "step"
    assert config.lr_gamma == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("command", "schedule", "start"),
    [
        # Train.py: decay began at 200 for non-IID runs and 100 otherwise.
        ("--comm_style d-sgd --noniid 1 --degree_noniid 0.9", "step", 200),
        ("--comm_style d-sgd --noniid 0 --degree_noniid 0", "step", 100),
        # TrainSlowdown.py was identical but started at 50; --slowdown is the
        # only thing distinguishing the two scripts on a command line.
        ("--weight_type swift --noniid 0 --degree_noniid 0 --slowdown 4", "step", 50),
        # A 1.0 command line that omitted --noniid got its default of 1.
        ("--comm_style d-sgd --degree_noniid 0.5", "step", 200),
    ],
)
def test_schedule_is_inferred_when_custom_lr_is_absent(
    command: str, schedule: str, start: int
) -> None:
    """1.0 defaulted --customLR to 0, so an omitted flag means the step row.

    Getting this wrong is silent: the run trains happily on the wrong schedule.
    """
    config, _ = parse_config(command.split())
    assert config.lr_schedule == schedule
    assert config.lr_decay_start == start
    assert config.lr_decay_every == 10
    assert config.lr_gamma == pytest.approx(0.5)


def test_new_schedule_flags_win_over_inference() -> None:
    config, _ = parse_config(
        ["--comm_style", "d-sgd", "--noniid", "1",
         "--lr-schedule", "step", "--lr-decay-start", "77"]
    )
    assert config.lr_decay_start == 77


def test_schedule_is_untouched_without_legacy_flags() -> None:
    """A purely modern command line keeps the dataclass defaults."""
    config, _ = parse_config(["--algorithm", "d-sgd", "--epochs", "5"])
    assert config.lr_schedule == "multistep"


def test_new_flags_win_over_legacy_gates() -> None:
    config, _ = parse_config(["--customLR", "1", "--lr-schedule", "constant"])
    assert config.lr_schedule == "constant"

    config, _ = parse_config(["--noniid", "0", "--non-iid", "0.4"])
    assert config.non_iid == pytest.approx(0.4)


@pytest.mark.parametrize(
    "removed",
    ["--max_sgd 10", "--personalize 0", "--unordered_epochs 1",
     "--model res", "--savePath /tmp/x", "--warmup", "--p", "-p"],
)
def test_removed_flags_are_consumed_not_fatal(removed: str) -> None:
    config, _ = parse_config([*removed.split(), "--epochs", "3"])
    assert config.epochs == 3


def test_store_true_flags_keep_their_meaning() -> None:
    """1.0's bare --nesterov meant on; dropping it would turn it off."""
    config, _ = parse_config(["--nesterov", "--epochs", "3"])
    assert config.nesterov is True
    assert config.epochs == 3

    config, _ = parse_config(["--nesterov", "0"])
    assert config.nesterov is False


def test_short_forms_are_translated() -> None:
    config, _ = parse_config(["-n", "myrun", "-e", "7"])
    assert config.name == "myrun"
    assert config.epochs == 7


def test_algorithm_and_weight_values_are_renamed() -> None:
    config, _ = parse_config(
        ["--comm_style", "pd-sgd", "--weight_type", "uniform", "--i1", "1"]
    )
    assert config.algorithm == "pa-sgd"
    assert config.weights == "neighborhood-uniform"

    config, _ = parse_config(["--weight_type", "uniform-symmetric",
                              "--comm_style", "d-sgd"])
    assert config.weights == "global-uniform"
