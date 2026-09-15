"""Command line interface: ``swift-train``.

A run is described by a :class:`~swift.config.Config`. Values come from, in
increasing precedence: the dataclass defaults, a ``--config`` YAML file, and
explicit flags. Running without ``--config`` reproduces the paper's baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args, get_type_hints

from .config import Config

# Spellings used by the pre-2.0 scripts, kept so archived commands still run.
_LEGACY_ALIASES = {
    "--resSize": "--resnet-depth",
    "--comm_style": "--algorithm",
    "--degree_noniid": "--non-iid",
    "--sgd_steps": "--local-steps",
    "--bs": "--batch-size",
    "--randomSeed": "--seed",
    "--outputFolder": "--output-dir",
    "--datasetRoot": "--data-dir",
    "--graph": "--topology",
    "--num_clusters": "--clusters",
    "--memory_efficient": "--low-memory",
    "--wb": "--weight-boost",
    "--weight_type": "--weights",
    "--epoch": "--epochs",
    "--description": "--notes",
    "--downloadCifar": "--download",
    # 1.0 short forms.
    "-n": "--name",
    "-e": "--epochs",
}

# 1.0 store_true flags whose feature survives; they need a value in 2.0.
_LEGACY_STORE_TRUE = {"--nesterov": "1"}
_LEGACY_VALUES = {
    "--algorithm": {"pd-sgd": "pa-sgd"},
    "--weights": {"swift": "ccs", "uniform": "neighborhood-uniform",
                  "uniform-symmetric": "global-uniform"},
}

# Pre-2.0 flags that no longer map onto anything. Consumed with a warning so an
# archived command line still runs rather than dying on an unknown argument.
_LEGACY_DROPPED_WITH_VALUE = {
    "--max_sgd": "adaptive local steps were part of the personalization path",
    "--personalize": "the personalization path was never reachable",
    "--unordered_epochs": "only the removed ModelAvg.py consulted it",
    "--model": "only ResNet is implemented",
    "--savePath": "use --output-dir",
}
_LEGACY_DROPPED_FLAGS = {
    "--warmup": "warmup asserted on arguments its only caller never passed",
    "--p": "the dataset is always partitioned",
    "-p": "the dataset is always partitioned",
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _add_field_arguments(parser: argparse.ArgumentParser) -> None:
    """Derive one flag per Config field, so the two can never drift apart."""
    docs = Config.field_docs()
    defaults = Config()
    # Annotations are strings under `from __future__ import annotations`;
    # resolve them so Literal choices reach argparse.
    hints = get_type_hints(Config)
    for field in dataclasses.fields(Config):
        kind: Any = hints.get(field.name, field.type)
        default = getattr(defaults, field.name)
        help_text = docs.get(field.name, "")
        if default is not None:
            help_text = f"{help_text} (default: {default})".strip()
        options: dict[str, Any] = {"default": None, "dest": field.name}

        if kind is bool or kind == "bool":
            options["type"] = _parse_bool
            options["metavar"] = "{0,1}"
        elif field.name == "lr_milestones":
            options["type"] = int
            options["nargs"] = "+"
        elif field.name in {"data_dir", "output_dir"}:
            options["type"] = Path
        elif field.name in {"lr", "momentum", "weight_decay", "non_iid",
                            "slowdown", "edge_prob", "lr_gamma"}:
            options["type"] = float
        elif field.name in {"seed", "epochs", "batch_size", "resnet_depth",
                            "local_steps", "i1", "i2", "clusters", "eval_subset",
                            "num_workers", "lr_decay_start", "lr_decay_every",
                            "max_pending_sends"}:
            options["type"] = int
        else:
            options["type"] = str
            literal_values = [v for v in get_args(kind) if isinstance(v, str)]
            if literal_values:
                options["choices"] = literal_values

        parser.add_argument(_flag(field.name), help=help_text, **options)


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def _translate_legacy(
    argv: Sequence[str],
) -> tuple[list[str], dict[str, Any], list[str]]:
    """Rewrite pre-2.0 flags, reporting what changed so users can update.

    Returns the rewritten argv, overrides that cannot be expressed as a simple
    rename, and the warnings to print. Two 1.0 flags gate other flags rather
    than carrying a value of their own:

    ``--noniid 0|1``
        gated ``--degree_noniid``; at 0 the partition was IID whatever the
        degree said.
    ``--customLR 0|1``
        chose between the two schedules in Table 8 -- multistep at 1, and the
        every-N-epochs step decay at 0.
    """
    tokens = list(argv)
    # A new-style flag on the same command line always wins over a 1.0 gate, so
    # note which ones were given explicitly before rewriting anything.
    explicit = {token.partition("=")[0] for token in tokens if token.startswith("--")}
    translated: list[str] = []
    overrides: dict[str, Any] = {}
    warnings: list[str] = []
    index = 0
    legacy_seen = False
    # 1.0 argparse defaults, used when a command line omits the gate.
    noniid_gate = True
    custom_lr = False

    def take_value() -> str | None:
        """Consume the value belonging to tokens[index], if it has one."""
        nonlocal index
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            index += 1
            return tokens[index]
        return None

    while index < len(tokens):
        token = tokens[index]
        key, sep, inline = token.partition("=")
        value = inline if sep else None

        if key == "--noniid":
            legacy_seen = True
            if value is None:
                value = take_value()
            noniid_gate = value is None or _parse_bool(value)
            if value is not None and not _parse_bool(value):
                if "--non-iid" not in explicit:
                    overrides["non_iid"] = 0.0
                warnings.append("--noniid 0 is deprecated, use --non-iid 0")
            else:
                warnings.append(
                    "--noniid is deprecated; --non-iid now carries the degree"
                )
        elif key == "--customLR":
            legacy_seen = True
            if value is None:
                value = take_value()
            custom_lr = value is not None and _parse_bool(value)
        elif key in _LEGACY_STORE_TRUE and value is None and (
            index + 1 >= len(tokens) or tokens[index + 1].startswith("-")
        ):
            # A 1.0 store_true flag: it meant "on", so say so rather than
            # dropping it and silently turning the feature off.
            legacy_seen = True
            translated.extend([key, _LEGACY_STORE_TRUE[key]])
            warnings.append(f"{key} now takes a value, use {key} 1")
        elif key in _LEGACY_DROPPED_WITH_VALUE:
            legacy_seen = True
            if value is None:
                take_value()
            warnings.append(f"{key} was removed: {_LEGACY_DROPPED_WITH_VALUE[key]}")
        elif key in _LEGACY_DROPPED_FLAGS and value is None and (
            index + 1 >= len(tokens) or tokens[index + 1].startswith("--")
        ):
            legacy_seen = True
            warnings.append(f"{key} was removed: {_LEGACY_DROPPED_FLAGS[key]}")
        elif key in _LEGACY_ALIASES:
            legacy_seen = key != _LEGACY_ALIASES[key] or legacy_seen
            replacement = _LEGACY_ALIASES[key]
            if replacement != key:
                warnings.append(f"{key} is deprecated, use {replacement}")
            translated.append(replacement + sep + inline if sep else replacement)
        else:
            translated.append(token)
        index += 1

    if legacy_seen:
        schedule = _legacy_schedule(custom_lr, noniid_gate, "--slowdown" in explicit)
        for field, value in schedule.items():
            if _flag(field) not in explicit:
                overrides[field] = value
        warnings.append(
            "1.0 chose the schedule from --customLR; this resolves to "
            f"--lr-schedule {schedule['lr_schedule']}"
            + (f" --lr-decay-start {schedule['lr_decay_start']}"
               if "lr_decay_start" in schedule else "")
        )

    # Rename renamed values (pd-sgd -> pa-sgd, and the weight schemes).
    for position, token in enumerate(translated[:-1]):
        mapping = _LEGACY_VALUES.get(token)
        if mapping and translated[position + 1] in mapping:
            old = translated[position + 1]
            new = mapping[old]
            if new != old:
                warnings.append(f"{token} {old} is deprecated, use {token} {new}")
                translated[position + 1] = new
    return translated, overrides, warnings


def _legacy_schedule(
    custom_lr: bool, noniid: bool, slowdown: bool
) -> dict[str, Any]:
    """The schedule a 1.0 command line implied (paper Table 8).

    Train.py branched on ``--customLR``: set, it dropped the rate by 1/10 after
    epochs 81 and 122; unset, it halved every 10 epochs starting at 200 for
    non-IID runs and 100 otherwise. TrainSlowdown.py was identical but started
    at 50, and ``--slowdown`` is what tells the two scripts apart.
    """
    if custom_lr:
        return {"lr_schedule": "multistep", "lr_gamma": 0.1,
                "lr_milestones": [81, 122]}
    return {
        "lr_schedule": "step",
        "lr_gamma": 0.5,
        "lr_decay_every": 10,
        "lr_decay_start": 200 if noniid else (50 if slowdown else 100),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swift-train",
        description=(
            "Train a decentralised federated model with SWIFT or one of the "
            "synchronous baselines. Launch with mpirun; one rank per client."
        ),
        epilog="Example: mpirun -np 4 swift-train --config configs/baseline-ring.yaml",
    )
    parser.add_argument(
        "--config", type=Path, help="YAML file of Config values to start from"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="resolve the configuration, print it, and exit without training",
    )
    _add_field_arguments(parser)
    return parser


def parse_config(argv: Sequence[str] | None = None) -> tuple[Config, bool]:
    """Resolve defaults, YAML and flags into one Config."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    argv, legacy_overrides, warnings = _translate_legacy(argv)
    args = build_parser().parse_args(argv)

    # Merge before constructing, not after: building a Config and then
    # replacing fields would bake in defaults resolved against the wrong
    # values (weights is derived from algorithm, so --algorithm d-sgd on top
    # of a swift config would silently keep CCS weights).
    base = Config.raw_from_yaml(args.config) if args.config else {}
    overrides = {
        name: value
        for name, value in vars(args).items()
        if name in Config.field_names() and value is not None
    }
    # A 1.0 gate outranks the 1.0 flag it gated (--noniid 0 beats
    # --degree_noniid 0.7), so it is merged last. _translate_legacy already
    # dropped any gate that a new-style flag supersedes.
    config = Config(**(base | overrides | legacy_overrides))

    for warning in warnings:
        print(f"[swift] warning: {warning}", flush=True)
    return config, args.print_config


def main(argv: Sequence[str] | None = None) -> int:
    config, print_only = parse_config(argv)

    if print_only:
        import json

        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0

    from .train import run

    run(config)
    return 0
