"""Run configuration: one dataclass, loadable from YAML and overridable on the CLI."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import textwrap
from dataclasses import dataclass, field
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import yaml

# Kept here rather than imported from swift.model so that --help does not pay
# for importing torch. tests/test_config.py asserts the two agree.
RESNET_DEPTHS = (18, 34, 50, 101, 152)

Algorithm = Literal["swift", "d-sgd", "pa-sgd", "ld-sgd"]
Topology = Literal["ring", "clique-ring", "fully-connected", "erdos-renyi"]
WeightScheme = Literal["ccs", "neighborhood-uniform", "global-uniform"]
LRSchedule = Literal["constant", "step", "multistep"]


@dataclass
class Config:
    """Every knob for a single training run.

    Defaults reproduce the paper's baseline experiment (Table 3): 16 clients in
    a ring, ResNet-18 on IID CIFAR-10, 200 epochs.
    """

    # --- experiment identity -------------------------------------------------
    name: str = "swift"
    """Run name; also the output sub-directory."""
    notes: str = ""
    """Free-text description stored alongside the results."""
    seed: int = 9001
    output_dir: Path = Path("outputs")

    # --- algorithm -----------------------------------------------------------
    algorithm: Algorithm = "swift"
    """swift | d-sgd | pa-sgd | ld-sgd."""
    local_steps: int = 1
    """SWIFT: gradient steps between averaging rounds (|C_s| in the paper)."""
    weight_boost: bool = True
    """SWIFT: renormalise weights over the neighbours actually heard from.

    When False, a client that hears from nobody falls back on its own model
    with the missing neighbours' weight added to its self-weight.
    """
    low_memory: bool = False
    """SWIFT: keep one receive buffer instead of one model per neighbour.

    Saves ``degree - 1`` model copies at the cost of averaging only over
    neighbours heard from this round.
    """
    i1: int = 0
    """PA-SGD / LD-SGD: local updates between averaging rounds. Must be >= 1."""
    i2: int = 1
    """LD-SGD: consecutive averaging rounds after the local updates. Must be >= 2."""

    # --- topology ------------------------------------------------------------
    topology: Topology = "ring"
    clusters: int = 1
    """clique-ring: number of cliques in the ring."""
    edge_prob: float | None = None
    """erdos-renyi: edge probability. Defaults to ``3 / num_clients``."""
    weights: WeightScheme | None = None
    """Mixing weights. Defaults to ``ccs`` (Algorithm 2) for SWIFT and to
    ``neighborhood-uniform`` for the synchronous baselines."""

    # --- model ---------------------------------------------------------------
    resnet_depth: int = 18
    """18 | 34 | 50 | 101 | 152 (CIFAR-style ResNet)."""

    # --- data ----------------------------------------------------------------
    dataset: Literal["cifar10"] = "cifar10"
    data_dir: Path = Path("data")
    download: bool = True
    batch_size: int = 32
    non_iid: float = 0.0
    """Fraction of each client's data drawn from its own label bin (0 = IID)."""
    num_workers: int = 0
    """DataLoader worker processes. Leave at 0 under MPI.

    Workers are forked after MPI has initialised, which MPI implementations
    document as unsafe and which hangs with Open MPI on macOS. 0 is also what
    the paper's timings were measured with.
    """

    # --- optimisation --------------------------------------------------------
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = False
    lr_schedule: LRSchedule = "multistep"
    lr_gamma: float = 0.1
    """Multiplicative decay applied at each decay point."""
    lr_milestones: list[int] = field(default_factory=lambda: [81, 122])
    """multistep: epochs after which the learning rate drops by ``lr_gamma``."""
    lr_decay_start: int = 100
    """step: first epoch at which decay begins."""
    lr_decay_every: int = 10
    """step: epochs between successive decays."""

    # --- evaluation ----------------------------------------------------------
    eval_subset: int = 500
    """Test images used for the per-epoch loss. 0 evaluates the full test set.

    The paper's runs used 500, drawn identically on every client.
    """

    # --- runtime -------------------------------------------------------------
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    slowdown: float = 1.0
    """Artificial straggler factor applied to rank 0 (Section 6.2). 1 = none."""
    cudnn_benchmark: bool = True
    deterministic: bool = False
    """Force deterministic kernels. Slower, and disables cudnn autotuning."""
    max_pending_sends: int = 1024
    """SWIFT: floor on the in-flight broadcast queue before backpressure.

    Raised automatically when the topology and ``local_steps`` need more.
    """

    # ------------------------------------------------------------------ utils
    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.lr_milestones = sorted(int(m) for m in self.lr_milestones)
        if not 0.0 <= self.non_iid <= 1.0:
            raise ValueError(f"non_iid must be in [0, 1], got {self.non_iid}")
        if self.local_steps < 1:
            raise ValueError(f"local_steps must be >= 1, got {self.local_steps}")
        if self.i2 < 1:
            raise ValueError(f"i2 must be >= 1, got {self.i2}")
        if self.slowdown < 1.0:
            raise ValueError(f"slowdown must be >= 1, got {self.slowdown}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.eval_subset < 0:
            raise ValueError(
                f"eval_subset must be >= 0 (0 means the full test set), "
                f"got {self.eval_subset}"
            )
        if self.clusters < 1:
            raise ValueError(f"clusters must be >= 1, got {self.clusters}")
        if self.lr_decay_every < 1:
            # lr_at divides by this.
            raise ValueError(
                f"lr_decay_every must be >= 1, got {self.lr_decay_every}"
            )
        if self.resnet_depth not in RESNET_DEPTHS:
            raise ValueError(
                f"resnet_depth must be one of {list(RESNET_DEPTHS)}, "
                f"got {self.resnet_depth}"
            )
        # Without these, the communicator schedule collapses onto D-SGD's and
        # the run looks fine while measuring the wrong algorithm.
        if self.algorithm in {"pa-sgd", "ld-sgd"} and self.i1 < 1:
            raise ValueError(
                f"{self.algorithm} needs i1 >= 1 local updates between averaging "
                f"rounds (got {self.i1}); i1 = 0 is D-SGD"
            )
        if self.algorithm == "ld-sgd" and self.i2 < 2:
            raise ValueError(
                f"ld-sgd needs i2 >= 2 averaging rounds (got {self.i2}); "
                f"i2 = 1 is pa-sgd"
            )
        if self.weights is None:
            self.weights = "ccs" if self.algorithm == "swift" else "neighborhood-uniform"
        if self.algorithm == "swift" and self.weights != "ccs":
            raise ValueError("SWIFT requires weights='ccs' (Algorithm 2)")

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.name

    def lr_at(self, epoch: int) -> float:
        """Learning rate for ``epoch``, matching the paper's Table 8 schedules."""
        if self.lr_schedule == "constant":
            return self.lr
        if self.lr_schedule == "multistep":
            drops = sum(1 for m in self.lr_milestones if m < epoch)
            return self.lr * self.lr_gamma**drops
        # "step": decay begins the epoch after lr_decay_start, then every
        # lr_decay_every epochs.
        drops = max(0, (epoch - self.lr_decay_start) // self.lr_decay_every)
        return self.lr * self.lr_gamma**drops

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in out.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in dataclasses.fields(cls)}

    @staticmethod
    @cache
    def field_docs() -> dict[str, str]:
        """The docstring written under each field, for use in ``--help``.

        Read from the source so the CLI help and the dataclass cannot drift.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(Config)))
        body = tree.body[0].body  # type: ignore[attr-defined]
        docs: dict[str, str] = {}
        for statement, following in pairwise(body):
            is_field = isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            )
            is_doc = isinstance(following, ast.Expr) and isinstance(
                following.value, ast.Constant
            )
            if is_field and is_doc:
                text = str(following.value.value).strip()  # type: ignore[attr-defined]
                docs[statement.target.id] = " ".join(text.split())  # type: ignore[attr-defined]
        return docs

    @classmethod
    def raw_from_yaml(cls, path: str | Path) -> dict[str, Any]:
        """Validated field values from ``path``, left unresolved.

        Returned as a plain dict rather than a Config so that CLI overrides can
        be merged in before construction. Fields whose default depends on
        another field -- ``weights`` on ``algorithm`` -- must be resolved once,
        against the final values.
        """
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - cls.field_names()
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        return raw

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return cls(**cls.raw_from_yaml(path))
