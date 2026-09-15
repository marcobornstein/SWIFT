"""Communicators: one wait-free (SWIFT) and one synchronous (the baselines)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from mpi4py import MPI

from ..config import Config
from ..topology import Topology
from .swift import SwiftCommunicator
from .sync import SyncCommunicator
from .tensors import copy_into_model, flatten_tensors, parameter_list, unflatten_tensors

__all__ = [
    "Communicator",
    "SwiftCommunicator",
    "SyncCommunicator",
    "build_communicator",
    "copy_into_model",
    "flatten_tensors",
    "parameter_list",
    "synchronize_model",
    "unflatten_tensors",
]

# (i1, i2) for each synchronous baseline. See comm/sync.py.
_SYNC_SCHEDULES = {"d-sgd": (0, 1), "pa-sgd": (None, 1), "ld-sgd": (None, None)}


class Communicator(Protocol):
    sync_time: float
    """Cumulative seconds spent waiting for other clients, outside communicate()."""

    def communicate(self, model: torch.nn.Module) -> float: ...
    def finish(self, model: torch.nn.Module) -> None: ...


def build_communicator(
    config: Config,
    rank: int,
    size: int,
    comm: MPI.Comm,
    topology: Topology,
    initial_model: np.ndarray,
) -> Communicator:
    """Pick the communicator for ``config.algorithm``."""
    if config.algorithm == "swift":
        return SwiftCommunicator(
            rank=rank,
            size=size,
            comm=comm,
            topology=topology,
            initial_model=initial_model,
            local_steps=config.local_steps,
            weight_boost=config.weight_boost,
            low_memory=config.low_memory,
            max_pending_sends=config.max_pending_sends,
        )

    if config.algorithm not in _SYNC_SCHEDULES:
        raise ValueError(f"unknown algorithm {config.algorithm!r}")

    default_i1, default_i2 = _SYNC_SCHEDULES[config.algorithm]
    i1 = config.i1 if default_i1 is None else default_i1
    i2 = config.i2 if default_i2 is None else default_i2
    return SyncCommunicator(rank, size, comm, topology, i1=i1, i2=i2)


def synchronize_model(
    model: torch.nn.Module, size: int, comm: MPI.Comm
) -> np.ndarray:
    """Average every client's parameters so all of them start from one model.

    Returns the agreed model as a flat float32 vector, which SWIFT uses to seed
    its per-neighbour stores.
    """
    params = parameter_list(model)
    flat = flatten_tensors(params).detach().cpu().numpy().astype(np.float32)

    total = np.empty_like(flat)
    comm.Allreduce(flat, total, op=MPI.SUM)
    total /= float(size)

    device = params[0].device
    copy_into_model(torch.from_numpy(total).to(device), params)
    return total
