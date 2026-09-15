"""Synchronous decentralised baselines: D-SGD, PA-SGD and LD-SGD.

All three are the same loop with different communication schedules, set by
``i1`` (local updates between averaging rounds) and ``i2`` (consecutive
averaging rounds that follow):

===========  ====  ====
Algorithm     i1    i2
===========  ====  ====
D-SGD           0     1
PA-SGD        > 0     1
LD-SGD        > 0   > 1
===========  ====  ====
"""

from __future__ import annotations

import time

import numpy as np
import torch
from mpi4py import MPI

from ..topology import Topology
from .tensors import copy_into_model, flatten_tensors, parameter_list

# Explicit tag: mpi4py's Sendrecv defaults to ANY_TAG, which would happily match
# any other message in flight from the same neighbour.
_MODEL_TAG = 11


class SyncCommunicator:
    """Neighbourhood averaging in lockstep: every client waits for every other."""

    def __init__(
        self, rank: int, size: int, comm: MPI.Comm, topology: Topology, i1: int, i2: int
    ) -> None:
        self.rank = rank
        self.size = size
        self.comm = comm
        self.topology = topology
        self.i1 = i1
        self.i2 = i2
        self.step = 0
        self.rounds = 0
        self.sync_time = 0.0
        """Seconds spent waiting for the slowest client at the pre-exchange
        barrier. Excluded from the reported communication time, as in 1.0 and
        the paper's tables, and recorded separately so it is not invisible."""

    def communicate(self, model: torch.nn.Module) -> float:
        """Average with neighbours if this step calls for it. Returns seconds spent."""
        self.step += 1
        # i1 + 1 rather than i1 so that i1 = 0 (D-SGD) averages every step.
        if self.step % (self.i1 + 1) != 0:
            return 0.0

        self.rounds += 1
        elapsed = self._average(model)

        if self.rounds % self.i2 == 0:
            self.rounds = 0
        else:
            # Roll the counter back so the next step averages again, giving the
            # i2 consecutive averaging rounds LD-SGD asks for.
            self.step -= 1
        return elapsed

    def _average(self, model: torch.nn.Module) -> float:
        params = parameter_list(model)
        send_buffer = flatten_tensors(params).detach().cpu()

        # Everyone starts the exchange together, so the timer below measures
        # only the exchange. Waiting for the slowest client is the cost SWIFT
        # exists to avoid, so it is measured rather than dropped.
        sync_start = time.perf_counter()
        self.comm.Barrier()
        start = time.perf_counter()
        self.sync_time += start - sync_start

        averaged = send_buffer * self.topology.self_weight
        send_numpy = send_buffer.numpy()
        received = np.empty_like(send_numpy)
        for idx, node in enumerate(self.topology.neighbors):
            self.comm.Sendrecv(
                sendbuf=send_numpy,
                dest=node,
                sendtag=_MODEL_TAG,
                recvbuf=received,
                source=node,
                recvtag=_MODEL_TAG,
            )
            averaged.add_(torch.from_numpy(received), alpha=self.topology.weights[idx])

        self.comm.Barrier()
        elapsed = time.perf_counter() - start

        copy_into_model(averaged.to(next(model.parameters()).device), params)
        return elapsed

    def finish(self, model: torch.nn.Module) -> None:
        """No teardown needed: nothing is ever left in flight."""
        del model
