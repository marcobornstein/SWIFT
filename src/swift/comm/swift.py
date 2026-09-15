"""SWIFT: wait-free asynchronous decentralised communication.

A client never blocks on a neighbour. Each step it broadcasts its own model
(one non-blocking send per neighbour) and, on averaging steps, takes whatever
has already arrived. Neighbours that have not sent anything since the last
round contribute either their last-known model or nothing at all, depending on
``low_memory``.

This is Algorithm 1 of the paper; :class:`swift.topology.Topology` supplies the
mixing weights from Algorithm 2.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch
from mpi4py import MPI

from ..topology import Topology
from .tensors import copy_into_model, flatten_tensors, parameter_list

# Tag offsets. Model messages use the sender's rank directly, so the exit
# protocol is shifted clear of them.
_EXIT_TAG = 2
_COUNT_TAG = 3
_POLL_INTERVAL = 0.5


class _SendQueue:
    """In-flight broadcasts, reaped as they complete.

    Each request is held together with the buffer it is sending from: MPI reads
    that memory asynchronously, so releasing it before the request completes
    would corrupt the message. Sends are matched in order per destination, so
    completed requests always sit at the front.

    When the queue is full the oldest send is waited on, which throttles a
    client whose neighbour has stopped listening. That wait is only safe
    because it cannot happen on both sides of an edge at once: a model is
    larger than any eager threshold, so a send completes only once the peer
    receives it, and two peers each blocked on a send to the other would
    deadlock. :class:`SwiftCommunicator` sizes the queue past the most a
    healthy client can accumulate between averaging rounds, so reaching the
    cap means the peer really has stalled -- and a stalled peer is not itself
    blocked in a send, so it still drains.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._queue: deque[tuple[MPI.Request, np.ndarray]] = deque()

    def reap(self) -> None:
        """Retire sends that have landed."""
        while self._queue and self._queue[0][0].Test():
            self._queue.popleft()

    @property
    def full(self) -> bool:
        self.reap()
        return len(self._queue) > self.capacity

    def add(self, request: MPI.Request, payload: np.ndarray,
            block: bool = True) -> None:
        self._queue.append((request, payload))
        self.reap()
        while block and len(self._queue) > self.capacity:
            self._queue.popleft()[0].Wait()

    def wait_all(self) -> None:
        if self._queue:
            MPI.Request.Waitall([request for request, _ in self._queue])
            self._queue.clear()


class SwiftCommunicator:
    """Wait-free neighbourhood averaging."""

    def __init__(
        self,
        rank: int,
        size: int,
        comm: MPI.Comm,
        topology: Topology,
        initial_model: np.ndarray,
        local_steps: int = 1,
        weight_boost: bool = True,
        low_memory: bool = False,
        max_pending_sends: int = 1024,
    ) -> None:
        self.rank = rank
        self.size = size
        self.comm = comm
        self.topology = topology
        self.local_steps = local_steps
        self.weight_boost = weight_boost
        self.low_memory = low_memory

        self.step = 0
        self.missed_messages = 0
        self.sync_time = 0.0
        """Always zero: SWIFT never waits for a neighbour, so any time it
        spends on communication is already inside the reported figure."""
        # A client posts `degree` sends per step and drains every neighbour
        # every `local_steps` steps, so the queue holds at most
        # degree * local_steps in normal operation. Keep several times that in
        # reserve: see the note on blocking in _SendQueue.
        headroom = 4 * self.topology.degree * max(local_steps, 1) + 16
        self._sends = _SendQueue(max(max_pending_sends, headroom))
        self._sent = np.zeros(self.topology.degree, dtype=np.int64)
        self._received = np.zeros(self.topology.degree, dtype=np.int64)

        model_size = initial_model.size
        self._scratch = np.empty(model_size, dtype=initial_model.dtype)
        if low_memory:
            # One shared buffer: only neighbours heard from this round count.
            self._neighbor_models = np.empty(model_size, dtype=initial_model.dtype)
        else:
            # One slot per neighbour, seeded with the common initial model so
            # that a silent neighbour still contributes something sensible.
            self._neighbor_models = np.tile(
                initial_model, (self.topology.degree, 1)
            )

    # ------------------------------------------------------------------ public
    def communicate(self, model: torch.nn.Module) -> float:
        """Broadcast, and average on averaging steps. Returns seconds spent."""
        self.step += 1
        params = parameter_list(model)
        flat = flatten_tensors(params).detach().cpu()

        # flatten_tensors returns a fresh buffer, so this is not a view of the
        # model and will not be mutated underneath an in-flight send. The queue
        # holds a reference to it until MPI is done reading.
        elapsed = self._broadcast(flat.numpy())
        if self.step % self.local_steps == 0:
            elapsed += self._average(model, params, flat)
        return elapsed

    def finish(self, model: torch.nn.Module) -> None:
        """Announce completion, keep feeding neighbours, then settle all traffic."""
        degree = self.topology.degree
        if degree == 0:
            return

        flat = flatten_tensors(parameter_list(model)).detach().cpu().numpy()
        outgoing = np.ones(1, dtype=np.float64)
        incoming = [np.zeros(1, dtype=np.float64) for _ in range(degree)]

        exit_sends = [
            self.comm.Isend(outgoing, dest=node, tag=self.rank + _EXIT_TAG * self.size)
            for node in self.topology.neighbors
        ]
        exit_recvs = [
            self.comm.Irecv(incoming[idx], source=node, tag=node + _EXIT_TAG * self.size)
            for idx, node in enumerate(self.topology.neighbors)
        ]

        finished = [False] * degree
        # At most one model in flight per neighbour. A finished client's model
        # never changes, so queueing more conveys nothing -- and it outruns the
        # receiver: a large message only moves while the *sender* is inside an
        # MPI call, so a neighbour polling on a timer delivers more slowly than
        # it posts, and the receiver's drain loop never empties.
        in_flight: list[MPI.Request | None] = [None] * degree

        while not all(finished):
            for idx, node in enumerate(self.topology.neighbors):
                # Keep draining even after finishing. A client in finish() has
                # stopped averaging, but if it also stopped receiving, its
                # neighbours' sends would never complete -- and a neighbour
                # blocked on one of those sends never reaches the test below.
                self._drain_latest(idx, node, self._scratch)
                if finished[idx]:
                    continue
                if exit_recvs[idx].Test():
                    finished[idx] = True
                elif in_flight[idx] is None or in_flight[idx].Test():
                    # Still training: keep giving them something to average.
                    in_flight[idx] = self.comm.Isend(flat, dest=node, tag=self.rank)
                    self._sent[idx] += 1
            if not all(finished):
                time.sleep(_POLL_INTERVAL)

        # These are tracked here rather than in the send queue, so hand them
        # over for the final reconciliation.
        for request in in_flight:
            if request is not None:
                self._sends.add(request, flat, block=False)

        MPI.Request.Waitall(exit_sends)
        self._settle()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "missed_messages": int(self.missed_messages),
            "messages_sent": int(self._sent.sum()),
            "messages_received": int(self._received.sum()),
        }

    # ----------------------------------------------------------------- internal
    def _send_to(
        self, idx: int, node: int, payload: np.ndarray, block: bool = True
    ) -> bool:
        """Queue a non-blocking send, keeping ``payload`` alive until it lands.

        With ``block`` false the send is skipped rather than waited on when the
        queue is full. :meth:`finish` needs that: waiting there would stop it
        reaching the test that ends the loop.
        """
        if not block and self._sends.full:
            return False
        self._sends.add(
            self.comm.Isend(payload, dest=node, tag=self.rank), payload, block
        )
        self._sent[idx] += 1
        return True

    def _broadcast(self, payload: np.ndarray) -> float:
        start = time.perf_counter()
        for idx, node in enumerate(self.topology.neighbors):
            self._send_to(idx, node, payload)
        return time.perf_counter() - start

    def _average(
        self, model: torch.nn.Module, params: list[torch.Tensor], flat: torch.Tensor
    ) -> float:
        start = time.perf_counter()
        averaged = (
            self._average_low_memory(flat)
            if self.low_memory
            else self._average_stored(flat)
        )
        elapsed = time.perf_counter() - start

        copy_into_model(averaged.to(params[0].device), params)
        return elapsed

    def _average_stored(self, flat: torch.Tensor) -> torch.Tensor:
        """Average over every neighbour, falling back on their last-known model."""
        averaged = torch.zeros_like(flat)
        weights = self.topology.weights

        for idx, node in enumerate(self.topology.neighbors):
            if self._drain_latest(idx, node, self._scratch):
                self._neighbor_models[idx] = self._scratch
            else:
                self.missed_messages += 1
            averaged.add_(
                torch.from_numpy(self._neighbor_models[idx]), alpha=weights[idx]
            )

        averaged.add_(flat, alpha=self.topology.self_weight)
        return averaged

    def _average_low_memory(self, flat: torch.Tensor) -> torch.Tensor:
        """Average over only the neighbours heard from, using one buffer.

        With ``weight_boost`` the surviving weights are scaled up so they still
        form a convex combination; without it, the absent neighbours' weight is
        handed back to the client's own model.
        """
        averaged = torch.zeros_like(flat)
        weights = self.topology.weights

        heard: list[int] = []
        for idx, node in enumerate(self.topology.neighbors):
            if self._drain_latest(idx, node, self._neighbor_models):
                heard.append(idx)
                averaged.add_(torch.from_numpy(self._neighbor_models), alpha=weights[idx])
            else:
                self.missed_messages += 1

        if self.weight_boost:
            boost = (len(heard) + 1) / (self.topology.degree + 1)
            averaged.div_(boost)
            self_weight = self.topology.self_weight / boost
        else:
            self_weight = 1.0 - float(np.sum(weights[heard]))

        averaged.add_(flat, alpha=self_weight)
        return averaged

    def _drain_latest(self, idx: int, node: int, buffer: np.ndarray) -> bool:
        """Take the newest model queued from ``node``, discarding older ones.

        Returns whether anything arrived. Stale models are dropped rather than
        averaged in, so a fast neighbour cannot weight the average by sending
        more often. Every receive is guarded by ``Iprobe``, so none of them
        block: this is the wait-free part of SWIFT.
        """
        received = False
        while self.comm.Iprobe(source=node, tag=node):
            self.comm.Recv(buffer, source=node, tag=node)
            self._received[idx] += 1
            received = True
        return received

    def _settle(self) -> None:
        """Consume every model a neighbour sent us, then close our own sends.

        Neighbours exchange exact send counts, so each client knows how many
        messages are still outstanding and can receive precisely that many. A
        blind drain would race against messages still in flight.
        """
        for idx, node in enumerate(self.topology.neighbors):
            their_count = np.zeros(1, dtype=np.int64)
            self.comm.Sendrecv(
                sendbuf=self._sent[idx : idx + 1],
                dest=node,
                sendtag=self.rank + _COUNT_TAG * self.size,
                recvbuf=their_count,
                source=node,
                recvtag=node + _COUNT_TAG * self.size,
            )
            for _ in range(int(their_count[0]) - int(self._received[idx])):
                self.comm.Recv(self._scratch, source=node, tag=node)
                self._received[idx] += 1

        self._sends.wait_all()
