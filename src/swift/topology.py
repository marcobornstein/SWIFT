"""Communication graphs and the mixing weights defined on them.

Two pieces:

* :func:`build_edges` turns a topology name into an edge list. Every rank must
  agree on the result, so the one random topology (Erdos-Renyi) is drawn on
  rank 0 and broadcast.
* :class:`Topology` computes a client's neighbours and the weight it assigns to
  each of them. The ``ccs`` scheme is Algorithm 2 (Communication Coefficient
  Selection) of the paper: a distributed protocol that makes the expected
  mixing matrix doubly stochastic even though each round's matrix is not.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from mpi4py import MPI

# Tag offsets, kept far apart from the model-exchange tags used in swift.py.
_DEGREE_TAG = 100
_WEIGHT_TAG = 200

_TOPOLOGIES = ("ring", "clique-ring", "fully-connected", "erdos-renyi")


def build_edges(
    topology: str,
    size: int,
    comm: MPI.Comm,
    rank: int,
    clusters: int = 1,
    edge_prob: float | None = None,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """Edge list for ``topology`` over ``size`` clients, identical on every rank."""
    if topology not in _TOPOLOGIES:
        raise ValueError(
            f"unknown topology {topology!r}; expected one of {sorted(_TOPOLOGIES)}"
        )
    if size < 2:
        # A lone client has nobody to talk to. Worth special-casing because
        # nx.cycle_graph(1) is a self-loop rather than an empty graph.
        return []

    if topology == "fully-connected":
        return list(nx.complete_graph(size).edges)

    if topology == "ring":
        return list(nx.cycle_graph(size).edges)

    if topology == "clique-ring":
        return _clique_ring(size, clusters)

    return _erdos_renyi(size, comm, rank, edge_prob, seed)


def _clique_ring(size: int, clusters: int) -> list[tuple[int, int]]:
    """``clusters`` cliques joined into a ring (the paper's ROC-xC topology)."""
    if clusters < 1:
        raise ValueError(f"clusters must be >= 1, got {clusters}")
    if size < clusters:
        raise ValueError(f"cannot split {size} clients into {clusters} cliques")

    per_clique, remainder = divmod(size, clusters)
    edges: list[tuple[int, int]] = []
    for i in range(clusters):
        last = i == clusters - 1
        members = per_clique + remainder if last else per_clique
        clique = nx.convert_node_labels_to_integers(
            nx.complete_graph(members), first_label=i * per_clique
        )
        edges += list(clique.edges)
        if not last:
            # Bridge to the next clique.
            edges.append((i * per_clique + per_clique - 1, i * per_clique + per_clique))
        elif clusters > 2:
            # Close the ring. With two cliques the bridge above already joins
            # them, and with one there is nothing to close.
            edges.append((size - 1, 0))
    return edges


def _erdos_renyi(
    size: int, comm: MPI.Comm, rank: int, edge_prob: float | None, seed: int | None
) -> list[tuple[int, int]]:
    """A connected random graph, drawn on rank 0 and broadcast to everyone."""
    p = edge_prob if edge_prob is not None else 3.0 / size
    num_edges = np.zeros(1, dtype=np.int64)

    if rank == 0:
        rng = np.random.default_rng(seed)
        while True:
            graph = nx.erdos_renyi_graph(size, p, seed=int(rng.integers(2**31)))
            if nx.is_connected(graph):
                break
        edges = list(graph.edges)
        num_edges[0] = len(edges)

    comm.Bcast(num_edges, root=0)
    buffer = (
        np.array(edges, dtype=np.int64)
        if rank == 0
        else np.empty((int(num_edges[0]), 2), dtype=np.int64)
    )
    comm.Bcast(buffer, root=0)
    return [(int(a), int(b)) for a, b in buffer]


class Topology:
    """One client's view of the communication graph: its neighbours and weights."""

    def __init__(
        self,
        rank: int,
        size: int,
        comm: MPI.Comm,
        edges: list[tuple[int, int]],
        scheme: str = "ccs",
    ) -> None:
        self.rank = rank
        self.size = size
        self.comm = comm
        self.edges = edges
        self.neighbors = self._neighbors_of(rank, size, edges)
        self.weights = (
            self._compute_weights(scheme)
            if size > 1
            else np.zeros(0, dtype=np.float64)
        )

    @property
    def degree(self) -> int:
        return len(self.neighbors)

    @property
    def self_weight(self) -> float:
        """The weight a client keeps on its own model: ``1 - sum(neighbours)``."""
        return float(1.0 - np.sum(self.weights))

    @staticmethod
    def _neighbors_of(
        rank: int, size: int, edges: list[tuple[int, int]]
    ) -> list[int]:
        neighbors: list[list[int]] = [[] for _ in range(size)]
        for a, b in edges:
            if a == b:
                raise ValueError(f"graph contains a self-loop at node {a}")
            neighbors[a].append(b)
            neighbors[b].append(a)
        return sorted(neighbors[rank])

    def _compute_weights(self, scheme: str) -> np.ndarray:
        if scheme == "ccs":
            return self._ccs_weights()
        if scheme == "global-uniform":
            return np.full(self.degree, 1.0 / self.size)
        if scheme == "neighborhood-uniform":
            return np.full(self.degree, 1.0 / (self.degree + 1))
        raise ValueError(f"unknown weight scheme {scheme!r}")

    def _ccs_weights(self) -> np.ndarray:
        """Algorithm 2: Communication Coefficient Selection.

        Clients settle weights in descending order of degree. The highest-degree
        client in a neighbourhood proposes ``1 / (degree + 1)`` to everyone; each
        remaining client waits to hear from all its higher-degree neighbours,
        then splits whatever weight is left uniformly over the rest. Ties are
        broken by both parties taking the smaller of the two proposals, so the
        protocol always terminates.

        Assumes uniform client influence scores, which is what every experiment
        in the paper uses.
        """
        comm, degree = self.comm, self.degree
        if degree == 0:
            # Isolated client: it only ever averages with itself.
            return np.zeros(0, dtype=np.float64)
        pending: list[MPI.Request] = []

        # Round 1: everyone learns their neighbours' degrees.
        send_degree = np.array([degree], dtype=np.float64)
        for node in self.neighbors:
            tag = self.rank + _DEGREE_TAG * self.size
            pending.append(comm.Isend(send_degree, dest=node, tag=tag))

        neighbor_degrees = np.empty(degree, dtype=np.float64)
        scratch = np.empty(1, dtype=np.float64)
        for idx, node in enumerate(self.neighbors):
            comm.Recv(scratch, source=node, tag=node + _DEGREE_TAG * self.size)
            neighbor_degrees[idx] = scratch[0]

        MPI.Request.Waitall(pending)
        pending.clear()

        # Neighbours sorted by descending degree, carrying their index into
        # self.neighbors so weights land in the right slot.
        order = np.argsort(-neighbor_degrees, kind="stable")
        sorted_degrees = neighbor_degrees[order]
        sorted_nodes = np.asarray(self.neighbors)[order]

        my_tag = self.rank + _WEIGHT_TAG * self.size
        weights = np.zeros(degree, dtype=np.float64)

        if degree >= sorted_degrees[0]:
            # Highest degree locally: propose a uniform neighbourhood weight.
            proposal = 1.0 / (degree + 1)
            weights[:] = proposal
            send_weight = np.array([proposal], dtype=np.float64)
            for node in sorted_nodes:
                pending.append(comm.Isend(send_weight, dest=node, tag=my_tag))
        else:
            # Wait for every strictly-higher-degree neighbour to commit first.
            while sorted_degrees.size and degree < sorted_degrees[0]:
                comm.Recv(
                    scratch,
                    source=int(sorted_nodes[0]),
                    tag=int(sorted_nodes[0]) + _WEIGHT_TAG * self.size,
                )
                weights[order[0]] = scratch[0]
                order, sorted_nodes, sorted_degrees = (
                    order[1:],
                    sorted_nodes[1:],
                    sorted_degrees[1:],
                )

            if sorted_degrees.size:
                remaining = 1.0 - float(np.sum(weights))
                proposal = remaining / (sorted_degrees.size + 1)

                if degree == sorted_degrees[0]:
                    # Same-degree neighbours propose simultaneously; both sides
                    # adopt the smaller proposal so the weights stay consistent.
                    # Sends are posted before the receives, so neither side can
                    # block waiting for the other.
                    tied = sorted_nodes[sorted_degrees == degree]
                    # One buffer, held until Waitall: MPI reads it
                    # asynchronously, so a per-send temporary could be collected
                    # while the send is still in flight. Rebinding `proposal`
                    # below does not touch it.
                    tied_buffer = np.array([proposal], dtype=np.float64)
                    tied_requests = [
                        comm.Isend(tied_buffer, dest=int(node), tag=my_tag)
                        for node in tied
                    ]
                    for node in tied:
                        comm.Recv(
                            scratch,
                            source=int(node),
                            tag=int(node) + _WEIGHT_TAG * self.size,
                        )
                        proposal = min(proposal, float(scratch[0]))
                    MPI.Request.Waitall(tied_requests)

                # `order` now holds exactly the neighbours still unset: the
                # higher-degree ones were sliced off as they committed.
                weights[order] = proposal
                send_weight = np.array([proposal], dtype=np.float64)
                for node in sorted_nodes[sorted_degrees != degree]:
                    comm.Send(send_weight, dest=int(node), tag=my_tag)

        comm.Barrier()
        MPI.Request.Waitall(pending)
        self._discard_unclaimed_proposals()
        return weights

    def _discard_unclaimed_proposals(self) -> None:
        """Drop weight proposals nobody was waiting for.

        On a regular graph every client is the highest-degree client in its own
        neighbourhood, so all of them propose and none of them listen. The
        proposals are small enough to be buffered and would otherwise sit in the
        unexpected-message queue, where a later receive could match them instead
        of a model.
        """
        status = MPI.Status()
        for node in self.neighbors:
            tag = node + _WEIGHT_TAG * self.size
            while self.comm.Iprobe(source=node, tag=tag, status=status):
                # Size the buffer from the message rather than assuming, so an
                # unexpected sender cannot truncate the receive.
                count = status.Get_count(MPI.DOUBLE)
                self.comm.Recv(np.empty(count, dtype=np.float64), source=node, tag=tag)
