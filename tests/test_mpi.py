"""Tests that need real ranks. Run with mpirun:

    mpirun -np 4 --oversubscribe python -m pytest tests/test_mpi.py -q

Under a single rank they skip, so a plain ``pytest`` run stays green.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from mpi4py import MPI

from swift.comm import build_communicator, synchronize_model
from swift.config import Config
from swift.topology import Topology, build_edges

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()

pytestmark = pytest.mark.skipif(
    SIZE < 2, reason="needs at least 2 MPI ranks (run under mpirun)"
)


def _topology(name: str = "ring", scheme: str = "ccs", clusters: int = 1) -> Topology:
    edges = build_edges(name, SIZE, COMM, RANK, clusters=clusters, seed=0)
    return Topology(RANK, SIZE, COMM, edges, scheme=scheme)


def _tiny_model(fill: float) -> nn.Module:
    model = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        model.weight.fill_(fill)
    return model


def _large_model() -> nn.Module:
    """Bigger than any eager threshold, so sends need a matching receive."""
    model = nn.Linear(2048, 128, bias=False)   # 256k floats, 1 MB
    with torch.no_grad():
        model.weight.fill_(1.0)
    return model


def _flat(model: nn.Module) -> np.ndarray:
    return (
        torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        .cpu()
        .numpy()
        .copy()
    )


@pytest.mark.parametrize("scheme", ["ccs", "neighborhood-uniform"])
def test_mixing_rows_sum_to_one(scheme: str) -> None:
    topology = _topology(scheme=scheme)
    assert topology.self_weight + float(np.sum(topology.weights)) == pytest.approx(1.0)
    assert topology.self_weight > 0
    assert all(w > 0 for w in topology.weights)


def test_ccs_weights_are_symmetric_and_doubly_stochastic() -> None:
    """Contribution (2) of the paper: Algorithm 2 makes W symmetric and
    doubly stochastic, even though the graph need not be regular."""
    topology = _topology("clique-ring", scheme="ccs", clusters=min(3, SIZE))

    matrix = _mixing_matrix(topology)

    assert np.allclose(matrix.sum(axis=1), 1.0), "rows are not stochastic"
    assert np.allclose(matrix.sum(axis=0), 1.0), "columns are not stochastic"
    assert np.allclose(matrix, matrix.T), "matrix is not symmetric"
    # The paper requires w_ii >= 1/n so a client never discards its own model.
    assert np.all(np.diag(matrix) >= 1.0 / SIZE - 1e-9)
    _assert_weight_only_on_edges(matrix, topology.edges)


def _mixing_matrix(topology: Topology) -> np.ndarray:
    """Gather every client's row of the mixing matrix onto every rank."""
    row = np.zeros(SIZE)
    row[RANK] = topology.self_weight
    for node, weight in zip(topology.neighbors, topology.weights, strict=True):
        row[node] = weight
    matrix = np.zeros((SIZE, SIZE))
    COMM.Allgather(row, matrix)
    return matrix


def _assert_weight_only_on_edges(
    matrix: np.ndarray, edges: list[tuple[int, int]]
) -> None:
    """A client may only put weight on itself and its one-hop neighbours."""
    adjacency = {frozenset(edge) for edge in edges}
    for i in range(SIZE):
        for j in range(SIZE):
            if i == j:
                continue
            connected = frozenset((i, j)) in adjacency
            assert (matrix[i, j] > 0) == connected, (
                f"weight {matrix[i, j]} between {i} and {j}, "
                f"which are {'' if connected else 'not '}neighbours"
            )


def test_erdos_renyi_graph_is_agreed_and_connected() -> None:
    """The one random topology: drawn on rank 0, so every rank must match."""
    import networkx as nx

    edges = build_edges("erdos-renyi", SIZE, COMM, RANK, edge_prob=0.5, seed=7)
    gathered = COMM.allgather(sorted(map(tuple, edges)))
    assert all(other == gathered[0] for other in gathered), "ranks disagree"

    graph = nx.Graph(edges)
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == SIZE


def test_ccs_holds_on_an_irregular_random_graph() -> None:
    """CCS on a graph whose degrees genuinely differ.

    Ring and clique-ring are regular or nearly so, which lets every client take
    the highest-degree branch of Algorithm 2. Erdos-Renyi forces the protocol
    to actually negotiate.
    """
    edges = build_edges("erdos-renyi", SIZE, COMM, RANK, edge_prob=0.5, seed=11)
    matrix = _mixing_matrix(Topology(RANK, SIZE, COMM, edges, scheme="ccs"))

    assert np.allclose(matrix.sum(axis=1), 1.0), "rows are not stochastic"
    assert np.allclose(matrix.sum(axis=0), 1.0), "columns are not stochastic"
    assert np.allclose(matrix, matrix.T), "matrix is not symmetric"
    assert np.all(np.diag(matrix) >= 1.0 / SIZE - 1e-9)
    _assert_weight_only_on_edges(matrix, edges)


def test_synchronize_model_agrees_across_ranks() -> None:
    model = _tiny_model(float(RANK))
    agreed = synchronize_model(model, SIZE, COMM)

    expected = float(sum(range(SIZE))) / SIZE
    assert np.allclose(agreed, expected)
    assert np.allclose(_flat(model), expected)

    gathered = COMM.allgather(_flat(model))
    assert all(np.array_equal(gathered[0], other) for other in gathered)


@pytest.mark.parametrize("algorithm", ["swift", "d-sgd", "pa-sgd", "ld-sgd"])
def test_consensus_is_a_fixed_point(algorithm: str) -> None:
    """If every client already holds the same model, averaging must not move it.

    This is the sharpest check on the weights: any row that does not sum to one
    shows up immediately as drift.
    """
    model = _tiny_model(2.5)
    config = Config(algorithm=algorithm, i1=1, i2=2, epochs=1)
    topology = _topology(scheme=config.weights)
    communicator = build_communicator(
        config, RANK, SIZE, COMM, topology, _flat(model)
    )

    for _ in range(6):
        communicator.communicate(model)
        COMM.Barrier()
    communicator.finish(model)

    assert np.allclose(_flat(model), 2.5, atol=1e-5), f"{algorithm} drifted off consensus"


def test_dsgd_averaging_is_the_weighted_mean() -> None:
    """One D-SGD round equals one explicit multiplication by the mixing row."""
    model = _tiny_model(float(RANK))
    config = Config(algorithm="d-sgd")
    topology = _topology(scheme=config.weights)
    communicator = build_communicator(
        config, RANK, SIZE, COMM, topology, _flat(model)
    )

    pairs = zip(topology.neighbors, topology.weights, strict=True)
    expected = RANK * topology.self_weight + sum(node * w for node, w in pairs)
    communicator.communicate(model)
    assert np.allclose(_flat(model), expected, atol=1e-5)


def test_finish_drains_while_waiting() -> None:
    """A client in finish() keeps receiving.

    If it stopped, a neighbour's sends would never complete, and a neighbour
    blocked on one of those sends never reaches its own shutdown.
    """
    model = _large_model()
    config = Config(algorithm="swift")
    topology = _topology()
    communicator = build_communicator(
        config, RANK, SIZE, COMM, topology, _flat(model)
    )
    # Rank 0 lags, so everyone else sits in finish() while it keeps broadcasting.
    for _ in range(25 if RANK == 0 else 1):
        communicator.communicate(model)

    communicator.finish(model)
    COMM.Barrier()
    sent = np.array(COMM.allgather(communicator.stats["messages_sent"])).sum()
    received = np.array(COMM.allgather(communicator.stats["messages_received"])).sum()
    assert sent == received, "messages were left in flight"


@pytest.mark.parametrize("low_memory", [False, True])
@pytest.mark.parametrize("weight_boost", [False, True])
def test_swift_terminates_and_settles_all_traffic(
    low_memory: bool, weight_boost: bool
) -> None:
    """Every broadcast must be matched, or MPI_Finalize would hang."""
    model = _tiny_model(1.0)
    config = Config(
        algorithm="swift",
        low_memory=low_memory,
        weight_boost=weight_boost,
        local_steps=2,
    )
    topology = _topology()
    communicator = build_communicator(
        config, RANK, SIZE, COMM, topology, _flat(model)
    )

    for _ in range(5):
        communicator.communicate(model)
    communicator.finish(model)

    stats = communicator.stats
    sent = np.array(COMM.allgather(stats["messages_sent"])).sum()
    received = np.array(COMM.allgather(stats["messages_received"])).sum()
    assert sent == received, "messages were left in flight"
    COMM.Barrier()
