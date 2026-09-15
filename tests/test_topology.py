"""Graph construction and mixing weights.

The single-rank cases here need no MPI communication; the CCS protocol itself
is exercised by tests/test_mpi.py under mpirun.
"""

from __future__ import annotations

import networkx as nx
import pytest
from mpi4py import MPI

from swift.topology import Topology, build_edges


def _edges(topology: str, size: int, **kwargs) -> list[tuple[int, int]]:
    return build_edges(topology, size, MPI.COMM_SELF, 0, **kwargs)


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_ring_is_connected_and_two_regular(size: int) -> None:
    graph = nx.Graph(_edges("ring", size))
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == size
    # A two-client "ring" is a single edge, so each client has one neighbour.
    expected = 2 if size > 2 else 1
    assert all(degree == expected for _, degree in graph.degree)


@pytest.mark.parametrize(("size", "clusters"), [(10, 3), (16, 2), (16, 4), (8, 1)])
def test_clique_ring_is_connected(size: int, clusters: int) -> None:
    graph = nx.Graph(_edges("clique-ring", size, clusters=clusters))
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == size


def test_clique_ring_single_cluster_is_fully_connected() -> None:
    assert set(map(frozenset, _edges("clique-ring", 6, clusters=1))) == set(
        map(frozenset, _edges("fully-connected", 6))
    )


def test_unknown_topology_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown topology"):
        _edges("mobius-strip", 4)


@pytest.mark.parametrize(
    "topology", ["ring", "clique-ring", "fully-connected", "erdos-renyi"]
)
def test_a_lone_client_has_no_edges(topology: str) -> None:
    """nx.cycle_graph(1) is a self-loop, which the neighbour scan rejects."""
    assert _edges(topology, 1) == []


@pytest.mark.parametrize("size", [2, 3, 8])
def test_no_topology_produces_a_self_loop(size: int) -> None:
    for topology in ("ring", "clique-ring", "fully-connected"):
        edges = _edges(topology, size, clusters=1) if topology == "clique-ring" \
            else _edges(topology, size)
        assert all(a != b for a, b in edges), f"{topology} n={size}"


def test_self_loops_are_rejected() -> None:
    with pytest.raises(ValueError, match="self-loop"):
        Topology._neighbors_of(0, 2, [(0, 0)])


@pytest.mark.parametrize("scheme", ["neighborhood-uniform", "global-uniform"])
def test_uniform_weights_are_a_valid_mixing_row(scheme: str) -> None:
    edges = _edges("ring", 8)
    # Build the neighbour view directly: these schemes need no communication.
    topology = Topology.__new__(Topology)
    topology.rank, topology.size, topology.edges = 0, 8, edges
    topology.neighbors = Topology._neighbors_of(0, 8, edges)
    topology.weights = topology._compute_weights(scheme)

    assert len(topology.weights) == topology.degree
    assert all(weight > 0 for weight in topology.weights)
    # A row of a stochastic matrix: neighbours plus self must sum to one.
    assert topology.self_weight + sum(topology.weights) == pytest.approx(1.0)
    assert topology.self_weight > 0


def test_neighborhood_uniform_matches_degree() -> None:
    edges = _edges("ring", 6)
    topology = Topology.__new__(Topology)
    topology.rank, topology.size, topology.edges = 0, 6, edges
    topology.neighbors = Topology._neighbors_of(0, 6, edges)
    weights = topology._compute_weights("neighborhood-uniform")
    assert weights == pytest.approx([1 / 3, 1 / 3])
