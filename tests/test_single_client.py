"""A one-client run has no neighbours, which is the degenerate case every
communicator has to survive. It needs no extra ranks, so it runs under plain
pytest against MPI.COMM_SELF."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from mpi4py import MPI

from swift.comm import build_communicator, synchronize_model
from swift.config import Config
from swift.topology import Topology, build_edges

COMM = MPI.COMM_SELF


def _model(fill: float = 3.0) -> nn.Module:
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(fill)
    return model


def _flat(model: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().copy()


@pytest.mark.parametrize("topology", ["ring", "clique-ring", "fully-connected"])
def test_single_client_topology_has_no_neighbours(topology: str) -> None:
    edges = build_edges(topology, 1, COMM, 0, clusters=1)
    assert edges == []

    graph = Topology(0, 1, COMM, edges, scheme="ccs")
    assert graph.neighbors == []
    assert graph.degree == 0
    assert len(graph.weights) == 0
    # It keeps all its own weight, so averaging is the identity.
    assert graph.self_weight == pytest.approx(1.0)


@pytest.mark.parametrize("algorithm", ["swift", "d-sgd", "pa-sgd", "ld-sgd"])
def test_single_client_training_step_is_a_no_op(algorithm: str) -> None:
    """With nobody to average against, communicate() must leave the model alone."""
    model = _model()
    config = Config(algorithm=algorithm, i1=1, i2=2)
    topology = Topology(0, 1, COMM, [], scheme=config.weights)
    communicator = build_communicator(config, 0, 1, COMM, topology, _flat(model))

    for _ in range(4):
        communicator.communicate(model)
    communicator.finish(model)

    assert np.allclose(_flat(model), 3.0)


def test_single_client_synchronize_is_the_identity() -> None:
    model = _model(1.25)
    agreed = synchronize_model(model, 1, COMM)
    assert np.allclose(agreed, 1.25)
    assert np.allclose(_flat(model), 1.25)


def test_swift_reports_no_traffic_for_a_lone_client() -> None:
    model = _model()
    config = Config(algorithm="swift")
    topology = Topology(0, 1, COMM, [], scheme="ccs")
    communicator = build_communicator(config, 0, 1, COMM, topology, _flat(model))
    communicator.communicate(model)
    communicator.finish(model)
    assert communicator.stats["messages_sent"] == 0
    assert communicator.stats["messages_received"] == 0


def test_threads_are_shared_between_ranks_on_a_node(monkeypatch) -> None:
    """Each rank taking the whole machine makes CPU timings meaningless."""
    import os

    import torch

    from swift.train import configure_threads

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    before = torch.get_num_threads()
    try:
        threads = configure_threads(torch.device("cpu"), COMM, 0)
        # COMM_SELF is one rank, so it gets the whole node.
        assert threads == max(1, os.cpu_count() or 1)
        assert threads >= 1
    finally:
        torch.set_num_threads(before)


def test_explicit_thread_count_is_respected(monkeypatch) -> None:
    import torch

    from swift.train import configure_threads

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    before = torch.get_num_threads()
    try:
        assert configure_threads(torch.device("cpu"), COMM, 0) == before
    finally:
        torch.set_num_threads(before)


def test_non_cpu_devices_are_left_alone(monkeypatch) -> None:
    import torch

    from swift.train import configure_threads

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    before = torch.get_num_threads()
    assert configure_threads(torch.device("meta"), COMM, 0) == before
    assert torch.get_num_threads() == before
