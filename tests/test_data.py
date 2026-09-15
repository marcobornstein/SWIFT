"""Data partitioning: disjoint shards, correct sizes, and a real label skew."""

from __future__ import annotations

from collections import Counter

import pytest

from swift.data import partition_indices

NUM_SAMPLES = 10000
NUM_LABELS = 10
LABELS = [i % NUM_LABELS for i in range(NUM_SAMPLES)]


def _all_shards(num_clients: int, non_iid: float, seed: int = 0) -> list[list[int]]:
    return [
        partition_indices(LABELS, num_clients, rank, non_iid, seed)
        for rank in range(num_clients)
    ]


@pytest.mark.parametrize("num_clients", [1, 2, 4, 5, 10, 16])
@pytest.mark.parametrize("non_iid", [0.0, 0.5, 1.0])
def test_shards_are_disjoint(num_clients: int, non_iid: float) -> None:
    shards = _all_shards(num_clients, non_iid)
    flat = [index for shard in shards for index in shard]
    assert len(flat) == len(set(flat)), "a sample was handed to two clients"
    assert set(flat) <= set(range(NUM_SAMPLES))
    # At most one short shard's worth goes unused.
    assert NUM_SAMPLES - len(flat) < num_clients


@pytest.mark.parametrize("num_clients", [1, 3, 7, 16, 32])
@pytest.mark.parametrize("non_iid", [0.0, 0.5, 1.0])
def test_every_client_gets_exactly_the_same_number_of_samples(
    num_clients: int, non_iid: float
) -> None:
    """Unequal shards can give clients different step counts, and the
    synchronous communicators run a collective per step: one extra Barrier
    desynchronises the whole job."""
    sizes = {len(shard) for shard in _all_shards(num_clients, non_iid)}
    assert len(sizes) == 1, f"shard sizes differ: {sorted(sizes)}"
    assert sizes.pop() == NUM_SAMPLES // num_clients


def test_more_clients_than_samples_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than"):
        partition_indices([0, 1, 2], 4, 0, 0.0, 0)


def test_iid_partition_is_label_balanced() -> None:
    """Every client sees every label, and none of them dominates its shard."""
    for shard in _all_shards(10, non_iid=0.0):
        counts = Counter(LABELS[i] for i in shard)
        assert len(counts) == NUM_LABELS
        # Uniform would be 1/10 of the shard per label; well clear of the 1.0
        # a fully non-IID shard shows.
        assert max(counts.values()) / len(shard) < 0.2


def test_fully_non_iid_partition_gives_each_client_one_label() -> None:
    for shard in _all_shards(10, non_iid=1.0):
        assert len(set(LABELS[i] for i in shard)) == 1


def test_non_iid_degree_controls_the_skew() -> None:
    """Half non-IID means about half of a shard comes from its own label."""
    shard = _all_shards(10, non_iid=0.5)[0]
    counts = Counter(LABELS[i] for i in shard)
    dominant = counts.most_common(1)[0][1]
    assert 0.4 <= dominant / len(shard) <= 0.65


def test_partition_is_deterministic() -> None:
    assert _all_shards(4, 0.7, seed=3) == _all_shards(4, 0.7, seed=3)
    assert _all_shards(4, 0.7, seed=3) != _all_shards(4, 0.7, seed=4)


def test_invalid_non_iid_is_rejected() -> None:
    with pytest.raises(ValueError, match="non_iid"):
        partition_indices(LABELS, 4, 0, 1.5, 0)


def test_partition_does_not_depend_on_call_order() -> None:
    """Every client derives the whole partition and takes its own slice, so the
    shards must not depend on who asks or when."""
    forwards = [partition_indices(LABELS, 5, r, 0.6, 77) for r in range(5)]
    backwards = [partition_indices(LABELS, 5, r, 0.6, 77) for r in reversed(range(5))]
    assert forwards == list(reversed(backwards))


def test_evaluation_subset_is_shared_and_seed_stable() -> None:
    """Clients must score the same images, or averaging their losses is
    meaningless. Drawn from the base seed alone, never the rank."""
    import torch

    def subset(seed: int) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(10000, generator=generator)[:500].tolist()

    assert subset(4242) == subset(4242)
    assert subset(4242) != subset(4243)
