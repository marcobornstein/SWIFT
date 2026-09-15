"""CIFAR-10 loading and the per-client partitioning described in Appendix A.2.

``non_iid`` controls what fraction of a client's shard is drawn from a small set
of labels assigned to it; the rest is sampled IID across all labels. At 0 the
partition is fully IID, at 1 each client sees only its own labels.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _transforms(train: bool) -> transforms.Compose:
    augment = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if train
        else []
    )
    return transforms.Compose(
        [*augment, transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def partition_indices(
    labels: list[int], num_clients: int, rank: int, non_iid: float, seed: int
) -> list[int]:
    """Indices assigned to ``rank`` under a ``non_iid``-degree label skew.

    Deterministic given ``(labels, num_clients, non_iid, seed)``: every rank
    derives the same global partition and returns its own slice of it, so the
    shards are disjoint without any communication.
    """
    if not 0.0 <= non_iid <= 1.0:
        raise ValueError(f"non_iid must be in [0, 1], got {non_iid}")

    generator = torch.Generator().manual_seed(seed)
    num_samples = len(labels)

    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)
    # Sort so the bins are in label order on every rank, independent of the
    # order the dataset happens to yield labels in.
    label_bins = [by_label[label] for label in sorted(by_label)]
    num_labels = len(label_bins)

    # Every client gets exactly the same number of samples, as Appendix A.2
    # specifies. The remainder (fewer than num_clients samples) goes unused.
    #
    # This is not only cosmetic. Shards differing by one sample can give
    # clients different numbers of batches, and the synchronous communicators
    # call a collective per step -- so one client would run an extra Barrier,
    # silently desynchronising the job and eventually hanging it.
    shard = num_samples // num_clients
    if shard == 0:
        raise ValueError(
            f"{num_clients} clients is more than the {num_samples} samples available"
        )
    shard_sizes = [shard] * num_clients

    # Each shard is part label-skewed and part IID.
    skew_sizes = [math.ceil(size * non_iid) for size in shard_sizes]
    iid_sizes = [size - skew for size, skew in zip(shard_sizes, skew_sizes, strict=True)]

    # Reserve the skewed portion of every label bin, shuffling first so the
    # choice of which samples are skewed is not an artefact of dataset order.
    total_skew = sum(skew_sizes)
    per_label_skew = _split_evenly(total_skew, [len(b) for b in label_bins])
    skew_pool: list[list[int]] = []
    iid_pool: list[int] = []
    for bin_indices, take in zip(label_bins, per_label_skew, strict=True):
        shuffled = _shuffle(bin_indices, generator)
        skew_pool.append(shuffled[:take])
        iid_pool.extend(shuffled[take:])
    iid_pool = _shuffle(iid_pool, generator)

    # Deal the skewed samples out label-by-label, cycling through the bins, so
    # consecutive clients get consecutive labels (Appendix A.2, step 2).
    selected: list[int] = []
    current_bin = 0
    for client in range(num_clients):
        wanted = skew_sizes[client]
        mine: list[int] = []
        # ceil(size * non_iid) <= size for non_iid <= 1, so the pool always
        # holds enough; the counter only guards against a future change making
        # that untrue, which would otherwise spin here forever.
        attempts = 0
        while wanted > 0:
            if attempts > num_labels:
                raise AssertionError(
                    f"skew pool exhausted with {wanted} samples still to place "
                    f"for client {client}"
                )
            available = skew_pool[current_bin]
            take = min(wanted, len(available))
            if take:
                mine.extend(available[len(available) - take :])
                del available[len(available) - take :]
                wanted -= take
                attempts = 0
            else:
                attempts += 1
            current_bin = (current_bin + 1) % num_labels
        if client == rank:
            selected = mine
            break

    iid_start = sum(iid_sizes[:rank])
    selected.extend(iid_pool[iid_start : iid_start + iid_sizes[rank]])
    return selected


def _split_evenly(total: int, capacities: list[int]) -> list[int]:
    """Spread ``total`` items over bins as evenly as their capacities allow."""
    n = len(capacities)
    base, remainder = divmod(total, n)
    take = [min(base + (1 if i < remainder else 0), capacities[i]) for i in range(n)]
    # Push any overflow from small bins onto bins with room left.
    shortfall = total - sum(take)
    i = 0
    while shortfall > 0:
        if take[i] < capacities[i]:
            take[i] += 1
            shortfall -= 1
        i = (i + 1) % n
    return take


def _shuffle(items: list[int], generator: torch.Generator) -> list[int]:
    order = torch.randperm(len(items), generator=generator).tolist()
    return [items[i] for i in order]


def prepare_dataset(data_dir: Path) -> None:
    """Fetch CIFAR-10 once.

    Call this from a single rank before the others build their loaders:
    torchvision writes to a fixed path, so concurrent downloads corrupt each
    other.
    """
    datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    datasets.CIFAR10(root=str(data_dir), train=False, download=True)


def build_loaders(
    rank: int,
    num_clients: int,
    data_dir: Path,
    batch_size: int,
    non_iid: float,
    seed: int,
    eval_subset: int = 500,
    num_workers: int = 0,
    download: bool = False,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Training loader for this client, plus shared eval and full-test loaders.

    The eval loader is a fixed subset used for the per-epoch test loss; it is
    drawn from ``seed`` alone so every client scores the same images and the
    average across clients is meaningful. The test loader is the full test set,
    used once at the end for the consensus model.
    """
    train_set = datasets.CIFAR10(
        root=str(data_dir), train=True, download=download, transform=_transforms(True)
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir), train=False, download=download, transform=_transforms(False)
    )

    indices = partition_indices(
        list(train_set.targets), num_clients, rank, non_iid, seed
    )

    # Seeded per rank so clients shuffle their own shard differently.
    shuffle_generator = torch.Generator().manual_seed(seed + rank)
    train_loader = DataLoader(
        Subset(train_set, indices),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        generator=shuffle_generator,
    )

    eval_set: Dataset = test_set
    if eval_subset:
        subset_generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(test_set), generator=subset_generator)
        eval_set = Subset(test_set, order[:eval_subset].tolist())

    eval_loader = DataLoader(
        eval_set, batch_size=256, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=num_workers
    )
    return train_loader, eval_loader, test_loader
