"""The training loop, run once per MPI rank (one rank = one federated client)."""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
from torch.utils.data import DataLoader

from .comm import build_communicator, synchronize_model
from .config import Config
from .data import build_loaders, prepare_dataset
from .model import build_model
from .recorder import EpochRecord, Recorder
from .topology import Topology, build_edges


def resolve_device(
    preference: str, rank: int, comm: MPI.Comm | None = None
) -> torch.device:
    """Pick this rank's device, spreading ranks across the GPUs on its node.

    GPUs are assigned by rank *within the node*, not by global rank: on a
    multi-node job the global rank can exceed the local GPU count, and two
    ranks on different nodes share a global index only by coincidence.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            preference = "cuda"
        elif torch.backends.mps.is_available():
            preference = "mps"
        else:
            preference = "cpu"

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but no CUDA device is visible")
        gpu = _local_rank(rank, comm) % torch.cuda.device_count()
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but MPS is not available")
    return torch.device(preference)


def _node_placement(rank: int, comm: MPI.Comm | None) -> tuple[int, int]:
    """This rank's index among the ranks sharing its node, and how many there are."""
    if comm is None:
        return rank, 1
    shared = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        return shared.Get_rank(), shared.Get_size()
    finally:
        shared.Free()


def _local_rank(rank: int, comm: MPI.Comm | None) -> int:
    return _node_placement(rank, comm)[0]


def configure_threads(device: torch.device, comm: MPI.Comm | None, rank: int) -> int:
    """Share this node's cores between the ranks running on it.

    PyTorch otherwise gives every rank the whole machine, so n clients on one
    node open n times more threads than there are cores. The contention makes
    the reported computation times measure oversubscription rather than the
    algorithm. An explicit OMP_NUM_THREADS is left alone.
    """
    if device.type != "cpu":
        return torch.get_num_threads()
    if os.environ.get("OMP_NUM_THREADS"):
        return torch.get_num_threads()

    _, ranks_on_node = _node_placement(rank, comm)
    threads = max(1, (os.cpu_count() or 1) // max(ranks_on_node, 1))
    torch.set_num_threads(threads)
    return threads


def seed_everything(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Mean loss and top-1 accuracy over ``loader``."""
    model.eval()
    total_loss = total_correct = total_seen = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item() * targets.size(0)
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_seen += targets.size(0)
    return total_loss / total_seen, 100.0 * total_correct / total_seen


def run(config: Config, comm: MPI.Comm | None = None) -> dict[str, object]:
    """Train one client. Every rank calls this; they differ only by ``rank``."""
    comm = comm or MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # Ranks are seeded apart so their models start from different points; the
    # all-reduce below then agrees on a common initial model.
    seed_everything(config.seed + rank, config.deterministic, config.cudnn_benchmark)
    device = resolve_device(config.device, rank, comm)
    threads = configure_threads(device, comm, rank)

    model = build_model(config.resnet_depth).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    initial_model = synchronize_model(model, size, comm)

    # One rank fetches the dataset; the rest wait rather than racing it.
    if config.download and rank == 0:
        prepare_dataset(config.data_dir)
    comm.Barrier()

    train_loader, eval_loader, test_loader = build_loaders(
        rank=rank,
        num_clients=size,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        non_iid=config.non_iid,
        seed=config.seed,
        eval_subset=config.eval_subset,
        num_workers=config.num_workers,
        download=False,
        # Pinned host memory only helps a device that DMAs from it; asking for
        # it on CPU just warns and costs a copy.
        pin_memory=device.type == "cuda",
    )

    # Defence in depth for the hazard partition_indices avoids: the synchronous
    # communicators run a collective per step, so unequal step counts would
    # desynchronise the job rather than fail cleanly.
    steps_per_epoch = len(train_loader)
    if comm.allreduce(steps_per_epoch, op=MPI.MIN) != comm.allreduce(
        steps_per_epoch, op=MPI.MAX
    ):
        raise RuntimeError(
            f"clients disagree on steps per epoch (rank {rank} has "
            f"{steps_per_epoch}); every client must take the same number of steps"
        )

    edges = build_edges(
        config.topology,
        size,
        comm,
        rank,
        clusters=config.clusters,
        edge_prob=config.edge_prob,
        seed=config.seed,
    )
    topology = Topology(rank, size, comm, edges, scheme=config.weights)
    communicator = build_communicator(
        config, rank, size, comm, topology, initial_model
    )

    recorder = Recorder(config, rank)
    if rank == 0 and config.num_workers:
        print(
            f"[swift] warning: --num-workers {config.num_workers} forks worker "
            f"processes after MPI has initialised, which MPI implementations "
            f"document as unsafe; it hangs with Open MPI on macOS. Use 0 if the "
            f"run stalls before the first epoch.",
            flush=True,
        )
    if rank == 0:
        config.run_dir.mkdir(parents=True, exist_ok=True)
        config.save(config.run_dir / "config.json")
        print(
            f"[swift] {config.name}: {config.algorithm} on {size} clients, "
            f"{config.topology} topology, ResNet-{config.resnet_depth}, "
            f"device={device.type}"
            + (f", {threads} threads/client" if device.type == "cpu" else ""),
            flush=True,
        )
    comm.Barrier()

    for epoch in range(config.epochs):
        lr = config.lr_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        epoch_start = time.perf_counter()
        sync_before = communicator.sync_time
        comp_time = comm_time = bookkeeping = 0.0
        loss_sum = correct = seen = 0.0
        model.train()

        for inputs, targets in train_loader:
            step_start = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Reading the loss synchronises on the forward pass, so this block
            # is timed and removed from comp_time below rather than counted as
            # part of the algorithm's cost.
            bookkeeping_start = time.perf_counter()
            batch = targets.size(0)
            loss_sum += loss.item() * batch
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            seen += batch
            bookkeeping += time.perf_counter() - bookkeeping_start

            loss.backward()

            communication_start = time.perf_counter()
            comm_time += communicator.communicate(model)
            communication_elapsed = time.perf_counter() - communication_start

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_time = time.perf_counter() - step_start - communication_elapsed
            comp_time += step_time

            if config.slowdown > 1.0 and rank == 0:
                # Artificial straggler (Section 6.2): stretch this client's
                # computation without counting the delay as work.
                time.sleep((config.slowdown - 1.0) * step_time)

        if not seen:
            raise RuntimeError(
                f"rank {rank} was assigned no training data: {size} clients is "
                f"too many for this dataset"
            )

        comp_time -= bookkeeping
        wall_time = time.perf_counter() - epoch_start
        test_loss, _ = evaluate(model, eval_loader, criterion, device)

        recorder.add(
            EpochRecord(
                epoch=epoch,
                lr=lr,
                train_loss=loss_sum / seen,
                train_acc=100.0 * correct / seen,
                test_loss=test_loss,
                comp_time=comp_time,
                comm_time=comm_time,
                sync_time=communicator.sync_time - sync_before,
                epoch_time=comp_time + comm_time,
                wall_time=wall_time,
            )
        )
        if rank == 0:
            print(
                f"[swift] epoch {epoch + 1}/{config.epochs} "
                f"lr {lr:.4g} train_loss {loss_sum / seen:.4f} "
                f"train_acc {100.0 * correct / seen:.2f} test_loss {test_loss:.4f} "
                f"comp {comp_time:.2f}s comm {comm_time:.2f}s",
                flush=True,
            )

    communicator.finish(model)
    comm.Barrier()

    # The reported number is the accuracy of the consensus model: average every
    # client's parameters, then score once on the full test set.
    synchronize_model(model, size, comm)
    consensus_loss, consensus_acc = evaluate(model, test_loader, criterion, device)

    summary: dict[str, object] = {
        "consensus_accuracy": consensus_acc,
        "consensus_loss": consensus_loss,
    }
    summary |= getattr(communicator, "stats", {})
    recorder.save(summary)

    if rank == 0:
        print(
            f"[swift] consensus model: accuracy {consensus_acc:.2f}% "
            f"loss {consensus_loss:.4f} -> {config.run_dir}",
            flush=True,
        )
    return summary
