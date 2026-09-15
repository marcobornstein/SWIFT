"""Check that the 2.0 communicators still compute what the 1.0 ones did.

Runs the original ICLR-2023 classes and the rewritten ones side by side on
identical inputs under the same MPI ranks, and compares the resulting models.
Unpack the 1.0 source tree somewhere and point SWIFT_V1_PATH at it::

    SWIFT_V1_PATH=../swift-v1 TOPOLOGY=clique-ring CLUSTERS=3 \
        mpirun -np 10 --oversubscribe python tools/verify_against_v1.py

Environment: SWIFT_V1_PATH (the 1.0 tree), TOPOLOGY, CLUSTERS.

Two details make the comparison fair rather than flattering:

* Each pairing gets freshly duplicated communicators. Both versions use the same
  tag scheme, 1.0 leaves unclaimed CCS proposals queued, and its Sendrecv has no
  receive tag -- so on a shared communicator one version's stale weight messages
  get matched as the other's model.
* Both sides are driven broadcast -> barrier -> average, so each sees the same
  set of arrivals and any difference is arithmetic rather than a race.

Agreement is to a few ulp rather than bitwise: 1.0 summed each neighbourhood in
edge order and 2.0 sorts for determinism, which reorders a float32 sum.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI

ORIG = os.path.abspath(os.environ.get("SWIFT_V1_PATH", "swift-v1"))
sys.path.insert(0, ORIG)

# The original code is GPU-only; make .cuda() a no-op so it runs on CPU.
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self

from Communicators.AsyncCommunicator import AsyncDecentralized as OldSwift  # noqa: E402
from Communicators.DSGD import decenCommunicator as OldSync  # noqa: E402
from GDM.GraphConstruct import GraphConstruct as OldGraph  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from swift.comm import (  # noqa: E402
    SwiftCommunicator,
    SyncCommunicator,
    flatten_tensors,
    parameter_list,
)
from swift.topology import Topology, build_edges  # noqa: E402

C = MPI.COMM_WORLD
R, S = C.Get_rank(), C.Get_size()
DIM, ROUNDS = 64, 6
TOPOLOGY = os.environ.get("TOPOLOGY", "ring")
CLUSTERS = int(os.environ.get("CLUSTERS", "1"))


def fresh_model(seed):
    torch.manual_seed(seed)
    model = nn.Linear(DIM, 4, bias=False)
    with torch.no_grad():
        model.weight.add_(float(R))     # a distinct model per rank
    return model


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().copy()


# float32 carries ~7 significant digits, and the two implementations sum each
# neighbourhood in a different order (the original used edge order, the rewrite
# sorts for determinism), so agreement is expected to a few ulp, not bitwise.
TOLERANCE_ULP = 8.0


def report(label, old_vals, new_vals):
    identical = np.array_equal(old_vals, new_vals)
    delta = float(np.max(np.abs(old_vals - new_vals)))
    scale = float(np.max(np.abs(old_vals))) or 1.0
    ulps = delta / (scale * np.finfo(np.float32).eps)
    close = ulps <= TOLERANCE_ULP
    rows = C.gather((R, label, identical, close, delta, ulps), root=0)
    if R == 0:
        for rank, lab, ident, ok, d, u in rows:
            verdict = "IDENTICAL" if ident else ("MATCH" if ok else "DIFFERS")
            print(f"  rank {rank:<3d} {lab:<16s} {verdict:<10s} "
                  f"max|old-new| = {d:.3e}  ({u:.2f} ulp)")
    return close


def topologies(scheme_old, scheme_new):
    """Fresh communicators, then the same graph built by each implementation."""
    old_comm, new_comm = C.Dup(), C.Dup()
    old = OldGraph(R, S, old_comm, TOPOLOGY, scheme_old, num_c=CLUSTERS)
    C.Barrier()
    new = Topology(R, S, new_comm,
                   build_edges(TOPOLOGY, S, new_comm, R, clusters=CLUSTERS),
                   scheme=scheme_new)
    C.Barrier()
    return old, old_comm, new, new_comm


ok = True
if R == 0:
    print(f"\n=== old vs new: {S} ranks, {TOPOLOGY}"
          f"{'-' + str(CLUSTERS) + 'C' if CLUSTERS > 1 else ''}, {ROUNDS} rounds ===")

# ------------------------------------------------------------------- weights
for scheme_old, scheme_new in (("uniform", "neighborhood-uniform"), ("swift", "ccs")):
    old_t, _, new_t, _ = topologies(scheme_old, scheme_new)
    if R == 0:
        print(f"\n[weights: {scheme_old} -> {scheme_new}]")
    assert sorted(old_t.neighbor_list) == sorted(new_t.neighbors)
    ok &= report("neighbor weights",
                 np.asarray(old_t.neighbor_weights, dtype=np.float64),
                 np.asarray(new_t.weights, dtype=np.float64))

# --------------------------------------------------------- sync communicators
for label, i1, i2 in (("d-sgd", 0, 1), ("pa-sgd", 1, 1), ("ld-sgd", 1, 2)):
    old_t, old_comm, new_t, new_comm = topologies("uniform", "neighborhood-uniform")
    if R == 0:
        print(f"\n[{label}]")
    old_model, new_model = fresh_model(7), fresh_model(7)
    old_c = OldSync(R, S, old_comm, old_t, i1, i2)
    new_c = SyncCommunicator(R, S, new_comm, new_t, i1, i2)
    for step in range(ROUNDS):
        with torch.no_grad():                   # stand-in for a gradient step
            old_model.weight.add_(0.01 * (step + 1))
            new_model.weight.add_(0.01 * (step + 1))
        old_c.communicate(old_model)
        C.Barrier()
        new_c.communicate(new_model)
        C.Barrier()
    ok &= report("model", flat(old_model), flat(new_model))

# -------------------------------------------------------- swift communicator
for label, boost, low_mem in (("stored", 1, 0), ("low-memory", 1, 1),
                              ("low-mem no-boost", 0, 1)):
    old_t, old_comm, new_t, new_comm = topologies("swift", "ccs")
    if R == 0:
        print(f"\n[swift {label}]")
    old_model, new_model = fresh_model(7), fresh_model(7)
    initial = flat(old_model)
    old_c = OldSwift(R, S, old_comm, old_t, 1, 10, boost, low_mem, initial)
    new_c = SwiftCommunicator(R, S, new_comm, new_t, initial, local_steps=1,
                              weight_boost=bool(boost), low_memory=bool(low_mem))
    for step in range(ROUNDS):
        with torch.no_grad():
            old_model.weight.add_(0.01 * (step + 1))
            new_model.weight.add_(0.01 * (step + 1))
        # Both implementations are driven broadcast -> barrier -> average, so
        # each sees every neighbour's message and the comparison is arithmetic
        # rather than a race. communicate() does not barrier, so the new side
        # is stepped through its two halves by hand.
        old_c.broadcast(old_model)
        new_params = parameter_list(new_model)
        new_flat = flatten_tensors(new_params).detach().cpu()
        new_c._broadcast(new_flat.numpy())
        C.Barrier()

        if low_mem:
            old_c.averaging_efficient(old_model)
        else:
            old_c.averaging_standard(old_model)
        new_c._average(new_model, new_params, new_flat)
        C.Barrier()
    ok &= report("model", flat(old_model), flat(new_model))
    new_c.finish(new_model)
    C.Barrier()

verdict = C.allreduce(ok, op=MPI.LAND)
if R == 0:
    print(f"\n=== {'ALL MATCH' if verdict else 'MISMATCH FOUND'} ===\n")
sys.exit(0 if verdict else 1)
