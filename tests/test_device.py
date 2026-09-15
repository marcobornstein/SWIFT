"""Device selection, and that an accelerator computes what the CPU does.

`--device auto` picks MPS on Apple Silicon and CUDA where present, so the first
run a new user makes is often not on the CPU path everything else is tested on.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from swift.model import build_model
from swift.train import resolve_device, seed_everything


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def test_cpu_is_always_available() -> None:
    assert resolve_device("cpu", 0, None).type == "cpu"


def test_auto_picks_an_accelerator_when_there_is_one() -> None:
    chosen = resolve_device("auto", 0, None).type
    assert chosen == (_accelerator() or "cpu")


def test_requesting_an_absent_device_fails_loudly() -> None:
    for name in ("cuda", "mps"):
        available = name == _accelerator()
        if available:
            continue
        with pytest.raises(RuntimeError, match=r"not available|no CUDA device"):
            resolve_device(name, 0, None)


def test_deterministic_disables_cudnn_autotuning() -> None:
    before = torch.backends.cudnn.benchmark
    try:
        seed_everything(1, deterministic=True, cudnn_benchmark=True)
        assert torch.backends.cudnn.benchmark is False
        seed_everything(1, deterministic=False, cudnn_benchmark=True)
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = before


@pytest.mark.skipif(_accelerator() is None, reason="no accelerator available")
def test_accelerator_training_matches_the_cpu() -> None:
    """A few real optimiser steps, compared against the CPU path.

    Guards against an unsupported or subtly wrong kernel silently changing what
    a run computes.
    """
    device = _accelerator()
    torch.manual_seed(0)
    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()

    def train_on(where: str) -> tuple[list[float], torch.Tensor]:
        torch.manual_seed(7)
        model = build_model(18).to(where)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        batch, labels = inputs.to(where), targets.to(where)
        losses = []
        for _ in range(3):
            model.train()
            loss = criterion(model(batch), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        flat = torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])
        return losses, flat

    cpu_losses, cpu_params = train_on("cpu")
    device_losses, device_params = train_on(device)

    # The losses are what the run reports, so hold them tightly.
    assert device_losses == pytest.approx(cpu_losses, rel=1e-4)

    # Weights are compared relative to their own scale: an absolute tolerance
    # says nothing about whether float32 backends have genuinely diverged, and
    # batch-norm plus momentum amplify the last bits over a few steps.
    scale = cpu_params.abs().max().item()
    drift = (cpu_params - device_params).abs().max().item() / scale
    assert drift < 1e-3, f"{device} drifted {drift:.2e} from the CPU in 3 steps"
