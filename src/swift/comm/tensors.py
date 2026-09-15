"""Flatten and unflatten model parameters into a single contiguous buffer.

Every communicator sends one flat vector per round rather than one message per
layer, which is what makes a model exchange a single MPI operation.

Adapted from https://github.com/facebookresearch/stochastic_gradient_push
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate dense tensors into one 1-D buffer."""
    if len(tensors) == 1:
        return tensors[0].reshape(-1).clone()
    return torch.cat([t.reshape(-1) for t in tensors], dim=0)


def unflatten_tensors(
    flat: torch.Tensor, tensors: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Split a flat buffer back into views shaped like ``tensors``."""
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def parameter_list(model: torch.nn.Module) -> list[torch.Tensor]:
    """The parameters a communicator averages, in a stable order."""
    return list(model.parameters())


def copy_into_model(flat: torch.Tensor, tensors: Iterable[torch.Tensor]) -> None:
    """Write a flat buffer back into the model's parameters in place."""
    tensors = list(tensors)
    with torch.no_grad():
        for source, target in zip(unflatten_tensors(flat, tensors), tensors, strict=True):
            target.copy_(source)
