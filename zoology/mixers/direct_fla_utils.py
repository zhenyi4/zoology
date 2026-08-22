"""Shared utilities for thin adapters around official FLA layers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch
from torch import nn

from fla.modules import RMSNorm


SUPPORTED_AUTOCAST_DTYPES = {"none", "bfloat16"}


class RMSNormSigmoidGate(nn.Module):
    """Apply FLA's head-wise RMSNorm followed by a sigmoid output gate."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(hidden_states) * torch.sigmoid(gate)


def validate_autocast_dtype(autocast_dtype: str) -> None:
    if autocast_dtype not in SUPPORTED_AUTOCAST_DTYPES:
        raise ValueError(
            "autocast_dtype must be one of "
            f"{sorted(SUPPORTED_AUTOCAST_DTYPES)}; got {autocast_dtype!r}."
        )


def autocast_context(
    hidden_states: torch.Tensor,
    autocast_dtype: str,
) -> ContextManager:
    if autocast_dtype == "none" or hidden_states.device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def unwrap_fla_output(outputs: object, layer_name: str) -> torch.Tensor:
    """Return element zero from FLA's ``(hidden, attention, cache)`` tuple."""
    if not isinstance(outputs, tuple) or not outputs:
        raise TypeError(
            f"FLA {layer_name} was expected to return a non-empty tuple; "
            f"got {type(outputs).__name__}."
        )
    hidden_states = outputs[0]
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(
            f"FLA {layer_name} tuple element 0 must be a Tensor; got "
            f"{type(hidden_states).__name__}."
        )
    return hidden_states
