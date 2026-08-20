"""Thin Zoology adapter around FLA's official Gated DeltaNet layer.

This module intentionally contains no copied Gated DeltaNet equations.  FLA
owns the projections, ShortConv, decay/gate initialization, recurrent kernel,
normalization, and output projection.  The adapter only translates Zoology's
``d_model`` argument, selects an optional mixed-precision context, unwraps
FLA's tuple return value, and reports the recurrent matrix-state size.

Keeping this adapter separate from ``zoology.mixers.gated_delta_net`` makes it
possible to test whether behavior comes from Zoology's copied implementation
without changing the historical baseline.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch
from torch import nn

from fla.layers.gated_deltanet import GatedDeltaNet as FLAGatedDeltaNet


class DirectFLAGatedDeltaNet(nn.Module):
    """Expose FLA's GatedDeltaNet through Zoology's sequence-mixer contract.

    ``autocast_dtype="bfloat16"`` runs the complete FLA layer under CUDA BF16
    autocast.  This differs deliberately from Zoology's historical wrapper,
    which performs the linear projections in FP32 and manually casts only the
    recurrent-kernel inputs.  Set ``autocast_dtype="none"`` to call FLA
    without an adapter-owned autocast context.
    """

    _SUPPORTED_AUTOCAST_DTYPES = {"none", "bfloat16"}

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        head_dim: int | None = None,
        expand_v: float = 2,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        autocast_dtype: str = "bfloat16",
        layer_idx: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive; got {d_model}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive; got {num_heads}.")
        if head_dim is None:
            if d_model % num_heads != 0:
                raise ValueError(
                    "head_dim was omitted, but d_model is not divisible by "
                    f"num_heads: {d_model} % {num_heads} != 0."
                )
            head_dim = d_model // num_heads
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive; got {head_dim}.")
        if expand_v <= 0:
            raise ValueError(f"expand_v must be positive; got {expand_v}.")
        if autocast_dtype not in self._SUPPORTED_AUTOCAST_DTYPES:
            raise ValueError(
                "autocast_dtype must be one of "
                f"{sorted(self._SUPPORTED_AUTOCAST_DTYPES)}; "
                f"got {autocast_dtype!r}."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.head_v_dim = int(head_dim * expand_v)
        if self.head_v_dim != head_dim * expand_v:
            raise ValueError(
                "head_dim * expand_v must be an integer because it defines "
                f"the per-head value dimension; got {head_dim} * {expand_v}."
            )
        self.autocast_dtype = autocast_dtype

        # Do not reproduce or modify the GDN internals here.  This direct FLA
        # construction is the experimental control being tested.
        self.fla_layer = FLAGatedDeltaNet(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            mode=mode,
            use_gate=use_gate,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            layer_idx=layer_idx,
            norm_eps=norm_eps,
            **kwargs,
        )

    def _autocast_context(self, hidden_states: torch.Tensor) -> ContextManager:
        if self.autocast_dtype == "none" or hidden_states.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        with self._autocast_context(hidden_states):
            outputs = self.fla_layer(hidden_states=hidden_states, **kwargs)

        if not isinstance(outputs, tuple) or not outputs:
            raise TypeError(
                "FLA GatedDeltaNet was expected to return a non-empty tuple; "
                f"got {type(outputs).__name__}."
            )
        mixed_hidden_states = outputs[0]
        if not isinstance(mixed_hidden_states, torch.Tensor):
            raise TypeError(
                "FLA GatedDeltaNet tuple element 0 must be a Tensor; got "
                f"{type(mixed_hidden_states).__name__}."
            )
        return mixed_hidden_states

    def state_size(self, sequence_length: int = 2048) -> int:
        """Return recurrent matrix-state scalars per layer (batch excluded)."""
        del sequence_length
        return self.num_heads * self.head_dim * self.head_v_dim

