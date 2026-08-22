"""Thin Zoology adapter around FLA's official Kimi Delta Attention layer.

KDA entered FLA in release v0.4.0.  The adapter deliberately contains no KDA
equations: FLA owns the projections, per-key-dimension decay gate, ShortConv,
delta-rule recurrence, sigmoid output gate, normalization, and initialization.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from zoology.mixers.direct_fla_utils import (
    autocast_context,
    unwrap_fla_output,
    validate_autocast_dtype,
)


class DirectFLAKimiDeltaAttention(nn.Module):
    """Expose FLA KimiDeltaAttention through Zoology's mixer contract."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        head_dim: int | None = None,
        num_v_heads: int | None = None,
        expand_v: float = 2,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        autocast_dtype: str = "bfloat16",
        layer_idx: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        try:
            from fla.layers.kda import KimiDeltaAttention as FLAKimiDeltaAttention
        except ModuleNotFoundError as error:
            if error.name not in {"fla", "fla.layers", "fla.layers.kda"}:
                raise
            raise ImportError(
                "DirectFLAKimiDeltaAttention requires "
                "flash-linear-attention>=0.4.0; the installed FLA package "
                "does not provide fla.layers.kda.KimiDeltaAttention."
            ) from error

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
        if num_heads * head_dim != d_model:
            raise ValueError(
                "This comparison adapter requires num_heads * head_dim == "
                f"d_model; got {num_heads} * {head_dim} != {d_model}."
            )

        if num_v_heads is None:
            num_v_heads = num_heads
        if num_v_heads <= 0:
            raise ValueError(
                f"num_v_heads must be positive; got {num_v_heads}."
            )
        if num_v_heads > num_heads and num_v_heads % num_heads != 0:
            raise ValueError(
                "When num_v_heads exceeds num_heads it must be divisible by "
                f"num_heads; got {num_v_heads} and {num_heads}."
            )
        if expand_v <= 0:
            raise ValueError(f"expand_v must be positive; got {expand_v}.")
        head_v_dim = int(head_dim * expand_v)
        if not math.isclose(head_dim * expand_v, head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"head_dim * expand_v must be integral; got "
                f"{head_dim * expand_v}."
            )
        validate_autocast_dtype(autocast_dtype)

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.head_dim = head_dim
        self.head_v_dim = head_v_dim
        self.expand_v = expand_v
        self.autocast_dtype = autocast_dtype

        self.fla_layer = FLAKimiDeltaAttention(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            mode=mode,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            allow_neg_eigval=allow_neg_eigval,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
            **kwargs,
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast_context(hidden_states, self.autocast_dtype):
            outputs = self.fla_layer(hidden_states=hidden_states, **kwargs)
        return unwrap_fla_output(outputs, "KimiDeltaAttention")

    def state_size(self, sequence_length: int = 2048) -> int:
        """Return recurrent matrix-state scalars per layer (batch excluded)."""
        del sequence_length
        return self.num_v_heads * self.head_dim * self.head_v_dim
