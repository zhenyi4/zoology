"""Thin Zoology adapter around FLA's official DeltaNet layer.

FLA owns all projections, ShortConv operations, delta-rule recurrence,
normalization, and initialization.  This adapter only translates Zoology's
``d_model`` convention, optionally selects the sigmoid output gate used by the
local KDA comparison, controls CUDA autocast, unwraps FLA's tuple return, and
reports recurrent-state size.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from fla.layers.delta_net import DeltaNet as FLADeltaNet
from zoology.mixers.direct_fla_utils import (
    RMSNormSigmoidGate,
    autocast_context,
    unwrap_fla_output,
    validate_autocast_dtype,
)


class DirectFLADeltaNet(nn.Module):
    """Expose FLA DeltaNet through Zoology's sequence-mixer contract."""

    _SUPPORTED_OUTPUT_GATE_ACTIVATIONS = {"swish", "sigmoid"}

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        head_dim: int | None = None,
        expand_k: float = 1,
        expand_v: float = 2,
        mode: str = "chunk",
        use_beta: bool = True,
        use_gate: bool = True,
        output_gate_activation: str = "sigmoid",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
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
        if expand_k <= 0 or expand_v <= 0:
            raise ValueError(
                "expand_k and expand_v must be positive; got "
                f"expand_k={expand_k}, expand_v={expand_v}."
            )

        key_dim = int(d_model * expand_k)
        value_dim = int(d_model * expand_v)
        if not math.isclose(d_model * expand_k, key_dim, rel_tol=1e-5):
            raise ValueError(
                f"d_model * expand_k must be integral; got {d_model * expand_k}."
            )
        if not math.isclose(d_model * expand_v, value_dim, rel_tol=1e-5):
            raise ValueError(
                f"d_model * expand_v must be integral; got {d_model * expand_v}."
            )
        if key_dim % num_heads != 0 or value_dim % num_heads != 0:
            raise ValueError(
                "DeltaNet key/value dimensions must be divisible by num_heads; "
                f"got key_dim={key_dim}, value_dim={value_dim}, "
                f"num_heads={num_heads}."
            )

        inferred_head_dim = key_dim // num_heads
        if head_dim is None:
            head_dim = inferred_head_dim
        elif head_dim != inferred_head_dim:
            raise ValueError(
                "DeltaNet head_dim is determined by d_model * expand_k / "
                f"num_heads; expected {inferred_head_dim}, got {head_dim}."
            )
        if output_gate_activation not in self._SUPPORTED_OUTPUT_GATE_ACTIVATIONS:
            raise ValueError(
                "output_gate_activation must be one of "
                f"{sorted(self._SUPPORTED_OUTPUT_GATE_ACTIVATIONS)}; "
                f"got {output_gate_activation!r}."
            )
        if not use_gate and output_gate_activation != "swish":
            raise ValueError(
                "output_gate_activation only applies when use_gate=True; "
                f"got use_gate={use_gate} and "
                f"output_gate_activation={output_gate_activation!r}."
            )
        validate_autocast_dtype(autocast_dtype)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_v_dim = value_dim // num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.autocast_dtype = autocast_dtype
        self.output_gate_activation = output_gate_activation

        self.fla_layer = FLADeltaNet(
            hidden_size=d_model,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            mode=mode,
            use_beta=use_beta,
            use_gate=use_gate,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            allow_neg_eigval=allow_neg_eigval,
            qk_activation=qk_activation,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
            **kwargs,
        )
        if use_gate and output_gate_activation == "sigmoid":
            self.fla_layer.o_norm = RMSNormSigmoidGate(
                self.head_v_dim,
                eps=norm_eps,
            )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast_context(hidden_states, self.autocast_dtype):
            outputs = self.fla_layer(hidden_states=hidden_states, **kwargs)
        return unwrap_fla_output(outputs, "DeltaNet")

    def state_size(self, sequence_length: int = 2048) -> int:
        """Return recurrent matrix-state scalars per layer (batch excluded)."""
        del sequence_length
        return self.num_heads * self.head_dim * self.head_v_dim
