"""Thin Zoology adapter around FLA's official Gated DeltaNet layer.

This module intentionally contains no copied Gated DeltaNet equations.  FLA
owns the projections, ShortConv, decay/gate initialization, recurrent kernel,
normalization, and output projection.  The adapter translates Zoology's
``d_model`` argument, optionally substitutes the paper-specified sigmoid
output-gate nonlinearity, selects a mixed-precision context, unwraps FLA's
tuple return value, and reports the recurrent matrix-state size.

Keeping this adapter separate from ``zoology.mixers.gated_delta_net`` makes it
possible to test whether behavior comes from Zoology's copied implementation
without changing the historical baseline.
"""

from __future__ import annotations

import torch
from torch import nn

from fla.layers.gated_deltanet import GatedDeltaNet as FLAGatedDeltaNet
from zoology.mixers.direct_fla_utils import (
    RMSNormSigmoidGate,
    autocast_context,
    unwrap_fla_output,
    validate_autocast_dtype,
)


class DirectFLAGatedDeltaNet(nn.Module):
    """Expose FLA's GatedDeltaNet through Zoology's sequence-mixer contract.

    ``autocast_dtype="bfloat16"`` runs the complete FLA layer under CUDA BF16
    autocast.  This differs deliberately from Zoology's historical wrapper,
    which performs the linear projections in FP32 and manually casts only the
    recurrent-kernel inputs.  Set ``autocast_dtype="none"`` to call FLA
    without an adapter-owned autocast context.
    """

    _SUPPORTED_OUTPUT_GATE_ACTIVATIONS = {"swish", "sigmoid"}

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        head_dim: int | None = None,
        expand_v: float = 2,
        mode: str = "chunk",
        use_gate: bool = True,
        output_gate_activation: str = "swish",
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
        if not float(expand_v).is_integer():
            raise ValueError(
                "DirectFLAGatedDeltaNet requires an integer expand_v for "
                "compatibility with the vendored FLA GDN layer; got "
                f"{expand_v}."
            )
        validate_autocast_dtype(autocast_dtype)
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

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Older FLA snapshots multiply key_dim by expand_v without converting
        # the result to int before passing it to nn.Linear. Normalize values
        # such as 1.0 and 2.0 here so out_features never becomes 256.0/512.0.
        self.expand_v = int(expand_v)
        self.head_v_dim = head_dim * self.expand_v
        self.autocast_dtype = autocast_dtype
        self.output_gate_activation = output_gate_activation

        # Keep the GDN projections, recurrence, and initialization in FLA. The
        # optional output-norm substitution below changes only the gate
        # nonlinearity when a paper-aligned control explicitly requests it.
        self.fla_layer = FLAGatedDeltaNet(
            hidden_size=d_model,
            expand_v=self.expand_v,
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
        if use_gate and output_gate_activation == "sigmoid":
            # The vendored FLA snapshot exposes only its historical fused
            # Swish gate. Kimi Linear reports adopting a sigmoid output gate
            # across its experiments, including the GDN baseline. Replacing
            # only o_norm leaves FLA's projections, ShortConv, recurrence,
            # initialization, and output projection untouched.
            self.fla_layer.o_norm = RMSNormSigmoidGate(
                self.head_v_dim,
                eps=norm_eps,
            )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast_context(hidden_states, self.autocast_dtype):
            outputs = self.fla_layer(hidden_states=hidden_states, **kwargs)
        return unwrap_fla_output(outputs, "GatedDeltaNet")

    def state_size(self, sequence_length: int = 2048) -> int:
        """Return recurrent matrix-state scalars per layer (batch excluded)."""
        del sequence_length
        return self.num_heads * self.head_dim * self.head_v_dim
