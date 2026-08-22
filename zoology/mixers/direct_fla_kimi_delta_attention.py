"""Thin Zoology adapter around FLA's official Kimi Delta Attention layer.

KDA entered FLA in release v0.4.0.  The adapter deliberately contains no KDA
equations: FLA owns the projections, per-key-dimension decay gate, ShortConv,
delta-rule recurrence, sigmoid output gate, normalization, and initialization.
The sole optional exception is an initialization-only diagnostic that replaces
FLA v0.4.2's zero ``dt_bias`` with the log-uniform time-step initialization
introduced by FLA v0.5.2.  It does not replace or modify the recurrence kernel.
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


SUPPORTED_DT_BIAS_INITS = {
    "fla_v042_zero",
    "fla_v052_log_uniform",
}


def _reinitialize_dt_bias_v052_(dt_bias: nn.Parameter) -> None:
    """Apply FLA v0.5.2's KDA ``dt_bias`` initialization in place.

    ``torch.random.fork_rng`` restores the caller's RNG state on exit.  This is
    important for a strict initialization-only ablation: sampling the corrected
    ``dt_bias`` must not shift the initialization of later layers or parameters.
    """

    dt_min = 0.001
    dt_max = 0.1
    dt_init_floor = 1e-4
    cuda_devices: list[int] = []
    if dt_bias.is_cuda:
        device_index = dt_bias.device.index
        cuda_devices = [
            torch.cuda.current_device() if device_index is None else device_index
        ]

    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        dt = torch.exp(
            torch.rand(
                dt_bias.shape,
                dtype=torch.float32,
                device=dt_bias.device,
            )
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus_dt = dt + torch.log(-torch.expm1(-dt))

    with torch.no_grad():
        dt_bias.copy_(inverse_softplus_dt.to(dtype=dt_bias.dtype))


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
        dt_bias_init: str = "fla_v042_zero",
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
        if dt_bias_init not in SUPPORTED_DT_BIAS_INITS:
            raise ValueError(
                "dt_bias_init must be one of "
                f"{sorted(SUPPORTED_DT_BIAS_INITS)}; got {dt_bias_init!r}."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.head_dim = head_dim
        self.head_v_dim = head_v_dim
        self.expand_v = expand_v
        self.autocast_dtype = autocast_dtype
        self.dt_bias_init = dt_bias_init

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
        if dt_bias_init == "fla_v052_log_uniform":
            _reinitialize_dt_bias_v052_(self.fla_layer.dt_bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast_context(hidden_states, self.autocast_dtype):
            outputs = self.fla_layer(hidden_states=hidden_states, **kwargs)
        return unwrap_fla_output(outputs, "KimiDeltaAttention")

    def state_size(self, sequence_length: int = 2048) -> int:
        """Return recurrent matrix-state scalars per layer (batch excluded)."""
        del sequence_length
        return self.num_v_heads * self.head_dim * self.head_v_dim
