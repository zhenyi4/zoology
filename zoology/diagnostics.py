"""Deterministic fingerprints for training repeatability diagnostics."""

from __future__ import annotations

import hashlib
import os
import platform
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


NamedTensor = Tuple[str, Optional[torch.Tensor]]


def named_tensors_sha256(named_tensors: Iterable[NamedTensor]) -> str:
    """Return a stable SHA256 over names, metadata, and exact tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if tensor is None:
            digest.update(b"<none>\0")
            continue

        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        # Viewing flattened storage as bytes also works for bfloat16 tensors,
        # whose dtype is not supported by every NumPy build.
        raw_bytes = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(raw_bytes)
        digest.update(b"\0")
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor, name: str = "tensor") -> str:
    return named_tensors_sha256([(name, tensor)])


def model_parameters_sha256(model: nn.Module) -> str:
    return named_tensors_sha256(model.named_parameters())


def model_gradients_sha256(model: nn.Module) -> str:
    return named_tensors_sha256(
        (name, parameter.grad) for name, parameter in model.named_parameters()
    )


def rng_state_sha256() -> str:
    states = [("torch_cpu", torch.get_rng_state())]
    if torch.cuda.is_available():
        states.extend(
            (f"torch_cuda_{device_idx}", state)
            for device_idx, state in enumerate(torch.cuda.get_rng_state_all())
        )
    return named_tensors_sha256(states)


def runtime_diagnostics() -> dict[str, object]:
    """Return environment facts that affect deterministic CUDA execution."""
    result: dict[str, object] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "none",
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", "unset"
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if torch.cuda.is_available():
        result.update(
            {
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "gpu_capability": str(
                    torch.cuda.get_device_capability(torch.cuda.current_device())
                ),
            }
        )
    return result
