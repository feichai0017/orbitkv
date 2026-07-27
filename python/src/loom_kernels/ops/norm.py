"""Normalization predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import math

import torch

from .._torch_dispatch import (
    _add_rms_norm_mut,
    _rms_norm,
    _rms_norm_dynamic_per_token_fp8,
)
from ._common import _DTYPE_NAMES, _require_inference_tensors


def supports_rms_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> bool:
    """Return whether tensors match Loom's inference RMSNorm contract."""
    return bool(
        weight is not None
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and input_tensor.device.type == "cuda"
        and weight.device == input_tensor.device
        and input_tensor.dtype in _DTYPE_NAMES
        and weight.dtype == input_tensor.dtype
        and input_tensor.dim() >= 1
        and input_tensor.numel() > 0
        and weight.dim() == 1
        and weight.shape[0] == input_tensor.shape[-1]
        and input_tensor.is_contiguous()
        and weight.is_contiguous()
        and not input_tensor.requires_grad
        and not weight.requires_grad
    )


def supports_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    """Shape/dtype predicate shared with the vLLM IR provider."""
    return bool(
        variance_size is None
        and weight is not None
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and input_tensor.device.type == "cuda"
        and residual.device == input_tensor.device
        and weight.device == input_tensor.device
        and input_tensor.dtype in _DTYPE_NAMES
        and residual.dtype == input_tensor.dtype
        and weight.dtype == input_tensor.dtype
        and input_tensor.dim() >= 1
        and input_tensor.numel() > 0
        and input_tensor.shape == residual.shape
        and weight.dim() == 1
        and weight.shape[0] == input_tensor.shape[-1]
        and input_tensor.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and not input_tensor.requires_grad
        and not residual.requires_grad
        and not weight.requires_grad
    )


def supports_rms_norm_dynamic_fp8(
    input_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> bool:
    """Return whether Loom can fuse optional residual Add+RMSNorm with FP8."""
    return bool(
        weight is not None
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and input_tensor.device.type == "cuda"
        and weight.device == input_tensor.device
        and input_tensor.dtype in _DTYPE_NAMES
        and weight.dtype == input_tensor.dtype
        and input_tensor.dim() >= 1
        and input_tensor.numel() > 0
        and weight.dim() == 1
        and weight.shape[0] == input_tensor.shape[-1]
        and input_tensor.is_contiguous()
        and weight.is_contiguous()
        and not input_tensor.requires_grad
        and not weight.requires_grad
        and (
            residual is None
            or (
                residual.device == input_tensor.device
                and residual.dtype == input_tensor.dtype
                and residual.shape == input_tensor.shape
                and residual.is_contiguous()
                and not residual.requires_grad
            )
        )
    )


def _validate_rms_norm_inputs(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    if not supports_rms_norm(input_tensor, weight, epsilon):
        raise ValueError(
            "Loom RMSNorm requires same-device contiguous CUDA tensors, "
            "matching F32/FP16/BF16 dtypes, a 1D hidden-size weight, and no "
            "gradients"
        )
    hidden_size = input_tensor.shape[-1]
    rows = input_tensor.numel() // hidden_size
    if rows > 0xFFFF_FFFF or hidden_size > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")


def _validate_dynamic_fp8_inputs(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None,
) -> int:
    if not supports_rms_norm_dynamic_fp8(
        input_tensor, weight, epsilon, residual
    ):
        raise ValueError(
            "Loom RMSNorm+FP8 requires same-device contiguous CUDA tensors, "
            "matching F32/FP16/BF16 dtypes, a 1D hidden-size weight, and an "
            "optional same-shape residual"
        )
    hidden_size = input_tensor.shape[-1]
    rows = input_tensor.numel() // hidden_size
    if rows > 0xFFFF_FFFF or hidden_size > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return rows


def add_rms_norm_(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update input/residual in place and return those same tensor objects."""
    _require_inference_tensors(input_tensor, residual, weight)
    _add_rms_norm_mut(input_tensor, residual, weight, float(epsilon))
    return input_tensor, residual


def rms_norm_out(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Write RMSNorm into caller-owned output storage."""
    _require_inference_tensors(input_tensor, weight, output)
    _rms_norm(input_tensor, weight, output, float(epsilon))
    return output


def rms_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Return inference RMSNorm with the input dtype and shape."""
    _validate_rms_norm_inputs(input_tensor, weight, epsilon)
    output = torch.empty_like(input_tensor)
    return rms_norm_out(input_tensor, weight, output, epsilon)


def rms_norm_dynamic_fp8_out(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write optional Add+RMSNorm and per-token FP8 into caller buffers."""
    inference_tensors = [input_tensor, weight, output, scales]
    if residual is not None:
        inference_tensors.append(residual)
    _require_inference_tensors(*inference_tensors)
    _rms_norm_dynamic_per_token_fp8(
        output,
        input_tensor,
        weight,
        scales,
        float(epsilon),
        None,
        residual,
    )
    return output, scales


def rms_norm_dynamic_fp8(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return optional Add+RMSNorm FP8 output and one F32 scale per row."""
    rows = _validate_dynamic_fp8_inputs(
        input_tensor, weight, epsilon, residual
    )
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, 1), device=input_tensor.device, dtype=torch.float32)
    return rms_norm_dynamic_fp8_out(
        input_tensor,
        weight,
        output,
        scales,
        float(epsilon),
        residual,
    )
