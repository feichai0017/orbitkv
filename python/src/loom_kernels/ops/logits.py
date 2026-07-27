"""Logits-processing predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import _logits_preprocess, _min_p_filter
from ._common import _DTYPE_NAMES


def supports_logits_preprocess(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    blocked_mask: torch.Tensor | None = None,
    bias_row_ids: torch.Tensor | None = None,
    bias_token_ids: torch.Tensor | None = None,
    bias_values: torch.Tensor | None = None,
    suppressed_row_ids: torch.Tensor | None = None,
    suppressed_token_ids: torch.Tensor | None = None,
) -> bool:
    """Return whether tensors match Loom's fused F32 preprocessing boundary."""
    if logits.dim() != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        return False
    if not (
        logits.device.type == "cuda"
        and logits.dtype == torch.float32
        and logits.shape[0] <= 0x7FFF_FFFF
        and logits.shape[1] <= 0x7FFF_FFFF
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and temperatures.device == logits.device
        and temperatures.dtype == torch.float32
        and temperatures.dim() == 1
        and temperatures.shape[0] == logits.shape[0]
        and temperatures.is_contiguous()
    ):
        return False
    if blocked_mask is not None and not (
        blocked_mask.device == logits.device
        and blocked_mask.dtype == torch.bool
        and blocked_mask.shape == logits.shape
        and blocked_mask.is_contiguous()
    ):
        return False

    bias_group = (bias_row_ids, bias_token_ids, bias_values)
    if any(value is None for value in bias_group):
        if not all(value is None for value in bias_group):
            return False
    else:
        assert bias_row_ids is not None
        assert bias_token_ids is not None
        assert bias_values is not None
        bias_count = bias_row_ids.numel()
        if not (
            bias_row_ids.device == logits.device
            and bias_token_ids.device == logits.device
            and bias_values.device == logits.device
            and bias_row_ids.dtype == torch.int32
            and bias_token_ids.dtype == torch.int32
            and bias_values.dtype == torch.float32
            and bias_row_ids.dim() == 1
            and bias_token_ids.dim() == 1
            and bias_values.dim() == 1
            and 0 < bias_count <= 0xFFFF_FFFF
            and bias_token_ids.numel() == bias_count
            and bias_values.numel() == bias_count
            and bias_row_ids.is_contiguous()
            and bias_token_ids.is_contiguous()
            and bias_values.is_contiguous()
        ):
            return False

    suppression_group = (suppressed_row_ids, suppressed_token_ids)
    if any(value is None for value in suppression_group):
        if not all(value is None for value in suppression_group):
            return False
    else:
        assert suppressed_row_ids is not None
        assert suppressed_token_ids is not None
        suppression_count = suppressed_row_ids.numel()
        if not (
            suppressed_row_ids.device == logits.device
            and suppressed_token_ids.device == logits.device
            and suppressed_row_ids.dtype == torch.int32
            and suppressed_token_ids.dtype == torch.int32
            and suppressed_row_ids.dim() == 1
            and suppressed_token_ids.dim() == 1
            and 0 < suppression_count <= 0xFFFF_FFFF
            and suppressed_token_ids.numel() == suppression_count
            and suppressed_row_ids.is_contiguous()
            and suppressed_token_ids.is_contiguous()
        ):
            return False

    tensors = [
        logits,
        temperatures,
        blocked_mask,
        bias_row_ids,
        bias_token_ids,
        bias_values,
        suppressed_row_ids,
        suppressed_token_ids,
    ]
    return not any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def _validate_logits_preprocess(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    blocked_mask: torch.Tensor | None,
    bias_row_ids: torch.Tensor | None,
    bias_token_ids: torch.Tensor | None,
    bias_values: torch.Tensor | None,
    suppressed_row_ids: torch.Tensor | None,
    suppressed_token_ids: torch.Tensor | None,
) -> None:
    if not supports_logits_preprocess(
        logits,
        temperatures,
        blocked_mask,
        bias_row_ids,
        bias_token_ids,
        bias_values,
        suppressed_row_ids,
        suppressed_token_ids,
    ):
        raise ValueError(
            "Loom logits preprocessing requires non-empty rank-2 F32 CUDA "
            "logits, contiguous F32 [rows] temperatures, an optional "
            "contiguous bool [rows, vocab] blocked mask, an optional non-empty "
            "int32/int32/F32 sparse bias triplet, and an optional non-empty "
            "int32/int32 sparse suppression pair on the same device"
        )


def logits_preprocess_(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    blocked_mask: torch.Tensor | None = None,
    bias_row_ids: torch.Tensor | None = None,
    bias_token_ids: torch.Tensor | None = None,
    bias_values: torch.Tensor | None = None,
    suppressed_row_ids: torch.Tensor | None = None,
    suppressed_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse token masking, sparse bias/suppression, and temperature in place."""
    _validate_logits_preprocess(
        logits,
        temperatures,
        blocked_mask,
        bias_row_ids,
        bias_token_ids,
        bias_values,
        suppressed_row_ids,
        suppressed_token_ids,
    )
    _logits_preprocess(
        logits,
        temperatures,
        blocked_mask,
        bias_row_ids,
        bias_token_ids,
        bias_values,
        suppressed_row_ids,
        suppressed_token_ids,
    )
    return logits


def supports_min_p_filter(logits: torch.Tensor, min_p: torch.Tensor) -> bool:
    """Return whether tensors match Loom's in-place min-p CUDA boundary."""
    if logits.dim() != 2 or min_p.dim() not in (1, 2):
        return False
    min_p_shape_matches = bool(
        (min_p.dim() == 1 and min_p.shape[0] == logits.shape[0])
        or (
            min_p.dim() == 2
            and min_p.shape[0] == logits.shape[0]
            and min_p.shape[1] == 1
        )
    )
    return bool(
        logits.device.type == "cuda"
        and logits.dtype in _DTYPE_NAMES
        and logits.dim() == 2
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and logits.shape[0] <= 0xFFFF_FFFF
        and logits.shape[1] <= 0xFFFF_FFFF
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and min_p.device == logits.device
        and min_p.dtype == torch.float32
        and min_p_shape_matches
        and min_p.is_contiguous()
        and not logits.requires_grad
        and not min_p.requires_grad
    )


def _validate_min_p_filter(
    logits: torch.Tensor,
    min_p: torch.Tensor,
) -> None:
    if not supports_min_p_filter(logits, min_p):
        raise ValueError(
            "Loom min-p filtering requires non-empty rank-2 F32/FP16/BF16 "
            "CUDA logits with unit vocabulary stride and same-device "
            "contiguous F32 probabilities shaped [rows] or [rows, 1]"
        )


def min_p_filter_(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Filter logits in place using each row's max-probability ratio."""
    _validate_min_p_filter(logits, min_p)
    _min_p_filter(logits, min_p)
    return logits
