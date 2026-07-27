"""Sampling predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import (
    _apply_token_penalties,
    _greedy_sample_logprobs,
    _selected_token_logprobs,
    _top_k_filter,
    _topk_sampled_logprobs,
)
from ._common import _DTYPE_NAMES


def supports_greedy_sample_logprobs(logits: torch.Tensor) -> bool:
    """Return whether logits match the deterministic greedy CUDA boundary."""
    return bool(
        logits.device.type == "cuda"
        and logits.dtype in _DTYPE_NAMES
        and logits.dim() == 2
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and logits.shape[0] <= 0xFFFF_FFFF
        and logits.shape[1] <= 0x7FFF_FFFF
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and not logits.requires_grad
    )


def supports_selected_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> bool:
    """Return whether one selected token per logits row can be normalized."""
    return bool(
        supports_greedy_sample_logprobs(logits)
        and token_ids.device == logits.device
        and token_ids.dtype == torch.int64
        and token_ids.dim() == 1
        and token_ids.shape[0] == logits.shape[0]
        and token_ids.is_contiguous()
        and not token_ids.requires_grad
    )


def supports_topk_sampled_logprobs(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    top_k: int,
) -> bool:
    """Return whether sampled-token plus top-k logprobs can be fused."""
    return bool(
        supports_selected_token_logprobs(logits, sampled_token_ids)
        and isinstance(top_k, int)
        and not isinstance(top_k, bool)
        and 1 <= top_k <= min(logits.shape[1], 32)
    )


def supports_top_k_filter(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
) -> bool:
    """Return whether tensors match the exact in-place top-k boundary."""
    return bool(
        supports_greedy_sample_logprobs(logits)
        and top_ks.device == logits.device
        and top_ks.dtype == torch.int32
        and top_ks.dim() == 1
        and top_ks.shape[0] == logits.shape[0]
        and top_ks.is_contiguous()
        and not top_ks.requires_grad
    )


def token_penalties_workspace_capacity(
    prompt_tokens: int,
    output_tokens: int,
) -> int:
    """Return packed int64 hash slots required per logits row."""
    if prompt_tokens <= 0 or output_tokens <= 0:
        raise ValueError("token-penalty history widths must be positive")
    return 1 << (2 * (prompt_tokens + output_tokens) - 1).bit_length()


def supports_apply_token_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    output_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    workspace: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the sparse F32 penalty boundary."""
    if not (
        logits.device.type == "cuda"
        and logits.dtype == torch.float32
        and logits.dim() == 2
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and logits.shape[0] <= 0x7FFF_FFFF
        and logits.shape[1] <= 0x7FFF_FFFF
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and not logits.requires_grad
    ):
        return False
    rows = logits.shape[0]
    matrices = (prompt_token_ids, output_token_ids)
    if not all(
        tensor.device == logits.device
        and tensor.dtype == torch.int64
        and tensor.dim() == 2
        and tensor.shape[0] == rows
        and tensor.shape[1] > 0
        and tensor.stride(1) == 1
        and tensor.stride(0) >= tensor.shape[1]
        and not tensor.requires_grad
        for tensor in matrices
    ):
        return False
    penalties = (
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
    if not all(
        tensor.device == logits.device
        and tensor.dtype == torch.float32
        and tensor.dim() == 1
        and tensor.shape[0] == rows
        and tensor.is_contiguous()
        and not tensor.requires_grad
        for tensor in penalties
    ):
        return False
    required = token_penalties_workspace_capacity(
        prompt_token_ids.shape[1], output_token_ids.shape[1]
    )
    return bool(
        workspace.device == logits.device
        and workspace.dtype == torch.int64
        and workspace.dim() == 2
        and workspace.shape[0] == rows
        and workspace.shape[1] >= required
        and workspace.shape[1] <= 0xFFFF_FFFF
        and (workspace.shape[1] & (workspace.shape[1] - 1)) == 0
        and workspace.stride(1) == 1
        and workspace.stride(0) >= workspace.shape[1]
        and not workspace.requires_grad
    )


def _validate_greedy_sample_logits(
    logits: torch.Tensor,
) -> None:
    if not supports_greedy_sample_logprobs(logits):
        raise ValueError(
            "Loom greedy sampling requires finite, non-empty rank-2 "
            "F32/FP16/BF16 CUDA logits with unit vocabulary stride, "
            "non-overlapping rows, and no gradients"
        )


def _validate_selected_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> None:
    if not supports_selected_token_logprobs(logits, token_ids):
        raise ValueError(
            "Loom selected-token logprobs require finite, non-empty rank-2 "
            "F32/FP16/BF16 CUDA logits with unit vocabulary stride and one "
            "same-device contiguous int64 token ID per row; token IDs must "
            "be in vocabulary range"
        )


def _validate_topk_sampled_logprobs(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    top_k: int,
) -> None:
    if not supports_topk_sampled_logprobs(logits, sampled_token_ids, top_k):
        maximum = min(logits.shape[1], 32) if logits.dim() == 2 else 32
        raise ValueError(
            "Loom top-k sampled logprobs require inference-only, non-empty "
            "rank-2 F32/FP16/BF16 CUDA logits with unit vocabulary stride; "
            "one same-device contiguous int64 sampled token ID per row; and "
            f"1 <= top_k <= {maximum}"
        )


def _validate_top_k_filter(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
) -> None:
    if not supports_top_k_filter(logits, top_ks):
        raise ValueError(
            "Loom top-k filtering requires inference-only, non-empty rank-2 "
            "F32/FP16/BF16 CUDA logits with unit vocabulary stride and one "
            "same-device contiguous int32 top-k value per row"
        )


def _validate_apply_token_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    output_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    if not supports_apply_token_penalties(
        logits,
        prompt_token_ids,
        output_token_ids,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        workspace,
    ):
        raise ValueError(
            "Loom token penalties require inference-only F32 rank-2 CUDA "
            "logits; same-device row-major int64 prompt/output IDs; one "
            "contiguous F32 value per row for presence, frequency, and "
            "repetition; and a same-device int64 power-of-two workspace with "
            "at least twice the combined padded history width"
        )


def greedy_sample_logprobs(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return greedy IDs, sampled logprobs, and vLLM-compatible tie ranks."""
    _validate_greedy_sample_logits(logits)
    return _greedy_sample_logprobs(logits)


def selected_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized logprobs and ranks for one selected token per row."""
    _validate_selected_token_logprobs(logits, token_ids)
    return _selected_token_logprobs(logits, token_ids)


def top_k_filter_(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
) -> torch.Tensor:
    """Apply exact per-row top-k thresholds in place, preserving ties."""
    _validate_top_k_filter(logits, top_ks)
    _top_k_filter(logits, top_ks)
    return logits


def topk_sampled_logprobs(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sampled token, deterministic top-k logprobs, and sampled ranks."""
    _validate_topk_sampled_logprobs(logits, sampled_token_ids, top_k)
    return _topk_sampled_logprobs(logits, sampled_token_ids, top_k)


def apply_token_penalties_(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    output_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Apply sparse vLLM-compatible penalties in place with caller workspace."""
    _validate_apply_token_penalties(
        logits,
        prompt_token_ids,
        output_token_ids,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        workspace,
    )
    _apply_token_penalties(
        logits,
        prompt_token_ids,
        output_token_ids,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        workspace,
    )
    return logits
