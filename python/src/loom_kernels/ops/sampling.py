"""Sampling predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import (
    _apply_token_penalties,
    _categorical_sample,
    _greedy_sample_logprobs,
    _selected_token_logprobs,
    _top_k_filter,
    _top_p_renorm,
    _topk_sampled_logprobs,
)
from ._common import _DTYPE_NAMES


def supports_categorical_sample(
    probabilities: torch.Tensor,
    rng_state: torch.Tensor,
) -> bool:
    """Return whether tensor metadata fits explicit-state sampling."""
    return bool(
        probabilities.device.type == "cuda"
        and probabilities.dtype == torch.float32
        and probabilities.dim() == 2
        and probabilities.shape[0] > 0
        and probabilities.shape[1] > 0
        and probabilities.shape[0] <= 0x7FFF_FFFF
        and probabilities.shape[1] <= 0x7FFF_FFFF
        and probabilities.is_contiguous()
        and not probabilities.requires_grad
        and rng_state.device == probabilities.device
        and rng_state.dtype == torch.int64
        and rng_state.shape == (probabilities.shape[0], 2)
        and rng_state.is_contiguous()
        and not rng_state.requires_grad
    )


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


def supports_top_p_renorm(
    logits: torch.Tensor,
    top_ps: torch.Tensor,
) -> bool:
    """Return whether tensors match fused top-p filtering and renormalization."""
    return bool(
        supports_greedy_sample_logprobs(logits)
        and top_ps.device == logits.device
        and top_ps.dtype == torch.float32
        and top_ps.dim() == 1
        and top_ps.shape[0] == logits.shape[0]
        and top_ps.is_contiguous()
        and not top_ps.requires_grad
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


def _validate_categorical_sample(
    probabilities: torch.Tensor,
    rng_state: torch.Tensor,
) -> None:
    if not supports_categorical_sample(probabilities, rng_state):
        raise ValueError(
            "Loom categorical sampling requires inference-only contiguous "
            "rank-2 F32 CUDA probabilities and same-device contiguous int64 "
            "RNG state shaped [rows, 2]"
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


def _validate_top_p_renorm(
    logits: torch.Tensor,
    top_ps: torch.Tensor,
) -> None:
    if not supports_top_p_renorm(logits, top_ps):
        raise ValueError(
            "Loom top-p renormalization requires inference-only, non-empty "
            "rank-2 F32/FP16/BF16 CUDA logits with unit vocabulary stride "
            "and one same-device contiguous F32 top-p value in (0, 1] per row"
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


def categorical_sample(
    probabilities: torch.Tensor,
    rng_state: torch.Tensor,
) -> torch.Tensor:
    """Sample one token per row and advance every explicit counter once.

    Probability values are a device-side precondition: every row must be
    finite, non-negative, contain positive mass, and sum to one within 1e-5.
    Seeds and counters must be non-negative, and counters must be below the
    int64 maximum.
    """
    _validate_categorical_sample(probabilities, rng_state)
    return _categorical_sample(probabilities, rng_state)


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


def top_p_renorm_(
    logits: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Filter logits in place and return F32 probabilities over the nucleus."""
    _validate_top_p_renorm(logits, top_ps)
    return _top_p_renorm(logits, top_ps)


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
