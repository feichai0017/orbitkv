"""MoE movement predicates and public APIs around vendor grouped GEMM."""

from __future__ import annotations

import torch

from .._torch_dispatch import _moe_combine, _moe_permute
from ._common import _DTYPE_NAMES, _require_inference_tensors

_MOE_PERMUTE_DTYPES = {*_DTYPE_NAMES, torch.float8_e4m3fn}


def supports_moe_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int | None = None,
    expert_map: torch.Tensor | None = None,
) -> bool:
    """Return whether inputs match Loom's stable expert-major permutation."""
    if num_local_experts is None:
        num_local_experts = num_experts
    if (
        isinstance(num_experts, bool)
        or not isinstance(num_experts, int)
        or isinstance(num_local_experts, bool)
        or not isinstance(num_local_experts, int)
        or hidden_states.dim() != 2
        or topk_ids.dim() != 2
        or hidden_states.numel() == 0
        or topk_ids.numel() == 0
    ):
        return False
    tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    assignments = tokens * top_k
    map_supported = (
        isinstance(expert_map, torch.Tensor)
        and expert_map.device == hidden_states.device
        and expert_map.dtype == torch.int32
        and expert_map.shape == (num_experts,)
        and expert_map.is_contiguous()
        and not expert_map.requires_grad
    )
    return bool(
        hidden_states.device.type == "cuda"
        and hidden_states.dtype in _MOE_PERMUTE_DTYPES
        and hidden_states.is_contiguous()
        and not hidden_states.requires_grad
        and topk_ids.device == hidden_states.device
        and topk_ids.dtype == torch.int32
        and topk_ids.shape[0] == tokens
        and topk_ids.is_contiguous()
        and not topk_ids.requires_grad
        and 0 < num_local_experts <= num_experts <= 0x7FFF_FFFF
        and 0 < top_k <= num_experts
        and tokens <= 0xFFFF_FFFF
        and hidden_size <= 0xFFFF_FFFF
        and assignments <= 0x7FFF_FFFF
        and (
            map_supported
            if expert_map is not None
            else num_local_experts == num_experts
        )
    )


def _validate_moe_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int,
    expert_map: torch.Tensor | None,
) -> None:
    if not supports_moe_permute(
        hidden_states,
        topk_ids,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        expert_map=expert_map,
    ):
        raise ValueError(
            "Loom moe_permute requires contiguous inference CUDA tensors: "
            "rank-2 F32/FP16/BF16/FP8-E4M3FN hidden states and same-device "
            "int32 top-k "
            "IDs [tokens, top_k]. Expert-parallel layouts additionally "
            "require a contiguous int32 global-to-local expert map."
        )


def moe_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int | None = None,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stably group assignments for an unchanged vendor grouped GEMM.

    The outputs are expert-major activations ``[tokens * top_k, hidden]``,
    int64 expert offsets, an int32 inverse permutation ``[tokens, top_k]``,
    and flattened assignment IDs in permuted order. With expert parallelism,
    remote assignments occupy a zero-filled tail and use ``tokens * top_k``
    as the assignment-ID sentinel.
    """
    if num_local_experts is None:
        num_local_experts = num_experts
    _validate_moe_permute(
        hidden_states,
        topk_ids,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        expert_map=expert_map,
    )
    return _moe_permute(
        hidden_states,
        topk_ids,
        int(num_experts),
        int(num_local_experts),
        expert_map,
    )


def supports_moe_combine(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> bool:
    """Return whether inputs match Loom's weighted inverse permutation."""
    if (
        expert_outputs.dim() != 2
        or routing_weights.dim() != 2
        or inverse_permutation.dim() != 2
        or expert_offsets.dim() != 1
        or expert_outputs.numel() == 0
        or routing_weights.numel() == 0
    ):
        return False
    tokens, top_k = routing_weights.shape
    return bool(
        expert_outputs.device.type == "cuda"
        and expert_outputs.dtype in _DTYPE_NAMES
        and expert_outputs.is_contiguous()
        and routing_weights.device == expert_outputs.device
        and routing_weights.dtype == torch.float32
        and routing_weights.is_contiguous()
        and inverse_permutation.device == expert_outputs.device
        and inverse_permutation.dtype == torch.int32
        and inverse_permutation.shape == routing_weights.shape
        and inverse_permutation.is_contiguous()
        and expert_offsets.device == expert_outputs.device
        and expert_offsets.dtype == torch.int64
        and expert_offsets.numel() >= 2
        and expert_offsets.is_contiguous()
        and expert_outputs.shape[0] == routing_weights.numel()
        and not expert_outputs.requires_grad
        and not routing_weights.requires_grad
        and not inverse_permutation.requires_grad
        and not expert_offsets.requires_grad
        and tokens <= 0xFFFF_FFFF
        and top_k <= 0xFFFF_FFFF
        and expert_outputs.shape[1] <= 0xFFFF_FFFF
        and expert_offsets.numel() - 1 <= 0xFFFF_FFFF
        and tokens * top_k <= 0x7FFF_FFFF
    )


def moe_combine(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Invert expert-major movement and reduce routes with F32 weights."""
    if not supports_moe_combine(
        expert_outputs, routing_weights, inverse_permutation, expert_offsets
    ):
        raise ValueError(
            "Loom moe_combine requires contiguous same-device CUDA tensors: "
            "rank-2 expert outputs, F32 routing weights [tokens, top_k], "
            "matching int32 inverse permutation, and int64 expert offsets."
        )
    _require_inference_tensors(
        expert_outputs, routing_weights, inverse_permutation, expert_offsets
    )
    return _moe_combine(
        expert_outputs, routing_weights, inverse_permutation, expert_offsets
    )
