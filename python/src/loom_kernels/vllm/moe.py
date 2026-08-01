"""Explicit vLLM MoE movement override around unchanged vendor GEMM."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .._torch_extension import load_torch_extension, torch_extension_available
from ._runtime import _env_enabled, supports_installed_vllm

MOE_MOVEMENT_OVERRIDE_KEY = "moe_movement"
MOE_MOVEMENT_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_MOE_MOVEMENT"

_MOE_MOVEMENT_REGISTERED = False
_MOE_MOVEMENT_ORIGINAL_PERMUTE: Any | None = None
_MOE_MOVEMENT_ORIGINAL_UNPERMUTE: Any | None = None
_MOE_MOVEMENT_PERMUTE_HITS = 0
_MOE_MOVEMENT_COMBINE_HITS = 0
_MOE_MOVEMENT_FIRST_CONTRACT: dict[str, Any] | None = None
_MOE_MOVEMENT_FIRST_REJECTION: dict[str, Any] | None = None


def _moe_movement_override_requested() -> bool:
    return _env_enabled(MOE_MOVEMENT_OVERRIDE_ENV)


def _record_rejection(reason: str, **metadata: Any) -> None:
    global _MOE_MOVEMENT_FIRST_REJECTION
    if _MOE_MOVEMENT_FIRST_REJECTION is None:
        _MOE_MOVEMENT_FIRST_REJECTION = {"reason": reason, **metadata}


def _valid_permute_output(
    tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    assignments: int,
) -> bool:
    return bool(
        tensor.device == hidden_states.device
        and tensor.dtype == hidden_states.dtype
        and tensor.shape == (assignments, hidden_states.shape[1])
        and tensor.is_contiguous()
        and not tensor.requires_grad
    )


def _valid_combine_output(
    output: torch.Tensor,
    expert_outputs: torch.Tensor,
    tokens: int,
) -> bool:
    return bool(
        output.device == expert_outputs.device
        and output.dtype == expert_outputs.dtype
        and output.shape == (tokens, expert_outputs.shape[1])
        and output.is_contiguous()
        and not output.requires_grad
    )


def register_vllm_moe_movement() -> str | None:
    """Replace only vLLM's supported MoE permute/combine movement boundary.

    The Cutlass or Humming grouped GEMM functions and expert weights remain
    owned by vLLM. Unsupported tensors call the original vLLM wrappers; an
    admitted Loom call is fail-closed and never silently falls back after
    launch.
    """

    global _MOE_MOVEMENT_ORIGINAL_PERMUTE
    global _MOE_MOVEMENT_ORIGINAL_UNPERMUTE
    global _MOE_MOVEMENT_REGISTERED

    if _MOE_MOVEMENT_REGISTERED:
        return MOE_MOVEMENT_OVERRIDE_KEY
    if not torch_extension_available() or not supports_installed_vllm():
        return None

    movement = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.moe_permute_unpermute"
    )
    original_permute = movement.moe_permute
    original_unpermute = movement.moe_unpermute

    from ..torch_ops import supports_moe_combine, supports_moe_permute

    load_torch_extension()
    permute = torch.ops.loom_kernels.moe_permute.default
    permute_out = torch.ops.loom_kernels.moe_permute.out
    combine_out = torch.ops.loom_kernels.moe_combine.out

    def loom_moe_permute(
        hidden_states: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
        n_expert: int,
        n_local_expert: int = -1,
        expert_map: torch.Tensor | None = None,
        permuted_hidden_states: torch.Tensor | None = None,
        scratch: Any | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        global _MOE_MOVEMENT_FIRST_CONTRACT
        global _MOE_MOVEMENT_PERMUTE_HITS

        local_experts = n_expert if n_local_expert == -1 else n_local_expert
        prepared_topk_ids = (
            scratch.prepare_topk_ids(topk_ids)
            if scratch is not None
            else topk_ids
            if topk_ids.dtype == torch.int32
            else topk_ids.to(torch.int32)
        )
        if not supports_moe_permute(
            hidden_states,
            prepared_topk_ids,
            num_experts=n_expert,
            num_local_experts=local_experts,
            expert_map=expert_map,
        ):
            _record_rejection(
                "unsupported permutation contract",
                hidden_dtype=str(hidden_states.dtype),
                hidden_shape=list(hidden_states.shape),
                topk_dtype=str(topk_ids.dtype),
                topk_shape=list(topk_ids.shape),
            )
            return original_permute(
                hidden_states,
                a1q_scale,
                topk_ids,
                n_expert,
                n_local_expert,
                expert_map,
                permuted_hidden_states,
                scratch,
            )

        tokens, hidden_size = hidden_states.shape
        top_k = prepared_topk_ids.shape[1]
        assignments = tokens * top_k
        if permuted_hidden_states is None and scratch is not None:
            scratch.validate(hidden_states, topk_ids)
            scratch_hidden = scratch.permuted_hidden_states
            if scratch_hidden is not None:
                permuted_hidden_states = scratch_hidden[
                    : assignments * hidden_size
                ].view(assignments, hidden_size)
        if permuted_hidden_states is not None and not _valid_permute_output(
            permuted_hidden_states, hidden_states, assignments
        ):
            _record_rejection(
                "unsupported caller-owned permutation output",
                output_shape=list(permuted_hidden_states.shape),
                output_dtype=str(permuted_hidden_states.dtype),
            )
            return original_permute(
                hidden_states,
                a1q_scale,
                topk_ids,
                n_expert,
                n_local_expert,
                expert_map,
                permuted_hidden_states,
                scratch,
            )

        if scratch is None:
            if permuted_hidden_states is None:
                outputs = permute(
                    hidden_states,
                    prepared_topk_ids,
                    n_expert,
                    local_experts,
                    expert_map,
                )
                (
                    permuted_hidden_states,
                    expert_offsets,
                    inverse_permutation,
                    assignment_ids,
                ) = outputs
            else:
                expert_offsets = torch.empty(
                    local_experts + 1,
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
                inverse_permutation = torch.empty_like(prepared_topk_ids)
                assignment_ids = torch.empty(
                    assignments,
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
                permute_out(
                    hidden_states,
                    prepared_topk_ids,
                    n_expert,
                    local_experts,
                    expert_map,
                    permuted_hidden_states,
                    expert_offsets,
                    inverse_permutation,
                    assignment_ids,
                )
        else:
            scratch.validate(hidden_states, topk_ids)
            if n_expert != scratch.num_experts or local_experts != scratch.num_local_experts:
                raise ValueError("Loom MoE scratch expert counts are stale")
            if permuted_hidden_states is None:
                permuted_hidden_states = torch.empty(
                    (assignments, hidden_size),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            expert_offsets = scratch.expert_first_token_offset
            inverse_permutation = scratch.inv_permuted_idx[:assignments].view(
                tokens, top_k
            )
            assignment_ids = scratch.permuted_idx[:assignments]
            permute_out(
                hidden_states,
                prepared_topk_ids,
                n_expert,
                local_experts,
                expert_map,
                permuted_hidden_states,
                expert_offsets,
                inverse_permutation,
                assignment_ids,
            )

        if a1q_scale is not None and a1q_scale.dim() > 1:
            source_rows = assignment_ids.clamp(max=assignments - 1) // top_k
            a1q_scale = a1q_scale[source_rows]
        _MOE_MOVEMENT_PERMUTE_HITS += 1
        if _MOE_MOVEMENT_FIRST_CONTRACT is None:
            _MOE_MOVEMENT_FIRST_CONTRACT = {
                "hidden_shape": list(hidden_states.shape),
                "hidden_dtype": str(hidden_states.dtype),
                "top_k": top_k,
                "global_experts": n_expert,
                "local_experts": local_experts,
                "expert_parallel": expert_map is not None,
                "caller_owned_output": scratch is not None,
            }
        return (
            permuted_hidden_states,
            a1q_scale,
            expert_offsets,
            inverse_permutation.flatten(),
            assignment_ids,
        )

    def loom_moe_unpermute(
        out: torch.Tensor,
        permuted_hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        expert_first_token_offset: torch.Tensor | None = None,
    ) -> None:
        global _MOE_MOVEMENT_COMBINE_HITS

        tokens, top_k = topk_weights.shape
        inverse = inv_permuted_idx.view(tokens, top_k)
        if (
            expert_first_token_offset is None
            or not supports_moe_combine(
                permuted_hidden_states,
                topk_weights,
                inverse,
                expert_first_token_offset,
            )
            or not _valid_combine_output(out, permuted_hidden_states, tokens)
        ):
            _record_rejection(
                "unsupported combine contract",
                expert_output_dtype=str(permuted_hidden_states.dtype),
                expert_output_shape=list(permuted_hidden_states.shape),
                weight_dtype=str(topk_weights.dtype),
                weight_shape=list(topk_weights.shape),
            )
            return original_unpermute(
                out,
                permuted_hidden_states,
                topk_weights,
                inv_permuted_idx,
                expert_first_token_offset,
            )
        combine_out(
            permuted_hidden_states,
            topk_weights,
            inverse,
            expert_first_token_offset,
            out,
        )
        _MOE_MOVEMENT_COMBINE_HITS += 1

    loom_moe_permute.__module__ = __name__
    loom_moe_unpermute.__module__ = __name__
    movement.moe_permute = loom_moe_permute
    movement.moe_unpermute = loom_moe_unpermute
    for module_name in (
        "vllm.model_executor.layers.fused_moe.experts.cutlass_moe",
        "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe",
    ):
        try:
            consumer = importlib.import_module(module_name)
        except ImportError:
            continue
        if getattr(consumer, "moe_permute", None) is original_permute:
            consumer.moe_permute = loom_moe_permute
        if getattr(consumer, "moe_unpermute", None) is original_unpermute:
            consumer.moe_unpermute = loom_moe_unpermute

    _MOE_MOVEMENT_ORIGINAL_PERMUTE = original_permute
    _MOE_MOVEMENT_ORIGINAL_UNPERMUTE = original_unpermute
    _MOE_MOVEMENT_REGISTERED = True
    return MOE_MOVEMENT_OVERRIDE_KEY


def _metadata() -> dict[str, object]:
    return {
        "moe_movement_override_requested": _moe_movement_override_requested(),
        "moe_movement_override": _MOE_MOVEMENT_REGISTERED,
        "moe_movement_permute_hits": _MOE_MOVEMENT_PERMUTE_HITS,
        "moe_movement_combine_hits": _MOE_MOVEMENT_COMBINE_HITS,
        "moe_movement_first_contract": _MOE_MOVEMENT_FIRST_CONTRACT,
        "moe_movement_first_rejection": _MOE_MOVEMENT_FIRST_REJECTION,
        "moe_grouped_gemm_owner": "vllm_vendor_backend",
    }
