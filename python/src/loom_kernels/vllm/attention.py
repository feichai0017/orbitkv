"""vLLM paged-decode attention registration and shape policy."""

from __future__ import annotations

from typing import Any

import torch

from .._torch_extension import torch_extension_available
from ._runtime import _env_enabled, supports_installed_vllm

PAGED_DECODE_OVERRIDE_KEY = "paged_decode_attention"
PAGED_DECODE_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION"
PAGED_DECODE_FAST_PATH_MAX_BATCH = 128
PAGED_DECODE_FAST_PATH_MAX_CONTEXT = 32

_PAGED_DECODE_REGISTERED = False
_PAGED_DECODE_ORIGINAL_FORWARD: Any | None = None
_PAGED_DECODE_CAN_USE_FAST_PATH: Any | None = None
_PAGED_DECODE_FIRST_CONTRACT: dict[str, Any] | None = None
_PAGED_DECODE_FIRST_REJECTION: dict[str, Any] | None = None


def _paged_decode_override_requested() -> bool:
    return _env_enabled(PAGED_DECODE_OVERRIDE_ENV)


def supports_vllm_paged_decode_shape(
    *,
    dtype: torch.dtype,
    batch: int,
    query_heads: int,
    kv_heads: int,
    head_size: int,
    block_size: int,
    max_sequence_length: int,
) -> bool:
    """Return whether a shape is inside the H20-qualified FA3 win region."""
    return bool(
        dtype in (torch.float16, torch.bfloat16)
        and 0 < batch <= PAGED_DECODE_FAST_PATH_MAX_BATCH
        and query_heads == 32
        and kv_heads == 8
        and head_size == 128
        and block_size in (16, 32)
        and 0 < max_sequence_length <= PAGED_DECODE_FAST_PATH_MAX_CONTEXT
    )


def register_vllm_paged_decode_attention() -> str | None:
    """Install a measured-shape vLLM 0.24/0.25 FlashAttention decode path."""
    global _PAGED_DECODE_CAN_USE_FAST_PATH
    global _PAGED_DECODE_ORIGINAL_FORWARD
    global _PAGED_DECODE_REGISTERED
    if _PAGED_DECODE_REGISTERED:
        return PAGED_DECODE_OVERRIDE_KEY
    if not torch_extension_available():
        return None

    from ..torch_ops import supports_paged_decode_attention

    if not supports_installed_vllm():
        return None

    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    implementation = torch.ops.loom_kernels.paged_decode_attention.default
    original_forward = FlashAttentionImpl.forward
    native_cache_dtypes = {
        "auto",
        "float16",
        "half",
        "bfloat16",
        torch.float16,
        torch.bfloat16,
    }

    def can_use_fast_path(
        attention: Any,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        if attn_metadata is None or kv_cache.dim() != 5:
            return False
        sequences = int(attn_metadata.seq_lens.shape[0])
        block_size = int(kv_cache.shape[2])
        if not supports_vllm_paged_decode_shape(
            dtype=query.dtype,
            batch=sequences,
            query_heads=attention.num_heads,
            kv_heads=attention.num_kv_heads,
            head_size=attention.head_size,
            block_size=block_size,
            max_sequence_length=attn_metadata.max_seq_len,
        ):
            return False
        # FA3's AOT scheduler tensor and the non-DCP zero context length are
        # execution hints, not attention semantics. Loom owns its scheduling;
        # the DCP world-size/cache-length and cascade gates below remain strict.
        return bool(
            output_scale is None
            and output_block_scale is None
            and attention.attn_type == AttentionType.DECODER
            and attention.num_heads == 32
            and attention.num_kv_heads == 8
            and attention.head_size == 128
            and attention.alibi_slopes is None
            and tuple(attention.sliding_window) == (-1, -1)
            and attention.logits_soft_cap == 0
            and attention.sinks is None
            and attention.kv_sharing_target_layer_name is None
            and attention.kv_cache_dtype in native_cache_dtypes
            and getattr(attention, "dcp_world_size", 1) == 1
            and attn_metadata.max_query_len == 1
            and attn_metadata.num_actual_tokens == sequences
            and attn_metadata.query_start_loc.shape[0] == sequences + 1
            and attn_metadata.block_table.shape[0] == sequences
            and not attn_metadata.use_cascade
            and attn_metadata.common_prefix_len == 0
            and attn_metadata.dcp_context_kv_lens is None
            and attn_metadata.prefix_scheduler_metadata is None
            and attn_metadata.causal is True
            and attn_metadata.mm_prefix_range_tensor is None
            and query.device.type == "cuda"
            and query.dtype == kv_cache.dtype
            and query.dim() == 3
            and query.shape[0] >= sequences
            and tuple(query.shape[1:]) == (32, 128)
            and query.is_contiguous()
            and not query.requires_grad
            and output.device == query.device
            and output.dtype == query.dtype
            and output.dim() == 3
            and output.shape[0] >= sequences
            and tuple(output.shape[1:]) == (32, 128)
            and output.is_contiguous()
            and kv_cache.device == query.device
            and tuple(kv_cache.shape[1:])
            == (2, block_size, 8, 128)
        )

    def forward(
        attention: Any,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global _PAGED_DECODE_FIRST_CONTRACT
        global _PAGED_DECODE_FIRST_REJECTION
        if not can_use_fast_path(
            attention,
            query,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        ):
            if _PAGED_DECODE_FIRST_REJECTION is None and attn_metadata is not None:
                _PAGED_DECODE_FIRST_REJECTION = {
                    "query_shape": list(query.shape),
                    "query_dtype": str(query.dtype),
                    "kv_cache_shape": list(kv_cache.shape),
                    "num_actual_tokens": attn_metadata.num_actual_tokens,
                    "max_query_len": attn_metadata.max_query_len,
                    "max_seq_len": attn_metadata.max_seq_len,
                    "use_cascade": attn_metadata.use_cascade,
                }
            return original_forward(
                attention,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        sequences = int(attn_metadata.seq_lens.shape[0])
        query_view = query[:sequences]
        output_view = output[:sequences]
        key_cache, value_cache = kv_cache.unbind(1)
        if not supports_paged_decode_attention(
            query_view,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            max_sequence_length=attn_metadata.max_seq_len,
        ):
            return original_forward(
                attention,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if _PAGED_DECODE_FIRST_CONTRACT is None:
            _PAGED_DECODE_FIRST_CONTRACT = {
                "query_shape": list(query_view.shape),
                "query_stride": list(query_view.stride()),
                "dtype": str(query_view.dtype),
                "kv_cache_shape": list(kv_cache.shape),
                "kv_cache_stride": list(kv_cache.stride()),
                "block_table_shape": list(attn_metadata.block_table.shape),
                "max_seq_len": attn_metadata.max_seq_len,
            }
        implementation(
            query_view,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            output_view,
            attn_metadata.max_seq_len,
            attention.scale,
        )
        return output

    forward.__module__ = __name__
    _PAGED_DECODE_ORIGINAL_FORWARD = original_forward
    _PAGED_DECODE_CAN_USE_FAST_PATH = can_use_fast_path
    FlashAttentionImpl.forward = forward
    _PAGED_DECODE_REGISTERED = True
    return PAGED_DECODE_OVERRIDE_KEY
def _metadata() -> dict[str, object]:
    return {
        "paged_decode_override_requested": _paged_decode_override_requested(),
        "paged_decode_override": _PAGED_DECODE_REGISTERED,
        "paged_decode_fast_path_max_batch": PAGED_DECODE_FAST_PATH_MAX_BATCH,
        "paged_decode_fast_path_max_context": PAGED_DECODE_FAST_PATH_MAX_CONTEXT,
        "paged_decode_first_contract": _PAGED_DECODE_FIRST_CONTRACT,
        "paged_decode_first_rejection": _PAGED_DECODE_FIRST_REJECTION,
    }
