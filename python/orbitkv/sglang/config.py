"""Bind SGLang model computation and GPU representation to a cache identity."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from orbitkv.identity import artifact_identity, model_identity, state_namespace

if TYPE_CHECKING:
    from .layout import GpuLayout


def derive_namespace(server_args: Any, params: Any, layout: GpuLayout) -> str:
    if server_args.enable_lora:
        raise ValueError(
            "OrbitKV requires immutable adapter identities; dynamic LoRA is unsupported"
        )
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp_rank = parallel.tp_rank
    tp_size = parallel.tp_size
    computation = {
        "weight_version": getattr(server_args, "weight_version", None),
        "quantization": getattr(server_args, "quantization", None),
        "model_overrides": getattr(server_args, "json_model_override_args", None),
        "dtype": server_args.dtype,
        "attention_backend": server_args.attention_backend,
        "prefill_attention_backend": server_args.prefill_attention_backend,
        "decode_attention_backend": server_args.decode_attention_backend,
        "mamba_backend": getattr(server_args, "mamba_backend", None),
        "linear_attention": {
            name: getattr(server_args, name, None)
            for name in (
                "linear_attn_backend",
                "linear_attn_decode_backend",
                "linear_attn_prefill_backend",
                "linear_attn_verify_backend",
                "enable_mamba_cache_stochastic_rounding",
                "mamba_cache_philox_rounds",
            )
        },
    }
    scale_path = getattr(server_args, "quantization_param_path", None)
    representation = {
        "kv_scale_artifact": artifact_identity(scale_path) if scale_path else None,
        "kv_cache_dtype": getattr(server_args, "kv_cache_dtype", None),
        "tp": [tp_rank, tp_size],
        "pp": [params.pp_rank, params.pp_size],
        "cp": [params.attn_cp_rank, params.attn_cp_size],
        "page_size": layout.page_size,
        "groups": [
            {
                "group": pool.group_id,
                "kind": pool.kind,
                "window": pool.window,
                "layers": sorted(pool.entry.layer_mapping.items()),
                "buffers": [
                    {
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape[1:]),
                        "stride": list(tensor.stride()[1:]),
                        "block_bytes": block_bytes,
                    }
                    for tensor, block_bytes in zip(
                        pool.entry.kv_buffer, pool.block_bytes, strict=True
                    )
                ],
            }
            for pool in layout.pools.values()
        ],
    }
    return state_namespace(
        engine="sglang",
        engine_version=version("sglang"),
        model=model_identity(
            server_args.model_path,
            revision=server_args.revision,
            tokenizer=server_args.tokenizer_path,
            tokenizer_revision=server_args.revision,
        ),
        computation=computation,
        representation=representation,
    )
