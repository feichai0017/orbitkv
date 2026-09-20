"""Bind SGLang model computation and GPU representation to a cache identity."""

from importlib.metadata import version
from typing import Any

from orbitkv.identity import model_identity, state_namespace

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
    }
    representation = {
        "kv_cache_dtype": getattr(server_args, "kv_cache_dtype", None),
        "tp": [tp_rank, tp_size],
        "pp": [params.pp_rank, params.pp_size],
        "cp": [params.attn_cp_rank, params.attn_cp_size],
        "page_size": layout.page_size,
        "buffers": [
            {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape[1:]),
                "stride": list(tensor.stride()[1:]),
                "block_bytes": block_bytes,
            }
            for tensor, block_bytes in zip(layout.pool.kv_buffer, layout.block_bytes, strict=True)
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
