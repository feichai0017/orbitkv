from __future__ import annotations

import os
import pkgutil
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

from ..pinned import validate_patched_checkout
from . import state as _state
from .state import RuntimeLimits, _config, _request_key, _runtime

SUPPORTED_ARCHITECTURES = ("Qwen2ForCausalLM", "GptOssForCausalLM")
ATTENTION_BACKENDS_BY_ARCHITECTURE = {
    "Qwen2ForCausalLM": ("flashinfer", "flashinfer"),
    "GptOssForCausalLM": ("fa3", "fa3"),
}
_ENTRYPOINT_NAME = "orbitkv_manager"

_PROPAGATED_ALIASES = (
    (
        "sglang.srt.managers.schedule_batch.alloc_for_extend",
        "sglang.srt.mem_cache.allocation.alloc_for_extend",
    ),
    (
        "sglang.srt.managers.schedule_batch.alloc_for_decode",
        "sglang.srt.mem_cache.allocation.alloc_for_decode",
    ),
    (
        "sglang.srt.managers.schedule_batch.release_kv_cache",
        "sglang.srt.mem_cache.common.release_kv_cache",
    ),
    (
        "sglang.srt.managers.scheduler.release_kv_cache",
        "sglang.srt.mem_cache.common.release_kv_cache",
    ),
    (
        "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache",
        "sglang.srt.mem_cache.common.release_kv_cache",
    ),
)


HOOK_TARGETS = (
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool_allocator",
    "sglang.srt.mem_cache.allocation.alloc_for_extend",
    "sglang.srt.mem_cache.allocation.alloc_for_decode",
    "sglang.srt.managers.schedule_batch.ScheduleBatch.maybe_evict_swa",
    "sglang.srt.mem_cache.common.release_kv_cache",
    "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
    "sglang.srt.managers.scheduler.Scheduler.run_batch",
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator.configure",
    "sglang.srt.managers.scheduler.Scheduler.get_internal_state",
)


def _validate_sglang_revision() -> None:
    import sglang

    value = os.environ.get("ORBITKV_SGLANG_ROOT")
    if not value:
        raise RuntimeError("ORBITKV_SGLANG_ROOT is required for the pinned adapter")
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"invalid ORBITKV_SGLANG_ROOT {value}: {error}") from error
    if not root.is_dir() or not (root / "python/sglang/__init__.py").is_file():
        raise RuntimeError("ORBITKV_SGLANG_ROOT is not an SGLang source checkout")
    imported = Path(sglang.__file__).resolve(strict=True)
    expected_package = (root / "python/sglang").resolve(strict=True)
    if not imported.is_relative_to(expected_package):
        raise RuntimeError("the imported SGLang package is outside ORBITKV_SGLANG_ROOT")
    validate_patched_checkout(root)


def _validate_plugin_selection() -> None:
    if os.environ.get("SGLANG_PLUGINS") != _ENTRYPOINT_NAME:
        raise RuntimeError(
            f"SGLANG_PLUGINS must be exactly {_ENTRYPOINT_NAME!r}; "
            "the canonical adapter cannot share a hook registry"
        )
    force_miss = os.environ.get("SGLANG_RADIX_FORCE_MISS")
    if force_miss is not None:
        normalized = force_miss.lower()
        if normalized not in ("true", "1", "yes", "y", "false", "0", "no", "n"):
            raise RuntimeError("SGLANG_RADIX_FORCE_MISS is not a valid boolean")
        if normalized in ("true", "1", "yes", "y"):
            raise RuntimeError("OrbitKV does not support SGLANG_RADIX_FORCE_MISS")


def _preflight_hook_targets(hook_registry: Any) -> None:
    already_patched = [
        target for target in HOOK_TARGETS if target in hook_registry._patched
    ]
    if already_patched:
        raise RuntimeError(
            "canonical hook targets were mutated before OrbitKV activation: "
            + ", ".join(already_patched)
        )
    for target in HOOK_TARGETS:
        value = _resolve_attribute(target)
        if not callable(value):
            raise TypeError(f"pinned hook target is not callable: {target}")


def _resolve_attribute(path: str) -> Any:
    object_path, attribute = path.rsplit(".", 1)
    return getattr(pkgutil.resolve_name(object_path), attribute)


def _validate_propagated_aliases() -> None:
    stale = [
        alias
        for alias, definition in _PROPAGATED_ALIASES
        if _resolve_attribute(alias) is not _resolve_attribute(definition)
    ]
    if stale:
        raise RuntimeError(
            "SGLang retained stale imported KV authority aliases: " + ", ".join(stale)
        )


def _validate_batch(batch: Any) -> None:
    config = _config()
    if bool(getattr(batch, "enable_overlap", False)):
        raise RuntimeError("OrbitKV does not support overlap scheduling")
    if not bool(batch.spec_algorithm.is_none()):
        raise RuntimeError("OrbitKV does not support speculative decoding")
    from .prefix_cache import OrbitKvPrefixCache

    if type(batch.tree_cache) is not OrbitKvPrefixCache:
        raise RuntimeError("OrbitKV requires its registered radix-cache backend")
    supports_swa = bool(batch.tree_cache.supports_swa())
    requires_swa = _config().sliding_class is not None
    if supports_swa != requires_swa:
        raise RuntimeError("SGLang cache type differs from the compiled KV classes")
    if batch.tree_cache.token_to_kv_pool_allocator is not _state._ALLOCATOR:
        raise RuntimeError("SGLang batch references a foreign KV allocator")
    if int(batch.tree_cache.page_size) != config.page_tokens:
        raise RuntimeError("OrbitKV and SGLang page sizes differ")
    if bool(getattr(batch.model_config, "is_encoder_decoder", False)):
        raise RuntimeError("OrbitKV does not support encoder-decoder models")
    if bool(getattr(batch, "is_dllm", lambda: False)()):
        raise RuntimeError("OrbitKV does not support diffusion models")


def _integer_vector(name: str, value: Any, expected: int) -> tuple[int, ...]:
    if value is None:
        raise RuntimeError(f"SGLang {name} is missing")
    try:
        if hasattr(value, "detach"):
            if str(getattr(value, "device", "cpu")) != "cpu":
                raise RuntimeError(f"SGLang {name} must be a CPU mirror")
            raw = value.detach().tolist()
        else:
            raw = list(value)
    except Exception as error:
        raise RuntimeError(f"SGLang {name} is not a readable vector") from error
    if not isinstance(raw, (list, tuple)) or len(raw) != expected:
        raise RuntimeError(f"SGLang {name} cardinality differs from the request batch")
    result: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise RuntimeError(f"SGLang {name} must contain integers")
        result.append(int(item))
    return tuple(result)


def _validate_device_vector(
    name: str, value: Any, expected: int, batch_device: Any
) -> None:
    import torch

    if value is None or not isinstance(value, torch.Tensor):
        raise RuntimeError(f"SGLang {name} must be a device tensor")
    if value.ndim != 1 or int(value.numel()) != expected:
        raise RuntimeError(f"SGLang {name} cardinality differs from the request batch")
    if value.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"SGLang {name} must use an integer tensor dtype")
    actual_device = value.device
    expected_device = torch.device(batch_device)
    if actual_device.type != expected_device.type or (
        expected_device.index is not None
        and actual_device.index != expected_device.index
    ):
        raise RuntimeError(f"SGLang {name} is on a foreign device")


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise RuntimeError(f"SGLang {name} must be a positive integer")
    return int(value)


def _preflight_extend_batch(
    batch: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    batch_size = len(batch.reqs)
    if batch_size <= 0:
        raise RuntimeError("OrbitKV cannot allocate an empty extend batch")
    prefix_lens = _integer_vector("prefix_lens", batch.prefix_lens, batch_size)
    extend_lens = _integer_vector("extend_lens", batch.extend_lens, batch_size)
    targets = _integer_vector("seq_lens_cpu", batch.seq_lens_cpu, batch_size)
    _validate_device_vector("seq_lens", batch.seq_lens, batch_size, batch.device)
    if any(value < 0 for value in prefix_lens):
        raise RuntimeError("SGLang prefix lengths must be nonnegative")
    if any(value <= 0 for value in extend_lens):
        raise RuntimeError("SGLang extend lengths must be positive")
    if any(
        target != prefix + extension
        for prefix, extension, target in zip(
            prefix_lens, extend_lens, targets, strict=True
        )
    ):
        raise RuntimeError("SGLang extend boundaries disagree with extend lengths")
    extend_num_tokens = _positive_integer("extend_num_tokens", batch.extend_num_tokens)
    if extend_num_tokens != sum(extend_lens):
        raise RuntimeError("SGLang extend_num_tokens differs from extend_lens")
    maximum = int(batch.req_to_token_pool.max_context_len)
    for req, prefix, target in zip(batch.reqs, prefix_lens, targets, strict=True):
        if target > maximum:
            raise RuntimeError("SGLang extend boundary exceeds ReqToToken capacity")
        try:
            prefix_entries = len(req.prefix_indices)
        except Exception as error:
            raise RuntimeError("SGLang request prefix mirror is unreadable") from error
        if prefix_entries != prefix:
            raise RuntimeError(
                "SGLang request prefix mirror length differs from prefix_lens"
            )
    return prefix_lens, extend_lens, targets


def _preflight_decode_batch(batch: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    batch_size = len(batch.reqs)
    if batch_size <= 0:
        raise RuntimeError("OrbitKV cannot allocate an empty decode batch")
    previous = _integer_vector("seq_lens_cpu", batch.seq_lens_cpu, batch_size)
    _validate_device_vector("seq_lens", batch.seq_lens, batch_size, batch.device)
    _validate_device_vector(
        "req_pool_indices", batch.req_pool_indices, batch_size, batch.device
    )
    req_pool_indices = _integer_vector(
        "req_pool_indices_cpu", batch.req_pool_indices_cpu, batch_size
    )
    if any(value < 0 for value in previous):
        raise RuntimeError("SGLang decode sequence lengths must be nonnegative")
    if any(value <= 0 for value in req_pool_indices):
        raise RuntimeError("SGLang request-pool indices must exclude the dummy row")
    row_capacity = int(batch.req_to_token_pool.req_to_token.shape[0])
    maximum = int(batch.req_to_token_pool.max_context_len)
    if len(set(req_pool_indices)) != batch_size:
        raise RuntimeError("SGLang decode request-pool indices alias")
    for req, request_pool_index, boundary in zip(
        batch.reqs, req_pool_indices, previous, strict=True
    ):
        if request_pool_index >= row_capacity:
            raise RuntimeError("SGLang decode request-pool index is out of range")
        if boundary + 1 > maximum:
            raise RuntimeError("SGLang decode boundary exceeds ReqToToken capacity")
        if req.req_pool_idx is None or int(req.req_pool_idx) != request_pool_index:
            raise RuntimeError("SGLang request-pool identity differs from the batch")
        if req.kv is None or int(req.kv.kv_allocated_len) != boundary:
            raise RuntimeError("SGLang request KV boundary differs from the batch")
    _runtime().bind_request_rows(
        tuple(
            (_request_key(req), request_pool_index, False)
            for req, request_pool_index in zip(
                batch.reqs, req_pool_indices, strict=True
            )
        )
    )
    return previous, req_pool_indices


def _dtype_bytes(dtype: Any) -> int:
    import torch

    try:
        size = int(torch.empty((), dtype=dtype, device="cpu").element_size())
    except Exception as error:
        raise RuntimeError(
            "cannot determine the SGLang KV cache element size"
        ) from error
    if size <= 0:
        raise RuntimeError("SGLang KV cache dtype has an invalid element size")
    return size


def _is_cuda_platform() -> bool:
    from sglang.srt.platforms import current_platform

    return bool(current_platform.is_cuda())


def _uses_hnd_kv_cache() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_USE_HND_KVCACHE.get())


def _checkpoint_architecture(model: Any) -> str:
    architectures = list(getattr(model.hf_config, "architectures", ()) or ())
    if len(architectures) != 1 or architectures[0] not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError("OrbitKV supports only Qwen2 and GPT-OSS checkpoints")
    return architectures[0]


def _validate_attention_backend_contract(configurator: Any) -> str:
    architecture = _checkpoint_architecture(configurator.model_config)
    backends = tuple(configurator.server_args.get_attention_backends())
    if len(backends) != 2:
        raise RuntimeError("SGLang returned an invalid attention backend pair")
    expected = ATTENTION_BACKENDS_BY_ARCHITECTURE[architecture]
    if backends != expected:
        raise RuntimeError(
            f"{architecture} requires SGLang attention backends {expected}, "
            f"got {backends}"
        )
    return architecture


def _validate_checkpoint_geometry(configurator: Any) -> None:
    plan = _config()
    model = configurator.model_config
    text = model.hf_text_config
    if int(text.num_hidden_layers) != plan.num_hidden_layers:
        raise RuntimeError("checkpoint layer count differs from KvPlanInput.layers")
    architecture = _checkpoint_architecture(model)
    retentions = tuple(item.retention for item in plan.classes)
    all_layers = tuple(range(plan.num_hidden_layers))
    if architecture == "Qwen2ForCausalLM":
        if retentions != ("full",) or plan.classes[0].layers != all_layers:
            raise RuntimeError("Qwen2 requires one Full class covering every layer")
        if bool(model.is_hybrid_swa):
            raise RuntimeError("Qwen2 unexpectedly resolved hybrid SWA storage")
    elif architecture == "GptOssForCausalLM":
        if retentions != ("full", "sliding") or not bool(model.is_hybrid_swa):
            raise RuntimeError("GPT-OSS requires ordered Full+SWA classes")
        full, sliding = plan.classes
        if (
            tuple(model.full_attention_layer_ids) != full.layers
            or tuple(model.swa_attention_layer_ids) != sliding.layers
        ):
            raise RuntimeError("GPT-OSS layer partition differs from KvPlanInput")
        if int(model.sliding_window_size) != int(sliding.window_tokens):
            raise RuntimeError("GPT-OSS sliding window differs from KvPlanInput")
        if bool(getattr(model, "disable_hybrid_swa_memory", False)):
            raise RuntimeError("GPT-OSS hybrid SWA memory is disabled")

    if bool(getattr(model, "is_deepseek_v4_arch", False)) or bool(
        getattr(model, "is_hybrid_swa_compress", False)
    ):
        raise RuntimeError("OrbitKV does not support compressed attention storage")
    if getattr(model, "attention_chunk_size", None) is not None:
        raise RuntimeError("OrbitKV does not support attention chunking")

    kv_heads = int(text.num_key_value_heads)
    dtype_bytes = _dtype_bytes(configurator.kv_cache_dtype)
    full_bytes = (
        kv_heads
        * (int(model.head_dim) + int(getattr(model, "v_head_dim", model.head_dim)))
        * dtype_bytes
    )
    swa_bytes = (
        kv_heads
        * (
            int(getattr(model, "swa_head_dim", model.head_dim))
            + int(
                getattr(
                    model,
                    "swa_v_head_dim",
                    getattr(model, "v_head_dim", model.head_dim),
                )
            )
        )
        * dtype_bytes
    )
    for class_config in plan.classes:
        actual = full_bytes if class_config.retention == "full" else swa_bytes
        if class_config.bytes_per_token_per_layer != actual:
            raise RuntimeError(
                f"{class_config.name} KV geometry differs from KvPlanInput"
            )


def _resolve_runtime_limits(configurator: Any) -> RuntimeLimits:
    server = configurator.server_args
    maximum_requests = int(server.max_running_requests or 0)
    chunk_tokens = int(server.chunked_prefill_size or 0)
    context_tokens = int(configurator.model_config.context_len or 0)
    if maximum_requests <= 0 or chunk_tokens <= 0 or context_tokens <= 0:
        raise RuntimeError(
            "OrbitKV requires explicit positive max_running_requests, "
            "chunked_prefill_size, and model context length"
        )
    return RuntimeLimits(
        maximum_running_requests=maximum_requests,
        chunked_prefill_tokens=chunk_tokens,
        maximum_context_tokens=context_tokens,
    )


def _validate_physical_pool(
    pool: Any, *, expected_tokens: int, expected_dtype: Any, name: str
) -> None:
    if pool is None:
        raise RuntimeError(f"SGLang did not construct the {name} KV pool")
    if int(pool.size) != int(expected_tokens):
        raise RuntimeError(f"SGLang {name} KV pool capacity changed")
    if int(pool.page_size) != _config().page_tokens:
        raise RuntimeError(f"SGLang {name} KV pool page size changed")
    if pool.dtype is not expected_dtype:
        raise RuntimeError(f"SGLang {name} KV pool dtype changed")
    if getattr(pool, "kv_cache_layout", None) != "nhd":
        raise RuntimeError(f"SGLang {name} KV pool is not NHD")


def _validate_configurator(
    original_fn: Callable[..., Any], configurator: Any, *args: Any, **kwargs: Any
) -> Any:
    import torch

    config = _config()
    server = configurator.server_args
    graph = server.cuda_graph_config
    _validate_attention_backend_contract(configurator)
    required = {
        "CUDA platform": _is_cuda_platform(),
        "CUDA device": str(configurator.device).startswith("cuda"),
        "bfloat16 KV cache": configurator.kv_cache_dtype is torch.bfloat16,
        "NHD KV layout": not _uses_hnd_kv_cache(),
        "radix cache": not bool(server.disable_radix_cache),
        "radix backend": getattr(server, "radix_cache_backend", None) == "orbitkv",
        "FCFS scheduling": getattr(server, "schedule_policy", None) == "fcfs",
        "thinking-cache trimming": not bool(
            getattr(server, "strip_thinking_cache", False)
        ),
        "LoRA": not bool(getattr(server, "enable_lora", False))
        and not bool(getattr(server, "lora_paths", ())),
        "overlap schedule": bool(server.disable_overlap_schedule),
        "disaggregation": getattr(
            server.disaggregation_mode, "value", server.disaggregation_mode
        )
        == "null",
        "speculative decoding": bool(configurator.spec_algorithm.is_none()),
        "page size": int(configurator.page_size) == config.page_tokens,
        "chunked prefill": int(server.chunked_prefill_size or 0) > 0,
        "maximum requests": int(server.max_running_requests or 0) > 0,
        "mixed chunked prefill": not bool(server.enable_mixed_chunk),
        "decode CUDA Graph": str(graph.decode.backend) == "disabled",
        "prefill CUDA Graph": str(graph.prefill.backend) == "disabled",
        "hierarchical cache": not bool(server.enable_hierarchical_cache),
        "streaming session": not bool(server.enable_streaming_session),
        "unified memory": not bool(server.enable_unified_memory),
        "PD multiplexing": not bool(server.enable_pdmux),
        "decode context parallelism": int(server.dcp_size) == 1,
        "tensor parallelism": int(server.tp_size) == 1,
        "pipeline parallelism": int(server.pp_size) == 1,
        "DP attention": not bool(server.enable_dp_attention),
        "LMCache": not bool(server.enable_lmcache),
        "HiSparse": not bool(server.enable_hisparse),
        "page-major KV": not bool(server.enable_page_major_kv_layout),
        "embedding mode": not bool(server.is_embedding),
        "draft worker": not bool(configurator.is_draft_worker),
        "MLA": not bool(configurator.use_mla_backend),
        "hybrid compression": not bool(configurator.is_hybrid_swa_compress),
        "Mamba": configurator.mambaish_config is None
        and configurator.hybrid_gdn_config is None,
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError("OrbitKV runtime contract failed: " + ", ".join(failed))
    _validate_checkpoint_geometry(configurator)
    resolved_limits = _resolve_runtime_limits(configurator)
    if _state._LIMITS is not None and _state._LIMITS != resolved_limits:
        raise RuntimeError("OrbitKV runtime capacities changed after initialization")
    _state._LIMITS = resolved_limits
    result = original_fn(configurator, *args, **kwargs)
    allocator = result.token_to_kv_pool_allocator
    if allocator is not _state._ALLOCATOR or _state._RUNTIME is None:
        raise RuntimeError("SGLang did not install the OrbitKV arena facades")
    if int(allocator.page_size) != config.page_tokens:
        raise RuntimeError("OrbitKV arena page size changed during configuration")
    kv_pool = allocator.get_kvcache()
    if config.full_class is not None and config.sliding_class is not None:
        if int(result.full_max_total_num_tokens) != int(allocator.size_full) or int(
            result.swa_max_total_num_tokens
        ) != int(allocator.size_swa):
            raise RuntimeError("SGLang Hybrid result capacity differs from its arenas")
        _validate_physical_pool(
            getattr(kv_pool, "full_kv_pool", None),
            expected_tokens=allocator.size_full,
            expected_dtype=configurator.kv_cache_dtype,
            name="Full",
        )
        _validate_physical_pool(
            getattr(kv_pool, "swa_kv_pool", None),
            expected_tokens=allocator.size_swa,
            expected_dtype=configurator.kv_cache_dtype,
            name="SWA",
        )
    elif config.full_class is not None:
        if int(result.max_total_num_tokens) != int(allocator.size):
            raise RuntimeError("SGLang Full result capacity differs from its arena")
        _validate_physical_pool(
            kv_pool,
            expected_tokens=allocator.size,
            expected_dtype=configurator.kv_cache_dtype,
            name="Full",
        )
    else:
        if int(result.swa_max_total_num_tokens) != int(allocator.size_swa):
            raise RuntimeError("SGLang SWA result capacity differs from its arena")
        _validate_physical_pool(
            getattr(kv_pool, "swa_kv_pool", None),
            expected_tokens=allocator.size_swa,
            expected_dtype=configurator.kv_cache_dtype,
            name="SWA",
        )
    if int(result.max_running_requests) != resolved_limits.maximum_running_requests:
        raise RuntimeError("SGLang request capacity differs from the manager capacity")
    return result
