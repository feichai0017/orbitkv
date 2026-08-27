from __future__ import annotations

import os
import pkgutil
from dataclasses import dataclass
from math import prod
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

from ..pinned import validate_patched_checkout
from ..runtime_policy import GDN_FIXED_STATE_BACKEND_PROFILE
from ..executor_capabilities import admit_runtime_config
from . import state as _state
from .state import RuntimeLimits, _config, _request_key, _runtime

SUPPORTED_ATTENTION_BACKENDS = frozenset(("flashinfer", "fa3"))
_SPARSE_RETAINED_SLOT_BACKENDS = frozenset(("flashinfer",))
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
    "sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new",
    "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.alloc",
    "sglang.srt.model_executor.model_runner.ModelRunner._maybe_execute_deferred_mamba_cow_and_clear",
    "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.clear",
    "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.free_mamba_cache",
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
    if (
        len(architectures) != 1
        or not isinstance(architectures[0], str)
        or not architectures[0]
    ):
        raise RuntimeError("OrbitKV requires one explicit checkpoint architecture")
    return architectures[0]


def _validate_attention_backend_contract(configurator: Any) -> str:
    architecture = _checkpoint_architecture(configurator.model_config)
    backends = tuple(configurator.server_args.get_attention_backends())
    if bool(getattr(configurator, "use_mla_backend", False)):
        if (
            len(backends) != 2
            or backends[0] != backends[1]
            or backends[0] not in SUPPORTED_ATTENTION_BACKENDS
        ):
            raise RuntimeError(
                f"{architecture} requires one uniform MLA backend from "
                f"{sorted(SUPPORTED_ATTENTION_BACKENDS)}, got {backends}"
            )
        return architecture
    if (
        len(backends) != 2
        or backends[0] != backends[1]
        or backends[0] not in SUPPORTED_ATTENTION_BACKENDS
    ):
        raise RuntimeError(
            f"{architecture} requires one uniform token-KV backend from "
            f"{sorted(SUPPORTED_ATTENTION_BACKENDS)}, got {backends}"
        )
    if bool(getattr(configurator.model_config, "has_attention_sinks", False)) and (
        backends != ("fa3", "fa3")
    ):
        raise RuntimeError(f"{architecture} attention sinks require SGLang FA3")
    return architecture


def _validate_token_reclamation_backend_contract(configurator: Any) -> None:
    config = _config()
    policy = config.token_reclamation
    if policy.mode != "naive":
        return

    retained = int(policy.retained_per_page)
    page_tokens = int(config.page_tokens)
    if not 0 < retained < page_tokens:
        return
    backends = tuple(configurator.server_args.get_attention_backends())
    incompatible = tuple(
        sorted(set(backends) - _SPARSE_RETAINED_SLOT_BACKENDS)
    )
    if incompatible:
        raise RuntimeError(
            "token_reclamation.mode='naive' requires token-granular KV "
            "addressing for sparse retained slots; selected page-granular "
            f"attention backend(s) {incompatible} cannot represent "
            f"retained_per_page={retained} within page_tokens={page_tokens}; "
            "use mode='relocate' or a backend with sparse-slot support"
        )


@dataclass(frozen=True, slots=True)
class _FixedStateRuntimeGeometry:
    layers: tuple[int, ...]
    conv_shapes: tuple[tuple[int, ...], ...]
    temporal_shape: tuple[int, ...]
    conv_dtype: Any
    temporal_dtype: Any
    conv_bytes_per_layer: int
    recurrent_bytes_per_layer: int
    kernel_width: int


def _positive_shape(name: str, value: Any) -> tuple[int, ...]:
    try:
        values = tuple(value)
    except Exception as error:
        raise RuntimeError(f"SGLang {name} is not a readable shape") from error
    if not values:
        raise RuntimeError(f"SGLang {name} must be nonempty")
    return tuple(_positive_integer(f"{name} dimension", item) for item in values)


def _fixed_state_runtime_geometry(params: Any) -> _FixedStateRuntimeGeometry:
    shape = getattr(params, "shape", None)
    dtype = getattr(params, "dtype", None)
    try:
        raw_conv_shapes = tuple(shape.conv)
    except Exception as error:
        raise RuntimeError(
            "SGLang fixed-state convolution geometry is missing"
        ) from error
    if not raw_conv_shapes:
        raise RuntimeError("SGLang fixed-state convolution geometry is empty")
    conv_shapes = tuple(
        _positive_shape(f"fixed-state convolution shape {index}", value)
        for index, value in enumerate(raw_conv_shapes)
    )
    temporal_shape = _positive_shape(
        "fixed-state recurrent shape", getattr(shape, "temporal", None)
    )
    conv_dtype = getattr(dtype, "conv", None)
    temporal_dtype = getattr(dtype, "temporal", None)
    return _FixedStateRuntimeGeometry(
        tuple(params.layers),
        conv_shapes,
        temporal_shape,
        conv_dtype,
        temporal_dtype,
        sum(prod(value) for value in conv_shapes) * _dtype_bytes(conv_dtype),
        prod(temporal_shape) * _dtype_bytes(temporal_dtype),
        _positive_integer(
            "fixed-state convolution kernel width",
            getattr(shape, "conv_kernel", None),
        ),
    )


def _is_exact_gdn_fixed_state_profile(configurator: Any) -> bool:
    fixed_states = _config().fixed_states
    recurrent = tuple(
        item for item in fixed_states if item.kind != "convolution"
    )
    convolution = tuple(
        item for item in fixed_states if item.kind == "convolution"
    )
    mambaish = getattr(configurator, "mambaish_config", None)
    return (
        len(recurrent) == 1
        and recurrent[0].kind == "gdn"
        and len(convolution) == 1
        and recurrent[0].layers == convolution[0].layers
        and recurrent[0].checkpoint_slots_per_request == 2
        and convolution[0].checkpoint_slots_per_request == 2
        and mambaish is not None
        and getattr(configurator, "hybrid_gdn_config", None) is mambaish
    )


def _validate_checkpoint_geometry(configurator: Any) -> None:
    plan = _config()
    model = configurator.model_config
    text = model.hf_text_config
    if int(text.num_hidden_layers) != plan.num_hidden_layers:
        raise RuntimeError("checkpoint layer count differs from KvPlanInput.layers")
    _checkpoint_architecture(model)
    if bool(getattr(model, "is_deepseek_v4_arch", False)) or bool(
        getattr(model, "is_hybrid_swa_compress", False)
    ):
        raise RuntimeError("OrbitKV does not support compressed attention storage")
    if getattr(model, "attention_chunk_size", None) is not None:
        raise RuntimeError("OrbitKV does not support attention chunking")
    retentions = tuple(item.retention for item in plan.classes)
    all_layers = tuple(range(plan.num_hidden_layers))
    token_layers = tuple(
        sorted(layer for item in plan.classes for layer in item.layers)
    )
    storage = {item.storage for item in plan.classes}
    if storage == {"latent_kv"}:
        from sglang.srt.configs.model_config import is_deepseek_dsa

        if (
            len(plan.classes) != 1
            or plan.classes[0].retention != "full"
            or plan.classes[0].layers != token_layers
            or bool(plan.fixed_states)
            or not bool(configurator.use_mla_backend)
            or bool(model.is_hybrid_swa)
            or is_deepseek_dsa(model.hf_config)
        ):
            raise RuntimeError(
                "supported MLA profile requires one Full latent_kv class covering every layer"
            )
        latent = int(model.kv_lora_rank) * _dtype_bytes(configurator.kv_cache_dtype)
        rope = int(model.qk_rope_head_dim) * _dtype_bytes(configurator.kv_cache_dtype)
        class_config = plan.classes[0]
        if class_config.components != (("latent", latent), ("rope", rope)):
            raise RuntimeError("MLA latent/RoPE geometry differs from KvPlanInput")
        if class_config.bytes_per_token_per_layer != latent + rope:
            raise RuntimeError("MLA aggregate geometry differs from KvPlanInput")
        return
    if storage != {"token_kv"}:
        raise RuntimeError("OrbitKV does not support mixed token storage backends")
    if retentions == ("full",):
        if plan.classes[0].layers != token_layers:
            raise RuntimeError("Full profile token layer identity changed")
        if not plan.fixed_states and token_layers != all_layers:
            raise RuntimeError("Full profile requires one class covering every layer")
        if bool(model.is_hybrid_swa) and not plan.fixed_states:
            raise RuntimeError("Full profile unexpectedly resolved hybrid SWA storage")
    elif retentions == ("sliding",):
        sliding = plan.classes[0]
        if (
            not bool(model.is_hybrid_swa)
            or tuple(getattr(model, "full_attention_layer_ids", ()))
            or tuple(getattr(model, "swa_attention_layer_ids", ()))
            != sliding.layers
            or token_layers != all_layers
            or bool(plan.fixed_states)
        ):
            raise RuntimeError(
                "pure sliding profile requires every model layer to use SWA"
            )
        if int(model.sliding_window_size) != int(sliding.window_tokens):
            raise RuntimeError(
                "SGLang sliding window differs from KvPlanInput"
            )
        if bool(getattr(model, "disable_hybrid_swa_memory", False)):
            raise RuntimeError("SGLang hybrid SWA memory is disabled")
    elif retentions == ("full", "sliding"):
        if not bool(model.is_hybrid_swa):
            raise RuntimeError("Hybrid profile requires ordered Full+SWA classes")
        full, sliding = plan.classes
        if (
            tuple(model.full_attention_layer_ids) != full.layers
            or tuple(model.swa_attention_layer_ids) != sliding.layers
        ):
            raise RuntimeError("SGLang Full/SWA partition differs from KvPlanInput")
        if int(model.sliding_window_size) != int(sliding.window_tokens):
            raise RuntimeError("SGLang sliding window differs from KvPlanInput")
        if bool(getattr(model, "disable_hybrid_swa_memory", False)):
            raise RuntimeError("SGLang hybrid SWA memory is disabled")
    else:
        raise RuntimeError(
            "OrbitKV SGLang supports Full, pure sliding, or ordered Full+SWA"
        )

    dtype_bytes = _dtype_bytes(configurator.kv_cache_dtype)
    kv_heads = int(text.num_key_value_heads)
    full_bytes = kv_heads * (
        int(model.head_dim) + int(getattr(model, "v_head_dim", model.head_dim))
    ) * dtype_bytes
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
    if plan.fixed_states:
        if configurator.mambaish_config is None:
            raise RuntimeError(
                "attention-state plan requires SGLang fixed-state storage"
            )
        params = configurator.mambaish_config.mamba2_cache_params
        fixed_layers = tuple(
            sorted({layer for item in plan.fixed_states for layer in item.layers})
        )
        if tuple(params.layers) != fixed_layers:
            raise RuntimeError("SGLang fixed-state layers differ from the state plan")
        if int(params.mamba_cache_per_req) != plan.fixed_state_byte_count:
            raise RuntimeError(
                "SGLang fixed-state byte geometry differs from the state plan"
            )
        if (
            set(token_layers) | set(fixed_layers) != set(all_layers)
            or set(token_layers) & set(fixed_layers)
        ):
            raise RuntimeError("hybrid state roles do not cover the model")
        if tuple(
            getattr(configurator.mambaish_config, "full_attention_layer_ids", ())
        ) != token_layers:
            raise RuntimeError("SGLang token-state layers differ from the state plan")
        recurrent = tuple(
            item for item in plan.fixed_states if item.kind != "convolution"
        )
        convolution = tuple(
            item for item in plan.fixed_states if item.kind == "convolution"
        )
        runtime_is_kda = bool(getattr(params, "is_kda", False))
        if runtime_is_kda:
            raise RuntimeError(
                "SGLang KDA family differs from the fixed-state plan"
            )
        hybrid_gdn = getattr(configurator, "hybrid_gdn_config", None)
        mamba_profile = (
            bool(recurrent)
            and all(item.kind == "mamba" for item in recurrent)
            and hybrid_gdn is None
        )
        gdn_profile = _is_exact_gdn_fixed_state_profile(configurator)
        if not mamba_profile and not gdn_profile:
            raise RuntimeError(
                "fixed-state profile requires the Mamba family or exact GDN "
                "recurrent+convolution components"
            )
        if gdn_profile:
            import torch

            geometry = _fixed_state_runtime_geometry(params)
            runtime_kernel_width = _positive_integer(
                "GDN runtime convolution kernel width",
                getattr(hybrid_gdn, "linear_conv_kernel_dim", None),
            )
            plan_kernel_width = convolution[0].kernel_width
            if (
                geometry.layers != recurrent[0].layers
                or geometry.conv_dtype is not torch.bfloat16
                or geometry.temporal_dtype is not torch.float32
                or recurrent[0].state_bytes_per_layer
                != geometry.recurrent_bytes_per_layer
                or convolution[0].state_bytes_per_layer
                != geometry.conv_bytes_per_layer
                or plan_kernel_width != runtime_kernel_width
                or plan_kernel_width != geometry.kernel_width
                or len(geometry.conv_shapes) != 1
                or len(geometry.conv_shapes[0]) != 2
                or geometry.conv_shapes[0][-1] + 1 != plan_kernel_width
            ):
                raise RuntimeError(
                    "SGLang GDN component geometry differs from the state plan"
                )


def _validate_gdn_fixed_state_backend_contract(configurator: Any) -> None:
    if not _is_exact_gdn_fixed_state_profile(configurator):
        return

    import torch

    profile = GDN_FIXED_STATE_BACKEND_PROFILE
    server = configurator.server_args
    mambaish = getattr(configurator, "mambaish_config", None)
    params = getattr(mambaish, "mamba2_cache_params", None)
    actual_dtype = getattr(getattr(params, "dtype", None), "temporal", None)
    required = {
        "Full attention FA3": tuple(server.get_attention_backends())
        == (profile["attention_backend"],) * 2,
        "linear_attn_backend=triton": getattr(
            server, "linear_attn_backend", None
        )
        == profile["linear_attn_backend"],
        "linear_attn_decode_backend=triton": getattr(
            server, "linear_attn_decode_backend", None
        )
        == profile["linear_attn_decode_backend"],
        "linear_attn_prefill_backend=triton": getattr(
            server, "linear_attn_prefill_backend", None
        )
        == profile["linear_attn_prefill_backend"],
        "mamba_ssm_dtype=float32": getattr(server, "mamba_ssm_dtype", None)
        == profile["mamba_ssm_dtype"],
        "mamba2_cache_params.dtype.temporal=float32": actual_dtype
        is getattr(torch, profile["mamba_ssm_dtype"]),
        "mamba_radix_cache_strategy=no_buffer": getattr(
            server, "mamba_radix_cache_strategy", None
        )
        == profile["mamba_radix_cache_strategy"],
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError(
            "GDN fixed-state backend contract failed: "
            + ", ".join(failed)
        )


def _validate_radix_cache_contract(configurator: Any) -> None:
    server = configurator.server_args
    requires_disabled_radix = _state._requires_disabled_radix_cache()
    if bool(server.disable_radix_cache) != requires_disabled_radix:
        required = "true" if requires_disabled_radix else "false"
        raise RuntimeError(
            f"--disable-radix-cache must be {required} for this OrbitKV plan"
        )
    if getattr(server, "radix_cache_backend", None) != "orbitkv":
        raise RuntimeError("--radix-cache-backend must remain 'orbitkv'")


def _validate_fixed_state_options(configurator: Any) -> None:
    if not _config().fixed_states:
        return
    server = configurator.server_args
    unsupported = {
        "Mamba extra buffer": bool(server.enable_mamba_extra_buffer()),
        "Mamba lazy extra buffer": bool(server.enable_mamba_extra_buffer_lazy()),
        "ReplaySSM": bool(getattr(server, "enable_linear_replayssm", False)),
        "ReplaySSM speculation": bool(
            getattr(server, "enable_linear_replayssm_spec", False)
        ),
        "Mamba int8 checkpoint": bool(
            getattr(server, "enable_int8_mamba_checkpoint", False)
        ),
    }
    failed = [name for name, enabled in unsupported.items() if enabled]
    if failed:
        raise RuntimeError(
            "OrbitKV restricted fixed-state profile rejects: " + ", ".join(failed)
        )


def _validate_fixed_state_pool(req_pool: Any, config: Any, params: Any) -> None:
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    if (
        not isinstance(req_pool, HybridReqToTokenPool)
        or req_pool.mamba_ckpt_pool is not None
        or bool(req_pool.enable_mamba_extra_buffer)
        or bool(req_pool.enable_mamba_extra_buffer_lazy)
        or bool(req_pool.mamba_pool.enable_linear_replayssm)
        or bool(req_pool.mamba_pool.enable_linear_replayssm_spec)
        or not callable(getattr(req_pool.mamba_pool, "clear_slots", None))
        or not callable(getattr(req_pool.mamba_pool, "copy_from", None))
    ):
        raise RuntimeError("SGLang fixed-state pool is outside the restricted profile")
    mamba = req_pool.mamba_pool
    conv_tensors = tuple(mamba.mamba_cache.conv)
    temporal = mamba.mamba_cache.temporal
    tensors = conv_tensors + (temporal,)
    slot_count = int(req_pool.mamba_pool.size)
    geometry = _fixed_state_runtime_geometry(params)
    expected_conv_shapes = tuple(
        (len(geometry.layers), slot_count + 1, *shape)
        for shape in geometry.conv_shapes
    )
    expected_temporal_shape = (
        len(geometry.layers),
        slot_count + 1,
        *geometry.temporal_shape,
    )
    if (
        len(conv_tensors) != len(expected_conv_shapes)
        or any(
            tuple(tensor.shape) != expected_shape
            or tensor.dtype is not geometry.conv_dtype
            for tensor, expected_shape in zip(
                conv_tensors, expected_conv_shapes, strict=True
            )
        )
        or tuple(temporal.shape) != expected_temporal_shape
        or temporal.dtype is not geometry.temporal_dtype
    ):
        raise RuntimeError("SGLang fixed-state tensor geometry changed")
    conv_bytes = sum(
        int(tensor[0, 0].numel()) * int(tensor.element_size())
        for tensor in conv_tensors
        if int(tensor.numel()) > 0
    ) * int(tensors[0].shape[0])
    recurrent_bytes = (
        int(temporal[0, 0].numel())
        * int(temporal.element_size())
        * int(temporal.shape[0])
    )
    recurrent = tuple(
        item for item in config.fixed_states if item.kind != "convolution"
    )
    aggregate_mamba = (
        len(config.fixed_states) == 1
        and len(recurrent) == 1
        and recurrent[0].kind == "mamba"
    )
    if aggregate_mamba:
        geometry_matches = (
            conv_bytes + recurrent_bytes == config.fixed_state_byte_count
        )
    else:
        expected_conv = sum(
            item.byte_count
            for item in config.fixed_states
            if item.kind == "convolution"
        )
        expected_recurrent = sum(item.byte_count for item in recurrent)
        geometry_matches = (
            conv_bytes == expected_conv and recurrent_bytes == expected_recurrent
        )
    if not geometry_matches:
        raise RuntimeError(
            "SGLang fixed-state component geometry differs from the state plan"
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
    pool: Any,
    *,
    expected_tokens: int,
    expected_dtype: Any,
    name: str,
    storage: str = "token_kv",
) -> None:
    if pool is None:
        raise RuntimeError(f"SGLang did not construct the {name} KV pool")
    if int(pool.size) != int(expected_tokens):
        raise RuntimeError(f"SGLang {name} KV pool capacity changed")
    if int(pool.page_size) != _config().page_tokens:
        raise RuntimeError(f"SGLang {name} KV pool page size changed")
    if pool.dtype is not expected_dtype:
        raise RuntimeError(f"SGLang {name} KV pool dtype changed")
    if storage == "latent_kv":
        _validate_mla_pool_geometry(pool, _config().classes[0])
        return
    if getattr(pool, "kv_cache_layout", None) != "nhd":
        raise RuntimeError(f"SGLang {name} KV pool is not NHD")


def _validate_full_physical_pool(
    pool: Any, *, expected_tokens: int, expected_dtype: Any, storage: str
) -> None:
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    physical = pool
    if isinstance(pool, HybridLinearKVPool):
        expected_layers = _config().full_class.layers
        mapping = getattr(pool, "full_attention_layer_id_mapping", None)
        if (int(pool.size) != expected_tokens
                or int(pool.page_size) != _config().page_tokens
                or pool.dtype is not expected_dtype
                or bool(getattr(pool, "use_mla", True))
                or mapping != {layer: index for index, layer in enumerate(expected_layers)}):
            raise RuntimeError("SGLang hybrid-linear KV pool envelope changed")
        physical = pool.full_kv_pool
    _validate_physical_pool(
        physical, expected_tokens=expected_tokens, expected_dtype=expected_dtype,
        name="Full", storage=storage,
    )


def _validate_mla_pool_geometry(pool: Any, class_config: Any) -> None:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    expected = class_config.components_by_name
    dtype_bytes = _dtype_bytes(getattr(pool, "dtype", None))
    if (
        not isinstance(pool, MLATokenToKVPool)
        or int(getattr(pool, "layer_num", 0)) != len(class_config.layers)
        or len(getattr(pool, "kv_buffer", ())) != len(class_config.layers)
        or int(getattr(pool, "kv_lora_rank", 0)) * dtype_bytes
        != expected.get("latent")
        or int(getattr(pool, "qk_rope_head_dim", 0)) * dtype_bytes
        != expected.get("rope")
        or bool(getattr(pool, "use_dsa", False))
        or bool(getattr(pool, "dsa_kv_cache_store_fp8", False))
        or not callable(getattr(pool, "move_kv_cache", None))
    ):
        raise RuntimeError("SGLang MLA pool geometry changed")


def _validate_configurator(
    original_fn: Callable[..., Any], configurator: Any, *args: Any, **kwargs: Any
) -> Any:
    import torch

    config = _config()
    if config.runtime_manifest_path is not None:
        try:
            admit_runtime_config(config)
        except ValueError as error:
            raise RuntimeError(
                f"OrbitKV executor target admission failed: {error}"
            ) from error
    server = configurator.server_args
    graph = server.cuda_graph_config
    _validate_gdn_fixed_state_backend_contract(configurator)
    _validate_radix_cache_contract(configurator)
    _validate_attention_backend_contract(configurator)
    _validate_token_reclamation_backend_contract(configurator)
    required = {
        "CUDA platform": _is_cuda_platform(),
        "CUDA device": str(configurator.device).startswith("cuda"),
        "bfloat16 KV cache": configurator.kv_cache_dtype is torch.bfloat16,
        "NHD KV layout": not _uses_hnd_kv_cache(),
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
        "attention storage": bool(configurator.use_mla_backend)
        == all(item.storage == "latent_kv" for item in config.classes),
        "hybrid compression": not bool(configurator.is_hybrid_swa_compress),
        "Mamba/state plan": (configurator.mambaish_config is not None)
        == bool(config.fixed_states),
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError("OrbitKV runtime contract failed: " + ", ".join(failed))
    _validate_fixed_state_options(configurator)
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
        _validate_full_physical_pool(
            kv_pool,
            expected_tokens=allocator.size,
            expected_dtype=configurator.kv_cache_dtype,
            storage=config.full_class.storage,
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
    if config.fixed_states:
        req_pool = result.req_to_token_pool
        _validate_fixed_state_pool(
            req_pool, config, configurator.mambaish_config.mamba2_cache_params
        )
        if _state._FIXED_STATE is None:
            from .state import _new_fixed_state

            _new_fixed_state(req_pool, device_module=torch)
    if _state._DATA_PLANE is None and _state._uses_structured_data_plane():
        from .state import _new_data_plane

        _new_data_plane(kv_pool, device_module=torch)
    return result
