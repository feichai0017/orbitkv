from __future__ import annotations

import os
from dataclasses import dataclass
from math import prod
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

from ..pinned import validate_patched_source
from ..runtime_policy import GDN_FIXED_STATE_BACKEND_PROFILE
from ..runtime_admission import admit_runtime_config
from . import state as _state
from .state import RuntimeLimits, _config, _request_key, _runtime

_SPARSE_RETAINED_SLOT_BACKENDS = frozenset(("flashinfer",))


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
    validate_patched_source(root)


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
    chunked = getattr(_config(), "chunked_class", None)
    if chunked is not None:
        chunk_tokens = _positive_integer(
            "compiled chunk size", getattr(chunked, "chunk_tokens", None)
        )
        for previous, target in zip(prefix_lens, targets, strict=True):
            if previous // chunk_tokens != (target - 1) // chunk_tokens:
                raise RuntimeError(
                    "SGLang extend crosses the compiled attention chunk boundary"
                )
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


def _validate_attention_backend_contract(configurator: Any) -> None:
    backends = tuple(configurator.server_args.get_attention_backends())
    config = _config()
    latent = bool(config.classes) and all(
        item.storage == "latent_kv" for item in config.classes
    )
    fixed = bool(config.fixed_states)
    chunked = getattr(config, "chunked_class", None) is not None
    sinks = bool(getattr(configurator.model_config, "has_attention_sinks", False))
    if bool(getattr(configurator, "use_mla_backend", False)) != latent:
        raise RuntimeError(
            "SGLang MLA mode differs from the compiled attention storage"
        )
    if len(backends) != 2 or backends[0] != backends[1]:
        capability = "latent-KV" if latent else "token-KV"
        raise RuntimeError(
            f"{capability} capability requires one uniform SGLang backend, "
            f"got {backends}"
        )
    expected = "flashinfer" if latent else "fa3"
    if backends != (expected, expected):
        backend_name = "FlashInfer" if expected == "flashinfer" else "FA3"
        capability = (
            "latent-KV"
            if latent
            else "fixed-state"
            if fixed
            else "chunked local attention"
            if chunked
            else "attention sinks"
            if sinks
            else "token-KV"
        )
        raise RuntimeError(
            f"{capability} capability requires SGLang {backend_name}, got {backends}"
        )


def _validate_chunked_scheduler_contract(configurator: Any) -> None:
    chunked = getattr(_config(), "chunked_class", None)
    if chunked is None:
        return

    chunk_tokens = _positive_integer(
        "compiled chunk size", getattr(chunked, "chunk_tokens", None)
    )
    server = configurator.server_args
    chunked_prefill_size = getattr(server, "chunked_prefill_size", None)
    prefill_max_requests = getattr(server, "prefill_max_requests", None)
    max_prefill_tokens = getattr(server, "max_prefill_tokens", None)
    failed: list[str] = []
    if (
        isinstance(chunked_prefill_size, bool)
        or not isinstance(chunked_prefill_size, Integral)
        or int(chunked_prefill_size) != chunk_tokens
    ):
        failed.append(f"chunked_prefill_size={chunk_tokens}")
    if (
        isinstance(prefill_max_requests, bool)
        or not isinstance(prefill_max_requests, Integral)
        or int(prefill_max_requests) != 1
    ):
        failed.append("prefill_max_requests=1")
    if getattr(server, "enable_dynamic_chunking", None) is not False:
        failed.append("enable_dynamic_chunking=false")
    if getattr(server, "enable_mixed_chunk", None) is not False:
        failed.append("enable_mixed_chunk=false")
    # In the pinned SGLang source, max_prefill_tokens is the aggregate batch
    # budget while chunked_prefill_size is the per-request truncation bound.
    # One admitted request may therefore use a larger aggregate budget, but it
    # must be large enough to execute one complete compiled semantic chunk.
    if (
        isinstance(max_prefill_tokens, bool)
        or not isinstance(max_prefill_tokens, Integral)
        or int(max_prefill_tokens) < chunk_tokens
    ):
        failed.append(f"max_prefill_tokens>={chunk_tokens}")
    if failed:
        raise RuntimeError(
            "chunked scheduler contract failed: " + ", ".join(failed)
        )


def _validate_token_reclamation_backend_contract(configurator: Any) -> None:
    config = _config()
    policy = config.token_reclamation
    if getattr(config, "chunked_class", None) is not None and policy.mode != "off":
        raise RuntimeError(
            "chunked retention requires token reclamation to remain off"
        )
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


def _chunked_attention_layers(configurator: Any) -> tuple[Any, ...]:
    model = getattr(configurator, "model", None)
    named_modules = getattr(model, "named_modules", None)
    if model is None or not callable(named_modules):
        raise RuntimeError(
            "chunked local-attention contract is unavailable at "
            "KVCacheConfigurator.configure; a post-model-load/pre-KV-pool "
            "hook must expose every runtime attention layer's layer_id and "
            "use_irope"
        )
    try:
        from sglang.srt.layers.radix_attention import RadixAttention

        # The pinned source decides whether to apply chunk-local masking from the
        # loaded RadixAttention.use_irope value.  ModelConfig's local-attention
        # flag alone is architecture-derived and cannot prove the layer set.
        layers = tuple(
            module
            for _name, module in named_modules()
            if isinstance(module, RadixAttention)
        )
        if not layers:
            raise RuntimeError(
                "the loaded model exposes no SGLang RadixAttention layers"
            )
        return layers
    except Exception as error:
        raise RuntimeError(
            "chunked local-attention contract is unavailable at "
            "KVCacheConfigurator.configure; a post-model-load/pre-KV-pool "
            "hook must expose every runtime attention layer's layer_id and "
            "use_irope"
        ) from error


def _validate_chunked_local_attention_contract(configurator: Any) -> None:
    chunked = getattr(_config(), "chunked_class", None)
    if chunked is None:
        return
    model_config = configurator.model_config
    if getattr(model_config, "is_local_attention_model", None) is not True:
        raise RuntimeError(
            "chunked local-attention contract requires "
            "model_config.is_local_attention_model=True"
        )

    expected_layers = tuple(chunked.layers)
    runtime_layers = _chunked_attention_layers(configurator)
    observed_layers: list[int] = []
    non_local_layers: list[int] = []
    for layer in runtime_layers:
        layer_id = getattr(layer, "layer_id", None)
        if (
            layer is None
            or isinstance(layer_id, bool)
            or not isinstance(layer_id, Integral)
        ):
            raise RuntimeError(
                "chunked local-attention contract requires every runtime "
                "attention layer to expose an integer layer_id and boolean "
                "use_irope"
            )
        layer_id = int(layer_id)
        use_irope = getattr(layer, "use_irope", None)
        if not isinstance(use_irope, bool):
            raise RuntimeError(
                "chunked local-attention contract requires every runtime "
                "attention layer to expose an integer layer_id and boolean "
                "use_irope"
            )
        observed_layers.append(layer_id)
        if not use_irope:
            non_local_layers.append(layer_id)

    observed_domain = tuple(sorted(observed_layers))
    if (
        len(set(observed_layers)) != len(observed_layers)
        or observed_domain != expected_layers
    ):
        raise RuntimeError(
            "chunked local-attention runtime layers do not exactly cover the "
            f"compiled layer domain: expected {expected_layers}, got "
            f"{observed_domain}"
        )
    if non_local_layers:
        raise RuntimeError(
            "chunked local-attention contract requires use_irope=True for "
            "every compiled layer; non-local layer ids: "
            + ", ".join(str(layer) for layer in non_local_layers)
        )


def _authoritative_chunked_execution_layers(model_runner: Any) -> tuple[Any, ...]:
    """Resolve the exact RadixAttention objects used by pinned model execution."""

    try:
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.model_executor.model_runner_components.layer_setup import (
            compute_attention_and_moe_layers,
        )
        from sglang.srt.model_loader.utils import resolve_language_model

        model = model_runner.model
        named_modules = model.named_modules
        language_model = resolve_language_model(model)
        while not hasattr(language_model, "layers") and hasattr(
            language_model, "model"
        ):
            language_model = language_model.model
        if not hasattr(language_model, "layers"):
            raise RuntimeError("the loaded language model has no layer sequence")
        mapping = compute_attention_and_moe_layers(language_model)
        execution_layers = tuple(mapping.attention_layers)
        companions = tuple(mapping.mha_companion_layers)
        loaded_layers = tuple(
            module
            for _name, module in named_modules()
            if isinstance(module, RadixAttention)
        )
    except Exception as error:
        raise RuntimeError(
            "pinned SGLang cannot prove the loaded attention execution mapping"
        ) from error

    if (
        not execution_layers
        or len(companions) != len(execution_layers)
        or any(item is not None for item in companions)
        or any(type(item) is not RadixAttention for item in execution_layers)
        or len(loaded_layers) != len(execution_layers)
        or {id(item) for item in loaded_layers}
        != {id(item) for item in execution_layers}
    ):
        raise RuntimeError(
            "loaded model attention implementations do not exactly match the "
            "pinned execution mapping"
        )
    return execution_layers


def _validate_loaded_runtime_backend(
    model_runner: Any,
) -> _state.RuntimeAttentionBackendProof:
    """Prove the concrete backend and layer dispatch after backend construction."""

    chunked = getattr(_config(), "chunked_class", None)
    if chunked is None:
        raise RuntimeError("runtime backend proof is only defined for chunked plans")
    try:
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )
        from sglang.srt.layers.radix_attention import RadixAttention
    except Exception as error:
        raise RuntimeError("pinned SGLang FA3 proof types are unavailable") from error

    backend = getattr(model_runner, "attn_backend", None)
    if type(backend) is not FlashAttentionBackend:
        raise RuntimeError(
            "chunked runtime requires the concrete SGLang FlashAttentionBackend"
        )
    if getattr(model_runner, "decode_attn_backend", None) is not None:
        raise RuntimeError("chunked runtime rejects a separate decode backend")
    decode_group = getattr(model_runner, "decode_attn_backend_group", None)
    if not isinstance(decode_group, list) or decode_group:
        raise RuntimeError("chunked runtime rejects a decode backend group")

    prefill = getattr(model_runner, "prefill_attention_backend_str", None)
    decode = getattr(model_runner, "decode_attention_backend_str", None)
    backend_prefill = getattr(backend, "prefill_attention_backend_str", None)
    backend_decode = getattr(backend, "decode_attention_backend_str", None)
    if (prefill, decode, backend_prefill, backend_decode) != (
        "fa3",
        "fa3",
        "fa3",
        "fa3",
    ):
        raise RuntimeError(
            "chunked runtime resolved attention backend IDs are not fa3/fa3"
        )
    if getattr(backend, "fa_impl_ver", None) != 3:
        raise RuntimeError("chunked runtime did not construct FA3 implementation 3")

    chunk_tokens = _positive_integer(
        "compiled chunk size", getattr(chunked, "chunk_tokens", None)
    )
    backend_chunk = getattr(backend, "attention_chunk_size", None)
    backend_page = getattr(backend, "page_size", None)
    if getattr(backend, "has_local_attention", None) is not True:
        raise RuntimeError("loaded FA3 backend did not enable local attention")
    if (
        isinstance(backend_chunk, bool)
        or not isinstance(backend_chunk, Integral)
        or int(backend_chunk) != chunk_tokens
    ):
        raise RuntimeError("loaded FA3 attention chunk size differs from the plan")
    if (
        isinstance(backend_page, bool)
        or not isinstance(backend_page, Integral)
        or int(backend_page) != _config().page_tokens
    ):
        raise RuntimeError("loaded FA3 page size differs from the plan")
    if getattr(backend, "use_sliding_window_kv_pool", None) is not False:
        raise RuntimeError("chunked runtime unexpectedly uses an SWA KV pool")
    if getattr(backend, "has_swa", None) is not False:
        raise RuntimeError("chunked runtime unexpectedly enabled SWA dispatch")

    expected_ids = tuple(chunked.layers)
    execution_layers = _authoritative_chunked_execution_layers(model_runner)
    observed_ids: list[int] = []
    use_irope_ids: list[int] = []
    for layer in execution_layers:
        if type(layer) is not RadixAttention:
            raise RuntimeError("chunked execution contains a foreign attention layer")
        layer_id = getattr(layer, "layer_id", None)
        use_irope = getattr(layer, "use_irope", None)
        if isinstance(layer_id, bool) or not isinstance(layer_id, Integral):
            raise RuntimeError(
                "chunked execution mapping requires integer layer IDs"
            )
        if not isinstance(use_irope, bool):
            raise RuntimeError(
                "chunked execution mapping requires boolean use_irope"
            )
        observed_ids.append(int(layer_id))
        if use_irope:
            use_irope_ids.append(int(layer_id))
    if tuple(observed_ids) != expected_ids or len(set(observed_ids)) != len(
        observed_ids
    ):
        raise RuntimeError(
            "loaded attention execution mapping does not exactly cover compiled layers"
        )
    if tuple(use_irope_ids) != expected_ids:
        raise RuntimeError(
            "loaded attention execution mapping is not chunk-local on every layer"
        )

    backend_type = type(backend)
    return _state.RuntimeAttentionBackendProof(
        backend_class=backend_type.__name__,
        backend_module=backend_type.__module__,
        prefill_backend=prefill,
        decode_backend=decode,
        has_local_attention=True,
        attention_chunk_size=int(backend_chunk),
        page_size=int(backend_page),
        compiled_layer_ids=expected_ids,
        use_irope_layer_ids=tuple(use_irope_ids),
    )


def _validate_runtime_attention_backend(
    original_fn: Callable[..., Any], model_runner: Any, *args: Any, **kwargs: Any
) -> Any:
    """Complete chunked initialization only after the loaded backend is proved."""

    if getattr(_config(), "chunked_class", None) is None:
        return original_fn(model_runner, *args, **kwargs)
    token = _state._pending_initialization_token()
    try:
        result = original_fn(model_runner, *args, **kwargs)
        proof = _validate_loaded_runtime_backend(model_runner)
        _state._publish_runtime_backend_proof(proof, token=token)
    except Exception as error:
        _state._rollback_initialization(error, token=token)
        try:
            _state._finish_initialization(token)
        except Exception as finish_error:
            error.add_note(
                "OrbitKV initialization transaction cleanup also failed: "
                f"{finish_error!r}"
            )
        raise
    _state._finish_initialization(token)
    return result


def _validate_checkpoint_geometry(configurator: Any) -> None:
    plan = _config()
    model = configurator.model_config
    text = model.hf_text_config
    if int(text.num_hidden_layers) != plan.num_hidden_layers:
        raise RuntimeError("checkpoint layer count differs from KvPlanInput.layers")
    if bool(getattr(model, "is_hybrid_swa_compress", False)):
        raise RuntimeError("OrbitKV does not support compressed attention storage")
    retentions = tuple(item.retention for item in plan.classes)
    attention_chunk_size = getattr(model, "attention_chunk_size", None)
    if retentions != ("chunked",) and attention_chunk_size is not None:
        raise RuntimeError("non-chunked OrbitKV plans reject attention chunking")
    all_layers = tuple(range(plan.num_hidden_layers))
    token_layers = tuple(
        sorted(layer for item in plan.classes for layer in item.layers)
    )
    storage = {item.storage for item in plan.classes}
    if storage == {"latent_kv"}:
        index_topk = getattr(model.hf_config, "index_topk", None)
        checkpoint_latent = _positive_integer(
            "checkpoint kv_lora_rank",
            getattr(model.hf_config, "kv_lora_rank", None),
        )
        checkpoint_rope = _positive_integer(
            "checkpoint qk_rope_head_dim",
            getattr(model.hf_config, "qk_rope_head_dim", None),
        )
        if (
            len(plan.classes) != 1
            or plan.classes[0].retention != "full"
            or plan.classes[0].layers != token_layers
            or bool(plan.fixed_states)
            or not bool(configurator.use_mla_backend)
            or bool(model.is_hybrid_swa)
            or index_topk is not None
        ):
            raise RuntimeError(
                "supported MLA profile requires one Full latent_kv class covering every layer"
            )
        if (
            int(model.kv_lora_rank) != checkpoint_latent
            or int(model.qk_rope_head_dim) != checkpoint_rope
        ):
            raise RuntimeError(
                "SGLang latent-KV geometry differs from the checkpoint"
            )
        latent = checkpoint_latent * _dtype_bytes(configurator.kv_cache_dtype)
        rope = checkpoint_rope * _dtype_bytes(configurator.kv_cache_dtype)
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
    elif retentions == ("chunked",):
        chunked = plan.classes[0]
        chunk_tokens = getattr(chunked, "chunk_tokens", None)
        blocks_per_epoch = getattr(chunked, "blocks_per_epoch", None)
        if (
            chunked.layers != all_layers
            or token_layers != all_layers
            or bool(plan.fixed_states)
            or bool(model.is_hybrid_swa)
            or getattr(model, "sliding_window_size", None) is not None
            or tuple(getattr(model, "full_attention_layer_ids", ()))
            or tuple(getattr(model, "swa_attention_layer_ids", ()))
            or isinstance(chunk_tokens, bool)
            or not isinstance(chunk_tokens, Integral)
            or isinstance(blocks_per_epoch, bool)
            or not isinstance(blocks_per_epoch, Integral)
            or int(chunk_tokens) <= 0
            or int(blocks_per_epoch) <= 0
            or int(chunk_tokens) != int(blocks_per_epoch) * plan.page_tokens
        ):
            raise RuntimeError(
                "chunked profile requires one page-aligned token_kv class "
                "covering every model layer"
            )
        if (
            isinstance(attention_chunk_size, bool)
            or not isinstance(attention_chunk_size, Integral)
            or int(attention_chunk_size) != int(chunk_tokens)
        ):
            raise RuntimeError(
                "SGLang attention chunk size differs from the compiled chunk"
            )
        _validate_chunked_local_attention_contract(configurator)
    else:
        raise RuntimeError(
            "OrbitKV SGLang supports Full, pure sliding, chunked, or ordered "
            "Full+SWA"
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
        actual = (
            swa_bytes if class_config.retention == "sliding" else full_bytes
        )
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
    force_miss = os.environ.get("SGLANG_RADIX_FORCE_MISS")
    if force_miss is not None:
        normalized = force_miss.lower()
        if normalized not in ("true", "1", "yes", "y", "false", "0", "no", "n"):
            raise RuntimeError("SGLANG_RADIX_FORCE_MISS is not a valid boolean")
        if normalized in ("true", "1", "yes", "y"):
            raise RuntimeError("OrbitKV does not support SGLANG_RADIX_FORCE_MISS")
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
        full = _config().full_class
        if full is None:
            raise RuntimeError(
                "non-Full plan received an SGLang hybrid-linear KV pool"
            )
        expected_layers = full.layers
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


def _validate_configurator_once(
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
    _validate_chunked_scheduler_contract(configurator)
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
    elif config.full_class is not None or getattr(
        config, "chunked_class", None
    ) is not None:
        if int(result.max_total_num_tokens) != int(allocator.size):
            raise RuntimeError(
                "SGLang paged result capacity differs from its arena"
            )
        primary = getattr(config, "primary_class", None)
        if primary is None:
            raise RuntimeError("OrbitKV plan has no primary KV class")
        _validate_full_physical_pool(
            kv_pool,
            expected_tokens=allocator.size,
            expected_dtype=configurator.kv_cache_dtype,
            storage=primary.storage,
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
    return result


def _validate_configurator(
    original_fn: Callable[..., Any], configurator: Any, *args: Any, **kwargs: Any
) -> Any:
    """Run SGLang configuration as one failure-atomic initialization."""

    token = _state._begin_initialization()
    try:
        result = _validate_configurator_once(
            original_fn, configurator, *args, **kwargs
        )
    except Exception as error:
        _state._rollback_initialization(error, token=token)
        _state._finish_initialization(token)
        raise
    if getattr(_config(), "chunked_class", None) is not None:
        return result
    _state._finish_initialization(token)
    return result
