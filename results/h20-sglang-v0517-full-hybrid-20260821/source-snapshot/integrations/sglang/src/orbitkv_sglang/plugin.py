from __future__ import annotations

import os
import pkgutil
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import ManagerPlanConfig, load_config
from .ffi import CtypesManagerFactory
from .pinned import validate_patched_checkout
from .runtime import (
    ArenaRegistration,
    CanonicalRuntime,
    FailStopped,
    LoweringPlan,
    ManagerCreateSettings,
    ManagerFactoryProtocol,
    ReclamationCertificate,
    StepRecord,
    sglang_page_id,
)


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


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    maximum_running_requests: int
    chunked_prefill_tokens: int
    maximum_context_tokens: int

_CONFIG: ManagerPlanConfig | None = None
_LIMITS: RuntimeLimits | None = None
_RUNTIME: CanonicalRuntime | None = None
_FACTORY: ManagerFactoryProtocol = CtypesManagerFactory()
_ALLOCATOR: Any = None
_FACADE_TYPES: tuple[type, type, type] | None = None


HOOK_TARGETS = (
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool_allocator",
    "sglang.srt.mem_cache.allocation.alloc_for_extend",
    "sglang.srt.mem_cache.allocation.alloc_for_decode",
    "sglang.srt.managers.schedule_batch.ScheduleBatch.maybe_evict_swa",
    "sglang.srt.mem_cache.common.release_kv_cache",
    "sglang.srt.managers.scheduler.Scheduler.run_batch",
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator.configure",
    "sglang.srt.managers.scheduler.Scheduler.get_internal_state",
)


def _config() -> ManagerPlanConfig:
    if _CONFIG is None:
        raise RuntimeError("OrbitKV manager plan is not loaded")
    return _CONFIG


def _runtime() -> CanonicalRuntime:
    if _RUNTIME is None:
        raise RuntimeError("OrbitKV KV arena is not initialized")
    return _RUNTIME


def _limits() -> RuntimeLimits:
    if _LIMITS is None:
        raise RuntimeError("OrbitKV runtime capacities are not resolved")
    return _LIMITS


def _request_key(req: Any) -> tuple[str, str | bytes | int]:
    value = getattr(req, "rid", None)
    if isinstance(value, bool):
        raise RuntimeError("SGLang request rid must not be boolean")
    if isinstance(value, str):
        if not value:
            raise RuntimeError("SGLang request rid must not be empty")
        return ("str", value)
    if isinstance(value, bytes):
        if not value:
            raise RuntimeError("SGLang request rid must not be empty")
        return ("bytes", value)
    if isinstance(value, int) and value >= 0:
        return ("int", value)
    raise RuntimeError("SGLang request rid must be a stable str, bytes, or integer")


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
            "SGLang retained stale imported KV authority aliases: "
            + ", ".join(stale)
        )


def _new_runtime(registrations: Sequence[ArenaRegistration]) -> CanonicalRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        raise RuntimeError("OrbitKV manager is already initialized")
    values = tuple(registrations)
    if not values or len(values) != len(_config().classes):
        raise RuntimeError("one physical arena is required for every KV class")
    total_pages = sum(item.page_count for item in values)
    if total_pages <= 0:
        raise RuntimeError("OrbitKV physical arenas must be nonempty")
    limits = _limits()
    settings = ManagerCreateSettings(
        maximum_requests=limits.maximum_running_requests,
        maximum_operations=limits.maximum_running_requests,
        maximum_reclamations=total_pages,
        maximum_step_tokens=limits.chunked_prefill_tokens,
    )
    manager = _FACTORY.create(_config(), settings, values)
    _RUNTIME = CanonicalRuntime(_config(), manager)
    for registration, identity in zip(values, _RUNTIME.arenas, strict=True):
        if (
            identity.class_id != registration.class_id
            or identity.pool_id != registration.pool_id
            or identity.backend_domain != registration.backend_domain
            or identity.page_count != registration.page_count
            or identity.backend_base_index != registration.backend_base_index
        ):
            _RUNTIME.fail_stop("manager arena differs from SGLang physical storage")
            raise FailStopped(_RUNTIME.failure_reason or "arena identity mismatch")
    return _RUNTIME


def _arena_available_tokens(class_id: int) -> int:
    runtime = _runtime()
    runtime.poll()
    runtime.stats()
    matches = [
        item for item in runtime.manager.arena_stats() if item.class_id == class_id
    ]
    if len(matches) != 1:
        runtime.fail_stop("manager returned ambiguous per-class arena stats")
        raise FailStopped(runtime.failure_reason or "ambiguous arena stats")
    return matches[0].free_pages * runtime.page_tokens


class _NativeAuthorityForbidden:
    @staticmethod
    def _native_authority_error(operation: str) -> None:
        raise RuntimeError(
            f"SGLang native KV authority was invoked ({operation}); "
            "OrbitKV manager hooks were bypassed"
        )

    def alloc(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("alloc")

    def alloc_extend(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("alloc_extend")

    def alloc_decode(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("alloc_decode")

    def alloc_extend_swa_tail(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("alloc_extend_swa_tail")

    def free(self, *_args: Any, **_kwargs: Any) -> None:
        self._native_authority_error("free")

    def free_segment(self, *_args: Any, **_kwargs: Any) -> None:
        self._native_authority_error("free_segment")

    def free_segments(self, *_args: Any, **_kwargs: Any) -> None:
        self._native_authority_error("free_segments")

    def free_swa(self, *_args: Any, **_kwargs: Any) -> None:
        self._native_authority_error("free_swa")

    def clear(self, *_args: Any, **_kwargs: Any) -> None:
        self._native_authority_error("clear")

    def resize(self, *_args: Any, **_kwargs: Any) -> None:
        self._native_authority_error("resize")

    def get_cpu_copy(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("get_cpu_copy")

    def load_cpu_copy(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("load_cpu_copy")

    def translate_kv_indices_for_transfer(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("translate_kv_indices_for_transfer")

    def free_group_begin(self) -> None:
        return None

    def free_group_end(self) -> None:
        return None


class _ArenaAvailability(_NativeAuthorityForbidden):
    def __init__(self, class_id: int, size: int, page_size: int) -> None:
        self.class_id = int(class_id)
        self.size = int(size)
        self.page_size = int(page_size)
        self.num_pages = self.size // self.page_size
        self.free_pages = None
        self.release_pages = None

    def available_size(self) -> int:
        return _arena_available_tokens(self.class_id)


def _initialize_facade(
    facade: Any,
    *,
    size: int,
    page_size: int,
    dtype: Any,
    device: Any,
    kvcache: Any,
    need_sort: bool,
) -> None:
    facade.page_size = page_size
    facade.dtype = dtype
    facade.device = device
    facade.need_sort = bool(need_sort)
    facade.num_pages = size // page_size
    facade._kvcache = kvcache
    facade.free_pages = None
    facade.release_pages = None
    facade.is_not_in_free_group = True
    facade.free_group = []


def _facade_types() -> tuple[type, type, type]:
    global _FACADE_TYPES
    if _FACADE_TYPES is not None:
        return _FACADE_TYPES

    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.swa import (
        PureSWATokenToKVPoolAllocator,
        SWATokenToKVPoolAllocator,
    )

    class OrbitKvPagedTokenToKVPoolAllocator(
        _NativeAuthorityForbidden, PagedTokenToKVPoolAllocator
    ):
        def __init__(
            self,
            size: int,
            page_size: int,
            dtype: Any,
            device: Any,
            kvcache: Any,
            need_sort: bool,
            *,
            class_id: int,
        ) -> None:
            _initialize_facade(
                self,
                size=size,
                page_size=page_size,
                dtype=dtype,
                device=device,
                kvcache=kvcache,
                need_sort=need_sort,
            )
            self.size = size
            self._orbitkv_class_id = int(class_id)

        def available_size(self) -> int:
            return _arena_available_tokens(self._orbitkv_class_id)

        def get_kvcache(self) -> Any:
            return self._kvcache

    class OrbitKvSWATokenToKVPoolAllocator(
        _NativeAuthorityForbidden, SWATokenToKVPoolAllocator
    ):
        def __init__(
            self,
            size: int,
            size_swa: int,
            page_size: int,
            dtype: Any,
            device: Any,
            kvcache: Any,
            need_sort: bool,
            *,
            full_class_id: int,
            swa_class_id: int,
        ) -> None:
            import torch

            _initialize_facade(
                self,
                size=min(size, size_swa),
                page_size=page_size,
                dtype=dtype,
                device=device,
                kvcache=kvcache,
                need_sort=need_sort,
            )
            self._size_full = size
            self._size_swa = size_swa
            self.full_attn_allocator = _ArenaAvailability(
                full_class_id, size, page_size
            )
            self.swa_attn_allocator = _ArenaAvailability(
                swa_class_id, size_swa, page_size
            )
            mapping = torch.zeros(
                size + page_size + 1,
                dtype=torch.int64,
                device=device,
            )
            mapping[-1] = -1
            self.full_to_swa_index_mapping = mapping
            kvcache.register_mapping(mapping)

        @property
        def size_full(self) -> int:
            return self._size_full

        @property
        def size_swa(self) -> int:
            return self._size_swa

        def available_size(self) -> int:
            return min(self.full_available_size(), self.swa_available_size())

        def full_available_size(self) -> int:
            return self.full_attn_allocator.available_size()

        def swa_available_size(self) -> int:
            return self.swa_attn_allocator.available_size()

        def _conserve_full_available_size(self) -> int:
            return self.full_available_size()

        def _conserve_swa_available_size(self) -> int:
            return self.swa_available_size()

        def new_pages_available(
            self, num_full_pages: int, num_swa_pages: int
        ) -> bool:
            return (
                int(num_full_pages)
                <= self.full_available_size() // self.page_size
                and int(num_swa_pages)
                <= self.swa_available_size() // self.page_size
            )

        def get_kvcache(self) -> Any:
            return self._kvcache

        def translate_loc_from_full_to_swa(self, kv_indices: Any) -> Any:
            return self.full_to_swa_index_mapping[kv_indices]

        def set_full_to_swa_mapping(
            self, full_indices: Any, swa_indices: Any
        ) -> None:
            self.full_to_swa_index_mapping[full_indices] = swa_indices

    class OrbitKvPureSWATokenToKVPoolAllocator(
        _NativeAuthorityForbidden, PureSWATokenToKVPoolAllocator
    ):
        def __init__(
            self,
            size_swa: int,
            page_size: int,
            dtype: Any,
            device: Any,
            kvcache: Any,
            need_sort: bool,
            *,
            class_id: int,
        ) -> None:
            import torch

            _initialize_facade(
                self,
                size=size_swa,
                page_size=page_size,
                dtype=dtype,
                device=device,
                kvcache=kvcache,
                need_sort=need_sort,
            )
            self._size_full = size_swa
            self._size_swa = size_swa
            self._orbitkv_class_id = int(class_id)
            self.swa_attn_allocator = _ArenaAvailability(
                class_id, size_swa, page_size
            )
            self.full_attn_allocator = self.swa_attn_allocator
            mapping = torch.arange(
                size_swa + page_size + 1,
                dtype=torch.int64,
                device=device,
            )
            mapping[-1] = -1
            self.full_to_swa_index_mapping = mapping
            kvcache.register_mapping(mapping)

        @property
        def size_full(self) -> int:
            return self._size_full

        @property
        def size_swa(self) -> int:
            return self._size_swa

        def available_size(self) -> int:
            return _arena_available_tokens(self._orbitkv_class_id)

        def full_available_size(self) -> int:
            return self.available_size()

        def swa_available_size(self) -> int:
            return self.available_size()

        def new_pages_available(
            self, num_full_pages: int, num_swa_pages: int
        ) -> bool:
            available = self.available_size() // self.page_size
            return max(int(num_full_pages), int(num_swa_pages)) <= available

        def get_kvcache(self) -> Any:
            return self._kvcache

        def translate_loc_from_full_to_swa(self, kv_indices: Any) -> Any:
            return kv_indices

    _FACADE_TYPES = (
        OrbitKvPagedTokenToKVPoolAllocator,
        OrbitKvSWATokenToKVPoolAllocator,
        OrbitKvPureSWATokenToKVPoolAllocator,
    )
    return _FACADE_TYPES


def _pool_tokens(name: str, value: Any, page_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"SGLang {name} must be a positive integer")
    if value % page_size:
        raise RuntimeError(f"SGLang {name} is not page aligned")
    return value


def _validate_sliding_pool_floor(class_config: Any, size: int) -> None:
    limits = _limits()
    minimum = class_config.minimum_sliding_pool_tokens(
        maximum_running_requests=limits.maximum_running_requests,
        chunked_prefill_tokens=limits.chunked_prefill_tokens,
    )
    if size < minimum:
        raise RuntimeError(
            "SGLang SWA KV capacity is below the compiled resident/staging "
            f"floor: capacity={size} minimum={minimum}"
        )


def _build_token_to_kv_pool_allocator(
    configurator: Any,
    *,
    sizes: Any,
    token_to_kv_pool: Any,
    is_dsv4_model: bool,
    req_to_token_pool: Any,
    token_to_kv_pool_allocator: Any,
) -> Any:
    del req_to_token_pool
    global _ALLOCATOR
    if _ALLOCATOR is not None or _RUNTIME is not None:
        raise RuntimeError("OrbitKV allocator builder ran more than once")
    if token_to_kv_pool_allocator is not None:
        raise RuntimeError("OrbitKV refuses a preexisting SGLang KV allocator")
    if bool(is_dsv4_model) or bool(getattr(configurator, "is_draft_worker", False)):
        raise RuntimeError("OrbitKV does not support DSV4 or draft allocators")
    page_size = int(getattr(configurator, "page_size", _config().page_tokens))
    if page_size != _config().page_tokens:
        raise RuntimeError("OrbitKV and SGLang page sizes differ")
    device = configurator.device
    if not str(device).startswith("cuda"):
        raise RuntimeError("OrbitKV requires CUDA KV storage")
    dtype = configurator.kv_cache_dtype
    full_type, hybrid_type, pure_type = _facade_types()
    retentions = tuple(item.retention for item in _config().classes)
    registrations: tuple[ArenaRegistration, ...]

    if retentions == ("full",):
        if bool(getattr(configurator, "is_hybrid_swa", False)):
            raise RuntimeError("Full-only plan cannot use SGLang hybrid storage")
        size = _pool_tokens(
            "full KV capacity", int(sizes.max_total_num_tokens), page_size
        )
        class_config = _config().classes[0]
        allocator = full_type(
            size,
            page_size,
            dtype,
            device,
            token_to_kv_pool,
            False,
            class_id=class_config.class_id,
        )
        registrations = (
            ArenaRegistration(
                class_config.class_id,
                class_config.pool_id,
                class_config.backend_domain,
                size // page_size,
            ),
        )
    elif retentions == ("full", "sliding"):
        if not bool(getattr(configurator, "is_hybrid_swa", False)):
            raise RuntimeError("Full+SWA plan requires SGLang hybrid storage")
        full_size = _pool_tokens(
            "full KV capacity", int(sizes.full_max_total_num_tokens), page_size
        )
        swa_size = _pool_tokens(
            "SWA KV capacity", int(sizes.swa_max_total_num_tokens), page_size
        )
        full_class, swa_class = _config().classes
        _validate_sliding_pool_floor(swa_class, swa_size)
        allocator = hybrid_type(
            full_size,
            swa_size,
            page_size,
            dtype,
            device,
            token_to_kv_pool,
            False,
            full_class_id=full_class.class_id,
            swa_class_id=swa_class.class_id,
        )
        registrations = (
            ArenaRegistration(
                full_class.class_id,
                full_class.pool_id,
                full_class.backend_domain,
                full_size // page_size,
            ),
            ArenaRegistration(
                swa_class.class_id,
                swa_class.pool_id,
                swa_class.backend_domain,
                swa_size // page_size,
            ),
        )
    elif retentions == ("sliding",):
        if not bool(getattr(configurator, "is_hybrid_swa", False)):
            raise RuntimeError("SWA plan requires SGLang SWA storage")
        size = _pool_tokens(
            "SWA KV capacity", int(sizes.swa_max_total_num_tokens), page_size
        )
        class_config = _config().classes[0]
        _validate_sliding_pool_floor(class_config, size)
        allocator = pure_type(
            size,
            page_size,
            dtype,
            device,
            token_to_kv_pool,
            False,
            class_id=class_config.class_id,
        )
        registrations = (
            ArenaRegistration(
                class_config.class_id,
                class_config.pool_id,
                class_config.backend_domain,
                size // page_size,
            ),
        )
    else:
        raise RuntimeError("unsupported OrbitKV attention-class layout")

    _new_runtime(registrations)
    _ALLOCATOR = allocator
    return allocator


def _validate_batch(batch: Any) -> None:
    config = _config()
    if bool(getattr(batch, "enable_overlap", False)):
        raise RuntimeError("OrbitKV does not support overlap scheduling")
    if not bool(batch.spec_algorithm.is_none()):
        raise RuntimeError("OrbitKV does not support speculative decoding")
    if not bool(batch.tree_cache.is_chunk_cache()):
        raise RuntimeError("OrbitKV requires ChunkCache")
    supports_swa = bool(batch.tree_cache.supports_swa())
    requires_swa = _config().sliding_class is not None
    if supports_swa != requires_swa:
        raise RuntimeError("SGLang cache type differs from the compiled KV classes")
    if batch.tree_cache.token_to_kv_pool_allocator is not _ALLOCATOR:
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
        expected_device.index is not None and actual_device.index != expected_device.index
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
    extend_num_tokens = _positive_integer(
        "extend_num_tokens", batch.extend_num_tokens
    )
    if extend_num_tokens != sum(extend_lens):
        raise RuntimeError("SGLang extend_num_tokens differs from extend_lens")
    maximum = int(batch.req_to_token_pool.max_context_len)
    for req, prefix, target in zip(
        batch.reqs, prefix_lens, targets, strict=True
    ):
        if target > maximum:
            raise RuntimeError("SGLang extend boundary exceeds ReqToToken capacity")
        try:
            prefix_entries = len(req.prefix_indices)
        except Exception as error:
            raise RuntimeError("SGLang request prefix mirror is unreadable") from error
        if prefix_entries != prefix:
            raise RuntimeError("SGLang request prefix mirror length differs from prefix_lens")
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


def _wait_previous_steps(batch: Any) -> None:
    runtime = _runtime()
    runtime.poll()
    for req in batch.reqs:
        if hasattr(req, "_orbitkv_request_lease"):
            key = _request_key(req)
            if getattr(req, "_orbitkv_request_key", None) != key:
                raise RuntimeError("SGLang request rid changed after KV acquisition")
            runtime.wait(key)


def _synchronize_mirror(req_to_token_pool: Any) -> None:
    device = getattr(req_to_token_pool, "device", None)
    if device is not None and str(device).startswith("cuda"):
        import torch

        torch.get_device_module(device).current_stream(device).synchronize()


def _mapping_clear(mapping: Any, locations: Any) -> None:
    import torch

    indices = locations.to(dtype=torch.int64)
    indices = indices[indices > 0]
    if int(indices.numel()):
        mapping[indices] = 0


def _validate_certificate_mirror(
    mirror: Any,
    allocator: Any,
    certificate: ReclamationCertificate,
    *,
    hybrid: bool,
) -> None:
    import torch

    begin = int(certificate.token_begin)
    end = int(certificate.token_end_exclusive)
    arena = _runtime().arenas_by_class[certificate.class_id]
    page = sglang_page_id(certificate.backend_index, arena.backend_base_index)
    expected = torch.arange(
        page * _config().page_tokens,
        page * _config().page_tokens + (end - begin),
        dtype=torch.int64,
        device=mirror.device,
    )
    full_locations = mirror[begin:end].to(dtype=torch.int64)
    actual = (
        allocator.full_to_swa_index_mapping[full_locations]
        if hybrid
        else full_locations
    )
    if not torch.equal(actual, expected):
        raise RuntimeError("reclamation certificate disagrees with the SGLang mirror")


def _clear_reclamation_mirrors(
    req: Any,
    req_to_token_pool: Any,
    allocator: Any,
    certificates: Sequence[ReclamationCertificate],
    releasing: bool,
) -> None:
    if req.req_pool_idx is None:
        raise RuntimeError("ReqToToken row disappeared before reclamation")
    row = int(req.req_pool_idx)
    if not 0 < row < int(req_to_token_pool.req_to_token.shape[0]):
        raise RuntimeError("ReqToToken reclamation names a dummy or foreign row")
    mirror = req_to_token_pool.req_to_token[row]
    maximum = int(req_to_token_pool.max_context_len)
    classes = _config().classes_by_id
    sliding = _config().sliding_class
    hybrid = _config().full_class is not None and sliding is not None

    for certificate in certificates:
        try:
            class_config = classes[certificate.class_id]
        except KeyError as error:
            raise RuntimeError("reclamation names an unknown KV class") from error
        begin = int(certificate.token_begin)
        end = int(certificate.token_end_exclusive)
        if (
            begin < 0
            or end <= begin
            or end > maximum
            or begin % _config().page_tokens
            or (not releasing and end % _config().page_tokens)
        ):
            raise RuntimeError("manager reclamation exceeds a whole mirror page")
        if not releasing and class_config.retention != "sliding":
            raise RuntimeError("Full KV was retired before request release")
        if class_config.retention == "sliding":
            _validate_certificate_mirror(
                mirror,
                allocator,
                certificate,
                hybrid=hybrid,
            )
            if not releasing and hybrid:
                _mapping_clear(
                    allocator.full_to_swa_index_mapping,
                    mirror[begin:end],
                )
            elif not releasing:
                mirror[begin:end].zero_()
                prefix = getattr(req, "prefix_indices", None)
                if prefix is not None:
                    prefix_begin = min(begin, len(prefix))
                    prefix_end = min(end, len(prefix))
                    if prefix_begin < prefix_end:
                        prefix[prefix_begin:prefix_end].zero_()

    if releasing:
        boundary = _runtime().record_for(_request_key(req)).boundary
        if not 0 <= boundary <= maximum:
            raise RuntimeError("manager release boundary exceeds ReqToToken")
        if hybrid and boundary:
            _mapping_clear(
                allocator.full_to_swa_index_mapping,
                mirror[:boundary],
            )
        mirror.zero_()
        prefix = getattr(req, "prefix_indices", None)
        if prefix is not None and hasattr(prefix, "zero_"):
            prefix.zero_()

    _synchronize_mirror(req_to_token_pool)
    if not releasing and certificates:
        if req.kv is None or sliding is None:
            raise RuntimeError("SWA reclamation lost its request KV metadata")
        frontier = max(
            int(item.token_end_exclusive)
            for item in certificates
            if classes[item.class_id].retention == "sliding"
        )
        req.kv.swa_evicted_seqlen = max(
            int(req.kv.swa_evicted_seqlen), frontier
        )


def _prepare_steps(
    batch: Any, previous_boundaries: Sequence[int], targets: Sequence[int]
) -> tuple[list[StepRecord], list[Any]]:
    if len(batch.reqs) != len(previous_boundaries) or len(batch.reqs) != len(targets):
        raise RuntimeError("OrbitKV batch boundary cardinality changed")
    runtime = _runtime()
    records: list[StepRecord] = []
    plans: list[LoweringPlan] = []
    acquired: list[bool] = []
    try:
        for req, previous, target in zip(
            batch.reqs, previous_boundaries, targets, strict=True
        ):
            key = _request_key(req)
            was_acquired = hasattr(req, "_orbitkv_request_lease")
            if runtime.has_request(key) != was_acquired:
                raise RuntimeError("duplicate live request rid or stale OrbitKV lease")
            manager_record = runtime.acquire(key)
            previous_key = getattr(req, "_orbitkv_request_key", key)
            if previous_key != key:
                raise RuntimeError("SGLang request rid changed after KV acquisition")
            req._orbitkv_request_key = key
            if was_acquired and req._orbitkv_request_lease != manager_record.lease:
                raise RuntimeError("SGLang request carries a foreign OrbitKV lease")
            req._orbitkv_request_lease = manager_record.lease
            acquired.append(not was_acquired)
            runtime.bind_reclamation_cleanup(
                key,
                lambda certificates, releasing, current_req=req: _clear_reclamation_mirrors(
                    current_req,
                    batch.req_to_token_pool,
                    _ALLOCATOR,
                    certificates,
                    releasing,
                ),
            )
            if int(previous) != manager_record.boundary:
                raise RuntimeError(
                    "SGLang prefix boundary differs from the manager-published root"
                )
            pending, plan = runtime.prepare(key, int(target))
            records.append(pending)
            plans.append(plan)
    except Exception:
        if runtime.failure_reason is None:
            if records:
                runtime.abort_unobserved(records)
            for req, is_new in zip(batch.reqs, acquired, strict=False):
                if is_new:
                    runtime.release(_request_key(req))
                    delattr(req, "_orbitkv_request_lease")
                    delattr(req, "_orbitkv_request_key")
        raise
    return records, plans


def _free_new_req_rows(batch: Any, new_req_slots: Sequence[bool]) -> None:
    """Return only SGLang's non-authoritative request-table rows."""

    for req, is_new in zip(batch.reqs, new_req_slots, strict=True):
        if is_new and req.req_pool_idx is not None:
            key = _request_key(req)
            row = int(req.req_pool_idx)
            try:
                batch.req_to_token_pool.free(req)
                _runtime().unbind_request_row(key, row)
            except Exception as error:
                _runtime().fail_stop(
                    f"ReqToToken row rollback became uncertain: {error}"
                )
                raise FailStopped(
                    _runtime().failure_reason or "request-row rollback failed"
                ) from error


def _lower_extend_class(
    batch: Any,
    prefix_lens_cpu: Any,
    targets_cpu: Any,
    extend_num_tokens: int,
    plans: Sequence[LoweringPlan],
    class_id: int,
) -> Any:
    import torch
    from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
    from sglang.srt.utils import next_power_of_2

    class_plans = [plan.by_class[class_id] for plan in plans]
    bs = len(plans)
    prefix_lens = prefix_lens_cpu.to(batch.device, non_blocking=True)
    targets = targets_cpu.to(batch.device, non_blocking=True)
    last_loc = torch.tensor(
        [class_plan.last_location for class_plan in class_plans],
        dtype=torch.int64,
        device=batch.device,
    )
    page_ids = [
        page for class_plan in class_plans for page in class_plan.exact_new_pages
    ]
    exact_pages = torch.tensor(page_ids, dtype=torch.int64, device=batch.device)
    out_cache_loc = torch.empty(
        (int(extend_num_tokens),), dtype=torch.int64, device=batch.device
    )
    alloc_extend_kernel[(bs,)](
        prefix_lens,
        targets,
        last_loc,
        exact_pages,
        out_cache_loc,
        next_power_of_2(bs),
        _config().page_tokens,
    )
    return out_cache_loc


def _lower_decode_class(
    batch: Any,
    targets_cpu: Any,
    plans: Sequence[LoweringPlan],
    class_id: int,
) -> Any:
    import torch
    from sglang.kernels.ops.memory.allocator import alloc_decode_kernel
    from sglang.srt.utils import next_power_of_2

    class_plans = [plan.by_class[class_id] for plan in plans]
    bs = len(plans)
    targets = targets_cpu.to(batch.device, non_blocking=True)
    last_loc = torch.tensor(
        [class_plan.last_location for class_plan in class_plans],
        dtype=torch.int64,
        device=batch.device,
    )
    page_ids = [
        page for class_plan in class_plans for page in class_plan.exact_new_pages
    ]
    exact_pages = torch.tensor(page_ids, dtype=torch.int64, device=batch.device)
    out_cache_loc = torch.empty((bs,), dtype=torch.int64, device=batch.device)
    alloc_decode_kernel[(bs,)](
        targets,
        last_loc,
        exact_pages,
        out_cache_loc,
        next_power_of_2(bs),
        _config().page_tokens,
    )
    return out_cache_loc


def _lower_all_extend(
    batch: Any,
    prefix_lens_cpu: Any,
    targets_cpu: Any,
    extend_num_tokens: int,
    plans: Sequence[LoweringPlan],
) -> dict[int, Any]:
    return {
        item.class_id: _lower_extend_class(
            batch,
            prefix_lens_cpu,
            targets_cpu,
            extend_num_tokens,
            plans,
            item.class_id,
        )
        for item in _config().classes
    }


def _lower_all_decode(
    batch: Any,
    targets_cpu: Any,
    plans: Sequence[LoweringPlan],
) -> dict[int, Any]:
    return {
        item.class_id: _lower_decode_class(
            batch, targets_cpu, plans, item.class_id
        )
        for item in _config().classes
    }


def _primary_locations(locations: dict[int, Any]) -> Any:
    class_config = _config().full_class or _config().sliding_class
    if class_config is None or set(locations) != {
        item.class_id for item in _config().classes
    }:
        raise RuntimeError("lowering did not return every compiled KV class")
    return locations[class_config.class_id]


def _write_hybrid_lut(locations: dict[int, Any]) -> None:
    import torch

    full = _config().full_class
    sliding = _config().sliding_class
    if full is None or sliding is None:
        return
    full_locations = locations[full.class_id]
    swa_locations = locations[sliding.class_id]
    if int(full_locations.numel()) != int(swa_locations.numel()):
        raise RuntimeError("Full and SWA lowering cardinalities differ")
    _ALLOCATOR.set_full_to_swa_mapping(
        full_locations.to(dtype=torch.int64),
        swa_locations.to(dtype=torch.int64),
    )


def _submit_steps(records: Sequence[StepRecord]) -> list[Any]:
    submitted = []
    try:
        for record in records:
            submitted.append(_runtime().submit(record))
    except Exception as error:
        _runtime().submission_batch_failed(records, error)
        raise FailStopped(
            _runtime().failure_reason or "manager submission failed"
        ) from error
    return submitted


def _alloc_for_extend(batch: Any) -> tuple[Any, Any, Any]:
    import torch
    import sglang.srt.mem_cache.allocation as allocation
    from sglang.srt.managers.schedule_batch import ReqKvInfo

    _validate_batch(batch)
    prefix_values, extend_values, target_values = _preflight_extend_batch(batch)
    batch.maybe_evict_swa()
    prefix_tensors = [req.prefix_indices for req in batch.reqs]
    prefix_lens_cpu = torch.tensor(prefix_values, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(extend_values, dtype=torch.int64)
    targets_cpu = torch.tensor(target_values, dtype=torch.int64)
    targets_device = targets_cpu.to(batch.device, non_blocking=True)
    batch.seq_lens = targets_device

    new_req_slots = [req.req_pool_idx is None for req in batch.reqs]
    try:
        req_pool_indices = allocation.alloc_req_slots(
            batch.req_to_token_pool, batch.reqs, batch.tree_cache
        )
        req_pool_values = _integer_vector(
            "allocated req_pool_indices", req_pool_indices, len(batch.reqs)
        )
        row_capacity = int(batch.req_to_token_pool.req_to_token.shape[0])
        if len(set(req_pool_values)) != len(req_pool_values):
            raise RuntimeError("SGLang allocated duplicate request-pool rows")
        if any(value <= 0 or value >= row_capacity for value in req_pool_values):
            raise RuntimeError("SGLang allocated an out-of-range request-pool row")
        if any(
            req.req_pool_idx is None or int(req.req_pool_idx) != value
            for req, value in zip(batch.reqs, req_pool_values, strict=True)
        ):
            raise RuntimeError("SGLang allocated request-pool identity is inconsistent")
    except Exception as error:
        _runtime().fail_stop(f"SGLang request-row allocation became uncertain: {error}")
        raise FailStopped(
            _runtime().failure_reason or "request-row allocation became uncertain"
        ) from error
    _runtime().bind_request_rows(
        tuple(
            (_request_key(req), row, is_new)
            for req, row, is_new in zip(
                batch.reqs, req_pool_values, new_req_slots, strict=True
            )
        )
    )

    records: list[StepRecord] = []
    try:
        req_pool_indices_cpu = torch.tensor(req_pool_values, dtype=torch.int64)
        req_pool_indices_device = req_pool_indices_cpu.to(
            batch.device, non_blocking=True
        )
        records, plans = _prepare_steps(
            batch,
            prefix_values,
            target_values,
        )
    except Exception:
        if _runtime().failure_reason is None:
            _free_new_req_rows(batch, new_req_slots)
        raise

    try:
        locations = _lower_all_extend(
            batch,
            prefix_lens_cpu,
            targets_cpu,
            int(batch.extend_num_tokens),
            plans,
        )
        out_cache_loc = _primary_locations(locations)
        for record in records:
            _runtime().mark_lowered(record)
    except Exception as error:
        _runtime().lowering_failed(records, error)
        raise FailStopped(_runtime().failure_reason or "extend lowering failed") from error


    _submit_steps(records)

    try:
        allocation.write_cache_indices(
            out_cache_loc,
            req_pool_indices_device,
            req_pool_indices_cpu,
            prefix_lens_cpu.to(batch.device, non_blocking=True),
            prefix_lens_cpu,
            targets_device,
            targets_cpu,
            extend_lens_cpu.to(batch.device, non_blocking=True),
            extend_lens_cpu,
            prefix_tensors,
            batch.req_to_token_pool,
        )
        _write_hybrid_lut(locations)
        for req, target in zip(batch.reqs, targets_cpu.tolist(), strict=True):
            if req.kv is None:
                req.kv = ReqKvInfo(kv_allocated_len=int(target), swa_evicted_seqlen=0)
            else:
                req.kv.kv_allocated_len = int(target)
        batch._orbitkv_steps = tuple(records)
    except Exception as error:
        _runtime().candidate_mirror_failed(records, error)
        raise FailStopped(
            _runtime().failure_reason or "candidate mirror failed"
        ) from error
    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def _alloc_for_decode(batch: Any, token_per_req: int) -> Any:
    _validate_batch(batch)
    if int(token_per_req) != 1:
        raise RuntimeError("OrbitKV supports one decode token per request")
    previous, req_pool_values = _preflight_decode_batch(batch)
    batch.maybe_evict_swa()
    targets = [value + 1 for value in previous]

    import torch

    targets_cpu = torch.tensor(targets, dtype=torch.int64)
    previous_device = torch.tensor(previous, dtype=torch.int64).to(
        batch.device, non_blocking=True
    )
    req_pool_indices_device = torch.tensor(
        req_pool_values, dtype=torch.int64
    ).to(batch.device, non_blocking=True)
    batch.seq_lens = previous_device
    batch.req_pool_indices = req_pool_indices_device
    records, plans = _prepare_steps(batch, previous, targets)
    try:
        locations = _lower_all_decode(batch, targets_cpu, plans)
        out_cache_loc = _primary_locations(locations)
        for record in records:
            _runtime().mark_lowered(record)
    except Exception as error:
        _runtime().lowering_failed(records, error)
        raise FailStopped(_runtime().failure_reason or "decode lowering failed") from error


    _submit_steps(records)

    try:
        if batch.model_config.is_encoder_decoder:
            raise RuntimeError("OrbitKV does not support encoder-decoder models")
        batch.req_to_token_pool.write(
            (batch.req_pool_indices, previous_device),
            out_cache_loc.to(torch.int32),
        )
        _write_hybrid_lut(locations)
        for req in batch.reqs:
            req.kv.kv_allocated_len += 1
        batch._orbitkv_steps = tuple(records)
    except Exception as error:
        _runtime().candidate_mirror_failed(records, error)
        raise FailStopped(
            _runtime().failure_reason or "decode mirror failed"
        ) from error
    return out_cache_loc


def _manager_maybe_evict_swa(batch: Any) -> None:
    _validate_batch(batch)
    _wait_previous_steps(batch)


def _completion_domain(scheduler: Any) -> int:
    device = getattr(scheduler, "device", None)
    if isinstance(device, str):
        index = None
    else:
        index = getattr(device, "index", None)
    if index is None and isinstance(device, str) and ":" in device:
        suffix = device.rsplit(":", 1)[-1]
        index = int(suffix) if suffix.isdigit() else 0
    return int(index or 0) + 1


def _run_batch(
    original_fn: Callable[..., Any], scheduler: Any, batch: Any, *args: Any, **kwargs: Any
) -> Any:
    runtime = _runtime()
    records = tuple(getattr(batch, "_orbitkv_steps", ()))
    try:
        _validate_batch(batch)
        runtime.poll()
        if batch.reqs and not records:
            raise RuntimeError("OrbitKV forward has no submitted manager step")
        expected_keys = tuple(_request_key(req) for req in batch.reqs)
        if len(records) != len(expected_keys) or tuple(
            record.key for record in records
        ) != expected_keys:
            raise RuntimeError("OrbitKV forward records do not match batch request order")
        runtime.mark_forward(records)
    except Exception as error:
        if records:
            runtime.forward_failed(records, error)
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            runtime.failure_reason or "pre-forward manager state became uncertain"
        ) from error
    try:
        result = original_fn(scheduler, batch, *args, **kwargs)
    except Exception as error:
        runtime.forward_failed(records, error)
        raise FailStopped(runtime.failure_reason or "forward failed") from error
    try:
        launch_stream = scheduler.device_module.current_stream(scheduler.device)
        event = scheduler.device_module.Event()
        event.record(stream=launch_stream)
        runtime.register_event(records, event, _completion_domain(scheduler))
        batch._orbitkv_steps = ()
    except Exception as error:
        runtime.event_registration_failed(records, error)
        raise FailStopped(runtime.failure_reason or "event registration failed") from error
    return result


def _release_kv_cache(req: Any, tree_cache: Any, is_insert: bool = True) -> None:
    del is_insert
    if req.req_pool_idx is None:
        runtime = _runtime()
        key = _request_key(req)
        carries_lease = hasattr(req, "_orbitkv_request_lease") or hasattr(
            req, "_orbitkv_request_key"
        )
        if req.kv is None and not carries_lease and not runtime.has_request(key):
            return
        runtime.fail_stop(
            "SGLang dropped a ReqToToken row while OrbitKV request state remained"
        )
        raise FailStopped(runtime.failure_reason or "release identity was lost")
    if not tree_cache.is_chunk_cache():
        raise RuntimeError("OrbitKV release requires ChunkCache")
    if bool(tree_cache.supports_swa()) != (_config().sliding_class is not None):
        raise RuntimeError("SGLang release cache type differs from the plan")
    if tree_cache.token_to_kv_pool_allocator is not _ALLOCATOR:
        raise RuntimeError("SGLang release references a foreign KV allocator")
    if not hasattr(req, "_orbitkv_request_lease"):
        raise RuntimeError("SGLang tried to release KV without an OrbitKV lease")

    key = _request_key(req)
    if getattr(req, "_orbitkv_request_key", None) != key:
        raise RuntimeError("SGLang request rid changed before release")
    runtime = _runtime()
    row = int(req.req_pool_idx)
    runtime.release(key)
    try:
        tree_cache.req_to_token_pool.free(req)
        runtime.unbind_request_row(key, row)
        req.kv = None
        req.prefix_indices = req.prefix_indices[:0]
        delattr(req, "_orbitkv_request_lease")
        delattr(req, "_orbitkv_request_key")
    except Exception as error:
        runtime.fail_stop(f"ReqToToken release mirror became uncertain: {error}")
        raise FailStopped(runtime.failure_reason or "release mirror failed") from error


def _dtype_bytes(dtype: Any) -> int:
    import torch

    try:
        size = int(torch.empty((), dtype=dtype, device="cpu").element_size())
    except Exception as error:
        raise RuntimeError("cannot determine the SGLang KV cache element size") from error
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
        if tuple(model.full_attention_layer_ids) != full.layers or tuple(
            model.swa_attention_layer_ids
        ) != sliding.layers:
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

    global _LIMITS
    config = _config()
    server = configurator.server_args
    graph = server.cuda_graph_config
    _validate_attention_backend_contract(configurator)
    required = {
        "CUDA platform": _is_cuda_platform(),
        "CUDA device": str(configurator.device).startswith("cuda"),
        "bfloat16 KV cache": configurator.kv_cache_dtype is torch.bfloat16,
        "NHD KV layout": not _uses_hnd_kv_cache(),
        "radix cache": bool(server.disable_radix_cache),
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
    if _LIMITS is not None and _LIMITS != resolved_limits:
        raise RuntimeError("OrbitKV runtime capacities changed after initialization")
    _LIMITS = resolved_limits
    result = original_fn(configurator, *args, **kwargs)
    allocator = result.token_to_kv_pool_allocator
    if allocator is not _ALLOCATOR or _RUNTIME is None:
        raise RuntimeError("SGLang did not install the OrbitKV arena facades")
    if int(allocator.page_size) != config.page_tokens:
        raise RuntimeError("OrbitKV arena page size changed during configuration")
    kv_pool = allocator.get_kvcache()
    if config.full_class is not None and config.sliding_class is not None:
        if (
            int(result.full_max_total_num_tokens) != int(allocator.size_full)
            or int(result.swa_max_total_num_tokens) != int(allocator.size_swa)
        ):
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


def _get_internal_state(
    original_fn: Callable[..., Any], scheduler: Any, *args: Any, **kwargs: Any
) -> Any:
    result = original_fn(scheduler, *args, **kwargs)
    state = getattr(result, "internal_state", None)
    if not isinstance(state, dict) or "orbitkv_manager" in state:
        raise RuntimeError("SGLang returned an invalid internal-state namespace")
    runtime = _runtime()
    stats = runtime.stats()
    swa_activity = runtime.swa_activity()
    arena_stats = tuple(runtime.manager.arena_stats())
    if tuple(item.class_id for item in arena_stats) != tuple(
        item.class_id for item in runtime.arenas
    ):
        raise RuntimeError("manager internal-state arena order changed")
    state["orbitkv_manager"] = {
        "abi_version": 4,
        "identities": [
            {
                "engine_epoch": item.engine_epoch,
                "pool_epoch": item.pool_epoch,
                "pool_id": item.pool_id,
                "class_id": item.class_id,
                "backend_domain": item.backend_domain,
                "page_count": item.page_count,
                "page_tokens": item.page_tokens,
                "backend_base_index": item.backend_base_index,
                "first_page_id": item.first_page_id,
            }
            for item in runtime.arenas
        ],
        "arena_stats": [
            {
                "engine_epoch": item.engine_epoch,
                "pool_epoch": item.pool_epoch,
                "pool_id": item.pool_id,
                "page_count": item.page_count,
                "class_id": item.class_id,
                "backend_domain": item.backend_domain,
                "first_page_id": item.first_page_id,
                "free_pages": item.free_pages,
                "reserved_pages": item.reserved_pages,
                "writing_pages": item.writing_pages,
                "active_pages": item.active_pages,
                "retiring_pages": item.retiring_pages,
                "quarantined_pages": item.quarantined_pages,
                "exhausted_pages": item.exhausted_pages,
            }
            for item in arena_stats
        ],
        "manager_stats": {
            "active_requests": stats.active_requests,
            "prepared_steps": stats.prepared_steps,
            "submitted_steps": stats.submitted_steps,
            "free_pages": stats.free_pages,
            "reserved_pages": stats.reserved_pages,
            "writing_pages": stats.writing_pages,
            "active_pages": stats.active_pages,
            "retiring_pages": stats.retiring_pages,
            "quarantined_pages": stats.quarantined_pages,
            "exhausted_pages": stats.exhausted_pages,
            "pending_reclamations": stats.pending_reclamations,
        },
        "swa_activity": {
            "status": "exposed",
            "applicable": _config().sliding_class is not None,
            "swa_retirement_certificates": swa_activity.retirement_certificates,
            "swa_pages_reclaimed": swa_activity.pages_reclaimed,
            "swa_wrap_events": swa_activity.wrap_events,
        },
    }
    return result


def _register() -> None:
    global _CONFIG
    if _CONFIG is not None:
        raise RuntimeError("OrbitKV plugin was registered more than once")
    _validate_plugin_selection()
    _validate_sglang_revision()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _preflight_hook_targets(HookRegistry)
    _CONFIG = load_config()

    hooks = (
        (HOOK_TARGETS[0], _build_token_to_kv_pool_allocator, HookType.REPLACE),
        (HOOK_TARGETS[1], _alloc_for_extend, HookType.REPLACE),
        (HOOK_TARGETS[2], _alloc_for_decode, HookType.REPLACE),
        (HOOK_TARGETS[3], _manager_maybe_evict_swa, HookType.REPLACE),
        (HOOK_TARGETS[4], _release_kv_cache, HookType.REPLACE),
        (HOOK_TARGETS[5], _run_batch, HookType.AROUND),
        (HOOK_TARGETS[6], _validate_configurator, HookType.AROUND),
        (HOOK_TARGETS[7], _get_internal_state, HookType.AROUND),
    )
    for target, hook, hook_type in hooks:
        HookRegistry.register(target, hook, hook_type)
    HookRegistry.apply_hooks()
    missing = [target for target in HOOK_TARGETS if target not in HookRegistry._patched]
    if missing:
        raise RuntimeError(
            "SGLang silently rejected canonical OrbitKV hooks: " + ", ".join(missing)
        )
    _validate_propagated_aliases()


def register() -> None:
    try:
        _register()
    except Exception as error:
        raise SystemExit(f"OrbitKV canonical adapter activation failed: {error}") from error


def _install_test_state(
    *,
    config: ManagerPlanConfig | None = None,
    limits: RuntimeLimits | None = None,
    runtime: CanonicalRuntime | None = None,
    factory: ManagerFactoryProtocol | None = None,
) -> None:
    global _CONFIG, _LIMITS, _RUNTIME, _FACTORY, _ALLOCATOR
    _CONFIG = config
    _LIMITS = limits
    _RUNTIME = runtime
    _ALLOCATOR = None
    if factory is not None:
        _FACTORY = factory
