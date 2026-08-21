from __future__ import annotations

from typing import Any

from ..runtime import ArenaRegistration, FailStopped
from . import lowering
from . import state as _state
from .state import (
    _arena_available_tokens,
    _arena_available_tokens_batch,
    _config,
    _limits,
    _new_runtime,
    _runtime,
)
from .mirror_cleanup import _mirror_cleanup_coordinator

_FACADE_TYPES: tuple[type, type, type] | None = None


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
        runtime = _runtime()
        runtime.poll()
        stats, arena_stats = runtime.census()
        live = (
            stats.active_requests,
            stats.active_snapshots,
            stats.active_prefixes,
            stats.evicted_prefixes,
            stats.prepared_steps,
            stats.submitted_steps,
            stats.reserved_pages,
            stats.writing_pages,
            stats.active_pages,
            stats.retiring_pages,
            stats.quarantined_pages,
            stats.exhausted_pages,
            stats.pending_reclamations,
            stats.total_request_page_refs,
            stats.total_prefix_page_refs,
            stats.total_reader_pins,
        )
        if any(live) or any(
            item.free_pages != item.page_count
            or item.reserved_pages
            or item.writing_pages
            or item.active_pages
            or item.retiring_pages
            or item.quarantined_pages
            or item.exhausted_pages
            or item.request_page_refs
            or item.prefix_page_refs
            or item.reader_pins
            for item in arena_stats
        ):
            runtime.fail_stop("SGLang cleared a non-quiescent OrbitKV arena")
            raise FailStopped(runtime.failure_reason or "non-quiescent arena clear")
        mapping = getattr(self, "full_to_swa_index_mapping", None)
        if mapping is not None and _config().full_class is not None:
            import torch

            if (
                type(mapping) is not torch.Tensor
                or int(mapping.numel()) <= 1
                or not torch.equal(mapping[:-1], torch.zeros_like(mapping[:-1]))
                or not torch.equal(
                    mapping[-1:],
                    torch.full_like(mapping[-1:], -1),
                )
            ):
                runtime.fail_stop("Hybrid mirror was not empty at allocator clear")
                raise FailStopped(runtime.failure_reason or "nonempty Hybrid mirror")

    def resize(self, *_args: Any, **_kwargs: Any) -> None:
        self._native_authority_error("resize")

    def get_cpu_copy(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("get_cpu_copy")

    def load_cpu_copy(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("load_cpu_copy")

    def translate_kv_indices_for_transfer(self, *_args: Any, **_kwargs: Any) -> Any:
        self._native_authority_error("translate_kv_indices_for_transfer")

    def free_group_begin(self) -> None:
        if getattr(self, "_orbitkv_free_group_state", None) != "idle":
            runtime = _runtime()
            runtime.fail_stop("SGLang nested an OrbitKV release group")
            raise FailStopped(runtime.failure_reason or "nested release group")
        self._orbitkv_free_group_state = "collecting"
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        if getattr(self, "_orbitkv_free_group_state", None) != "collecting":
            runtime = _runtime()
            runtime.fail_stop("SGLang ended an inactive OrbitKV release group")
            raise FailStopped(runtime.failure_reason or "inactive release group")
        candidates = tuple(self.free_group)
        self._orbitkv_free_group_state = "flushing"
        self.is_not_in_free_group = True
        self.free_group = []
        try:
            if candidates:
                lowering._flush_release_group(candidates)
        except Exception as error:
            # A failed flush is terminal.  Keep the facade out of IDLE so no
            # later scheduler action can accidentally reuse this group.
            runtime = _runtime()
            if runtime.failure_reason is None:
                runtime.fail_stop(f"OrbitKV release-group flush failed: {error}")
            if isinstance(error, FailStopped):
                raise
            raise FailStopped(
                runtime.failure_reason or "release-group flush failed"
            ) from error
        self._orbitkv_free_group_state = "idle"


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
    facade._orbitkv_free_group_state = "idle"


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
            return min(self._orbitkv_available_sizes())

        def _orbitkv_available_sizes(self) -> tuple[int, int]:
            return _arena_available_tokens_batch(
                (
                    self.full_attn_allocator.class_id,
                    self.swa_attn_allocator.class_id,
                )
            )

        def full_available_size(self) -> int:
            return self.full_attn_allocator.available_size()

        def swa_available_size(self) -> int:
            return self.swa_attn_allocator.available_size()

        def _conserve_full_available_size(self) -> int:
            return self.full_available_size()

        def _conserve_swa_available_size(self) -> int:
            return self.swa_available_size()

        def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
            full_available, swa_available = self._orbitkv_available_sizes()
            return (
                int(num_full_pages) <= full_available // self.page_size
                and int(num_swa_pages) <= swa_available // self.page_size
            )

        def get_kvcache(self) -> Any:
            return self._kvcache

        def translate_loc_from_full_to_swa(self, kv_indices: Any) -> Any:
            return self.full_to_swa_index_mapping[kv_indices]

        def set_full_to_swa_mapping(self, full_indices: Any, swa_indices: Any) -> None:
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
            self.swa_attn_allocator = _ArenaAvailability(class_id, size_swa, page_size)
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

        def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
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
    if _state._ALLOCATOR is not None or _state._RUNTIME is not None:
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
    _state._ALLOCATOR = allocator
    coordinator = _mirror_cleanup_coordinator(req_to_token_pool, allocator)
    _runtime().bind_prefix_eviction_cleanup(coordinator)
    return allocator
