from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..config import ManagerPlanConfig
from ..ffi import CtypesManagerFactory
from ..runtime import (
    ArenaRegistration,
    CanonicalRuntime,
    FailStopped,
    ManagerCreateSettings,
    ManagerFactoryProtocol,
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
_MIRROR_CLEANUP: Any = None
_COUNTER_NAMES = (
    "prefix_matches",
    "prefix_hits",
    "prefix_publishes",
    "prefix_evictions",
    "prefix_evicted_full_tokens",
    "prefix_evicted_swa_tokens",
    "cow_copy_intents",
    "cow_move_calls",
    "cow_copied_tokens",
    "mirror_validation_calls",
    "mirror_syncs",
    "prefix_global_alias_scans",
)
_COUNTERS = {name: 0 for name in _COUNTER_NAMES}


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
        maximum_prefixes=total_pages,
        maximum_reclamations=total_pages,
        maximum_step_tokens=limits.chunked_prefill_tokens,
    )
    manager = _FACTORY.create(_config(), settings, values)
    try:
        runtime = CanonicalRuntime(_config(), manager)
        for registration, identity in zip(values, runtime.arenas, strict=True):
            if (
                identity.class_id != registration.class_id
                or identity.pool_id != registration.pool_id
                or identity.backend_domain != registration.backend_domain
                or identity.page_count != registration.page_count
                or identity.backend_base_index != registration.backend_base_index
            ):
                runtime.fail_stop(
                    "manager arena differs from SGLang physical storage"
                )
                raise FailStopped(runtime.failure_reason or "arena identity mismatch")
    except Exception:
        # Factory creation transfers the native handle to this function.  Do
        # not publish a partial runtime, and always terminate that handle when
        # constructor or post-construction identity validation fails.
        manager.destroy()
        raise
    _RUNTIME = runtime
    return runtime


def _arena_available_tokens(class_id: int) -> int:
    return _arena_available_tokens_batch((class_id,))[0]


def _arena_available_tokens_batch(class_ids: Sequence[int]) -> tuple[int, ...]:
    requested = tuple(class_ids)
    if not requested or len(set(requested)) != len(requested):
        raise RuntimeError("arena availability classes must be nonempty and unique")
    runtime = _runtime()
    runtime.poll()
    _manager_stats, arena_values = runtime.census()
    by_class = {item.class_id: item for item in arena_values}
    if len(by_class) != len(arena_values) or any(
        class_id not in by_class for class_id in requested
    ):
        runtime.fail_stop("manager returned ambiguous per-class arena stats")
        raise FailStopped(runtime.failure_reason or "ambiguous arena stats")
    return tuple(
        by_class[class_id].free_pages * runtime.page_tokens
        for class_id in requested
    )


def _counter_add(name: str, value: int = 1) -> None:
    if name not in _COUNTERS or isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("OrbitKV activity counter identity changed")
    if value < 0:
        raise RuntimeError("OrbitKV activity counters cannot decrease")
    _COUNTERS[name] += value


def _activity_counters() -> dict[str, int]:
    return dict(_COUNTERS)


def _install_test_state(
    *,
    config: ManagerPlanConfig | None = None,
    limits: RuntimeLimits | None = None,
    runtime: CanonicalRuntime | None = None,
    factory: ManagerFactoryProtocol | None = None,
) -> None:
    global _CONFIG, _LIMITS, _RUNTIME, _FACTORY, _ALLOCATOR, _MIRROR_CLEANUP
    _CONFIG = config
    _LIMITS = limits
    _RUNTIME = runtime
    _ALLOCATOR = None
    _MIRROR_CLEANUP = None
    for name in _COUNTERS:
        _COUNTERS[name] = 0
    if factory is not None:
        _FACTORY = factory
