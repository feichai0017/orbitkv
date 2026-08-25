from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Sequence

from ..config import ManagerPlanConfig
from ..ffi import CtypesManagerFactory, CtypesStatePool
from ..runtime import (
    ArenaRegistration,
    CanonicalRuntime,
    FailStopped,
    ManagerCreateSettings,
    ManagerFactoryProtocol,
    StatePoolConfig,
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
_FIXED_STATE: Any = None
_DATA_PLANE: Any = None
_STRUCTURED_ARENAS: tuple[Any, ...] = ()
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
    "token_disposition_batches",
    "token_policy_evictions",
    "relocation_batches",
    "relocation_moves",
    "relocation_reclaimed_pages",
    "relocation_copy_events",
    "relocation_copy_tokens",
    "fixed_state_prepares",
    "fixed_state_clears",
    "fixed_state_copies",
    "fixed_state_events",
    "fixed_state_retirements",
    "fixed_state_acks",
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


def _data_plane() -> Any:
    if _DATA_PLANE is None:
        raise RuntimeError("OrbitKV structured data plane is not initialized")
    return _DATA_PLANE


def _uses_structured_data_plane() -> bool:
    config = _config()
    enabled = os.environ.get("ORBITKV_STRUCTURED_DATA_PLANE", "0").strip().lower()
    if enabled not in ("0", "1", "false", "true"):
        raise RuntimeError(
            "ORBITKV_STRUCTURED_DATA_PLANE must be 0/1/false/true"
        )
    if enabled in ("0", "false"):
        return False
    retentions = tuple(item.retention for item in config.classes)
    reclamation = getattr(config, "token_reclamation", None)
    if (
        retentions not in (("full",), ("full", "sliding"))
        or not all(item.storage == "token_kv" for item in config.classes)
        or getattr(reclamation, "mode", "off") != "off"
    ):
        raise RuntimeError(
            "structured data plane requires Full or Full+SWA token_kv "
            "with token relocation disabled"
        )
    return True


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


def _new_fixed_state(req_to_token_pool: Any, *, device_module: Any | None = None) -> Any:
    global _FIXED_STATE
    if not _config().fixed_states:
        return None
    if _FIXED_STATE is not None:
        raise RuntimeError("OrbitKV fixed-state adapter is already initialized")
    runtime = _runtime()
    byte_count = _config().fixed_state_byte_count
    slot_count = int(getattr(req_to_token_pool.mamba_pool, "size", 0))
    if byte_count <= 0 or slot_count < 2:
        raise RuntimeError("OrbitKV fixed-state geometry is invalid")
    from .fixed_state import FixedStateCoordinator

    pool = CtypesStatePool(
        _config().library_path,
        StatePoolConfig(
            runtime.engine_epoch,
            runtime.engine_epoch + 2,
            byte_count,
            len(_config().classes) + 1,
            slot_count,
        ),
    )
    try:
        coordinator = FixedStateCoordinator(
            pool,
            req_to_token_pool,
            failure_sink=runtime.fail_stop,
            device_module=device_module,
        )
        coordinator.install_allocator_facade()
    except Exception:
        pool.close()
        raise
    _FIXED_STATE = coordinator
    return coordinator


def _new_data_plane(
    token_to_kv_pool: Any, *, device_module: Any
) -> Any | None:
    """Install the scoped structured-arena write bridge after pool validation."""

    global _DATA_PLANE, _STRUCTURED_ARENAS
    if not _uses_structured_data_plane():
        return None
    if _DATA_PLANE is not None or _STRUCTURED_ARENAS:
        raise RuntimeError("OrbitKV structured data plane was initialized twice")
    try:
        from .external_append import SglangExternalWriteAdapter
        from .structured_arena import build_sglang_structured_arenas
    except ModuleNotFoundError as error:
        if error.name == "orbitkv_runtime":
            raise RuntimeError(
                "structured data plane requires the structured-data-plane extra"
            ) from error
        raise

    arenas = build_sglang_structured_arenas(
        _config(), _runtime(), token_to_kv_pool
    )
    component_device = arenas[0].components[0].tensor.device
    adapter = SglangExternalWriteAdapter(
        arenas,
        device_module=device_module,
        device=component_device,
        adapter_id="sglang-v0517",
    )
    _STRUCTURED_ARENAS = arenas
    _DATA_PLANE = adapter
    return adapter


def _close_owned_runtime(runtime: Any) -> None:
    """Close every adapter-owned resource even if an earlier close fails."""

    failures: list[tuple[str, BaseException]] = []
    if _DATA_PLANE is not None:
        try:
            _DATA_PLANE.close()
        except Exception as error:
            failures.append(("structured data plane", error))
    if _FIXED_STATE is not None:
        try:
            _FIXED_STATE.shutdown()
        except Exception as error:
            failures.append(("fixed state", error))
    try:
        runtime.close()
    except Exception as error:
        for label, previous in failures:
            error.add_note(f"{label} shutdown also failed: {previous!r}")
        raise
    if failures:
        label, error = failures[0]
        for other_label, other in failures[1:]:
            error.add_note(f"{other_label} shutdown also failed: {other!r}")
        raise error


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


def _requires_disabled_radix_cache() -> bool:
    """Whether the plan requires request-private, no-prefix cache mode."""

    config = _config()
    reclamation = getattr(config, "token_reclamation", None)
    return bool(getattr(config, "fixed_states", ())) or (
        getattr(reclamation, "mode", "off") != "off"
    )


def _fixed_state_descriptors() -> list[dict[str, Any]]:
    """Return the loaded fixed-width state projection as JSON-safe values."""

    return [
        {
            "name": item.name,
            "kind": item.kind,
            "layers": list(item.layers),
            "state_bytes_per_layer": item.state_bytes_per_layer,
            "checkpoint_slots_per_request": item.checkpoint_slots_per_request,
            "kernel_width": item.kernel_width,
            "byte_count": item.byte_count,
        }
        for item in _config().fixed_states
    ]


def _tree_cache_type(tree_cache: Any) -> dict[str, str]:
    """Return the concrete cache type without relying on its repr."""

    tree_cache_type = type(tree_cache)
    return {
        "module": tree_cache_type.__module__,
        "qualname": tree_cache_type.__qualname__,
    }


def _install_test_state(
    *,
    config: ManagerPlanConfig | None = None,
    limits: RuntimeLimits | None = None,
    runtime: CanonicalRuntime | None = None,
    factory: ManagerFactoryProtocol | None = None,
) -> None:
    global _CONFIG, _LIMITS, _RUNTIME, _FACTORY, _ALLOCATOR, _MIRROR_CLEANUP
    global _FIXED_STATE, _DATA_PLANE, _STRUCTURED_ARENAS
    _CONFIG = config
    _LIMITS = limits
    _RUNTIME = runtime
    _ALLOCATOR = None
    _MIRROR_CLEANUP = None
    _FIXED_STATE = None
    _DATA_PLANE = None
    _STRUCTURED_ARENAS = ()
    for name in _COUNTERS:
        _COUNTERS[name] = 0
    if factory is not None:
        _FACTORY = factory
