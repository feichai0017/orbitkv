from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Sequence

from ..config import RuntimeConfig
from ..ffi import CtypesRuntimeSession, CtypesStatePool
from ..runtime import (
    ArenaRegistration,
    CacheSharingPolicy,
    FailStopped,
    ManagerCreateSettings,
    SessionCreateSettings,
    StatePoolConfig,
)


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    maximum_running_requests: int
    chunked_prefill_tokens: int
    maximum_context_tokens: int


@dataclass(frozen=True, slots=True)
class RuntimeAttentionBackendProof:
    backend_class: str
    backend_module: str
    prefill_backend: str
    decode_backend: str
    has_local_attention: bool
    attention_chunk_size: int
    page_size: int
    compiled_layer_ids: tuple[int, ...]
    use_irope_layer_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_class": self.backend_class,
            "backend_module": self.backend_module,
            "prefill_backend": self.prefill_backend,
            "decode_backend": self.decode_backend,
            "has_local_attention": self.has_local_attention,
            "attention_chunk_size": self.attention_chunk_size,
            "page_size": self.page_size,
            "compiled_layer_ids": list(self.compiled_layer_ids),
            "use_irope_layer_ids": list(self.use_irope_layer_ids),
        }


_CONFIG: RuntimeConfig | None = None
_PRODUCT_PROFILE: Any = None
_LIMITS: RuntimeLimits | None = None
_RUNTIME: Any = None
_RUNTIME_BACKEND_PROOF: RuntimeAttentionBackendProof | None = None
_ALLOCATOR: Any = None
_MIRROR_CLEANUP: Any = None
_FIXED_STATE: Any = None
_FIXED_STATE_ALLOCATOR_RESTORE: tuple[Any, Any, Any] | None = None
_INITIALIZATION_LOCK = Lock()
_INITIALIZATION_TOKEN: object | None = None
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


def _config() -> RuntimeConfig:
    if _CONFIG is None:
        raise RuntimeError("OrbitKV manager plan is not loaded")
    return _CONFIG


def _runtime() -> Any:
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


def _selected_product_runtime_profile() -> Any | None:
    """Return the exact admitted product profile, detecting later drift."""

    from ..runtime_admission import product_takeover_profile

    selected = product_takeover_profile(_config())
    if _PRODUCT_PROFILE is not None and selected != _PRODUCT_PROFILE:
        raise RuntimeError(
            "OrbitKV configured runtime differs from its admitted product profile"
        )
    return _PRODUCT_PROFILE if _PRODUCT_PROFILE is not None else selected


def _uses_runtime_session() -> bool:
    """Route declared product profiles exclusively to the native session."""

    return _selected_product_runtime_profile() is not None


def _admit_product_runtime_profile() -> Any:
    """Require the configured direct-source product runtime profile."""

    from ..runtime_admission import admit_product_takeover_config

    admitted = admit_product_takeover_config(_config())
    if _PRODUCT_PROFILE is not None and admitted != _PRODUCT_PROFILE:
        raise RuntimeError(
            "OrbitKV runtime admission changed after product initialization"
        )
    return admitted


def _cache_sharing_policy(profile: Any | None = None) -> CacheSharingPolicy:
    """Resolve native cache visibility only from a selected product profile."""

    selected = _selected_product_runtime_profile() if profile is None else profile
    value = getattr(selected, "cache_policy", None)
    policies = {
        "request_private": CacheSharingPolicy.REQUEST_PRIVATE,
        "shared_prefix": CacheSharingPolicy.SHARED_PREFIX,
    }
    policy = policies.get(value)
    if policy is None:
        raise RuntimeError(
            "OrbitKV product cache policy is unavailable or unsupported"
        )
    return policy


def _bind_session_cleanup(callback: Any) -> None:
    """Bind the sole mirror authority before a session can acquire work."""

    runtime = _runtime()
    if not callable(callback):
        raise TypeError(
            "OrbitKV runtime-session mirror cleanup must be callable"
        )
    runtime.bind_mirror_cleanup(callback)


def _new_runtime(
    registrations: Sequence[ArenaRegistration], *, mirror_cleanup: Any = None
) -> Any:
    global _RUNTIME
    if _RUNTIME is not None:
        raise RuntimeError("OrbitKV manager is already initialized")
    profile = _admit_product_runtime_profile()
    values = tuple(registrations)
    if not values or len(values) != len(_config().classes):
        raise RuntimeError("one physical arena is required for every KV class")
    total_pages = sum(item.page_count for item in values)
    if total_pages <= 0:
        raise RuntimeError("OrbitKV physical arenas must be nonempty")
    limits = _limits()
    cache_sharing_policy = _cache_sharing_policy(profile)
    settings = SessionCreateSettings(
        manager=ManagerCreateSettings(
            maximum_requests=limits.maximum_running_requests,
            maximum_operations=limits.maximum_running_requests,
            maximum_prefixes=total_pages,
            maximum_reclamations=total_pages,
            maximum_step_tokens=limits.chunked_prefill_tokens,
        ),
        cache_sharing_policy=cache_sharing_policy,
    )
    session = CtypesRuntimeSession.create(_config(), settings, values)
    try:
        from ..session_runtime import SessionRuntime, unbound_mirror_cleanup

        bootstrap_cleanup = (
            unbound_mirror_cleanup if mirror_cleanup is None else mirror_cleanup
        )
        runtime = SessionRuntime(session, mirror_cleanup=bootstrap_cleanup)
        if runtime.cache_sharing_policy is not cache_sharing_policy:
            raise RuntimeError(
                "native runtime-session cache policy differs from product admission"
            )
    except Exception as error:
        # Session creation transfers the sole native handle here. Until
        # SessionRuntime is successfully published, this function owns it.
        try:
            session.close()
        except Exception as close_error:
            error.add_note(
                "runtime-session construction rollback also failed: "
                f"{close_error!r}"
            )
        raise
    _RUNTIME = runtime
    return runtime


def _new_fixed_state(req_to_token_pool: Any, *, device_module: Any | None = None) -> Any:
    global _FIXED_STATE, _FIXED_STATE_ALLOCATOR_RESTORE
    if not _config().fixed_states:
        return None
    if _FIXED_STATE is not None:
        raise RuntimeError("OrbitKV fixed-state adapter is already initialized")
    if _FIXED_STATE_ALLOCATOR_RESTORE is not None:
        raise RuntimeError("OrbitKV fixed-state allocator restore is still pending")
    runtime = _runtime()
    byte_count = _config().fixed_state_byte_count
    slot_count = int(getattr(req_to_token_pool.mamba_pool, "size", 0))
    if byte_count <= 0 or slot_count < 2:
        raise RuntimeError("OrbitKV fixed-state geometry is invalid")
    from .fixed_state import FixedStateCoordinator

    original_allocator = req_to_token_pool.mamba_allocator
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
    except Exception as error:
        try:
            req_to_token_pool.mamba_allocator = original_allocator
        except Exception as restore_error:
            error.add_note(
                "fixed-state allocator rollback also failed: "
                f"{restore_error!r}"
            )
        try:
            pool.close()
        except Exception as close_error:
            error.add_note(
                f"fixed-state pool rollback also failed: {close_error!r}"
            )
        raise
    installed_allocator = req_to_token_pool.mamba_allocator
    if installed_allocator is original_allocator:
        pool.close()
        raise RuntimeError("OrbitKV fixed-state allocator facade was not installed")
    _FIXED_STATE_ALLOCATOR_RESTORE = (
        req_to_token_pool,
        original_allocator,
        installed_allocator,
    )
    _FIXED_STATE = coordinator
    return coordinator


_NO_EXPECTED_RUNTIME = object()


def _begin_initialization() -> object:
    """Claim the process-global initialization transaction."""

    global _INITIALIZATION_TOKEN
    if not _INITIALIZATION_LOCK.acquire(blocking=False):
        raise RuntimeError("OrbitKV manager initialization is already in progress")
    try:
        if any(
            (
                _LIMITS is not None,
                _RUNTIME is not None,
                _RUNTIME_BACKEND_PROOF is not None,
                _ALLOCATOR is not None,
                _MIRROR_CLEANUP is not None,
                _FIXED_STATE is not None,
                _FIXED_STATE_ALLOCATOR_RESTORE is not None,
            )
        ):
            raise RuntimeError("OrbitKV manager is already initialized")
        token = object()
        _INITIALIZATION_TOKEN = token
        return token
    except Exception:
        _INITIALIZATION_LOCK.release()
        raise


def _finish_initialization(token: object) -> None:
    """Release one successfully claimed initialization transaction."""

    global _INITIALIZATION_TOKEN
    if _INITIALIZATION_TOKEN is not token:
        raise RuntimeError("OrbitKV initialization transaction identity changed")
    _INITIALIZATION_TOKEN = None
    _INITIALIZATION_LOCK.release()


def _pending_initialization_token() -> object:
    """Return the transaction held between chunked pool and backend init."""

    if _INITIALIZATION_TOKEN is None:
        raise RuntimeError("OrbitKV has no pending initialization transaction")
    return _INITIALIZATION_TOKEN


def _detach_initialization_state(
    expected_runtime: Any = _NO_EXPECTED_RUNTIME,
) -> tuple[Any | None, Any | None, tuple[Any, Any, Any] | None]:
    """Unpublish initialization state before any fallible resource cleanup."""

    global _LIMITS, _RUNTIME, _RUNTIME_BACKEND_PROOF
    global _ALLOCATOR, _MIRROR_CLEANUP
    global _FIXED_STATE, _FIXED_STATE_ALLOCATOR_RESTORE
    if (
        expected_runtime is not _NO_EXPECTED_RUNTIME
        and _RUNTIME is not expected_runtime
    ):
        raise RuntimeError(
            "OrbitKV runtime shutdown does not match the published runtime"
        )
    runtime = _RUNTIME
    fixed_state = _FIXED_STATE
    fixed_state_allocator_restore = _FIXED_STATE_ALLOCATOR_RESTORE
    _LIMITS = None
    _RUNTIME = None
    _RUNTIME_BACKEND_PROOF = None
    _ALLOCATOR = None
    _MIRROR_CLEANUP = None
    _FIXED_STATE = None
    _FIXED_STATE_ALLOCATOR_RESTORE = None
    return runtime, fixed_state, fixed_state_allocator_restore


def _close_initialization_resources(
    runtime: Any | None,
    fixed_state: Any | None,
    fixed_state_allocator_restore: tuple[Any, Any, Any] | None,
) -> None:
    """Close detached resources even if an earlier close fails."""

    failures: list[tuple[str, BaseException]] = []
    if fixed_state_allocator_restore is not None:
        req_to_token_pool, original_allocator, installed_allocator = (
            fixed_state_allocator_restore
        )
        try:
            if req_to_token_pool.mamba_allocator is not installed_allocator:
                raise RuntimeError(
                    "SGLang Mamba allocator changed before OrbitKV shutdown"
                )
            req_to_token_pool.mamba_allocator = original_allocator
        except Exception as error:
            failures.append(("fixed-state allocator", error))
    if fixed_state is not None:
        try:
            fixed_state.shutdown()
        except Exception as error:
            failures.append(("fixed state", error))
    if runtime is not None:
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


def _close_owned_runtime(runtime: Any) -> None:
    """Unpublish and close every adapter-owned runtime resource."""

    detached = _detach_initialization_state(runtime)
    _close_initialization_resources(*detached)


def _rollback_initialization(
    error: BaseException, *, token: object | None = None
) -> None:
    """Restore retryable globals while preserving the initialization error."""

    if token is not None and _INITIALIZATION_TOKEN is not token:
        error.add_note("OrbitKV initialization rollback lost transaction ownership")
        return
    detached = _detach_initialization_state()
    try:
        _close_initialization_resources(*detached)
    except Exception as cleanup_error:
        error.add_note(
            f"OrbitKV initialization rollback also failed: {cleanup_error!r}"
        )


def _publish_runtime_backend_proof(
    proof: RuntimeAttentionBackendProof, *, token: object
) -> None:
    """Publish one fully validated backend proof inside the init transaction."""

    global _RUNTIME_BACKEND_PROOF
    if _INITIALIZATION_TOKEN is not token:
        raise RuntimeError("OrbitKV backend proof lost initialization ownership")
    if _RUNTIME is None or _ALLOCATOR is None:
        raise RuntimeError("OrbitKV backend proof preceded KV initialization")
    if _RUNTIME_BACKEND_PROOF is not None:
        raise RuntimeError("OrbitKV runtime backend proof was published twice")
    if not isinstance(proof, RuntimeAttentionBackendProof):
        raise TypeError("OrbitKV runtime backend proof has an invalid type")
    _RUNTIME_BACKEND_PROOF = proof


def _runtime_backend_proof() -> dict[str, Any] | None:
    """Return a detached JSON-safe copy of the loaded runtime proof."""

    if _RUNTIME_BACKEND_PROOF is None:
        if getattr(_config(), "chunked_class", None) is not None:
            raise RuntimeError("chunked runtime backend proof is not published")
        return None
    return _RUNTIME_BACKEND_PROOF.as_dict()


def _arena_available_tokens(class_id: int) -> int:
    return _arena_available_tokens_batch((class_id,))[0]


def _arena_available_tokens_batch(class_ids: Sequence[int]) -> tuple[int, ...]:
    requested = tuple(class_ids)
    if not requested or len(set(requested)) != len(requested):
        raise RuntimeError("arena availability classes must be nonempty and unique")
    runtime = _runtime()
    arena_values = runtime.capacity_arena_stats()
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
    """Whether the admitted product policy requires no-prefix mode."""

    return _cache_sharing_policy() is CacheSharingPolicy.REQUEST_PRIVATE


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
    config: RuntimeConfig | None = None,
    limits: RuntimeLimits | None = None,
    runtime: Any = None,
    product_profile: Any = None,
) -> None:
    global _CONFIG, _PRODUCT_PROFILE, _LIMITS, _RUNTIME, _RUNTIME_BACKEND_PROOF
    global _ALLOCATOR, _MIRROR_CLEANUP
    global _FIXED_STATE, _FIXED_STATE_ALLOCATOR_RESTORE
    _CONFIG = config
    _PRODUCT_PROFILE = product_profile
    _LIMITS = limits
    _RUNTIME = runtime
    _RUNTIME_BACKEND_PROOF = None
    _ALLOCATOR = None
    _MIRROR_CLEANUP = None
    _FIXED_STATE = None
    _FIXED_STATE_ALLOCATOR_RESTORE = None
    for name in _COUNTERS:
        _COUNTERS[name] = 0
