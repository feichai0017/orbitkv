from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .ffi import WIRE_VERSION
from .runtime import (
    CacheSharingPolicy,
    disabled_pressure_report,
    pressure_enabled_from_environment,
)
from .session_runtime import SessionRuntime
from .bridge import state as _state
from .bridge.state import _config, _runtime


_SESSION_COUNTER_KEYS = frozenset(
    {
        "forward_events",
        "completion_values",
        "event_queries",
        "event_waits",
        "fail_stop_count",
    }
)


def _cache_policy(tree_cache: Any) -> str:
    """Validate the concrete cache facade against the compiled policy."""

    request_private = _state._requires_disabled_radix_cache()
    policy = "request_private" if request_private else "shared_prefix"
    expected_disabled = bool(request_private)
    no_prefix = getattr(tree_cache, "_no_prefix", None)
    finished_insert_disabled = getattr(
        tree_cache, "disable_finished_insert", None
    )
    if (
        type(no_prefix) is not bool
        or no_prefix is not expected_disabled
        or type(finished_insert_disabled) is not bool
        or finished_insert_disabled is not expected_disabled
    ):
        expected = "request_private" if request_private else "shared_prefix"
        raise RuntimeError(
            "OrbitKV cache facade differs from the compiled "
            f"{expected} policy"
        )
    return policy


def augment_internal_state(
    native_get_internal_state: Callable[..., Any],
    scheduler: Any,
    *args: Any,
    owner: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Attach native OrbitKV session state to SGLang diagnostics."""

    result = native_get_internal_state(scheduler, *args, **kwargs)
    state = getattr(result, "internal_state", None)
    if not isinstance(state, dict) or "orbitkv_manager" in state:
        raise RuntimeError("SGLang returned an invalid internal-state namespace")
    runtime = _runtime()
    if not isinstance(runtime, SessionRuntime):
        raise RuntimeError(
            "OrbitKV product runtime is not a native SessionRuntime"
        )
    tree_cache = getattr(scheduler, "tree_cache", None)
    cache_policy = _cache_policy(tree_cache)
    expected_native_policy = (
        CacheSharingPolicy.REQUEST_PRIVATE
        if cache_policy == "request_private"
        else CacheSharingPolicy.SHARED_PREFIX
    )
    if getattr(runtime, "cache_sharing_policy", None) is not expected_native_policy:
        raise RuntimeError(
            "native runtime-session cache policy differs from product admission"
        )
    if getattr(runtime, "mirror_cleanup_bound", None) is not True:
        raise RuntimeError(
            "native runtime-session cleanup authority is not bound"
        )
    if pressure_enabled_from_environment():
        raise RuntimeError(
            "native runtime sessions do not support pressure telemetry"
        )
    runtime.poll()
    if _state._FIXED_STATE is not None:
        _state._FIXED_STATE.poll()
    stats, arena_stats = runtime.census()
    if tuple(item.class_id for item in arena_stats) != tuple(
        item.class_id for item in runtime.arenas
    ):
        raise RuntimeError("manager internal-state arena order changed")
    backend_proof = _state._runtime_backend_proof()
    runtime_proof = None
    if backend_proof is not None:
        max_prefill_tokens = getattr(scheduler, "max_prefill_tokens", None)
        max_running_requests = getattr(scheduler, "max_running_requests", None)
        effective_requests = state.get("effective_max_running_requests_per_dp")
        for name, value in (
            ("max_prefill_tokens", max_prefill_tokens),
            ("max_running_requests", max_running_requests),
            (
                "effective_max_running_requests_per_dp",
                effective_requests,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(
                    f"SGLang scheduler {name} is not a positive integer"
                )
        if effective_requests != max_running_requests:
            raise RuntimeError(
                "SGLang effective request census differs from scheduler state"
            )
        runtime_proof = {
            "actual_attention_backend": backend_proof,
            "effective_scheduler": {
                "max_prefill_tokens": max_prefill_tokens,
                "max_running_requests": max_running_requests,
                "effective_max_running_requests_per_dp": effective_requests,
            },
        }
    config = _config()
    swa_activity = runtime.swa_activity(config.classes)
    runtime_binding = config.runtime_binding
    if not isinstance(runtime_binding, Mapping):
        raise RuntimeError("canonical runtime binding is unavailable")
    runtime_binding_fingerprint = runtime_binding.get("fingerprint")
    if not isinstance(runtime_binding_fingerprint, str):
        raise RuntimeError("canonical runtime binding fingerprint is unavailable")
    if not isinstance(config.runtime_manifest_fingerprint, str):
        raise RuntimeError("canonical runtime manifest fingerprint is unavailable")
    owner_proof = None
    if owner is not None:
        from .engine import get_owner

        allocator = _state._ALLOCATOR
        owner_proof = {
            "module": type(owner).__module__,
            "type": type(owner).__qualname__,
            "owner_is_process_singleton": get_owner() is owner,
            "config_is_canonical": _state._CONFIG is owner.config,
            "allocator_owned": (
                allocator is not None
                and getattr(allocator, "_orbitkv_lifecycle_owner", None) is owner
            ),
            "tree_cache_owned": (
                tree_cache is not None
                and getattr(tree_cache, "_orbitkv_lifecycle_owner", None) is owner
            ),
        }
        if not all(
            owner_proof[field]
            for field in (
                "owner_is_process_singleton",
                "config_is_canonical",
                "allocator_owned",
                "tree_cache_owned",
            )
        ):
            raise RuntimeError("direct-source lifecycle ownership is inconsistent")
    completion_evidence = runtime.completion_evidence(external=False)
    batch_counters = runtime.performance_counters()
    if not isinstance(batch_counters, dict) or set(batch_counters) != (
        _SESSION_COUNTER_KEYS
    ):
        raise RuntimeError(
            "native runtime-session performance counter schema changed"
        )
    swa_report = {
        "status": (
            "exposed" if swa_activity.applicable else "not_applicable"
        ),
        "applicable": swa_activity.applicable,
        "source": "native_runtime_session",
        "derived": False,
        "swa_retirement_certificates": swa_activity.retirement_certificates,
        "swa_pages_reclaimed": swa_activity.pages_reclaimed,
        "swa_wrap_events": swa_activity.wrap_events,
        "swa_page_reuse_events": swa_activity.page_reuse_events,
    }
    pressure = disabled_pressure_report()
    state["orbitkv_manager"] = {
        "lifecycle_route": "native_session",
        "cache_policy": cache_policy,
        "direct_source_owner": owner_proof,
        "completion_evidence": completion_evidence,
        "wire_version": WIRE_VERSION,
        "manager_input_fingerprint": config.plan_fingerprint,
        "runtime_manifest_fingerprint": config.runtime_manifest_fingerprint,
        "runtime_binding_fingerprint": runtime_binding_fingerprint,
        "fixed_state_byte_count": config.fixed_state_byte_count,
        "fixed_state_descriptors": _state._fixed_state_descriptors(),
        "tree_cache_type": _state._tree_cache_type(scheduler.tree_cache),
        "runtime_proof": runtime_proof,
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
                "request_page_refs": item.request_page_refs,
                "prefix_page_refs": item.prefix_page_refs,
                "reader_pins": item.reader_pins,
            }
            for item in arena_stats
        ],
        "manager_stats": {
            "active_requests": stats.active_requests,
            "active_snapshots": stats.active_snapshots,
            "active_prefixes": stats.active_prefixes,
            "evicted_prefixes": stats.evicted_prefixes,
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
            "total_request_page_refs": stats.total_request_page_refs,
            "total_prefix_page_refs": stats.total_prefix_page_refs,
            "total_reader_pins": stats.total_reader_pins,
        },
        "swa_activity": swa_report,
        "batch_counters": batch_counters,
        "pressure": pressure,
    }
    if _state._FIXED_STATE is not None:
        state["orbitkv_manager"]["fixed_state"] = _state._FIXED_STATE.census()
    return result


__all__ = ("augment_internal_state",)
