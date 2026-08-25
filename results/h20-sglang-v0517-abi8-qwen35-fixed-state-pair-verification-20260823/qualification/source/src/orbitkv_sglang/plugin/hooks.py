from __future__ import annotations

from typing import Any, Callable

from ..config import load_config
from . import state as _state
from .facade import _build_token_to_kv_pool_allocator
from .fixed_state import (
    _attach_fixed_state_forward_batch,
    _execute_fixed_state_deferred,
    _fixed_state_alloc,
    _fixed_state_free,
    _fixed_state_pool_clear,
)
from .lowering import (
    _alloc_for_decode,
    _alloc_for_extend,
    _get_next_batch_to_run,
    _manager_maybe_evict_swa,
    _release_kv_cache,
    _run_batch,
)
from .prefix_cache import _build_prefix_cache
from .relocation import _active_forward_lengths
from .state import _config, _runtime
from .validation import (
    HOOK_TARGETS,
    _preflight_hook_targets,
    _validate_configurator,
    _validate_plugin_selection,
    _validate_propagated_aliases,
    _validate_sglang_revision,
)


def _get_internal_state(
    original_fn: Callable[..., Any], scheduler: Any, *args: Any, **kwargs: Any
) -> Any:
    result = original_fn(scheduler, *args, **kwargs)
    state = getattr(result, "internal_state", None)
    if not isinstance(state, dict) or "orbitkv_manager" in state:
        raise RuntimeError("SGLang returned an invalid internal-state namespace")
    runtime = _runtime()
    runtime.poll()
    if _state._FIXED_STATE is not None:
        _state._FIXED_STATE.poll()
    stats, arena_stats = runtime.census()
    swa_activity = runtime.swa_activity()
    if tuple(item.class_id for item in arena_stats) != tuple(
        item.class_id for item in runtime.arenas
    ):
        raise RuntimeError("manager internal-state arena order changed")
    state["orbitkv_manager"] = {
        "abi_version": 8,
        "plan_fingerprint": _config().plan_fingerprint,
        "state_plan_fingerprint": _config().state_plan_fingerprint,
        "fixed_state_byte_count": _config().fixed_state_byte_count,
        "fixed_state_descriptors": _state._fixed_state_descriptors(),
        "tree_cache_type": _state._tree_cache_type(scheduler.tree_cache),
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
        "swa_activity": {
            "status": "exposed",
            "applicable": _config().sliding_class is not None,
            "swa_retirement_certificates": swa_activity.retirement_certificates,
            "swa_pages_reclaimed": swa_activity.pages_reclaimed,
            "swa_wrap_events": swa_activity.wrap_events,
        },
        "batch_counters": {
            **runtime.performance_counters(),
            **_state._activity_counters(),
        },
    }
    if _state._FIXED_STATE is not None:
        state["orbitkv_manager"]["fixed_state"] = (
            _state._FIXED_STATE.census()
        )
    return result


def _register() -> None:
    if _state._CONFIG is not None:
        raise RuntimeError("OrbitKV plugin was registered more than once")
    _validate_plugin_selection()
    _validate_sglang_revision()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.mem_cache.registry import register_radix_cache_backend

    _preflight_hook_targets(HookRegistry)
    _state._CONFIG = load_config()
    register_radix_cache_backend("orbitkv", _build_prefix_cache)

    hooks = (
        (HOOK_TARGETS[0], _build_token_to_kv_pool_allocator, HookType.REPLACE),
        (HOOK_TARGETS[1], _alloc_for_extend, HookType.REPLACE),
        (HOOK_TARGETS[2], _alloc_for_decode, HookType.REPLACE),
        (HOOK_TARGETS[3], _manager_maybe_evict_swa, HookType.REPLACE),
        (HOOK_TARGETS[4], _release_kv_cache, HookType.REPLACE),
        (HOOK_TARGETS[5], _get_next_batch_to_run, HookType.AROUND),
        (HOOK_TARGETS[6], _run_batch, HookType.AROUND),
        (HOOK_TARGETS[7], _validate_configurator, HookType.AROUND),
        (HOOK_TARGETS[8], _get_internal_state, HookType.AROUND),
        (HOOK_TARGETS[9], _active_forward_lengths, HookType.AFTER),
        (HOOK_TARGETS[9], _attach_fixed_state_forward_batch, HookType.AFTER),
        (HOOK_TARGETS[10], _fixed_state_alloc, HookType.AROUND),
        (HOOK_TARGETS[11], _execute_fixed_state_deferred, HookType.AROUND),
        (HOOK_TARGETS[12], _fixed_state_pool_clear, HookType.AROUND),
        (HOOK_TARGETS[13], _fixed_state_free, HookType.AROUND),
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
        raise SystemExit(
            f"OrbitKV canonical adapter activation failed: {error}"
        ) from error
