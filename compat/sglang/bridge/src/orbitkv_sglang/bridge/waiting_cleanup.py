from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ffi.session_types import (
    EnginePendingAttachCancel,
    EnginePendingAttachCancelDisposition,
    EnginePendingAttachCancelOutcome,
)
from ..runtime import FailStopped, RetryableConflict
from ..session_runtime import ReleaseRetryPending
from . import session_cache, state as _state
from .private_prefix import clear_request_identity
from .state import _request_key, _runtime


_PENDING_REMOVAL = "_orbitkv_pending_waiting_removal"


@dataclass(slots=True)
class _PendingAttachRemoval:
    key: Any
    request_id: Any
    pending: Any
    shared: Any
    node: Any
    empty_prefix: Any
    expected: EnginePendingAttachCancel
    native_canceled: bool = False
    local_cleaned: bool = False


@dataclass(frozen=True, slots=True)
class _PendingMissRemoval:
    key: Any
    request_id: Any


def _validate_outcome(
    expected: EnginePendingAttachCancel,
    outcome: Any,
    dispositions: tuple[EnginePendingAttachCancelDisposition, ...],
) -> None:
    if (
        type(outcome) is not EnginePendingAttachCancelOutcome
        or outcome.identity != expected
        or outcome.disposition not in dispositions
    ):
        raise RuntimeError(
            "committed waiting attach returned a hostile cancel outcome"
        )


def _new_pending_attach_removal(
    cache: Any, req: Any, pending: Any, shared: Any
) -> _PendingAttachRemoval:
    runtime = _runtime()
    key = _request_key(req)
    binding = runtime.binding_for(key)
    view = runtime.view_for(key)
    exact_shared = session_cache.pending_shared_prefix(cache, req)
    prefix = getattr(req, "prefix_indices", None)
    if (
        pending.req is not req
        or pending.key != key
        or pending.request_id != shared.request_id
        or exact_shared is not shared
        or binding.request_id != pending.request_id
        or binding.request_row is not None
        or view.request_id != pending.request_id
        or int(view.boundary) != 0
        or int(view.resident_count) != 0
        or getattr(req, "req_pool_idx", None) is not None
        or getattr(req, "kv", None) is not None
    ):
        raise RuntimeError(
            "committed waiting attach changed before cancellation"
        )
    node = cache._preflight_release_node(req, provisional=True)
    if node is not shared.node:
        raise RuntimeError(
            "committed waiting attach changed its provisional Prefix lock"
        )
    try:
        empty_prefix = prefix[:0]
    except BaseException as error:
        raise RuntimeError(
            "committed waiting attach has no sliceable Prefix mirror"
        ) from error
    materialized = shared.plan.requests[0]
    expected = EnginePendingAttachCancel(
        shared.control_id,
        pending.request_id,
        shared.prefix_id,
        int(materialized.view_version),
        int(materialized.boundary),
        int(materialized.resident_count),
    )
    return _PendingAttachRemoval(
        key, pending.request_id, pending, shared, node, empty_prefix, expected
    )


def _finish_pending_attach_removal(
    cache: Any, req: Any, work: _PendingAttachRemoval
) -> bool:
    runtime = _runtime()
    if not work.native_canceled:
        outcome = runtime.cancel_pending_attach(work.expected, work.key)
        _validate_outcome(
            work.expected,
            outcome,
            (
                EnginePendingAttachCancelDisposition.RECYCLE_PENDING,
                EnginePendingAttachCancelDisposition.FINALIZED,
            ),
        )
        work.native_canceled = True
    if not work.local_cleaned:
        pending = getattr(cache, "_session_pending_requests", None)
        shared = getattr(cache, "_session_pending_shared_prefix", None)
        active = getattr(cache, "_session_active_shared_prefix", None)
        if (
            not isinstance(pending, dict)
            or not isinstance(shared, dict)
            or not isinstance(active, dict)
            or pending.get(work.key) is not work.pending
            or shared.get(work.key) is not work.shared
            or work.key in active
        ):
            raise RuntimeError(
                "committed waiting attach changed before local cleanup"
            )
        installed_pending = dict(pending)
        installed_shared = dict(shared)
        del installed_pending[work.key]
        del installed_shared[work.key]
        cache._commit_release_node(req, work.node, provisional=True)
        cache._session_pending_requests = installed_pending
        cache._session_pending_shared_prefix = installed_shared
        req.prefix_indices = work.empty_prefix
        req.cache_protected_len = 0
        root = getattr(cache, "root_node", None)
        for name in ("last_node", "last_host_node", "best_match_node"):
            if hasattr(req, name):
                setattr(req, name, root)
        clear_request_identity(req)
        work.local_cleaned = True
    finalized = runtime.finalize_pending_attach_cancel(work.expected, work.key)
    _validate_outcome(
        work.expected,
        finalized,
        (EnginePendingAttachCancelDisposition.FINALIZED,),
    )
    delattr(req, _PENDING_REMOVAL)
    return True


def prepare_waiting_request_removal(req: Any, tree_cache: Any) -> bool:
    """Finish ownership before removal; return false while retry is pending."""

    if not _state._uses_runtime_session():
        from .lowering import _release_kv_cache

        _release_kv_cache(req, tree_cache, is_insert=False)
        return True

    work = getattr(req, _PENDING_REMOVAL, None)
    try:
        if type(work) is _PendingAttachRemoval:
            return _finish_pending_attach_removal(tree_cache, req, work)
        if type(work) is _PendingMissRemoval:
            session_cache.cancel_pending_request(tree_cache, req)
            delattr(req, _PENDING_REMOVAL)
            return True
        if work is not None:
            raise RuntimeError("waiting request retained malformed removal work")

        key = _request_key(req)
        pending = getattr(tree_cache, "_session_pending_requests", None)
        shared = getattr(tree_cache, "_session_pending_shared_prefix", None)
        if not isinstance(pending, dict) or not isinstance(shared, dict):
            raise RuntimeError(
                "waiting request removal requires a bound session cache"
            )
        pending_entry = pending.get(key)
        shared_entry = shared.get(key)
        if pending_entry is not None:
            if shared_entry is not None:
                work = _new_pending_attach_removal(
                    tree_cache, req, pending_entry, shared_entry
                )
                setattr(req, _PENDING_REMOVAL, work)
                return _finish_pending_attach_removal(tree_cache, req, work)
            setattr(
                req,
                _PENDING_REMOVAL,
                _PendingMissRemoval(key, pending_entry.request_id),
            )
            session_cache.cancel_pending_request(tree_cache, req)
            delattr(req, _PENDING_REMOVAL)
            return True
        if shared_entry is not None:
            raise RuntimeError(
                "waiting request has a committed attach without its "
                "pending request identity"
            )
        if (
            not hasattr(req, "_orbitkv_request_key")
            and not hasattr(req, "_orbitkv_engine_request_id")
            and getattr(req, "req_pool_idx", None) is None
            and getattr(req, "kv", None) is None
        ):
            return True
        from .lowering import _release_kv_cache

        _release_kv_cache(req, tree_cache, is_insert=False)
        return True
    except (ReleaseRetryPending, RetryableConflict):
        return False
    except BaseException as error:
        detail = (
            "waiting request removal became uncertain: "
            f"{type(error).__name__}: {error}"
        )
        session_cache.fail_stop_session_cache(tree_cache, detail)
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(_runtime().failure_reason or detail) from error


__all__ = ["prepare_waiting_request_removal"]
