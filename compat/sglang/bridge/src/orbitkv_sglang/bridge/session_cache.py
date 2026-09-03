from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import torch

from ..ffi.session_types import (
    EngineControlDisposition,
    EngineControlId,
    EngineControlOutcome,
    EngineMaterializationPlan,
    EngineMaterializedRequest,
    EnginePrefixId,
)
from ..runtime import FailStopped
from ..session_runtime import (
    ReleaseRetryPending,
    SessionMaterializationUpdate,
    SessionMirrorUpdate,
    collective_mirror_cleanup,
)
from . import state as _state
from .private_prefix import (
    ENGINE_REQUEST_ID_MARKER,
    ENGINE_PREFIX_ID_MARKER,
    PRIVATE_PREFIX_MARKER,
    clear_request_identity,
    request_identity,
    update_private_prefix,
    validate_private_prefix,
    validate_shared_prefix_metadata,
)
from .state import _request_key, _runtime
from .request_rows import validate_private_resident_count


@dataclass(frozen=True, slots=True)
class _RequestEntry:
    req: Any
    key: tuple[str, str | bytes | int]
    request_id: Any
    row: int


@dataclass(frozen=True, slots=True)
class _PendingRequestEntry:
    req: Any
    key: tuple[str, str | bytes | int]
    request_id: Any


@dataclass(frozen=True, slots=True)
class _PendingSharedPrefixEntry:
    req: Any
    key: tuple[str, str | bytes | int]
    request_id: Any
    prefix_id: EnginePrefixId
    semantic: Any
    node: Any
    boundary: int
    control_id: EngineControlId | None
    plan: EngineMaterializationPlan | None
    swa_evicted_seqlen: int


@dataclass(frozen=True, slots=True)
class _PreparedMaterializationMirror:
    entry: _PendingSharedPrefixEntry
    update: SessionMaterializationUpdate
    row: Any
    indices: Any


@dataclass(frozen=True, slots=True)
class ReleaseCandidate:
    req: Any
    tree_cache: Any
    req_to_token_pool: Any
    key: tuple[str, str | bytes | int]
    request_id: Any
    row: int
    boundary: int
    prefix_indices: Any
    empty_prefix_indices: Any
    prefix_node: Any
    is_insert: bool


class SessionCacheMixin:
    """Narrow methods consumed by session lowering and release dispatch."""

    def _register_pending_session_request(
        self, req: Any
    ) -> _PendingRequestEntry:
        return register_pending_request(self, req)

    def _promote_pending_session_requests(
        self, requests: Sequence[Any]
    ) -> tuple[_RequestEntry, ...]:
        return promote_pending_requests(self, requests)

    def _cancel_pending_session_request(self, req: Any) -> None:
        cancel_pending_request(self, req)

    def _unregister_session_request(self, req: Any) -> None:
        unregister_request(self, req)

    def _session_cleanup_context(self, update: Any) -> Any:
        return cleanup_context(self, update)

    def _session_materialization_callback(self, updates: Any) -> bool:
        return _materialization_callback(self, updates)


def bind_session_cache(cache: Any) -> None:
    """Create the context registry and bind the sole cleanup callback."""

    from .mirror_cleanup import _mirror_cleanup_coordinator

    if not _state._uses_runtime_session():
        raise RuntimeError(
            "session cache binding requires the runtime-session profile"
        )
    if hasattr(cache, "_session_requests") or hasattr(
        cache, "_session_pending_requests"
    ):
        raise RuntimeError("session cache context registry was bound twice")
    try:
        cache._session_requests = {}
        cache._session_pending_requests = {}
        cache._session_pending_shared_prefix = {}
        cache._session_active_shared_prefix = {}
        coordinator = _mirror_cleanup_coordinator(
            cache.req_to_token_pool, cache.token_to_kv_pool_allocator
        )
        _state._bind_session_cleanup(
            collective_mirror_cleanup(coordinator, cache._session_cleanup_context)
        )
        _runtime().bind_materialization(cache._session_materialization_callback)
    except BaseException:
        for name in (
            "_session_requests",
            "_session_pending_requests",
            "_session_pending_shared_prefix",
            "_session_active_shared_prefix",
        ):
            registry = getattr(cache, name, None)
            if isinstance(registry, dict):
                registry.clear()
            if hasattr(cache, name):
                delattr(cache, name)
        raise


def clear_session_cache(cache: Any) -> None:
    for name in (
        "_session_requests",
        "_session_pending_requests",
        "_session_pending_shared_prefix",
        "_session_active_shared_prefix",
    ):
        registry = getattr(cache, name, None)
        if registry is not None:
            registry.clear()


def fail_stop_session_cache(cache: Any, reason: str) -> None:
    """Poison the session runtime and discard Python request references."""

    runtime = _runtime()
    try:
        runtime.fail_stop(reason)
    finally:
        clear_session_cache(cache)


def _registry(cache: Any) -> dict[Any, _RequestEntry]:
    registry = getattr(cache, "_session_requests", None)
    if not isinstance(registry, dict):
        raise RuntimeError("session cache context registry is not bound")
    return registry


def _pending_registry(cache: Any) -> dict[Any, _PendingRequestEntry]:
    registry = getattr(cache, "_session_pending_requests", None)
    if not isinstance(registry, dict):
        raise RuntimeError("session cache pending registry is not bound")
    return registry


def _pending_shared_prefix_registry(
    cache: Any,
) -> dict[Any, _PendingSharedPrefixEntry]:
    registry = getattr(cache, "_session_pending_shared_prefix", None)
    if not isinstance(registry, dict):
        raise RuntimeError("session shared-prefix pending registry is not bound")
    return registry


def _active_shared_prefix_registry(cache: Any) -> dict[Any, _PendingSharedPrefixEntry]:
    registry = getattr(cache, "_session_active_shared_prefix", None)
    if not isinstance(registry, dict):
        raise RuntimeError("session shared-prefix active registry is not bound")
    return registry


def register_pending_shared_prefix(
    cache: Any,
    *,
    req: Any,
    prefix_id: EnginePrefixId,
    semantic: Any,
    node: Any,
    boundary: int,
    control_id: EngineControlId,
    plan: EngineMaterializationPlan,
    swa_evicted_seqlen: int = 0,
) -> _PendingSharedPrefixEntry:
    key = _request_key(req)
    request_id = getattr(req, ENGINE_REQUEST_ID_MARKER, None)
    if (
        type(control_id) is not EngineControlId
        or isinstance(swa_evicted_seqlen, bool)
        or not isinstance(swa_evicted_seqlen, Integral)
        or not 0 <= int(swa_evicted_seqlen) <= int(boundary)
    ):
        raise RuntimeError("shared Prefix pending entry requires one control id")
    entry = _PendingSharedPrefixEntry(
        req,
        key,
        request_id,
        prefix_id,
        semantic,
        node,
        int(boundary),
        control_id,
        plan,
        int(swa_evicted_seqlen),
    )
    _planned_shared_prefix(entry)
    pending = _pending_shared_prefix_registry(cache)
    active = _active_shared_prefix_registry(cache)
    if key in pending or key in active:
        raise RuntimeError("session shared Prefix was registered twice")
    pending[key] = entry
    return entry


def _planned_shared_prefix(
    entry: _PendingSharedPrefixEntry,
) -> EngineMaterializedRequest:
    plan = entry.plan
    if (
        entry.control_id is None
        or type(plan) is not EngineMaterializationPlan
        or plan.control_id != entry.control_id
        or len(plan.requests) != 1
    ):
        raise RuntimeError("shared Prefix pending entry has no exact attach plan")
    materialized = plan.requests[0]
    if (
        type(materialized) is not EngineMaterializedRequest
        or materialized.request_id != entry.request_id
        or int(materialized.boundary) != entry.boundary
        or int(materialized.resident_count) < 0
        or int(materialized.resident_count)
        != getattr(entry.node, "resident_count", None)
        or len(materialized.pages) != int(materialized.resident_count)
        or isinstance(entry.swa_evicted_seqlen, bool)
        or not isinstance(entry.swa_evicted_seqlen, Integral)
        or not 0 <= int(entry.swa_evicted_seqlen) <= entry.boundary
    ):
        raise RuntimeError("shared Prefix pending attach plan changed identity")
    return materialized


def pending_shared_prefix(
    cache: Any, req: Any
) -> _PendingSharedPrefixEntry | None:
    """Return and validate the exact pending attach for one request."""

    key = _request_key(req)
    entry = _pending_shared_prefix_registry(cache).get(key)
    if entry is None:
        return None
    if (
        entry.req is not req
        or entry.key != key
        or getattr(req, ENGINE_REQUEST_ID_MARKER, None) != entry.request_id
        or getattr(req, ENGINE_PREFIX_ID_MARKER, None) != entry.prefix_id
    ):
        raise RuntimeError("shared Prefix pending registry changed identity")
    _planned_shared_prefix(entry)
    return entry


def register_active_shared_prefix(
    cache: Any,
    *,
    req: Any,
    prefix_id: EnginePrefixId,
    semantic: Any,
    node: Any,
    boundary: int,
) -> _PendingSharedPrefixEntry:
    key = _request_key(req)
    request_id = getattr(req, ENGINE_REQUEST_ID_MARKER, None)
    entry = _PendingSharedPrefixEntry(
        req, key, request_id, prefix_id, semantic, node, int(boundary), None, None,
        int(getattr(getattr(req, "kv", None), "swa_evicted_seqlen", 0)),
    )
    pending = _pending_shared_prefix_registry(cache)
    active = _active_shared_prefix_registry(cache)
    if key in pending or key in active:
        raise RuntimeError("session shared Prefix was registered twice")
    installed = dict(active)
    installed[key] = entry
    cache._session_active_shared_prefix = installed
    return entry


def _validate_confirmed_shared_prefix(
    cache: Any, entry: _PendingSharedPrefixEntry
) -> None:
    """Validate native confirmation before changing local ownership state."""

    pending_request = _pending_registry(cache).get(entry.key)
    runtime = _runtime()
    binding = runtime.binding_for(entry.key)
    view = runtime.view_for(entry.key)
    planned = _planned_shared_prefix(entry)
    raw_row = getattr(entry.req, "req_pool_idx", None)
    if isinstance(raw_row, bool) or not isinstance(raw_row, Integral):
        raise RuntimeError("shared Prefix confirmed request has no exact row")
    row_index = int(raw_row)
    row = cache.req_to_token_pool.req_to_token[row_index]
    prefix = getattr(entry.req, "prefix_indices", None)
    if (
        pending_request is None
        or pending_request.req is not entry.req
        or pending_request.key != entry.key
        or pending_request.request_id != entry.request_id
        or row_index <= 0
        or binding.request_id != entry.request_id
        or binding.request_row != row_index
        or view.request_id != entry.request_id
        or int(view.view_version) != int(planned.view_version)
        or int(view.boundary) != entry.boundary
        or int(view.resident_count) != int(planned.resident_count)
        or type(prefix) is not torch.Tensor
        or int(prefix.numel()) != entry.boundary
        or not torch.equal(
            prefix, row[: entry.boundary].to(dtype=prefix.dtype)
        )
    ):
        raise RuntimeError("shared Prefix confirmed view or mirror changed identity")
    validate_shared_prefix_metadata(
        entry.req,
        key=entry.key,
        request_id=entry.request_id,
        prefix_id=entry.prefix_id,
        boundary=entry.boundary,
        node=entry.node,
        semantic=entry.semantic,
        provisional=True,
    )
    lock_ref = getattr(entry.node, "lock_ref", None)
    if (
        isinstance(lock_ref, bool)
        or not isinstance(lock_ref, Integral)
        or int(lock_ref) < 2
    ):
        raise RuntimeError(
            "shared Prefix activation has no waiting and run locks"
        )


def _activate_confirmed_shared_prefix(
    cache: Any, entry: _PendingSharedPrefixEntry
) -> _PendingSharedPrefixEntry:
    """Activate one confirmed attach and drop exactly its waiting lock."""

    pending = _pending_shared_prefix_registry(cache)
    active = _active_shared_prefix_registry(cache)
    if pending.get(entry.key) is not entry or entry.key in active:
        raise RuntimeError("shared Prefix registry changed before activation")
    installed_pending = dict(pending)
    installed_active = dict(active)
    del installed_pending[entry.key]
    installed_active[entry.key] = entry
    cache.dec_lock_ref(entry.node)
    entry.req._orbitkv_provisional_prefix_lock = False
    entry.req._orbitkv_prefix_lock_held = True
    cache._session_pending_shared_prefix = installed_pending
    cache._session_active_shared_prefix = installed_active
    validate_shared_prefix_metadata(
        entry.req,
        key=entry.key,
        request_id=entry.request_id,
        prefix_id=entry.prefix_id,
        boundary=entry.boundary,
        node=entry.node,
        semantic=entry.semantic,
        provisional=False,
    )
    return entry


def confirm_pending_shared_prefix(
    cache: Any, req: Any
) -> _PendingSharedPrefixEntry:
    key = _request_key(req)
    pending = _pending_shared_prefix_registry(cache)
    entry = pending.get(key)
    if entry is None or entry.req is not req:
        raise RuntimeError("shared Prefix request has no pending materialization")
    if entry.control_id is None or type(entry.plan) is not EngineMaterializationPlan:
        raise RuntimeError("shared Prefix request has no pending attach control")
    runtime = _runtime()
    binding = runtime.binding_for(key)
    pending_request = _pending_registry(cache).get(key)
    raw_row = getattr(req, "req_pool_idx", None)
    if (
        pending_request is None
        or pending_request.req is not req
        or pending_request.key != entry.key
        or pending_request.request_id != entry.request_id
        or isinstance(raw_row, bool)
        or not isinstance(raw_row, Integral)
        or int(raw_row) <= 0
        or binding.request_id != entry.request_id
        or binding.request_row != int(raw_row)
    ):
        raise RuntimeError("shared Prefix request is not row bound for confirmation")
    validate_shared_prefix_metadata(
        req,
        key=entry.key,
        request_id=entry.request_id,
        prefix_id=entry.prefix_id,
        boundary=entry.boundary,
        node=entry.node,
        semantic=entry.semantic,
        provisional=True,
    )
    outcome = runtime.confirm_control(entry.control_id)
    if (
        type(outcome) is not EngineControlOutcome
        or outcome.control_id != entry.control_id
        or outcome.disposition is not EngineControlDisposition.MATERIALIZED
    ):
        raise RuntimeError("shared Prefix confirmation returned a hostile outcome")
    _validate_confirmed_shared_prefix(cache, entry)
    return _activate_confirmed_shared_prefix(cache, entry)


def clear_shared_prefix(cache: Any, req: Any) -> None:
    key = _request_key(req)
    pending = _pending_shared_prefix_registry(cache)
    active = _active_shared_prefix_registry(cache)
    if key in pending:
        installed = dict(pending)
        del installed[key]
        cache._session_pending_shared_prefix = installed
    if key in active:
        installed = dict(active)
        del installed[key]
        cache._session_active_shared_prefix = installed


def require_shared_prefix(
    cache: Any, req: Any, *, allow_pending: bool
) -> _PendingSharedPrefixEntry:
    key = _request_key(req)
    entry = _active_shared_prefix_registry(cache).get(key)
    if entry is None and allow_pending:
        entry = _pending_shared_prefix_registry(cache).get(key)
    if (
        entry is None
        or entry.req is not req
        or getattr(req, ENGINE_PREFIX_ID_MARKER, None) != entry.prefix_id
        or getattr(req, ENGINE_REQUEST_ID_MARKER, None) != entry.request_id
    ):
        raise RuntimeError("shared Prefix request has no exact session registry")
    validate_shared_prefix_metadata(
        req,
        key=entry.key,
        request_id=entry.request_id,
        prefix_id=entry.prefix_id,
        boundary=entry.boundary,
        node=entry.node,
        semantic=entry.semantic,
        provisional=entry.key in _pending_shared_prefix_registry(cache),
    )
    return entry


def _materialization_callback(cache: Any, updates: Any) -> bool:
    values = tuple(updates)
    if not values:
        return True
    table = cache.req_to_token_pool.req_to_token
    pending = _pending_shared_prefix_registry(cache)
    runtime = _runtime()
    prepared: list[_PreparedMaterializationMirror] = []
    seen_keys: set[Any] = set()
    seen_rows: set[int] = set()
    for item in values:
        if type(item) is not SessionMaterializationUpdate:
            raise RuntimeError("session materialization supplied an invalid update")
        entry = pending.get(item.key)
        if entry is None:
            raise RuntimeError("session materialization has no pending shared Prefix")
        planned = _planned_shared_prefix(entry)
        if (
            item.key in seen_keys
            or int(item.request_row) in seen_rows
            or entry.request_id != item.request_id
            or entry.boundary != int(item.boundary)
            or int(item.view_version) != int(planned.view_version)
            or int(item.resident_count) != int(planned.resident_count)
            or tuple(item.pages) != tuple(planned.pages)
            or int(item.request_row) <= 0
            or int(item.request_row) >= int(table.shape[0])
            or runtime.binding_for(item.key).request_row != int(item.request_row)
        ):
            raise RuntimeError("session materialization changed shared Prefix identity")
        row = table[int(item.request_row)]
        indices = cache._materialize_prefix_pages(
            item.pages, int(item.boundary), int(item.resident_count)
        )
        validate_shared_prefix_metadata(
            entry.req,
            key=entry.key,
            request_id=entry.request_id,
            prefix_id=entry.prefix_id,
            node=entry.node,
            semantic=entry.semantic,
            boundary=int(item.boundary),
            provisional=True,
        )
        mirror = getattr(entry.req, "prefix_indices", None)
        if (
            type(indices) is not torch.Tensor
            or indices.ndim != 1
            or int(indices.numel()) != int(item.boundary)
            or type(mirror) is not torch.Tensor
            or int(mirror.numel()) != int(item.boundary)
            or not torch.equal(mirror, indices.to(dtype=mirror.dtype))
        ):
            raise RuntimeError(
                "session materialization differs from the provisional prefix mirror"
            )
        prepared.append(_PreparedMaterializationMirror(entry, item, row, indices))
        seen_keys.add(item.key)
        seen_rows.add(int(item.request_row))

    # Every update is now validated.  Only ReqToToken is written here; the
    # provisional markers and pending registry remain authoritative until the
    # native confirmation succeeds and installs the public confirmed view.
    for value in prepared:
        boundary = int(value.update.boundary)
        value.row[:boundary] = value.indices.to(dtype=value.row.dtype)
    for value in prepared:
        mirror = getattr(value.entry.req, "prefix_indices", None)
        boundary = int(value.update.boundary)
        if not torch.equal(
            mirror, value.row[:boundary].to(dtype=mirror.dtype)
        ):
            raise RuntimeError("session materialization did not install an exact prefix mirror")
    return True


def _request_boundary(req: Any, *, allow_empty_kv: bool) -> int:
    kv = getattr(req, "kv", None)
    if kv is None and allow_empty_kv:
        return 0
    boundary = getattr(kv, "kv_allocated_len", None)
    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, Integral)
        or int(boundary) < 0
    ):
        raise RuntimeError(
            "SGLang request KV boundary is not a nonnegative integer"
        )
    return int(boundary)


def preflight_register_requests(
    cache: Any, assignments: Sequence[tuple[Any, Any, int]]
) -> None:
    """Validate a new request batch without changing engine or cache state."""

    if not _state._uses_runtime_session():
        raise RuntimeError(
            "session request contexts require the runtime-session profile"
        )
    registry = _registry(cache)
    pending = _pending_registry(cache)
    table = cache.req_to_token_pool.req_to_token
    values = tuple(assignments)
    keys: list[Any] = []
    requests: list[int] = []
    rows: list[int] = []
    for item in values:
        if not isinstance(item, tuple) or len(item) != 3:
            raise RuntimeError(
                "session context assignments must be (request, key, row) triples"
            )
        req, key, row = item
        pending_request_id = getattr(req, ENGINE_REQUEST_ID_MARKER, None)
        if (
            _request_key(req) != key
            or isinstance(row, bool)
            or not isinstance(row, Integral)
            or not 0 < int(row) < int(table.shape[0])
            or key in registry
            or key in pending
            or any(entry.req is req for entry in registry.values())
            or any(entry.req is req for entry in pending.values())
            or any(
                pending_request_id is not None
                and entry.request_id == pending_request_id
                for entry in pending.values()
            )
            or any(entry.row == int(row) for entry in registry.values())
            or hasattr(req, "_orbitkv_request_key")
            or hasattr(req, ENGINE_REQUEST_ID_MARKER)
            or hasattr(req, "_orbitkv_request_lease")
            or not hasattr(req, "__dict__")
        ):
            raise RuntimeError(
                "new session request context is invalid or already registered"
            )
        keys.append(key)
        requests.append(id(req))
        rows.append(int(row))
    if (
        len(set(keys)) != len(values)
        or len(set(requests)) != len(values)
        or len(set(rows)) != len(values)
    ):
        raise RuntimeError("new session request contexts contain aliases")


def register_pending_request(cache: Any, req: Any) -> _PendingRequestEntry:
    """Register one acquired request that does not own a ReqToToken row yet."""

    if not _state._uses_runtime_session():
        raise RuntimeError(
            "session request contexts require the runtime-session profile"
        )
    key = _request_key(req)
    request_id = getattr(req, ENGINE_REQUEST_ID_MARKER, None)
    if (
        getattr(req, "_orbitkv_request_key", None) != key
        or request_id is None
        or isinstance(request_id, bool)
        or not isinstance(request_id, int)
        or request_id <= 0
        or getattr(req, "req_pool_idx", None) is not None
        or getattr(req, "kv", None) is not None
        or hasattr(req, "_orbitkv_request_lease")
        or not hasattr(req, "__dict__")
    ):
        raise RuntimeError(
            "pending session request has an invalid unbound identity"
        )
    runtime = _runtime()
    binding = runtime.binding_for(key)
    view = runtime.view_for(key)
    if (
        binding.request_id != request_id
        or binding.request_row is not None
        or view.request_id != request_id
        or int(view.boundary) != 0
        or int(view.resident_count) != 0
    ):
        raise RuntimeError(
            "pending session request differs from runtime authority"
        )
    registry = _registry(cache)
    pending = _pending_registry(cache)
    current = pending.get(key)
    if current is not None:
        if current.req is not req or current.request_id != request_id:
            raise RuntimeError(
                "pending session request registry changed identity"
            )
        raise RuntimeError("pending session request was registered twice")
    if (
        key in registry
        or any(
            entry.req is req or entry.request_id == request_id
            for entry in registry.values()
        )
        or any(
            entry.req is req or entry.request_id == request_id
            for entry in pending.values()
        )
    ):
        raise RuntimeError(
            "pending session request aliases an existing context"
        )
    entry = _PendingRequestEntry(req, key, request_id)
    try:
        installed = dict(pending)
        installed[key] = entry
        cache._session_pending_requests = installed
    except BaseException as error:
        fail_stop_session_cache(
            cache,
            "pending session request local installation became uncertain: "
            f"{type(error).__name__}: {error}",
        )
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            _runtime().failure_reason
            or "pending request local installation failed"
        ) from error
    return entry


def _preflight_pending_request_promotion(
    cache: Any, req: Any
) -> tuple[_PendingRequestEntry, _RequestEntry]:
    """Validate one row-bound pending identity without changing registries."""

    if not _state._uses_runtime_session():
        raise RuntimeError(
            "session request contexts require the runtime-session profile"
        )
    key = _request_key(req)
    pending = _pending_registry(cache)
    entry = pending.get(key)
    request_id = getattr(req, ENGINE_REQUEST_ID_MARKER, None)
    raw_row = getattr(req, "req_pool_idx", None)
    table = cache.req_to_token_pool.req_to_token
    binding = _runtime().binding_for(key)
    if (
        entry is None
        or entry.req is not req
        or entry.key != key
        or entry.request_id != request_id
        or getattr(req, "_orbitkv_request_key", None) != key
        or hasattr(req, "_orbitkv_request_lease")
        or getattr(req, "kv", None) is not None
        or isinstance(raw_row, bool)
        or not isinstance(raw_row, Integral)
        or not 0 < int(raw_row) < int(table.shape[0])
        or binding.request_id != entry.request_id
        or binding.request_row != int(raw_row)
        or any(
            current is not entry
            and (current.req is req or current.request_id == entry.request_id)
            for current in pending.values()
        )
    ):
        raise RuntimeError(
            "pending session request cannot be promoted from its current identity"
        )
    row = int(raw_row)
    registry = _registry(cache)
    if (
        key in registry
        or any(
            current.req is req
            or current.request_id == entry.request_id
            or current.row == row
            for current in registry.values()
        )
    ):
        raise RuntimeError(
            "pending session request promotion aliases an existing context"
        )
    return entry, _RequestEntry(req, key, entry.request_id, row)


def promote_pending_requests(
    cache: Any, requests: Sequence[Any]
) -> tuple[_RequestEntry, ...]:
    """Collectively promote row-bound identities after attach confirmation."""

    values = tuple(requests)
    if not values or len({id(req) for req in values}) != len(values):
        raise RuntimeError("pending session promotion requires unique requests")
    prepared = tuple(
        _preflight_pending_request_promotion(cache, req) for req in values
    )
    active = _active_shared_prefix_registry(cache)
    shared = tuple(active.get(entry.key) for entry, _promoted in prepared)
    if any(
        item is not None
        and (item.req is not entry.req or item.request_id != entry.request_id)
        for (entry, _promoted), item in zip(prepared, shared, strict=True)
    ):
        raise RuntimeError("confirmed shared Prefix changed before promotion")
    pending = _pending_registry(cache)
    registry = _registry(cache)
    try:
        installed_registry = dict(registry)
        installed_pending = dict(pending)
        for entry, promoted in prepared:
            if installed_pending.get(entry.key) is not entry:
                raise RuntimeError(
                    "pending session request changed during promotion"
                )
            installed_registry[entry.key] = promoted
            del installed_pending[entry.key]
        cache._session_requests = installed_registry
        cache._session_pending_requests = installed_pending
        if any(item is not None for item in shared):
            from sglang.srt.managers.schedule_batch import ReqKvInfo

            for req, item in zip(values, shared, strict=True):
                if item is not None:
                    req.kv = ReqKvInfo(
                        kv_allocated_len=item.boundary,
                        swa_evicted_seqlen=item.swa_evicted_seqlen,
                    )
    except BaseException as error:
        fail_stop_session_cache(
            cache,
            "collective pending session request promotion became uncertain: "
            f"{type(error).__name__}: {error}",
        )
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            _runtime().failure_reason or "pending request promotion failed"
        ) from error
    return tuple(promoted for _entry, promoted in prepared)


def cancel_pending_request(cache: Any, req: Any) -> None:
    """Release one never-row-bound request without touching ReqToToken."""

    runtime = _runtime()
    key = _request_key(req)
    pending = _pending_registry(cache)
    entry = pending.get(key)
    binding = runtime.binding_for(key)
    view = runtime.view_for(key)
    if (
        entry is None
        or entry.req is not req
        or entry.key != key
        or getattr(req, "_orbitkv_request_key", None) != key
        or getattr(req, ENGINE_REQUEST_ID_MARKER, None) != entry.request_id
        or getattr(req, "req_pool_idx", None) is not None
        or getattr(req, "kv", None) is not None
        or binding.request_id != entry.request_id
        or binding.request_row is not None
        or view.request_id != entry.request_id
        or int(view.boundary) != 0
        or int(view.resident_count) != 0
    ):
        raise RuntimeError(
            "pending session request changed before cancellation"
        )
    plan = runtime.prepare_release((key,))
    if (
        len(plan.releases) != 1
        or plan.releases[0].request_id != entry.request_id
        or tuple(plan.releases[0].detached)
        or tuple(plan.retirements)
    ):
        fail_stop_session_cache(
            cache,
            "pending session request release unexpectedly owns resident state",
        )
        raise FailStopped(
            runtime.failure_reason or "invalid pending request release"
        )
    try:
        runtime.confirm_release(plan)
        if pending.get(key) is not entry:
            raise RuntimeError(
                "pending session request changed during cancellation"
            )
        clear_shared_prefix(cache, req)
        del pending[key]
        clear_request_identity(req)
    except ReleaseRetryPending:
        raise
    except BaseException as error:
        fail_stop_session_cache(
            cache,
            "pending session request cancellation became uncertain: "
            f"{type(error).__name__}: {error}",
        )
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            runtime.failure_reason or "pending request cancellation failed"
        ) from error


def require_request(cache: Any, req: Any) -> _RequestEntry:
    key = _request_key(req)
    entry = _registry(cache).get(key)
    if (
        entry is None
        or entry.req is not req
        or getattr(req, "_orbitkv_request_key", None) != key
        or getattr(req, ENGINE_REQUEST_ID_MARKER, None) != entry.request_id
        or getattr(req, "req_pool_idx", None) != entry.row
    ):
        raise RuntimeError(
            "session request context changed or was not registered"
        )
    return entry


def unregister_request(cache: Any, req: Any) -> None:
    entry = require_request(cache, req)
    del _registry(cache)[entry.key]


def cleanup_context(cache: Any, update: Any) -> Any:
    from .mirror_cleanup import _MirrorCleanupContext

    if not isinstance(update, SessionMirrorUpdate):
        raise RuntimeError("session cleanup supplied an invalid update")
    entry = _registry(cache).get(update.key)
    binding = _runtime().binding_for(update.key)
    if (
        entry is None
        or entry.request_id != update.request_id
        or entry.row != update.request_row
        or binding.request_id != update.request_id
        or binding.request_row != update.request_row
        or _request_boundary(entry.req, allow_empty_kv=True)
        != int(update.boundary)
    ):
        raise RuntimeError(
            "session cleanup update has no exact request context"
        )
    require_request(cache, entry.req)
    return _MirrorCleanupContext(
        entry.req, entry.row, entry.key, entry.request_id
    )


def cache_unfinished_request(cache: Any, req: Any) -> None:
    runtime = _runtime()
    key = _request_key(req)
    entry = require_request(cache, req)
    runtime.wait_requests((key,))
    binding = runtime.binding_for(key)
    view = runtime.view_for(key)
    boundary = int(view.boundary)
    validate_private_resident_count(_state._config(), view, boundary)
    if (
        binding.request_id != entry.request_id
        or binding.request_row != entry.row
        or getattr(req, ENGINE_REQUEST_ID_MARKER, None) != entry.request_id
        or getattr(req, "_orbitkv_request_key", None) != key
        or _request_boundary(req, allow_empty_kv=False) != boundary
    ):
        raise RuntimeError("unfinished session request identity changed")
    update_private_prefix(
        req,
        cache.req_to_token_pool.req_to_token[entry.row],
        key,
        entry.request_id,
        boundary,
    )


def preflight_release_node(cache: Any, req: Any) -> None:
    key = _request_key(req)
    entry = require_request(cache, req)
    runtime = _runtime()
    binding = runtime.binding_for(key)
    view = runtime.view_for(key)
    identity = getattr(req, ENGINE_REQUEST_ID_MARKER, None)
    boundary = int(view.boundary)
    if (
        binding.request_id != entry.request_id
        or binding.request_row != entry.row
        or identity != entry.request_id
        or _request_boundary(req, allow_empty_kv=True) != boundary
    ):
        raise RuntimeError("request-private session identity changed")
    validate_private_prefix(
        req,
        cache.req_to_token_pool.req_to_token[entry.row],
        key,
        identity,
        boundary,
    )


def commit_release_node(
    _cache: Any, req: Any, node: Any, *, provisional: bool
) -> None:
    prefix = getattr(req, "prefix_indices", None)
    if (
        node is not None
        or provisional
        or getattr(req, PRIVATE_PREFIX_MARKER, None) is not None
        or prefix is None
        or int(prefix.numel()) != 0
    ):
        raise RuntimeError(
            "session release cleanup did not commit its private prefix"
        )


def release_candidate(
    req: Any, tree_cache: Any, *, is_insert: bool
) -> ReleaseCandidate | None:
    """Capture one strict runtime-session release candidate without waiting."""

    from .prefix_cache import OrbitKvPrefixCache

    if not _state._uses_runtime_session():
        raise RuntimeError(
            "session release candidate requires a runtime-session profile"
        )
    if type(tree_cache) is not OrbitKvPrefixCache:
        raise RuntimeError(
            "session release requires the OrbitKV cache facade"
        )
    if tree_cache.token_to_kv_pool_allocator is not _state._ALLOCATOR:
        raise RuntimeError("session release requires the OrbitKV allocator facade")
    if tree_cache._no_prefix and not tree_cache.disable_finished_insert:
        raise RuntimeError("session release cache is not request-private")
    key = _request_key(req)
    entry = _registry(tree_cache).get(key)
    pending_entry = _pending_registry(tree_cache).get(key)
    pending_shared = _pending_shared_prefix_registry(tree_cache).get(key)
    active_shared = _active_shared_prefix_registry(tree_cache).get(key)
    has_metadata = (
        hasattr(req, "_orbitkv_request_key")
        or hasattr(req, ENGINE_REQUEST_ID_MARKER)
        or hasattr(req, ENGINE_PREFIX_ID_MARKER)
        or entry is not None
        or pending_entry is not None
        or pending_shared is not None
        or active_shared is not None
    )
    if not has_metadata:
        if getattr(req, "req_pool_idx", None) is None and getattr(
            req, "kv", None
        ) is None:
            return None
        raise RuntimeError(
            "unregistered session request retained engine KV state"
        )
    if pending_entry is not None:
        if (
            pending_entry.req is not req
            or getattr(req, "_orbitkv_request_key", None) != key
            or getattr(req, ENGINE_REQUEST_ID_MARKER, None)
            != pending_entry.request_id
        ):
            raise RuntimeError(
                "pending session release request changed identity"
            )
        raise RuntimeError(
            "pending session request must use the unbound cancel path"
        )
    entry = require_request(tree_cache, req)
    runtime = _runtime()
    binding = runtime.binding_for(key)
    if (
        binding.request_id != entry.request_id
        or binding.request_row != entry.row
        or request_identity(req) != entry.request_id
    ):
        raise RuntimeError(
            "session release request differs from runtime authority"
        )
    boundary = _request_boundary(req, allow_empty_kv=True)
    view = runtime.view_for(key)
    if int(view.boundary) > boundary:
        raise RuntimeError(
            "session release boundary precedes its confirmed request view"
        )
    prefix_indices = getattr(req, "prefix_indices", None)
    try:
        empty_prefix_indices = prefix_indices[:0]
    except Exception as error:
        raise RuntimeError(
            "SGLang release prefix is not sliceable"
        ) from error
    prefix_node = None
    if tree_cache._no_prefix:
        validate_private_prefix(
            req,
            tree_cache.req_to_token_pool.req_to_token[entry.row],
            key,
            entry.request_id,
            boundary,
        )
    else:
        prefix_node = tree_cache._preflight_release_node(req, provisional=False)
        if prefix_node is not None:
            shared_entry = require_shared_prefix(tree_cache, req, allow_pending=False)
            if (
                shared_entry.req is not req
                or shared_entry.key != key
                or shared_entry.request_id != entry.request_id
                or shared_entry.node is not prefix_node
                or shared_entry.boundary != getattr(prefix_node, "boundary", None)
            ):
                raise RuntimeError(
                    "shared session release request differs from its exact registry"
                )
    return ReleaseCandidate(
        req,
        tree_cache,
        tree_cache.req_to_token_pool,
        key,
        entry.request_id,
        entry.row,
        boundary,
        prefix_indices,
        empty_prefix_indices,
        prefix_node,
        bool(is_insert),
    )


def _candidate_identity_changed(
    candidate: ReleaseCandidate, runtime: Any, *, confirmed: bool
) -> bool:
    req = candidate.req
    cache = candidate.tree_cache
    try:
        entry = require_request(cache, req)
        binding = runtime.binding_for(candidate.key)
        boundary = _request_boundary(req, allow_empty_kv=True)
        raw_row = getattr(req, "req_pool_idx", None)
        if cache._no_prefix:
            validate_private_prefix(
                req,
                candidate.req_to_token_pool.req_to_token[candidate.row],
                candidate.key,
                candidate.request_id,
                boundary,
            )
            prefix_node = None
        else:
            prefix_node = cache._preflight_release_node(req, provisional=False)
    except Exception:
        return True
    return bool(
        entry.key != candidate.key
        or entry.request_id != candidate.request_id
        or entry.row != candidate.row
        or raw_row != candidate.row
        or binding.request_id != candidate.request_id
        or binding.request_row != candidate.row
        or boundary != candidate.boundary
        or getattr(req, "prefix_indices", None) is not candidate.prefix_indices
        or cache.token_to_kv_pool_allocator is not _state._ALLOCATOR
        or prefix_node is not candidate.prefix_node
        or confirmed
        and int(runtime.view_for(candidate.key).boundary)
        != candidate.boundary
    )


def _pending_candidate_identity_changed(
    candidate: ReleaseCandidate, runtime: Any
) -> bool:
    """Validate identity after release cleanup has already committed."""

    req = candidate.req
    try:
        entry = require_request(candidate.tree_cache, req)
        binding = runtime.binding_for(candidate.key)
        view = runtime.view_for(candidate.key)
        boundary = _request_boundary(req, allow_empty_kv=True)
        raw_row = getattr(req, "req_pool_idx", None)
        if candidate.tree_cache._no_prefix:
            prefix_node = None
        else:
            prefix_node = candidate.tree_cache._preflight_release_node(
                req, provisional=False
            )
    except Exception:
        return True
    return bool(
        entry.key != candidate.key
        or entry.request_id != candidate.request_id
        or entry.row != candidate.row
        or raw_row != candidate.row
        or binding.request_id != candidate.request_id
        or binding.request_row != candidate.row
        or view.request_id != candidate.request_id
        or int(view.boundary) != candidate.boundary
        or boundary != candidate.boundary
        or getattr(req, "prefix_indices", None) is not candidate.prefix_indices
        or prefix_node is not candidate.prefix_node
        or candidate.tree_cache.req_to_token_pool
        is not candidate.req_to_token_pool
        or candidate.tree_cache.token_to_kv_pool_allocator
        is not _state._ALLOCATOR
    )


def _preflight_release_rows(
    values: tuple[ReleaseCandidate, ...], pool: Any
) -> Any:
    """Validate every SGLang row before making any of them reusable."""

    import torch

    free_request = getattr(pool, "free", None)
    table = getattr(pool, "req_to_token", None)
    free_slots = getattr(pool, "free_slots", None)
    if (
        not callable(free_request)
        or type(table) is not torch.Tensor
        or table.ndim != 2
        or free_slots is not None
        and not isinstance(free_slots, list)
    ):
        raise RuntimeError("session release lost its ReqToToken row authority")
    for candidate in values:
        entry = require_request(candidate.tree_cache, candidate.req)
        raw_row = getattr(candidate.req, "req_pool_idx", None)
        prefix = getattr(candidate.req, "prefix_indices", None)
        base_invalid = (
            entry.key != candidate.key
            or entry.request_id != candidate.request_id
            or entry.row != candidate.row
            or isinstance(raw_row, bool)
            or not isinstance(raw_row, Integral)
            or int(raw_row) != candidate.row
            or not 0 < candidate.row < int(table.shape[0])
            or int(torch.count_nonzero(table[candidate.row, : candidate.boundary]))
            != 0
            or free_slots is not None
            and candidate.row in free_slots
        )
        if candidate.tree_cache._no_prefix:
            invalid = (
                base_invalid
                or getattr(candidate.req, PRIVATE_PREFIX_MARKER, None) is not None
                or type(prefix) is not torch.Tensor
                or int(prefix.numel()) != 0
            )
        else:
            prefix_count = int(prefix.numel()) if type(prefix) is torch.Tensor else -1
            invalid = (
                base_invalid
                or getattr(candidate.req, PRIVATE_PREFIX_MARKER, None) is not None
                or type(prefix) is not torch.Tensor
                or prefix is not candidate.prefix_indices
                or prefix_count < 0
                or (
                    min(prefix_count, candidate.boundary) > 0
                    and int(
                        torch.count_nonzero(
                            prefix[: min(prefix_count, candidate.boundary)]
                        )
                    )
                    != 0
                )
                or candidate.tree_cache._preflight_release_node(
                    candidate.req, provisional=False
                )
                is not candidate.prefix_node
            )
        if invalid:
            raise RuntimeError(
                "session release row is not cleared and exclusively owned"
            )
    return free_request


def flush_release_group(candidates: Sequence[ReleaseCandidate]) -> None:
    """Release one SGLang free group through the native session."""

    values = tuple(candidates)
    if not values:
        raise RuntimeError("OrbitKV session release group must be nonempty")
    runtime = _runtime()
    if any(type(candidate) is not ReleaseCandidate for candidate in values):
        runtime.fail_stop(
            "SGLang session release group contains foreign candidates"
        )
        raise FailStopped(
            runtime.failure_reason or "foreign session release candidate"
        )
    keys = tuple(candidate.key for candidate in values)
    rows = {candidate.row for candidate in values}
    request_ids = {candidate.request_id for candidate in values}
    requests = {id(candidate.req) for candidate in values}
    cache = values[0].tree_cache
    pool = values[0].req_to_token_pool
    if (
        not _state._uses_runtime_session()
        or len(set(keys)) != len(values)
        or len(rows) != len(values)
        or len(request_ids) != len(values)
        or len(requests) != len(values)
        or any(
            candidate.tree_cache is not cache
            or candidate.req_to_token_pool is not pool
            or type(candidate.is_insert) is not bool
            or (
                candidate.tree_cache._no_prefix
                and candidate.prefix_node is not None
            )
            for candidate in values
        )
    ):
        fail_stop_session_cache(
            cache, "SGLang session release group contains aliased identities"
        )
        raise FailStopped(
            runtime.failure_reason or "aliased session release group"
        )
    try:
        plan = runtime.pending_release(keys)
        pending = plan is not None
        if not pending:
            if any(
                _candidate_identity_changed(
                    candidate, runtime, confirmed=False
                )
                for candidate in values
            ):
                raise RuntimeError(
                    "session release group identity changed before preparation"
                )
            if cache._no_prefix:
                plan = runtime.prepare_release(keys)
            else:
                publications = tuple(
                    (
                        candidate,
                        candidate.tree_cache.publication_for_release(
                            candidate.req, is_insert=candidate.is_insert
                        ),
                    )
                    for candidate in values
                )
                publishable = (
                    all(publication is not None for _candidate, publication in publications)
                    and len(
                        {
                            publication.semantic
                            for _candidate, publication in publications
                            if publication is not None
                        }
                    )
                    == len(values)
                )
                if publishable:
                    publish_release = runtime.prepare_prefix_publish_release(
                        tuple(
                            (candidate.key, publication.semantic)
                            for candidate, publication in publications
                            if publication is not None
                        )
                    )
                    if len(publish_release.outputs) != len(values):
                        raise RuntimeError(
                            "session prefix publish-release changed cardinality"
                        )
                    for (candidate, publication), output in zip(
                        publications, publish_release.outputs, strict=True
                    ):
                        assert publication is not None
                        candidate.tree_cache.accept_release_publication(
                            output, publication.tokens
                        )
                    plan = runtime.pending_release(keys)
                    if plan is None:
                        raise RuntimeError(
                            "session prefix publish-release did not retain a release plan"
                        )
                else:
                    plan = runtime.prepare_release(keys)
        if tuple(item.request_id for item in plan.releases) != tuple(
            candidate.request_id for candidate in values
        ):
            raise RuntimeError(
                "session release plan changed request ordering"
            )
        if pending:
            if any(
                _pending_candidate_identity_changed(candidate, runtime)
                for candidate in values
            ):
                raise RuntimeError(
                    "session release group identity changed before recycle retry"
                )
        elif any(
            _candidate_identity_changed(candidate, runtime, confirmed=True)
            for candidate in values
        ):
            raise RuntimeError(
                "session release group identity changed before confirmation"
            )
        # The runtime callback has cleared and synchronized every mirror before
        # confirm_release returns from the native ACK.
        runtime.confirm_release(plan)
        free_request = _preflight_release_rows(values, pool)
        for candidate in values:
            candidate.tree_cache._commit_release_node(
                candidate.req, candidate.prefix_node, provisional=False
            )
            if not candidate.tree_cache._no_prefix:
                clear_shared_prefix(candidate.tree_cache, candidate.req)
            unregister_request(candidate.tree_cache, candidate.req)
        for candidate in values:
            free_request(candidate.req)
            if candidate.req.req_pool_idx is not None:
                raise RuntimeError(
                    "ReqToToken release did not clear its request row"
                )
            candidate.req.kv = None
            candidate.req.prefix_indices = candidate.empty_prefix_indices
            clear_request_identity(candidate.req)
    except ReleaseRetryPending:
        raise
    except Exception as error:
        fail_stop_session_cache(
            cache,
            f"ReqToToken session release group became uncertain: {error}",
        )
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            runtime.failure_reason or "session release group failed"
        ) from error


__all__ = [
    "SessionCacheMixin",
    "bind_session_cache",
    "cache_unfinished_request",
    "cancel_pending_request",
    "clear_session_cache",
    "flush_release_group",
    "preflight_register_requests",
    "promote_pending_requests",
    "register_pending_request",
    "release_candidate",
]
