from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

from ..execution_plan import (
    BindingResult,
    StepExecutionResult,
    confirm_execution,
    expected_bindings,
    lower_batch_plan,
)
from ..runtime import FailStopped, ManagerError
from ..session_runtime import ReleaseRetryPending
from . import session_cache, state as _state
from .execution_context import (
    ForwardExecutionContext,
    capture_forward_context,
    clear_forward_context,
)
from .location_validation import validate_locations as _validate_location_tensors
from .request_rows import (
    free_new_rows as _free_new_rows_impl,
    request_row_tensors as _request_row_tensors,
    validate_allocated_rows as _validate_allocated_rows_impl,
    validate_private_admission as _validate_private_admission,
)
from .state import _config, _request_key, _runtime
from .validation import (
    _integer_vector,
    _preflight_extend_batch,
    _validate_batch,
    _validate_device_vector,
)


@dataclass(frozen=True, slots=True)
class _PreparedSessionLowering:
    native_plan: Any
    lowered: Any
    bindings: tuple[tuple[Any, ...], ...]
    cow_mirror_plan: Any


@dataclass(frozen=True, slots=True)
class _PendingNewRequestRelease:
    batch: Any
    requests: tuple[Any, ...]
    plan: Any


def _shared_lowering() -> Any:
    # Imported lazily because lowering.py owns the shared tensor/COW helpers and
    # dispatches into this module only after its own module initialization.
    from . import lowering

    return lowering


def _validate_profile(batch: Any) -> None:
    if not _state._uses_runtime_session():
        raise RuntimeError("runtime-session lowering requires its selected profile")
    shape = tuple(
        (item.class_id, item.retention, item.storage) for item in _config().classes
    )
    if shape not in (
        ((0, "full", "token_kv"),),
        ((0, "full", "latent_kv"),),
        ((0, "full", "token_kv"), (1, "sliding", "token_kv")),
        ((0, "sliding", "token_kv"),),
        ((0, "chunked", "token_kv"),),
    ):
        raise RuntimeError(
            "runtime-session requires Full token_kv/latent_kv, ordered "
            "Full+SWA token_kv, pure SWA token_kv, or Chunked token_kv classes"
        )
    expected_private = _state._requires_disabled_radix_cache()
    if getattr(batch.tree_cache, "_no_prefix", None) is not expected_private:
        raise RuntimeError(
            "runtime-session radix mode differs from the selected profile"
        )


def _session_prefix_authoritative(
    req: Any,
    key: Any,
    boundary: int,
    *,
    pending_attach: Any | None,
) -> bool:
    """Validate and freeze one pre-row shared-Prefix authority decision."""

    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, Integral)
        or int(boundary) < 0
    ):
        raise RuntimeError("shared Prefix authority has an invalid boundary")
    boundary = int(boundary)
    if hasattr(req, "_orbitkv_request_lease"):
        raise RuntimeError(
            "runtime-session request retained a legacy request lease; "
            "shared Prefix authority is invalid"
        )
    request_id = getattr(req, "_orbitkv_engine_request_id", None)
    if (
        getattr(req, "_orbitkv_request_key", None) != key
        or isinstance(request_id, bool)
        or not isinstance(request_id, Integral)
        or int(request_id) <= 0
    ):
        raise RuntimeError("shared Prefix authority has an invalid session identity")
    runtime = _runtime()
    binding = runtime.binding_for(key)
    view = runtime.view_for(key)
    raw_row = getattr(req, "req_pool_idx", None)
    expected_view_boundary = 0 if pending_attach is not None else boundary
    if (
        binding.request_id != request_id
        or view.request_id != request_id
        or int(view.boundary) != expected_view_boundary
        or pending_attach is not None
        and int(view.resident_count) != 0
        or raw_row is None
        and binding.request_row is not None
        or raw_row is not None
        and (
            isinstance(raw_row, bool)
            or not isinstance(raw_row, Integral)
            or binding.request_row != int(raw_row)
        )
    ):
        raise RuntimeError("shared Prefix authority differs from runtime identity")

    prefix = getattr(req, "prefix_indices", None)
    try:
        prefix_count = len(prefix)
    except BaseException as error:
        raise RuntimeError("shared Prefix authority has an unreadable mirror") from error
    node = getattr(req, "_orbitkv_prefix_node", None)
    semantic = getattr(req, "_orbitkv_prefix_semantic", None)
    last_node = getattr(req, "last_node", None)
    provisional = getattr(req, "_orbitkv_provisional_prefix_lock", False)
    held = getattr(req, "_orbitkv_prefix_lock_held", False)
    node_is_non_root = node is not None and getattr(node, "parent", None) is not None
    last_is_non_root = (
        last_node is not None and getattr(last_node, "parent", None) is not None
    )
    shared_claimed = (
        node_is_non_root
        or semantic is not None
        or hasattr(req, "_orbitkv_provisional_prefix_lock")
        or hasattr(req, "_orbitkv_prefix_lock_held")
        or prefix_count > 0
        or last_is_non_root
    )
    if not shared_claimed:
        if boundary != 0 or prefix_count != 0 or pending_attach is not None:
            raise RuntimeError(
                "shared Prefix authority is missing for a nonempty boundary"
            )
        return False
    if pending_attach is None and raw_row is None:
        raise RuntimeError(
            "rowless shared Prefix authority has no pending attach plan"
        )
    if (
        boundary <= 0
        or prefix_count != boundary
        or not node_is_non_root
        or bool(getattr(node, "evicted", False))
        or getattr(node, "prefix", None) is None
        or last_node is not node
        or getattr(semantic, "boundary", None) != boundary
        or getattr(node, "boundary", None) != boundary
        or getattr(semantic, "digest", None) != getattr(node, "digest", None)
        or type(provisional) is not bool
        or type(held) is not bool
        or provisional == held
        or raw_row is None
        and not provisional
        or raw_row is not None
        and not held
        or isinstance(getattr(node, "lock_ref", None), bool)
        or not isinstance(getattr(node, "lock_ref", None), Integral)
        or int(node.lock_ref) <= 0
    ):
        raise RuntimeError("shared Prefix authority is incomplete or inconsistent")
    return True


def _preflight_pending_admission(
    batch: Any,
    keys: Sequence[Any],
    boundaries: Sequence[int],
    new_req_slots: Sequence[bool],
) -> tuple[tuple[Any | None, ...], tuple[bool, ...]]:
    """Snapshot exact pending identities and shared-Prefix authority."""

    pending = getattr(batch.tree_cache, "_session_pending_requests", None)
    if not isinstance(pending, dict):
        raise RuntimeError("session cache pending registry is not bound")
    entries: list[Any | None] = []
    authoritative: list[bool] = []
    runtime = _runtime()
    for req, key, boundary, is_new in zip(
        batch.reqs, keys, boundaries, new_req_slots, strict=True
    ):
        pending_attach = session_cache.pending_shared_prefix(
            batch.tree_cache, req
        )
        if _state._requires_disabled_radix_cache():
            authoritative.append(
                _validate_private_admission(
                    batch.tree_cache,
                    req,
                    key,
                    int(boundary),
                    pending_attach=pending_attach,
                    runtime=runtime,
                    config=_config(),
                )
            )
        else:
            authoritative.append(
                _session_prefix_authoritative(
                    req, key, int(boundary), pending_attach=pending_attach
                )
            )
        entry = pending.get(key) if is_new else None
        if is_new:
            request_id = getattr(req, "_orbitkv_engine_request_id", None)
            binding = runtime.binding_for(key)
            view = runtime.view_for(key)
            if (
                entry is None
                or getattr(entry, "req", None) is not req
                or getattr(entry, "key", None) != key
                or getattr(entry, "request_id", None) != request_id
                or binding.request_id != request_id
                or binding.request_row is not None
                or view.request_id != request_id
                or int(view.boundary) != 0
                or int(view.resident_count) != 0
                or (pending_attach is None) != (int(boundary) == 0)
            ):
                raise RuntimeError(
                    "rowless session request has no exact pending identity"
                )
        elif entry is not None:
            raise RuntimeError(
                "row-bound session request retained pending identity"
            )
        elif pending_attach is not None:
            raise RuntimeError(
                "row-bound session request retained a pending shared Prefix"
            )
        entries.append(entry)
    return tuple(entries), tuple(authoritative)


def _validate_allocated_rows(
    batch: Any, req_pool_indices: Any
) -> tuple[int, ...]:
    return _validate_allocated_rows_impl(
        batch, req_pool_indices, integer_vector=_integer_vector
    )


def _free_new_rows(batch: Any, new_req_slots: Sequence[bool]) -> None:
    _free_new_rows_impl(batch, new_req_slots, runtime=_runtime())


def _release_new_requests(batch: Any, requests: Sequence[Any]) -> None:
    """Release acquired requests after a provably unobserved failure."""

    values = tuple(requests)
    if not values:
        return
    runtime = _runtime()
    release = None
    try:
        release = runtime.prepare_release(
            tuple(_request_key(req) for req in values)
        )
        runtime.confirm_release(release)
        _finish_new_request_release(batch, values)
    except ReleaseRetryPending as error:
        if release is None or error.plan is not release:
            runtime.fail_stop(
                "session request rollback returned a foreign retry plan"
            )
            raise FailStopped(
                runtime.failure_reason or "session request rollback changed plan"
            ) from error
        _retain_pending_new_request_release(batch, values, release)
        raise
    except BaseException as error:
        runtime.fail_stop(f"session request rollback became uncertain: {error}")
        raise FailStopped(
            runtime.failure_reason or "session request rollback failed"
        ) from error


def _cancel_unbound_new_requests(
    batch: Any, requests: Sequence[Any], new_req_slots: Sequence[bool]
) -> None:
    """Return unbound SGLang rows, then cancel their native identities."""

    runtime = _runtime()
    try:
        _free_new_rows(batch, new_req_slots)
        for req in requests:
            batch.tree_cache._cancel_pending_session_request(req)
    except BaseException as error:
        session_cache.fail_stop_session_cache(
            batch.tree_cache,
            "unbound session admission rollback became uncertain: "
            f"{type(error).__name__}: {error}",
        )
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            runtime.failure_reason or "unbound session admission rollback failed"
        ) from error


def _finish_new_request_release(batch: Any, requests: Sequence[Any]) -> None:
    if not getattr(batch.tree_cache, "_no_prefix", True):
        prefix_work = tuple(
            (
                req,
                batch.tree_cache._preflight_release_node(
                    req,
                    provisional=bool(
                        getattr(req, "_orbitkv_provisional_prefix_lock", False)
                    ),
                ),
                bool(
                    getattr(req, "_orbitkv_provisional_prefix_lock", False)
                ),
            )
            for req in requests
        )
        for req, node, provisional in prefix_work:
            batch.tree_cache._commit_release_node(
                req, node, provisional=provisional
            )
    for req in requests:
        batch.tree_cache._unregister_session_request(req)
    for req in requests:
        batch.req_to_token_pool.free(req)
        req.kv = None
        for name in (
            "_orbitkv_engine_request_id",
            "_orbitkv_request_key",
        ):
            if hasattr(req, name):
                delattr(req, name)


def _retain_pending_new_request_release(
    batch: Any, requests: tuple[Any, ...], plan: Any
) -> None:
    allocator = _state._ALLOCATOR
    group_state = getattr(allocator, "_orbitkv_free_group_state", None)
    retained = getattr(allocator, "_orbitkv_pending_release_work", None)
    free_group = getattr(allocator, "free_group", None)
    work = _PendingNewRequestRelease(batch, requests, plan)
    valid = bool(requests) and free_group == []
    if group_state == "retry_pending":
        valid = (
            valid
            and type(retained) is _PendingNewRequestRelease
            and retained.batch is batch
            and retained.plan is plan
            and _same_release_candidates(retained.requests, requests)
        )
    else:
        valid = valid and group_state == "idle" and retained is None
    if not valid:
        runtime = _runtime()
        runtime.fail_stop(
            "SGLang lost ownership of a pending request rollback"
        )
        raise FailStopped(
            runtime.failure_reason or "pending request rollback was lost"
        )
    allocator._orbitkv_pending_release_work = work
    allocator._orbitkv_free_group_state = "retry_pending"
    allocator.is_not_in_free_group = True


def _abort_prepared(
    batch: Any, native_plan: Any, new_requests: Sequence[Any]
) -> None:
    runtime = _runtime()
    try:
        runtime.abort_prepared(native_plan)
    except BaseException as error:
        runtime.fail_stop(f"session prepared abort became uncertain: {error}")
        raise FailStopped(
            runtime.failure_reason or "session prepared abort failed"
        ) from error
    _release_new_requests(batch, new_requests)


def _fail_committed_admission(batch: Any, error: BaseException) -> None:
    """Quarantine every retained attach and poison a bound admission."""

    runtime = _runtime()
    quarantine_errors: list[str] = []
    pending = getattr(batch.tree_cache, "_session_pending_shared_prefix", None)
    entries = tuple(pending.values()) if isinstance(pending, dict) else ()
    for entry in entries:
        control_id = getattr(entry, "control_id", None)
        if control_id is None:
            continue
        try:
            runtime.quarantine_control(control_id)
        except BaseException as caught:
            quarantine_errors.append(
                f"{type(caught).__name__}: {caught}"
            )
    reason = (
        "committed session attach handoff failed: "
        f"{type(error).__name__}: {error}"
    )
    if quarantine_errors:
        reason += "; attach quarantine failed: " + "; ".join(quarantine_errors)
    runtime.fail_stop(reason)
    raise FailStopped(runtime.failure_reason or reason) from error


def _has_pending_shared_attach(batch: Any, requests: Sequence[Any]) -> bool:
    pending = getattr(batch.tree_cache, "_session_pending_shared_prefix", None)
    if not isinstance(pending, dict):
        return False
    request_ids = {id(req) for req in requests}
    return any(
        id(getattr(entry, "req", None)) in request_ids
        and getattr(entry, "control_id", None) is not None
        for entry in pending.values()
    )


def _prepare_lowering(
    batch: Any,
    previous: Sequence[int],
    targets: Sequence[int],
    req_pool_values: Sequence[int],
    new_req_slots: Sequence[bool],
    execution_context: ForwardExecutionContext | None = None,
    pending_entries: Sequence[Any | None] | None = None,
    prefix_authoritative: Sequence[bool] | None = None,
) -> _PreparedSessionLowering:
    """Bind pending rows and lower a native plan before observation."""

    runtime = _runtime()
    keys: tuple[Any, ...] = ()
    new_requests: tuple[Any, ...] = ()
    new_assignments: tuple[tuple[Any, int], ...] = ()
    rows_bound = False
    promoted = False
    native_plan = None
    committed_attach = False
    pending_attach_entries: tuple[Any, ...] = ()
    try:
        keys = tuple(_request_key(req) for req in batch.reqs)
        if len(set(keys)) != len(keys):
            raise RuntimeError("session lowering contains duplicate request keys")
        pending_values = (
            (None,) * len(keys)
            if pending_entries is None
            else tuple(pending_entries)
        )
        prefix_values = (
            (False,) * len(keys)
            if prefix_authoritative is None
            else tuple(prefix_authoritative)
        )
        if (
            len(pending_values) != len(keys)
            or len(prefix_values) != len(keys)
            or any(type(value) is not bool for value in prefix_values)
        ):
            raise RuntimeError("session admission snapshot cardinality changed")
        new_requests = tuple(
            req
            for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
            if is_new
        )
        committed_attach = _has_pending_shared_attach(batch, new_requests)
        pending_attach_entries = tuple(
            entry
            for req in new_requests
            if (entry := session_cache.pending_shared_prefix(
                batch.tree_cache, req
            )) is not None
        )
        new_assignments = tuple(
            (key, int(row))
            for key, row, is_new in zip(
                keys, req_pool_values, new_req_slots, strict=True
            )
            if is_new
        )
        pending_registry = getattr(
            batch.tree_cache, "_session_pending_requests", None
        )
        if not isinstance(pending_registry, dict):
            raise RuntimeError("session cache pending registry is not bound")
        for req, key, row, is_new, pending_entry in zip(
            batch.reqs,
            keys,
            req_pool_values,
            new_req_slots,
            pending_values,
            strict=True,
        ):
            if hasattr(req, "_orbitkv_request_lease"):
                raise RuntimeError(
                    "runtime-session request retained a legacy request lease"
                )
            if is_new:
                request_id = getattr(req, "_orbitkv_engine_request_id", None)
                binding = runtime.binding_for(key)
                if (
                    pending_entry is None
                    or pending_registry.get(key) is not pending_entry
                    or getattr(pending_entry, "req", None) is not req
                    or getattr(pending_entry, "key", None) != key
                    or getattr(pending_entry, "request_id", None) != request_id
                    or getattr(req, "_orbitkv_request_key", None) != key
                    or binding.request_id != request_id
                    or binding.request_row is not None
                ):
                    raise RuntimeError(
                        "new SGLang request has no exact pending session identity"
                    )
                continue
            if pending_entry is not None or key in pending_registry:
                raise RuntimeError(
                    "existing SGLang request retained pending session identity"
                )
            binding = runtime.binding_for(key)
            entry = session_cache.require_request(batch.tree_cache, req)
            if (
                binding.request_row != int(row)
                or entry.key != key
                or entry.request_id != binding.request_id
                or entry.row != binding.request_row
                or getattr(req, "_orbitkv_request_key", None) != key
                or getattr(req, "_orbitkv_engine_request_id", None)
                != binding.request_id
            ):
                raise RuntimeError(
                    "existing SGLang request differs from its session binding"
                )

        if new_assignments:
            try:
                runtime.bind_request_rows(new_assignments)
            except ManagerError:
                raise
            except BaseException as error:
                session_cache.fail_stop_session_cache(
                    batch.tree_cache,
                    "session request-row binding became uncertain: "
                    f"{type(error).__name__}: {error}",
                )
                raise FailStopped(
                    runtime.failure_reason or "session request-row binding failed"
                ) from error
            rows_bound = True
            try:
                for pending_entry in (
                    item for item in pending_values if item is not None
                ):
                    if pending_registry.get(pending_entry.key) is not pending_entry:
                        raise RuntimeError(
                            "pending session identity changed before promotion"
                        )
                for req in new_requests:
                    entry = next(
                        (
                            candidate
                            for candidate in pending_attach_entries
                            if candidate.req is req
                        ),
                        None,
                    )
                    if entry is not None:
                        session_cache.confirm_pending_shared_prefix(
                            batch.tree_cache, req
                        )
                batch.tree_cache._promote_pending_session_requests(new_requests)
            except BaseException as error:
                _fail_committed_admission(batch, error)
            promoted = True

        native_plan = runtime.prepare(
            tuple(zip(keys, (int(value) for value in targets), strict=True))
        )
        views = runtime.views_for(keys)
        if len(views) != len(keys):
            raise RuntimeError("session view cardinality changed before lowering")
        expected_views = {}
        for req, key, row, boundary, view in zip(
            batch.reqs, keys, req_pool_values, previous, views, strict=True
        ):
            binding = runtime.binding_for(key)
            if (
                binding.request_row != int(row)
                or binding.request_id != view.request_id
                or getattr(req, "_orbitkv_request_key", None) != key
                or getattr(req, "_orbitkv_engine_request_id", None)
                != binding.request_id
                or int(view.boundary) != int(boundary)
                or view.request_id in expected_views
            ):
                raise RuntimeError(
                    "session request view differs from the SGLang mirror"
                )
            expected_views[view.request_id] = view
        lowered = lower_batch_plan(
            native_plan, _config(), runtime.arenas, expected_views
        )
        shared = _shared_lowering()
        if (
            _config().full_class is not None
            and _config().sliding_class is not None
        ):
            shared._validate_joint_hybrid_tails(lowered.steps)
        cow_mirror_plan = shared._preflight_cow_mirrors(
            batch,
            lowered.steps,
            prefix_values,
            execution_context,
        )
        bindings = expected_bindings(native_plan, lowered, runtime.arenas)
        return _PreparedSessionLowering(
            native_plan, lowered, bindings, cow_mirror_plan
        )
    except BaseException as error:
        if runtime.failure_reason is not None:
            raise
        if committed_attach and native_plan is not None:
            _quarantine_prepared(native_plan, error)
        elif committed_attach:
            _fail_committed_admission(
                batch, error,
            )
        elif native_plan is not None:
            _abort_prepared(batch, native_plan, new_requests)
        elif rows_bound and promoted:
            _release_new_requests(batch, new_requests)
        elif rows_bound:
            _fail_committed_admission(batch, RuntimeError(
                "session row binding completed without a prepared append"
            ))
        else:
            _cancel_unbound_new_requests(
                batch, new_requests, new_req_slots
            )
        raise


def _execution_results(
    prepared: _PreparedSessionLowering,
) -> tuple[StepExecutionResult, ...]:
    """Record success against the fixed, pre-mapped SGLang arena.

    Arena registration is the mapping/writability proof for each selected
    backend index.  Callers additionally validate the kernel-produced token
    locations and enqueue every exact copy before constructing these results.
    """

    return tuple(
        StepExecutionResult(
            step.request_id,
            tuple(
                BindingResult(
                    item.page,
                    item.backend_domain,
                    item.backend_index,
                    True,
                    True,
                )
                for item in bindings
            ),
            tuple(
                intent
                for class_spec in step.class_specs
                for intent in class_spec.copy_intents
            ),
            True,
        )
        for step, bindings in zip(
            prepared.lowered.steps, prepared.bindings, strict=True
        )
    )


def _validate_locations(
    locations: Any,
    lowered: Any,
    execution_context: ForwardExecutionContext | None = None,
) -> None:
    config = _config()
    _validate_location_tensors(
        locations,
        lowered,
        tuple(item.class_id for item in config.classes),
        config.page_tokens,
        execution_context,
    )


def _quarantine_prepared(native_plan: Any, error: BaseException) -> None:
    runtime = _runtime()
    reason = f"session prepared execution failed: {type(error).__name__}: {error}"
    quarantine_error: BaseException | None = None
    try:
        runtime.quarantine_prepared(native_plan.batch_id)
    except FailStopped:
        pass
    except BaseException as caught:
        quarantine_error = caught
    if runtime.failure_reason is None:
        if quarantine_error is not None:
            reason += (
                "; prepared quarantine became uncertain: "
                f"{type(quarantine_error).__name__}: {quarantine_error}"
            )
        runtime.fail_stop(reason)
    raise FailStopped(
        runtime.failure_reason or reason
    ) from error


def _quarantine_submitted(ticket: Any, error: BaseException) -> None:
    runtime = _runtime()
    reason = f"session submitted execution failed: {type(error).__name__}: {error}"
    quarantine_error: BaseException | None = None
    try:
        runtime.quarantine_submitted(ticket)
    except FailStopped:
        pass
    except BaseException as caught:
        quarantine_error = caught
    if runtime.failure_reason is None:
        if quarantine_error is not None:
            reason += (
                "; submitted quarantine became uncertain: "
                f"{type(quarantine_error).__name__}: {quarantine_error}"
            )
        runtime.fail_stop(reason)
    raise FailStopped(
        runtime.failure_reason or reason
    ) from error


def _discard_unsubmitted_context(
    batch: Any, context: ForwardExecutionContext
) -> None:
    """Remove a context only while no native plan or ticket exists."""

    try:
        clear_forward_context(batch, context)
    except BaseException as error:
        runtime = _runtime()
        runtime.fail_stop(
            "session forward context cleanup became uncertain: "
            f"{type(error).__name__}: {error}"
        )
        raise FailStopped(
            runtime.failure_reason or "session forward context cleanup failed"
        ) from error


def _fail_prepared(
    batch: Any,
    context: ForwardExecutionContext,
    native_plan: Any,
    error: BaseException,
) -> None:
    try:
        clear_forward_context(batch, context)
    except BaseException as cleanup_error:
        error = RuntimeError(
            "session prepared execution failed and context cleanup became "
            f"uncertain: {type(cleanup_error).__name__}: {cleanup_error}"
        )
    _quarantine_prepared(native_plan, error)


def _fail_submitted(
    batch: Any,
    context: ForwardExecutionContext,
    ticket: Any,
    error: BaseException,
) -> None:
    try:
        clear_forward_context(batch, context)
    except BaseException as cleanup_error:
        error = RuntimeError(
            "session submitted execution failed and context cleanup became "
            f"uncertain: {type(cleanup_error).__name__}: {cleanup_error}"
        )
    _quarantine_submitted(ticket, error)


def _publish_submitted(
    batch: Any, context: ForwardExecutionContext, ticket: Any
) -> None:
    context.bind_ticket(ticket)
    if hasattr(batch, "_orbitkv_session_ticket"):
        raise RuntimeError("batch already has a session ticket")
    setattr(batch, "_orbitkv_session_ticket", ticket)
    if getattr(batch, "_orbitkv_session_ticket", None) is not ticket:
        raise RuntimeError("batch did not retain the issued session ticket")


def alloc_for_extend(batch: Any) -> tuple[Any, Any, Any]:
    import sglang.srt.mem_cache.allocation as allocation
    import torch
    from sglang.srt.managers.schedule_batch import ReqKvInfo

    _validate_batch(batch)
    _validate_profile(batch)
    prefix_values, extend_values, target_values = _preflight_extend_batch(batch)
    new_req_slots = tuple(req.req_pool_idx is None for req in batch.reqs)
    try:
        keys = tuple(_request_key(req) for req in batch.reqs)
        pending_entries, prefix_authoritative = _preflight_pending_admission(
            batch, keys, prefix_values, new_req_slots
        )
    except BaseException as error:
        new_requests = tuple(
            req
            for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
            if is_new
        )
        if _has_pending_shared_attach(batch, new_requests):
            _fail_committed_admission(batch, error)
        pending = getattr(batch.tree_cache, "_session_pending_requests", None)
        if isinstance(pending, dict) and any(
            getattr(entry, "req", None) is req
            for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
            if is_new
            for entry in pending.values()
        ):
            session_cache.fail_stop_session_cache(
                batch.tree_cache,
                "pending session admission identity became uncertain: "
                f"{type(error).__name__}: {error}",
            )
            if isinstance(error, FailStopped):
                raise
            raise FailStopped(
                _runtime().failure_reason
                or "pending session admission identity failed"
            ) from error
        raise
    batch.maybe_evict_swa()
    context = capture_forward_context(batch)
    try:
        prefix_tensors = [req.prefix_indices for req in batch.reqs]
        prefix_lens_cpu = torch.tensor(prefix_values, dtype=torch.int64)
        extend_lens_cpu = torch.tensor(extend_values, dtype=torch.int64)
        targets_cpu = torch.tensor(target_values, dtype=torch.int64)
        context.validate_current()
        targets_device = targets_cpu.to(batch.device, non_blocking=True)
    except BaseException as error:
        try:
            new_requests = tuple(
                req for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
                if is_new
            )
            if _has_pending_shared_attach(batch, new_requests):
                _fail_committed_admission(batch, error)
            _cancel_unbound_new_requests(batch, new_requests, new_req_slots)
        finally:
            _discard_unsubmitted_context(batch, context)
        raise

    try:
        req_pool_indices = allocation.alloc_req_slots(
            batch.req_to_token_pool, batch.reqs, batch.tree_cache
        )
        req_pool_values = _validate_allocated_rows(batch, req_pool_indices)
    except BaseException as error:
        try:
            new_requests = tuple(
                req
                for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
                if is_new
            )
            if _has_pending_shared_attach(batch, new_requests):
                _fail_committed_admission(batch, error)
            _cancel_unbound_new_requests(batch, new_requests, new_req_slots)
        finally:
            _discard_unsubmitted_context(batch, context)
        raise

    try:
        req_pool_indices_cpu, req_pool_indices_device = _request_row_tensors(
            req_pool_values, batch.device, context
        )
    except BaseException as error:
        try:
            new_requests = tuple(
                req
                for req, is_new in zip(batch.reqs, new_req_slots, strict=True)
                if is_new
            )
            if _has_pending_shared_attach(batch, new_requests):
                _fail_committed_admission(batch, error)
            _cancel_unbound_new_requests(batch, new_requests, new_req_slots)
        finally:
            _discard_unsubmitted_context(batch, context)
        raise
    try:
        prepared = _prepare_lowering(
            batch,
            prefix_values,
            target_values,
            req_pool_values,
            new_req_slots,
            context,
            pending_entries,
            prefix_authoritative,
        )
    except BaseException:
        _discard_unsubmitted_context(batch, context)
        raise
    shared = _shared_lowering()
    try:
        locations = shared._lower_all_extend(
            batch,
            prefix_lens_cpu,
            targets_cpu,
            int(batch.extend_num_tokens),
            prepared.lowered.steps,
            context,
        )
        _validate_locations(locations, prepared.lowered, context)
        out_cache_loc = shared._primary_locations(locations)
        # move_kv_cache has no receipt API. A normal return on the current
        # forward stream is the backend success witness for these exact intents.
        cow_activity = shared._execute_cow_copies(
            batch, prepared.lowered.steps, context
        )
        evidence = confirm_execution(
            prepared.native_plan,
            prepared.lowered,
            _runtime().arenas,
            _execution_results(prepared),
        )
        context.validate_current()
        ticket = _runtime().submit(evidence)
    except BaseException as error:
        _fail_prepared(batch, context, prepared.native_plan, error)

    try:
        _publish_submitted(batch, context, ticket)
        shared._record_cow_activity(cow_activity)
        context.validate_current()
        prefix_lens_device = prefix_lens_cpu.to(
            batch.device, non_blocking=True
        )
        context.validate_current()
        extend_lens_device = extend_lens_cpu.to(
            batch.device, non_blocking=True
        )
        context.validate_current()
        allocation.write_cache_indices(
            out_cache_loc,
            req_pool_indices_device,
            req_pool_indices_cpu,
            prefix_lens_device,
            prefix_lens_cpu,
            targets_device,
            targets_cpu,
            extend_lens_device,
            extend_lens_cpu,
            prefix_tensors,
            batch.req_to_token_pool,
        )
        if (
            _config().full_class is not None
            and _config().sliding_class is not None
        ):
            context.validate_current()
            shared._write_hybrid_lut(locations)
            context.validate_current()
        shared._commit_cow_mirrors(prepared.cow_mirror_plan, context)
        for req, target in zip(batch.reqs, target_values, strict=True):
            if req.kv is None:
                req.kv = ReqKvInfo(
                    kv_allocated_len=int(target), swa_evicted_seqlen=0
                )
            else:
                req.kv.kv_allocated_len = int(target)
        batch.seq_lens = targets_device
    except BaseException as error:
        _fail_submitted(batch, context, ticket, error)
    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def _preflight_decode_batch(
    batch: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    batch_size = len(batch.reqs)
    if batch_size <= 0:
        raise RuntimeError("OrbitKV cannot allocate an empty decode batch")
    previous = _integer_vector(
        "seq_lens_cpu", batch.seq_lens_cpu, batch_size
    )
    _validate_device_vector(
        "seq_lens", batch.seq_lens, batch_size, batch.device
    )
    _validate_device_vector(
        "req_pool_indices", batch.req_pool_indices, batch_size, batch.device
    )
    req_pool_indices = _integer_vector(
        "req_pool_indices_cpu", batch.req_pool_indices_cpu, batch_size
    )
    if any(value < 0 for value in previous):
        raise RuntimeError("SGLang decode sequence lengths must be nonnegative")
    if any(value <= 0 for value in req_pool_indices):
        raise RuntimeError(
            "SGLang request-pool indices must exclude the dummy row"
        )
    row_capacity = int(batch.req_to_token_pool.req_to_token.shape[0])
    maximum = int(batch.req_to_token_pool.max_context_len)
    if len(set(req_pool_indices)) != batch_size:
        raise RuntimeError("SGLang decode request-pool indices alias")
    for req, request_pool_index, boundary in zip(
        batch.reqs, req_pool_indices, previous, strict=True
    ):
        if isinstance(request_pool_index, bool) or not isinstance(
            request_pool_index, Integral
        ):
            raise RuntimeError("SGLang request-pool index is not an integer")
        if request_pool_index >= row_capacity:
            raise RuntimeError(
                "SGLang decode request-pool index is out of range"
            )
        if boundary + 1 > maximum:
            raise RuntimeError(
                "SGLang decode boundary exceeds ReqToToken capacity"
            )
        if req.req_pool_idx is None or int(req.req_pool_idx) != request_pool_index:
            raise RuntimeError(
                "SGLang request-pool identity differs from the batch"
            )
        if req.kv is None or int(req.kv.kv_allocated_len) != boundary:
            raise RuntimeError(
                "SGLang request KV boundary differs from the batch"
            )
    return previous, req_pool_indices


def alloc_for_decode(batch: Any, token_per_req: int) -> Any:
    import torch

    _validate_batch(batch)
    _validate_profile(batch)
    if int(token_per_req) != 1:
        raise RuntimeError("OrbitKV supports one decode token per request")
    previous, req_pool_values = _preflight_decode_batch(batch)
    batch.maybe_evict_swa()
    targets = tuple(value + 1 for value in previous)
    context = capture_forward_context(batch)
    try:
        targets_cpu = torch.tensor(targets, dtype=torch.int64)
        previous_cpu = torch.tensor(previous, dtype=torch.int64)
        context.validate_current()
        previous_device = previous_cpu.to(batch.device, non_blocking=True)
        req_pool_indices_cpu = torch.tensor(
            req_pool_values, dtype=torch.int64
        )
        context.validate_current()
        req_pool_indices_device = req_pool_indices_cpu.to(
            batch.device, non_blocking=True
        )
        context.validate_current()
        active_previous_device = torch.tensor(
            previous, dtype=torch.int64, device=batch.device
        )
        batch.seq_lens = previous_device
        batch.req_pool_indices = req_pool_indices_device
    except BaseException:
        _discard_unsubmitted_context(batch, context)
        raise

    try:
        prepared = _prepare_lowering(
            batch,
            previous,
            targets,
            req_pool_values,
            (False,) * len(batch.reqs),
            context,
            (None,) * len(batch.reqs),
            (False,) * len(batch.reqs),
        )
    except BaseException:
        _discard_unsubmitted_context(batch, context)
        raise
    shared = _shared_lowering()
    try:
        locations = shared._lower_all_decode(
            batch, targets_cpu, prepared.lowered.steps, context
        )
        _validate_locations(locations, prepared.lowered, context)
        out_cache_loc = shared._primary_locations(locations)
        # See alloc_for_extend: successful enqueue on this stream is the only
        # copy receipt exposed by the current SGLang KV-pool API.
        cow_activity = shared._execute_cow_copies(
            batch, prepared.lowered.steps, context
        )
        evidence = confirm_execution(
            prepared.native_plan,
            prepared.lowered,
            _runtime().arenas,
            _execution_results(prepared),
        )
        context.validate_current()
        ticket = _runtime().submit(evidence)
    except BaseException as error:
        _fail_prepared(batch, context, prepared.native_plan, error)

    try:
        _publish_submitted(batch, context, ticket)
        shared._record_cow_activity(cow_activity)
        context.validate_current()
        out_cache_loc_i32 = out_cache_loc.to(torch.int32)
        context.validate_current()
        batch.req_to_token_pool.write(
            (batch.req_pool_indices, active_previous_device),
            out_cache_loc_i32,
        )
        if (
            _config().full_class is not None
            and _config().sliding_class is not None
        ):
            context.validate_current()
            shared._write_hybrid_lut(locations)
            context.validate_current()
        shared._commit_cow_mirrors(prepared.cow_mirror_plan, context)
        for req, target in zip(batch.reqs, targets, strict=True):
            req.kv.kv_allocated_len = int(target)
    except BaseException as error:
        _fail_submitted(batch, context, ticket, error)
    return out_cache_loc


def manager_maybe_evict_swa(batch: Any) -> None:
    _validate_batch(batch)
    runtime = _runtime()
    runtime.poll()
    keys = []
    for req in batch.reqs:
        request_id = getattr(req, "_orbitkv_engine_request_id", None)
        if request_id is None:
            continue
        key = _request_key(req)
        binding = runtime.binding_for(key)
        if getattr(req, "_orbitkv_request_key", None) != key or (
            binding.request_id != request_id
        ):
            raise RuntimeError("SGLang request differs from its session binding")
        if binding.request_row is None:
            pending = getattr(batch.tree_cache, "_session_pending_requests", None)
            entry = None if not isinstance(pending, dict) else pending.get(key)
            if (
                req.req_pool_idx is not None
                or entry is None
                or getattr(entry, "req", None) is not req
                or getattr(entry, "request_id", None) != request_id
            ):
                raise RuntimeError(
                    "SGLang pending request differs from its session binding"
                )
        elif req.req_pool_idx is None or binding.request_row != int(
            req.req_pool_idx
        ):
            raise RuntimeError("SGLang request differs from its session binding")
        keys.append(key)
    if keys:
        runtime.wait_requests(tuple(keys))


def get_next_batch_to_run(
    original_fn: Any, scheduler: Any, *args: Any, **kwargs: Any
) -> Any:
    runtime = _runtime()
    try:
        retry_pending_release()
        with runtime.scheduler_turn():
            return original_fn(scheduler, *args, **kwargs)
    except ReleaseRetryPending:
        raise
    except BaseException as error:
        runtime.fail_stop(f"SGLang session pre-forward failed: {error}")
        raise FailStopped(
            runtime.failure_reason or "session pre-forward failed"
        ) from error


def _same_release_candidates(
    left: Sequence[Any], right: Sequence[Any]
) -> bool:
    return len(left) == len(right) and all(
        first is second for first, second in zip(left, right, strict=True)
    )


def _retain_pending_release_group(candidates: Sequence[Any]) -> None:
    values = tuple(candidates)
    allocator = _state._ALLOCATOR
    group_state = getattr(allocator, "_orbitkv_free_group_state", None)
    retained = getattr(allocator, "free_group", None)
    valid = bool(values) and group_state in {"idle", "flushing", "retry_pending"}
    if group_state == "retry_pending":
        valid = valid and isinstance(retained, tuple) and _same_release_candidates(
            retained, values
        )
    elif retained not in (None, []):
        valid = (
            group_state == "flushing"
            and isinstance(retained, tuple)
            and _same_release_candidates(retained, values)
        )
    if not valid:
        runtime = _runtime()
        runtime.fail_stop(
            "SGLang lost ownership of a pending session release group"
        )
        raise FailStopped(
            runtime.failure_reason or "pending session release group was lost"
        )
    allocator.free_group = values
    allocator._orbitkv_free_group_state = "retry_pending"
    allocator.is_not_in_free_group = True


def _retry_pending_new_request_release(
    work: _PendingNewRequestRelease, allocator: Any
) -> None:
    runtime = _runtime()
    values = work.requests
    try:
        keys = tuple(_request_key(req) for req in values)
        if (
            not values
            or getattr(allocator, "free_group", None) != []
            or runtime.pending_release(keys) is not work.plan
            or tuple(item.request_id for item in work.plan.releases)
            != tuple(
                session_cache.require_request(work.batch.tree_cache, req).request_id
                for req in values
            )
            or any(
                runtime.binding_for(key).request_row
                != session_cache.require_request(work.batch.tree_cache, req).row
                for req, key in zip(values, keys, strict=True)
            )
        ):
            raise RuntimeError(
                "pending session request rollback changed identity"
            )
        allocator._orbitkv_free_group_state = "flushing"
        runtime.confirm_release(work.plan)
        _finish_new_request_release(work.batch, values)
    except ReleaseRetryPending as error:
        if error.plan is not work.plan:
            runtime.fail_stop(
                "pending session request rollback changed its retry plan"
            )
            allocator._orbitkv_free_group_state = "flushing"
            raise FailStopped(
                runtime.failure_reason or "pending request rollback changed plan"
            ) from error
        allocator._orbitkv_pending_release_work = work
        allocator._orbitkv_free_group_state = "retry_pending"
        allocator.is_not_in_free_group = True
        raise
    except Exception as error:
        runtime.fail_stop(
            f"pending session request rollback became uncertain: {error}"
        )
        allocator._orbitkv_free_group_state = "flushing"
        raise FailStopped(
            runtime.failure_reason or "pending request rollback failed"
        ) from error
    allocator._orbitkv_pending_release_work = None
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True


def retry_pending_release() -> None:
    allocator = _state._ALLOCATOR
    if getattr(allocator, "_orbitkv_free_group_state", None) != "retry_pending":
        return
    work = getattr(allocator, "_orbitkv_pending_release_work", None)
    if work is not None:
        if type(work) is not _PendingNewRequestRelease:
            runtime = _runtime()
            runtime.fail_stop(
                "SGLang retained malformed pending session release work"
            )
            allocator._orbitkv_free_group_state = "flushing"
            raise FailStopped(
                runtime.failure_reason or "pending session release was malformed"
            )
        _retry_pending_new_request_release(work, allocator)
        return
    candidates = getattr(allocator, "free_group", None)
    if not isinstance(candidates, tuple) or not candidates:
        runtime = _runtime()
        runtime.fail_stop(
            "SGLang lost its retained pending session release group"
        )
        raise FailStopped(
            runtime.failure_reason or "pending session release group was lost"
        )
    allocator._orbitkv_free_group_state = "flushing"
    try:
        session_cache.flush_release_group(candidates)
    except ReleaseRetryPending:
        allocator.free_group = candidates
        allocator._orbitkv_free_group_state = "retry_pending"
        allocator.is_not_in_free_group = True
        raise
    except Exception:
        allocator.free_group = []
        raise
    allocator.free_group = []
    allocator._orbitkv_free_group_state = "idle"
    allocator.is_not_in_free_group = True


def flush_release_group(candidates: Sequence[Any]) -> None:
    from .session_cache import flush_release_group as flush

    values = tuple(candidates)
    try:
        flush(values)
    except ReleaseRetryPending:
        _retain_pending_release_group(values)
        raise


def release_kv_cache(
    req: Any, tree_cache: Any, *, is_insert: bool
) -> None:
    from .session_cache import release_candidate

    allocator = _state._ALLOCATOR
    group_state = getattr(allocator, "_orbitkv_free_group_state", "idle")
    if group_state == "retry_pending":
        raise RuntimeError(
            "SGLang cannot release new KV while recycle is pending"
        )
    candidate = release_candidate(req, tree_cache, is_insert=bool(is_insert))
    if candidate is None:
        return
    if group_state == "collecting":
        if any(
            existing.key == candidate.key
            or existing.request_id == candidate.request_id
            or existing.req is candidate.req
            or candidate.row is not None and existing.row == candidate.row
            for existing in allocator.free_group
        ):
            runtime = _runtime()
            runtime.fail_stop(
                "SGLang duplicated a session request inside a release group"
            )
            raise FailStopped(
                runtime.failure_reason or "duplicate session release"
            )
        allocator.free_group.append(candidate)
        return
    if group_state != "idle":
        runtime = _runtime()
        runtime.fail_stop(
            "SGLang released session KV while a group was flushing"
        )
        raise FailStopped(
            runtime.failure_reason or "session release raced group flush"
        )
    flush_release_group((candidate,))


__all__ = [
    "alloc_for_decode",
    "alloc_for_extend",
    "flush_release_group",
    "get_next_batch_to_run",
    "manager_maybe_evict_swa",
    "release_kv_cache",
]
