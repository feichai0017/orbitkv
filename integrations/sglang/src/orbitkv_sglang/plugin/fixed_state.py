from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from threading import RLock
from typing import Any, Callable, Hashable, Sequence

from ..runtime import (
    FailStopped,
    ManagerError,
    StateCompletionReceipt,
    StateCopyIntent,
    StateCopyReceipt,
    StatePoolIdentity,
    StateRetirementCertificate,
    StateSlotLease,
)


@dataclass(slots=True)
class _PendingState:
    key: Hashable
    owner_id: int
    request: Any
    intent: StateCopyIntent
    snapshot: _RequestStateSnapshot
    submitted: bool = False


@dataclass(frozen=True, slots=True)
class _RequestStateSnapshot:
    row: int
    mamba_pool_idx: Any
    mamba_needs_clear: Any
    mamba_cow_src_index: Any
    mapping: int
    state_owner_id: Any
    state_transition: Any
    state_key: Any
    state_lease: Any


_MISSING = object()


@dataclass(slots=True)
class _StateEventGroup:
    event: Any
    keys: tuple[Hashable, ...]
    records: tuple[_PendingState, ...]
    completion_domain: int
    device_module: Any
    device: Any
    completion_value: int | None = None


class _OrbitKvMambaAllocator:
    """Read-only SGLang census facade; OrbitKV owns allocation and reuse."""

    def __init__(self, coordinator: FixedStateCoordinator, size: int, device: Any):
        self._coordinator = coordinator
        self.size = int(size)
        self.device = device

    @property
    def free_slots(self) -> Any:
        import torch

        # The ABI exposes an exact count but deliberately not reusable slot
        # identities. Returning no identities is conservative and prevents
        # diagnostics from becoming an allocation side channel.
        return torch.tensor(
            (),
            dtype=torch.int64,
            device=self.device,
        )

    def available_size(self) -> int:
        return self._coordinator.available_size()

    def schedulable_available_size(self) -> int:
        return self.available_size()

    def alloc(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("SGLang native Mamba allocation bypassed OrbitKV")

    def free(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("SGLang native Mamba release bypassed OrbitKV")

    def clear(self) -> None:
        self._coordinator.require_quiescent()

    def alloc_group_begin(self, _num_reqs: int) -> None:
        return None

    def alloc_group_end(self) -> None:
        return None


class FixedStateCoordinator:
    """Generation-checked request-state seam over SGLang's real MambaPool."""

    def __init__(
        self,
        state_pool: Any,
        req_to_token_pool: Any,
        *,
        failure_sink: Callable[[str], None] | None = None,
        device_module: Any | None = None,
    ) -> None:
        self._state_pool = state_pool
        self.req_to_token_pool = req_to_token_pool
        self.mamba_pool = req_to_token_pool.mamba_pool
        self.identity: StatePoolIdentity = state_pool.identity
        self._failure_sink = failure_sink
        if device_module is None:
            import torch

            device_module = torch.get_device_module(req_to_token_pool.device)
        elif callable(getattr(device_module, "get_device_module", None)):
            device_module = device_module.get_device_module(req_to_token_pool.device)
        self._device_module = device_module
        self._current_stream = lambda: self._device_module.current_stream(
            self.req_to_token_pool.device
        )
        self._lock = RLock()
        self._owners: dict[Hashable, int] = {}
        self._owner_keys: dict[int, Hashable] = {}
        self._pending: dict[Hashable, _PendingState] = {}
        self._events: list[_StateEventGroup] = []
        self._owner_completions: dict[Hashable, tuple[int, int]] = {}
        self._completion_value = 1
        self._failed: str | None = None

    @property
    def failed(self) -> str | None:
        return self._failed

    def fail_stop(self, reason: str) -> None:
        if self._failed is None:
            self._failed = reason
            if self._failure_sink is not None:
                self._failure_sink(reason)

    def poison_unobserved(
        self, records: Sequence[_PendingState], reason: str
    ) -> None:
        with self._lock:
            values = tuple(records)
            if values:
                try:
                    self._state_pool.submit_batch(
                        tuple(
                            StateCopyReceipt(
                                item.intent.transition,
                                item.intent.source,
                                item.intent.destination,
                                item.intent.byte_count,
                                observed=0,
                                written=0,
                            )
                            for item in values
                        )
                    )
                except Exception:
                    pass
            self.fail_stop(reason)

    def _fail(self, reason: str, error: BaseException | None = None) -> FailStopped:
        self.fail_stop(reason)
        failure = FailStopped(self._failed or reason)
        if error is not None:
            failure.__cause__ = error
        return failure

    def _healthy(self) -> None:
        if self._failed is not None:
            raise FailStopped("OrbitKV fixed-state adapter is poisoned: " + self._failed)

    def available_size(self) -> int:
        with self._lock:
            self._healthy()
            return int(self._state_pool.stats().free_slots)

    def require_quiescent(self) -> None:
        with self._lock:
            stats = self._state_pool.stats()
            live = (
                stats.reserved_slots,
                stats.relocating_slots,
                stats.live_slots,
                stats.retiring_slots,
                stats.quarantined_slots,
                stats.active_owners,
                stats.pending_transitions,
                stats.pending_retirements,
            )
            if any(live) or stats.free_slots != stats.identity.slot_count:
                self.fail_stop("SGLang cleared a non-quiescent fixed-state pool")
                raise FailStopped(self._failed or "non-quiescent fixed-state pool")

    def census(self) -> dict[str, Any]:
        stats = self._state_pool.stats()
        return {
            "status": "host_seam",
            "identity": {
                "engine_epoch": stats.identity.engine_epoch,
                "pool_epoch": stats.identity.pool_epoch,
                "pool_id": stats.identity.pool_id,
                "byte_count": stats.identity.byte_count,
                "slot_count": stats.identity.slot_count,
            },
            "free_slots": stats.free_slots,
            "reserved_slots": stats.reserved_slots,
            "relocating_slots": stats.relocating_slots,
            "live_slots": stats.live_slots,
            "retiring_slots": stats.retiring_slots,
            "quarantined_slots": stats.quarantined_slots,
            "active_owners": stats.active_owners,
            "pending_transitions": stats.pending_transitions,
            "pending_retirements": stats.pending_retirements,
        }

    def install_allocator_facade(self) -> None:
        native = self.req_to_token_pool.mamba_allocator
        if int(native.size) != self.identity.slot_count:
            raise RuntimeError("SGLang Mamba slot capacity differs from state pool")
        self.req_to_token_pool.mamba_allocator = _OrbitKvMambaAllocator(
            self, native.size, native.device
        )

    def owns_pool(self, pool: Any) -> bool:
        return pool is self.req_to_token_pool

    def note_prepare(self, count: int) -> None:
        from .state import _counter_add

        _counter_add("fixed_state_prepares", count)

    def free_request_rows(self, requests: Sequence[Any]) -> None:
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        for request in requests:
            if getattr(request, "mamba_pool_idx", None) is not None:
                raise ManagerError("fixed-state slot was not retired before row release")
            ReqToTokenPool.free(self.req_to_token_pool, request)

    def alloc_request_rows(self, requests: Sequence[Any]) -> list[int] | None:
        """Allocate only ReqToToken rows; state slots remain OrbitKV-owned."""

        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        return ReqToTokenPool.alloc(self.req_to_token_pool, list(requests))

    def _request_owner(self, request: Any) -> int:
        lease = getattr(request, "_orbitkv_request_lease", None)
        if (
            lease is None
            or isinstance(getattr(lease, "slot", None), bool)
            or isinstance(getattr(lease, "generation", None), bool)
            or not isinstance(getattr(lease, "slot", None), int)
            or not isinstance(getattr(lease, "generation", None), int)
            or int(getattr(lease, "engine_epoch", 0)) != self.identity.engine_epoch
            or not 0 <= lease.slot < 1 << 32
            or not 0 < lease.generation < 1 << 32
        ):
            raise ManagerError("fixed-state owner lacks a canonical request lease")
        return (lease.generation << 32) | lease.slot

    def physical_slot(self, lease: StateSlotLease) -> int:
        if not isinstance(lease, StateSlotLease):
            raise ManagerError("fixed-state slot is not a generation-checked lease")
        if (
            lease.engine_epoch != self.identity.engine_epoch
            or lease.pool_epoch != self.identity.pool_epoch
            or lease.pool_id != self.identity.pool_id
            or not 0 <= lease.slot_id < self.identity.slot_count
            or lease.generation <= 0
        ):
            raise ManagerError("fixed-state slot belongs to another pool or generation")
        return lease.slot_id + 1

    def prepare_batch(
        self,
        requests: Sequence[tuple[Hashable, Any, int]],
        *,
        replace: bool = False,
    ) -> tuple[_PendingState, ...]:
        with self._lock:
            self._healthy()
            values = tuple(requests)
            if not values or len({key for key, _req, _row in values}) != len(values):
                raise ManagerError("fixed-state prepare keys must be nonempty and unique")
            if any(key in self._pending for key, _req, _row in values):
                raise ManagerError("fixed-state request already has a pending transition")
            row_capacity = int(self.req_to_token_pool.req_to_token.shape[0])
            rows = []
            for _key, _req, row in values:
                if (
                    isinstance(row, bool)
                    or not isinstance(row, Integral)
                    or not 0 < int(row) < row_capacity
                ):
                    raise ManagerError("fixed-state request row is invalid")
                rows.append(int(row))
            if (
                len({id(req) for _key, req, _row in values}) != len(values)
                or len(set(rows)) != len(values)
            ):
                raise ManagerError("fixed-state requests and rows must be unique")
            owners = tuple(self._request_owner(req) for _key, req, _row in values)
            if len(set(owners)) != len(owners):
                raise ManagerError("canonical request leases alias state owners")
            expected = self._state_pool.current_batch(owners)
            if tuple(owner for owner, _slot in expected) != owners:
                self.fail_stop("fixed-state current lookup changed owner order")
                raise FailStopped(self._failed or "fixed-state lookup failed")
            selected = tuple(
                (value, owner, current)
                for value, owner, (_observed_owner, current) in zip(
                    values, owners, expected, strict=True
                )
                if (current is not None) == replace
            )
            for (key, req, row), owner, (_observed_owner, current) in zip(
                values, owners, expected, strict=True
            ):
                if current is None:
                    mapping = int(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            int(row)
                        ]
                    )
                    if (
                        key in self._owners
                        or hasattr(req, "_orbitkv_state_lease")
                        or getattr(req, "mamba_pool_idx", None) is not None
                        or mapping != 0
                    ):
                        raise ManagerError("fixed-state owner lost its published slot")
                    continue
                if (
                    self._owners.get(key) != owner
                    or self._owner_keys.get(owner) != key
                    or getattr(req, "_orbitkv_state_lease", None) != current
                    or self.physical_slot(current) != int(req.mamba_pool_idx)
                    or int(self.req_to_token_pool.req_index_to_mamba_index_mapping[int(row)])
                    != self.physical_slot(current)
                ):
                    raise ManagerError("fixed-state live request identity changed")
            if not selected:
                return ()
            snapshots = tuple(
                _RequestStateSnapshot(
                    row=int(row),
                    mamba_pool_idx=getattr(req, "mamba_pool_idx", None),
                    mamba_needs_clear=getattr(req, "mamba_needs_clear", False),
                    mamba_cow_src_index=getattr(req, "mamba_cow_src_index", None),
                    mapping=int(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[int(row)]
                    ),
                    state_owner_id=getattr(req, "_orbitkv_state_owner_id", _MISSING),
                    state_transition=getattr(
                        req, "_orbitkv_state_transition", _MISSING
                    ),
                    state_key=getattr(req, "_orbitkv_state_key", _MISSING),
                    state_lease=getattr(req, "_orbitkv_state_lease", _MISSING),
                )
                for (
                    _key,
                    req,
                    row,
                ), _owner, _current in selected
            )
            intents = self._state_pool.prepare_batch(
                tuple((owner, current) for _value, owner, current in selected)
            )
            selected_owners = tuple(owner for _value, owner, _current in selected)
            try:
                if (
                    len(intents) != len(selected)
                    or tuple(intent.owner_id for intent in intents) != selected_owners
                    or any(
                        intent.source != current
                        for intent, (_value, _owner, current) in zip(
                            intents, selected, strict=True
                        )
                    )
                ):
                    raise ManagerError("fixed-state prepare output identity changed")
                records = tuple(
                    _PendingState(key, owner, req, intent, snapshot)
                    for (key, req, _row), owner, intent, snapshot in zip(
                        (value for value, _owner, _current in selected),
                        selected_owners,
                        intents,
                        snapshots,
                        strict=True,
                    )
                )
                physical = tuple(
                    self.physical_slot(record.intent.destination) for record in records
                )
                if len(set(physical)) != len(physical):
                    raise ManagerError("fixed-state destinations alias")
                for (_key, req, row), record, slot in zip(
                    (value for value, _owner, _current in selected),
                    records,
                    physical,
                    strict=True,
                ):
                    req.mamba_pool_idx = self._slot_tensor(slot)
                    req.mamba_needs_clear = record.intent.source is None
                    req.mamba_cow_src_index = (
                        None
                        if record.intent.source is None
                        else self._slot_tensor(self.physical_slot(record.intent.source)).unsqueeze(0)
                    )
                    req._orbitkv_state_owner_id = record.owner_id
                    req._orbitkv_state_transition = record.intent.transition
                    req._orbitkv_state_key = record.key
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        int(row)
                    ] = slot
                    self._owners[record.key] = record.owner_id
                    self._owner_keys[record.owner_id] = record.key
                    self._pending[record.key] = record
            except Exception as error:
                failures = []
                try:
                    self._state_pool.abort_batch(tuple(item.transition for item in intents))
                except Exception as abort_error:
                    failures.append(f"native abort: {abort_error}")
                for index, ((key, req, _row), owner, _current) in enumerate(
                    selected
                ):
                    snapshot = snapshots[index]
                    try:
                        pending = self._pending.get(key)
                        if pending is not None:
                            self._restore_pending(pending)
                        else:
                            self._restore_snapshot(
                                key, owner, req, snapshot, initial=_current is None
                            )
                    except Exception as restore_error:
                        failures.append(f"mirror restore: {restore_error}")
                if failures:
                    self.fail_stop(
                        "fixed-state prepare rollback failed: " + "; ".join(failures)
                    )
                    raise FailStopped(self._failed or "fixed-state rollback failed") from error
                raise
            return records

    def prepare_for_allocated_rows(
        self, requests: Sequence[Any], rows: Sequence[int]
    ) -> tuple[_PendingState, ...]:
        values = tuple(requests)
        row_values = tuple(int(row) for row in rows)
        if len(values) != len(row_values):
            raise ManagerError("fixed-state request-row cardinality changed")
        return self.prepare_batch(
            tuple(
                (_request_key(req), req, row)
                for req, row in zip(values, row_values, strict=True)
            )
        )

    def prepare_replacement_batch(
        self, requests: Sequence[Any], rows: Sequence[int]
    ) -> tuple[_PendingState, ...]:
        values = tuple(requests)
        row_values = tuple(int(row) for row in rows)
        if len(values) != len(row_values):
            raise ManagerError("fixed-state replacement cardinality changed")
        return self.prepare_batch(
            tuple(
                (_request_key(req), req, row)
                for req, row in zip(values, row_values, strict=True)
            ),
            replace=True,
        )

    def _slot_tensor(self, value: int) -> Any:
        import torch

        return torch.tensor(value, dtype=torch.int64, device=self.req_to_token_pool.device)

    def mirror_failed(self, records: Sequence[_PendingState], error: BaseException) -> None:
        with self._lock:
            values = tuple(records)
            try:
                if values:
                    self._state_pool.abort_batch(
                        tuple(item.intent.transition for item in values)
                    )
            except Exception as abort_error:
                raise self._fail(
                    f"fixed-state mirror rollback became uncertain: {abort_error}",
                    abort_error,
                ) from error
            for record in values:
                self._discard_pending(record)
            self.fail_stop(f"fixed-state mirror publication failed: {error}")

    def mirror_failed_after_external_authorization(
        self, records: Sequence[_PendingState], error: BaseException
    ) -> None:
        """Contain fixed state when token destinations may already mutate."""

        with self._lock:
            values = tuple(records)
            if values:
                self.poison_unobserved(
                    values,
                    f"fixed-state mirror failed after external authorization: {error}",
                )

    def abort_batch(self, records: Sequence[_PendingState]) -> None:
        with self._lock:
            values = tuple(records)
            self._state_pool.abort_batch(tuple(item.intent.transition for item in values))
            for record in values:
                self._discard_pending(record)

    def rollback_new_requests(self, requests: Sequence[Any]) -> None:
        with self._lock:
            records = tuple(
                self._pending[_request_key(req)]
                for req in requests
                if _request_key(req) in self._pending
            )
            if not records:
                return
            try:
                self._state_pool.abort_batch(
                    tuple(item.intent.transition for item in records)
                )
                for record in records:
                    self._discard_pending(record)
            except Exception as error:
                raise self._fail(
                    f"fixed-state request rollback became uncertain: {error}", error
                )

    def preflight_deferred(
        self, model_runner: Any, forward_batch: Any
    ) -> tuple[_PendingState, ...]:
        with self._lock:
            self._healthy()
            records = tuple(
                getattr(forward_batch, "_orbitkv_state_records", ())
            )
            mode = forward_batch.forward_mode
            inactive = (
                bool(model_runner.is_draft_worker)
                or not bool(mode.is_extend())
                or bool(mode.is_target_verify())
                or bool(mode.is_draft_extend_v2())
            )
            if not records:
                # SGLang carries the previous extend batch's deferred Mamba
                # tensors into decode ForwardBatch objects, but its native
                # helper returns before observing them outside this exact
                # target-extend domain. Match that execution boundary while
                # retaining the ownership check wherever an operation could
                # actually execute.
                if inactive:
                    return ()
                if any(
                    getattr(forward_batch, name, None) is not None
                    for name in (
                        "mamba_clear_indices",
                        "mamba_cow_src_indices",
                        "mamba_cow_dst_indices",
                    )
                ):
                    raise ManagerError("unowned deferred Mamba operation")
                return ()
            if any(
                not isinstance(record, _PendingState)
                or self._pending.get(record.key) is not record
                for record in records
            ):
                self.fail_stop("forward batch lost fixed-state transitions")
                raise FailStopped(self._failed or "fixed-state transition missing")
            pending = records
            if model_runner.req_to_token_pool is not self.req_to_token_pool:
                raise ManagerError("model runner references a foreign fixed-state pool")
            if inactive:
                raise ManagerError("fixed-state transition reached an unsupported forward mode")
            carried_keys = tuple(
                getattr(forward_batch, "_orbitkv_state_forward_keys", ())
            )
            carried_rows = tuple(
                getattr(forward_batch, "_orbitkv_state_forward_rows", ())
            )
            raw_rids = getattr(forward_batch, "rids", None)
            if (
                not carried_keys
                or len(carried_keys) != len(carried_rows)
                or len(set(carried_keys)) != len(carried_keys)
                or raw_rids is None
                or tuple(_request_key_from_rid(value) for value in raw_rids)
                != carried_keys
            ):
                raise ManagerError("forward request identity metadata changed")
            cursor = 0
            for record in pending:
                while cursor < len(carried_keys) and carried_keys[cursor] != record.key:
                    cursor += 1
                if cursor == len(carried_keys):
                    raise ManagerError(
                        "fixed-state transitions are not an ordered forward subset"
                    )
                if (
                    carried_rows[cursor] != record.snapshot.row
                    or _request_key(record.request) != record.key
                ):
                    raise ManagerError(
                        "fixed-state transition request or row identity changed"
                    )
                cursor += 1
                row = getattr(record.request, "req_pool_idx", None)
                destination = self.physical_slot(record.intent.destination)
                if (
                    isinstance(row, bool)
                    or not isinstance(row, Integral)
                    or int(row) != record.snapshot.row
                    or int(record.request.mamba_pool_idx) != destination
                    or int(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            record.snapshot.row
                        ]
                    )
                    != destination
                ):
                    raise ManagerError(
                        "fixed-state forward row or destination identity changed"
                    )
            initial = tuple(item for item in pending if item.intent.source is None)
            replacements = tuple(
                item for item in pending if item.intent.source is not None
            )
            expected_clear = self._slot_tensor_batch(
                tuple(self.physical_slot(item.intent.destination) for item in initial)
            )
            expected_src = self._slot_tensor_batch(
                tuple(
                    self.physical_slot(item.intent.source)
                    for item in replacements
                    if item.intent.source is not None
                )
            )
            expected_dst = self._slot_tensor_batch(
                tuple(self.physical_slot(item.intent.destination) for item in replacements)
            )
            if not self._same_indices(
                getattr(forward_batch, "mamba_clear_indices", None), expected_clear
            ) or not self._same_indices(
                getattr(forward_batch, "mamba_cow_src_indices", None), expected_src
            ) or not self._same_indices(
                getattr(forward_batch, "mamba_cow_dst_indices", None), expected_dst
            ):
                raise ManagerError("SGLang deferred Mamba operation differs from state intents")
            return pending

    @staticmethod
    def _same_indices(actual: Any, expected: Any) -> bool:
        if int(expected.numel()) == 0:
            return actual is None or int(actual.numel()) == 0
        return bool(
            actual is not None
            and int(actual.numel()) == int(expected.numel())
            and (actual.to(dtype=expected.dtype) == expected).all()
        )

    def accept_deferred(
        self, records: Sequence[_PendingState], *, clear_count: int, copy_count: int
    ) -> None:
        with self._lock:
            values = tuple(records)
            self._state_pool.submit_batch(
                tuple(
                    StateCopyReceipt(
                        record.intent.transition,
                        record.intent.source,
                        record.intent.destination,
                        record.intent.byte_count,
                    )
                    for record in values
                )
            )
            for record in values:
                record.submitted = True
            from .state import _counter_add

            _counter_add("fixed_state_clears", clear_count)
            _counter_add("fixed_state_copies", copy_count)

    def deferred_failed(
        self, records: Sequence[_PendingState], error: BaseException
    ) -> None:
        with self._lock:
            values = tuple(records)
            try:
                self._state_pool.submit_batch(
                    tuple(
                        StateCopyReceipt(
                            record.intent.transition,
                            record.intent.source,
                            record.intent.destination,
                            record.intent.byte_count,
                            observed=0,
                            written=0,
                        )
                        for record in values
                    )
                )
            except Exception:
                pass
            self.fail_stop(f"fixed-state clear/copy became uncertain: {error}")

    def records_for_schedule_batch(self, batch: Any) -> tuple[_PendingState, ...]:
        values = tuple(getattr(batch, "_orbitkv_state_records", ()))
        return tuple(item for item in values if item.submitted)

    def register_event(
        self,
        keys: Sequence[Hashable],
        records: Sequence[_PendingState],
        event: Any,
        completion_domain: int,
        device_module: Any,
        device: Any,
        completion_value: int | None = None,
    ) -> None:
        with self._lock:
            self._healthy()
            key_values = tuple(keys)
            values = tuple(records)
            if (
                not key_values
                or len(set(key_values)) != len(key_values)
                or any(key not in self._owners for key in key_values)
                or any(item.key not in key_values for item in values)
                or any(
                    not item.submitted or self._pending.get(item.key) is not item
                    for item in values
                )
            ):
                raise ManagerError("fixed-state event records are invalid")
            if completion_value is not None and (
                isinstance(completion_value, bool)
                or not isinstance(completion_value, int)
                or completion_value <= 0
                or completion_value >= 1 << 64
                or completion_value < self._completion_value
                or any(
                    group.completion_domain == int(completion_domain)
                    and group.completion_value is not None
                    and group.completion_value >= completion_value
                    for group in self._events
                )
            ):
                raise ManagerError(
                    "fixed-state external completion point did not advance"
                )
            self._events.append(
                _StateEventGroup(
                    event,
                    key_values,
                    values,
                    int(completion_domain),
                    device_module,
                    device,
                    completion_value,
                )
            )
            from .state import _counter_add

            _counter_add("fixed_state_events")

    def register_external_event(
        self,
        keys: Sequence[Hashable],
        records: Sequence[_PendingState],
        event: Any,
        fence: Any,
        device_module: Any,
        device: Any,
        *,
        adapter: Any,
    ) -> None:
        from orbitkv_runtime import CompletionFence

        if not isinstance(fence, CompletionFence):
            raise ManagerError(
                "fixed-state external completion requires a CompletionFence"
            )
        if (
            getattr(adapter, "adapter_id", None) != fence.adapter_id
            or not callable(getattr(adapter, "event_for", None))
            or adapter.event_for(fence) is not event
        ):
            raise ManagerError(
                "fixed-state completion lacks issuing-adapter evidence"
            )
        if fence.engine_epoch != self.identity.engine_epoch:
            raise ManagerError(
                "fixed-state external completion has another engine epoch"
            )
        self.register_event(
            keys,
            records,
            event,
            fence.completion_domain,
            device_module,
            device,
            fence.completion_value,
        )

    def forward_failed(
        self, records: Sequence[_PendingState], error: BaseException
    ) -> None:
        with self._lock:
            if records:
                self.fail_stop(f"fixed-state forward became uncertain: {error}")

    def event_registration_failed(
        self, records: Sequence[_PendingState], error: BaseException
    ) -> None:
        with self._lock:
            if records:
                self.fail_stop(
                    f"fixed-state CUDA event registration became uncertain: {error}"
                )

    def pre_forward_failed(self, error: BaseException) -> None:
        with self._lock:
            records = tuple(self._pending.values())
            if records:
                submitted = tuple(item for item in records if item.submitted)
                prepared = tuple(item for item in records if not item.submitted)
                if prepared:
                    try:
                        self._state_pool.abort_batch(
                            tuple(item.intent.transition for item in prepared)
                        )
                        for record in prepared:
                            self._discard_pending(record)
                    except Exception as abort_error:
                        self.fail_stop(
                            "fixed-state pre-forward rollback became uncertain: "
                            + str(abort_error)
                        )
                if submitted:
                    self.fail_stop(
                        f"fixed-state pre-forward scheduling failed after submit: {error}"
                    )

    def poll(self) -> None:
        with self._lock:
            self._healthy()
            for group in tuple(self._events):
                try:
                    ready = bool(group.event.query())
                except Exception as error:
                    self.fail_stop(f"fixed-state CUDA event query failed: {error}")
                    raise FailStopped(self._failed or "fixed-state event query failed") from error
                if ready:
                    self._complete_group(group)

    def wait_keys(self, keys: Sequence[Hashable]) -> None:
        with self._lock:
            self._healthy()
            requested = set(keys)
            for group in tuple(self._events):
                if not any(key in requested for key in group.keys):
                    continue
                try:
                    group.event.synchronize()
                except Exception as error:
                    self.fail_stop(f"fixed-state CUDA event wait failed: {error}")
                    raise FailStopped(self._failed or "fixed-state event wait failed") from error
                self._complete_group(group)

    def _complete_group(self, group: _StateEventGroup) -> None:
        if group not in self._events:
            return
        completion_value = (
            self._completion_value
            if group.completion_value is None
            else group.completion_value
        )
        receipt = StateCompletionReceipt(
            self.identity.engine_epoch,
            group.completion_domain,
            completion_value,
        )
        try:
            publications = (
                self._state_pool.complete_batch(
                    receipt, tuple(item.intent.transition for item in group.records)
                )
                if group.records
                else ()
            )
            if len(publications) != len(group.records):
                raise ManagerError("fixed-state publication cardinality changed")
            retirements = []
            for record, publication in zip(group.records, publications, strict=True):
                if (
                    publication.owner_id != record.owner_id
                    or publication.slot != record.intent.destination
                ):
                    raise ManagerError("fixed-state publication identity changed")
                record.request._orbitkv_state_lease = publication.slot
                record.request._orbitkv_state_transition = None
                if publication.retirement is not None:
                    retirements.append(publication.retirement)
                self._pending.pop(record.key)
            if retirements:
                self._clear_and_ack(
                    tuple(retirements),
                    group.completion_domain,
                    group.device_module,
                    group.device,
                )
            for key in group.keys:
                self._owner_completions[key] = (
                    group.completion_domain, completion_value
                )
            self._completion_value = max(
                self._completion_value, completion_value + 1
            )
            self._events.remove(group)
        except Exception as error:
            self.fail_stop(f"fixed-state publication became uncertain: {error}")
            raise FailStopped(self._failed or "fixed-state publication failed") from error

    def preflight_retire(
        self, requests: Sequence[tuple[Hashable, Any]]
    ) -> tuple[tuple[int, StateSlotLease], ...]:
        with self._lock:
            self._healthy()
            values = tuple(requests)
            if (
                not values
                or len({key for key, _req in values}) != len(values)
                or len({id(req) for _key, req in values}) != len(values)
            ):
                raise ManagerError("fixed-state release requests must be unique")
            items = []
            for key, req in values:
                owner = self._owners.get(key)
                lease = getattr(req, "_orbitkv_state_lease", None)
                if (
                    owner is None
                    or self._owner_keys.get(owner) != key
                    or key in self._pending
                    or not isinstance(lease, StateSlotLease)
                    or self.physical_slot(lease) != int(req.mamba_pool_idx)
                    or getattr(req, "_orbitkv_state_key", None) != key
                ):
                    raise ManagerError("fixed-state release identity changed")
                items.append((owner, lease))
            observed = self._state_pool.current_batch(
                tuple(owner for owner, _lease in items)
            )
            if observed != tuple(
                (owner, lease) for owner, lease in items
            ):
                raise ManagerError("fixed-state release differs from native authority")
            return tuple(items)

    def retire_batch(
        self,
        requests: Sequence[tuple[Hashable, Any]],
        *,
        items: Sequence[tuple[int, StateSlotLease]] | None = None,
    ) -> None:
        with self._lock:
            self._healthy()
            values = tuple(requests)
            self.wait_keys(tuple(key for key, _req in values))
            expected_items = self.preflight_retire(values)
            if items is not None and tuple(items) != expected_items:
                raise ManagerError("fixed-state release preflight changed")
            frontiers = tuple(self._owner_completions.get(key) for key, _req in values)
            if any(frontier is None for frontier in frontiers):
                raise ManagerError("fixed-state release lacks a completed forward")
            domains = {frontier[0] for frontier in frontiers if frontier is not None}
            if len(domains) != 1:
                raise ManagerError("fixed-state release spans completion domains")
            domain = next(iter(domains))
            completion_value = max(
                frontier[1] for frontier in frontiers if frontier is not None
            )
            receipt = StateCompletionReceipt(
                self.identity.engine_epoch, domain, completion_value
            )
            certificates = self._state_pool.retire_owners_batch(receipt, expected_items)
            from .state import _counter_add

            _counter_add("fixed_state_retirements", len(certificates))
            self._clear_and_ack(
                tuple(certificates),
                domain,
                self._device_module,
                self.req_to_token_pool.device,
            )
            for key, req in values:
                row = int(req.req_pool_idx)
                self.req_to_token_pool.req_index_to_mamba_index_mapping[row] = 0
                req.mamba_pool_idx = None
                for name in (
                    "_orbitkv_state_owner_id",
                    "_orbitkv_state_transition",
                    "_orbitkv_state_lease",
                    "_orbitkv_state_key",
                ):
                    if hasattr(req, name):
                        delattr(req, name)
                owner = self._owners.pop(key)
                self._owner_keys.pop(owner)
                self._owner_completions.pop(key)

    def close(self) -> None:
        with self._lock:
            error: BaseException | None = None
            try:
                stats = self._state_pool.stats()
                if (
                    self._failed is None
                    and (
                        stats.free_slots != stats.identity.slot_count
                        or stats.active_owners
                        or stats.pending_transitions
                        or stats.pending_retirements
                        or stats.quarantined_slots
                    )
                ):
                    message = "shutdown encountered live fixed-state ownership"
                    self.fail_stop(message)
                    error = FailStopped(message)
            except Exception as stats_error:
                self.fail_stop(
                    f"fixed-state shutdown census became uncertain: {stats_error}"
                )
                error = stats_error
            try:
                self._state_pool.close()
            except Exception as close_error:
                if error is not None:
                    close_error.add_note(
                        f"fixed-state pre-close failure: {error!r}"
                    )
                raise
            if error is not None:
                raise error

    def shutdown(self) -> None:
        from . import state as plugin_state

        try:
            self.close()
        finally:
            if plugin_state._FIXED_STATE is self:
                plugin_state._FIXED_STATE = None

    def _clear_and_ack(
        self,
        certificates: tuple[StateRetirementCertificate, ...],
        completion_domain: int,
        device_module: Any,
        device: Any,
    ) -> None:
        if not certificates:
            return
        try:
            indices = self._slot_tensor_batch(
                tuple(self.physical_slot(item.slot) for item in certificates)
            )
            self.mamba_pool.clear_slots(indices)
            event = device_module.Event()
            stream = (
                self._current_stream()
                if device_module is self._device_module
                else device_module.current_stream(device)
            )
            event.record(stream=stream)
            event.synchronize()
            self._state_pool.acknowledge_batch(certificates)
            from .state import _counter_add

            _counter_add("fixed_state_acks", len(certificates))
        except Exception as error:
            self.fail_stop(
                f"fixed-state retirement clear/ACK failed in domain {completion_domain}: {error}"
            )
            raise FailStopped(self._failed or "fixed-state retirement failed") from error

    def _slot_tensor_batch(self, values: tuple[int, ...]) -> Any:
        import torch

        return torch.tensor(values, dtype=torch.int64, device=self.req_to_token_pool.device)

    def _discard_pending(self, record: _PendingState) -> None:
        self._restore_pending(record)

    def _restore_pending(self, record: _PendingState) -> None:
        current = self._pending.get(record.key)
        if current is not None and current is not record:
            raise ManagerError("fixed-state pending transition identity changed")
        self._pending.pop(record.key, None)
        self._restore_snapshot(
            record.key,
            record.owner_id,
            record.request,
            record.snapshot,
            initial=record.intent.source is None,
        )

    def _restore_snapshot(
        self,
        key: Hashable,
        owner_id: int,
        req: Any,
        snapshot: _RequestStateSnapshot,
        *,
        initial: bool,
    ) -> None:
        self.req_to_token_pool.req_index_to_mamba_index_mapping[
            snapshot.row
        ] = snapshot.mapping
        req.mamba_pool_idx = snapshot.mamba_pool_idx
        req.mamba_needs_clear = snapshot.mamba_needs_clear
        req.mamba_cow_src_index = snapshot.mamba_cow_src_index
        for name, value in (
            ("_orbitkv_state_owner_id", snapshot.state_owner_id),
            ("_orbitkv_state_transition", snapshot.state_transition),
            ("_orbitkv_state_key", snapshot.state_key),
            ("_orbitkv_state_lease", snapshot.state_lease),
        ):
            if value is _MISSING:
                if hasattr(req, name):
                    delattr(req, name)
            else:
                setattr(req, name, value)
        if initial:
            self._owners.pop(key, None)
            self._owner_keys.pop(owner_id, None)
        else:
            self._owners[key] = owner_id
            self._owner_keys[owner_id] = key

    def stats(self) -> Any:
        return self._state_pool.stats()


def _execute_fixed_state_deferred(
    original_fn: Callable[..., Any], model_runner: Any, forward_batch: Any
) -> Any:
    from . import state as plugin_state

    coordinator = plugin_state._FIXED_STATE
    if coordinator is None:
        return original_fn(model_runner, forward_batch)
    try:
        records = coordinator.preflight_deferred(model_runner, forward_batch)
        clear_count = (
            0
            if getattr(forward_batch, "mamba_clear_indices", None) is None
            else int(forward_batch.mamba_clear_indices.numel())
        )
        copy_count = (
            0
            if getattr(forward_batch, "mamba_cow_src_indices", None) is None
            else int(forward_batch.mamba_cow_src_indices.numel())
        )
        result = original_fn(model_runner, forward_batch)
        if records:
            coordinator.accept_deferred(
                records, clear_count=clear_count, copy_count=copy_count
            )
        return result
    except Exception as error:
        records = locals().get("records", ())
        if records:
            coordinator.deferred_failed(records, error)
        else:
            coordinator.fail_stop(f"fixed-state deferred preflight failed: {error}")
        if isinstance(error, FailStopped):
            raise
        raise FailStopped(
            coordinator.failed or "fixed-state deferred operation failed"
        ) from error


def _request_key(req: Any) -> tuple[str, str | bytes | int]:
    from .state import _request_key as key

    return key(req)


def _request_key_from_rid(value: Any) -> tuple[str, str | bytes | int]:
    if isinstance(value, bool):
        raise ManagerError("forward request rid must not be boolean")
    if isinstance(value, str):
        if not value:
            raise ManagerError("forward request rid must not be empty")
        return ("str", value)
    if isinstance(value, bytes):
        if not value:
            raise ManagerError("forward request rid must not be empty")
        return ("bytes", value)
    if isinstance(value, int) and value >= 0:
        return ("int", value)
    raise ManagerError("forward request rid is not stable")


def _fixed_state_alloc(
    original_fn: Callable[..., Any], pool: Any, requests: Sequence[Any]
) -> list[int] | None:
    from . import state as plugin_state

    coordinator = plugin_state._FIXED_STATE
    if coordinator is None:
        return original_fn(pool, requests)
    if not coordinator.owns_pool(pool):
        raise RuntimeError("OrbitKV fixed-state allocator received a foreign pool")
    return coordinator.alloc_request_rows(requests)


def _fixed_state_free(
    original_fn: Callable[..., Any],
    pool: Any,
    request: Any,
    mamba_ping_pong_track_buffer_to_keep: Any = None,
) -> Any:
    from . import state as plugin_state

    coordinator = plugin_state._FIXED_STATE
    if coordinator is None:
        return original_fn(
            pool, request, mamba_ping_pong_track_buffer_to_keep
        )
    if not coordinator.owns_pool(pool):
        raise RuntimeError("OrbitKV fixed-state release received a foreign pool")
    raise RuntimeError(
        "SGLang native Mamba release bypassed OrbitKV retirement/ACK"
    )


def _attach_fixed_state_forward_batch(
    result: Any, _cls: Any, schedule_batch: Any, _model_runner: Any, **_kwargs: Any
) -> Any:
    from . import state as plugin_state

    if plugin_state._FIXED_STATE is not None:
        result._orbitkv_state_records = tuple(
            getattr(schedule_batch, "_orbitkv_state_records", ())
        )
        result._orbitkv_state_forward_keys = tuple(
            _request_key(req) for req in schedule_batch.reqs
        )
        rows = tuple(getattr(req, "req_pool_idx", None) for req in schedule_batch.reqs)
        if any(isinstance(row, bool) or not isinstance(row, Integral) for row in rows):
            raise ManagerError("fixed-state ForwardBatch has an invalid request row")
        result._orbitkv_state_forward_rows = tuple(int(row) for row in rows)
    return result


def _fixed_state_pool_clear(original_fn: Callable[..., Any], pool: Any) -> Any:
    from . import state as plugin_state

    coordinator = plugin_state._FIXED_STATE
    if coordinator is None:
        return original_fn(pool)
    if not coordinator.owns_pool(pool):
        raise RuntimeError("OrbitKV fixed-state clear received a foreign pool")
    coordinator.require_quiescent()
    # ReqToToken rows and their device mirror are non-authoritative, but the
    # original method would also call the forbidden native Mamba free-list clear.
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    ReqToTokenPool.clear(pool)
    pool.req_index_to_mamba_index_mapping.zero_()
    return None


__all__ = [
    "FixedStateCoordinator",
    "_execute_fixed_state_deferred",
    "_attach_fixed_state_forward_batch",
    "_fixed_state_alloc",
    "_fixed_state_free",
    "_fixed_state_pool_clear",
]
