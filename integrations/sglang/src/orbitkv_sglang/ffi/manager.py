from __future__ import annotations

import ctypes
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Sequence

from orbitkv_sglang.runtime import (
    ArenaIdentity,
    ArenaRegistration,
    ArenaStats,
    AttachedPrefix,
    BackendBindReceipt,
    BackendCopyReceipt,
    BackendUnobservedReceipt,
    BatchCompletionReceipt,
    ClassLowering,
    CompletionBatch,
    CopyIntent,
    DetachedBinding,
    EvictedPrefix,
    FailStopped,
    ForkedRequest,
    ManagerCreateSettings,
    ManagerError,
    ManagerFactoryProtocol,
    ManagerProtocol,
    ManagerStats,
    MaterializedRequestView,
    PageLease,
    PrefixAttachItem,
    PrefixEvictionBatch,
    PrefixLease,
    PrefixLookupHint,
    PrefixPublishItem,
    PrefixPublishRelease,
    PrefixPublishReleaseBatch,
    PrefixSemanticKey,
    PreparedStep,
    PublishedPrefix,
    ReclamationCertificate,
    ReclamationLease,
    ReclamationReceipt,
    ReleaseBatchCompletion,
    ReleaseBatchItem,
    ReleaseCompletion,
    RequestForkItem,
    RequestLease,
    RequestView,
    RetryableConflict,
    SnapshotLease,
    SnapshotPage,
    StepCompletion,
    StepLease,
    SubmissionLease,
    SubmittedStep,
    TailAction,
    WriteIntent,
)

from . import layouts as L
from .library import (
    ERROR_BUFFER_BYTES,
    STATUS_BUFFER_TOO_SMALL,
    STATUS_FAIL_STOPPED,
    STATUS_INVALID_ARGUMENT,
    STATUS_MANAGER_ERROR,
    STATUS_OK,
    STATUS_PANIC,
    STATUS_RETRYABLE_CONFLICT,
    LoadedLibrary,
)
from .workspace import HotBounds, HotWorkspace, array, cold_materialization, cold_reclamation


def _uint(name: str, value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << bits:
        raise ManagerError(f"{name} is outside uint{bits}_t")
    return value


def _discard_created_handle(loaded: LoadedLibrary, handle: ctypes.c_void_p) -> None:
    """Best-effort consume a create result that cannot be handed to a caller.

    Creation has exclusive ownership of any non-null output handle.  Even an
    unusable or lost-return result must therefore attempt the ABI6 consuming
    destroy operation; retaining the pointer would leak authority, while a
    second attempt would risk using an already-consumed pointer.
    """

    if not handle.value:
        return
    error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
    try:
        loaded.cdll.orbitkv_manager_destroy(handle, error, len(error))
    except BaseException:
        pass
    finally:
        handle.value = None


def _lease_to_c(value: Any) -> Any:
    layouts = (
        (RequestLease, L.RequestLeaseLayout),
        (SnapshotLease, L.SnapshotLeaseLayout),
        (StepLease, L.StepLeaseLayout),
        (SubmissionLease, L.SubmissionLeaseLayout),
        (ReclamationLease, L.ReclamationLeaseLayout),
        (PrefixLease, L.PrefixLeaseLayout),
    )
    layout = next((item for kind, item in layouts if isinstance(value, kind)), None)
    if layout is None:
        raise ManagerError("value is not an ABI6 lease DTO")
    return layout(
        _uint("lease engine epoch", value.engine_epoch, 64),
        _uint("lease slot", value.slot, 32),
        _uint("lease generation", value.generation, 32),
    )


def _request(value: Any) -> RequestLease:
    return RequestLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _snapshot(value: Any) -> SnapshotLease:
    return SnapshotLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _step(value: Any) -> StepLease:
    return StepLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _submission(value: Any) -> SubmissionLease:
    return SubmissionLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _reclamation(value: Any) -> ReclamationLease:
    return ReclamationLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _prefix(value: Any) -> PrefixLease:
    return PrefixLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def _page_to_c(value: PageLease) -> L.PageLeaseLayout:
    return L.PageLeaseLayout(
        _uint("page engine epoch", value.engine_epoch, 64),
        _uint("page pool epoch", value.pool_epoch, 64),
        _uint("page generation", value.generation, 64),
        _uint("page id", value.page_id, 32),
        _uint("page pool id", value.pool_id, 32),
    )


def _page(value: L.PageLeaseLayout) -> PageLease:
    return PageLease(
        int(value.engine_epoch),
        int(value.pool_epoch),
        int(value.generation),
        int(value.page_id),
        int(value.pool_id),
    )


def _key_to_c(value: PrefixSemanticKey) -> L.PrefixKeyLayout:
    if not isinstance(value.namespace, bytes) or len(value.namespace) != 32:
        raise ManagerError("prefix namespace must contain exactly 32 bytes")
    if not isinstance(value.digest, bytes) or len(value.digest) != 32:
        raise ManagerError("prefix digest must contain exactly 32 bytes")
    result = L.PrefixKeyLayout()
    result.namespace_bytes[:] = value.namespace
    result.digest[:] = value.digest
    result.boundary = _uint("prefix boundary", value.boundary, 64)
    return result


def _key(value: L.PrefixKeyLayout) -> PrefixSemanticKey:
    return PrefixSemanticKey(bytes(value.namespace_bytes), bytes(value.digest), int(value.boundary))


def _view(value: L.RequestViewLayout) -> RequestView:
    if int(value.reserved) != 0:
        raise ManagerError("request view reserved field is nonzero")
    return RequestView(
        _request(value.request),
        _snapshot(value.snapshot),
        int(value.view_version),
        int(value.boundary),
        int(value.resident_count),
    )


def _snapshot_page(value: L.SnapshotPageLayout) -> SnapshotPage:
    if int(value.reserved) != 0:
        raise ManagerError("snapshot page reserved field is nonzero")
    return SnapshotPage(
        _page(value.page),
        int(value.logical_ordinal),
        int(value.temporal_cell_index),
        int(value.temporal_cycle),
        int(value.backend_index),
        int(value.class_id),
        int(value.backend_domain),
        int(value.valid_token_count),
        int(value.visible_token_offset),
        int(value.visible_token_count),
    )


def _tail(value: L.TailActionLayout) -> TailAction:
    return TailAction(
        int(value.class_id),
        int(value.kind),
        int(value.valid_token_count),
        int(value.logical_ordinal),
        _page(value.source),
        _page(value.destination),
        int(value.reserved),
    )


def _copy_intent(value: L.CopyIntentLayout) -> CopyIntent:
    return CopyIntent(
        class_id=int(value.class_id),
        backend_domain=int(value.backend_domain),
        token_count=int(value.token_count),
        source_token_offset=int(value.source_token_offset),
        destination_token_offset=int(value.destination_token_offset),
        source=_page(value.source),
        destination=_page(value.destination),
        source_backend_index=int(value.source_backend_index),
        destination_backend_index=int(value.destination_backend_index),
        reserved=int(value.reserved),
    )


def _detached(value: L.DetachedBindingLayout) -> DetachedBinding:
    return DetachedBinding(
        _page(value.old),
        _page(value.replacement),
        int(value.logical_ordinal),
        int(value.old_backend_index),
        int(value.replacement_backend_index),
        int(value.token_begin),
        int(value.token_end_exclusive),
        int(value.class_id),
        int(value.backend_domain),
        int(value.action),
        int(value.reason),
        int(value.reserved),
    )


def _certificate(value: L.ReclamationCertificateLayout) -> ReclamationCertificate:
    if int(value.reserved32) != 0:
        raise ManagerError("reclamation certificate reserved field is nonzero")
    return ReclamationCertificate(
        _reclamation(value.reclamation),
        _page(value.page),
        int(value.class_id),
        int(value.backend_domain),
        int(value.logical_ordinal),
        int(value.backend_index),
        int(value.token_begin),
        int(value.token_end_exclusive),
        int(value.completion_domain),
        int(value.completion_value),
    )


def _published(value: L.PublishedPrefixLayout) -> PublishedPrefix:
    if int(value.reserved) != 0:
        raise ManagerError("published prefix reserved field is nonzero")
    return PublishedPrefix(_prefix(value.prefix), _key(value.key), int(value.resident_count))


class CtypesManager(ManagerProtocol):
    def __init__(
        self,
        loaded: LoadedLibrary,
        handle: ctypes.c_void_p,
        registrations: Sequence[ArenaRegistration],
        page_tokens: int,
        settings: ManagerCreateSettings,
    ):
        self._loaded = loaded
        self._library = loaded.cdll
        self._handle = handle
        self._lock = RLock()
        self._poisoned: str | None = None
        self._destroy_attempted = False
        self._error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
        self._arena_count = len(registrations)
        self._physical_pages = sum(item.page_count for item in registrations)
        self._request_capacity = int(settings.maximum_requests)
        self._operation_capacity = min(
            self._request_capacity, int(settings.maximum_operations)
        )
        self._prefix_capacity = int(settings.maximum_prefixes)
        bounds = HotBounds.compile(
            maximum_batch=self._operation_capacity,
            class_count=self._arena_count,
            maximum_step_tokens=int(settings.maximum_step_tokens),
            page_tokens=page_tokens,
            physical_pages=self._physical_pages,
        )
        self._hot = HotWorkspace(bounds)
        self._request_views = array(L.RequestViewLayout, self._request_capacity)
        self._request_leases = array(L.RequestLeaseLayout, self._request_capacity)
        self._abort_receipts = array(L.UnobservedReceiptLayout, self._operation_capacity)
        self._step_leases = array(L.StepLeaseLayout, self._operation_capacity)
        self._submission_leases = array(L.SubmissionLeaseLayout, self._operation_capacity)
        self._reclamation_receipts = array(L.ReclamationReceiptLayout, self._physical_pages)
        self._counters: dict[str, int] = {
            "request_acquire_batch_calls": 0,
            "request_fork_batch_calls": 0,
            "prepare_batch_calls": 0,
            "submit_batch_calls": 0,
            "complete_batch_calls": 0,
            "abort_steps_batch_calls": 0,
            "quarantine_steps_batch_calls": 0,
            "quarantine_submissions_batch_calls": 0,
            "release_batch_calls": 0,
            "acknowledge_reclamations_batch_calls": 0,
            "recycle_requests_batch_calls": 0,
            "prefix_lookup_batch_calls": 0,
            "prefix_attach_batch_calls": 0,
            "prefix_publish_batch_calls": 0,
            "prefix_publish_release_batch_calls": 0,
            "prefix_evict_batch_calls": 0,
            "prefix_recycle_batch_calls": 0,
            "buffer_too_small_preflights": 0,
            "retryable_conflicts": 0,
            "fail_stops": 0,
            # Construction allocates the reusable workspace.  These counters
            # describe allocations/capacity work performed by lifecycle calls.
            "hot_workspace_allocations": 0,
            "capacity_memset_bytes": 0,
            "root_entries_crossed": 0,
            "cold_workspace_allocations": 0,
            "materialized_page_objects": 0,
        }
        self._arenas = self._load_arenas(registrations, page_tokens)
        self._arenas_by_class = {item.class_id: item for item in self._arenas}

    @property
    def arenas(self) -> tuple[ArenaIdentity, ...]:
        return self._arenas

    @property
    def arenas_by_class(self) -> dict[int, ArenaIdentity]:
        return dict(self._arenas_by_class)

    @property
    def performance_counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def _require_handle(self, *, allow_poisoned: bool = False) -> ctypes.c_void_p:
        if not self._handle or not self._handle.value:
            raise ManagerError("OrbitKV manager handle is closed")
        if self._poisoned is not None and not allow_poisoned:
            raise FailStopped("OrbitKV ABI6 handle is poisoned: " + self._poisoned)
        return self._handle

    def _message(self) -> str:
        return self._error.value.decode("utf-8", errors="replace")

    def _poison(self, reason: str) -> FailStopped:
        if self._poisoned is None:
            self._poisoned = reason
            self._counters["fail_stops"] += 1
        return FailStopped("OrbitKV ABI6 handle is poisoned: " + self._poisoned)

    def _call(
        self,
        operation: str,
        function: Callable[..., Any],
        *args: Any,
        allow_short: bool = False,
        allow_poisoned: bool = False,
        counter: str | None = None,
    ) -> int:
        self._require_handle(allow_poisoned=allow_poisoned)
        if counter is not None:
            self._counters[counter] += 1
        ctypes.memset(self._error, 0, len(self._error))
        try:
            status = int(function(*args, self._error, len(self._error)))
        except BaseException as error:
            raise self._poison(f"{operation} call outcome is unknown: {error}") from error
        message = self._message()
        if status == STATUS_OK:
            if message:
                raise self._poison(f"{operation} succeeded with an error payload")
            return status
        if status == STATUS_BUFFER_TOO_SMALL and allow_short:
            self._counters["buffer_too_small_preflights"] += 1
            return status
        if status == STATUS_RETRYABLE_CONFLICT:
            self._counters["retryable_conflicts"] += 1
            raise RetryableConflict(f"{operation}: {message or 'retryable conflict'}")
        if status in (STATUS_INVALID_ARGUMENT, STATUS_MANAGER_ERROR):
            raise ManagerError(f"{operation} failed with status {status}: {message or 'no detail'}")
        if status in (STATUS_PANIC, STATUS_FAIL_STOPPED):
            kind = "unknown panic" if status == STATUS_PANIC else "core fail-stop"
            raise self._poison(f"{operation} returned {kind}: {message or 'no detail'}")
        raise self._poison(f"{operation} returned unknown status {status}: {message}")

    @staticmethod
    def _count(values: Sequence[Any], label: str, capacity: int) -> int:
        count = len(values)
        if count <= 0 or count > capacity:
            raise ManagerError(f"{label} cardinality exceeds its configured batch bound")
        return count

    @staticmethod
    def _span(offset: int, count: int, cursor: int, total: int, label: str) -> int:
        end = offset + count
        if offset != cursor or end < offset or end > total:
            raise ManagerError(f"{label} spans are not canonical")
        return end

    def _cold_preflight(
        self,
        operation: str,
        status: int,
        required: Sequence[ctypes.c_uint32],
        bounds: Sequence[int],
    ) -> tuple[int, ...]:
        values = tuple(int(value.value) for value in required)
        if (
            status != STATUS_BUFFER_TOO_SMALL
            or len(values) != len(bounds)
            or any(value > bound for value, bound in zip(values, bounds, strict=True))
        ):
            raise self._poison(
                f"{operation} preflight exceeded its configured cold output bound"
            )
        return values

    def _load_arenas(
        self, registrations: Sequence[ArenaRegistration], page_tokens: int
    ) -> tuple[ArenaIdentity, ...]:
        raw = array(L.ArenaIdentityLayout, self._arena_count)
        count = ctypes.c_uint32()
        self._call(
            "arena identities",
            self._library.orbitkv_manager_arena_identities,
            self._handle,
            raw,
            self._arena_count,
            ctypes.byref(count),
        )
        if int(count.value) != self._arena_count:
            raise self._poison("arena identity cardinality changed after create")
        result = tuple(
            ArenaIdentity(
                int(item.engine_epoch),
                int(item.pool_epoch),
                int(item.pool_id),
                int(item.class_id),
                int(item.backend_domain),
                int(item.page_count),
                int(item.page_tokens),
                int(item.backend_base_index),
                int(item.first_page_id),
            )
            for item in raw
        )
        for item, registration in zip(result, registrations, strict=True):
            if (
                item.class_id != registration.class_id
                or item.pool_id != registration.pool_id
                or item.backend_domain != registration.backend_domain
                or item.page_count != registration.page_count
                or item.backend_base_index != registration.backend_base_index
                or item.page_tokens != page_tokens
            ):
                raise self._poison("arena identity differs from its registration")
        return result

    def arena_stats(self) -> tuple[ArenaStats, ...]:
        with self._lock:
            raw = array(L.ArenaStatsLayout, self._arena_count)
            count = ctypes.c_uint32()
            self._call(
                "arena stats",
                self._library.orbitkv_manager_arena_stats,
                self._require_handle(),
                raw,
                self._arena_count,
                ctypes.byref(count),
            )
            if int(count.value) != self._arena_count:
                raise self._poison("arena stats cardinality changed")
            return tuple(
                ArenaStats(
                    int(item.engine_epoch), int(item.pool_epoch), int(item.pool_id),
                    int(item.page_count), int(item.class_id), int(item.backend_domain),
                    int(item.first_page_id), int(item.free_pages), int(item.reserved_pages),
                    int(item.writing_pages), int(item.active_pages), int(item.retiring_pages),
                    int(item.quarantined_pages), int(item.exhausted_pages),
                    int(item.request_page_refs), int(item.prefix_page_refs),
                    int(item.reader_pins),
                )
                for item in raw
            )

    def request_acquire_batch(self, request_count: int) -> tuple[RequestView, ...]:
        with self._lock:
            count = _uint("request acquire count", request_count, 32)
            if count <= 0 or count > self._request_capacity:
                raise ManagerError(
                    "request acquire cardinality exceeds its configured batch bound"
                )
            out = ctypes.c_uint32()
            self._call(
                "request acquire batch",
                self._library.orbitkv_manager_request_acquire_batch,
                self._require_handle(), count, self._request_views, self._request_capacity,
                ctypes.byref(out), counter="request_acquire_batch_calls",
            )
            if int(out.value) != count:
                raise self._poison("request acquire returned invalid cardinality")
            try:
                return tuple(_view(self._request_views[index]) for index in range(count))
            except Exception as error:
                raise self._poison(f"request acquire output is invalid: {error}") from error

    def request_fork_batch(self, items: Sequence[RequestForkItem]) -> tuple[ForkedRequest, ...]:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "request fork", self._request_capacity)
            raw = (L.RequestForkItemLayout * count)(
                *(
                    L.RequestForkItemLayout(
                        _lease_to_c(item.source_request), _lease_to_c(item.expected_source_head),
                        _lease_to_c(item.target_empty_request), _lease_to_c(item.expected_target_head),
                    )
                    for item in values
                )
            )
            required_items = ctypes.c_uint32()
            required_pages = ctypes.c_uint32()
            status = self._call(
                "request fork preflight", self._library.orbitkv_manager_request_fork_batch,
                self._require_handle(), raw, count, None, 0, ctypes.byref(required_items),
                None, 0, ctypes.byref(required_pages), allow_short=True,
                counter="request_fork_batch_calls",
            )
            capacities = self._cold_preflight(
                "request fork",
                status,
                (required_items, required_pages),
                (count, count * self._physical_pages),
            )
            if capacities[0] != count:
                raise self._poison("request fork preflight changed item cardinality")
            cold = cold_materialization(L.ForkedItemLayout, count, capacities[1])
            self._counters["cold_workspace_allocations"] += 1
            out_items = ctypes.c_uint32()
            out_pages = ctypes.c_uint32()
            self._call(
                "request fork batch", self._library.orbitkv_manager_request_fork_batch,
                self._require_handle(), raw, count, cold.items, count, ctypes.byref(out_items),
                cold.pages_or_detached, capacities[1], ctypes.byref(out_pages),
            )
            return self._materialized_fork_outputs(cold, count, out_items, out_pages)

    def _materialized_fork_outputs(
        self, cold: Any, count: int, out_items: Any, out_pages: Any
    ) -> tuple[ForkedRequest, ...]:
        if int(out_items.value) != count:
            raise self._poison("fork output cardinality changed")
        total = int(out_pages.value)
        cursor = 0
        result = []
        try:
            for index in range(count):
                item = cold.items[index]
                end = self._span(int(item.page_offset), int(item.page_count), cursor, total, "fork page")
                pages = tuple(_snapshot_page(cold.pages_or_detached[pos]) for pos in range(cursor, end))
                view = _view(item.target)
                result.append(ForkedRequest(_request(item.source), MaterializedRequestView(view, pages)))
                cursor = end
            if cursor != total:
                raise ManagerError("fork page spans do not cover output")
        except Exception as error:
            raise self._poison(f"fork output is invalid: {error}") from error
        self._counters["materialized_page_objects"] += total
        return tuple(result)

    def prepare_batch(self, items: Sequence[Any]) -> tuple[PreparedStep, ...]:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "prepare", self._operation_capacity)
            for index, item in enumerate(values):
                self._hot.prepare_items[index] = L.PrepareItemLayout(
                    _lease_to_c(item.request), _lease_to_c(item.expected_head),
                    _uint("prepare target boundary", item.target_boundary, 64), 0,
                )
            counts = [ctypes.c_uint32() for _ in range(5)]
            b = self._hot.bounds
            self._call(
                "prepare batch", self._library.orbitkv_manager_prepare_batch,
                self._require_handle(), self._hot.prepare_items, count,
                self._hot.prepared, b.batch, ctypes.byref(counts[0]),
                self._hot.class_lowerings, b.class_outputs, ctypes.byref(counts[1]),
                self._hot.tail_actions, b.class_outputs, ctypes.byref(counts[2]),
                self._hot.copy_intents, b.copy_outputs, ctypes.byref(counts[3]),
                self._hot.write_intents, b.write_outputs, ctypes.byref(counts[4]),
                counter="prepare_batch_calls",
            )
            totals = tuple(int(value.value) for value in counts)
            if totals[0] != count:
                raise self._poison("prepare output cardinality changed")
            try:
                return self._decode_prepared(count, *totals[1:])
            except Exception as error:
                raise self._poison(f"prepare output is invalid: {error}") from error

    def _decode_prepared(
        self, count: int, class_total: int, tail_total: int, copy_total: int, write_total: int
    ) -> tuple[PreparedStep, ...]:
        cursors = [0, 0, 0, 0]
        result = []
        for index in range(count):
            item = self._hot.prepared[index]
            ends = [
                self._span(int(item.class_offset), int(item.class_count), cursors[0], class_total, "prepare class"),
                self._span(int(item.tail_offset), int(item.tail_count), cursors[1], tail_total, "prepare tail"),
                self._span(int(item.copy_offset), int(item.copy_count), cursors[2], copy_total, "prepare copy"),
                self._span(int(item.write_offset), int(item.write_count), cursors[3], write_total, "prepare write"),
            ]
            classes = []
            local_tail = local_copy = local_write = 0
            for position in range(cursors[0], ends[0]):
                raw = self._hot.class_lowerings[position]
                next_tail = self._span(
                    int(raw.tail_offset), int(raw.tail_count), cursors[1] + local_tail, ends[1], "class tail"
                )
                next_copy = self._span(
                    int(raw.copy_offset), int(raw.copy_count), cursors[2] + local_copy, ends[2], "class copy"
                )
                next_write = self._span(
                    int(raw.write_offset), int(raw.write_count), cursors[3] + local_write, ends[3], "class write"
                )
                classes.append(
                    ClassLowering(
                        int(raw.class_id), int(raw.flags), local_tail, int(raw.tail_count),
                        local_copy, int(raw.copy_count), local_write, int(raw.write_count),
                        int(raw.reserved),
                    )
                )
                local_tail = next_tail - cursors[1]
                local_copy = next_copy - cursors[2]
                local_write = next_write - cursors[3]
            if (local_tail, local_copy, local_write) != (
                ends[1] - cursors[1], ends[2] - cursors[2], ends[3] - cursors[3]
            ):
                raise ManagerError("class spans do not cover prepared item outputs")
            result.append(
                PreparedStep(
                    _step(item.step), _request(item.request), _snapshot(item.base_snapshot),
                    _snapshot(item.target_snapshot), int(item.base_view_version),
                    int(item.target_view_version), int(item.previous_boundary),
                    int(item.target_boundary), tuple(classes),
                    tuple(_tail(self._hot.tail_actions[p]) for p in range(cursors[1], ends[1])),
                    tuple(_copy_intent(self._hot.copy_intents[p]) for p in range(cursors[2], ends[2])),
                    tuple(
                        WriteIntent(
                            int(self._hot.write_intents[p].page_generation),
                            int(self._hot.write_intents[p].page_id),
                            int(self._hot.write_intents[p].reserved),
                        )
                        for p in range(cursors[3], ends[3])
                    ),
                )
            )
            cursors = ends
        if tuple(cursors) != (class_total, tail_total, copy_total, write_total):
            raise ManagerError("prepare flat outputs are not fully partitioned")
        return tuple(result)

    def submit_batch(
        self,
        items: Sequence[
            tuple[StepLease, Sequence[BackendBindReceipt], Sequence[BackendCopyReceipt]]
        ],
    ) -> tuple[SubmittedStep, ...]:
        with self._lock:
            values = tuple((step, tuple(binds), tuple(copies)) for step, binds, copies in items)
            count = self._count(values, "submit", self._operation_capacity)
            bind_total = sum(len(item[1]) for item in values)
            copy_total = sum(len(item[2]) for item in values)
            if bind_total > self._hot.bounds.bind_outputs or copy_total > self._hot.bounds.copy_outputs:
                raise ManagerError("submit receipts exceed the compiled hot bound")
            bind_cursor = copy_cursor = 0
            for index, (step, binds, copies) in enumerate(values):
                self._hot.submit_items[index] = L.SubmitItemLayout(
                    _lease_to_c(step), bind_cursor, len(binds), copy_cursor, len(copies)
                )
                for value in binds:
                    self._hot.bind_receipts[bind_cursor] = L.BindReceiptLayout(
                        _lease_to_c(value.step), _page_to_c(value.page),
                        _uint("bind backend domain", value.backend_domain, 16),
                        _uint("bind mapped", value.mapped, 8),
                        _uint("bind writable", value.writable, 8),
                        _uint("bind reserved", value.reserved, 32),
                        _uint("bind backend index", value.backend_index, 64),
                    )
                    bind_cursor += 1
                for value in copies:
                    self._hot.copy_receipts[copy_cursor] = L.CopyReceiptLayout(
                        _lease_to_c(value.step), _uint("copy class", value.class_id, 16),
                        _uint("copy domain", value.backend_domain, 16),
                        _uint("copy token count", value.token_count, 32),
                        _uint("copy source offset", value.source_token_offset, 32),
                        _uint("copy destination offset", value.destination_token_offset, 32),
                        _uint("copy observed", value.observed, 8),
                        _uint("copy completed", value.copied, 8),
                        _uint("copy ordering", value.ordered_before_writes, 8),
                        _uint("copy reserved8", value.reserved8, 8),
                        _uint("copy reserved32", value.reserved32, 32),
                        _page_to_c(value.source), _page_to_c(value.destination),
                        _uint("copy source backend", value.source_backend_index, 64),
                        _uint("copy destination backend", value.destination_backend_index, 64),
                    )
                    copy_cursor += 1
            out = ctypes.c_uint32()
            self._call(
                "submit batch", self._library.orbitkv_manager_submit_batch,
                self._require_handle(), self._hot.submit_items, count,
                self._hot.bind_receipts, bind_total, self._hot.copy_receipts, copy_total,
                self._hot.submitted, self._hot.bounds.batch, ctypes.byref(out),
                counter="submit_batch_calls",
            )
            if int(out.value) != count:
                raise self._poison("submit output cardinality changed")
            return tuple(
                SubmittedStep(
                    _submission(self._hot.submitted[index].submission),
                    _request(self._hot.submitted[index].request),
                    _snapshot(self._hot.submitted[index].target_snapshot),
                )
                for index in range(count)
            )

    def complete_batch(
        self, receipt: BatchCompletionReceipt, submissions: Sequence[SubmissionLease]
    ) -> CompletionBatch:
        with self._lock:
            values = tuple(submissions)
            count = self._count(values, "completion", self._operation_capacity)
            for index, value in enumerate(values):
                self._hot.complete_items[index] = L.CompleteItemLayout(_lease_to_c(value))
            raw_receipt = L.CompletionReceiptLayout(
                _uint("completion engine", receipt.engine_epoch, 64),
                _uint("completion domain", receipt.completion_domain, 64),
                _uint("completion value", receipt.completion_value, 64),
                _uint("completion confirmed", receipt.confirmed, 32),
                _uint("completion reserved", receipt.reserved, 32),
            )
            output_counts = [ctypes.c_uint32() for _ in range(3)]
            b = self._hot.bounds
            self._call(
                "complete batch", self._library.orbitkv_manager_complete_batch,
                self._require_handle(), raw_receipt, self._hot.complete_items, count,
                self._hot.completed, b.batch, ctypes.byref(output_counts[0]),
                self._hot.detached, b.completion_detached, ctypes.byref(output_counts[1]),
                self._hot.retirements, b.completion_retirements, ctypes.byref(output_counts[2]),
                counter="complete_batch_calls",
            )
            totals = tuple(int(item.value) for item in output_counts)
            if totals[0] != count:
                raise self._poison("completion output cardinality changed")
            try:
                cursor = 0
                completions = []
                for index in range(count):
                    item = self._hot.completed[index]
                    if int(item.reserved) != 0:
                        raise ManagerError("completed item reserved field is nonzero")
                    end = self._span(
                        int(item.detached_offset), int(item.detached_count), cursor,
                        totals[1], "completion detached",
                    )
                    completions.append(
                        StepCompletion(
                            _submission(item.submission), _request(item.request),
                            _snapshot(item.detached_snapshot), _snapshot(item.published_snapshot),
                            int(item.published_view_version), int(item.published_boundary),
                            int(item.resident_count),
                            tuple(_detached(self._hot.detached[p]) for p in range(cursor, end)),
                        )
                    )
                    cursor = end
                if cursor != totals[1]:
                    raise ManagerError("completion detach spans do not cover flat output")
                retirements = tuple(
                    _certificate(self._hot.retirements[index]) for index in range(totals[2])
                )
                return CompletionBatch(tuple(completions), retirements)
            except Exception as error:
                raise self._poison(f"completion output is invalid: {error}") from error

    def abort_steps_batch(self, receipts: Sequence[BackendUnobservedReceipt]) -> None:
        with self._lock:
            values = tuple(receipts)
            count = self._count(values, "abort", self._operation_capacity)
            for index, item in enumerate(values):
                self._abort_receipts[index] = L.UnobservedReceiptLayout(
                    _lease_to_c(item.step), _uint("unobserved", item.backend_unobserved, 32),
                    _uint("abort reserved", item.reserved, 32),
                )
            self._call(
                "abort steps batch", self._library.orbitkv_manager_abort_steps_batch,
                self._require_handle(), self._abort_receipts, count,
                counter="abort_steps_batch_calls",
            )

    def quarantine_steps_batch(self, steps: Sequence[StepLease]) -> None:
        self._lease_call(
            tuple(steps), self._step_leases, self._operation_capacity,
            "quarantine steps batch", self._library.orbitkv_manager_quarantine_steps_batch,
            "quarantine_steps_batch_calls",
        )

    def quarantine_submissions_batch(self, submissions: Sequence[SubmissionLease]) -> None:
        self._lease_call(
            tuple(submissions), self._submission_leases, self._operation_capacity,
            "quarantine submissions batch",
            self._library.orbitkv_manager_quarantine_submissions_batch,
            "quarantine_submissions_batch_calls",
        )

    def _lease_call(
        self, values: Sequence[Any], workspace: Any, capacity: int,
        operation: str, function: Callable[..., Any], counter: str,
    ) -> None:
        with self._lock:
            count = self._count(values, operation, capacity)
            for index, value in enumerate(values):
                workspace[index] = _lease_to_c(value)
            self._call(
                operation, function, self._require_handle(), workspace, count, counter=counter
            )

    def release_batch(self, items: Sequence[ReleaseBatchItem]) -> ReleaseBatchCompletion:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "release", self._request_capacity)
            raw = (L.ReleaseItemLayout * count)(
                *(L.ReleaseItemLayout(_lease_to_c(item.request), _lease_to_c(item.expected_head)) for item in values)
            )
            required = [ctypes.c_uint32() for _ in range(3)]
            status = self._call(
                "release preflight", self._library.orbitkv_manager_release_batch,
                self._require_handle(), raw, count, None, 0, ctypes.byref(required[0]),
                None, 0, ctypes.byref(required[1]), None, 0, ctypes.byref(required[2]),
                allow_short=True, counter="release_batch_calls",
            )
            capacities = self._cold_preflight(
                "release",
                status,
                required,
                (count, count * self._physical_pages, self._physical_pages),
            )
            if capacities[0] != count:
                raise self._poison("release preflight changed item cardinality")
            cold = cold_reclamation(
                L.ReleasedItemLayout, count, capacities[1], capacities[2]
            )
            self._counters["cold_workspace_allocations"] += 1
            out = [ctypes.c_uint32() for _ in range(3)]
            self._call(
                "release batch", self._library.orbitkv_manager_release_batch,
                self._require_handle(), raw, count, cold.items, count, ctypes.byref(out[0]),
                cold.pages_or_detached, capacities[1], ctypes.byref(out[1]),
                cold.retirements, capacities[2], ctypes.byref(out[2]),
            )
            return self._decode_release(cold, count, out)

    def _decode_release(self, cold: Any, count: int, out: Sequence[Any]) -> ReleaseBatchCompletion:
        actual = tuple(int(value.value) for value in out)
        detached_capacity = (
            len(cold.pages_or_detached) if cold.pages_or_detached is not None else 0
        )
        retirement_capacity = len(cold.retirements) if cold.retirements is not None else 0
        if (
            actual[0] != count
            or actual[1] != detached_capacity
            or actual[2] > retirement_capacity
        ):
            raise self._poison("release second pass changed required capacities")
        cursor = 0
        releases = []
        try:
            total = int(out[1].value)
            for index in range(count):
                item = cold.items[index]
                if int(item.reserved) != 0:
                    raise ManagerError("released item reserved field is nonzero")
                end = self._span(int(item.detached_offset), int(item.detached_count), cursor, total, "release detached")
                releases.append(
                    ReleaseCompletion(
                        _request(item.request), _snapshot(item.detached_snapshot),
                        tuple(_detached(cold.pages_or_detached[p]) for p in range(cursor, end)),
                    )
                )
                cursor = end
            return ReleaseBatchCompletion(
                tuple(releases),
                tuple(_certificate(cold.retirements[p]) for p in range(int(out[2].value))),
            )
        except Exception as error:
            raise self._poison(f"release output is invalid: {error}") from error

    def acknowledge_reclamations_batch(self, receipts: Sequence[ReclamationReceipt]) -> None:
        with self._lock:
            values = tuple(receipts)
            count = self._count(values, "reclamation acknowledgement", self._physical_pages)
            for index, item in enumerate(values):
                self._reclamation_receipts[index] = L.ReclamationReceiptLayout(
                    _lease_to_c(item.reclamation), _page_to_c(item.page),
                    _uint("reclamation domain", item.backend_domain, 16),
                    _uint("reclamation acknowledged", item.acknowledged, 8),
                    _uint("reclamation reserved8", item.reserved8, 8),
                    _uint("reclamation reserved32", item.reserved32, 32),
                    _uint("reclamation backend", item.backend_index, 64),
                )
            self._call(
                "acknowledge reclamations batch",
                self._library.orbitkv_manager_acknowledge_reclamations_batch,
                self._require_handle(), self._reclamation_receipts, count,
                counter="acknowledge_reclamations_batch_calls",
            )

    def recycle_requests_batch(self, requests: Sequence[RequestLease]) -> None:
        self._lease_call(
            tuple(requests), self._request_leases, self._request_capacity,
            "recycle requests batch", self._library.orbitkv_manager_recycle_requests_batch,
            "recycle_requests_batch_calls",
        )

    def prefix_lookup_batch(self, keys: Sequence[PrefixSemanticKey]) -> tuple[PrefixLookupHint, ...]:
        with self._lock:
            values = tuple(keys)
            count = self._count(values, "prefix lookup", self._prefix_capacity)
            raw = (L.PrefixKeyLayout * count)(*(_key_to_c(value) for value in values))
            output = array(L.PrefixLookupHintLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "prefix lookup batch", self._library.orbitkv_manager_prefix_lookup_batch,
                self._require_handle(), raw, count, output, count, ctypes.byref(out),
                counter="prefix_lookup_batch_calls",
            )
            if int(out.value) != count:
                raise self._poison("prefix lookup cardinality changed")
            try:
                result = []
                for item in output:
                    if int(item.reserved) or int(item.reserved_padding) or int(item.candidate_present) not in (0, 1):
                        raise ManagerError("prefix lookup reserved/presence field is invalid")
                    candidate = _prefix(item.candidate) if int(item.candidate_present) else None
                    result.append(PrefixLookupHint(_key(item.key), candidate, int(item.resident_count)))
                return tuple(result)
            except Exception as error:
                raise self._poison(f"prefix lookup output is invalid: {error}") from error

    def prefix_attach_batch(self, items: Sequence[PrefixAttachItem]) -> tuple[AttachedPrefix, ...]:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "prefix attach", min(self._prefix_capacity, self._request_capacity))
            raw = (L.PrefixAttachItemLayout * count)(
                *(
                    L.PrefixAttachItemLayout(
                        _lease_to_c(item.request), _lease_to_c(item.expected_empty_head),
                        self._hint_to_c(item.hint),
                    )
                    for item in values
                )
            )
            required = [ctypes.c_uint32(), ctypes.c_uint32()]
            status = self._call(
                "prefix attach preflight", self._library.orbitkv_manager_prefix_attach_batch,
                self._require_handle(), raw, count, None, 0, ctypes.byref(required[0]),
                None, 0, ctypes.byref(required[1]), allow_short=True,
                counter="prefix_attach_batch_calls",
            )
            capacities = self._cold_preflight(
                "prefix attach",
                status,
                required,
                (count, count * self._physical_pages),
            )
            if capacities[0] != count:
                raise self._poison("prefix attach preflight changed item cardinality")
            cold = cold_materialization(L.AttachedPrefixLayout, count, capacities[1])
            self._counters["cold_workspace_allocations"] += 1
            out = [ctypes.c_uint32(), ctypes.c_uint32()]
            self._call(
                "prefix attach batch", self._library.orbitkv_manager_prefix_attach_batch,
                self._require_handle(), raw, count, cold.items, count, ctypes.byref(out[0]),
                cold.pages_or_detached, capacities[1], ctypes.byref(out[1]),
            )
            if int(out[0].value) != count or int(out[1].value) != int(required[1].value):
                raise self._poison("prefix attach second pass changed capacities")
            cursor = 0
            result = []
            try:
                for index in range(count):
                    item = cold.items[index]
                    end = self._span(int(item.page_offset), int(item.page_count), cursor, int(out[1].value), "attach pages")
                    result.append(
                        AttachedPrefix(
                            _prefix(item.prefix),
                            MaterializedRequestView(
                                _view(item.target),
                                tuple(_snapshot_page(cold.pages_or_detached[p]) for p in range(cursor, end)),
                            ),
                        )
                    )
                    cursor = end
                return tuple(result)
            except Exception as error:
                raise self._poison(f"prefix attach output is invalid: {error}") from error

    @staticmethod
    def _hint_to_c(value: PrefixLookupHint) -> L.PrefixLookupHintLayout:
        candidate = (
            L.PrefixLeaseLayout()
            if value.candidate is None
            else _lease_to_c(value.candidate)
        )
        return L.PrefixLookupHintLayout(
            _key_to_c(value.key), candidate, _uint("hint residents", value.resident_count, 32),
            0 if value.candidate is None else 1, 0, 0,
        )

    def _publish_inputs(self, values: Sequence[PrefixPublishItem]) -> Any:
        return (L.PrefixPublishItemLayout * len(values))(
            *(
                L.PrefixPublishItemLayout(
                    _lease_to_c(item.request), _lease_to_c(item.expected_head), _key_to_c(item.key)
                )
                for item in values
            )
        )

    def prefix_publish_batch(self, items: Sequence[PrefixPublishItem]) -> tuple[PublishedPrefix, ...]:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "prefix publish", min(self._prefix_capacity, self._request_capacity))
            raw = self._publish_inputs(values)
            output = array(L.PublishedPrefixLayout, count)
            out = ctypes.c_uint32()
            self._call(
                "prefix publish batch", self._library.orbitkv_manager_prefix_publish_batch,
                self._require_handle(), raw, count, output, count, ctypes.byref(out),
                counter="prefix_publish_batch_calls",
            )
            if int(out.value) != count:
                raise self._poison("prefix publish cardinality changed")
            try:
                return tuple(_published(output[index]) for index in range(count))
            except Exception as error:
                raise self._poison(f"prefix publish output is invalid: {error}") from error

    def prefix_publish_release_batch(
        self, items: Sequence[PrefixPublishItem]
    ) -> PrefixPublishReleaseBatch:
        with self._lock:
            values = tuple(items)
            count = self._count(values, "prefix publish-release", min(self._prefix_capacity, self._request_capacity))
            raw = self._publish_inputs(values)
            required = [ctypes.c_uint32() for _ in range(3)]
            status = self._call(
                "prefix publish-release preflight",
                self._library.orbitkv_manager_prefix_publish_release_batch,
                self._require_handle(), raw, count, None, 0, ctypes.byref(required[0]),
                None, 0, ctypes.byref(required[1]), None, 0, ctypes.byref(required[2]),
                allow_short=True, counter="prefix_publish_release_batch_calls",
            )
            capacities = self._cold_preflight(
                "publish-release",
                status,
                required,
                (count, count * self._physical_pages, self._physical_pages),
            )
            if capacities[0] != count:
                raise self._poison("publish-release preflight changed item cardinality")
            cold = cold_reclamation(
                L.PrefixPublishReleaseLayout, count,
                capacities[1], capacities[2],
            )
            self._counters["cold_workspace_allocations"] += 1
            out = [ctypes.c_uint32() for _ in range(3)]
            self._call(
                "prefix publish-release batch",
                self._library.orbitkv_manager_prefix_publish_release_batch,
                self._require_handle(), raw, count, cold.items, count, ctypes.byref(out[0]),
                cold.pages_or_detached, capacities[1], ctypes.byref(out[1]),
                cold.retirements, capacities[2], ctypes.byref(out[2]),
            )
            actual = tuple(int(item.value) for item in out)
            if (
                actual[0] != count
                or actual[1] != capacities[1]
                or actual[2] > capacities[2]
            ):
                raise self._poison("publish-release second pass changed capacities")
            cursor = 0
            outputs = []
            try:
                for index in range(count):
                    item = cold.items[index]
                    if int(item.reserved) != 0:
                        raise ManagerError("publish-release reserved field is nonzero")
                    end = self._span(int(item.detached_offset), int(item.detached_count), cursor, int(out[1].value), "publish-release detached")
                    outputs.append(
                        PrefixPublishRelease(
                            _published(item.publication),
                            ReleaseCompletion(
                                _request(item.request), _snapshot(item.detached_snapshot),
                                tuple(_detached(cold.pages_or_detached[p]) for p in range(cursor, end)),
                            ),
                        )
                    )
                    cursor = end
                return PrefixPublishReleaseBatch(
                    tuple(outputs),
                    tuple(_certificate(cold.retirements[p]) for p in range(int(out[2].value))),
                )
            except Exception as error:
                raise self._poison(f"publish-release output is invalid: {error}") from error

    def prefix_evict_batch(self, prefixes: Sequence[PrefixLease]) -> PrefixEvictionBatch:
        with self._lock:
            values = tuple(prefixes)
            count = self._count(values, "prefix evict", self._prefix_capacity)
            raw = (L.PrefixLeaseLayout * count)(
                *(_lease_to_c(value) for value in values)
            )
            required = [ctypes.c_uint32(), ctypes.c_uint32()]
            status = self._call(
                "prefix evict preflight", self._library.orbitkv_manager_prefix_evict_batch,
                self._require_handle(), raw, count, None, 0, ctypes.byref(required[0]),
                None, 0, ctypes.byref(required[1]), allow_short=True,
                counter="prefix_evict_batch_calls",
            )
            capacities = self._cold_preflight(
                "prefix evict", status, required, (count, self._physical_pages)
            )
            if capacities[0] != count:
                raise self._poison("prefix evict preflight changed item cardinality")
            output = array(L.EvictedPrefixLayout, count)
            certs = array(L.ReclamationCertificateLayout, capacities[1])
            self._counters["cold_workspace_allocations"] += 1
            out = [ctypes.c_uint32(), ctypes.c_uint32()]
            self._call(
                "prefix evict batch", self._library.orbitkv_manager_prefix_evict_batch,
                self._require_handle(), raw, count, output, count, ctypes.byref(out[0]),
                certs, capacities[1], ctypes.byref(out[1]),
            )
            actual = tuple(int(item.value) for item in out)
            if actual[0] != count or actual[1] > capacities[1]:
                raise self._poison("prefix evict second pass changed capacities")
            try:
                return PrefixEvictionBatch(
                    tuple(EvictedPrefix(_prefix(output[p].prefix), _key(output[p].key)) for p in range(count)),
                    tuple(_certificate(certs[p]) for p in range(int(out[1].value))),
                )
            except Exception as error:
                raise self._poison(f"prefix evict output is invalid: {error}") from error

    def prefix_recycle_batch(self, prefixes: Sequence[PrefixLease]) -> None:
        values = tuple(prefixes)
        workspace = array(L.PrefixLeaseLayout, len(values))
        self._lease_call(
            values, workspace, self._prefix_capacity, "prefix recycle batch",
            self._library.orbitkv_manager_prefix_recycle_batch,
            "prefix_recycle_batch_calls",
        )

    def stats(self) -> ManagerStats:
        with self._lock:
            raw = L.ManagerStatsLayout()
            self._call(
                "manager stats", self._library.orbitkv_manager_stats,
                self._require_handle(allow_poisoned=True), ctypes.byref(raw),
                allow_poisoned=True,
            )
            return ManagerStats(*(int(getattr(raw, name)) for name, _ctype in raw._fields_))

    def destroy(self) -> None:
        with self._lock:
            if not self._handle or not self._handle.value or self._destroy_attempted:
                return
            handle = self._handle
            self._destroy_attempted = True
            try:
                self._call(
                    "destroy manager",
                    self._library.orbitkv_manager_destroy,
                    handle,
                    allow_poisoned=True,
                )
            finally:
                # Destruction consumes the exclusive native pointer even when
                # its return path is lost or malformed.  Retaining the address
                # would turn a later stats call into a use-after-free.
                self._handle = ctypes.c_void_p()


class CtypesManagerFactory(ManagerFactoryProtocol):
    def create(
        self,
        config: Any,
        settings: ManagerCreateSettings,
        arenas: Sequence[ArenaRegistration],
    ) -> ManagerProtocol:
        loaded = LoadedLibrary(Path(config.library_path))
        plan = bytes(config.plan_json)
        if not plan:
            raise ManagerError("canonical KvPlanInput JSON is empty")
        registrations = tuple(arenas)
        if not registrations or len(registrations) != len(config.classes):
            raise ManagerError("one ABI6 arena is required for every plan class")
        if tuple(item.class_id for item in registrations) != tuple(range(len(registrations))):
            raise ManagerError("ABI6 arena registrations must be class-id ordered")
        total_pages = sum(item.page_count for item in registrations)
        for registration, class_config in zip(registrations, config.classes, strict=True):
            if (
                registration.class_id != class_config.class_id
                or registration.pool_id != class_config.pool_id
                or registration.backend_domain != class_config.backend_domain
                or registration.page_count <= 0
                or registration.backend_base_index < 0
            ):
                raise ManagerError("ABI6 arena registration differs from the plan")
        if settings.maximum_reclamations < total_pages:
            raise ManagerError("maximum_reclamations must cover all physical pages")
        manager_config = L.ManagerConfigLayout(
            _uint("maximum requests", settings.maximum_requests, 32),
            _uint("maximum operations", settings.maximum_operations, 32),
            _uint("maximum prefixes", settings.maximum_prefixes, 32),
            _uint("maximum reclamations", settings.maximum_reclamations, 32),
            _uint("maximum step tokens", settings.maximum_step_tokens, 32),
        )
        backend = (L.BackendArenaRegistrationLayout * len(registrations))(
            *(
                L.BackendArenaRegistrationLayout(
                    _uint("pool id", item.pool_id, 32),
                    _uint("class id", item.class_id, 16),
                    _uint("backend domain", item.backend_domain, 16),
                    _uint("page count", item.page_count, 32), 0,
                    _uint("backend base index", item.backend_base_index, 64),
                )
                for item in registrations
            )
        )
        plan_buffer = (ctypes.c_uint8 * len(plan)).from_buffer_copy(plan)
        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(ERROR_BUFFER_BYTES)
        try:
            status = int(
                loaded.cdll.orbitkv_manager_create(
                    plan_buffer, len(plan), ctypes.byref(manager_config), backend,
                    len(registrations), ctypes.byref(handle), error, len(error),
                )
            )
        except BaseException as failure:
            _discard_created_handle(loaded, handle)
            raise FailStopped(f"manager create outcome is unknown: {failure}") from failure
        message = error.value.decode("utf-8", errors="replace")
        if status != STATUS_OK or not handle.value or message:
            _discard_created_handle(loaded, handle)
            if status in (
                STATUS_BUFFER_TOO_SMALL,
                STATUS_RETRYABLE_CONFLICT,
                STATUS_INVALID_ARGUMENT,
                STATUS_MANAGER_ERROR,
            ):
                raise ManagerError(
                    f"manager create failed with status {status}: {message or 'no detail'}"
                )
            raise FailStopped(
                f"manager create outcome is unusable (status {status}): "
                f"{message or 'no detail'}"
            )
        try:
            return CtypesManager(
                loaded, handle, registrations, int(config.page_tokens), settings
            )
        except Exception as primary:
            _discard_created_handle(loaded, handle)
            raise primary


__all__ = ["CtypesManager", "CtypesManagerFactory"]
