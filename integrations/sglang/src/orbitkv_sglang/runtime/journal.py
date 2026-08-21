from __future__ import annotations

from threading import RLock
from typing import Any, Hashable, Sequence

from .completion import (
    BatchCompletionReceipt,
    BatchRecord,
    CompletionBatch,
    CompletionRuntimeMixin,
    CursorDelta,
    EventGroup,
    StepPhase,
    StepRecord,
    completion_cursor_delta,
)
from .census import CensusRuntimeMixin
from .identity import (
    DETACHED_CLEAR,
    DETACHED_COPY_ON_WRITE,
    DETACHED_PREFIX_TRANSFER,
    DETACHED_REPLACE,
    DETACHED_REQUEST_RELEASE,
    DETACHED_RETENTION,
    FailStopped,
    ManagerError,
    ManagerProtocol,
    PageLease,
    PrefixLease,
    PrefixSemanticKey,
    ReclamationLease,
    RequestLease,
    RetryableConflict,
    SnapshotLease,
    StepLease,
)
from .identity_index import IdentityIndexMixin
from .reclamation import (
    DetachedBinding,
    MirrorCleanupBinding,
    MirrorCandidateTransition,
    MirrorCleanupItem,
    MirrorCleanupProtocol,
    PrefixEvictionBatch,
    PrefixPublishReleaseBatch,
    ReclamationCertificate,
    ReleaseBatchItem,
    reclamation_receipts,
)
from .page_registry import (
    PageRefPlan,
    PhysicalPageRegistry,
    canonical_shadows,
    shadow_identities,
)
from .records import MirrorTransaction, RequestRecord
from .snapshot_shadow import (
    AttachedPrefix,
    ForkedRequest,
    LoweringPlan,
    MaterializedRequestView,
    PageShadow,
    PrefixAttachItem,
    PrefixLookupHint,
    PrefixPublishItem,
    PrepareBatchItem,
    PreparedStep,
    PublishedPrefix,
    RequestCursor,
    RequestForkItem,
    RequestView,
    _decode_prepared,
    _nonnegative,
    _positive,
    page_shadow_from_snapshot,
)

_ZERO_SNAPSHOT = SnapshotLease(0, 0, 0)


class CanonicalRuntime(IdentityIndexMixin, CensusRuntimeMixin, CompletionRuntimeMixin):
    """Fail-closed host journal around the ABI6 canonical manager."""

    def __init__(self, config: Any, manager: ManagerProtocol):
        if not isinstance(manager, ManagerProtocol):
            raise TypeError("manager does not implement the canonical protocol")
        self.config = config
        self.manager = manager
        self.arenas = tuple(manager.arenas)
        self.arenas_by_class = dict(manager.arenas_by_class)
        expected_classes = tuple(config.classes)
        if len(self.arenas) != len(expected_classes):
            raise ManagerError("manager returned the wrong arena count")
        if tuple(item.class_id for item in self.arenas) != tuple(
            item.class_id for item in expected_classes
        ):
            raise ManagerError("manager arenas are not in compiled class order")
        if self.arenas_by_class != {item.class_id: item for item in self.arenas}:
            raise ManagerError("manager arena index is incomplete or inconsistent")

        engine_epochs: set[int] = set()
        pool_ids: set[int] = set()
        page_ranges: list[tuple[int, int]] = []
        for class_config, arena in zip(expected_classes, self.arenas, strict=True):
            for name, value in (
                ("engine epoch", arena.engine_epoch),
                ("pool epoch", arena.pool_epoch),
                ("pool id", arena.pool_id),
                ("backend domain", arena.backend_domain),
                ("page count", arena.page_count),
                ("first page id", arena.first_page_id),
            ):
                _positive(name, value)
            _nonnegative("backend base index", arena.backend_base_index)
            if arena.page_tokens != int(config.page_tokens):
                raise ManagerError("manager arena does not match the compiled page size")
            if (
                arena.class_id != class_config.class_id
                or arena.pool_id != class_config.pool_id
                or arena.backend_domain != class_config.backend_domain
            ):
                raise ManagerError("manager arena does not match its compiled KV class")
            engine_epochs.add(arena.engine_epoch)
            if arena.pool_id in pool_ids:
                raise ManagerError("manager returned duplicate pool ids")
            pool_ids.add(arena.pool_id)
            page_range = (arena.first_page_id, arena.first_page_id + arena.page_count)
            if any(
                page_range[0] < end and start < page_range[1]
                for start, end in page_ranges
            ):
                raise ManagerError("manager arena page-id ranges overlap")
            page_ranges.append(page_range)
        if len(engine_epochs) != 1:
            raise ManagerError("manager arenas do not share one engine epoch")

        self.engine_epoch = next(iter(engine_epochs))
        self.page_tokens = int(config.page_tokens)
        self.page_count = sum(item.page_count for item in self.arenas)
        self._requests: dict[Hashable, RequestRecord] = {}
        self._initialize_identity_indexes()
        self._candidate_pages: dict[tuple[int, int], PageShadow] = {}
        self._page_registry = PhysicalPageRegistry()
        self._request_rows: dict[Hashable, int] = {}
        self._row_owners: dict[int, Hashable] = {}
        self._prefix_eviction_cleanup: MirrorCleanupProtocol | None = None
        self._events: list[EventGroup] = []
        self._completion_value = 1
        self._swa_retirement_certificates = 0
        self._swa_pages_reclaimed = 0
        self._swa_wrap_events = 0
        self._runtime_counters = {
            "forward_events": 0,
            "completion_values": 0,
            "event_queries": 0,
            "event_waits": 0,
            "quarantine_count": 0,
            "fail_stop_count": 0,
        }
        self._failure: str | None = None
        self._lock = RLock()

    def _healthy(self) -> None:
        if self._failure is not None:
            raise FailStopped("OrbitKV manager is fail-stopped: " + self._failure)

    @property
    def failure_reason(self) -> str | None:
        return self._failure

    def fail_stop(self, reason: str) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = str(reason)
                self._runtime_counters["fail_stop_count"] += 1

    def record_for(self, key: Hashable) -> RequestRecord:
        with self._lock:
            self._healthy()
            try:
                record = self._requests[key]
            except KeyError as error:
                raise ManagerError("request is not acquired") from error
            self._require_indexed_record(record)
            return record

    def has_request(self, key: Hashable) -> bool:
        with self._lock:
            self._healthy()
            return key in self._requests

    def bind_request_rows(self, assignments: Sequence[tuple[Hashable, int, bool]]) -> None:
        """Install new rows or revalidate rows already owned by a request.

        The boolean describes row ownership, not manager-request ownership.  A
        prefix hit deliberately creates and attaches the manager request before
        SGLang allocates its ReqToToken row.
        """

        with self._lock:
            self._healthy()
            values = tuple(assignments)
            keys = {key for key, _row, _new in values}
            rows = {row for _key, row, _new in values}
            valid = bool(values) and len(keys) == len(values) and len(rows) == len(values)
            if valid:
                for key, row, install_row in values:
                    if (
                        not isinstance(install_row, bool)
                        or isinstance(row, bool)
                        or not isinstance(row, int)
                        or row <= 0
                    ):
                        valid = False
                        break
                    if install_row:
                        valid = (
                            key not in self._request_rows
                            and row not in self._row_owners
                        )
                    else:
                        valid = (
                            key in self._requests
                            and self._request_rows.get(key) == row
                            and self._row_owners.get(row) == key
                        )
                    if not valid:
                        break
            if not valid:
                self.fail_stop("ReqToToken row ownership became uncertain")
                raise FailStopped(self._failure or "request-row binding failed")
            for key, row, install_row in values:
                if install_row:
                    self._request_rows[key] = row
                    self._row_owners[row] = key

    def rollback_request_rows(
        self, assignments: Sequence[tuple[Hashable, int]]
    ) -> None:
        """Undo rows installed before a pre-commit prepare failure.

        A live request is eligible only while it is idle and has never gained
        mirror-cleanup authority.  This is the precise state of a prefix attach
        waiting for admission; running or previously lowered requests cannot
        use this rollback path.
        """

        with self._lock:
            self._healthy()
            values = tuple(assignments)
            if (
                not values
                or len({key for key, _ in values}) != len(values)
                or len({row for _, row in values}) != len(values)
            ):
                raise ManagerError("request-row rollback must be nonempty and unique")
            for key, row in values:
                record = self._requests.get(key)
                live_is_rollback_safe = record is None or (
                    record.pending is None and record.reclamation_cleanup is None
                )
                if (
                    not live_is_rollback_safe
                    or self._request_rows.get(key) != row
                    or self._row_owners.get(row) != key
                ):
                    self.fail_stop("ReqToToken row rollback changed ownership identity")
                    raise FailStopped(self._failure or "request-row rollback failed")
            for key, row in values:
                del self._request_rows[key]
                del self._row_owners[row]

    def unbind_request_rows(self, assignments: Sequence[tuple[Hashable, int]]) -> None:
        with self._lock:
            self._healthy()
            values = tuple(assignments)
            if (
                not values
                or len({key for key, _ in values}) != len(values)
                or len({row for _, row in values}) != len(values)
                or any(
                    key in self._requests
                    or self._request_rows.get(key) != row
                    or self._row_owners.get(row) != key
                    for key, row in values
                )
            ):
                self.fail_stop("ReqToToken row release changed ownership identity")
                raise FailStopped(self._failure or "request-row release failed")
            for key, row in values:
                del self._request_rows[key]
                del self._row_owners[row]

    def bind_reclamation_cleanup(self, key: Hashable, cleanup: MirrorCleanupBinding) -> None:
        with self._lock:
            self._healthy()
            if not isinstance(cleanup, MirrorCleanupBinding) or not isinstance(
                cleanup.coordinator, MirrorCleanupProtocol
            ):
                raise TypeError("reclamation cleanup must be a collective binding")
            record = self.record_for(key)
            if record.reclamation_cleanup is not None:
                raise ManagerError("cannot replace an installed mirror cleanup")
            if record.pending is not None and record.pending.phase is not StepPhase.PREPARED:
                raise ManagerError("cannot install mirror cleanup after lowering")
            record.reclamation_cleanup = cleanup

    def bind_prefix_eviction_cleanup(
        self, coordinator: MirrorCleanupProtocol
    ) -> None:
        """Install the sole global mirror authority for prefix-last-ref reuse."""

        with self._lock:
            self._healthy()
            if not isinstance(coordinator, MirrorCleanupProtocol):
                raise TypeError("prefix eviction cleanup must be a collective coordinator")
            if self._prefix_eviction_cleanup is not None:
                raise ManagerError("cannot replace the prefix eviction cleanup coordinator")
            self._prefix_eviction_cleanup = coordinator

    def _acquire_missing(self, keys: Sequence[Hashable]) -> dict[Hashable, RequestRecord]:
        if not keys:
            return {}
        try:
            views = tuple(self.manager.request_acquire_batch(len(keys)))
        except (RetryableConflict, ManagerError):
            raise
        except Exception as error:
            self.fail_stop(f"request batch acquisition became uncertain: {error}")
            raise FailStopped(self._failure or "request batch acquisition failed") from error
        try:
            if len(views) != len(keys) or len({view.request for view in views}) != len(views):
                raise ManagerError("manager returned invalid acquired-request cardinality")
            result: dict[Hashable, RequestRecord] = {}
            live = self._request_leases
            live_snapshots = self._snapshot_leases
            acquired_snapshots: set[SnapshotLease] = set()
            for key, view in zip(keys, views, strict=True):
                self._validate_empty_view(view)
                if view.request in live:
                    raise ManagerError("manager returned an already-owned request")
                if (
                    view.snapshot in live_snapshots
                    or view.snapshot in acquired_snapshots
                ):
                    raise ManagerError("manager reused an acquired snapshot identity")
                acquired_snapshots.add(view.snapshot)
                result[key] = RequestRecord(RequestCursor.from_view(view))
            return result
        except Exception as error:
            self.fail_stop(f"manager returned invalid acquired views: {error}")
            raise FailStopped(self._failure or "invalid acquired views") from error

    def request_acquire_batch(
        self, keys: Sequence[Hashable]
    ) -> tuple[RequestView, ...]:
        """Acquire an ordered set of empty request heads without preparing a step."""

        with self._lock:
            self._healthy()
            values = tuple(keys)
            if not values or len(set(values)) != len(values):
                raise ManagerError("request acquire keys must be nonempty and unique")
            if any(key in self._requests for key in values):
                raise ManagerError("request acquire names an already-live key")
            acquired = self._acquire_missing(values)
            self._requests.update(acquired)
            self._register_acquired(acquired.values())
            return tuple(
                RequestView(
                    acquired[key].lease,
                    acquired[key].head,
                    acquired[key].cursor.view_version,
                    acquired[key].boundary,
                    len(acquired[key].cursor.pages),
                )
                for key in values
            )

    def prepare_batch(
        self, items: Sequence[tuple[Hashable, int]]
    ) -> tuple[BatchRecord, tuple[LoweringPlan, ...]]:
        with self._lock:
            self._healthy()
            values = tuple(items)
            if not values or len({key for key, _ in values}) != len(values):
                raise ManagerError("prepare batch keys must be nonempty and unique")
            new_keys: list[Hashable] = []
            records: list[RequestRecord | None] = []
            for key, target in values:
                if isinstance(target, bool) or not isinstance(target, int):
                    raise ManagerError("step target must be an integer")
                record = self._requests.get(key)
                if record is None:
                    new_keys.append(key)
                elif record.pending is not None:
                    raise ManagerError("request already has a pending step")
                elif target <= record.boundary:
                    raise ManagerError("step target must advance the request boundary")
                records.append(record)
            acquired = self._acquire_missing(new_keys)
            self._requests.update(acquired)
            self._register_acquired(acquired.values())
            ordered = tuple(
                record if record is not None else acquired[key]
                for (key, _target), record in zip(values, records, strict=True)
            )
            for record in ordered:
                self._require_indexed_record(record)
            try:
                outputs = tuple(
                    self.manager.prepare_batch(
                        tuple(
                            PrepareBatchItem(record.lease, record.head, target)
                            for record, (_key, target) in zip(ordered, values, strict=True)
                        )
                    )
                )
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"manager batch preparation became uncertain: {error}")
                raise FailStopped(self._failure or "manager batch preparation failed") from error

            pending_records: list[StepRecord] = []
            plans: list[LoweringPlan] = []
            batch_physical: set[tuple[int, int]] = set()
            try:
                if len(outputs) != len(values):
                    raise ManagerError("manager returned the wrong prepared item count")
                occupied_snapshots = self._snapshot_leases
                occupied_steps = self._step_leases
                batch_snapshots: set[SnapshotLease] = set()
                batch_steps: set[StepLease] = set()
                for (key, target), record, prepared in zip(
                    values, ordered, outputs, strict=True
                ):
                    self._validate_prepared(record, prepared, target)
                    if (
                        prepared.target_snapshot in occupied_snapshots
                        or prepared.target_snapshot in batch_snapshots
                    ):
                        raise ManagerError("prepare reused a live target snapshot")
                    if prepared.step in occupied_steps or prepared.step in batch_steps:
                        raise ManagerError("prepare reused a live step identity")
                    batch_snapshots.add(prepared.target_snapshot)
                    batch_steps.add(prepared.step)
                    plan, new_pages = _decode_prepared(
                        record.cursor, prepared, self.arenas_by_class, self.config
                    )
                    for shadow in new_pages:
                        physical = self._page_registry.physical_key(shadow)
                        if (
                            physical in batch_physical
                            or physical in self._candidate_pages
                            or self._page_registry.contains_physical(shadow)
                        ):
                            raise ManagerError("prepare aliases a reserved physical page")
                        batch_physical.add(physical)
                    plans.append(plan)
                    pending_records.append(StepRecord(key, prepared, new_pages))
            except Exception as error:
                self._best_effort_quarantine_steps(
                    tuple(
                        output.step
                        for output in outputs
                        if isinstance(getattr(output, "step", None), StepLease)
                    )
                )
                self.fail_stop(f"manager returned an invalid prepare batch: {error}")
                raise FailStopped(self._failure or "invalid prepare batch") from error

            batch = BatchRecord(tuple(key for key, _ in values), tuple(pending_records))
            for record, pending in zip(ordered, batch.records, strict=True):
                record.pending = pending
                self._register_prepared(pending)
                for shadow in pending.new_pages:
                    self._candidate_pages[
                        self._page_registry.physical_key(shadow)
                    ] = shadow
            return batch, tuple(plans)

    def request_fork_batch(
        self, items: Sequence[tuple[Hashable, Hashable]]
    ) -> tuple[ForkedRequest, ...]:
        with self._lock:
            self._healthy()
            values = tuple(items)
            if not values or len({target for _source, target in values}) != len(values):
                raise ManagerError("fork targets must be nonempty and unique")
            try:
                sources = tuple(self._requests[source] for source, _target in values)
                targets = tuple(self._requests[target] for _source, target in values)
            except KeyError as error:
                raise ManagerError("fork names an unknown request") from error
            if any(record.pending is not None for record in sources + targets):
                raise ManagerError("fork request has a pending step")
            for record in sources + targets:
                self._require_indexed_record(record)
            for target in targets:
                self._require_empty_record(target)
            raw = tuple(
                RequestForkItem(source.lease, source.head, target.lease, target.head)
                for source, target in zip(sources, targets, strict=True)
            )
            try:
                outputs = tuple(self.manager.request_fork_batch(raw))
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"request fork became uncertain: {error}")
                raise FailStopped(self._failure or "request fork failed") from error
            try:
                if len(outputs) != len(values):
                    raise ManagerError("manager returned wrong fork cardinality")
                cursors = tuple(
                    self._cursor_from_materialized(target.lease, output.target)
                    for target, output in zip(targets, outputs, strict=True)
                )
                occupied_snapshots = self._snapshot_leases
                if (
                    len({cursor.snapshot for cursor in cursors}) != len(cursors)
                    or any(cursor.snapshot in occupied_snapshots for cursor in cursors)
                ):
                    raise ManagerError("fork output reused a live snapshot identity")
                for source, target, cursor, output in zip(
                    sources, targets, cursors, outputs, strict=True
                ):
                    if (
                        output.source != source.lease
                        or cursor.boundary != source.boundary
                        or cursor.view_version != target.cursor.view_version + 1
                        or cursor.snapshot.engine_epoch != self.engine_epoch
                        or cursor.snapshot.generation <= 0
                        or cursor.snapshot == target.head
                        or shadow_identities(tuple(cursor.pages.values()))
                        != shadow_identities(tuple(source.cursor.pages.values()))
                    ):
                        raise ManagerError("fork output changed source identity")
                page_refs = self._page_registry.plan(
                    (), tuple(page for cursor in cursors for page in cursor.pages.values())
                )
            except Exception as error:
                self.fail_stop(f"manager returned invalid fork output: {error}")
                raise FailStopped(self._failure or "invalid fork output") from error
            self._page_registry.commit(page_refs)
            old_heads = tuple(target.head for target in targets)
            self._replace_heads(old_heads, tuple(cursor.snapshot for cursor in cursors))
            for target, cursor in zip(targets, cursors, strict=True):
                target.cursor = cursor
            return outputs

    def prefix_lookup_batch(
        self, keys: Sequence[PrefixSemanticKey]
    ) -> tuple[PrefixLookupHint, ...]:
        with self._lock:
            self._healthy()
            values = tuple(keys)
            try:
                outputs = tuple(self.manager.prefix_lookup_batch(values))
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"prefix lookup became uncertain: {error}")
                raise FailStopped(self._failure or "prefix lookup failed") from error
            try:
                if len(outputs) != len(values):
                    raise ManagerError("prefix lookup output cardinality changed")
                for key, output in zip(values, outputs, strict=True):
                    if output.key != key:
                        raise ManagerError("prefix lookup changed the semantic key")
                    if output.candidate is None:
                        if output.resident_count != 0:
                            raise ManagerError("prefix miss returned resident pages")
                        continue
                    pages = self._page_registry.prefix_pages(output.candidate)
                    if pages is None or output.resident_count != len(pages):
                        raise ManagerError("prefix lookup returned an unknown identity")
                return outputs
            except Exception as error:
                self.fail_stop(f"prefix lookup output became uncertain: {error}")
                raise FailStopped(self._failure or "invalid prefix lookup output") from error

    def prefix_attach_batch(
        self, items: Sequence[tuple[Hashable, PrefixLookupHint]]
    ) -> tuple[AttachedPrefix, ...]:
        with self._lock:
            self._healthy()
            values = tuple(items)
            if not values or len({key for key, _hint in values}) != len(values):
                raise ManagerError("prefix attach targets must be nonempty and unique")
            try:
                records = tuple(self._requests[key] for key, _hint in values)
            except KeyError as error:
                raise ManagerError("prefix attach names an unknown request") from error
            for record in records:
                self._require_indexed_record(record)
                self._require_empty_record(record)
            raw = tuple(
                PrefixAttachItem(record.lease, record.head, hint)
                for record, (_key, hint) in zip(records, values, strict=True)
            )
            try:
                outputs = tuple(self.manager.prefix_attach_batch(raw))
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"prefix attach became uncertain: {error}")
                raise FailStopped(self._failure or "prefix attach failed") from error
            try:
                if len(outputs) != len(records):
                    raise ManagerError("manager returned wrong attach cardinality")
                cursors = tuple(
                    self._cursor_from_materialized(record.lease, output.target)
                    for record, output in zip(records, outputs, strict=True)
                )
                occupied_snapshots = self._snapshot_leases
                if (
                    len({cursor.snapshot for cursor in cursors}) != len(cursors)
                    or any(cursor.snapshot in occupied_snapshots for cursor in cursors)
                ):
                    raise ManagerError("prefix attach reused a live snapshot identity")
                for record, (_key, hint), cursor, output in zip(
                    records, values, cursors, outputs, strict=True
                ):
                    expected_pages = self._page_registry.prefix_pages(output.prefix)
                    if (
                        hint.candidate is None
                        or output.prefix != hint.candidate
                        or expected_pages is None
                        or cursor.boundary != hint.key.boundary
                        or cursor.view_version != record.cursor.view_version + 1
                        or cursor.snapshot.engine_epoch != self.engine_epoch
                        or cursor.snapshot.generation <= 0
                        or cursor.snapshot == record.head
                        or shadow_identities(tuple(cursor.pages.values()))
                        != shadow_identities(expected_pages or ())
                    ):
                        raise ManagerError("prefix attach changed the resolved prefix")
                page_refs = self._page_registry.plan(
                    (), tuple(page for cursor in cursors for page in cursor.pages.values())
                )
            except Exception as error:
                self.fail_stop(f"manager returned invalid attach output: {error}")
                raise FailStopped(self._failure or "invalid attach output") from error
            self._page_registry.commit(page_refs)
            old_heads = tuple(record.head for record in records)
            self._replace_heads(old_heads, tuple(cursor.snapshot for cursor in cursors))
            for record, cursor in zip(records, cursors, strict=True):
                record.cursor = cursor
            return outputs

    def prefix_publish_batch(
        self, items: Sequence[tuple[Hashable, PrefixSemanticKey]]
    ) -> tuple[PublishedPrefix, ...]:
        with self._lock:
            self._healthy()
            values = tuple(items)
            records = self._records_for_idle_batch(tuple(key for key, _ in values))
            raw = tuple(
                PrefixPublishItem(record.lease, record.head, semantic)
                for record, (_key, semantic) in zip(records, values, strict=True)
            )
            try:
                outputs = tuple(self.manager.prefix_publish_batch(raw))
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"prefix publication became uncertain: {error}")
                raise FailStopped(self._failure or "prefix publication failed") from error
            try:
                if len(outputs) != len(records):
                    raise ManagerError("prefix publication output cardinality changed")
                if len({item.prefix for item in outputs}) != len(outputs):
                    raise ManagerError("prefix publication duplicated an identity")
                for record, (_key, semantic), output in zip(
                    records, values, outputs, strict=True
                ):
                    if (
                        output.key != semantic
                        or output.prefix.engine_epoch != self.engine_epoch
                        or output.prefix.slot < 0
                        or output.prefix.generation <= 0
                        or output.resident_count != len(record.cursor.pages)
                    ):
                        raise ManagerError("prefix publication output identity changed")
                prefix_pages = tuple(
                    tuple(record.cursor.pages.values()) for record in records
                )
                if any(self._page_registry.has_prefix(item.prefix) for item in outputs):
                    raise ManagerError("prefix publication reused a live identity")
                page_refs = self._page_registry.plan(
                    (), tuple(page for pages in prefix_pages for page in pages)
                )
            except Exception as error:
                self.fail_stop(f"prefix publication output became uncertain: {error}")
                raise FailStopped(self._failure or "prefix publication failed") from error
            self._page_registry.commit(page_refs)
            for output, pages in zip(outputs, prefix_pages, strict=True):
                self._page_registry.install_prefix(output.prefix, pages)
            return outputs

    def prefix_publish_release_batch(
        self, items: Sequence[tuple[Hashable, PrefixSemanticKey]]
    ) -> PrefixPublishReleaseBatch:
        with self._lock:
            self._healthy()
            values = tuple(items)
            records = self._records_for_idle_batch(tuple(key for key, _ in values))
            raw = tuple(
                PrefixPublishItem(record.lease, record.head, semantic)
                for record, (_key, semantic) in zip(records, values, strict=True)
            )
            try:
                output = self.manager.prefix_publish_release_batch(raw)
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"prefix publish-release became uncertain: {error}")
                raise FailStopped(self._failure or "prefix publish-release failed") from error
            try:
                if len(output.outputs) != len(records):
                    raise ManagerError("publish-release output cardinality changed")
                publications = tuple(item.publication for item in output.outputs)
                if len({item.prefix for item in publications}) != len(publications):
                    raise ManagerError("publish-release duplicated a prefix identity")
                prefix_pages = tuple(
                    tuple(record.cursor.pages.values()) for record in records
                )
                for record, (_key, semantic), publication in zip(
                    records, values, publications, strict=True
                ):
                    if (
                        publication.key != semantic
                        or publication.prefix.engine_epoch != self.engine_epoch
                        or publication.prefix.slot < 0
                        or publication.prefix.generation <= 0
                        or publication.resident_count != len(record.cursor.pages)
                        or self._page_registry.has_prefix(publication.prefix)
                    ):
                        raise ManagerError("publish-release output identity changed")
                releases = tuple(item.release for item in output.outputs)
                self._consume_release(
                    tuple(key for key, _semantic in values),
                    records,
                    releases,
                    output.retirements,
                    tuple(zip(publications, prefix_pages, strict=True)),
                )
                self.manager.recycle_requests_batch(tuple(record.lease for record in records))
            except Exception as error:
                self.fail_stop(f"prefix publish-release consumption failed: {error}")
                raise FailStopped(self._failure or "prefix publish-release failed") from error
            for key, _semantic in values:
                record = self._requests[key]
                self._drop_request_identity(record)
                del self._requests[key]
            return output

    def prefix_evict_batch(self, prefixes: Sequence[PrefixLease]) -> PrefixEvictionBatch:
        with self._lock:
            self._healthy()
            values = tuple(prefixes)
            if (
                not values
                or len(set(values)) != len(values)
                or any(not self._page_registry.has_prefix(value) for value in values)
            ):
                raise ManagerError("prefix eviction names a non-live prefix")
            try:
                output = self.manager.prefix_evict_batch(values)
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"prefix eviction became uncertain: {error}")
                raise FailStopped(self._failure or "prefix eviction failed") from error
            try:
                if (
                    not isinstance(output, PrefixEvictionBatch)
                    or tuple(item.prefix for item in output.evicted) != values
                    or len(set(values)) != len(values)
                ):
                    raise ManagerError("prefix eviction output identity changed")
                pages = tuple(
                    page
                    for prefix in values
                    for page in (self._page_registry.prefix_pages(prefix) or ())
                )
                page_refs = self._page_registry.plan(pages, ())
                self._validate_certificates(
                    output.retirements, (), None, page_refs, pages
                )
                transaction = self._preflight_prefix_eviction(output.retirements)
                self._commit_mirrors(transaction)
                self._page_registry.commit(page_refs)
                for prefix in values:
                    self._page_registry.remove_prefix(prefix)
                self._finalize_mirrors(transaction)
                receipts = reclamation_receipts(output.retirements)
                if receipts:
                    self.manager.acknowledge_reclamations_batch(receipts)
                return output
            except Exception as error:
                self.fail_stop(f"prefix eviction became uncertain: {error}")
                raise FailStopped(self._failure or "prefix eviction failed") from error

    def prefix_recycle_batch(self, prefixes: Sequence[PrefixLease]) -> None:
        with self._lock:
            self._healthy()
            values = tuple(prefixes)
            if any(self._page_registry.has_prefix(prefix) for prefix in values):
                raise ManagerError("cannot recycle a prefix before eviction")
            try:
                self.manager.prefix_recycle_batch(values)
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"prefix recycling became uncertain: {error}")
                raise FailStopped(self._failure or "prefix recycling failed") from error

    def release_batch(self, keys: Sequence[Hashable]) -> None:
        with self._lock:
            self._healthy()
            values = tuple(keys)
            records = self._records_for_batch(values)
            prepared = tuple(
                record.pending
                for record in records
                if record.pending is not None and record.pending.phase is StepPhase.PREPARED
            )
            if prepared:
                self.abort_unobserved(
                    BatchRecord(tuple(item.key for item in prepared), prepared)
                )
            event_keys = tuple(
                key
                for key, record in zip(values, records, strict=True)
                if record.pending is not None and record.pending.phase is StepPhase.EVENT
            )
            if event_keys:
                self.wait_batch(event_keys)
            if any(record.pending is not None for record in records):
                self._quarantine_submitted(
                    tuple(record.pending for record in records if record.pending is not None)
                )
                self.fail_stop("request release raced an unproven submission")
                raise FailStopped(self._failure or "release raced submission")
            try:
                output = self.manager.release_batch(
                    tuple(ReleaseBatchItem(record.lease, record.head) for record in records)
                )
            except (RetryableConflict, ManagerError):
                raise
            except Exception as error:
                self.fail_stop(f"request release became uncertain: {error}")
                raise FailStopped(self._failure or "request release failed") from error
            try:
                self._consume_release(
                    values, records, output.releases, output.retirements
                )
                self.manager.recycle_requests_batch(tuple(record.lease for record in records))
            except Exception as error:
                self.fail_stop(f"request release consumption failed: {error}")
                raise FailStopped(self._failure or "request release failed") from error
            for key in values:
                record = self._requests[key]
                self._drop_request_identity(record)
                del self._requests[key]

    def _accept_completion_batch(
        self,
        group: EventGroup,
        records: Sequence[RequestRecord],
        receipt: BatchCompletionReceipt,
        output: CompletionBatch,
    ) -> None:
        if not isinstance(output, CompletionBatch) or len(output.completions) != len(records):
            raise ManagerError("manager returned the wrong completion cardinality")
        cursor_deltas: list[CursorDelta] = []
        detached_bindings: list[DetachedBinding] = []
        mirror_items: list[
            tuple[
                Hashable,
                RequestRecord,
                tuple[DetachedBinding, ...],
                bool,
                int,
                tuple[MirrorCandidateTransition, ...],
            ]
        ] = []
        for pending, record, completion in zip(
            group.records, records, output.completions, strict=True
        ):
            cursor_delta = completion_cursor_delta(
                record.cursor,
                pending,
                completion,
                self.arenas_by_class,
                self.config.classes,
                self.page_tokens,
                self._zero_page(),
            )
            cursor_deltas.append(cursor_delta)
            detached_bindings.extend(completion.detached)
            copy_by_destination = {
                intent.destination: intent for intent in pending.prepared.copy_intents
            }
            if len(copy_by_destination) != len(pending.prepared.copy_intents):
                raise ManagerError("COW intents duplicate a destination page")
            retired_candidates = {page.page for page in cursor_delta.retired_transient}
            candidate_transitions: list[MirrorCandidateTransition] = []
            for candidate in pending.new_pages:
                arena = self.arenas_by_class[candidate.class_id]
                begin = candidate.logical_ordinal * self.page_tokens
                end = min(begin + self.page_tokens, completion.published_boundary)
                if end <= begin:
                    raise ManagerError("candidate mirror span is empty")
                copy = copy_by_destination.get(candidate.page)
                if copy is None:
                    source = self._zero_page()
                    source_backend_index = 0
                    copied_begin = copied_end = 0
                else:
                    if (
                        copy.class_id != candidate.class_id
                        or copy.backend_domain != arena.backend_domain
                        or copy.destination_backend_index != candidate.backend_index
                    ):
                        raise ManagerError("COW candidate identity changed")
                    source = copy.source
                    source_backend_index = copy.source_backend_index
                    copied_begin = begin + copy.destination_token_offset
                    copied_end = copied_begin + copy.token_count
                    if (
                        copied_begin < begin
                        or copied_end > end
                        or copy.source_token_offset != copy.destination_token_offset
                    ):
                        raise ManagerError("COW candidate token span changed")
                candidate_transitions.append(
                    MirrorCandidateTransition(
                        destination=candidate.page,
                        source=source,
                        logical_ordinal=candidate.logical_ordinal,
                        destination_backend_index=candidate.backend_index,
                        source_backend_index=source_backend_index,
                        token_begin=begin,
                        token_end_exclusive=end,
                        copied_token_begin=copied_begin,
                        copied_token_end_exclusive=copied_end,
                        class_id=candidate.class_id,
                        backend_domain=arena.backend_domain,
                        retiring=candidate.page in retired_candidates,
                    )
                )
            mirror_items.append(
                (
                    pending.key,
                    record,
                    completion.detached,
                    False,
                    completion.published_boundary,
                    tuple(candidate_transitions),
                )
            )
        page_refs = self._page_registry.plan(
            tuple(page for delta in cursor_deltas for page in delta.removed),
            tuple(page for delta in cursor_deltas for page in delta.added),
            tuple(page for delta in cursor_deltas for page in delta.transient),
        )
        self._validate_certificates(
            output.retirements,
            tuple(detached_bindings),
            receipt,
            page_refs,
            transient_pages=tuple(
                (page, completion.published_boundary)
                for delta, completion in zip(
                    cursor_deltas, output.completions, strict=True
                )
                for page in delta.retired_transient
            ),
        )
        sliding_ids = {
            item.class_id for item in self.config.classes if item.retention == "sliding"
        }
        sliding = sum(item.class_id in sliding_ids for item in output.retirements)
        cycle_updates: list[tuple[RequestRecord, int, int, int]] = []
        for pending, record in zip(group.records, records, strict=True):
            for class_id in sliding_ids:
                period = int(self.config.classes_by_id[class_id].period_blocks)
                cycle = (
                    (pending.prepared.target_boundary - 1)
                    // self.page_tokens
                    // period
                )
                previous = record.swa_temporal_cycles.get(class_id, 0)
                if cycle < previous:
                    raise ManagerError("SWA temporal cycle moved backwards")
                cycle_updates.append((record, class_id, cycle, cycle - previous))
        transaction = self._preflight_mirrors(mirror_items, output.retirements)
        self._commit_mirrors(transaction)
        self._page_registry.commit(page_refs)
        for pending, record, completion, cursor_delta in zip(
            group.records, records, output.completions, cursor_deltas, strict=True
        ):
            cursor_delta.apply(record.cursor.pages)
            record.cursor.snapshot = completion.published_snapshot
            record.cursor.view_version = completion.published_view_version
            record.cursor.boundary = completion.published_boundary
            record.completion_domain = receipt.completion_domain
            record.completion_value = receipt.completion_value
            self._discard_candidates(pending)
        self._swa_retirement_certificates += sliding
        self._swa_pages_reclaimed += sliding
        for record, class_id, cycle, wrap_events in cycle_updates:
            self._swa_wrap_events += wrap_events
            record.swa_temporal_cycles[class_id] = cycle
        receipts = reclamation_receipts(output.retirements)
        self._finalize_mirrors(transaction)
        if receipts:
            self.manager.acknowledge_reclamations_batch(receipts)

    def _consume_release(
        self,
        keys: Sequence[Hashable],
        records: Sequence[RequestRecord],
        releases: Sequence[Any],
        retirements: Sequence[ReclamationCertificate],
        prefix_installations: Sequence[
            tuple[PublishedPrefix, Sequence[PageShadow]]
        ] = (),
    ) -> None:
        if len(releases) != len(records):
            raise ManagerError("manager returned wrong release cardinality")
        detached_bindings: list[DetachedBinding] = []
        mirror_items = []
        for key, record, release in zip(keys, records, releases, strict=True):
            if release.request != record.lease or release.detached_snapshot != record.head:
                raise ManagerError("release changed request or snapshot identity")
            projection = dict(record.cursor.pages)
            detached = self._apply_detached(
                projection, {}, release.detached, record.boundary
            )
            if projection or len(detached) != len(record.cursor.pages):
                raise ManagerError("release did not detach the whole request view")
            if any(
                item.action != DETACHED_CLEAR
                or item.reason not in (DETACHED_REQUEST_RELEASE, DETACHED_PREFIX_TRANSFER)
                for item in release.detached
            ):
                raise ManagerError("release returned an invalid detach reason")
            detached_bindings.extend(release.detached)
            mirror_items.append(
                (key, record, release.detached, True, record.boundary, ())
            )
        normalized_prefixes = tuple(
            (publication, canonical_shadows(tuple(pages)))
            for publication, pages in prefix_installations
        )
        if (
            len({publication.prefix for publication, _pages in normalized_prefixes})
            != len(normalized_prefixes)
            or any(
                self._page_registry.has_prefix(publication.prefix)
                for publication, _pages in normalized_prefixes
            )
        ):
            raise ManagerError("release transaction reused a prefix identity")
        page_refs = self._page_registry.plan(
            tuple(page for record in records for page in record.cursor.pages.values()),
            tuple(page for _publication, pages in normalized_prefixes for page in pages),
        )
        self._validate_certificates(
            retirements, tuple(detached_bindings), None, page_refs
        )
        transaction = self._preflight_mirrors(mirror_items, retirements)
        self._commit_mirrors(transaction)
        self._page_registry.commit(page_refs)
        for publication, pages in normalized_prefixes:
            self._page_registry.install_prefix(publication.prefix, pages)
        for record in records:
            record.cursor.pages.clear()
        receipts = reclamation_receipts(retirements)
        self._finalize_mirrors(transaction)
        if receipts:
            self.manager.acknowledge_reclamations_batch(receipts)

    def _apply_detached(
        self,
        projection: dict[tuple[int, int], PageShadow],
        candidates: dict[tuple[int, int], PageShadow],
        bindings: Sequence[DetachedBinding],
        resident_boundary: int,
    ) -> tuple[PageShadow, ...]:
        previous_key: tuple[int, int, int, int, int] | None = None
        detached: list[PageShadow] = []
        seen: set[tuple[int, int]] = set()
        cleared: set[tuple[int, int]] = set()
        for item in bindings:
            key = (item.class_id, item.logical_ordinal)
            order = (
                item.class_id,
                item.logical_ordinal,
                item.action,
                item.old.pool_id,
                item.old.page_id,
            )
            if item.reserved != 0 or previous_key is not None and order <= previous_key:
                raise ManagerError("detached bindings are not canonical")
            previous_key = order
            if key in seen:
                raise ManagerError("detached bindings duplicate a logical page")
            seen.add(key)
            try:
                old = projection[key]
            except KeyError as error:
                raise ManagerError("detached binding names an absent logical page") from error
            arena = self.arenas_by_class.get(item.class_id)
            if (
                arena is None
                or old.page != item.old
                or old.backend_index != item.old_backend_index
                or item.backend_domain != arena.backend_domain
                or item.token_begin != item.logical_ordinal * self.page_tokens
                or item.token_end_exclusive
                != min(
                    (item.logical_ordinal + 1) * self.page_tokens,
                    resident_boundary,
                )
            ):
                raise ManagerError("detached binding does not match the shadow view")
            detached.append(old)
            if item.action == DETACHED_CLEAR:
                if item.replacement != self._zero_page() or item.replacement_backend_index != 0:
                    raise ManagerError("clear detach carries a replacement")
                if item.reason not in (
                    DETACHED_RETENTION,
                    DETACHED_REQUEST_RELEASE,
                    DETACHED_PREFIX_TRANSFER,
                ):
                    raise ManagerError("clear detach has an invalid reason")
                del projection[key]
                cleared.add(key)
            elif item.action == DETACHED_REPLACE:
                try:
                    replacement = candidates[key]
                except KeyError as error:
                    raise ManagerError("replace detach lacks a prepared destination") from error
                if (
                    item.reason != DETACHED_COPY_ON_WRITE
                    or item.replacement != replacement.page
                    or item.replacement_backend_index != replacement.backend_index
                ):
                    raise ManagerError("replace detach changed the COW destination")
                projection[key] = replacement
            else:
                raise ManagerError("detached binding has an unknown action")
        for key, candidate in candidates.items():
            if key not in cleared and projection.get(key) != candidate:
                raise ManagerError("completion did not publish a prepared candidate")
        return tuple(detached)

    def _validate_certificates(
        self,
        certificates: Sequence[ReclamationCertificate],
        detached: Sequence[DetachedBinding],
        receipt: BatchCompletionReceipt | None,
        page_refs: PageRefPlan,
        prefix_pages: Sequence[PageShadow] = (),
        transient_pages: Sequence[tuple[PageShadow, int]] = (),
    ) -> None:
        authorities: dict[PageLease, set[tuple[int, int, int, int, int, int]]] = {}
        for item in detached:
            authority = (
                item.class_id,
                item.backend_domain,
                item.logical_ordinal,
                item.old_backend_index,
                item.token_begin,
                item.token_end_exclusive,
            )
            authorities.setdefault(item.old, set()).add(authority)
        for shadow in prefix_pages:
            arena = self.arenas_by_class[shadow.class_id]
            token_begin = shadow.logical_ordinal * self.page_tokens
            authority = (
                shadow.class_id,
                arena.backend_domain,
                shadow.logical_ordinal,
                shadow.backend_index,
                token_begin,
                token_begin + self.page_tokens,
            )
            authorities.setdefault(shadow.page, set()).add(authority)
        for shadow, boundary in transient_pages:
            arena = self.arenas_by_class[shadow.class_id]
            token_begin = shadow.logical_ordinal * self.page_tokens
            token_end = min(token_begin + self.page_tokens, boundary)
            if token_end <= token_begin:
                raise ManagerError("transient candidate has an empty token span")
            authority = (
                shadow.class_id,
                arena.backend_domain,
                shadow.logical_ordinal,
                shadow.backend_index,
                token_begin,
                token_end,
            )
            authorities.setdefault(shadow.page, set()).add(authority)
        seen_reclamations: set[ReclamationLease] = set()
        seen_pages = set()
        previous: tuple[int, ...] | None = None
        for certificate in certificates:
            # Completion uses its transition key; page-owner release/eviction
            # uses the PageLease BTree key.  Both suffixes use PageLease's ABI6
            # repr/Ord field order, never a backend-local page-number order.
            page_key = (
                certificate.page.engine_epoch,
                certificate.page.pool_epoch,
                certificate.page.generation,
                certificate.page.page_id,
                certificate.page.pool_id,
            )
            physical = (
                (certificate.class_id, certificate.logical_ordinal) + page_key
                if receipt is not None
                else page_key
            )
            if previous is not None and physical <= previous:
                raise ManagerError("reclamation certificates are not globally canonical")
            previous = physical
            if (
                certificate.reclamation in seen_reclamations
                or certificate.page in seen_pages
            ):
                raise ManagerError("reclamation batch duplicated identity")
            seen_reclamations.add(certificate.reclamation)
            seen_pages.add(certificate.page)
            if certificate.reclamation.engine_epoch != self.engine_epoch:
                raise ManagerError("reclamation belongs to another engine")
            arena = self.arenas_by_class.get(certificate.class_id)
            if arena is None or certificate.backend_domain != arena.backend_domain:
                raise ManagerError("reclamation names an unknown class or domain")
            if (
                certificate.page.engine_epoch != self.engine_epoch
                or certificate.page.pool_epoch != arena.pool_epoch
                or certificate.page.pool_id != arena.pool_id
                or certificate.page.generation <= 0
                or not arena.first_page_id
                <= certificate.page.page_id
                < arena.first_page_id + arena.page_count
            ):
                raise ManagerError("reclamation page is outside its class arena")
            if certificate.backend_index != (
                arena.backend_base_index + certificate.page.page_id - arena.first_page_id
            ):
                raise ManagerError("reclamation backend index changed")
            if (
                isinstance(certificate.logical_ordinal, bool)
                or not isinstance(certificate.logical_ordinal, int)
                or certificate.logical_ordinal < 0
                or certificate.token_begin
                != certificate.logical_ordinal * self.page_tokens
                or certificate.token_end_exclusive <= certificate.token_begin
                or certificate.token_end_exclusive
                > certificate.token_begin + self.page_tokens
            ):
                raise ManagerError("reclamation token span changed")
            if receipt is not None and (
                certificate.completion_domain != receipt.completion_domain
                or certificate.completion_value != receipt.completion_value
            ):
                raise ManagerError("reclamation completion identity changed")
            authority = (
                certificate.class_id,
                certificate.backend_domain,
                certificate.logical_ordinal,
                certificate.backend_index,
                certificate.token_begin,
                certificate.token_end_exclusive,
            )
            if authority not in authorities.get(certificate.page, set()):
                raise ManagerError("reclamation certificate has no exact batch authority")
        if seen_pages != set(self._page_registry.expected_retirements(page_refs)):
            raise ManagerError("reclamation certificates do not match last-reference pages")

    def _preflight_mirrors(
        self,
        values: Sequence[
            tuple[
                Hashable,
                RequestRecord,
                Sequence[DetachedBinding],
                bool,
                int,
                Sequence[MirrorCandidateTransition],
            ]
        ],
        retirements: Sequence[ReclamationCertificate],
    ) -> MirrorTransaction:
        grouped: dict[int, tuple[MirrorCleanupProtocol, list[MirrorCleanupItem]]] = {}
        for key, record, detached, releasing, boundary, candidates in values:
            if not detached and not candidates:
                continue
            binding = record.reclamation_cleanup
            if binding is None:
                if key in self._request_rows:
                    raise ManagerError("request mirror detach has no cleanup authority")
                continue
            identity = id(binding.coordinator)
            grouped.setdefault(identity, (binding.coordinator, []))[1].append(
                MirrorCleanupItem(
                    binding.context,
                    tuple(detached),
                    releasing,
                    boundary,
                    tuple(candidates),
                )
            )
        entries: list[tuple[MirrorCleanupProtocol, Any]] = []
        for coordinator, items in grouped.values():
            entries.append((coordinator, coordinator.preflight(tuple(items), tuple(retirements))))
        return MirrorTransaction(tuple(entries))

    def _preflight_prefix_eviction(
        self, retirements: Sequence[ReclamationCertificate]
    ) -> MirrorTransaction:
        coordinator = self._prefix_eviction_cleanup
        if not retirements:
            return MirrorTransaction(())
        if coordinator is None:
            raise ManagerError(
                "prefix eviction retirement has no global cleanup authority"
            )
        plan = coordinator.preflight((), tuple(retirements))
        return MirrorTransaction(((coordinator, plan),))

    @staticmethod
    def _commit_mirrors(transaction: MirrorTransaction) -> None:
        for coordinator, plan in transaction.entries:
            coordinator.commit(plan)
        for coordinator, plan in transaction.entries:
            coordinator.synchronize(plan)

    @staticmethod
    def _finalize_mirrors(transaction: MirrorTransaction) -> None:
        for coordinator, plan in transaction.entries:
            coordinator.finalize(plan)

    def _cursor_from_materialized(
        self, expected_request: RequestLease, materialized: MaterializedRequestView
    ) -> RequestCursor:
        view = materialized.view
        if view.request != expected_request or len(materialized.pages) != view.resident_count:
            raise ManagerError("materialized view cardinality or request changed")
        cursor = RequestCursor.from_view(view)
        previous: tuple[int, int] | None = None
        physical = set()
        for page in materialized.pages:
            order = (page.class_id, page.logical_ordinal)
            if previous is not None and order <= previous:
                raise ManagerError("materialized pages are not canonical")
            previous = order
            try:
                arena = self.arenas_by_class[page.class_id]
                class_config = self.config.classes_by_id[page.class_id]
            except KeyError as error:
                raise ManagerError("materialized page names an unknown class") from error
            shadow = page_shadow_from_snapshot(expected_request, page, arena)
            token_begin = page.logical_ordinal * self.page_tokens
            token_end = min(token_begin + self.page_tokens, view.boundary)
            retained_start = (
                0
                if class_config.retention == "full"
                else max(0, view.boundary - (int(class_config.window_tokens) - 1))
            )
            visible_begin = max(retained_start, token_begin)
            period = class_config.period_blocks
            temporal_cell = (
                page.logical_ordinal
                if period is None
                else page.logical_ordinal % int(period)
            )
            temporal_cycle = 0 if period is None else page.logical_ordinal // int(period)
            if (
                token_begin >= view.boundary
                or visible_begin >= token_end
                or page.valid_token_count != token_end - token_begin
                or page.visible_token_offset != visible_begin - token_begin
                or page.visible_token_count != token_end - visible_begin
                or page.temporal_cell_index != temporal_cell
                or page.temporal_cycle != temporal_cycle
            ):
                raise ManagerError("materialized page geometry changed")
            physical_key = (shadow.page.pool_id, shadow.page.page_id)
            if physical_key in physical:
                raise ManagerError("materialized view aliases a physical page")
            physical.add(physical_key)
            cursor.pages[order] = shadow
        return cursor

    def _validate_empty_view(self, view: RequestView) -> None:
        if (
            view.request.engine_epoch != self.engine_epoch
            or view.snapshot.engine_epoch != self.engine_epoch
            or view.request.generation <= 0
            or view.snapshot.generation <= 0
            or view.view_version != 0
            or view.boundary != 0
            or view.resident_count != 0
        ):
            raise ManagerError("acquired request view is not a valid empty head")
        _nonnegative("request slot", view.request.slot)
        _nonnegative("snapshot slot", view.snapshot.slot)

    def _validate_prepared(
        self, record: RequestRecord, prepared: PreparedStep, target: int
    ) -> None:
        if (
            prepared.request != record.lease
            or prepared.base_snapshot != record.head
            or prepared.step.engine_epoch != self.engine_epoch
            or prepared.target_snapshot.engine_epoch != self.engine_epoch
            or prepared.previous_boundary != record.boundary
            or prepared.target_boundary != target
            or prepared.base_view_version != record.cursor.view_version
            or prepared.target_view_version != prepared.base_view_version + 1
        ):
            raise ManagerError("prepared step boundary or snapshot lineage is invalid")
        _nonnegative("step slot", prepared.step.slot)
        _positive("step generation", prepared.step.generation)
        _nonnegative("target snapshot slot", prepared.target_snapshot.slot)
        _positive("target snapshot generation", prepared.target_snapshot.generation)

    def _require_empty_record(self, record: RequestRecord) -> None:
        if (
            record.pending is not None
            or record.boundary != 0
            or record.cursor.view_version != 0
            or record.cursor.pages
        ):
            raise ManagerError("target request is not empty")

    def _records_for_batch(self, keys: Sequence[Hashable]) -> tuple[RequestRecord, ...]:
        if not keys or len(set(keys)) != len(keys):
            raise ManagerError("request batch must be nonempty and unique")
        try:
            records = tuple(self._requests[key] for key in keys)
        except KeyError as error:
            raise ManagerError("batch names an unknown request") from error
        for record in records:
            self._require_indexed_record(record)
        return records

    def _records_for_idle_batch(self, keys: Sequence[Hashable]) -> tuple[RequestRecord, ...]:
        records = self._records_for_batch(keys)
        if any(record.pending is not None for record in records):
            raise ManagerError("request batch contains a pending step")
        return records

    def _discard_candidates(self, pending: StepRecord) -> None:
        for shadow in pending.new_pages:
            removed = self._candidate_pages.pop(
                self._page_registry.physical_key(shadow), None
            )
            if removed != shadow:
                raise ManagerError("candidate page journal changed identity")

    @staticmethod
    def _zero_page() -> Any:
        from .identity import PageLease

        return PageLease(0, 0, 0, 0, 0)

    def close(self) -> None:
        with self._lock:
            if self._failure is None and (
                self._events
                or self._requests
                or self._candidate_pages
                or self._page_registry
                or self._request_rows
                or self._row_owners
                or self._identity_indexes_live()
            ):
                raise ManagerError("cannot destroy a manager with live requests")
            self.manager.destroy()


__all__ = [
    "BatchRecord",
    "CanonicalRuntime",
    "RequestRecord",
    "StepPhase",
    "StepRecord",
]
