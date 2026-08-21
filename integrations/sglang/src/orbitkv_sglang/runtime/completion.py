from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Hashable, Mapping, Sequence

from .identity import (
    DETACHED_CLEAR,
    DETACHED_COPY_ON_WRITE,
    DETACHED_REPLACE,
    DETACHED_RETENTION,
    ArenaIdentity,
    FailStopped,
    ManagerError,
    PageLease,
    RequestLease,
    SnapshotLease,
    StepLease,
    SubmissionLease,
)
from .reclamation import DetachedBinding, ReclamationCertificate
from .snapshot_shadow import (
    CopyIntent,
    PageShadow,
    PreparedStep,
    RequestCursor,
    _positive,
)


@dataclass(frozen=True, slots=True)
class BackendBindReceipt:
    step: StepLease
    page: PageLease
    backend_domain: int
    mapped: int = 1
    writable: int = 1
    reserved: int = 0
    backend_index: int = 0


@dataclass(frozen=True, slots=True)
class BackendCopyReceipt:
    step: StepLease
    class_id: int
    backend_domain: int
    token_count: int
    source_token_offset: int
    destination_token_offset: int
    source: PageLease
    destination: PageLease
    source_backend_index: int
    destination_backend_index: int
    observed: int = 1
    copied: int = 1
    ordered_before_writes: int = 1
    reserved8: int = 0
    reserved32: int = 0


@dataclass(frozen=True, slots=True)
class SubmittedStep:
    submission: SubmissionLease
    request: RequestLease
    target_snapshot: SnapshotLease


@dataclass(frozen=True, slots=True)
class BatchCompletionReceipt:
    engine_epoch: int
    completion_domain: int
    completion_value: int
    confirmed: int = 1
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class BackendUnobservedReceipt:
    step: StepLease
    backend_unobserved: int = 1
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class StepCompletion:
    submission: SubmissionLease
    request: RequestLease
    detached_snapshot: SnapshotLease
    published_snapshot: SnapshotLease
    published_view_version: int
    published_boundary: int
    resident_count: int
    detached: tuple[DetachedBinding, ...]


@dataclass(frozen=True, slots=True)
class CompletionBatch:
    completions: tuple[StepCompletion, ...]
    retirements: tuple[ReclamationCertificate, ...]


class StepPhase(Enum):
    PREPARED = auto()
    LOWERED = auto()
    SUBMITTED = auto()
    FORWARD = auto()
    EVENT = auto()
    COMPLETED = auto()
    ABORTED = auto()
    QUARANTINED = auto()


@dataclass(slots=True)
class StepRecord:
    key: Hashable
    prepared: PreparedStep
    new_pages: tuple[PageShadow, ...] = ()
    phase: StepPhase = StepPhase.PREPARED
    bind_receipts: tuple[BackendBindReceipt, ...] = ()
    copy_receipts: tuple[BackendCopyReceipt, ...] = ()
    submitted: SubmittedStep | None = None


@dataclass(frozen=True, slots=True)
class BatchRecord:
    keys: tuple[Hashable, ...]
    records: tuple[StepRecord, ...]

    def __post_init__(self) -> None:
        if not self.records or len(self.keys) != len(self.records):
            raise ManagerError("batch journal cardinality is invalid")
        if tuple(record.key for record in self.records) != self.keys:
            raise ManagerError("batch journal keys do not match its ordered records")


@dataclass(slots=True)
class EventGroup:
    event: Any
    batch: BatchRecord
    completion_domain: int

    @property
    def records(self) -> tuple[StepRecord, ...]:
        return self.batch.records


@dataclass(frozen=True, slots=True)
class CursorDelta:
    removed: tuple[PageShadow, ...]
    added: tuple[PageShadow, ...]
    transient: tuple[PageShadow, ...]
    retired_transient: tuple[PageShadow, ...]
    detached: tuple[PageShadow, ...]

    def apply(self, pages: dict[tuple[int, int], PageShadow]) -> None:
        for shadow in self.removed:
            key = (shadow.class_id, shadow.logical_ordinal)
            if pages.pop(key) != shadow:
                raise AssertionError("validated cursor removal changed identity")
        for shadow in self.added:
            key = (shadow.class_id, shadow.logical_ordinal)
            if key in pages:
                raise AssertionError("validated cursor addition replaced a live page")
            pages[key] = shadow


def completion_cursor_delta(
    cursor: RequestCursor,
    pending: StepRecord,
    item: StepCompletion,
    arenas: Mapping[int, ArenaIdentity],
    classes: Sequence[Any],
    page_tokens: int,
    zero_page: PageLease,
) -> CursorDelta:
    """Validate a publication without traversing the resident snapshot."""

    if pending.submitted is None:
        raise ManagerError("completion lacks a submitted operation")
    if (
        item.submission != pending.submitted.submission
        or item.request != cursor.lease
        or item.detached_snapshot != pending.prepared.base_snapshot
        or item.published_snapshot != pending.prepared.target_snapshot
        or item.published_view_version != pending.prepared.target_view_version
        or item.published_boundary != pending.prepared.target_boundary
    ):
        raise ManagerError("completion identity or snapshot lineage changed")

    candidates = {
        (shadow.class_id, shadow.logical_ordinal): shadow
        for shadow in pending.new_pages
    }
    if len(candidates) != len(pending.new_pages):
        raise ManagerError("completion candidate aliases a logical page")

    target_ranges: dict[int, tuple[int, int]] = {}
    expected_removed: set[tuple[int, int]] = set()
    expected_resident_count = 0
    for class_config in classes:
        class_id = int(class_config.class_id)
        old_end = (pending.prepared.previous_boundary + page_tokens - 1) // page_tokens
        new_end = (pending.prepared.target_boundary + page_tokens - 1) // page_tokens
        if class_config.retention == "full":
            old_start = new_start = 0
        elif class_config.retention == "sliding":
            window = int(class_config.window_tokens)
            old_start = max(
                0, pending.prepared.previous_boundary - (window - 1)
            ) // page_tokens
            new_start = max(
                0, pending.prepared.target_boundary - (window - 1)
            ) // page_tokens
        else:
            raise ManagerError("completion names an unsupported retention class")
        if old_start > old_end or new_start > new_end:
            raise ManagerError("completion retention range is invalid")
        target_ranges[class_id] = (new_start, new_end)
        expected_removed.update(
            (class_id, ordinal)
            for ordinal in range(old_start, min(old_end, new_start))
        )
        expected_resident_count += new_end - new_start
    if item.resident_count != expected_resident_count:
        raise ManagerError("completion returned the wrong retained-root census")

    cow_sources = {intent.source for intent in pending.prepared.copy_intents}

    previous_order: tuple[int, int, int, int, int] | None = None
    seen: set[tuple[int, int]] = set()
    consumed_candidates: set[tuple[int, int]] = set()
    removed: dict[tuple[int, int], PageShadow] = {}
    cleared_published: set[tuple[int, int]] = set()
    added: dict[tuple[int, int], PageShadow] = {}
    detached: list[PageShadow] = []
    for binding in item.detached:
        key = (binding.class_id, binding.logical_ordinal)
        order = (
            binding.class_id,
            binding.logical_ordinal,
            binding.action,
            binding.old.pool_id,
            binding.old.page_id,
        )
        if (
            binding.reserved != 0
            or previous_order is not None
            and order <= previous_order
            or key in seen
        ):
            raise ManagerError("detached bindings are not canonical")
        previous_order = order
        seen.add(key)

        published = cursor.pages.get(key)
        candidate = candidates.get(key)
        old = published if published is not None else candidate
        if old is None:
            raise ManagerError("detached binding names an absent logical page")
        arena = arenas.get(binding.class_id)
        if (
            arena is None
            or old.page != binding.old
            or old.backend_index != binding.old_backend_index
            or binding.backend_domain != arena.backend_domain
            or binding.token_begin != binding.logical_ordinal * page_tokens
        ):
            raise ManagerError("detached binding does not match the shadow view")
        resident_boundary = (
            pending.prepared.previous_boundary
            if binding.old in cow_sources
            else item.published_boundary
        )
        expected_token_end = min(
            (binding.logical_ordinal + 1) * page_tokens, resident_boundary
        )
        if binding.token_end_exclusive != expected_token_end:
            raise ManagerError("detached binding changed the resident token span")
        detached.append(old)

        if binding.action == DETACHED_CLEAR:
            if (
                binding.replacement != zero_page
                or binding.replacement_backend_index != 0
                or binding.reason != DETACHED_RETENTION
            ):
                raise ManagerError("completion clear detach is invalid")
            if published is not None:
                removed[key] = published
                cleared_published.add(key)
            if candidate is not None:
                consumed_candidates.add(key)
        elif binding.action == DETACHED_REPLACE:
            if published is None or candidate is None:
                raise ManagerError("replace detach lacks an old or destination page")
            if (
                binding.old not in cow_sources
                or binding.reason != DETACHED_COPY_ON_WRITE
                or binding.replacement != candidate.page
                or binding.replacement_backend_index != candidate.backend_index
            ):
                raise ManagerError("replace detach changed the COW destination")
            removed[key] = published
            added[key] = candidate
            consumed_candidates.add(key)
        else:
            raise ManagerError("detached binding has an unknown action")

    for key, candidate in candidates.items():
        retained = key[0] in target_ranges and (
            target_ranges[key[0]][0] <= key[1] < target_ranges[key[0]][1]
        )
        if retained:
            if key in consumed_candidates and key not in added:
                raise ManagerError("completion retired a retained candidate")
            if key in cursor.pages and key not in added:
                raise ManagerError("completion overwrote a page without a replace detach")
            added.setdefault(key, candidate)
        elif key not in consumed_candidates:
            raise ManagerError("completion retained an out-of-window candidate")

    if cleared_published != expected_removed:
        raise ManagerError("completion detached the wrong retention ordinals")
    resident_count = len(cursor.pages) - len(removed) + len(added)
    if resident_count != expected_resident_count:
        raise ManagerError("completion delta disagrees with the retained-root census")
    return CursorDelta(
        tuple(removed.values()),
        tuple(added.values()),
        pending.new_pages,
        tuple(
            candidate for key, candidate in candidates.items() if key not in added
        ),
        tuple(detached),
    )


def bind_receipts(
    prepared: PreparedStep,
    new_pages: Sequence[PageShadow],
    arenas: Mapping[int, ArenaIdentity],
) -> tuple[BackendBindReceipt, ...]:
    return tuple(
        BackendBindReceipt(
            step=prepared.step,
            page=shadow.page,
            backend_domain=arenas[shadow.class_id].backend_domain,
            backend_index=shadow.backend_index,
        )
        for shadow in new_pages
    )


def copy_receipts(
    prepared: PreparedStep,
    intents: Sequence[CopyIntent] | None = None,
) -> tuple[BackendCopyReceipt, ...]:
    values = prepared.copy_intents if intents is None else tuple(intents)
    return tuple(
        BackendCopyReceipt(
            step=prepared.step,
            class_id=intent.class_id,
            backend_domain=intent.backend_domain,
            token_count=intent.token_count,
            source_token_offset=intent.source_token_offset,
            destination_token_offset=intent.destination_token_offset,
            source=intent.source,
            destination=intent.destination,
            source_backend_index=intent.source_backend_index,
            destination_backend_index=intent.destination_backend_index,
        )
        for intent in values
    )


class CompletionRuntimeMixin:
    def mark_lowered(self, batch: BatchRecord) -> None:
        """Record that every advertised bind and COW copy completed exactly."""

        with self._lock:
            self._healthy()
            self._validate_batch_record(batch)
            for pending in batch.records:
                self._require_pending(pending, StepPhase.PREPARED)
            for pending in batch.records:
                pending.bind_receipts = bind_receipts(
                    pending.prepared, pending.new_pages, self.arenas_by_class
                )
                pending.copy_receipts = copy_receipts(pending.prepared)
                pending.phase = StepPhase.LOWERED

    def lowering_failed(self, batch: BatchRecord, error: BaseException) -> None:
        with self._lock:
            steps = tuple(
                pending.prepared.step
                for pending in batch.records
                if pending.phase in (StepPhase.PREPARED, StepPhase.LOWERED)
            )
            self._best_effort_quarantine_steps(steps)
            for pending in batch.records:
                if pending.phase in (StepPhase.PREPARED, StepPhase.LOWERED):
                    pending.phase = StepPhase.QUARANTINED
            self.fail_stop(f"backend lowering became uncertain: {error}")

    def submit_batch(self, batch: BatchRecord) -> tuple[SubmittedStep, ...]:
        with self._lock:
            self._healthy()
            self._validate_batch_record(batch)
            records = tuple(
                self._require_pending(pending, StepPhase.LOWERED)
                for pending in batch.records
            )
            assumed = tuple(
                SubmissionLease(
                    pending.prepared.step.engine_epoch,
                    pending.prepared.step.slot,
                    pending.prepared.step.generation,
                )
                for pending in batch.records
            )
            try:
                submitted_values = tuple(
                    self.manager.submit_batch(
                        tuple(
                            (
                                pending.prepared.step,
                                pending.bind_receipts,
                                pending.copy_receipts,
                            )
                            for pending in batch.records
                        )
                    )
                )
            except Exception as error:
                self._best_effort_quarantine_submissions(assumed)
                for pending in batch.records:
                    pending.phase = StepPhase.QUARANTINED
                self.fail_stop(f"manager batch submission became uncertain: {error}")
                raise FailStopped(self._failure or "manager batch submission failed") from error
            try:
                if len(submitted_values) != len(batch.records):
                    raise ManagerError("manager returned the wrong submitted item count")
                for record, pending, submitted in zip(
                    records, batch.records, submitted_values, strict=True
                ):
                    self._validate_submitted(record, pending, submitted)
            except Exception as error:
                self._best_effort_quarantine_submissions(assumed)
                for pending in batch.records:
                    pending.phase = StepPhase.QUARANTINED
                self.fail_stop(f"manager returned an invalid submission batch: {error}")
                raise FailStopped(self._failure or "invalid submission batch") from error
            for pending, submitted in zip(batch.records, submitted_values, strict=True):
                pending.submitted = submitted
                pending.phase = StepPhase.SUBMITTED
            return submitted_values

    def candidate_mirror_failed(self, batch: BatchRecord, error: BaseException) -> None:
        with self._lock:
            self._quarantine_submitted(batch.records)
            self.fail_stop(f"ReqToToken candidate mirror became uncertain: {error}")

    def pre_forward_failed(self, error: BaseException) -> None:
        with self._lock:
            unobserved = tuple(
                record.pending
                for record in self._requests.values()
                if record.pending is not None
                and record.pending.phase in (StepPhase.SUBMITTED, StepPhase.FORWARD)
            )
            self._quarantine_submitted(unobserved)
            self.fail_stop(f"pre-forward scheduling became uncertain: {error}")
            raise FailStopped(self._failure or "pre-forward scheduling failed") from error

    def mark_forward(self, batch: BatchRecord) -> None:
        with self._lock:
            self._healthy()
            self._validate_batch_record(batch)
            for pending in batch.records:
                self._require_pending(pending, StepPhase.SUBMITTED)
            for pending in batch.records:
                pending.phase = StepPhase.FORWARD

    def forward_failed(self, batch: BatchRecord, error: BaseException) -> None:
        with self._lock:
            self._quarantine_submitted(batch.records)
            self.fail_stop(f"forward launch became uncertain: {error}")

    def register_event(
        self, batch: BatchRecord, event: Any, completion_domain: int
    ) -> None:
        with self._lock:
            self._healthy()
            _positive("completion domain", int(completion_domain))
            self._validate_batch_record(batch)
            for pending in batch.records:
                self._require_pending(pending, StepPhase.FORWARD)
            self._events.append(EventGroup(event, batch, int(completion_domain)))
            self._runtime_counters["forward_events"] += 1
            for pending in batch.records:
                pending.phase = StepPhase.EVENT

    def event_registration_failed(
        self, batch: BatchRecord, error: BaseException
    ) -> None:
        with self._lock:
            self._quarantine_submitted(batch.records)
            self.fail_stop(f"CUDA completion event registration became uncertain: {error}")

    def abort_unobserved(self, batch: BatchRecord) -> None:
        with self._lock:
            self._healthy()
            self._validate_batch_record(batch)
            for pending in batch.records:
                self._require_pending(pending, StepPhase.PREPARED)
            try:
                self.manager.abort_steps_batch(
                    tuple(
                        BackendUnobservedReceipt(pending.prepared.step)
                        for pending in batch.records
                    )
                )
            except Exception as error:
                self._best_effort_quarantine_steps(
                    tuple(pending.prepared.step for pending in batch.records)
                )
                self.fail_stop(f"unobserved step abort became uncertain: {error}")
                raise FailStopped(self._failure or "step abort failed") from error
            for pending in batch.records:
                self._discard_candidates(pending)
                self._abort_prepared_identity(pending)
                self._requests[pending.key].pending = None
                pending.phase = StepPhase.ABORTED

    def poll(self) -> None:
        with self._lock:
            self._healthy()
            ready: list[EventGroup] = []
            for group in tuple(self._events):
                try:
                    self._runtime_counters["event_queries"] += 1
                    complete = bool(group.event.query())
                except Exception as error:
                    self._quarantine_submitted(group.records)
                    self.fail_stop(f"CUDA event query became uncertain: {error}")
                    raise FailStopped(self._failure or "event query failed") from error
                if complete:
                    ready.append(group)
            for group in ready:
                self._complete_group(group)

    def wait(self, key: Hashable) -> None:
        self.wait_batch((key,))

    def wait_batch(self, keys: Sequence[Hashable]) -> None:
        with self._lock:
            self._healthy()
            requested = set(keys)
            groups = [
                group
                for group in self._events
                if any(record.key in requested for record in group.records)
            ]
            for group in groups:
                try:
                    self._runtime_counters["event_waits"] += 1
                    group.event.synchronize()
                except Exception as error:
                    self._quarantine_submitted(group.records)
                    self.fail_stop(f"CUDA event synchronization became uncertain: {error}")
                    raise FailStopped(self._failure or "event synchronization failed") from error
                self._complete_group(group)

    def _complete_group(self, group: EventGroup) -> None:
        if group not in self._events:
            return
        receipt = BatchCompletionReceipt(
            engine_epoch=self.engine_epoch,
            completion_domain=group.completion_domain,
            completion_value=self._completion_value,
        )
        try:
            records = tuple(
                self._require_pending(pending, StepPhase.EVENT)
                for pending in group.records
            )
            submissions = tuple(
                pending.submitted.submission
                for pending in group.records
                if pending.submitted is not None
            )
            if len(submissions) != len(group.records):
                raise ManagerError("event group lost a submitted identity")
            output = self.manager.complete_batch(receipt, submissions)
            self._accept_completion_batch(group, records, receipt, output)
            self._completion_value += 1
            self._runtime_counters["completion_values"] += 1
            for pending in group.records:
                self._complete_prepared_identity(pending)
                self._requests[pending.key].pending = None
                pending.phase = StepPhase.COMPLETED
            self._events.remove(group)
        except Exception as error:
            remaining = [
                pending
                for pending in group.records
                if pending.submitted is not None and pending.phase is StepPhase.EVENT
            ]
            self._quarantine_submitted(remaining)
            self.fail_stop(f"GPU completion publication became uncertain: {error}")
            raise FailStopped(self._failure or "completion failed") from error

    def _validate_submitted(self, record: Any, pending: StepRecord, submitted: SubmittedStep) -> None:
        expected_lease = SubmissionLease(
            pending.prepared.step.engine_epoch,
            pending.prepared.step.slot,
            pending.prepared.step.generation,
        )
        if (
            submitted.request != record.lease
            or submitted.submission != expected_lease
            or submitted.target_snapshot != pending.prepared.target_snapshot
        ):
            raise ManagerError("submission identity or target snapshot changed")

    def _require_pending(self, pending: StepRecord, expected: StepPhase) -> Any:
        try:
            record = self._requests[pending.key]
        except KeyError as error:
            raise ManagerError("step request has already been released") from error
        if record.pending is not pending or pending.phase is not expected:
            raise ManagerError(
                f"step phase mismatch: expected {expected.name.lower()}, "
                f"got {pending.phase.name.lower()}"
            )
        self._require_indexed_record(record)
        return record

    @staticmethod
    def _validate_batch_record(batch: BatchRecord) -> None:
        if not isinstance(batch, BatchRecord):
            raise ManagerError("operation requires one ABI6 batch journal")
        if len({id(record) for record in batch.records}) != len(batch.records):
            raise ManagerError("step batch contains a duplicate record")
        if len(set(batch.keys)) != len(batch.keys):
            raise ManagerError("step batch contains a duplicate request key")

    def _quarantine_submitted(self, records: Sequence[StepRecord]) -> None:
        steps: list[StepLease] = []
        submissions: list[SubmissionLease] = []
        for pending in records:
            if pending.phase is StepPhase.QUARANTINED:
                continue
            if pending.submitted is None:
                steps.append(pending.prepared.step)
            else:
                submissions.append(pending.submitted.submission)
            pending.phase = StepPhase.QUARANTINED
            self._runtime_counters["quarantine_count"] += 1
        self._best_effort_quarantine_steps(tuple(steps))
        self._best_effort_quarantine_submissions(tuple(submissions))

    def _best_effort_quarantine_steps(self, steps: Sequence[StepLease]) -> None:
        if steps:
            try:
                self.manager.quarantine_steps_batch(tuple(steps))
            except Exception:
                pass

    def _best_effort_quarantine_submissions(
        self, submissions: Sequence[SubmissionLease]
    ) -> None:
        if submissions:
            try:
                self.manager.quarantine_submissions_batch(tuple(submissions))
            except Exception:
                pass


__all__ = [
    "BackendBindReceipt",
    "BackendCopyReceipt",
    "BackendUnobservedReceipt",
    "BatchCompletionReceipt",
    "BatchRecord",
    "CompletionBatch",
    "EventGroup",
    "StepCompletion",
    "StepPhase",
    "StepRecord",
    "SubmittedStep",
    "bind_receipts",
    "copy_receipts",
]
