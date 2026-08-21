from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum, auto
from threading import RLock
from typing import Any, Hashable, Mapping, Protocol, Sequence, runtime_checkable


CLASS_LOWERING_HAS_PREVIOUS_TAIL = 1 << 0


class ManagerError(RuntimeError):
    """The canonical manager rejected an operation or returned invalid data."""


class FailStopped(RuntimeError):
    """The adapter quarantined uncertain state and cannot safely continue."""


@dataclass(frozen=True, slots=True)
class ArenaIdentity:
    engine_epoch: int
    pool_epoch: int
    pool_id: int
    class_id: int
    backend_domain: int
    page_count: int
    page_tokens: int
    backend_base_index: int
    first_page_id: int


@dataclass(frozen=True, slots=True)
class ArenaRegistration:
    class_id: int
    pool_id: int
    backend_domain: int
    page_count: int
    backend_base_index: int = 0


@dataclass(frozen=True, slots=True)
class ArenaStats:
    engine_epoch: int
    pool_epoch: int
    pool_id: int
    page_count: int
    class_id: int
    backend_domain: int
    first_page_id: int
    free_pages: int
    reserved_pages: int
    writing_pages: int
    active_pages: int
    retiring_pages: int
    quarantined_pages: int
    exhausted_pages: int


@dataclass(frozen=True, slots=True)
class ManagerCreateSettings:
    maximum_requests: int
    maximum_operations: int
    maximum_reclamations: int
    maximum_step_tokens: int


@dataclass(frozen=True, slots=True)
class RequestLease:
    engine_epoch: int
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class StepLease:
    engine_epoch: int
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class SubmissionLease:
    engine_epoch: int
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class ReclamationLease:
    engine_epoch: int
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class PageLease:
    engine_epoch: int
    pool_epoch: int
    generation: int
    page_id: int
    pool_id: int


@dataclass(frozen=True, slots=True)
class WriteIntent:
    page_generation: int
    page_id: int
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class ClassLowering:
    class_id: int
    flags: int
    write_offset: int
    write_count: int
    previous_tail_page_id: int
    previous_tail_generation: int


@dataclass(frozen=True, slots=True)
class PreparedStep:
    step: StepLease
    request: RequestLease
    base_view_version: int
    target_view_version: int
    previous_boundary: int
    target_boundary: int
    class_lowerings: tuple[ClassLowering, ...]
    write_intents: tuple[WriteIntent, ...]


@dataclass(frozen=True, slots=True)
class PrepareBatchItem:
    request: RequestLease
    target_boundary: int


@dataclass(frozen=True, slots=True)
class BackendBindReceipt:
    step: StepLease
    page: PageLease
    backend_domain: int
    mapped: int
    writable: int
    reserved: int
    backend_index: int


@dataclass(frozen=True, slots=True)
class SubmittedStep:
    submission: SubmissionLease
    request: RequestLease


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
class ReclamationCertificate:
    reclamation: ReclamationLease
    request: RequestLease
    page: PageLease
    class_id: int
    backend_domain: int
    logical_ordinal: int
    backend_index: int
    token_begin: int
    token_end_exclusive: int
    completion_domain: int
    completion_value: int


@dataclass(frozen=True, slots=True)
class ReclamationReceipt:
    reclamation: ReclamationLease
    page: PageLease
    backend_domain: int
    acknowledged: int
    reserved8: int
    reserved32: int
    backend_index: int


@dataclass(frozen=True, slots=True)
class StepCompletion:
    submission: SubmissionLease
    request: RequestLease
    published_view_version: int
    published_boundary: int
    resident_count: int
    retirements: tuple[ReclamationCertificate, ...] = ()


@dataclass(frozen=True, slots=True)
class ReleaseCompletion:
    request: RequestLease
    retirements: tuple[ReclamationCertificate, ...] = ()


@dataclass(frozen=True, slots=True)
class ManagerStats:
    active_requests: int
    prepared_steps: int
    submitted_steps: int
    free_pages: int
    reserved_pages: int
    writing_pages: int
    active_pages: int
    retiring_pages: int
    quarantined_pages: int
    exhausted_pages: int
    pending_reclamations: int


@dataclass(frozen=True, slots=True)
class SwaActivity:
    retirement_certificates: int
    pages_reclaimed: int
    wrap_events: int


@dataclass(frozen=True, slots=True)
class MirrorCleanupItem:
    context: Any
    certificates: tuple[ReclamationCertificate, ...]
    releasing: bool
    boundary: int


@runtime_checkable
class MirrorCleanupProtocol(Protocol):
    def preflight(self, items: Sequence[MirrorCleanupItem]) -> Any: ...

    def commit(self, plan: Any) -> None: ...

    def synchronize(self, plan: Any) -> None: ...

    def finalize(self, plan: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class MirrorCleanupBinding:
    coordinator: MirrorCleanupProtocol
    context: Any


@runtime_checkable
class ManagerProtocol(Protocol):
    @property
    def arenas(self) -> tuple[ArenaIdentity, ...]: ...

    @property
    def arenas_by_class(self) -> dict[int, ArenaIdentity]: ...

    def arena_stats(self) -> tuple[ArenaStats, ...]: ...

    def request_acquire_batch(self, request_count: int) -> tuple[RequestLease, ...]: ...

    def prepare_batch(
        self, items: Sequence[PrepareBatchItem]
    ) -> tuple[PreparedStep, ...]: ...

    def submit_batch(
        self,
        items: Sequence[tuple[StepLease, Sequence[BackendBindReceipt]]],
    ) -> tuple[SubmittedStep, ...]: ...

    def complete_batch(
        self,
        receipt: BatchCompletionReceipt,
        submissions: Sequence[SubmissionLease],
    ) -> tuple[StepCompletion, ...]: ...

    def abort_steps(self, receipts: Sequence[BackendUnobservedReceipt]) -> None: ...

    def quarantine_steps(self, steps: Sequence[StepLease]) -> None: ...

    def quarantine_submissions(
        self, submissions: Sequence[SubmissionLease]
    ) -> None: ...

    def release_batch(
        self, requests: Sequence[RequestLease]
    ) -> tuple[ReleaseCompletion, ...]: ...

    def acknowledge_reclamations(
        self, receipts: Sequence[ReclamationReceipt]
    ) -> None: ...

    def recycle_requests(self, requests: Sequence[RequestLease]) -> None: ...

    def stats(self) -> ManagerStats: ...

    def destroy(self) -> None: ...


@runtime_checkable
class ManagerFactoryProtocol(Protocol):
    def create(
        self,
        config: Any,
        settings: ManagerCreateSettings,
        arenas: Sequence[ArenaRegistration],
    ) -> ManagerProtocol: ...


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


@dataclass(frozen=True, slots=True)
class PageShadow:
    request: RequestLease
    class_id: int
    logical_ordinal: int
    page: PageLease
    backend_index: int


@dataclass(slots=True)
class RequestCursor:
    lease: RequestLease
    view_version: int = 0
    boundary: int = 0
    pages: dict[tuple[int, int], PageShadow] = field(default_factory=dict)


@dataclass(slots=True)
class RequestRecord:
    cursor: RequestCursor
    pending: StepRecord | None = None
    completion_domain: int = 0
    completion_value: int = 0
    reclamation_cleanup: MirrorCleanupBinding | None = None
    swa_temporal_cycles: dict[int, int] = field(default_factory=dict)

    @property
    def lease(self) -> RequestLease:
        return self.cursor.lease

    @property
    def boundary(self) -> int:
        return self.cursor.boundary


@dataclass(slots=True)
class EventGroup:
    event: Any
    batch: BatchRecord
    completion_domain: int

    @property
    def records(self) -> tuple[StepRecord, ...]:
        return self.batch.records


@dataclass(frozen=True, slots=True)
class ClassLoweringSpec:
    class_id: int
    pool_id: int
    last_location: int
    exact_new_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LoweringPlan:
    request: RequestLease
    previous_boundary: int
    target_boundary: int
    class_specs: tuple[ClassLoweringSpec, ...]

    @property
    def by_class(self) -> dict[int, ClassLoweringSpec]:
        return {item.class_id: item for item in self.class_specs}


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManagerError(f"{name} must be a positive integer")


def _nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManagerError(f"{name} must be a nonnegative integer")


def _ceil_pages(tokens: int, page_tokens: int) -> int:
    return (tokens + page_tokens - 1) // page_tokens


def expected_new_ordinals(previous: int, target: int, page_tokens: int) -> tuple[int, ...]:
    if previous < 0 or target < previous:
        raise ManagerError("step boundaries are not monotonic")
    return tuple(range(_ceil_pages(previous, page_tokens), _ceil_pages(target, page_tokens)))


def sglang_page_id(backend_index: int, backend_base_index: int) -> int:
    page_id = backend_index - backend_base_index + 1
    if page_id <= 0:
        raise ManagerError("manager backend index cannot lower into the SGLang arena")
    return page_id


def _shadow_from_intent(
    request: RequestLease,
    class_id: int,
    logical_ordinal: int,
    intent: WriteIntent,
    arena: ArenaIdentity,
) -> PageShadow:
    if intent.reserved != 0 or intent.page_generation <= 0:
        raise ManagerError("write intent has an invalid generation or reserved field")
    if not arena.first_page_id <= intent.page_id < arena.first_page_id + arena.page_count:
        raise ManagerError("write intent page is outside its class arena")
    backend_index = arena.backend_base_index + intent.page_id - arena.first_page_id
    return PageShadow(
        request=request,
        class_id=class_id,
        logical_ordinal=logical_ordinal,
        page=PageLease(
            engine_epoch=request.engine_epoch,
            pool_epoch=arena.pool_epoch,
            generation=intent.page_generation,
            page_id=intent.page_id,
            pool_id=arena.pool_id,
        ),
        backend_index=backend_index,
    )


def _decode_prepared(
    cursor: RequestCursor,
    prepared: PreparedStep,
    arenas: Mapping[int, ArenaIdentity],
    config: Any,
) -> tuple[LoweringPlan, tuple[PageShadow, ...]]:
    if len(prepared.class_lowerings) != len(config.classes):
        raise ManagerError("prepare returned the wrong class cardinality")
    class_specs: list[ClassLoweringSpec] = []
    new_pages: list[PageShadow] = []
    physical: set[tuple[int, int]] = set()
    write_cursor = 0
    expected_new = expected_new_ordinals(
        prepared.previous_boundary, prepared.target_boundary, int(config.page_tokens)
    )
    for class_config, lowering in zip(
        config.classes, prepared.class_lowerings, strict=True
    ):
        arena = arenas[class_config.class_id]
        if lowering.class_id != class_config.class_id:
            raise ManagerError("prepare class lowerings are not in compiled order")
        if lowering.flags & ~CLASS_LOWERING_HAS_PREVIOUS_TAIL:
            raise ManagerError("prepare returned unknown class-lowering flags")
        if lowering.write_offset != write_cursor:
            raise ManagerError("prepare class write spans are not gap-free")
        write_end = lowering.write_offset + lowering.write_count
        if write_end < lowering.write_offset or write_end > len(prepared.write_intents):
            raise ManagerError("prepare class write span is out of range")
        if lowering.write_count != len(expected_new):
            raise ManagerError("prepare reserved the wrong number of class pages")

        has_tail = bool(prepared.previous_boundary % arena.page_tokens)
        expected_flags = CLASS_LOWERING_HAS_PREVIOUS_TAIL if has_tail else 0
        if lowering.flags != expected_flags:
            raise ManagerError("prepare returned the wrong previous-tail presence")
        if has_tail:
            tail_key = (
                class_config.class_id,
                prepared.previous_boundary // arena.page_tokens,
            )
            try:
                tail = cursor.pages[tail_key]
            except KeyError as error:
                raise ManagerError("published cursor does not contain its partial tail") from error
            if (
                lowering.previous_tail_page_id != tail.page.page_id
                or lowering.previous_tail_generation != tail.page.generation
            ):
                raise ManagerError("prepare changed the published previous tail")
            last_location = (
                sglang_page_id(tail.backend_index, arena.backend_base_index)
                * arena.page_tokens
                + prepared.previous_boundary % arena.page_tokens
                - 1
            )
        else:
            if lowering.previous_tail_page_id or lowering.previous_tail_generation:
                raise ManagerError("prepare returned tail identity at an aligned boundary")
            last_location = -1

        exact_new_pages: list[int] = []
        for ordinal, intent in zip(
            expected_new,
            prepared.write_intents[lowering.write_offset:write_end],
            strict=True,
        ):
            shadow = _shadow_from_intent(
                prepared.request, class_config.class_id, ordinal, intent, arena
            )
            physical_key = (shadow.page.pool_id, shadow.page.page_id)
            if physical_key in physical:
                raise ManagerError("prepare aliases a physical page within the request")
            physical.add(physical_key)
            new_pages.append(shadow)
            exact_new_pages.append(
                sglang_page_id(shadow.backend_index, arena.backend_base_index)
            )
        class_specs.append(
            ClassLoweringSpec(
                class_id=class_config.class_id,
                pool_id=class_config.pool_id,
                last_location=last_location,
                exact_new_pages=tuple(exact_new_pages),
            )
        )
        write_cursor = write_end
    if write_cursor != len(prepared.write_intents):
        raise ManagerError("prepare class spans do not cover all write intents")
    return (
        LoweringPlan(
            request=prepared.request,
            previous_boundary=prepared.previous_boundary,
            target_boundary=prepared.target_boundary,
            class_specs=tuple(class_specs),
        ),
        tuple(new_pages),
    )


def lowering_plan(
    cursor: RequestCursor,
    prepared: PreparedStep,
    arenas: Mapping[int, ArenaIdentity],
    config: Any,
) -> LoweringPlan:
    return _decode_prepared(cursor, prepared, arenas, config)[0]


def bind_receipts(
    prepared: PreparedStep,
    new_pages: Sequence[PageShadow],
    arenas: Mapping[int, ArenaIdentity],
) -> tuple[BackendBindReceipt, ...]:
    result: list[BackendBindReceipt] = []
    for shadow in new_pages:
        arena = arenas[shadow.class_id]
        result.append(
            BackendBindReceipt(
                step=prepared.step,
                page=shadow.page,
                backend_domain=arena.backend_domain,
                mapped=1,
                writable=1,
                reserved=0,
                backend_index=shadow.backend_index,
            )
        )
    return tuple(result)


def reclamation_receipts(
    certificates: Sequence[ReclamationCertificate],
) -> tuple[ReclamationReceipt, ...]:
    return tuple(
        ReclamationReceipt(
            reclamation=certificate.reclamation,
            page=certificate.page,
            backend_domain=certificate.backend_domain,
            acknowledged=1,
            reserved8=0,
            reserved32=0,
            backend_index=certificate.backend_index,
        )
        for certificate in certificates
    )


class CanonicalRuntime:
    """Host transaction journal around the sole canonical manager authority."""

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
        if self.arenas_by_class != {
            item.class_id: item for item in self.arenas
        }:
            raise ManagerError("manager arena index is incomplete or inconsistent")

        engine_epochs: set[int] = set()
        pool_ids: set[int] = set()
        page_ranges: list[tuple[int, int]] = []
        for class_config, arena in zip(
            expected_classes, self.arenas, strict=True
        ):
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
                raise ManagerError(
                    "manager arena does not match the compiled page size"
                )
            if (
                arena.class_id != class_config.class_id
                or arena.pool_id != class_config.pool_id
                or arena.backend_domain != class_config.backend_domain
            ):
                raise ManagerError(
                    "manager arena does not match its compiled KV class"
                )
            engine_epochs.add(arena.engine_epoch)
            if arena.pool_id in pool_ids:
                raise ManagerError("manager returned duplicate pool ids")
            pool_ids.add(arena.pool_id)
            page_range = (
                arena.first_page_id,
                arena.first_page_id + arena.page_count,
            )
            if any(
                page_range[0] < existing_end
                and existing_start < page_range[1]
                for existing_start, existing_end in page_ranges
            ):
                raise ManagerError("manager arena page-id ranges overlap")
            page_ranges.append(page_range)
        if len(engine_epochs) != 1:
            raise ManagerError("manager arenas do not share one engine epoch")
        self.engine_epoch = next(iter(engine_epochs))
        self.page_tokens = int(config.page_tokens)
        self.page_count = sum(item.page_count for item in self.arenas)
        self._requests: dict[Hashable, RequestRecord] = {}
        self._page_shadows: dict[tuple[int, int], PageShadow] = {}
        self._request_rows: dict[Hashable, int] = {}
        self._row_owners: dict[int, Hashable] = {}
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

    def performance_counters(self) -> dict[str, int]:
        with self._lock:
            counters = dict(
                getattr(self.manager, "performance_counters", {})
            )
            counters.update(self._runtime_counters)
            return counters

    def swa_activity(self) -> SwaActivity:
        with self._lock:
            self._healthy()
            return SwaActivity(
                retirement_certificates=self._swa_retirement_certificates,
                pages_reclaimed=self._swa_pages_reclaimed,
                wrap_events=self._swa_wrap_events,
            )

    def stats(self) -> ManagerStats:
        with self._lock:
            self._healthy()
            stats = self.manager.stats()
            counts = tuple(getattr(stats, field.name) for field in fields(ManagerStats))
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            ):
                raise ManagerError("manager stats contain an invalid counter")
            arena_stats = tuple(self.manager.arena_stats())
            if len(arena_stats) != len(self.arenas):
                raise ManagerError("manager returned the wrong arena-stats count")
            phase_names = (
                "free_pages",
                "reserved_pages",
                "writing_pages",
                "active_pages",
                "retiring_pages",
                "quarantined_pages",
                "exhausted_pages",
            )
            phase_totals = {name: 0 for name in phase_names}
            for identity, item in zip(self.arenas, arena_stats, strict=True):
                identity_fields = (
                    "engine_epoch",
                    "pool_epoch",
                    "pool_id",
                    "page_count",
                    "class_id",
                    "backend_domain",
                    "first_page_id",
                )
                if any(
                    getattr(item, name) != getattr(identity, name)
                    for name in identity_fields
                ):
                    raise ManagerError("arena stats changed arena identity")
                phases = tuple(getattr(item, name) for name in phase_names)
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in phases
                ):
                    raise ManagerError("arena stats contain an invalid counter")
                if sum(phases) != identity.page_count:
                    raise ManagerError("arena page census does not match its identity")
                for name, value in zip(phase_names, phases, strict=True):
                    phase_totals[name] += value
            if any(
                getattr(stats, name) != phase_totals[name] for name in phase_names
            ):
                raise ManagerError("aggregate stats disagree with per-arena census")
            return stats

    def record_for(self, key: Hashable) -> RequestRecord:
        with self._lock:
            self._healthy()
            try:
                return self._requests[key]
            except KeyError as error:
                raise ManagerError("request is not acquired") from error

    def has_request(self, key: Hashable) -> bool:
        with self._lock:
            self._healthy()
            return key in self._requests

    def bind_request_rows(
        self, assignments: Sequence[tuple[Hashable, int, bool]]
    ) -> None:
        """Bind SGLang ReqToToken rows to request identities for their full lifetime.

        ``is_new`` is the adapter's observation from before SGLang row allocation.
        A disagreement is an ownership-ambiguous allocator result, so it is
        fail-stopped rather than repaired from Python.
        """

        with self._lock:
            self._healthy()
            keys: set[Hashable] = set()
            rows: set[int] = set()
            normalized: list[tuple[Hashable, int, bool]] = []
            failure: str | None = None
            for key, raw_row, is_new in assignments:
                if not isinstance(is_new, bool):
                    failure = "request-row newness is not boolean"
                    break
                if isinstance(raw_row, bool) or not isinstance(raw_row, int):
                    failure = "request-row identity is not an integer"
                    break
                row = int(raw_row)
                if row <= 0:
                    failure = "request-row identity names the dummy row"
                    break
                if key in keys or row in rows:
                    failure = "request-row assignments alias within the batch"
                    break
                keys.add(key)
                rows.add(row)
                existing_row = self._request_rows.get(key)
                existing_owner = self._row_owners.get(row)
                if is_new:
                    if (
                        key in self._requests
                        or existing_row is not None
                        or existing_owner is not None
                    ):
                        failure = "new request-row assignment aliases live state"
                        break
                elif (
                    key not in self._requests
                    or existing_row != row
                    or existing_owner != key
                ):
                    failure = "live request-row identity changed"
                    break
                normalized.append((key, row, is_new))
            if failure is not None:
                self.fail_stop(f"ReqToToken row ownership became uncertain: {failure}")
                raise FailStopped(self._failure or failure)
            for key, row, is_new in normalized:
                if is_new:
                    self._request_rows[key] = row
                    self._row_owners[row] = key

    def unbind_request_row(self, key: Hashable, row: int) -> None:
        """Forget a row only after SGLang has successfully returned it."""

        with self._lock:
            self._healthy()
            if (
                key in self._requests
                or self._request_rows.get(key) != row
                or self._row_owners.get(row) != key
            ):
                self.fail_stop("ReqToToken row release changed ownership identity")
                raise FailStopped(self._failure or "request-row release failed")
            del self._request_rows[key]
            del self._row_owners[row]

    def bind_reclamation_cleanup(
        self, key: Hashable, cleanup: MirrorCleanupBinding
    ) -> None:
        with self._lock:
            self._healthy()
            if not isinstance(cleanup, MirrorCleanupBinding) or not isinstance(
                cleanup.coordinator, MirrorCleanupProtocol
            ):
                raise TypeError("reclamation cleanup must be a collective binding")
            record = self.record_for(key)
            if record.reclamation_cleanup is not None:
                raise ManagerError("cannot replace an installed mirror cleanup")
            if (
                record.pending is not None
                and record.pending.phase is not StepPhase.PREPARED
            ):
                raise ManagerError("cannot install mirror cleanup after lowering")
            record.reclamation_cleanup = cleanup

    def prepare_batch(
        self, items: Sequence[tuple[Hashable, int]]
    ) -> tuple[BatchRecord, tuple[LoweringPlan, ...]]:
        """Acquire missing requests and prepare one ordered engine batch."""

        with self._lock:
            self._healthy()
            values = tuple(items)
            if not values or len({key for key, _ in values}) != len(values):
                raise ManagerError("prepare batch keys must be nonempty and unique")
            existing_records: list[RequestRecord | None] = []
            new_keys: list[Hashable] = []
            for key, target_boundary in values:
                if isinstance(target_boundary, bool) or not isinstance(
                    target_boundary, int
                ):
                    raise ManagerError("step target must be an integer")
                record = self._requests.get(key)
                if record is None:
                    new_keys.append(key)
                else:
                    if record.pending is not None:
                        raise ManagerError("request already has a pending step")
                    if target_boundary <= record.boundary:
                        raise ManagerError(
                            "step target must advance the request boundary"
                        )
                existing_records.append(record)

            new_records: dict[Hashable, RequestRecord] = {}
            if new_keys:
                try:
                    leases = tuple(
                        self.manager.request_acquire_batch(len(new_keys))
                    )
                    if len(leases) != len(new_keys) or len(set(leases)) != len(leases):
                        raise ManagerError(
                            "manager returned invalid acquired-request cardinality"
                        )
                    if any(
                        lease == live.lease
                        for lease in leases
                        for live in self._requests.values()
                    ):
                        raise ManagerError(
                            "manager returned a request lease already owned by live state"
                        )
                    for key, lease in zip(new_keys, leases, strict=True):
                        if lease.engine_epoch != self.engine_epoch:
                            raise ManagerError(
                                "request lease belongs to a stale engine epoch"
                            )
                        _nonnegative("request slot", lease.slot)
                        _positive("request generation", lease.generation)
                        new_records[key] = RequestRecord(
                            cursor=RequestCursor(lease=lease)
                        )
                except Exception as error:
                    self.fail_stop(f"request batch acquisition became uncertain: {error}")
                    raise FailStopped(
                        self._failure or "request batch acquisition failed"
                    ) from error
                self._requests.update(new_records)

            records = tuple(
                existing if existing is not None else new_records[key]
                for (key, _), existing in zip(
                    values, existing_records, strict=True
                )
            )
            try:
                prepared_values = tuple(
                    self.manager.prepare_batch(
                        tuple(
                            PrepareBatchItem(record.lease, target)
                            for record, (_, target) in zip(
                                records, values, strict=True
                            )
                        )
                    )
                )
            except Exception as error:
                # Crossing the manager call boundary makes the outcome unknown:
                # the core may have committed the whole prepare batch before a
                # wrapper/output-observation failure surfaced.  Without a typed
                # pre-commit error, neither rollback nor retry is safe.
                self.fail_stop(f"manager batch preparation became uncertain: {error}")
                raise FailStopped(
                    self._failure or "manager batch preparation failed"
                ) from error

            pending_records: list[StepRecord] = []
            plans: list[LoweringPlan] = []
            decoded_pages: list[tuple[PageShadow, ...]] = []
            batch_physical: set[tuple[int, int]] = set()
            batch_steps: set[StepLease] = set()
            try:
                if len(prepared_values) != len(values):
                    raise ManagerError("manager returned the wrong prepared item count")
                for (key, target), record, prepared in zip(
                    values, records, prepared_values, strict=True
                ):
                    self._validate_prepared(record, prepared, target)
                    if prepared.step in batch_steps:
                        raise ManagerError("prepare batch duplicated a step lease")
                    batch_steps.add(prepared.step)
                    plan, new_pages = _decode_prepared(
                        record.cursor,
                        prepared,
                        self.arenas_by_class,
                        self.config,
                    )
                    for shadow in new_pages:
                        physical_key = (shadow.page.pool_id, shadow.page.page_id)
                        if (
                            physical_key in batch_physical
                            or physical_key in self._page_shadows
                        ):
                            raise ManagerError(
                                "prepare aliases a physical page owned by live state"
                            )
                        batch_physical.add(physical_key)
                    plans.append(plan)
                    decoded_pages.append(new_pages)
                    pending_records.append(
                        StepRecord(
                            key=key,
                            prepared=prepared,
                            new_pages=new_pages,
                        )
                    )
            except Exception as error:
                steps = tuple(
                    value.step
                    for value in prepared_values
                    if isinstance(getattr(value, "step", None), StepLease)
                )
                self._best_effort_quarantine_steps(steps)
                self.fail_stop(f"manager returned an invalid prepare batch: {error}")
                raise FailStopped(
                    self._failure or "invalid prepare batch"
                ) from error

            batch = BatchRecord(
                tuple(key for key, _ in values), tuple(pending_records)
            )
            for record, pending in zip(records, batch.records, strict=True):
                record.pending = pending
            for new_pages in decoded_pages:
                for shadow in new_pages:
                    self._page_shadows[(shadow.page.pool_id, shadow.page.page_id)] = shadow
            return batch, tuple(plans)

    def mark_lowered(self, batch: BatchRecord) -> None:
        with self._lock:
            self._healthy()
            self._validate_batch_record(batch)
            receipts: list[tuple[BackendBindReceipt, ...]] = []
            for pending in batch.records:
                self._require_pending(pending, StepPhase.PREPARED)
                receipts.append(
                    bind_receipts(
                        pending.prepared,
                        pending.new_pages,
                        self.arenas_by_class,
                    )
                )
            for pending, item_receipts in zip(
                batch.records, receipts, strict=True
            ):
                pending.bind_receipts = item_receipts
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
                            (pending.prepared.step, pending.bind_receipts)
                            for pending in batch.records
                        )
                    )
                )
                if len(submitted_values) != len(batch.records):
                    raise ManagerError("manager returned the wrong submitted item count")
                for record, pending, submitted in zip(
                    records, batch.records, submitted_values, strict=True
                ):
                    self._validate_submitted(record, pending, submitted)
            except Exception as error:
                # Submit semantic errors are quarantined atomically by core.
                # A lost return is equally uncertain, so never downgrade this
                # path to an unobserved abort.
                self._best_effort_quarantine_submissions(assumed)
                for pending in batch.records:
                    pending.phase = StepPhase.QUARANTINED
                self.fail_stop(f"manager batch submission became uncertain: {error}")
                raise FailStopped(
                    self._failure or "manager batch submission failed"
                ) from error
            for pending, submitted in zip(
                batch.records, submitted_values, strict=True
            ):
                pending.submitted = submitted
                pending.phase = StepPhase.SUBMITTED
            return submitted_values

    def candidate_mirror_failed(
        self, batch: BatchRecord, error: BaseException
    ) -> None:
        with self._lock:
            self._quarantine_submitted(batch.records)
            self.fail_stop(f"ReqToToken candidate mirror became uncertain: {error}")

    def pre_forward_failed(self, error: BaseException) -> None:
        """Fail closed after scheduling submitted work but before event ownership."""

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
                self.manager.abort_steps(
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
                for shadow in pending.new_pages:
                    self._page_shadows.pop(
                        (shadow.page.pool_id, shadow.page.page_id), None
                    )
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

    def release(self, key: Hashable) -> None:
        with self._lock:
            self._healthy()
            record = self._requests.get(key)
            if record is None:
                return
            if record.pending is not None:
                pending = record.pending
                pending_batch = BatchRecord((key,), (pending,))
                if pending.phase is StepPhase.PREPARED:
                    self.abort_unobserved(pending_batch)
                elif pending.phase is StepPhase.EVENT:
                    self.wait(key)
                elif pending.phase is StepPhase.LOWERED:
                    self.lowering_failed(
                        pending_batch, RuntimeError("release raced lowering")
                    )
                    raise FailStopped(self._failure or "release raced lowering")
                else:
                    self._quarantine_submitted((pending,))
                    self.fail_stop("request release raced an unproven submission")
                    raise FailStopped(self._failure or "release raced submission")

            try:
                completions = tuple(self.manager.release_batch((record.lease,)))
                if len(completions) != 1:
                    raise ManagerError("manager returned the wrong release item count")
                completion = completions[0]
                released_pages = self._validate_release(record, completion)
                receipts = reclamation_receipts(completion.retirements)
                mirror_transaction = self._preflight_reclamation_mirrors(
                    ((record, completion.retirements, True, record.boundary),)
                )
                self._commit_reclamation_mirrors(mirror_transaction)
                self._finalize_reclamation_mirrors(mirror_transaction)
                if receipts:
                    self.manager.acknowledge_reclamations(receipts)
                self.manager.recycle_requests((record.lease,))
            except Exception as error:
                self.fail_stop(f"request release or reclamation became uncertain: {error}")
                raise FailStopped(self._failure or "request release failed") from error
            for shadow in released_pages:
                self._page_shadows.pop(
                    (shadow.page.pool_id, shadow.page.page_id), None
                )
            del self._requests[key]

    def close(self) -> None:
        with self._lock:
            self._healthy()
            if (
                self._events
                or self._requests
                or self._page_shadows
                or self._request_rows
                or self._row_owners
            ):
                raise ManagerError("cannot destroy a manager with live requests")
            self.manager.destroy()

    def _validate_prepared(
        self, record: RequestRecord, prepared: PreparedStep, target_boundary: int
    ) -> None:
        step = prepared.step
        if prepared.request != record.lease:
            raise ManagerError("prepared step belongs to another request")
        if step.engine_epoch != self.engine_epoch:
            raise ManagerError("prepared step belongs to another engine")
        _nonnegative("step slot", step.slot)
        _positive("step generation", step.generation)
        if (
            prepared.previous_boundary != record.boundary
            or prepared.target_boundary != target_boundary
            or prepared.base_view_version != record.cursor.view_version
            or prepared.target_view_version != prepared.base_view_version + 1
        ):
            raise ManagerError("prepared step boundary or view lineage is invalid")

    def _validate_submitted(
        self, record: RequestRecord, pending: StepRecord, submitted: SubmittedStep
    ) -> None:
        if submitted.request != record.lease:
            raise ManagerError("submitted step belongs to another request")
        expected_lease = SubmissionLease(
            pending.prepared.step.engine_epoch,
            pending.prepared.step.slot,
            pending.prepared.step.generation,
        )
        if submitted.submission != expected_lease:
            raise ManagerError("submission lease did not preserve the step generation")

    def _validate_completion(
        self,
        record: RequestRecord,
        pending: StepRecord,
        receipt: BatchCompletionReceipt,
        completion: StepCompletion,
    ) -> tuple[tuple[PageShadow, ...], int]:
        assert pending.submitted is not None
        if (
            completion.submission != pending.submitted.submission
            or completion.request != record.lease
            or completion.published_view_version
            != pending.prepared.target_view_version
            or completion.published_boundary != pending.prepared.target_boundary
        ):
            raise ManagerError("completion belongs to another submission")
        for shadow in pending.new_pages:
            logical_key = (shadow.class_id, shadow.logical_ordinal)
            physical_key = (shadow.page.pool_id, shadow.page.page_id)
            if logical_key in record.cursor.pages:
                raise ManagerError("completion candidate aliases a logical page")
            if self._page_shadows.get(physical_key) != shadow:
                raise ManagerError("completion lost a reserved page shadow")
        retired = self._expected_retired(record, pending)
        resident_count = (
            len(record.cursor.pages) + len(pending.new_pages) - len(retired)
        )
        if completion.resident_count != resident_count:
            raise ManagerError("completion published the wrong resident count")
        if len(completion.retirements) != len(retired):
            raise ManagerError("completion returned the wrong retirement cardinality")
        if len({item.reclamation for item in completion.retirements}) != len(
            completion.retirements
        ):
            raise ManagerError("completion duplicated a reclamation lease")
        for certificate, shadow in zip(
            completion.retirements, retired, strict=True
        ):
            physical_key = (shadow.page.pool_id, shadow.page.page_id)
            if self._page_shadows.get(physical_key) != shadow:
                raise ManagerError("completion lost a retiring page shadow")
            self._validate_certificate(
                certificate,
                record.lease,
                shadow,
                min(
                    pending.prepared.target_boundary,
                    (shadow.logical_ordinal + 1) * self.page_tokens,
                ),
                receipt.completion_domain,
                receipt.completion_value,
            )
        return retired, resident_count

    def _validate_release(
        self, record: RequestRecord, completion: ReleaseCompletion
    ) -> tuple[PageShadow, ...]:
        if completion.request != record.lease:
            raise ManagerError("release completion belongs to another request")
        pages = tuple(
            shadow
            for _, shadow in sorted(record.cursor.pages.items())
        )
        if len(completion.retirements) != len(pages):
            raise ManagerError("release returned the wrong retirement cardinality")
        if len({item.reclamation for item in completion.retirements}) != len(
            completion.retirements
        ):
            raise ManagerError("release duplicated a reclamation lease")
        for certificate, shadow in zip(
            completion.retirements, pages, strict=True
        ):
            self._validate_certificate(
                certificate,
                record.lease,
                shadow,
                min(
                    record.boundary,
                    (shadow.logical_ordinal + 1) * self.page_tokens,
                ),
                record.completion_domain,
                record.completion_value,
            )
        return pages

    def _validate_certificate(
        self,
        certificate: ReclamationCertificate,
        request: RequestLease,
        shadow: PageShadow,
        token_end_exclusive: int,
        completion_domain: int,
        completion_value: int,
    ) -> None:
        reclamation = certificate.reclamation
        if reclamation.engine_epoch != self.engine_epoch:
            raise ManagerError("reclamation belongs to another engine")
        try:
            arena = self.arenas_by_class[certificate.class_id]
        except KeyError as error:
            raise ManagerError("reclamation names an unknown KV class") from error
        _nonnegative("reclamation slot", reclamation.slot)
        _positive("reclamation generation", reclamation.generation)
        if (
            certificate.request != request
            or certificate.page != shadow.page
            or certificate.class_id != shadow.class_id
            or certificate.backend_domain != arena.backend_domain
            or certificate.logical_ordinal != shadow.logical_ordinal
            or certificate.backend_index != shadow.backend_index
            or certificate.token_begin != shadow.logical_ordinal * self.page_tokens
            or certificate.token_end_exclusive != token_end_exclusive
            or certificate.completion_domain != completion_domain
            or certificate.completion_value != completion_value
            or certificate.page.pool_id != arena.pool_id
            or certificate.page.pool_epoch != arena.pool_epoch
            or not (
                arena.first_page_id
                <= certificate.page.page_id
                < arena.first_page_id + arena.page_count
            )
            or not (
                arena.backend_base_index
                <= certificate.backend_index
                < arena.backend_base_index + arena.page_count
            )
            or (
                certificate.backend_index - arena.backend_base_index
                != certificate.page.page_id - arena.first_page_id
            )
        ):
            raise ManagerError("reclamation certificate changed physical identity")

    def _expected_retired(
        self, record: RequestRecord, pending: StepRecord
    ) -> tuple[PageShadow, ...]:
        target = pending.prepared.target_boundary
        pending_by_key = {
            (shadow.class_id, shadow.logical_ordinal): shadow
            for shadow in pending.new_pages
        }
        if len(pending_by_key) != len(pending.new_pages):
            raise ManagerError("completion candidate aliases a logical page")
        retired: list[PageShadow] = []
        for class_config in self.config.classes:
            if class_config.retention == "full":
                continue
            previous_start = max(
                0,
                record.boundary - (int(class_config.window_tokens) - 1),
            )
            retained_start = max(
                0,
                target - (int(class_config.window_tokens) - 1),
            )
            first_ordinal = previous_start // self.page_tokens
            last_retired = retained_start // self.page_tokens - 1
            for ordinal in range(first_ordinal, last_retired + 1):
                logical_key = (class_config.class_id, ordinal)
                shadow = record.cursor.pages.get(logical_key)
                if shadow is None:
                    shadow = pending_by_key.get(logical_key)
                if shadow is None:
                    raise ManagerError(
                        "completion retirement delta is missing a logical page"
                    )
                retired.append(shadow)
        return tuple(retired)

    def _complete_group(self, group: EventGroup) -> None:
        if group not in self._events:
            return
        completion_value = self._completion_value
        receipt = BatchCompletionReceipt(
            engine_epoch=self.engine_epoch,
            completion_domain=group.completion_domain,
            completion_value=completion_value,
        )
        completed: list[
            tuple[
                StepRecord,
                StepCompletion,
                tuple[PageShadow, ...],
                int,
            ]
        ] = []
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
            completion_values = tuple(
                self.manager.complete_batch(receipt, submissions)
            )
            if len(completion_values) != len(group.records):
                raise ManagerError("manager returned the wrong completion item count")
            for pending, record, completion in zip(
                group.records, records, completion_values, strict=True
            ):
                retired, resident_count = self._validate_completion(
                    record, pending, receipt, completion
                )
                completed.append(
                    (pending, completion, retired, resident_count)
                )

            all_certificates = tuple(
                certificate
                for _, completion, _, _ in completed
                for certificate in completion.retirements
            )
            if len({item.reclamation for item in all_certificates}) != len(
                all_certificates
            ):
                raise ManagerError(
                    "completion batch duplicated a reclamation lease"
                )
            sliding_class_ids = {
                item.class_id
                for item in self.config.classes
                if item.retention == "sliding"
            }
            sliding_certificates = tuple(
                item
                for item in all_certificates
                if item.class_id in sliding_class_ids
            )
            cycle_updates: list[tuple[RequestRecord, int, int]] = []
            wrap_events = 0
            for pending, _completion, _retired, _resident_count in completed:
                record = self._requests[pending.key]
                for class_id in sliding_class_ids:
                    previous_cycle = record.swa_temporal_cycles.get(class_id, 0)
                    period = int(
                        self.config.classes_by_id[class_id].period_blocks
                    )
                    current_cycle = (
                        (pending.prepared.target_boundary - 1)
                        // self.page_tokens
                        // period
                    )
                    if current_cycle < previous_cycle:
                        raise ManagerError("SWA temporal cycle moved backwards")
                    wrap_events += current_cycle - previous_cycle
                    cycle_updates.append((record, class_id, current_cycle))

            receipts = reclamation_receipts(all_certificates)
            # This is one transaction across the whole EventGroup.  Preflight
            # validates every request/certificate and the external GPU mirror
            # without mutation.  Only then may the coordinator launch all
            # clears, establish one shared completion dependency, and allow the
            # aggregate acknowledgement below.
            mirror_transaction = self._preflight_reclamation_mirrors(
                tuple(
                    (
                        self._requests[pending.key],
                        completion.retirements,
                        False,
                        pending.prepared.target_boundary,
                    )
                    for pending, completion, _, _ in completed
                )
            )
            self._commit_reclamation_mirrors(mirror_transaction)
            self._finalize_reclamation_mirrors(mirror_transaction)
            # Commit the already-preflighted host delta while this runtime lock
            # still hides it from observers. If even a Python allocation fails,
            # the aggregate ACK has not happened, so no physical page can be
            # reused and the process fail-stops safely.
            for pending, _completion, retired, resident_count in completed:
                record = self._requests[pending.key]
                for shadow in pending.new_pages:
                    record.cursor.pages[
                        (shadow.class_id, shadow.logical_ordinal)
                    ] = shadow
                for shadow in retired:
                    logical_key = (shadow.class_id, shadow.logical_ordinal)
                    removed = record.cursor.pages.pop(logical_key)
                    if removed != shadow:
                        raise ManagerError(
                            "completion commit lost a retired page shadow"
                        )
                    physical_key = (shadow.page.pool_id, shadow.page.page_id)
                    removed_physical = self._page_shadows.pop(physical_key)
                    if removed_physical != shadow:
                        raise ManagerError(
                            "completion commit changed physical page identity"
                        )
                if len(record.cursor.pages) != resident_count:
                    raise ManagerError(
                        "completion commit produced the wrong resident count"
                    )
                record.cursor.boundary = pending.prepared.target_boundary
                record.cursor.view_version = pending.prepared.target_view_version
                record.completion_domain = receipt.completion_domain
                record.completion_value = receipt.completion_value
            if receipts:
                self.manager.acknowledge_reclamations(receipts)
            self._swa_retirement_certificates += len(sliding_certificates)
            self._swa_pages_reclaimed += len(sliding_certificates)
            self._swa_wrap_events += wrap_events
            for record, class_id, current_cycle in cycle_updates:
                record.swa_temporal_cycles[class_id] = current_cycle

            # Keep every post-ACK bookkeeping action inside the same
            # fail-closed boundary.  The runtime lock prevents another
            # allocation from observing a page after ACK but before these
            # journals are settled; any failure below therefore terminates
            # the authority before reuse can occur.
            self._completion_value += 1
            self._runtime_counters["completion_values"] += 1
            for pending, _completion, _retired, _resident_count in completed:
                record = self._requests[pending.key]
                record.pending = None
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

    @staticmethod
    def _preflight_reclamation_mirrors(
        values: Sequence[
            tuple[
                RequestRecord,
                Sequence[ReclamationCertificate],
                bool,
                int,
            ]
        ],
    ) -> tuple[MirrorCleanupProtocol, Any] | None:
        coordinator: MirrorCleanupProtocol | None = None
        items: list[MirrorCleanupItem] = []
        for record, certificates, releasing, boundary in values:
            certificate_values = tuple(certificates)
            if not certificate_values and not releasing:
                continue
            binding = record.reclamation_cleanup
            if binding is None:
                raise ManagerError("request has no ReqToToken reclamation cleanup")
            if coordinator is None:
                coordinator = binding.coordinator
            elif binding.coordinator is not coordinator:
                raise ManagerError(
                    "event group spans multiple reclamation mirror authorities"
                )
            items.append(
                MirrorCleanupItem(
                    context=binding.context,
                    certificates=certificate_values,
                    releasing=releasing,
                    boundary=boundary,
                )
            )
        if coordinator is None:
            return None
        return coordinator, coordinator.preflight(tuple(items))

    @staticmethod
    def _commit_reclamation_mirrors(
        transaction: tuple[MirrorCleanupProtocol, Any] | None,
    ) -> None:
        if transaction is None:
            return
        coordinator, plan = transaction
        coordinator.commit(plan)
        coordinator.synchronize(plan)

    @staticmethod
    def _finalize_reclamation_mirrors(
        transaction: tuple[MirrorCleanupProtocol, Any] | None,
    ) -> None:
        if transaction is None:
            return
        coordinator, plan = transaction
        coordinator.finalize(plan)

    def _require_pending(
        self, pending: StepRecord, expected: StepPhase
    ) -> RequestRecord:
        try:
            record = self._requests[pending.key]
        except KeyError as error:
            raise ManagerError("step request has already been released") from error
        if record.pending is not pending or pending.phase is not expected:
            raise ManagerError(
                f"step phase mismatch: expected {expected.name.lower()}, "
                f"got {pending.phase.name.lower()}"
            )
        return record

    @staticmethod
    def _validate_batch_record(batch: BatchRecord) -> None:
        if not isinstance(batch, BatchRecord):
            raise ManagerError("operation requires one ABI5 batch journal")
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
        if not steps:
            return
        try:
            self.manager.quarantine_steps(tuple(steps))
        except Exception:
            pass

    def _best_effort_quarantine_submissions(
        self, submissions: Sequence[SubmissionLease]
    ) -> None:
        if not submissions:
            return
        try:
            self.manager.quarantine_submissions(tuple(submissions))
        except Exception:
            pass


__all__ = [
    "ArenaIdentity",
    "ArenaRegistration",
    "ArenaStats",
    "BatchCompletionReceipt",
    "BatchRecord",
    "BackendBindReceipt",
    "BackendUnobservedReceipt",
    "CanonicalRuntime",
    "ClassLowering",
    "ClassLoweringSpec",
    "FailStopped",
    "LoweringPlan",
    "ManagerCreateSettings",
    "ManagerError",
    "ManagerFactoryProtocol",
    "ManagerProtocol",
    "ManagerStats",
    "MirrorCleanupBinding",
    "MirrorCleanupItem",
    "MirrorCleanupProtocol",
    "PageLease",
    "PageShadow",
    "PrepareBatchItem",
    "PreparedStep",
    "ReclamationCertificate",
    "ReclamationLease",
    "ReclamationReceipt",
    "ReleaseCompletion",
    "RequestCursor",
    "RequestLease",
    "StepCompletion",
    "StepLease",
    "StepPhase",
    "StepRecord",
    "SubmissionLease",
    "SubmittedStep",
    "SwaActivity",
    "WriteIntent",
    "bind_receipts",
    "expected_new_ordinals",
    "lowering_plan",
    "reclamation_receipts",
    "sglang_page_id",
]
