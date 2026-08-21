from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum, auto
from threading import RLock
from typing import Any, Hashable, Mapping, Protocol, Sequence, runtime_checkable


DEVICE_VIEW_ABI_VERSION = 1
VIEW_PUBLISHED = 1 << 0
VIEW_CANDIDATE = 1 << 1
ACCESS_READ = 1 << 0
ACCESS_WRITE = 1 << 1
NEEDS_BINDING = 1 << 2
ACCESS_FLAGS = ACCESS_READ | ACCESS_WRITE | NEEDS_BINDING


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
class DeviceViewHeader:
    abi_version: int
    flags: int
    engine_epoch: int
    request_slot: int
    request_generation: int
    view_version: int
    base_frontier: int
    target_frontier: int
    page_tokens: int
    entry_count: int


@dataclass(frozen=True, slots=True)
class DeviceViewEntry:
    class_id: int
    backend_domain: int
    access_flags: int
    logical_ordinal: int
    token_begin: int
    valid_token_count: int
    visible_token_offset: int
    visible_token_count: int
    pool_id: int
    temporal_cell_index: int
    temporal_cycle: int
    pool_epoch: int
    page_generation: int
    backend_index: int
    page_id: int
    reserved: int = 0

    def page_lease(self, engine_epoch: int) -> PageLease:
        return PageLease(
            engine_epoch=engine_epoch,
            pool_epoch=self.pool_epoch,
            generation=self.page_generation,
            page_id=self.page_id,
            pool_id=self.pool_id,
        )


@dataclass(frozen=True, slots=True)
class DeviceKvView:
    header: DeviceViewHeader
    entries: tuple[DeviceViewEntry, ...]

    @classmethod
    def empty(cls, request: RequestLease, page_tokens: int) -> "DeviceKvView":
        return cls(
            DeviceViewHeader(
                abi_version=DEVICE_VIEW_ABI_VERSION,
                flags=VIEW_PUBLISHED,
                engine_epoch=request.engine_epoch,
                request_slot=request.slot,
                request_generation=request.generation,
                view_version=0,
                base_frontier=0,
                target_frontier=0,
                page_tokens=page_tokens,
                entry_count=0,
            ),
            (),
        )


@dataclass(frozen=True, slots=True)
class PreparedStep:
    step: StepLease
    request: RequestLease
    base_view_version: int
    target_view_version: int
    previous_boundary: int
    target_boundary: int
    view: DeviceKvView


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
    view: DeviceKvView


@dataclass(frozen=True, slots=True)
class CompletionReceipt:
    submission: SubmissionLease
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
    published_view: DeviceKvView
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


@runtime_checkable
class ManagerProtocol(Protocol):
    @property
    def arenas(self) -> tuple[ArenaIdentity, ...]: ...

    @property
    def arenas_by_class(self) -> dict[int, ArenaIdentity]: ...

    def arena_stats(self) -> tuple[ArenaStats, ...]: ...

    def request_acquire(self) -> RequestLease: ...

    def prepare_step(self, request: RequestLease, target_boundary: int) -> PreparedStep: ...

    def submit_step(
        self, step: StepLease, receipts: Sequence[BackendBindReceipt]
    ) -> SubmittedStep: ...

    def complete_step(self, receipt: CompletionReceipt) -> StepCompletion: ...

    def abort_step(self, receipt: BackendUnobservedReceipt) -> None: ...

    def quarantine_step(self, step: StepLease) -> None: ...

    def quarantine_submission(self, submission: SubmissionLease) -> None: ...

    def release_request(self, request: RequestLease) -> ReleaseCompletion: ...

    def commit_reclamations(
        self, receipts: Sequence[ReclamationReceipt]
    ) -> None: ...

    def recycle_request(self, request: RequestLease) -> None: ...

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
    phase: StepPhase = StepPhase.PREPARED
    bind_receipts: tuple[BackendBindReceipt, ...] = ()
    submitted: SubmittedStep | None = None


@dataclass(slots=True)
class RequestRecord:
    lease: RequestLease
    published_view: DeviceKvView
    boundary: int = 0
    pending: StepRecord | None = None
    completion_domain: int = 0
    completion_value: int = 0
    reclamation_cleanup: Any = None
    swa_temporal_cycles: dict[int, int] = field(default_factory=dict)


@dataclass(slots=True)
class EventGroup:
    event: Any
    records: tuple[StepRecord, ...]
    completion_domain: int


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


def validate_view(
    view: DeviceKvView,
    request: RequestLease,
    arenas: Mapping[int, ArenaIdentity],
    config: Any,
    *,
    expected_flags: int,
) -> None:
    header = view.header
    if header.abi_version != DEVICE_VIEW_ABI_VERSION:
        raise ManagerError("device view ABI version is invalid")
    if header.flags != expected_flags:
        raise ManagerError("device view publication state is invalid")
    if (
        header.engine_epoch != request.engine_epoch
        or header.request_slot != request.slot
        or header.request_generation != request.generation
        or not arenas
        or header.engine_epoch != next(iter(arenas.values())).engine_epoch
    ):
        raise ManagerError("device view belongs to a foreign request generation")
    if header.page_tokens != int(config.page_tokens) or header.entry_count != len(
        view.entries
    ):
        raise ManagerError("device view geometry changed across the ABI")
    if (
        header.view_version < 0
        or header.base_frontier < 0
        or header.target_frontier < header.base_frontier
    ):
        raise ManagerError("device view contains an invalid counter")
    if expected_flags == VIEW_PUBLISHED and header.base_frontier != header.target_frontier:
        raise ManagerError("published device view has two different frontiers")

    logical: set[tuple[int, int]] = set()
    physical: set[tuple[int, int]] = set()
    previous_key = (-1, -1)
    classes = config.classes_by_id
    for entry in view.entries:
        entry_key = (entry.class_id, entry.logical_ordinal)
        if entry_key <= previous_key:
            raise ManagerError(
                "device view entries are not ordered by class and ordinal"
            )
        previous_key = entry_key
        try:
            arena = arenas[entry.class_id]
            class_config = classes[entry.class_id]
        except KeyError as error:
            raise ManagerError("device view names an unknown KV class") from error
        if entry.reserved != 0 or entry.access_flags & ~ACCESS_FLAGS:
            raise ManagerError("device view entry flags or reserved field is invalid")
        if expected_flags == VIEW_PUBLISHED and entry.access_flags & (ACCESS_WRITE | NEEDS_BINDING):
            raise ManagerError("published view exposes a writable or unbound page")
        if entry.access_flags & NEEDS_BINDING and not entry.access_flags & ACCESS_WRITE:
            raise ManagerError("an unbound candidate page is not writable")
        if entry.token_begin != entry.logical_ordinal * header.page_tokens:
            raise ManagerError("device view entry is not page aligned")
        if not 0 < entry.valid_token_count <= arena.page_tokens:
            raise ManagerError("device view entry has an invalid token count")
        if (
            entry.visible_token_offset < 0
            or entry.visible_token_count < 0
            or entry.visible_token_offset + entry.visible_token_count
            > entry.valid_token_count
        ):
            raise ManagerError("device view entry has an invalid visible span")
        if bool(entry.access_flags & ACCESS_READ) != (entry.visible_token_count > 0):
            raise ManagerError("device view read access disagrees with its visible span")
        if (
            entry.backend_domain != arena.backend_domain
            or entry.pool_id != arena.pool_id
            or entry.pool_epoch != arena.pool_epoch
        ):
            raise ManagerError("device view entry belongs to another arena")
        if entry.page_generation <= 0:
            raise ManagerError("device view entry has a stale physical page")
        page_end = arena.first_page_id + arena.page_count
        if not arena.first_page_id <= entry.page_id < page_end:
            raise ManagerError("device view page id is outside its arena")
        backend_end = arena.backend_base_index + arena.page_count
        if not arena.backend_base_index <= entry.backend_index < backend_end:
            raise ManagerError("device view backend index is outside its arena")
        if (
            entry.backend_index - arena.backend_base_index
            != entry.page_id - arena.first_page_id
        ):
            raise ManagerError(
                "device view page id and backend index are not the same arena slot"
            )
        if class_config.retention == "full":
            temporal_valid = (
                entry.temporal_cell_index == entry.logical_ordinal
                and entry.temporal_cycle == 0
            )
        else:
            period = int(class_config.period_blocks)
            temporal_valid = (
                entry.temporal_cell_index == entry.logical_ordinal % period
                and entry.temporal_cycle == entry.logical_ordinal // period
            )
        if not temporal_valid:
            raise ManagerError("device view temporal address is invalid")
        logical_key = (entry.class_id, entry.logical_ordinal)
        # A physical slot can have only one live generation in one immutable
        # view. Including generation in this key would let a corrupted ABI
        # alias the same GPU slot under two logical pages by merely changing
        # the reported generation.
        physical_key = (entry.pool_id, entry.page_id)
        if logical_key in logical or physical_key in physical:
            raise ManagerError("device view aliases a live logical or physical page")
        logical.add(logical_key)
        physical.add(physical_key)


def sglang_page_id(backend_index: int, backend_base_index: int) -> int:
    page_id = backend_index - backend_base_index + 1
    if page_id <= 0:
        raise ManagerError("manager backend index cannot lower into the SGLang arena")
    return page_id


def token_location(
    view: DeviceKvView,
    position: int,
    class_id: int,
    backend_base_index: int = 0,
) -> int:
    if position < 0:
        return -1
    for entry in view.entries:
        if entry.class_id == class_id and (
            entry.token_begin <= position < entry.token_begin + entry.valid_token_count
        ):
            return (
                sglang_page_id(entry.backend_index, backend_base_index)
                * view.header.page_tokens
                + position
                - entry.token_begin
            )
    raise ManagerError(f"device view does not cover token position {position}")


def lowering_plan(
    published: DeviceKvView,
    prepared: PreparedStep,
    arenas: Mapping[int, ArenaIdentity],
    config: Any,
) -> LoweringPlan:
    validate_view(
        published, prepared.request, arenas, config, expected_flags=VIEW_PUBLISHED
    )
    validate_view(
        prepared.view, prepared.request, arenas, config, expected_flags=VIEW_CANDIDATE
    )
    physical_fields = (
        "class_id",
        "backend_domain",
        "pool_id",
        "temporal_cell_index",
        "temporal_cycle",
        "pool_epoch",
        "page_generation",
        "backend_index",
        "page_id",
    )
    class_specs: list[ClassLoweringSpec] = []
    for class_config in config.classes:
        arena = arenas[class_config.class_id]
        candidate_start = (
            0
            if class_config.retention == "full"
            else max(
                0,
                prepared.previous_boundary
                - (int(class_config.window_tokens) - 1),
            )
        )
        expected_candidate_ordinals = tuple(
            range(
                candidate_start // arena.page_tokens,
                (prepared.target_boundary - 1) // arena.page_tokens + 1,
            )
        )
        candidate_entries = tuple(
            entry
            for entry in prepared.view.entries
            if entry.class_id == class_config.class_id
        )
        if tuple(entry.logical_ordinal for entry in candidate_entries) != (
            expected_candidate_ordinals
        ):
            raise ManagerError("prepare returned the wrong per-class logical range")
        published_by_ordinal = {
            entry.logical_ordinal: entry
            for entry in published.entries
            if entry.class_id == class_config.class_id
        }
        for entry in candidate_entries:
            token_end = min(
                prepared.target_boundary, entry.token_begin + arena.page_tokens
            )
            visible_begin = max(candidate_start, entry.token_begin)
            visible_end = min(
                prepared.target_boundary, entry.token_begin + arena.page_tokens
            )
            expected_access = ACCESS_READ
            if entry.logical_ordinal >= (
                prepared.previous_boundary // arena.page_tokens
            ):
                expected_access |= ACCESS_WRITE
            previous_entry = published_by_ordinal.get(entry.logical_ordinal)
            if previous_entry is None:
                expected_access |= NEEDS_BINDING
            elif any(
                getattr(entry, field_name) != getattr(previous_entry, field_name)
                for field_name in physical_fields
            ):
                raise ManagerError(
                    "prepare remapped an already-published logical page"
                )
            if (
                entry.access_flags != expected_access
                or entry.valid_token_count != token_end - entry.token_begin
                or entry.visible_token_offset != visible_begin - entry.token_begin
                or entry.visible_token_count != visible_end - visible_begin
            ):
                raise ManagerError(
                    "prepare returned invalid candidate access geometry"
                )

        expected_new = expected_new_ordinals(
            prepared.previous_boundary,
            prepared.target_boundary,
            arena.page_tokens,
        )
        bindings = tuple(
            entry
            for entry in candidate_entries
            if entry.access_flags & NEEDS_BINDING
        )
        if tuple(entry.logical_ordinal for entry in bindings) != expected_new:
            raise ManagerError(
                "prepare did not reserve the exact per-class logical page set"
            )
        class_specs.append(
            ClassLoweringSpec(
                class_id=class_config.class_id,
                pool_id=class_config.pool_id,
                last_location=(
                    token_location(
                        published,
                        prepared.previous_boundary - 1,
                        class_config.class_id,
                        arena.backend_base_index,
                    )
                    if prepared.previous_boundary % arena.page_tokens
                    else -1
                ),
                exact_new_pages=tuple(
                    sglang_page_id(
                        entry.backend_index, arena.backend_base_index
                    )
                    for entry in bindings
                ),
            )
        )
    return LoweringPlan(
        request=prepared.request,
        previous_boundary=prepared.previous_boundary,
        target_boundary=prepared.target_boundary,
        class_specs=tuple(class_specs),
    )


def bind_receipts(prepared: PreparedStep) -> tuple[BackendBindReceipt, ...]:
    return tuple(
        BackendBindReceipt(
            step=prepared.step,
            page=entry.page_lease(prepared.request.engine_epoch),
            backend_domain=entry.backend_domain,
            mapped=1,
            writable=1,
            reserved=0,
            backend_index=entry.backend_index,
        )
        for entry in prepared.view.entries
        if entry.access_flags & NEEDS_BINDING
    )


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
        self._request_rows: dict[Hashable, int] = {}
        self._row_owners: dict[int, Hashable] = {}
        self._events: list[EventGroup] = []
        self._completion_value = 1
        self._swa_retirement_certificates = 0
        self._swa_pages_reclaimed = 0
        self._swa_wrap_events = 0
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

    def acquire(self, key: Hashable) -> RequestRecord:
        with self._lock:
            self._healthy()
            existing = self._requests.get(key)
            if existing is not None:
                return existing
            lease = self.manager.request_acquire()
            if lease.engine_epoch != self.engine_epoch:
                raise ManagerError("request lease belongs to a stale engine epoch")
            _nonnegative("request slot", lease.slot)
            _positive("request generation", lease.generation)
            record = RequestRecord(
                lease=lease,
                published_view=DeviceKvView.empty(lease, self.page_tokens),
            )
            self._requests[key] = record
            return record

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

    def bind_reclamation_cleanup(self, key: Hashable, cleanup: Any) -> None:
        with self._lock:
            self._healthy()
            if not callable(cleanup):
                raise TypeError("reclamation cleanup must be callable")
            record = self.record_for(key)
            if record.pending is not None:
                raise ManagerError("cannot replace mirror cleanup during a step")
            record.reclamation_cleanup = cleanup

    def prepare(
        self, key: Hashable, target_boundary: int
    ) -> tuple[StepRecord, LoweringPlan]:
        with self._lock:
            self._healthy()
            record = self.acquire(key)
            if record.pending is not None:
                raise ManagerError("request already has a pending step")
            if target_boundary <= record.boundary:
                raise ManagerError("step target must advance the request boundary")
            prepared = self.manager.prepare_step(record.lease, target_boundary)
            try:
                self._validate_prepared(record, prepared, target_boundary)
                lowering = lowering_plan(
                    record.published_view,
                    prepared,
                    self.arenas_by_class,
                    self.config,
                )
            except Exception as error:
                step = getattr(prepared, "step", None)
                if isinstance(step, StepLease):
                    self._best_effort_quarantine_step(step)
                self.fail_stop(f"manager returned an invalid prepare receipt: {error}")
                raise FailStopped(self._failure or "invalid prepare receipt") from error
            pending = StepRecord(key=key, prepared=prepared)
            record.pending = pending
            return pending, lowering

    def mark_lowered(self, pending: StepRecord) -> None:
        with self._lock:
            self._healthy()
            self._require_pending(pending, StepPhase.PREPARED)
            pending.bind_receipts = bind_receipts(pending.prepared)
            pending.phase = StepPhase.LOWERED

    def lowering_failed(self, records: Sequence[StepRecord], error: BaseException) -> None:
        with self._lock:
            for pending in records:
                if pending.phase in (StepPhase.PREPARED, StepPhase.LOWERED):
                    self._best_effort_quarantine_step(pending.prepared.step)
                    pending.phase = StepPhase.QUARANTINED
            self.fail_stop(f"backend lowering became uncertain: {error}")

    def submit(self, pending: StepRecord) -> SubmittedStep:
        with self._lock:
            self._healthy()
            record = self._require_pending(pending, StepPhase.LOWERED)
            assumed_submission = SubmissionLease(
                pending.prepared.step.engine_epoch,
                pending.prepared.step.slot,
                pending.prepared.step.generation,
            )
            try:
                submitted = self.manager.submit_step(
                    pending.prepared.step, pending.bind_receipts
                )
                self._validate_submitted(record, pending, submitted)
            except Exception as error:
                self._best_effort_quarantine_submission(assumed_submission)
                self.fail_stop(f"manager submission became uncertain: {error}")
                pending.phase = StepPhase.QUARANTINED
                raise FailStopped(self._failure or "manager submission failed") from error
            pending.submitted = submitted
            pending.phase = StepPhase.SUBMITTED
            return submitted

    def submission_batch_failed(
        self, records: Sequence[StepRecord], error: BaseException
    ) -> None:
        with self._lock:
            self._quarantine_submitted(records)
            self.fail_stop(f"manager batch submission became uncertain: {error}")

    def candidate_mirror_failed(
        self, records: Sequence[StepRecord], error: BaseException
    ) -> None:
        with self._lock:
            self._quarantine_submitted(records)
            self.fail_stop(f"ReqToToken candidate mirror became uncertain: {error}")

    def mark_forward(self, records: Sequence[StepRecord]) -> None:
        with self._lock:
            self._healthy()
            self._unique_records(records)
            for pending in records:
                self._require_pending(pending, StepPhase.SUBMITTED)
            for pending in records:
                pending.phase = StepPhase.FORWARD

    def forward_failed(self, records: Sequence[StepRecord], error: BaseException) -> None:
        with self._lock:
            self._quarantine_submitted(records)
            self.fail_stop(f"forward launch became uncertain: {error}")

    def register_event(
        self, records: Sequence[StepRecord], event: Any, completion_domain: int
    ) -> None:
        with self._lock:
            self._healthy()
            _positive("completion domain", int(completion_domain))
            self._unique_records(records)
            for pending in records:
                self._require_pending(pending, StepPhase.FORWARD)
            if not records:
                return
            self._events.append(EventGroup(event, tuple(records), int(completion_domain)))
            for pending in records:
                pending.phase = StepPhase.EVENT

    def event_registration_failed(
        self, records: Sequence[StepRecord], error: BaseException
    ) -> None:
        with self._lock:
            self._quarantine_submitted(records)
            self.fail_stop(f"CUDA completion event registration became uncertain: {error}")

    def abort_unobserved(self, records: Sequence[StepRecord]) -> None:
        with self._lock:
            self._healthy()
            self._unique_records(records)
            for pending in reversed(records):
                self._require_pending(pending, StepPhase.PREPARED)
                try:
                    self.manager.abort_step(BackendUnobservedReceipt(pending.prepared.step))
                except Exception as error:
                    self._best_effort_quarantine_step(pending.prepared.step)
                    self.fail_stop(f"unobserved step abort became uncertain: {error}")
                    raise FailStopped(self._failure or "step abort failed") from error
                self._requests[pending.key].pending = None
                pending.phase = StepPhase.ABORTED

    def poll(self) -> None:
        with self._lock:
            self._healthy()
            ready: list[EventGroup] = []
            for group in tuple(self._events):
                try:
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
        with self._lock:
            self._healthy()
            groups = [group for group in self._events if any(r.key == key for r in group.records)]
            for group in groups:
                try:
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
                if pending.phase is StepPhase.PREPARED:
                    self.abort_unobserved((pending,))
                elif pending.phase is StepPhase.EVENT:
                    self.wait(key)
                elif pending.phase is StepPhase.LOWERED:
                    self.lowering_failed((pending,), RuntimeError("release raced lowering"))
                    raise FailStopped(self._failure or "release raced lowering")
                else:
                    self._quarantine_submitted((pending,))
                    self.fail_stop("request release raced an unproven submission")
                    raise FailStopped(self._failure or "release raced submission")

            try:
                completion = self.manager.release_request(record.lease)
                self._validate_release(record, completion)
                self._clean_reclamation_mirror(
                    record, completion.retirements, releasing=True
                )
                receipts = reclamation_receipts(completion.retirements)
                if receipts:
                    self.manager.commit_reclamations(receipts)
                self.manager.recycle_request(record.lease)
            except Exception as error:
                self.fail_stop(f"request release or reclamation became uncertain: {error}")
                raise FailStopped(self._failure or "request release failed") from error
            del self._requests[key]

    def close(self) -> None:
        with self._lock:
            self._healthy()
            if self._events or self._requests or self._request_rows or self._row_owners:
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
            or prepared.base_view_version != record.published_view.header.view_version
            or prepared.target_view_version != prepared.base_view_version + 1
        ):
            raise ManagerError("prepared step boundary or view lineage is invalid")
        header = prepared.view.header
        if (
            header.view_version != prepared.target_view_version
            or header.base_frontier != prepared.previous_boundary
            or header.target_frontier != prepared.target_boundary
        ):
            raise ManagerError("prepared candidate header is inconsistent")

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
        validate_view(
            submitted.view,
            record.lease,
            self.arenas_by_class,
            self.config,
            expected_flags=VIEW_CANDIDATE,
        )
        expected_entries = tuple(
            replace(entry, access_flags=entry.access_flags & ~NEEDS_BINDING)
            for entry in pending.prepared.view.entries
        )
        if submitted.view.header != pending.prepared.view.header or submitted.view.entries != expected_entries:
            raise ManagerError("submit changed the manager-selected candidate mapping")

    def _validate_completion(
        self,
        record: RequestRecord,
        pending: StepRecord,
        receipt: CompletionReceipt,
        completion: StepCompletion,
    ) -> None:
        assert pending.submitted is not None
        if completion.submission != pending.submitted.submission or completion.request != record.lease:
            raise ManagerError("completion belongs to another submission")
        view = completion.published_view
        validate_view(
            view,
            record.lease,
            self.arenas_by_class,
            self.config,
            expected_flags=VIEW_PUBLISHED,
        )
        if (
            view.header.view_version != pending.prepared.target_view_version
            or view.header.base_frontier != pending.prepared.target_boundary
            or view.header.target_frontier != pending.prepared.target_boundary
        ):
            raise ManagerError("completion published the wrong root version")
        expected_entries = self._expected_published_entries(
            pending.submitted.view, pending.prepared.target_boundary
        )
        if view.entries != expected_entries:
            raise ManagerError("completion published an invalid retained root")
        published_pages = {
            (entry.class_id, entry.pool_id, entry.page_id, entry.page_generation)
            for entry in view.entries
        }
        retired_entries = tuple(
            entry
            for entry in pending.submitted.view.entries
            if (
                entry.class_id,
                entry.pool_id,
                entry.page_id,
                entry.page_generation,
            )
            not in published_pages
        )
        if len(completion.retirements) != len(retired_entries):
            raise ManagerError("completion returned the wrong retirement cardinality")
        if len({item.reclamation for item in completion.retirements}) != len(
            completion.retirements
        ):
            raise ManagerError("completion duplicated a reclamation lease")
        for certificate, entry in zip(completion.retirements, retired_entries, strict=True):
            self._validate_certificate(
                certificate, record.lease, entry, receipt.completion_domain, receipt.completion_value
            )

    def _validate_release(
        self, record: RequestRecord, completion: ReleaseCompletion
    ) -> None:
        if completion.request != record.lease:
            raise ManagerError("release completion belongs to another request")
        if len(completion.retirements) != len(record.published_view.entries):
            raise ManagerError("release returned the wrong retirement cardinality")
        if len({item.reclamation for item in completion.retirements}) != len(
            completion.retirements
        ):
            raise ManagerError("release duplicated a reclamation lease")
        for certificate, entry in zip(
            completion.retirements, record.published_view.entries, strict=True
        ):
            self._validate_certificate(
                certificate,
                record.lease,
                entry,
                record.completion_domain,
                record.completion_value,
            )

    def _validate_certificate(
        self,
        certificate: ReclamationCertificate,
        request: RequestLease,
        entry: DeviceViewEntry,
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
            or certificate.page != entry.page_lease(self.engine_epoch)
            or certificate.class_id != entry.class_id
            or certificate.backend_domain != entry.backend_domain
            or certificate.logical_ordinal != entry.logical_ordinal
            or certificate.backend_index != entry.backend_index
            or certificate.token_begin != entry.token_begin
            or certificate.token_end_exclusive
            != entry.token_begin + entry.valid_token_count
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

    def _expected_published_entries(
        self, candidate: DeviceKvView, target: int
    ) -> tuple[DeviceViewEntry, ...]:
        retained: list[DeviceViewEntry] = []
        classes = self.config.classes_by_id
        for entry in candidate.entries:
            class_config = classes[entry.class_id]
            arena = self.arenas_by_class[entry.class_id]
            retain_start = (
                0
                if class_config.retention == "full"
                else max(0, target - (int(class_config.window_tokens) - 1))
            )
            end = entry.token_begin + entry.valid_token_count
            last_ordinal = (target - 1) // arena.page_tokens
            partial_last = bool(target % arena.page_tokens) and (
                entry.logical_ordinal == last_ordinal
            )
            if (
                (end <= retain_start or entry.token_begin >= target)
                and not partial_last
            ):
                continue
            visible_begin = max(retain_start, entry.token_begin)
            visible_end = min(target, entry.token_begin + arena.page_tokens)
            visible_count = max(0, visible_end - visible_begin)
            retained.append(
                replace(
                    entry,
                    access_flags=ACCESS_READ if visible_count else 0,
                    valid_token_count=min(
                        arena.page_tokens, target - entry.token_begin
                    ),
                    visible_token_offset=visible_begin - entry.token_begin,
                    visible_token_count=visible_count,
                )
            )
        return tuple(retained)

    def _complete_group(self, group: EventGroup) -> None:
        if group not in self._events:
            return
        completed: list[tuple[StepRecord, CompletionReceipt, StepCompletion]] = []
        try:
            for pending in group.records:
                record = self._require_pending(pending, StepPhase.EVENT)
                assert pending.submitted is not None
                receipt = CompletionReceipt(
                    submission=pending.submitted.submission,
                    completion_domain=group.completion_domain,
                    completion_value=self._completion_value,
                )
                self._completion_value += 1
                completion = self.manager.complete_step(receipt)
                self._validate_completion(record, pending, receipt, completion)
                completed.append((pending, receipt, completion))
            for pending, receipt, completion in completed:
                record = self._requests[pending.key]
                self._clean_reclamation_mirror(
                    record, completion.retirements, releasing=False
                )
                record.published_view = completion.published_view
                record.boundary = pending.prepared.target_boundary
                record.completion_domain = receipt.completion_domain
                record.completion_value = receipt.completion_value
            all_certificates = tuple(
                certificate
                for _, _, completion in completed
                for certificate in completion.retirements
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
            for pending, _receipt, completion in completed:
                record = self._requests[pending.key]
                for class_id in sliding_class_ids:
                    previous_cycle = record.swa_temporal_cycles.get(class_id, 0)
                    current_cycle = max(
                        (
                            entry.temporal_cycle
                            for entry in completion.published_view.entries
                            if entry.class_id == class_id
                        ),
                        default=previous_cycle,
                    )
                    if current_cycle < previous_cycle:
                        raise ManagerError("SWA temporal cycle moved backwards")
                    wrap_events += current_cycle - previous_cycle
                    cycle_updates.append((record, class_id, current_cycle))
            receipts = reclamation_receipts(all_certificates)
            if receipts:
                self.manager.commit_reclamations(receipts)
            self._swa_retirement_certificates += len(sliding_certificates)
            self._swa_pages_reclaimed += len(sliding_certificates)
            self._swa_wrap_events += wrap_events
            for record, class_id, current_cycle in cycle_updates:
                record.swa_temporal_cycles[class_id] = current_cycle
        except Exception as error:
            remaining = [
                pending
                for pending in group.records
                if pending.submitted is not None and pending.phase is StepPhase.EVENT
            ]
            self._quarantine_submitted(remaining)
            self.fail_stop(f"GPU completion publication became uncertain: {error}")
            raise FailStopped(self._failure or "completion failed") from error

        for pending, _receipt, _completion in completed:
            record = self._requests[pending.key]
            record.pending = None
            pending.phase = StepPhase.COMPLETED
        self._events.remove(group)

    @staticmethod
    def _clean_reclamation_mirror(
        record: RequestRecord,
        certificates: Sequence[ReclamationCertificate],
        *,
        releasing: bool,
    ) -> None:
        if not certificates and not releasing:
            return
        if record.reclamation_cleanup is None:
            raise ManagerError("request has no ReqToToken reclamation cleanup")
        record.reclamation_cleanup(tuple(certificates), releasing)

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
    def _unique_records(records: Sequence[StepRecord]) -> None:
        if len({id(record) for record in records}) != len(records):
            raise ManagerError("step batch contains a duplicate record")

    def _quarantine_submitted(self, records: Sequence[StepRecord]) -> None:
        for pending in records:
            if pending.phase is StepPhase.QUARANTINED:
                continue
            if pending.submitted is None:
                self._best_effort_quarantine_step(pending.prepared.step)
            else:
                self._best_effort_quarantine_submission(pending.submitted.submission)
            pending.phase = StepPhase.QUARANTINED

    def _best_effort_quarantine_step(self, step: StepLease) -> None:
        try:
            self.manager.quarantine_step(step)
        except Exception:
            pass

    def _best_effort_quarantine_submission(self, submission: SubmissionLease) -> None:
        try:
            self.manager.quarantine_submission(submission)
        except Exception:
            pass


__all__ = [
    "ACCESS_READ",
    "ACCESS_WRITE",
    "ArenaIdentity",
    "ArenaRegistration",
    "ArenaStats",
    "BackendBindReceipt",
    "BackendUnobservedReceipt",
    "CanonicalRuntime",
    "CompletionReceipt",
    "DeviceKvView",
    "DeviceViewEntry",
    "DeviceViewHeader",
    "FailStopped",
    "ClassLoweringSpec",
    "LoweringPlan",
    "ManagerCreateSettings",
    "ManagerError",
    "ManagerFactoryProtocol",
    "ManagerProtocol",
    "ManagerStats",
    "NEEDS_BINDING",
    "PageLease",
    "PreparedStep",
    "ReclamationCertificate",
    "ReclamationLease",
    "ReclamationReceipt",
    "ReleaseCompletion",
    "RequestLease",
    "StepCompletion",
    "StepLease",
    "StepPhase",
    "StepRecord",
    "SubmissionLease",
    "SubmittedStep",
    "VIEW_CANDIDATE",
    "VIEW_PUBLISHED",
    "bind_receipts",
    "expected_new_ordinals",
    "lowering_plan",
    "reclamation_receipts",
    "sglang_page_id",
    "token_location",
    "validate_view",
]
