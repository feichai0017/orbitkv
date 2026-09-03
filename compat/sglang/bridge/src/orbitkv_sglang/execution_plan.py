from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import ClassConfig, RuntimeConfig
from .ffi.session_types import (
    EngineBatchPlan,
    EngineBatchId,
    EngineBindEvidence,
    EngineCopyEvidence,
    EngineRequestId,
    EngineRequestView,
    EngineStepExecutionEvidence,
    EngineStepPlan,
    ExecutionEvidence,
)
from .runtime.identity import (
    CLASS_LOWERING_PACKED,
    TAIL_COPY_ON_WRITE,
    TAIL_FRESH,
    TAIL_IN_PLACE,
    TAIL_NONE,
    ArenaIdentity,
    ManagerError,
    PageLease,
)
from .runtime.snapshot_shadow import (
    CLASS_LOWERING_EPOCH_START,
    CLASS_LOWERING_RESETTABLE,
    ClassLowering,
    CopyIntent,
    TailAction,
    WriteIntent,
)


_VALID_CLASS_LOWERING_FLAGS = frozenset(
    (
        0,
        CLASS_LOWERING_PACKED,
        CLASS_LOWERING_RESETTABLE,
        CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START,
    )
)
_ZERO_PAGE = PageLease(0, 0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class ClassSpec:
    """One validated class lowering in SGLang physical coordinates."""

    class_id: int
    pool_id: int
    last_location: int
    exact_new_pages: tuple[int, ...]
    tail_action: TailAction
    copy_intents: tuple[CopyIntent, ...]
    previous_layout_boundary: int
    target_layout_boundary: int


@dataclass(frozen=True, slots=True)
class StepPlan:
    """Lease-free physical lowering view for one runtime-session step."""

    request_id: EngineRequestId
    previous_boundary: int
    target_boundary: int
    class_specs: tuple[ClassSpec, ...]

    @property
    def by_class(self) -> dict[int, ClassSpec]:
        return {item.class_id: item for item in self.class_specs}


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Ordered physical lowering view for one runtime-session batch."""

    batch_id: EngineBatchId
    steps: tuple[StepPlan, ...]
    arenas: tuple[ArenaIdentity, ...]
    source_steps: tuple[EngineStepPlan, ...]


@dataclass(frozen=True, slots=True)
class BindingSpec:
    """One exact page binding selected by the runtime session."""

    page: PageLease
    backend_domain: int
    backend_index: int


@dataclass(frozen=True, slots=True)
class BindingResult:
    """One backend-observed page mapping and write permission."""

    page: PageLease
    backend_domain: int
    backend_index: int
    mapped: bool
    writable: bool


@dataclass(frozen=True, slots=True)
class StepExecutionResult:
    """Backend-observed success facts for one ordered execution step."""

    request_id: EngineRequestId
    bindings: tuple[BindingResult, ...]
    completed_copies: tuple[CopyIntent, ...]
    copies_ordered_before_writes: bool


def _integer(
    name: str,
    value: object,
    *,
    positive: bool = False,
    bits: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManagerError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ManagerError(f"{name} must be a {qualifier} integer")
    if bits is not None and value >= 1 << bits:
        raise ManagerError(f"{name} is outside uint{bits}_t")
    return value


def _class_table(
    config: RuntimeConfig, arenas: Sequence[ArenaIdentity]
) -> tuple[tuple[ClassConfig, ArenaIdentity], ...]:
    if not isinstance(config, RuntimeConfig):
        raise ManagerError("execution plan requires a RuntimeConfig")
    classes = tuple(config.classes)
    identities = tuple(arenas)
    if not classes or len(identities) != len(classes):
        raise ManagerError("one arena identity is required for every plan class")
    page_tokens = _integer(
        "runtime page tokens", config.page_tokens, positive=True, bits=32
    )
    result = []
    session_epoch = None
    page_ranges: list[tuple[int, int]] = []
    backend_ranges: list[tuple[int, int, int]] = []
    pool_ids: set[int] = set()
    for expected_class_id, (class_config, arena) in enumerate(
        zip(classes, identities, strict=True)
    ):
        if not isinstance(arena, ArenaIdentity):
            raise ManagerError("session arena identity has the wrong type")
        if not isinstance(class_config, ClassConfig):
            raise ManagerError("runtime plan class has the wrong type")
        class_id = _integer(
            "plan class id", class_config.class_id, bits=16
        )
        arena_class_id = _integer(
            "arena class id", arena.class_id, bits=16
        )
        if (
            class_id != expected_class_id
            or arena_class_id != expected_class_id
        ):
            raise ManagerError("plan classes and arenas must be class-id ordered")
        _integer(
            "class pool id", class_config.pool_id, positive=True, bits=32
        )
        _integer("class backend domain", class_config.backend_domain, bits=16)
        _integer("arena pool id", arena.pool_id, positive=True, bits=32)
        _integer("arena backend domain", arena.backend_domain, bits=16)
        if (
            arena.pool_id != class_config.pool_id
            or arena.backend_domain != class_config.backend_domain
        ):
            raise ManagerError("session arena differs from its plan class")
        engine_epoch = _integer(
            "arena engine epoch", arena.engine_epoch, positive=True, bits=64
        )
        _integer(
            "arena pool epoch", arena.pool_epoch, positive=True, bits=64
        )
        page_count = _integer(
            "arena page count", arena.page_count, positive=True, bits=32
        )
        first_page_id = _integer(
            "arena first page id", arena.first_page_id, positive=True, bits=32
        )
        backend_base = _integer(
            "arena backend base index", arena.backend_base_index, bits=64
        )
        _integer("arena page tokens", arena.page_tokens, positive=True, bits=32)
        if arena.page_tokens != page_tokens:
            raise ManagerError("session arena page size differs from the runtime plan")
        if session_epoch is None:
            session_epoch = engine_epoch
        elif engine_epoch != session_epoch:
            raise ManagerError("session arenas have different engine epochs")
        page_end = first_page_id + page_count
        if page_end > 1 << 32:
            raise ManagerError("session arena page-id range overflows uint32_t")
        if any(
            first_page_id < other_end and other_begin < page_end
            for other_begin, other_end in page_ranges
        ):
            raise ManagerError("session arena page-id ranges overlap")
        page_ranges.append((first_page_id, page_end))
        if arena.pool_id in pool_ids:
            raise ManagerError("session arenas have duplicate pool identities")
        pool_ids.add(arena.pool_id)
        backend_end = backend_base + page_count
        if backend_end > 1 << 64:
            raise ManagerError("session backend arena range overflows uint64_t")
        if any(
            domain == arena.backend_domain
            and backend_base < other_end
            and other_begin < backend_end
            for domain, other_begin, other_end in backend_ranges
        ):
            raise ManagerError("session backend arena ranges overlap")
        backend_ranges.append((arena.backend_domain, backend_base, backend_end))
        result.append((class_config, arena))
    return tuple(result)


def _backend_index(page_id: int, arena: ArenaIdentity) -> int:
    page_id = _integer("page id", page_id, positive=True, bits=32)
    if not arena.first_page_id <= page_id < arena.first_page_id + arena.page_count:
        raise ManagerError("page is outside its class arena")
    return arena.backend_base_index + page_id - arena.first_page_id


def _sglang_page_id(page_id: int, arena: ArenaIdentity) -> int:
    backend_index = _backend_index(page_id, arena)
    result = backend_index - arena.backend_base_index + 1
    if result <= 0 or result > arena.page_count:
        raise ManagerError("manager page cannot lower into the SGLang arena")
    return result


def _validate_page(
    page: PageLease, arena: ArenaIdentity, label: str, *, zero: bool = False
) -> int | None:
    if not isinstance(page, PageLease):
        raise ManagerError(f"{label} has the wrong page type")
    _integer(f"{label} engine epoch", page.engine_epoch, bits=64)
    _integer(f"{label} pool epoch", page.pool_epoch, bits=64)
    _integer(f"{label} generation", page.generation, bits=64)
    _integer(f"{label} page id", page.page_id, bits=32)
    _integer(f"{label} pool id", page.pool_id, bits=32)
    if zero:
        if page != _ZERO_PAGE:
            raise ManagerError(f"{label} must be the zero page")
        return None
    if (
        page.engine_epoch != arena.engine_epoch
        or page.pool_epoch != arena.pool_epoch
        or page.pool_id != arena.pool_id
    ):
        raise ManagerError(f"{label} belongs to another backend arena")
    _integer(f"{label} generation", page.generation, positive=True, bits=64)
    return _backend_index(page.page_id, arena)


def _span(
    offset: object,
    count: object,
    cursor: int,
    total: int,
    label: str,
) -> int:
    offset_value = _integer(f"{label} offset", offset, bits=32)
    count_value = _integer(f"{label} count", count, bits=32)
    if offset_value != cursor or count_value > total - cursor:
        raise ManagerError(f"{label} span is not canonical and in range")
    return cursor + count_value


def _validate_class_flags(
    class_config: ClassConfig,
    flags: object,
    previous: int,
    target: int,
    page_tokens: int,
) -> int:
    value = _integer("class lowering flags", flags, bits=16)
    if value not in _VALID_CLASS_LOWERING_FLAGS:
        raise ManagerError("class lowering flags are invalid")
    packed = bool(value & CLASS_LOWERING_PACKED)
    resettable = bool(value & CLASS_LOWERING_RESETTABLE)
    epoch_start = bool(value & CLASS_LOWERING_EPOCH_START)
    retention = class_config.retention
    if packed and retention != "full":
        raise ManagerError("packed lowering requires full retention")
    if resettable != (retention == "chunked"):
        raise ManagerError("resettable lowering disagrees with class retention")
    if epoch_start and not resettable:
        raise ManagerError("epoch-start lowering must be resettable")
    if resettable:
        chunk_tokens = getattr(class_config, "chunk_tokens", None)
        blocks_per_epoch = getattr(class_config, "blocks_per_epoch", None)
        if (
            isinstance(chunk_tokens, bool)
            or not isinstance(chunk_tokens, int)
            or chunk_tokens <= 0
            or isinstance(blocks_per_epoch, bool)
            or not isinstance(blocks_per_epoch, int)
            or blocks_per_epoch <= 0
            or chunk_tokens != blocks_per_epoch * page_tokens
            or previous // chunk_tokens != (target - 1) // chunk_tokens
            or epoch_start != (previous % chunk_tokens == 0)
        ):
            raise ManagerError("resettable lowering geometry is invalid")
    return value


def _write_page(
    intent: object, arena: ArenaIdentity
) -> tuple[PageLease, int]:
    if not isinstance(intent, WriteIntent):
        raise ManagerError("write intent has the wrong type")
    if _integer("write intent reserved field", intent.reserved, bits=32) != 0:
        raise ManagerError("write intent reserved field is nonzero")
    generation = _integer(
        "write intent page generation",
        intent.page_generation,
        positive=True,
        bits=64,
    )
    backend_index = _backend_index(intent.page_id, arena)
    return (
        PageLease(
            arena.engine_epoch,
            arena.pool_epoch,
            generation,
            intent.page_id,
            arena.pool_id,
        ),
        backend_index,
    )


def _class_spec(
    step: EngineStepPlan,
    class_config: ClassConfig,
    arena: ArenaIdentity,
    lowering: ClassLowering,
    cursors: tuple[int, int, int],
    page_tokens: int,
) -> tuple[ClassSpec, tuple[int, int, int]]:
    if not isinstance(lowering, ClassLowering):
        raise ManagerError("class lowering has the wrong type")
    class_id = _integer("plan class id", class_config.class_id, bits=16)
    lowering_class_id = _integer(
        "lowering class id", lowering.class_id, bits=16
    )
    if lowering_class_id != class_id:
        raise ManagerError("step class lowerings are not in compiled order")
    if _integer("class lowering reserved field", lowering.reserved, bits=32) != 0:
        raise ManagerError("class lowering reserved field is nonzero")
    previous = _integer(
        "class previous layout boundary",
        lowering.previous_layout_boundary,
        bits=64,
    )
    target = _integer(
        "class target layout boundary",
        lowering.target_layout_boundary,
        bits=64,
    )
    if (
        target < previous
        or target - previous != step.target_boundary - step.previous_boundary
    ):
        raise ManagerError("class layout boundaries changed the append delta")
    flags = _validate_class_flags(
        class_config, lowering.flags, previous, target, page_tokens
    )
    if class_config.retention != "full" or not (
        flags & CLASS_LOWERING_PACKED
    ):
        if previous != step.previous_boundary or target != step.target_boundary:
            raise ManagerError("dense class layout boundary diverged")

    tail_end = _span(
        lowering.tail_offset,
        lowering.tail_count,
        cursors[0],
        len(step.tail_actions),
        "class tail",
    )
    copy_end = _span(
        lowering.copy_offset,
        lowering.copy_count,
        cursors[1],
        len(step.copy_intents),
        "class copy",
    )
    write_end = _span(
        lowering.write_offset,
        lowering.write_count,
        cursors[2],
        len(step.write_intents),
        "class write",
    )
    if tail_end - cursors[0] != 1 or copy_end - cursors[1] > 1:
        raise ManagerError("class tail/copy spans are not canonical")
    expected_writes = (target + page_tokens - 1) // page_tokens - (
        (previous + page_tokens - 1) // page_tokens
    )
    if write_end - cursors[2] != expected_writes:
        raise ManagerError("class write span has the wrong page count")

    action = step.tail_actions[cursors[0]]
    copies = step.copy_intents[cursors[1]:copy_end]
    if not isinstance(action, TailAction):
        raise ManagerError("tail action has the wrong type")
    action_class_id = _integer("tail action class id", action.class_id, bits=16)
    action_kind = _integer("tail action kind", action.kind, bits=16)
    valid_tokens = _integer(
        "tail action valid token count", action.valid_token_count, bits=32
    )
    logical_ordinal = _integer(
        "tail action logical ordinal", action.logical_ordinal, bits=64
    )
    if action_class_id != class_id or _integer(
        "tail action reserved field", action.reserved, bits=64
    ) != 0:
        raise ManagerError("tail action does not belong to its class")
    partial = previous % page_tokens
    ordinal = previous // page_tokens if partial else 0
    last_location = -1
    if not partial:
        if (
            action_kind != TAIL_NONE
            or valid_tokens != 0
            or logical_ordinal != 0
            or copies
        ):
            raise ManagerError("aligned append returned a nonempty tail action")
        _validate_page(action.source, arena, "tail source", zero=True)
        _validate_page(action.destination, arena, "tail destination", zero=True)
    elif (
        logical_ordinal != ordinal
        or valid_tokens not in (0, partial)
    ):
        raise ManagerError("tail action geometry is invalid")
    elif action_kind == TAIL_IN_PLACE:
        if (
            valid_tokens != partial
            or action.source != action.destination
            or copies
        ):
            raise ManagerError("in-place tail action is inconsistent")
        backend_index = _validate_page(action.destination, arena, "tail page")
        assert backend_index is not None
        last_location = (
            (backend_index - arena.backend_base_index + 1) * page_tokens
            + partial - 1
        )
    elif action_kind == TAIL_COPY_ON_WRITE:
        if (
            valid_tokens != partial
            or action.source == action.destination
            or len(copies) != 1
        ):
            raise ManagerError("copy-on-write tail action is inconsistent")
        source_index = _validate_page(action.source, arena, "tail source")
        destination_index = _validate_page(
            action.destination, arena, "tail destination"
        )
        copy = copies[0]
        if not isinstance(copy, CopyIntent):
            raise ManagerError("copy intent has the wrong type")
        copy_class_id = _integer("copy intent class id", copy.class_id, bits=16)
        copy_backend_domain = _integer(
            "copy intent backend domain", copy.backend_domain, bits=16
        )
        copy_tokens = _integer(
            "copy intent token count", copy.token_count, bits=32
        )
        source_offset = _integer(
            "copy intent source token offset",
            copy.source_token_offset,
            bits=32,
        )
        destination_offset = _integer(
            "copy intent destination token offset",
            copy.destination_token_offset,
            bits=32,
        )
        copy_reserved = _integer(
            "copy intent reserved field", copy.reserved, bits=32
        )
        copy_source_index = _integer(
            "copy intent source backend index",
            copy.source_backend_index,
            bits=64,
        )
        copy_destination_index = _integer(
            "copy intent destination backend index",
            copy.destination_backend_index,
            bits=64,
        )
        if (
            copy_class_id != class_id
            or copy_backend_domain != arena.backend_domain
            or copy_tokens != partial
            or source_offset != 0
            or destination_offset != 0
            or copy_reserved != 0
            or copy.source != action.source
            or copy.destination != action.destination
            or copy_source_index != source_index
            or copy_destination_index != destination_index
        ):
            raise ManagerError("copy intent is not an exact tail echo")
        assert destination_index is not None
        last_location = (
            (destination_index - arena.backend_base_index + 1) * page_tokens
            + partial - 1
        )
    elif action_kind == TAIL_FRESH:
        if (
            valid_tokens != 0
            or copies
        ):
            raise ManagerError("fresh tail action is inconsistent")
        _validate_page(action.source, arena, "tail source", zero=True)
        destination_index = _validate_page(
            action.destination, arena, "tail destination"
        )
        assert destination_index is not None
        last_location = (
            (destination_index - arena.backend_base_index + 1) * page_tokens
            + partial - 1
        )
    else:
        raise ManagerError("tail action kind is invalid")

    exact_new_pages = []
    seen = set()
    if action_kind in (TAIL_IN_PLACE, TAIL_COPY_ON_WRITE):
        seen.add(action.source.page_id)
    if action_kind in (TAIL_COPY_ON_WRITE, TAIL_FRESH):
        seen.add(action.destination.page_id)
    for intent in step.write_intents[cursors[2]:write_end]:
        page, _backend = _write_page(intent, arena)
        if page.page_id in seen:
            raise ManagerError("class lowering aliases a destination page")
        seen.add(page.page_id)
        exact_new_pages.append(_sglang_page_id(page.page_id, arena))
    return (
        ClassSpec(
            class_id=class_id,
            pool_id=arena.pool_id,
            last_location=last_location,
            exact_new_pages=tuple(exact_new_pages),
            tail_action=action,
            copy_intents=copies,
            previous_layout_boundary=previous,
            target_layout_boundary=target,
        ),
        (tail_end, copy_end, write_end),
    )


def _step_plan(
    step: EngineStepPlan,
    classes: tuple[tuple[ClassConfig, ArenaIdentity], ...],
    page_tokens: int,
    expected_view: EngineRequestView,
) -> StepPlan:
    if not isinstance(step, EngineStepPlan):
        raise ManagerError("batch contains a non-EngineStepPlan value")
    request_id = _integer("engine request id", step.request_id, bits=64)
    base_version = _integer(
        "base view version", step.base_view_version, bits=64
    )
    target_version = _integer(
        "target view version", step.target_view_version, bits=64
    )
    previous = _integer("step previous boundary", step.previous_boundary, bits=64)
    target = _integer("step target boundary", step.target_boundary, bits=64)
    if not isinstance(expected_view, EngineRequestView):
        raise ManagerError("expected request view has the wrong type")
    expected_request_id = _integer(
        "expected request id", expected_view.request_id, bits=64
    )
    expected_version = _integer(
        "expected base view version", expected_view.view_version, bits=64
    )
    expected_boundary = _integer(
        "expected previous boundary", expected_view.boundary, bits=64
    )
    _integer(
        "expected resident count", expected_view.resident_count, bits=32
    )
    if (
        request_id != expected_request_id
        or base_version != expected_version
        or previous != expected_boundary
    ):
        raise ManagerError("step differs from the current request mirror")
    for name, values in (
        ("class lowerings", step.class_lowerings),
        ("tail actions", step.tail_actions),
        ("copy intents", step.copy_intents),
        ("write intents", step.write_intents),
    ):
        if not isinstance(values, tuple):
            raise ManagerError(f"step {name} must be a tuple")
    if (
        target <= previous
        or base_version == (1 << 64) - 1
        or target_version != base_version + 1
    ):
        raise ManagerError("step boundary or view version is not monotonic")
    if len(step.class_lowerings) != len(classes):
        raise ManagerError("step has the wrong class cardinality")
    if (
        expected_boundary == 0
        and expected_view.resident_count == 0
        and any(
            lowering.flags & CLASS_LOWERING_PACKED
            for lowering in step.class_lowerings
        )
    ):
        raise ManagerError("initial request view cannot use packed lowering")
    specs = []
    cursors = (0, 0, 0)
    destination_pages: set[int] = set()
    for (class_config, arena), lowering in zip(
        classes, step.class_lowerings, strict=True
    ):
        spec, cursors = _class_spec(
            step, class_config, arena, lowering, cursors, page_tokens
        )
        pages = (
            (spec.tail_action.destination.page_id,)
            if spec.tail_action.kind in (TAIL_COPY_ON_WRITE, TAIL_FRESH)
            else ()
        ) + tuple(
            step.write_intents[index].page_id
            for index in range(
                lowering.write_offset, lowering.write_offset + lowering.write_count
            )
        )
        if any(page in destination_pages for page in pages):
            raise ManagerError("step aliases a destination page across classes")
        destination_pages.update(pages)
        specs.append(spec)
    if cursors != (
        len(step.tail_actions), len(step.copy_intents), len(step.write_intents)
    ):
        raise ManagerError("class spans do not cover the step outputs")
    return StepPlan(EngineRequestId(request_id), previous, target, tuple(specs))


def lower_batch_plan(
    batch_plan: EngineBatchPlan,
    config: RuntimeConfig,
    arenas: Sequence[ArenaIdentity],
    expected_views: Mapping[EngineRequestId, EngineRequestView],
) -> BatchPlan:
    """Validate and lower a Rust session plan into SGLang page coordinates."""

    if not isinstance(batch_plan, EngineBatchPlan):
        raise ManagerError("execution lowering requires an EngineBatchPlan")
    if not isinstance(batch_plan.steps, tuple):
        raise ManagerError("batch steps must be a tuple")
    if not isinstance(expected_views, Mapping):
        raise ManagerError("expected request views must be a mapping")
    classes = _class_table(config, arenas)
    session_epoch = classes[0][1].engine_epoch
    if not isinstance(batch_plan.batch_id, EngineBatchId):
        raise ManagerError("batch identity has the wrong type")
    _integer(
        "batch session epoch",
        batch_plan.batch_id.session_epoch,
        positive=True,
        bits=64,
    )
    _integer(
        "batch sequence", batch_plan.batch_id.sequence, positive=True, bits=64
    )
    if (
        batch_plan.batch_id.session_epoch != session_epoch
        or not batch_plan.steps
    ):
        raise ManagerError("batch identity or cardinality is invalid")
    request_ids = tuple(step.request_id for step in batch_plan.steps)
    if set(expected_views) != set(request_ids):
        raise ManagerError("expected request views do not exactly match the batch")
    steps = tuple(
        _step_plan(
            step, classes, config.page_tokens, expected_views[step.request_id]
        )
        for step in batch_plan.steps
    )
    request_ids = tuple(step.request_id for step in steps)
    if len(set(request_ids)) != len(request_ids):
        raise ManagerError("batch contains duplicate request ids")
    destination_pages = [
        (spec.pool_id, page_id)
        for source_step, step in zip(batch_plan.steps, steps, strict=True)
        for lowering, spec in zip(
            source_step.class_lowerings, step.class_specs, strict=True
        )
        for page_id in (
            (spec.tail_action.destination.page_id,)
            if spec.tail_action.kind in (TAIL_COPY_ON_WRITE, TAIL_FRESH)
            else ()
        )
        + tuple(
            source_step.write_intents[index].page_id
            for index in range(
                lowering.write_offset,
                lowering.write_offset + lowering.write_count,
            )
        )
    ]
    if len(set(destination_pages)) != len(destination_pages):
        raise ManagerError("batch aliases a destination page across steps")
    destination_set = set(destination_pages)
    tail_sources = (
        (action.source.pool_id, action.source.page_id)
        for source_step in batch_plan.steps
        for action in source_step.tail_actions
        if action.kind in (TAIL_IN_PLACE, TAIL_COPY_ON_WRITE)
    )
    if any(page in destination_set for page in tail_sources):
        raise ManagerError("batch aliases a live tail and destination page")
    return BatchPlan(
        batch_plan.batch_id,
        steps,
        tuple(arena for _config, arena in classes),
        batch_plan.steps,
    )


def _expected_bindings(
    step: EngineStepPlan,
    lowered: StepPlan,
    arenas: Mapping[int, ArenaIdentity],
) -> tuple[BindingSpec, ...]:
    values = []
    for spec, lowering in zip(
        lowered.class_specs, step.class_lowerings, strict=True
    ):
        arena = arenas[spec.class_id]
        if spec.tail_action.kind in (TAIL_COPY_ON_WRITE, TAIL_FRESH):
            page = spec.tail_action.destination
            backend_index = _validate_page(page, arena, "tail bind page")
            assert backend_index is not None
            values.append(
                BindingSpec(page, arena.backend_domain, backend_index)
            )
        write_begin = lowering.write_offset
        write_end = write_begin + lowering.write_count
        for intent in step.write_intents[write_begin:write_end]:
            page, backend_index = _write_page(intent, arena)
            values.append(
                BindingSpec(page, arena.backend_domain, backend_index)
            )
    return tuple(values)


def _reconstructed_class_specs(
    step: EngineStepPlan, arenas: Mapping[int, ArenaIdentity]
) -> tuple[ClassSpec, ...]:
    """Rebuild every execution-facing field without trusting BatchPlan."""

    values = []
    cursors = (0, 0, 0)
    for lowering in step.class_lowerings:
        if lowering.class_id not in arenas:
            raise ManagerError("lowered plan names an unknown class arena")
        arena = arenas[lowering.class_id]
        tail_end = _span(
            lowering.tail_offset, lowering.tail_count, cursors[0],
            len(step.tail_actions), "confirmation class tail"
        )
        copy_end = _span(
            lowering.copy_offset, lowering.copy_count, cursors[1],
            len(step.copy_intents), "confirmation class copy"
        )
        write_end = _span(
            lowering.write_offset, lowering.write_count, cursors[2],
            len(step.write_intents), "confirmation class write"
        )
        if tail_end - cursors[0] != 1:
            raise ManagerError("confirmation tail span is not canonical")
        action = step.tail_actions[cursors[0]]
        copies = step.copy_intents[cursors[1]:copy_end]
        partial = lowering.previous_layout_boundary % arena.page_tokens
        last_location = -1
        if action.kind in (TAIL_IN_PLACE, TAIL_COPY_ON_WRITE, TAIL_FRESH):
            backend_index = _validate_page(
                action.destination, arena, "confirmation tail page"
            )
            assert backend_index is not None
            last_location = (
                (backend_index - arena.backend_base_index + 1)
                * arena.page_tokens
                + partial
                - 1
            )
        exact_new_pages = tuple(
            _sglang_page_id(intent.page_id, arena)
            for intent in step.write_intents[cursors[2]:write_end]
        )
        values.append(
            ClassSpec(
                lowering.class_id, arena.pool_id, last_location,
                exact_new_pages, action, copies,
                lowering.previous_layout_boundary,
                lowering.target_layout_boundary,
            )
        )
        cursors = (tail_end, copy_end, write_end)
    if cursors != (
        len(step.tail_actions), len(step.copy_intents), len(step.write_intents)
    ):
        raise ManagerError("confirmation spans do not cover plan outputs")
    return tuple(values)


def expected_bindings(
    batch_plan: EngineBatchPlan,
    lowered: BatchPlan,
    arenas: Sequence[ArenaIdentity],
) -> tuple[tuple[BindingSpec, ...], ...]:
    """Return exact ordered bindings for the backend mapping operation."""

    if not isinstance(batch_plan, EngineBatchPlan) or not isinstance(
        lowered, BatchPlan
    ):
        raise ManagerError("binding selection requires matching batch plans")
    identities = tuple(arenas)
    if (
        lowered.batch_id != batch_plan.batch_id
        or len(lowered.steps) != len(batch_plan.steps)
        or lowered.arenas != identities
        or lowered.source_steps != batch_plan.steps
        or not identities
        or any(not isinstance(item, ArenaIdentity) for item in identities)
        or tuple(item.class_id for item in identities)
        != tuple(range(len(identities)))
    ):
        raise ManagerError("binding selection identity or cardinality changed")
    table = {item.class_id: item for item in identities}
    expected_class_ids = tuple(range(len(identities)))
    values = []
    for source_step, step in zip(
        batch_plan.steps, lowered.steps, strict=True
    ):
        if (
            source_step.request_id != step.request_id
            or source_step.previous_boundary != step.previous_boundary
            or source_step.target_boundary != step.target_boundary
            or tuple(item.class_id for item in step.class_specs)
            != expected_class_ids
            or tuple(item.class_id for item in source_step.class_lowerings)
            != expected_class_ids
        ):
            raise ManagerError("binding selection lowered plan changed")
        if step.class_specs != _reconstructed_class_specs(source_step, table):
            raise ManagerError("binding selection class specs changed")
        values.append(_expected_bindings(source_step, step, table))
    return tuple(values)


def confirm_execution(
    batch_plan: EngineBatchPlan,
    lowered: BatchPlan,
    arenas: Sequence[ArenaIdentity],
    results: Sequence[StepExecutionResult],
) -> ExecutionEvidence:
    """Validate observed backend results and construct exact proof DTOs.

    The caller must obtain ``results`` only after successfully mapping every
    selected page and completing every advertised copy in write-before-use
    order. Any failure or uncertainty must quarantine the prepared batch.
    """

    if not isinstance(batch_plan, EngineBatchPlan):
        raise ManagerError("execution confirmation requires an EngineBatchPlan")
    if not isinstance(lowered, BatchPlan):
        raise ManagerError("execution confirmation requires a BatchPlan")
    identities = tuple(arenas)
    if not identities or any(
        not isinstance(item, ArenaIdentity) for item in identities
    ):
        raise ManagerError("execution confirmation requires arena identities")
    if tuple(item.class_id for item in identities) != tuple(range(len(identities))):
        raise ManagerError("execution arenas must be class-id ordered")
    arenas_by_class = {item.class_id: item for item in identities}
    expected_class_ids = tuple(range(len(identities)))
    values = tuple(results)
    if (
        lowered.batch_id != batch_plan.batch_id
        or len(lowered.steps) != len(batch_plan.steps)
        or lowered.arenas != identities
        or lowered.source_steps != batch_plan.steps
        or len(values) != len(batch_plan.steps)
    ):
        raise ManagerError("execution result cardinality or batch identity changed")
    evidence_steps = []
    for source_step, step, result in zip(
        batch_plan.steps, lowered.steps, values, strict=True
    ):
        if not isinstance(result, StepExecutionResult):
            raise ManagerError("execution result has the wrong type")
        if (
            source_step.request_id != step.request_id
            or result.request_id != step.request_id
            or source_step.previous_boundary != step.previous_boundary
            or source_step.target_boundary != step.target_boundary
            or tuple(item.class_id for item in step.class_specs)
            != expected_class_ids
            or tuple(item.class_id for item in source_step.class_lowerings)
            != expected_class_ids
        ):
            raise ManagerError("execution result or lowered plan changed")
        if step.class_specs != _reconstructed_class_specs(
            source_step, arenas_by_class
        ):
            raise ManagerError("execution lowered class specs changed")
        if not isinstance(result.copies_ordered_before_writes, bool):
            raise ManagerError("copy ordering witness must be a boolean")
        if not result.copies_ordered_before_writes:
            raise ManagerError("copies were not observed before writes")
        observed_bindings = tuple(result.bindings)
        observed_copies = tuple(result.completed_copies)
        if any(not isinstance(item, BindingResult) for item in observed_bindings):
            raise ManagerError("binding result has the wrong type")
        if any(not isinstance(item, CopyIntent) for item in observed_copies):
            raise ManagerError("completed copy result has the wrong type")

        selected_bindings = _expected_bindings(
            source_step, step, arenas_by_class
        )
        expected_results = tuple(
            BindingResult(
                item.page, item.backend_domain, item.backend_index, True, True
            )
            for item in selected_bindings
        )
        if observed_bindings != expected_results:
            raise ManagerError(
                "binding results do not exactly match selected pages"
            )
        expected_copies = tuple(
            intent for spec in step.class_specs for intent in spec.copy_intents
        )
        if observed_copies != expected_copies:
            raise ManagerError(
                "completed copy results do not exactly match copy intents"
            )
        binds = tuple(
            EngineBindEvidence(
                item.page, item.backend_domain, True, True, item.backend_index
            )
            for item in selected_bindings
        )
        copies = tuple(
            EngineCopyEvidence(
                intent.class_id,
                intent.backend_domain,
                intent.token_count,
                intent.source_token_offset,
                intent.destination_token_offset,
                True,
                True,
                True,
                intent.source,
                intent.destination,
                intent.source_backend_index,
                intent.destination_backend_index,
            )
            for intent in observed_copies
        )
        evidence_steps.append(
            EngineStepExecutionEvidence(step.request_id, binds, copies)
        )
    return ExecutionEvidence(batch_plan.batch_id, tuple(evidence_steps))


__all__ = [
    "BatchPlan",
    "BindingResult",
    "BindingSpec",
    "ClassSpec",
    "StepExecutionResult",
    "StepPlan",
    "confirm_execution",
    "expected_bindings",
    "lower_batch_plan",
]
