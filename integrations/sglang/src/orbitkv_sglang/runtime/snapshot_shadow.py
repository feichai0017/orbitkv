from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .identity import (
    TAIL_COPY_ON_WRITE,
    TAIL_FRESH,
    TAIL_IN_PLACE,
    TAIL_NONE,
    ArenaIdentity,
    ManagerError,
    PageLease,
    PrefixLease,
    PrefixSemanticKey,
    RequestLease,
    SnapshotLease,
    StepLease,
)


@dataclass(frozen=True, slots=True)
class RequestView:
    request: RequestLease
    snapshot: SnapshotLease
    view_version: int
    boundary: int
    resident_count: int


@dataclass(frozen=True, slots=True)
class SnapshotPage:
    page: PageLease
    logical_ordinal: int
    temporal_cell_index: int
    temporal_cycle: int
    backend_index: int
    class_id: int
    backend_domain: int
    valid_token_count: int
    visible_token_offset: int
    visible_token_count: int


@dataclass(frozen=True, slots=True)
class MaterializedRequestView:
    view: RequestView
    pages: tuple[SnapshotPage, ...]


@dataclass(frozen=True, slots=True)
class RequestForkItem:
    source_request: RequestLease
    expected_source_head: SnapshotLease
    target_empty_request: RequestLease
    expected_target_head: SnapshotLease


@dataclass(frozen=True, slots=True)
class ForkedRequest:
    source: RequestLease
    target: MaterializedRequestView


@dataclass(frozen=True, slots=True)
class PrepareBatchItem:
    request: RequestLease
    expected_head: SnapshotLease
    target_boundary: int


@dataclass(frozen=True, slots=True)
class WriteIntent:
    page_generation: int
    page_id: int
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class ClassLowering:
    class_id: int
    flags: int
    tail_offset: int
    tail_count: int
    copy_offset: int
    copy_count: int
    write_offset: int
    write_count: int
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class TailAction:
    class_id: int
    kind: int
    valid_token_count: int
    logical_ordinal: int
    source: PageLease
    destination: PageLease
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class CopyIntent:
    class_id: int
    backend_domain: int
    token_count: int
    source_token_offset: int
    destination_token_offset: int
    source: PageLease
    destination: PageLease
    source_backend_index: int
    destination_backend_index: int
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class PreparedStep:
    step: StepLease
    request: RequestLease
    base_snapshot: SnapshotLease
    target_snapshot: SnapshotLease
    base_view_version: int
    target_view_version: int
    previous_boundary: int
    target_boundary: int
    class_lowerings: tuple[ClassLowering, ...]
    tail_actions: tuple[TailAction, ...]
    copy_intents: tuple[CopyIntent, ...]
    write_intents: tuple[WriteIntent, ...]


@dataclass(frozen=True, slots=True)
class PrefixLookupHint:
    key: PrefixSemanticKey
    candidate: PrefixLease | None
    resident_count: int


@dataclass(frozen=True, slots=True)
class PrefixAttachItem:
    request: RequestLease
    expected_empty_head: SnapshotLease
    hint: PrefixLookupHint


@dataclass(frozen=True, slots=True)
class AttachedPrefix:
    prefix: PrefixLease
    target: MaterializedRequestView


@dataclass(frozen=True, slots=True)
class PrefixPublishItem:
    request: RequestLease
    expected_head: SnapshotLease
    key: PrefixSemanticKey


@dataclass(frozen=True, slots=True)
class PublishedPrefix:
    prefix: PrefixLease
    key: PrefixSemanticKey
    resident_count: int


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
    snapshot: SnapshotLease
    view_version: int = 0
    boundary: int = 0
    pages: dict[tuple[int, int], PageShadow] = field(default_factory=dict)

    @classmethod
    def from_view(cls, view: RequestView) -> RequestCursor:
        return cls(
            lease=view.request,
            snapshot=view.snapshot,
            view_version=view.view_version,
            boundary=view.boundary,
        )


@dataclass(frozen=True, slots=True)
class ClassLoweringSpec:
    class_id: int
    pool_id: int
    last_location: int
    exact_new_pages: tuple[int, ...]
    tail_action: TailAction
    copy_intents: tuple[CopyIntent, ...]


@dataclass(frozen=True, slots=True)
class LoweringPlan:
    request: RequestLease
    base_snapshot: SnapshotLease
    target_snapshot: SnapshotLease
    previous_boundary: int
    target_boundary: int
    class_specs: tuple[ClassLoweringSpec, ...]

    @property
    def by_class(self) -> dict[int, ClassLoweringSpec]:
        return {item.class_id: item for item in self.class_specs}


_ZERO_PAGE = PageLease(0, 0, 0, 0, 0)


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


def _backend_index(page: PageLease, arena: ArenaIdentity) -> int:
    if page.engine_epoch != arena.engine_epoch or page.pool_epoch != arena.pool_epoch:
        raise ManagerError("page lease belongs to another arena generation")
    if page.pool_id != arena.pool_id:
        raise ManagerError("page lease belongs to another class arena")
    if not arena.first_page_id <= page.page_id < arena.first_page_id + arena.page_count:
        raise ManagerError("page lease is outside its class arena")
    if page.generation <= 0:
        raise ManagerError("page lease generation is invalid")
    return arena.backend_base_index + page.page_id - arena.first_page_id


def _shadow(
    request: RequestLease,
    class_id: int,
    logical_ordinal: int,
    page: PageLease,
    arena: ArenaIdentity,
) -> PageShadow:
    return PageShadow(
        request=request,
        class_id=class_id,
        logical_ordinal=logical_ordinal,
        page=page,
        backend_index=_backend_index(page, arena),
    )


def page_shadow_from_snapshot(
    request: RequestLease, page: SnapshotPage, arena: ArenaIdentity
) -> PageShadow:
    if page.class_id != arena.class_id or page.backend_domain != arena.backend_domain:
        raise ManagerError("snapshot page names the wrong class or backend domain")
    if page.backend_index != _backend_index(page.page, arena):
        raise ManagerError("snapshot page backend index disagrees with its lease")
    return PageShadow(
        request=request,
        class_id=page.class_id,
        logical_ordinal=page.logical_ordinal,
        page=page.page,
        backend_index=page.backend_index,
    )


def _decode_prepared(
    cursor: RequestCursor,
    prepared: PreparedStep,
    arenas: Mapping[int, ArenaIdentity],
    config: Any,
) -> tuple[LoweringPlan, tuple[PageShadow, ...]]:
    classes = tuple(config.classes)
    if len(prepared.class_lowerings) != len(classes):
        raise ManagerError("prepare returned the wrong class cardinality")
    if prepared.base_snapshot != cursor.snapshot:
        raise ManagerError("prepare base snapshot differs from the request head")
    if prepared.target_snapshot == prepared.base_snapshot:
        raise ManagerError("prepare did not allocate a distinct target snapshot")

    expected_writes = expected_new_ordinals(
        prepared.previous_boundary, prepared.target_boundary, int(config.page_tokens)
    )
    class_specs: list[ClassLoweringSpec] = []
    new_pages: list[PageShadow] = []
    physical: set[tuple[int, int]] = set()
    tail_cursor = copy_cursor = write_cursor = 0

    for class_config, lowering in zip(classes, prepared.class_lowerings, strict=True):
        class_page_start = len(new_pages)
        arena = arenas[class_config.class_id]
        if lowering.class_id != class_config.class_id:
            raise ManagerError("prepare class lowerings are not in compiled order")
        if lowering.flags != 0 or lowering.reserved != 0:
            raise ManagerError("prepare returned nonzero class reserved fields")
        if (
            lowering.tail_offset != tail_cursor
            or lowering.copy_offset != copy_cursor
            or lowering.write_offset != write_cursor
            or lowering.tail_count != 1
            or lowering.copy_count > 1
        ):
            raise ManagerError("prepare class spans are not canonical")
        tail_end = tail_cursor + lowering.tail_count
        copy_end = copy_cursor + lowering.copy_count
        write_end = write_cursor + lowering.write_count
        if (
            tail_end > len(prepared.tail_actions)
            or copy_end > len(prepared.copy_intents)
            or write_end > len(prepared.write_intents)
        ):
            raise ManagerError("prepare class span is out of range")
        if lowering.write_count != len(expected_writes):
            raise ManagerError("prepare reserved the wrong number of class pages")

        action = prepared.tail_actions[tail_cursor]
        copies = prepared.copy_intents[copy_cursor:copy_end]
        if action.class_id != lowering.class_id or action.reserved != 0:
            raise ManagerError("tail action does not belong to its class")
        partial = prepared.previous_boundary % arena.page_tokens
        ordinal = prepared.previous_boundary // arena.page_tokens if partial else 0
        expected_tail = cursor.pages.get((lowering.class_id, ordinal)) if partial else None
        last_location = -1
        if not partial:
            if (
                action.kind != TAIL_NONE
                or action.valid_token_count != 0
                or action.logical_ordinal != 0
                or action.source != _ZERO_PAGE
                or action.destination != _ZERO_PAGE
                or copies
            ):
                raise ManagerError("aligned append returned a nonempty tail action")
        elif action.logical_ordinal != ordinal:
            raise ManagerError("tail action names the wrong logical ordinal")
        elif action.kind == TAIL_IN_PLACE:
            if (
                action.valid_token_count != partial
                or action.source != action.destination
                or expected_tail is None
                or action.source != expected_tail.page
                or not action.source.generation
                or copies
            ):
                raise ManagerError("in-place tail action is inconsistent")
            backend_index = _backend_index(action.destination, arena)
            last_location = (
                sglang_page_id(backend_index, arena.backend_base_index)
                * arena.page_tokens
                + partial
                - 1
            )
        elif action.kind == TAIL_COPY_ON_WRITE:
            if (
                action.valid_token_count != partial
                or expected_tail is None
                or action.source != expected_tail.page
                or action.source == action.destination
                or len(copies) != 1
            ):
                raise ManagerError("copy-on-write tail action is inconsistent")
            copy = copies[0]
            if (
                copy.class_id != action.class_id
                or copy.backend_domain != arena.backend_domain
                or copy.token_count != partial
                or copy.source_token_offset != 0
                or copy.destination_token_offset != 0
                or copy.reserved != 0
                or copy.source != action.source
                or copy.destination != action.destination
                or copy.source_backend_index != _backend_index(copy.source, arena)
                or copy.destination_backend_index
                != _backend_index(copy.destination, arena)
            ):
                raise ManagerError("copy intent is not an exact tail echo")
            shadow = _shadow(
                prepared.request, action.class_id, ordinal, action.destination, arena
            )
            new_pages.append(shadow)
            last_location = (
                sglang_page_id(shadow.backend_index, arena.backend_base_index)
                * arena.page_tokens
                + partial
                - 1
            )
        elif action.kind == TAIL_FRESH:
            if (
                action.valid_token_count != 0
                or expected_tail is not None
                or action.source != _ZERO_PAGE
                or not action.destination.generation
                or copies
            ):
                raise ManagerError("fresh tail action is inconsistent")
            shadow = _shadow(
                prepared.request, action.class_id, ordinal, action.destination, arena
            )
            new_pages.append(shadow)
            last_location = (
                sglang_page_id(shadow.backend_index, arena.backend_base_index)
                * arena.page_tokens
                + partial
                - 1
            )
        else:
            raise ManagerError("prepare returned an unknown tail action")

        exact_new_pages: list[int] = []
        for logical_ordinal, intent in zip(
            expected_writes,
            prepared.write_intents[write_cursor:write_end],
            strict=True,
        ):
            if intent.reserved != 0:
                raise ManagerError("write intent reserved field is nonzero")
            page = PageLease(
                prepared.request.engine_epoch,
                arena.pool_epoch,
                intent.page_generation,
                intent.page_id,
                arena.pool_id,
            )
            shadow = _shadow(
                prepared.request, lowering.class_id, logical_ordinal, page, arena
            )
            new_pages.append(shadow)
            exact_new_pages.append(
                sglang_page_id(shadow.backend_index, arena.backend_base_index)
            )

        for shadow in new_pages[class_page_start:]:
            key = (shadow.page.pool_id, shadow.page.page_id)
            if key in physical:
                raise ManagerError("prepare aliases a physical page within the request")
            physical.add(key)
        class_specs.append(
            ClassLoweringSpec(
                class_id=lowering.class_id,
                pool_id=arena.pool_id,
                last_location=last_location,
                exact_new_pages=tuple(exact_new_pages),
                tail_action=action,
                copy_intents=copies,
            )
        )
        tail_cursor, copy_cursor, write_cursor = tail_end, copy_end, write_end

    if (
        tail_cursor != len(prepared.tail_actions)
        or copy_cursor != len(prepared.copy_intents)
        or write_cursor != len(prepared.write_intents)
    ):
        raise ManagerError("prepare class spans do not cover the flat outputs")
    return (
        LoweringPlan(
            request=prepared.request,
            base_snapshot=prepared.base_snapshot,
            target_snapshot=prepared.target_snapshot,
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


__all__ = [
    "AttachedPrefix",
    "ClassLowering",
    "ClassLoweringSpec",
    "CopyIntent",
    "ForkedRequest",
    "LoweringPlan",
    "MaterializedRequestView",
    "PageShadow",
    "PrefixAttachItem",
    "PrefixLookupHint",
    "PrefixPublishItem",
    "PrepareBatchItem",
    "PreparedStep",
    "PublishedPrefix",
    "RequestCursor",
    "RequestForkItem",
    "RequestView",
    "SnapshotPage",
    "TailAction",
    "WriteIntent",
    "expected_new_ordinals",
    "lowering_plan",
    "page_shadow_from_snapshot",
    "sglang_page_id",
]
