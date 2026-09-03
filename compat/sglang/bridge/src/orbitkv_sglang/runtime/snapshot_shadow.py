from __future__ import annotations

from dataclasses import dataclass

from .identity import PageLease


CLASS_LOWERING_RESETTABLE = 2
CLASS_LOWERING_EPOCH_START = 4


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
    previous_layout_boundary: int = 0
    target_layout_boundary: int = 0


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


def sglang_page_id(backend_index: int, backend_base_index: int) -> int:
    page_id = backend_index - backend_base_index + 1
    if page_id <= 0:
        raise ValueError(
            "native backend index cannot lower into the SGLang arena"
        )
    return page_id


__all__ = [
    "CLASS_LOWERING_EPOCH_START",
    "CLASS_LOWERING_RESETTABLE",
    "ClassLowering",
    "CopyIntent",
    "SnapshotPage",
    "TailAction",
    "WriteIntent",
    "sglang_page_id",
]
