from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .completion import BatchCompletionReceipt
from .identity import PageLease, RelocationLease, RequestLease, SnapshotLease
from .reclamation import ReclamationCertificate
from .snapshot_shadow import RequestView


class TokenDispositionKind(IntEnum):
    RETAINED = 0
    SEMANTICALLY_DEAD = 1
    POLICY_EVICTED = 2


@dataclass(frozen=True, slots=True)
class TokenDisposition:
    kind: TokenDispositionKind
    policy_or_proof_id: int = 0
    version: int = 0
    quality_contract: int = 0


@dataclass(frozen=True, slots=True)
class TokenLocation:
    page: PageLease
    backend_index: int
    offset: int


@dataclass(frozen=True, slots=True)
class TokenPlacement:
    token_id: int
    disposition: TokenDisposition
    location: TokenLocation | None


@dataclass(frozen=True, slots=True)
class TokenViewQuery:
    request: RequestLease
    expected_snapshot: SnapshotLease
    class_id: int
    expected_boundary: int


@dataclass(frozen=True, slots=True)
class TokenView:
    class_id: int
    view_version: int
    page_tokens: int
    placements: tuple[TokenPlacement, ...]


@dataclass(frozen=True, slots=True)
class ClassTokenDispositionUpdate:
    class_id: int
    token_id: int
    disposition: TokenDisposition


@dataclass(frozen=True, slots=True)
class TokenDispositionBatchItem:
    request: RequestLease
    expected_snapshot: SnapshotLease
    updates: tuple[ClassTokenDispositionUpdate, ...]


@dataclass(frozen=True, slots=True)
class RelocationPolicy:
    maximum_source_pages: int = 32
    evacuation_headroom_pages: int = 8
    fragmentation_threshold_milli: int = 250
    full_evacuation: bool = True


@dataclass(frozen=True, slots=True)
class PrepareRelocationItem:
    request: RequestLease
    expected_snapshot: SnapshotLease
    class_id: int
    policy: RelocationPolicy


@dataclass(frozen=True, slots=True)
class TokenMove:
    token_id: int
    source: TokenLocation
    destination: TokenLocation


@dataclass(frozen=True, slots=True)
class PreparedRelocation:
    relocation: RelocationLease
    request: RequestLease
    base_snapshot: SnapshotLease
    target_snapshot: SnapshotLease
    base_view_version: int
    target_view_version: int
    source_pages: tuple[PageLease, ...]
    destination_pages: tuple[PageLease, ...]
    moves: tuple[TokenMove, ...]
    projected_reclaimed_pages: int
    fragmentation_milli: int
    class_id: int


@dataclass(frozen=True, slots=True)
class RelocationCopyReceipt:
    relocation: RelocationLease
    token_id: int
    source: TokenLocation
    destination: TokenLocation
    observed: int = 1
    copied: int = 1
    reserved16: int = 0
    reserved32: int = 0


@dataclass(frozen=True, slots=True)
class SubmittedRelocation:
    relocation: RelocationLease
    request: RequestLease
    target_snapshot: SnapshotLease


@dataclass(frozen=True, slots=True)
class RelocationUnobservedReceipt:
    relocation: RelocationLease
    backend_unobserved: int = 1
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class CompletedRelocationBatch:
    publications: tuple[RequestView, ...]
    retirements: tuple[ReclamationCertificate, ...]


def relocation_copy_receipts(
    prepared: PreparedRelocation,
) -> tuple[RelocationCopyReceipt, ...]:
    return tuple(
        RelocationCopyReceipt(
            relocation=prepared.relocation,
            token_id=movement.token_id,
            source=movement.source,
            destination=movement.destination,
        )
        for movement in prepared.moves
    )


__all__ = [
    "BatchCompletionReceipt",
    "ClassTokenDispositionUpdate",
    "CompletedRelocationBatch",
    "PrepareRelocationItem",
    "PreparedRelocation",
    "RelocationCopyReceipt",
    "RelocationPolicy",
    "RelocationUnobservedReceipt",
    "SubmittedRelocation",
    "TokenDisposition",
    "TokenDispositionBatchItem",
    "TokenDispositionKind",
    "TokenLocation",
    "TokenMove",
    "TokenPlacement",
    "TokenView",
    "TokenViewQuery",
    "relocation_copy_receipts",
]
