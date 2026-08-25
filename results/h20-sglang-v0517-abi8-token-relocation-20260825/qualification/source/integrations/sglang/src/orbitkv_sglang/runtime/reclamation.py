from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from .identity import (
    PageLease,
    PrefixLease,
    PrefixSemanticKey,
    ReclamationLease,
    RequestLease,
    SnapshotLease,
)
from .snapshot_shadow import PublishedPrefix


@dataclass(frozen=True, slots=True)
class DetachedBinding:
    old: PageLease
    replacement: PageLease
    logical_ordinal: int
    old_backend_index: int
    replacement_backend_index: int
    token_begin: int
    token_end_exclusive: int
    class_id: int
    backend_domain: int
    action: int
    reason: int
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class ReclamationCertificate:
    reclamation: ReclamationLease
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
    acknowledged: int = 1
    reserved8: int = 0
    reserved32: int = 0
    backend_index: int = 0


@dataclass(frozen=True, slots=True)
class ReleaseBatchItem:
    request: RequestLease
    expected_head: SnapshotLease


@dataclass(frozen=True, slots=True)
class ReleaseCompletion:
    request: RequestLease
    detached_snapshot: SnapshotLease
    detached: tuple[DetachedBinding, ...]


@dataclass(frozen=True, slots=True)
class ReleaseBatchCompletion:
    releases: tuple[ReleaseCompletion, ...]
    retirements: tuple[ReclamationCertificate, ...]


@dataclass(frozen=True, slots=True)
class PrefixPublishRelease:
    publication: PublishedPrefix
    release: ReleaseCompletion


@dataclass(frozen=True, slots=True)
class PrefixPublishReleaseBatch:
    outputs: tuple[PrefixPublishRelease, ...]
    retirements: tuple[ReclamationCertificate, ...]


@dataclass(frozen=True, slots=True)
class EvictedPrefix:
    prefix: PrefixLease
    key: PrefixSemanticKey


@dataclass(frozen=True, slots=True)
class PrefixEvictionBatch:
    evicted: tuple[EvictedPrefix, ...]
    retirements: tuple[ReclamationCertificate, ...]


@dataclass(frozen=True, slots=True)
class MirrorCandidateTransition:
    """Exact candidate state already installed in the backend mirror.

    ``source`` is the zero lease for a fresh page.  A nonzero source records a
    COW transition; the copied token span is then exact and the destination
    span describes every token location installed for this step.  ``retiring``
    says that the destination never became resident and must be cleaned before
    its reclamation certificate is acknowledged.
    """

    destination: PageLease
    source: PageLease
    logical_ordinal: int
    destination_backend_index: int
    source_backend_index: int
    token_begin: int
    token_end_exclusive: int
    copied_token_begin: int
    copied_token_end_exclusive: int
    class_id: int
    backend_domain: int
    retiring: bool
    reserved: int = 0


@dataclass(frozen=True, slots=True)
class MirrorCleanupItem:
    context: Any
    detached: tuple[DetachedBinding, ...]
    releasing: bool
    boundary: int
    candidates: tuple[MirrorCandidateTransition, ...] = ()


@runtime_checkable
class MirrorCleanupProtocol(Protocol):
    def preflight(
        self,
        items: Sequence[MirrorCleanupItem],
        retirements: Sequence[ReclamationCertificate],
    ) -> Any: ...

    def commit(self, plan: Any) -> None: ...

    def synchronize(self, plan: Any) -> None: ...

    def finalize(self, plan: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class MirrorCleanupBinding:
    coordinator: MirrorCleanupProtocol
    context: Any


def reclamation_receipts(
    certificates: Sequence[ReclamationCertificate],
) -> tuple[ReclamationReceipt, ...]:
    return tuple(
        ReclamationReceipt(
            reclamation=certificate.reclamation,
            page=certificate.page,
            backend_domain=certificate.backend_domain,
            backend_index=certificate.backend_index,
        )
        for certificate in certificates
    )


__all__ = [
    "DetachedBinding",
    "EvictedPrefix",
    "MirrorCleanupBinding",
    "MirrorCandidateTransition",
    "MirrorCleanupItem",
    "MirrorCleanupProtocol",
    "PrefixEvictionBatch",
    "PrefixPublishRelease",
    "PrefixPublishReleaseBatch",
    "ReclamationCertificate",
    "ReclamationReceipt",
    "ReleaseBatchCompletion",
    "ReleaseBatchItem",
    "ReleaseCompletion",
    "reclamation_receipts",
]
