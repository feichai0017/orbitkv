from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from .completion import (
        BackendBindReceipt,
        BackendCopyReceipt,
        BackendUnobservedReceipt,
        BatchCompletionReceipt,
        CompletionBatch,
        SubmittedStep,
    )
    from .reclamation import (
        PrefixEvictionBatch,
        PrefixPublishReleaseBatch,
        ReclamationReceipt,
        ReleaseBatchCompletion,
        ReleaseBatchItem,
    )
    from .snapshot_shadow import (
        AttachedPrefix,
        ForkedRequest,
        PrefixAttachItem,
        PrefixLookupHint,
        PrefixPublishItem,
        PrepareBatchItem,
        PreparedStep,
        PublishedPrefix,
        RequestForkItem,
        RequestView,
    )


TAIL_NONE = 0
TAIL_IN_PLACE = 1
TAIL_COPY_ON_WRITE = 2
TAIL_FRESH = 3

DETACHED_CLEAR = 1
DETACHED_REPLACE = 2
DETACHED_RETENTION = 1
DETACHED_COPY_ON_WRITE = 2
DETACHED_REQUEST_RELEASE = 3
DETACHED_PREFIX_TRANSFER = 4


class ManagerError(RuntimeError):
    """The canonical manager rejected an operation or returned invalid data."""


class RetryableConflict(ManagerError):
    """A proven precommit conflict left the whole manager batch unchanged."""


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
    request_page_refs: int
    prefix_page_refs: int
    reader_pins: int


@dataclass(frozen=True, slots=True)
class ManagerCreateSettings:
    maximum_requests: int
    maximum_operations: int
    maximum_prefixes: int
    maximum_reclamations: int
    maximum_step_tokens: int


@dataclass(frozen=True, slots=True)
class RequestLease:
    engine_epoch: int
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class SnapshotLease:
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
class PrefixLease:
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
class PrefixSemanticKey:
    namespace: bytes
    digest: bytes
    boundary: int


@dataclass(frozen=True, slots=True)
class ManagerStats:
    active_requests: int
    active_snapshots: int
    active_prefixes: int
    evicted_prefixes: int
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
    total_request_page_refs: int
    total_prefix_page_refs: int
    total_reader_pins: int


@runtime_checkable
class ManagerProtocol(Protocol):
    @property
    def arenas(self) -> tuple[ArenaIdentity, ...]: ...

    @property
    def arenas_by_class(self) -> dict[int, ArenaIdentity]: ...

    @property
    def performance_counters(self) -> dict[str, int]: ...

    def arena_stats(self) -> tuple[ArenaStats, ...]: ...

    def request_acquire_batch(self, request_count: int) -> tuple[RequestView, ...]: ...

    def request_fork_batch(
        self, items: Sequence[RequestForkItem]
    ) -> tuple[ForkedRequest, ...]: ...

    def prepare_batch(
        self, items: Sequence[PrepareBatchItem]
    ) -> tuple[PreparedStep, ...]: ...

    def submit_batch(
        self,
        items: Sequence[
            tuple[
                StepLease,
                Sequence[BackendBindReceipt],
                Sequence[BackendCopyReceipt],
            ]
        ],
    ) -> tuple[SubmittedStep, ...]: ...

    def complete_batch(
        self,
        receipt: BatchCompletionReceipt,
        submissions: Sequence[SubmissionLease],
    ) -> CompletionBatch: ...

    def abort_steps_batch(
        self, receipts: Sequence[BackendUnobservedReceipt]
    ) -> None: ...

    def quarantine_steps_batch(self, steps: Sequence[StepLease]) -> None: ...

    def quarantine_submissions_batch(
        self, submissions: Sequence[SubmissionLease]
    ) -> None: ...

    def release_batch(
        self, items: Sequence[ReleaseBatchItem]
    ) -> ReleaseBatchCompletion: ...

    def acknowledge_reclamations_batch(
        self, receipts: Sequence[ReclamationReceipt]
    ) -> None: ...

    def recycle_requests_batch(self, requests: Sequence[RequestLease]) -> None: ...

    def prefix_lookup_batch(
        self, keys: Sequence[PrefixSemanticKey]
    ) -> tuple[PrefixLookupHint, ...]: ...

    def prefix_attach_batch(
        self, items: Sequence[PrefixAttachItem]
    ) -> tuple[AttachedPrefix, ...]: ...

    def prefix_publish_batch(
        self, items: Sequence[PrefixPublishItem]
    ) -> tuple[PublishedPrefix, ...]: ...

    def prefix_publish_release_batch(
        self, items: Sequence[PrefixPublishItem]
    ) -> PrefixPublishReleaseBatch: ...

    def prefix_evict_batch(
        self, prefixes: Sequence[PrefixLease]
    ) -> PrefixEvictionBatch: ...

    def prefix_recycle_batch(self, prefixes: Sequence[PrefixLease]) -> None: ...

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


__all__ = [
    "ArenaIdentity",
    "ArenaRegistration",
    "ArenaStats",
    "FailStopped",
    "ManagerCreateSettings",
    "ManagerError",
    "ManagerFactoryProtocol",
    "ManagerProtocol",
    "ManagerStats",
    "PageLease",
    "PrefixLease",
    "PrefixSemanticKey",
    "ReclamationLease",
    "RequestLease",
    "RetryableConflict",
    "SnapshotLease",
    "StepLease",
    "SubmissionLease",
]
