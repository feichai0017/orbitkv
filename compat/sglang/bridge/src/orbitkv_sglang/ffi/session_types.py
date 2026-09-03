from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, NewType, Sequence, TypeVar

from orbitkv_sglang.runtime import (
    ClassLowering,
    CopyIntent,
    DetachedBinding,
    ManagerError,
    PageLease,
    PrefixSemanticKey,
    SnapshotPage,
    TailAction,
    WriteIntent,
)
from orbitkv_sglang.runtime.token_state import (
    RelocationPolicy,
    TokenDisposition,
    TokenLocation,
    TokenMove,
    TokenPlacement,
)

from . import layouts as L
from .codec import page as _page
from .codec import page_to_c as _page_to_c
from .codec import uint as _uint
from .library import (
    SESSION_PENDING_ATTACH_CANCEL_FINALIZED,
    SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING,
    SESSION_CONTROL_KIND_MATERIALIZATION,
    SESSION_CONTROL_KIND_PREFIX_EVICTION,
    SESSION_CONTROL_OUTCOME_EVICTED,
    SESSION_CONTROL_OUTCOME_MATERIALIZED,
    SESSION_RELEASE_COMPLETED,
    SESSION_RELEASE_RECYCLE_PENDING,
)


EngineRequestId = NewType("EngineRequestId", int)


@dataclass(frozen=True, slots=True)
class EngineBatchId:
    """Opaque batch identity minted by one native runtime session."""

    session_epoch: int
    sequence: int


@dataclass(frozen=True, slots=True)
class EnginePublicationId:
    """Opaque publication identity minted by one native runtime session."""

    session_epoch: int
    sequence: int


@dataclass(frozen=True, slots=True)
class EngineReleaseId:
    """Opaque release identity minted by one native runtime session."""

    session_epoch: int
    sequence: int


@dataclass(frozen=True, slots=True)
class EnginePrefixId:
    """Opaque prefix identity minted by one native runtime session."""

    session_epoch: int
    sequence: int


@dataclass(frozen=True, slots=True)
class EngineControlId:
    """Opaque control identity minted by one native runtime session."""

    session_epoch: int
    sequence: int


@dataclass(frozen=True, slots=True)
class EngineRelocationId:
    """Opaque relocation identity minted by one native runtime session."""

    session_epoch: int
    sequence: int


@dataclass(frozen=True, slots=True)
class EngineRequestView:
    request_id: EngineRequestId
    view_version: int
    boundary: int
    resident_count: int


@dataclass(frozen=True, slots=True)
class EngineTokenViewQuery:
    """Capability-free query against a session-owned current head."""

    request_id: EngineRequestId
    class_id: int
    expected_boundary: int


@dataclass(frozen=True, slots=True)
class EngineTokenView:
    request_id: EngineRequestId
    class_id: int
    view_version: int
    page_tokens: int
    placements: tuple[TokenPlacement, ...]


@dataclass(frozen=True, slots=True)
class EngineTokenDispositionUpdate:
    class_id: int
    token_id: int
    disposition: TokenDisposition


@dataclass(frozen=True, slots=True)
class EngineTokenDispositionBatchItem:
    request_id: EngineRequestId
    updates: tuple[EngineTokenDispositionUpdate, ...]


@dataclass(frozen=True, slots=True)
class EnginePrepareRelocationItem:
    request_id: EngineRequestId
    class_id: int
    policy: RelocationPolicy


@dataclass(frozen=True, slots=True)
class EngineRelocationPlan:
    request_id: EngineRequestId
    class_id: int
    base_view_version: int
    target_view_version: int
    fragmentation_milli: int
    source_pages: tuple[PageLease, ...]
    destination_pages: tuple[PageLease, ...]
    moves: tuple[TokenMove, ...]
    projected_reclaimed_pages: int


@dataclass(frozen=True, slots=True)
class EnginePreparedRelocation:
    relocation_id: EngineRelocationId
    plans: tuple[EngineRelocationPlan, ...]


@dataclass(frozen=True, slots=True)
class EngineRelocationCopyEvidence:
    token_id: int
    source: TokenLocation
    destination: TokenLocation
    observed: bool
    copied: bool


@dataclass(frozen=True, slots=True)
class EngineRelocationRequestEvidence:
    request_id: EngineRequestId
    copies: tuple[EngineRelocationCopyEvidence, ...]


@dataclass(frozen=True, slots=True)
class EngineRelocationExecutionEvidence:
    relocation_id: EngineRelocationId
    requests: tuple[EngineRelocationRequestEvidence, ...]


@dataclass(frozen=True, slots=True)
class EngineRelocationAbortEvidence:
    request_id: EngineRequestId
    backend_unobserved: bool


@dataclass(frozen=True, slots=True)
class EngineRelocationTicket:
    relocation_id: EngineRelocationId


@dataclass(frozen=True, slots=True)
class EngineRelocationRequestPublication:
    request_id: EngineRequestId
    view_version: int
    boundary: int
    resident_count: int


@dataclass(frozen=True, slots=True)
class EngineRelocationPublication:
    relocation_id: EngineRelocationId
    requests: tuple[EngineRelocationRequestPublication, ...]
    retirements: tuple[EngineRetirement, ...]


@dataclass(frozen=True, slots=True)
class EngineRelocationPublicationEvidence:
    relocation_id: EngineRelocationId
    mirror_cleanup_confirmed: bool
    reclamation_receipts: tuple[EngineRetirementEvidence, ...]


@dataclass(frozen=True, slots=True)
class EngineAppendIntent:
    request_id: EngineRequestId
    target_boundary: int


@dataclass(frozen=True, slots=True)
class EngineStepPlan:
    request_id: EngineRequestId
    base_view_version: int
    target_view_version: int
    previous_boundary: int
    target_boundary: int
    class_lowerings: tuple[ClassLowering, ...]
    tail_actions: tuple[TailAction, ...]
    copy_intents: tuple[CopyIntent, ...]
    write_intents: tuple[WriteIntent, ...]


@dataclass(frozen=True, slots=True)
class EngineBatchPlan:
    batch_id: EngineBatchId
    steps: tuple[EngineStepPlan, ...]


@dataclass(frozen=True, slots=True)
class EngineBindEvidence:
    page: PageLease
    backend_domain: int
    mapped: bool
    writable: bool
    backend_index: int


@dataclass(frozen=True, slots=True)
class EngineCopyEvidence:
    class_id: int
    backend_domain: int
    token_count: int
    source_token_offset: int
    destination_token_offset: int
    observed: bool
    copied: bool
    ordered_before_writes: bool
    source: PageLease
    destination: PageLease
    source_backend_index: int
    destination_backend_index: int


@dataclass(frozen=True, slots=True)
class EngineStepExecutionEvidence:
    request_id: EngineRequestId
    bind_receipts: tuple[EngineBindEvidence, ...]
    copy_receipts: tuple[EngineCopyEvidence, ...]


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    batch_id: EngineBatchId
    steps: tuple[EngineStepExecutionEvidence, ...]


@dataclass(frozen=True, slots=True)
class EngineStepAbortEvidence:
    request_id: EngineRequestId
    backend_unobserved: bool


@dataclass(frozen=True, slots=True)
class EngineBatchTicket:
    batch_id: EngineBatchId
    requests: tuple[EngineRequestId, ...]


@dataclass(frozen=True, slots=True)
class EngineCompletionEvidence:
    completion_domain: int
    completion_value: int
    confirmed: bool


@dataclass(frozen=True, slots=True)
class EngineRetirement:
    """Physical retirement fact with no manager reclamation capability."""

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
class EngineRetirementEvidence:
    page: PageLease
    backend_domain: int
    acknowledged: bool
    backend_index: int


@dataclass(frozen=True, slots=True)
class EngineStepPublication:
    request_id: EngineRequestId
    view_version: int
    boundary: int
    resident_count: int
    detached: tuple[DetachedBinding, ...]


@dataclass(frozen=True, slots=True)
class EngineBatchPublication:
    publication_id: EnginePublicationId
    batch_id: EngineBatchId
    steps: tuple[EngineStepPublication, ...]
    retirements: tuple[EngineRetirement, ...]


@dataclass(frozen=True, slots=True)
class EnginePublicationEvidence:
    publication_id: EnginePublicationId
    mirror_cleanup_confirmed: bool
    reclamation_receipts: tuple[EngineRetirementEvidence, ...]


@dataclass(frozen=True, slots=True)
class EngineReleasedRequest:
    request_id: EngineRequestId
    detached: tuple[DetachedBinding, ...]


@dataclass(frozen=True, slots=True)
class EngineReleasePlan:
    release_id: EngineReleaseId
    releases: tuple[EngineReleasedRequest, ...]
    retirements: tuple[EngineRetirement, ...]


@dataclass(frozen=True, slots=True)
class EngineReleaseEvidence:
    release_id: EngineReleaseId
    mirror_cleanup_confirmed: bool
    reclamation_receipts: tuple[EngineRetirementEvidence, ...]
    acknowledged_retry: bool = False


class EngineReleaseDisposition(IntEnum):
    """Native disposition after a release confirmation was accepted."""

    COMPLETED = SESSION_RELEASE_COMPLETED
    RECYCLE_PENDING = SESSION_RELEASE_RECYCLE_PENDING


@dataclass(frozen=True, slots=True)
class EngineReleaseOutcome:
    """Certain native result; only ``COMPLETED`` retires the release id."""

    release_id: EngineReleaseId
    disposition: EngineReleaseDisposition


@dataclass(frozen=True, slots=True)
class EnginePrefixLookup:
    key: PrefixSemanticKey
    candidate: EnginePrefixId | None
    resident_count: int


@dataclass(frozen=True, slots=True)
class EnginePrefixPublishItem:
    request_id: EngineRequestId
    key: PrefixSemanticKey


@dataclass(frozen=True, slots=True)
class EnginePublishedPrefix:
    prefix_id: EnginePrefixId
    key: PrefixSemanticKey
    resident_count: int


@dataclass(frozen=True, slots=True)
class EnginePublishedPrefixRelease:
    prefix_id: EnginePrefixId
    key: PrefixSemanticKey
    resident_count: int
    release: EngineReleasedRequest


@dataclass(frozen=True, slots=True)
class EnginePrefixPublishReleasePlan:
    release_id: EngineReleaseId
    outputs: tuple[EnginePublishedPrefixRelease, ...]


@dataclass(frozen=True, slots=True)
class EnginePrefixAttachItem:
    target_request_id: EngineRequestId
    prefix_id: EnginePrefixId
    key: PrefixSemanticKey
    resident_count: int


@dataclass(frozen=True, slots=True)
class EngineRequestForkItem:
    source_request_id: EngineRequestId
    target_request_id: EngineRequestId


class EngineControlKind(IntEnum):
    MATERIALIZATION = SESSION_CONTROL_KIND_MATERIALIZATION
    PREFIX_EVICTION = SESSION_CONTROL_KIND_PREFIX_EVICTION


@dataclass(frozen=True, slots=True)
class EngineControlPlanInfo:
    control_id: EngineControlId
    kind: EngineControlKind
    request_count: int
    page_count: int
    prefix_count: int
    retirement_count: int


@dataclass(frozen=True, slots=True)
class EngineMaterializedRequest:
    request_id: EngineRequestId
    view_version: int
    boundary: int
    resident_count: int
    pages: tuple[SnapshotPage, ...]


@dataclass(frozen=True, slots=True)
class EngineMaterializationPlan:
    control_id: EngineControlId
    requests: tuple[EngineMaterializedRequest, ...]


@dataclass(frozen=True, slots=True)
class EnginePendingAttachCancel:
    control_id: EngineControlId
    request_id: EngineRequestId
    prefix_id: EnginePrefixId
    view_version: int
    boundary: int
    resident_count: int


class EnginePendingAttachCancelDisposition(IntEnum):
    RECYCLE_PENDING = SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING
    FINALIZED = SESSION_PENDING_ATTACH_CANCEL_FINALIZED


@dataclass(frozen=True, slots=True)
class EnginePendingAttachCancelOutcome:
    control_id: EngineControlId
    request_id: EngineRequestId
    prefix_id: EnginePrefixId
    view_version: int
    boundary: int
    resident_count: int
    disposition: EnginePendingAttachCancelDisposition

    @property
    def identity(self) -> EnginePendingAttachCancel:
        return EnginePendingAttachCancel(
            self.control_id,
            self.request_id,
            self.prefix_id,
            self.view_version,
            self.boundary,
            self.resident_count,
        )


@dataclass(frozen=True, slots=True)
class EnginePrefixEvictionPlan:
    control_id: EngineControlId
    prefix_ids: tuple[EnginePrefixId, ...]
    retirements: tuple[EngineRetirement, ...]


@dataclass(frozen=True, slots=True)
class EngineControlEvidence:
    control_id: EngineControlId
    mirror_cleanup_confirmed: bool
    reclamation_receipts: tuple[EngineRetirementEvidence, ...]


class EngineControlDisposition(IntEnum):
    MATERIALIZED = SESSION_CONTROL_OUTCOME_MATERIALIZED
    EVICTED = SESSION_CONTROL_OUTCOME_EVICTED


@dataclass(frozen=True, slots=True)
class EngineControlOutcome:
    control_id: EngineControlId
    disposition: EngineControlDisposition


def retirement_evidence(
    retirements: Sequence[EngineRetirement],
) -> tuple[EngineRetirementEvidence, ...]:
    return tuple(
        EngineRetirementEvidence(
            item.page, item.backend_domain, True, item.backend_index
        )
        for item in retirements
    )


def _wire_bool(name: str, value: Any, bits: int) -> int:
    if not isinstance(value, bool):
        raise ManagerError(f"{name} must be a boolean")
    return _uint(name, int(value), bits)


_SessionIdT = TypeVar(
    "_SessionIdT",
    EngineBatchId,
    EnginePublicationId,
    EngineReleaseId,
    EnginePrefixId,
    EngineControlId,
    EngineRelocationId,
)


def _session_id_c(
    value: _SessionIdT,
    expected_type: type[_SessionIdT],
    layout: Any,
    label: str,
) -> Any:
    if type(value) is not expected_type:
        raise ManagerError(f"{label} must be an {expected_type.__name__}")
    epoch = _uint(f"{label} session epoch", value.session_epoch, 64)
    sequence = _uint(f"{label} sequence", value.sequence, 64)
    if epoch == 0 or sequence == 0:
        raise ManagerError(f"{label} fields must be nonzero")
    return layout(epoch, sequence)


_OperationIdT = TypeVar(
    "_OperationIdT",
    EngineBatchId,
    EnginePublicationId,
    EngineReleaseId,
    EngineRelocationId,
)


def _operation_id_c(
    value: _OperationIdT,
    expected_type: type[_OperationIdT],
    layout: Any,
    label: str,
) -> Any:
    return _session_id_c(value, expected_type, layout, label)


def _tail(value: Any) -> TailAction:
    if int(value.reserved) != 0:
        raise ManagerError("tail action reserved field is nonzero")
    return TailAction(
        int(value.class_id), int(value.kind), int(value.valid_token_count),
        int(value.logical_ordinal), _page(value.source), _page(value.destination), 0,
    )


def _copy_intent(value: Any) -> CopyIntent:
    if int(value.reserved) != 0:
        raise ManagerError("copy intent reserved field is nonzero")
    return CopyIntent(
        int(value.class_id), int(value.backend_domain), int(value.token_count),
        int(value.source_token_offset), int(value.destination_token_offset),
        _page(value.source), _page(value.destination),
        int(value.source_backend_index), int(value.destination_backend_index), 0,
    )


def _detached(value: Any) -> DetachedBinding:
    if int(value.reserved) != 0:
        raise ManagerError("detached binding reserved field is nonzero")
    return DetachedBinding(
        _page(value.old), _page(value.replacement), int(value.logical_ordinal),
        int(value.old_backend_index), int(value.replacement_backend_index),
        int(value.token_begin), int(value.token_end_exclusive), int(value.class_id),
        int(value.backend_domain), int(value.action), int(value.reason), 0,
    )


def _retirement(value: Any) -> EngineRetirement:
    if int(value.reserved32) != 0:
        raise ManagerError("session retirement reserved field is nonzero")
    return EngineRetirement(
        _page(value.page), int(value.class_id), int(value.backend_domain),
        int(value.logical_ordinal), int(value.backend_index), int(value.token_begin),
        int(value.token_end_exclusive), int(value.completion_domain),
        int(value.completion_value),
    )


def _retirement_evidence_c(
    value: EngineRetirementEvidence,
) -> L.SessionRetirementEvidenceLayout:
    if not isinstance(value, EngineRetirementEvidence):
        raise ManagerError("reclamation receipt must be EngineRetirementEvidence")
    return L.SessionRetirementEvidenceLayout(
        _page_to_c(value.page),
        _uint("retirement backend domain", value.backend_domain, 16),
        _wire_bool("retirement acknowledged", value.acknowledged, 8),
        0, 0, _uint("retirement backend index", value.backend_index, 64),
    )


__all__ = [
    "EngineAppendIntent",
    "EngineBatchId",
    "EngineBatchPlan",
    "EngineBatchPublication",
    "EngineBatchTicket",
    "EngineBindEvidence",
    "EngineCompletionEvidence",
    "EngineControlDisposition",
    "EngineControlEvidence",
    "EngineControlId",
    "EngineControlKind",
    "EngineControlOutcome",
    "EngineControlPlanInfo",
    "EngineCopyEvidence",
    "EngineMaterializedRequest",
    "EngineMaterializationPlan",
    "EnginePendingAttachCancel",
    "EnginePendingAttachCancelDisposition",
    "EnginePendingAttachCancelOutcome",
    "EnginePrepareRelocationItem",
    "EnginePublicationId",
    "EnginePublicationEvidence",
    "EnginePrefixAttachItem",
    "EnginePrefixEvictionPlan",
    "EnginePrefixId",
    "EnginePrefixLookup",
    "EnginePrefixPublishItem",
    "EnginePrefixPublishReleasePlan",
    "EnginePublishedPrefixRelease",
    "EnginePublishedPrefix",
    "EngineReleaseEvidence",
    "EngineReleaseId",
    "EngineReleaseDisposition",
    "EngineReleaseOutcome",
    "EngineReleasePlan",
    "EngineReleasedRequest",
    "EnginePreparedRelocation",
    "EngineRelocationAbortEvidence",
    "EngineRelocationCopyEvidence",
    "EngineRelocationExecutionEvidence",
    "EngineRelocationId",
    "EngineRelocationPlan",
    "EngineRelocationPublication",
    "EngineRelocationPublicationEvidence",
    "EngineRelocationRequestEvidence",
    "EngineRelocationRequestPublication",
    "EngineRelocationTicket",
    "EngineRequestForkItem",
    "EngineRequestId",
    "EngineRequestView",
    "EngineRetirement",
    "EngineRetirementEvidence",
    "EngineStepAbortEvidence",
    "EngineStepExecutionEvidence",
    "EngineStepPlan",
    "EngineStepPublication",
    "EngineTokenDispositionBatchItem",
    "EngineTokenDispositionUpdate",
    "EngineTokenView",
    "EngineTokenViewQuery",
    "ExecutionEvidence",
    "_session_id_c",
    "retirement_evidence",
]
