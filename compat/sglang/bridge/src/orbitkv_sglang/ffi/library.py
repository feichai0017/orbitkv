from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from . import layouts as L


WIRE_VERSION = 14
MANAGER_PLAN_FORMAT_KV_PLAN = 1
MANAGER_PLAN_FORMAT_RETENTION_IR = 2
SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING = 1
SESSION_PENDING_ATTACH_CANCEL_FINALIZED = 2
SESSION_CONTROL_KIND_MATERIALIZATION = 1
SESSION_CONTROL_KIND_PREFIX_EVICTION = 2
SESSION_CONTROL_OUTCOME_MATERIALIZED = 1
SESSION_CONTROL_OUTCOME_EVICTED = 2
SESSION_RELEASE_COMPLETED = 1
SESSION_RELEASE_RECYCLE_PENDING = 2
STATUS_OK = 0
STATUS_BUFFER_TOO_SMALL = 1
STATUS_RETRYABLE_CONFLICT = 2
STATUS_INVALID_ARGUMENT = -1
STATUS_MANAGER_ERROR = -2
STATUS_PANIC = -3
STATUS_FAIL_STOPPED = -4

ERROR_BUFFER_BYTES = 4096


class CanonicalWireUnavailable(RuntimeError):
    """The configured library does not expose the canonical wire surface."""


HANDLE = ctypes.c_void_p
PCHAR = ctypes.POINTER(ctypes.c_char)
PU32 = ctypes.POINTER(ctypes.c_uint32)


def _ptr(layout: Any) -> Any:
    return ctypes.POINTER(layout)


FUNCTION_SPECS: dict[str, tuple[Any, ...]] = {
    "orbitkv_session_create": (
        _ptr(ctypes.c_uint8), ctypes.c_size_t,
        _ptr(L.SessionCreateConfigLayout),
        _ptr(L.BackendArenaRegistrationLayout),
        ctypes.c_uint32, _ptr(HANDLE), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_arena_identities": (
        HANDLE, _ptr(L.ArenaIdentityLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_arena_stats": (
        HANDLE, _ptr(L.ArenaStatsLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_stats": (
        HANDLE, _ptr(L.ManagerStatsLayout), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_acquire_requests": (
        HANDLE, _ptr(ctypes.c_uint64), ctypes.c_uint32,
        _ptr(L.SessionRequestViewLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prepare_append": (
        HANDLE, _ptr(L.SessionAppendIntentLayout), ctypes.c_uint32,
        _ptr(L.SessionBatchIdLayout),
        _ptr(L.SessionPreparedStepLayout), ctypes.c_uint32, PU32,
        _ptr(L.ClassLoweringLayout), ctypes.c_uint32, PU32,
        _ptr(L.TailActionLayout), ctypes.c_uint32, PU32,
        _ptr(L.CopyIntentLayout), ctypes.c_uint32, PU32,
        _ptr(L.WriteIntentLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_submit_execution": (
        HANDLE, L.SessionBatchIdLayout,
        _ptr(L.SessionStepExecutionEvidenceLayout), ctypes.c_uint32,
        _ptr(L.SessionBindEvidenceLayout), ctypes.c_uint32,
        _ptr(L.SessionCopyEvidenceLayout), ctypes.c_uint32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_abort_prepared": (
        HANDLE, L.SessionBatchIdLayout,
        _ptr(L.SessionStepAbortEvidenceLayout), ctypes.c_uint32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_quarantine_prepared": (
        HANDLE, L.SessionBatchIdLayout, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_quarantine_submitted": (
        HANDLE, L.SessionBatchIdLayout, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_complete_execution": (
        HANDLE, L.SessionBatchIdLayout, L.SessionCompletionEvidenceLayout,
        _ptr(L.SessionPublicationIdLayout),
        _ptr(L.SessionStepPublicationLayout), ctypes.c_uint32, PU32,
        _ptr(L.DetachedBindingLayout), ctypes.c_uint32, PU32,
        _ptr(L.SessionRetirementLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_confirm_publication": (
        HANDLE, L.SessionPublicationEvidenceLayout,
        _ptr(L.SessionRetirementEvidenceLayout), ctypes.c_uint32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prepare_release": (
        HANDLE, _ptr(ctypes.c_uint64), ctypes.c_uint32,
        _ptr(L.SessionReleaseIdLayout),
        _ptr(L.SessionReleasedRequestLayout), ctypes.c_uint32, PU32,
        _ptr(L.DetachedBindingLayout), ctypes.c_uint32, PU32,
        _ptr(L.SessionRetirementLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_confirm_release": (
        HANDLE, L.SessionReleaseEvidenceLayout,
        _ptr(L.SessionRetirementEvidenceLayout), ctypes.c_uint32,
        _ptr(L.SessionReleaseOutcomeLayout),
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prefix_lookup_batch": (
        HANDLE, _ptr(L.PrefixKeyLayout), ctypes.c_uint32,
        _ptr(L.SessionPrefixLookupLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prefix_publish_batch": (
        HANDLE, _ptr(L.SessionPrefixPublishItemLayout), ctypes.c_uint32,
        _ptr(L.SessionPublishedPrefixLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prefix_publish_release_batch": (
        HANDLE, _ptr(L.SessionPrefixPublishItemLayout), ctypes.c_uint32,
        _ptr(L.SessionReleaseIdLayout),
        _ptr(L.SessionPublishedPrefixReleaseLayout), ctypes.c_uint32, PU32,
        _ptr(L.DetachedBindingLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prepare_prefix_attach": (
        HANDLE, _ptr(L.SessionPrefixAttachItemLayout), ctypes.c_uint32,
        _ptr(L.SessionControlIdLayout), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prepare_request_fork": (
        HANDLE, _ptr(L.SessionRequestForkItemLayout), ctypes.c_uint32,
        _ptr(L.SessionControlIdLayout), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prepare_prefix_evict": (
        HANDLE, _ptr(L.SessionPrefixIdLayout), ctypes.c_uint32,
        _ptr(L.SessionControlIdLayout), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_abort_control": (
        HANDLE, L.SessionControlIdLayout, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_commit_control": (
        HANDLE, L.SessionControlIdLayout,
        _ptr(L.SessionControlPlanInfoLayout), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_read_control_plan": (
        HANDLE, L.SessionControlIdLayout,
        _ptr(L.SessionMaterializedRequestLayout), ctypes.c_uint32, PU32,
        _ptr(L.SnapshotPageLayout), ctypes.c_uint32, PU32,
        _ptr(L.SessionPrefixIdLayout), ctypes.c_uint32, PU32,
        _ptr(L.SessionRetirementLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_cancel_pending_attach": (
        HANDLE, L.SessionPendingAttachCancelLayout,
        _ptr(L.SessionPendingAttachCancelOutcomeLayout),
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_finalize_pending_attach_cancel": (
        HANDLE, L.SessionPendingAttachCancelLayout,
        _ptr(L.SessionPendingAttachCancelOutcomeLayout),
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_confirm_control": (
        HANDLE, L.SessionControlEvidenceLayout,
        _ptr(L.SessionRetirementEvidenceLayout), ctypes.c_uint32,
        _ptr(L.SessionControlOutcomeLayout),
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_quarantine_control": (
        HANDLE, L.SessionControlIdLayout, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_token_views_batch": (
        HANDLE, _ptr(L.SessionTokenViewQueryLayout), ctypes.c_uint32,
        _ptr(L.SessionTokenViewLayout), ctypes.c_uint32, PU32,
        _ptr(L.TokenPlacementLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_mark_token_dispositions_batch": (
        HANDLE, _ptr(L.SessionTokenDispositionBatchItemLayout), ctypes.c_uint32,
        _ptr(L.ClassTokenDispositionUpdateLayout), ctypes.c_uint32,
        _ptr(L.SessionRequestViewLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_prepare_relocation_batch": (
        HANDLE, _ptr(L.SessionPrepareRelocationItemLayout), ctypes.c_uint32,
        _ptr(L.SessionRelocationIdLayout),
        _ptr(L.SessionRelocationPlanLayout), ctypes.c_uint32, PU32,
        _ptr(L.PageLeaseLayout), ctypes.c_uint32, PU32,
        _ptr(L.PageLeaseLayout), ctypes.c_uint32, PU32,
        _ptr(L.TokenMoveLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_abort_prepared_relocation": (
        HANDLE, L.SessionRelocationIdLayout,
        _ptr(L.SessionRelocationAbortEvidenceLayout), ctypes.c_uint32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_quarantine_relocation": (
        HANDLE, L.SessionRelocationIdLayout, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_submit_relocation": (
        HANDLE, L.SessionRelocationIdLayout,
        _ptr(L.SessionRelocationRequestEvidenceLayout), ctypes.c_uint32,
        _ptr(L.SessionRelocationCopyEvidenceLayout), ctypes.c_uint32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_complete_relocation": (
        HANDLE, L.SessionRelocationIdLayout, L.SessionCompletionEvidenceLayout,
        _ptr(L.SessionRelocationRequestPublicationLayout), ctypes.c_uint32, PU32,
        _ptr(L.SessionRetirementLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_confirm_relocation_publication": (
        HANDLE, L.SessionRelocationPublicationEvidenceLayout,
        _ptr(L.SessionRetirementEvidenceLayout), ctypes.c_uint32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_session_destroy": (HANDLE, PCHAR, ctypes.c_size_t),
    "orbitkv_state_pool_create": (
        _ptr(L.StatePoolConfigLayout), _ptr(HANDLE), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_identity": (
        HANDLE, _ptr(L.StatePoolIdentityLayout), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_stats": (
        HANDLE, _ptr(L.StatePoolStatsLayout), PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_prepare_batch": (
        HANDLE, _ptr(L.StatePrepareItemLayout), ctypes.c_uint32,
        _ptr(L.StateCopyIntentLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_submit_batch": (
        HANDLE, _ptr(L.StateCopyReceiptLayout), ctypes.c_uint32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_complete_batch": (
        HANDLE, _ptr(L.StateCompletionReceiptLayout),
        _ptr(L.StateTransitionLeaseLayout), ctypes.c_uint32,
        _ptr(L.StatePublicationLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_abort_batch": (
        HANDLE, _ptr(L.StateAbortItemLayout), ctypes.c_uint32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_retire_owners_batch": (
        HANDLE, _ptr(L.StateCompletionReceiptLayout),
        _ptr(L.StateRetireOwnerItemLayout), ctypes.c_uint32,
        _ptr(L.StateRetirementCertificateLayout), ctypes.c_uint32, PU32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_acknowledge_batch": (
        HANDLE, _ptr(L.StateRetirementCertificateLayout), ctypes.c_uint32,
        PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_current_batch": (
        HANDLE, _ptr(ctypes.c_uint64), ctypes.c_uint32,
        _ptr(L.StateCurrentLayout), ctypes.c_uint32, PU32, PCHAR, ctypes.c_size_t,
    ),
    "orbitkv_state_pool_destroy": (HANDLE, PCHAR, ctypes.c_size_t),
}


EXACT_SYMBOL_ALLOWLIST = frozenset({"orbitkv_wire_version", *FUNCTION_SPECS})


class LoadedLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        try:
            library = ctypes.CDLL(str(self.path))
        except OSError as error:
            raise CanonicalWireUnavailable(
                f"cannot load OrbitKV wire library {self.path}: {error}"
            ) from error
        try:
            wire = library.orbitkv_wire_version
            wire.argtypes = []
            wire.restype = ctypes.c_uint32
            actual = int(wire())
            if actual != WIRE_VERSION:
                raise CanonicalWireUnavailable(
                    f"OrbitKV wire version mismatch: library={actual}, "
                    f"adapter={WIRE_VERSION}"
                )
            for name, argtypes in FUNCTION_SPECS.items():
                function = getattr(library, name)
                function.argtypes = list(argtypes)
                function.restype = ctypes.c_int32
        except AttributeError as error:
            raise CanonicalWireUnavailable(
                f"OrbitKV library is missing a wire symbol: {error}"
            ) from error
        self.cdll = library

    def function(self, name: str) -> Any:
        if name not in EXACT_SYMBOL_ALLOWLIST:
            raise KeyError(f"symbol is outside the frozen wire allowlist: {name}")
        return getattr(self.cdll, name)


__all__ = [
    "WIRE_VERSION",
    "CanonicalWireUnavailable",
    "ERROR_BUFFER_BYTES",
    "EXACT_SYMBOL_ALLOWLIST",
    "FUNCTION_SPECS",
    "LoadedLibrary",
    "MANAGER_PLAN_FORMAT_KV_PLAN",
    "MANAGER_PLAN_FORMAT_RETENTION_IR",
    "SESSION_PENDING_ATTACH_CANCEL_FINALIZED",
    "SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING",
    "SESSION_CONTROL_KIND_PREFIX_EVICTION",
    "SESSION_CONTROL_KIND_MATERIALIZATION",
    "SESSION_CONTROL_OUTCOME_EVICTED",
    "SESSION_CONTROL_OUTCOME_MATERIALIZED",
    "SESSION_RELEASE_COMPLETED",
    "SESSION_RELEASE_RECYCLE_PENDING",
    "STATUS_BUFFER_TOO_SMALL",
    "STATUS_FAIL_STOPPED",
    "STATUS_INVALID_ARGUMENT",
    "STATUS_MANAGER_ERROR",
    "STATUS_OK",
    "STATUS_PANIC",
    "STATUS_RETRYABLE_CONFLICT",
]
