from __future__ import annotations

import ctypes


U8 = ctypes.c_uint8
U16 = ctypes.c_uint16
U32 = ctypes.c_uint32
U64 = ctypes.c_uint64


LEASE_FIELDS = [("engine_epoch", U64), ("slot", U32), ("generation", U32)]


class StateTransitionLeaseLayout(ctypes.Structure):
    _fields_ = LEASE_FIELDS


class StateRetirementLeaseLayout(ctypes.Structure):
    _fields_ = LEASE_FIELDS


class PageLeaseLayout(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", U64),
        ("pool_epoch", U64),
        ("generation", U64),
        ("page_id", U32),
        ("pool_id", U32),
    ]


class StatePoolConfigLayout(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", U64),
        ("pool_epoch", U64),
        ("byte_count", U64),
        ("pool_id", U32),
        ("slot_count", U32),
    ]


class StateSlotLeaseLayout(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", U64),
        ("pool_epoch", U64),
        ("generation", U64),
        ("slot_id", U32),
        ("pool_id", U32),
    ]


class StatePoolIdentityLayout(ctypes.Structure):
    _fields_ = StatePoolConfigLayout._fields_


class StatePoolStatsLayout(ctypes.Structure):
    _fields_ = [
        ("identity", StatePoolIdentityLayout),
        ("free_slots", U64),
        ("reserved_slots", U64),
        ("relocating_slots", U64),
        ("live_slots", U64),
        ("retiring_slots", U64),
        ("quarantined_slots", U64),
        ("active_owners", U64),
        ("pending_transitions", U64),
        ("pending_retirements", U64),
    ]


class StatePrepareItemLayout(ctypes.Structure):
    _fields_ = [
        ("owner_id", U64),
        ("expected", StateSlotLeaseLayout),
        ("expected_present", U32),
        ("reserved", U32),
    ]


class StateCopyIntentLayout(ctypes.Structure):
    _fields_ = [
        ("transition", StateTransitionLeaseLayout),
        ("owner_id", U64),
        ("source", StateSlotLeaseLayout),
        ("destination", StateSlotLeaseLayout),
        ("byte_count", U64),
        ("source_present", U32),
        ("reserved", U32),
    ]


class StateCopyReceiptLayout(ctypes.Structure):
    _fields_ = [
        ("transition", StateTransitionLeaseLayout),
        ("source", StateSlotLeaseLayout),
        ("destination", StateSlotLeaseLayout),
        ("byte_count", U64),
        ("source_present", U8),
        ("observed", U8),
        ("written", U8),
        ("reserved8", U8),
        ("reserved32", U32),
    ]


class StateCompletionReceiptLayout(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", U64),
        ("completion_domain", U64),
        ("completion_value", U64),
        ("confirmed", U32),
        ("reserved", U32),
    ]


class StateRetirementCertificateLayout(ctypes.Structure):
    _fields_ = [
        ("retirement", StateRetirementLeaseLayout),
        ("slot", StateSlotLeaseLayout),
        ("byte_count", U64),
        ("completion_domain", U64),
        ("completion_value", U64),
    ]


class StatePublicationLayout(ctypes.Structure):
    _fields_ = [
        ("owner_id", U64),
        ("slot", StateSlotLeaseLayout),
        ("retirement", StateRetirementCertificateLayout),
        ("retirement_present", U32),
        ("reserved", U32),
    ]


class StateAbortItemLayout(ctypes.Structure):
    _fields_ = [
        ("transition", StateTransitionLeaseLayout),
        ("backend_unobserved", U32),
        ("reserved", U32),
    ]


class StateRetireOwnerItemLayout(ctypes.Structure):
    _fields_ = [("owner_id", U64), ("expected", StateSlotLeaseLayout)]


class StateCurrentLayout(ctypes.Structure):
    _fields_ = [
        ("owner_id", U64),
        ("slot", StateSlotLeaseLayout),
        ("present", U32),
        ("reserved", U32),
    ]


class BackendArenaRegistrationLayout(ctypes.Structure):
    _fields_ = [
        ("pool_id", U32),
        ("class_id", U16),
        ("backend_domain", U16),
        ("page_count", U32),
        ("reserved", U32),
        ("backend_base_index", U64),
    ]


class ManagerConfigLayout(ctypes.Structure):
    _fields_ = [
        ("maximum_requests", U32),
        ("maximum_operations", U32),
        ("maximum_prefixes", U32),
        ("maximum_reclamations", U32),
        ("maximum_step_tokens", U32),
        ("plan_format", U32),
        ("reserved", U64),
    ]


class SessionCreateConfigLayout(ctypes.Structure):
    _fields_ = [
        ("manager", ManagerConfigLayout),
        ("cache_sharing_policy", U32),
        ("reserved", U32),
    ]


class ArenaIdentityLayout(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", U64),
        ("pool_epoch", U64),
        ("backend_base_index", U64),
        ("pool_id", U32),
        ("page_count", U32),
        ("page_tokens", U32),
        ("class_id", U16),
        ("backend_domain", U16),
        ("first_page_id", U32),
        ("reserved", U32),
    ]


class ArenaStatsLayout(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", U64),
        ("pool_epoch", U64),
        ("class_id", U16),
        ("backend_domain", U16),
        ("pool_id", U32),
        ("page_count", U32),
        ("first_page_id", U32),
        ("reserved", U32),
        ("reserved_padding", U32),
        ("free_pages", U64),
        ("reserved_pages", U64),
        ("writing_pages", U64),
        ("active_pages", U64),
        ("retiring_pages", U64),
        ("quarantined_pages", U64),
        ("exhausted_pages", U64),
        ("request_page_refs", U64),
        ("prefix_page_refs", U64),
        ("reader_pins", U64),
    ]


class SnapshotPageLayout(ctypes.Structure):
    _fields_ = [
        ("page", PageLeaseLayout),
        ("logical_ordinal", U64),
        ("temporal_cell_index", U64),
        ("temporal_cycle", U64),
        ("backend_index", U64),
        ("class_id", U16),
        ("backend_domain", U16),
        ("valid_token_count", U32),
        ("visible_token_offset", U32),
        ("visible_token_count", U32),
        ("reserved", U32),
    ]


class ClassLoweringLayout(ctypes.Structure):
    _fields_ = [
        ("class_id", U16),
        ("flags", U16),
        ("tail_offset", U32),
        ("tail_count", U32),
        ("copy_offset", U32),
        ("copy_count", U32),
        ("write_offset", U32),
        ("write_count", U32),
        ("reserved", U32),
        ("previous_layout_boundary", U64),
        ("target_layout_boundary", U64),
    ]


class TailActionLayout(ctypes.Structure):
    _fields_ = [
        ("class_id", U16),
        ("kind", U16),
        ("valid_token_count", U32),
        ("logical_ordinal", U64),
        ("source", PageLeaseLayout),
        ("destination", PageLeaseLayout),
        ("reserved", U64),
    ]


class CopyIntentLayout(ctypes.Structure):
    _fields_ = [
        ("class_id", U16),
        ("backend_domain", U16),
        ("token_count", U32),
        ("source_token_offset", U32),
        ("destination_token_offset", U32),
        ("reserved", U32),
        ("source", PageLeaseLayout),
        ("destination", PageLeaseLayout),
        ("source_backend_index", U64),
        ("destination_backend_index", U64),
    ]


class WriteIntentLayout(ctypes.Structure):
    _fields_ = [("page_generation", U64), ("page_id", U32), ("reserved", U32)]


class DetachedBindingLayout(ctypes.Structure):
    _fields_ = [
        ("old", PageLeaseLayout),
        ("replacement", PageLeaseLayout),
        ("logical_ordinal", U64),
        ("old_backend_index", U64),
        ("replacement_backend_index", U64),
        ("token_begin", U64),
        ("token_end_exclusive", U64),
        ("class_id", U16),
        ("backend_domain", U16),
        ("action", U16),
        ("reason", U16),
        ("reserved", U64),
    ]


class PrefixKeyLayout(ctypes.Structure):
    _fields_ = [("namespace_bytes", U8 * 32), ("digest", U8 * 32), ("boundary", U64)]


class ManagerStatsLayout(ctypes.Structure):
    _fields_ = [
        (name, U64)
        for name in (
            "active_requests",
            "active_snapshots",
            "active_prefixes",
            "evicted_prefixes",
            "prepared_steps",
            "submitted_steps",
            "free_pages",
            "reserved_pages",
            "writing_pages",
            "active_pages",
            "retiring_pages",
            "quarantined_pages",
            "exhausted_pages",
            "pending_reclamations",
            "total_request_page_refs",
            "total_prefix_page_refs",
            "total_reader_pins",
        )
    ]


SESSION_OPERATION_ID_FIELDS = [("session_epoch", U64), ("sequence", U64)]


class SessionBatchIdLayout(ctypes.Structure):
    _fields_ = SESSION_OPERATION_ID_FIELDS


class SessionPublicationIdLayout(ctypes.Structure):
    _fields_ = SESSION_OPERATION_ID_FIELDS


class SessionReleaseIdLayout(ctypes.Structure):
    _fields_ = SESSION_OPERATION_ID_FIELDS


class SessionPrefixIdLayout(ctypes.Structure):
    _fields_ = SESSION_OPERATION_ID_FIELDS


class SessionControlIdLayout(ctypes.Structure):
    _fields_ = SESSION_OPERATION_ID_FIELDS


class SessionRelocationIdLayout(ctypes.Structure):
    _fields_ = SESSION_OPERATION_ID_FIELDS


class SessionRequestViewLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("view_version", U64),
        ("boundary", U64),
        ("resident_count", U32),
        ("reserved", U32),
    ]


class SessionAppendIntentLayout(ctypes.Structure):
    _fields_ = [("request_id", U64), ("target_boundary", U64)]


class SessionPreparedStepLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("base_view_version", U64),
        ("target_view_version", U64),
        ("previous_boundary", U64),
        ("target_boundary", U64),
        ("class_offset", U32),
        ("class_count", U32),
        ("tail_offset", U32),
        ("tail_count", U32),
        ("copy_offset", U32),
        ("copy_count", U32),
        ("write_offset", U32),
        ("write_count", U32),
    ]


class SessionStepExecutionEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("bind_offset", U32),
        ("bind_count", U32),
        ("copy_offset", U32),
        ("copy_count", U32),
        ("reserved", U64),
    ]


class SessionBindEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("page", PageLeaseLayout),
        ("backend_domain", U16),
        ("mapped", U8),
        ("writable", U8),
        ("reserved", U32),
        ("backend_index", U64),
    ]


class SessionCopyEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("class_id", U16),
        ("backend_domain", U16),
        ("token_count", U32),
        ("source_token_offset", U32),
        ("destination_token_offset", U32),
        ("observed", U8),
        ("copied", U8),
        ("ordered_before_writes", U8),
        ("reserved8", U8),
        ("reserved32", U32),
        ("source", PageLeaseLayout),
        ("destination", PageLeaseLayout),
        ("source_backend_index", U64),
        ("destination_backend_index", U64),
    ]


class SessionStepAbortEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("backend_unobserved", U32),
        ("reserved", U32),
    ]


class SessionCompletionEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("completion_domain", U64),
        ("completion_value", U64),
        ("confirmed", U32),
        ("reserved", U32),
    ]


class SessionStepPublicationLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("view_version", U64),
        ("boundary", U64),
        ("resident_count", U32),
        ("detached_offset", U32),
        ("detached_count", U32),
        ("reserved", U32),
    ]


class SessionRetirementLayout(ctypes.Structure):
    _fields_ = [
        ("page", PageLeaseLayout),
        ("class_id", U16),
        ("backend_domain", U16),
        ("reserved32", U32),
        ("logical_ordinal", U64),
        ("backend_index", U64),
        ("token_begin", U64),
        ("token_end_exclusive", U64),
        ("completion_domain", U64),
        ("completion_value", U64),
    ]


class SessionRetirementEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("page", PageLeaseLayout),
        ("backend_domain", U16),
        ("acknowledged", U8),
        ("reserved8", U8),
        ("reserved32", U32),
        ("backend_index", U64),
    ]


class SessionPublicationEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("publication_id", SessionPublicationIdLayout),
        ("mirror_cleanup_confirmed", U32),
        ("reserved", U32),
    ]


class SessionReleasedRequestLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("detached_offset", U32),
        ("detached_count", U32),
    ]


class SessionReleaseEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("release_id", SessionReleaseIdLayout),
        ("mirror_cleanup_confirmed", U32),
        ("reserved", U32),
    ]


class SessionReleaseOutcomeLayout(ctypes.Structure):
    _fields_ = [
        ("release_id", SessionReleaseIdLayout),
        ("disposition", U32),
        ("reserved", U32),
    ]


class SessionPrefixLookupLayout(ctypes.Structure):
    _fields_ = [
        ("key", PrefixKeyLayout),
        ("candidate", SessionPrefixIdLayout),
        ("resident_count", U32),
        ("candidate_present", U32),
        ("reserved0", U32),
        ("reserved1", U32),
    ]


class SessionPrefixPublishItemLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("key", PrefixKeyLayout),
    ]


class SessionPublishedPrefixLayout(ctypes.Structure):
    _fields_ = [
        ("prefix_id", SessionPrefixIdLayout),
        ("key", PrefixKeyLayout),
        ("resident_count", U32),
        ("reserved", U32),
    ]


class SessionPublishedPrefixReleaseLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("prefix_id", SessionPrefixIdLayout),
        ("key", PrefixKeyLayout),
        ("resident_count", U32),
        ("detached_offset", U32),
        ("detached_count", U32),
        ("reserved", U32),
    ]


class SessionPrefixAttachItemLayout(ctypes.Structure):
    _fields_ = [
        ("target_request_id", U64),
        ("prefix_id", SessionPrefixIdLayout),
        ("key", PrefixKeyLayout),
        ("resident_count", U32),
        ("reserved", U32),
    ]


class SessionRequestForkItemLayout(ctypes.Structure):
    _fields_ = [
        ("source_request_id", U64),
        ("target_request_id", U64),
    ]


class SessionControlPlanInfoLayout(ctypes.Structure):
    _fields_ = [
        ("id", SessionControlIdLayout),
        ("kind", U32),
        ("reserved", U32),
        ("request_count", U32),
        ("page_count", U32),
        ("prefix_count", U32),
        ("retirement_count", U32),
    ]


class SessionMaterializedRequestLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("view_version", U64),
        ("boundary", U64),
        ("resident_count", U32),
        ("page_offset", U32),
        ("page_count", U32),
        ("reserved", U32),
    ]


class SessionPendingAttachCancelLayout(ctypes.Structure):
    _fields_ = [
        ("control_id", SessionControlIdLayout),
        ("request_id", U64),
        ("prefix_id", SessionPrefixIdLayout),
        ("view_version", U64),
        ("boundary", U64),
        ("resident_count", U32),
    ]


class SessionPendingAttachCancelOutcomeLayout(ctypes.Structure):
    _fields_ = [
        ("control_id", SessionControlIdLayout),
        ("request_id", U64),
        ("prefix_id", SessionPrefixIdLayout),
        ("view_version", U64),
        ("boundary", U64),
        ("resident_count", U32),
        ("disposition", U32),
    ]


class SessionControlEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("id", SessionControlIdLayout),
        ("mirror_cleanup_confirmed", U32),
        ("reserved", U32),
    ]


class SessionControlOutcomeLayout(ctypes.Structure):
    _fields_ = [
        ("id", SessionControlIdLayout),
        ("disposition", U32),
        ("reserved", U32),
    ]


class SessionTokenViewQueryLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("expected_boundary", U64),
        ("class_id", U16),
        ("reserved16", U16),
        ("reserved32", U32),
    ]


class SessionTokenViewLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("view_version", U64),
        ("placement_offset", U32),
        ("placement_count", U32),
        ("page_tokens", U32),
        ("class_id", U16),
        ("reserved16", U16),
        ("reserved32", U32),
    ]


class SessionTokenDispositionBatchItemLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("update_offset", U32),
        ("update_count", U32),
    ]


class SessionRelocationPlanLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("base_view_version", U64),
        ("target_view_version", U64),
        ("source_offset", U32),
        ("source_count", U32),
        ("destination_offset", U32),
        ("destination_count", U32),
        ("move_offset", U32),
        ("move_count", U32),
        ("projected_reclaimed_pages", U32),
        ("fragmentation_milli", U16),
        ("class_id", U16),
        ("reserved32", U32),
    ]


class SessionRelocationRequestEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("copy_offset", U32),
        ("copy_count", U32),
    ]


class SessionRelocationAbortEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("backend_unobserved", U32),
        ("reserved", U32),
    ]


class SessionRelocationRequestPublicationLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("view_version", U64),
        ("boundary", U64),
        ("resident_count", U32),
        ("reserved", U32),
    ]


class SessionRelocationPublicationEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("relocation_id", SessionRelocationIdLayout),
        ("mirror_cleanup_confirmed", U32),
        ("reserved", U32),
    ]


class TokenDispositionLayout(ctypes.Structure):
    _fields_ = [
        ("policy_or_proof_id", U64),
        ("version", U64),
        ("quality_contract", U64),
        ("kind", U16),
        ("reserved16", U16),
        ("reserved32", U32),
    ]


class TokenLocationLayout(ctypes.Structure):
    _fields_ = [
        ("page", PageLeaseLayout),
        ("backend_index", U64),
        ("offset", U32),
        ("reserved", U32),
    ]


class TokenPlacementLayout(ctypes.Structure):
    _fields_ = [
        ("token_id", U64),
        ("disposition", TokenDispositionLayout),
        ("location", TokenLocationLayout),
        ("location_present", U32),
        ("reserved", U32),
    ]


class ClassTokenDispositionUpdateLayout(ctypes.Structure):
    _fields_ = [
        ("token_id", U64),
        ("disposition", TokenDispositionLayout),
        ("class_id", U16),
        ("reserved16", U16),
        ("reserved32", U32),
    ]


class RelocationPolicyLayout(ctypes.Structure):
    _fields_ = [
        ("maximum_source_pages", U32),
        ("evacuation_headroom_pages", U32),
        ("fragmentation_threshold_milli", U16),
        ("full_evacuation", U8),
        ("reserved8", U8),
        ("reserved32", U32),
    ]


class SessionPrepareRelocationItemLayout(ctypes.Structure):
    _fields_ = [
        ("request_id", U64),
        ("policy", RelocationPolicyLayout),
        ("class_id", U16),
        ("reserved16", U16),
        ("reserved32", U32),
    ]


class SessionRelocationCopyEvidenceLayout(ctypes.Structure):
    _fields_ = [
        ("token_id", U64),
        ("source", TokenLocationLayout),
        ("destination", TokenLocationLayout),
        ("observed", U8),
        ("copied", U8),
        ("reserved16", U16),
        ("reserved32", U32),
    ]


class TokenMoveLayout(ctypes.Structure):
    _fields_ = [
        ("token_id", U64),
        ("source", TokenLocationLayout),
        ("destination", TokenLocationLayout),
    ]


FROZEN_LAYOUTS = {
    StateTransitionLeaseLayout: (16, 8),
    StateRetirementLeaseLayout: (16, 8),
    PageLeaseLayout: (32, 8),
    StatePoolConfigLayout: (32, 8),
    StateSlotLeaseLayout: (32, 8),
    StatePoolIdentityLayout: (32, 8),
    StatePoolStatsLayout: (104, 8),
    StatePrepareItemLayout: (48, 8),
    StateCopyIntentLayout: (104, 8),
    StateCopyReceiptLayout: (96, 8),
    StateCompletionReceiptLayout: (32, 8),
    StateRetirementCertificateLayout: (72, 8),
    StatePublicationLayout: (120, 8),
    StateAbortItemLayout: (24, 8),
    StateRetireOwnerItemLayout: (40, 8),
    StateCurrentLayout: (48, 8),
    BackendArenaRegistrationLayout: (24, 8),
    ManagerConfigLayout: (32, 8),
    SessionCreateConfigLayout: (40, 8),
    ArenaIdentityLayout: (48, 8),
    ArenaStatsLayout: (120, 8),
    SnapshotPageLayout: (88, 8),
    ClassLoweringLayout: (48, 8),
    TailActionLayout: (88, 8),
    CopyIntentLayout: (104, 8),
    WriteIntentLayout: (16, 8),
    DetachedBindingLayout: (120, 8),
    PrefixKeyLayout: (72, 8),
    ManagerStatsLayout: (136, 8),
    TokenDispositionLayout: (32, 8),
    TokenLocationLayout: (48, 8),
    TokenPlacementLayout: (96, 8),
    ClassTokenDispositionUpdateLayout: (48, 8),
    RelocationPolicyLayout: (16, 4),
    TokenMoveLayout: (104, 8),
    SessionBatchIdLayout: (16, 8),
    SessionPublicationIdLayout: (16, 8),
    SessionReleaseIdLayout: (16, 8),
    SessionPrefixIdLayout: (16, 8),
    SessionControlIdLayout: (16, 8),
    SessionRelocationIdLayout: (16, 8),
    SessionRequestViewLayout: (32, 8),
    SessionAppendIntentLayout: (16, 8),
    SessionPreparedStepLayout: (72, 8),
    SessionStepExecutionEvidenceLayout: (32, 8),
    SessionBindEvidenceLayout: (48, 8),
    SessionCopyEvidenceLayout: (104, 8),
    SessionStepAbortEvidenceLayout: (16, 8),
    SessionCompletionEvidenceLayout: (24, 8),
    SessionStepPublicationLayout: (40, 8),
    SessionRetirementLayout: (88, 8),
    SessionRetirementEvidenceLayout: (48, 8),
    SessionPublicationEvidenceLayout: (24, 8),
    SessionReleasedRequestLayout: (16, 8),
    SessionReleaseEvidenceLayout: (24, 8),
    SessionReleaseOutcomeLayout: (24, 8),
    SessionPrefixLookupLayout: (104, 8),
    SessionPrefixPublishItemLayout: (80, 8),
    SessionPublishedPrefixLayout: (96, 8),
    SessionPublishedPrefixReleaseLayout: (112, 8),
    SessionPrefixAttachItemLayout: (104, 8),
    SessionRequestForkItemLayout: (16, 8),
    SessionControlPlanInfoLayout: (40, 8),
    SessionMaterializedRequestLayout: (40, 8),
    SessionPendingAttachCancelLayout: (64, 8),
    SessionPendingAttachCancelOutcomeLayout: (64, 8),
    SessionControlEvidenceLayout: (24, 8),
    SessionControlOutcomeLayout: (24, 8),
    SessionTokenViewQueryLayout: (24, 8),
    SessionTokenViewLayout: (40, 8),
    SessionTokenDispositionBatchItemLayout: (16, 8),
    SessionPrepareRelocationItemLayout: (32, 8),
    SessionRelocationPlanLayout: (64, 8),
    SessionRelocationRequestEvidenceLayout: (16, 8),
    SessionRelocationCopyEvidenceLayout: (112, 8),
    SessionRelocationAbortEvidenceLayout: (16, 8),
    SessionRelocationRequestPublicationLayout: (32, 8),
    SessionRelocationPublicationEvidenceLayout: (24, 8),
}


def assert_frozen_layouts() -> None:
    for layout, expected in FROZEN_LAYOUTS.items():
        actual = (ctypes.sizeof(layout), ctypes.alignment(layout))
        if actual != expected:
            raise RuntimeError(
                f"wire layout drift for {layout.__name__}: {actual} != {expected}"
            )


assert_frozen_layouts()


__all__ = [name for name in globals() if name.endswith("Layout")] + [
    "FROZEN_LAYOUTS",
    "assert_frozen_layouts",
]
