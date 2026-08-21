from __future__ import annotations

import ctypes


U8 = ctypes.c_uint8
U16 = ctypes.c_uint16
U32 = ctypes.c_uint32
U64 = ctypes.c_uint64


LEASE_FIELDS = [("engine_epoch", U64), ("slot", U32), ("generation", U32)]


class RequestLeaseLayout(ctypes.Structure):
    _fields_ = LEASE_FIELDS


class SnapshotLeaseLayout(ctypes.Structure):
    _fields_ = LEASE_FIELDS


class StepLeaseLayout(ctypes.Structure):
    _fields_ = LEASE_FIELDS


class SubmissionLeaseLayout(ctypes.Structure):
    _fields_ = LEASE_FIELDS


class ReclamationLeaseLayout(ctypes.Structure):
    _fields_ = LEASE_FIELDS


class PrefixLeaseLayout(ctypes.Structure):
    _fields_ = LEASE_FIELDS


class PageLeaseLayout(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", U64),
        ("pool_epoch", U64),
        ("generation", U64),
        ("page_id", U32),
        ("pool_id", U32),
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


class RequestViewLayout(ctypes.Structure):
    _fields_ = [
        ("request", RequestLeaseLayout),
        ("snapshot", SnapshotLeaseLayout),
        ("view_version", U64),
        ("boundary", U64),
        ("resident_count", U32),
        ("reserved", U32),
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


class RequestForkItemLayout(ctypes.Structure):
    _fields_ = [
        ("source_request", RequestLeaseLayout),
        ("expected_source_head", SnapshotLeaseLayout),
        ("target_empty_request", RequestLeaseLayout),
        ("expected_target_head", SnapshotLeaseLayout),
    ]


class ForkedItemLayout(ctypes.Structure):
    _fields_ = [
        ("source", RequestLeaseLayout),
        ("target", RequestViewLayout),
        ("page_offset", U32),
        ("page_count", U32),
    ]


class PrepareItemLayout(ctypes.Structure):
    _fields_ = [
        ("request", RequestLeaseLayout),
        ("expected_head", SnapshotLeaseLayout),
        ("target_boundary", U64),
        ("reserved", U64),
    ]


class PreparedItemLayout(ctypes.Structure):
    _fields_ = [
        ("step", StepLeaseLayout),
        ("request", RequestLeaseLayout),
        ("base_snapshot", SnapshotLeaseLayout),
        ("target_snapshot", SnapshotLeaseLayout),
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


class BindReceiptLayout(ctypes.Structure):
    _fields_ = [
        ("step", StepLeaseLayout),
        ("page", PageLeaseLayout),
        ("backend_domain", U16),
        ("mapped", U8),
        ("writable", U8),
        ("reserved", U32),
        ("backend_index", U64),
    ]


class CopyReceiptLayout(ctypes.Structure):
    _fields_ = [
        ("step", StepLeaseLayout),
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


class SubmitItemLayout(ctypes.Structure):
    _fields_ = [
        ("step", StepLeaseLayout),
        ("receipt_offset", U32),
        ("receipt_count", U32),
        ("copy_receipt_offset", U32),
        ("copy_receipt_count", U32),
    ]


class SubmittedItemLayout(ctypes.Structure):
    _fields_ = [
        ("submission", SubmissionLeaseLayout),
        ("request", RequestLeaseLayout),
        ("target_snapshot", SnapshotLeaseLayout),
    ]


class CompletionReceiptLayout(ctypes.Structure):
    _fields_ = [
        ("engine_epoch", U64),
        ("completion_domain", U64),
        ("completion_value", U64),
        ("confirmed", U32),
        ("reserved", U32),
    ]


class CompleteItemLayout(ctypes.Structure):
    _fields_ = [("submission", SubmissionLeaseLayout)]


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


class ReclamationCertificateLayout(ctypes.Structure):
    _fields_ = [
        ("reclamation", ReclamationLeaseLayout),
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


class CompletedItemLayout(ctypes.Structure):
    _fields_ = [
        ("submission", SubmissionLeaseLayout),
        ("request", RequestLeaseLayout),
        ("detached_snapshot", SnapshotLeaseLayout),
        ("published_snapshot", SnapshotLeaseLayout),
        ("published_view_version", U64),
        ("published_boundary", U64),
        ("resident_count", U32),
        ("detached_offset", U32),
        ("detached_count", U32),
        ("reserved", U32),
    ]


class UnobservedReceiptLayout(ctypes.Structure):
    _fields_ = [("step", StepLeaseLayout), ("backend_unobserved", U32), ("reserved", U32)]


class ReleaseItemLayout(ctypes.Structure):
    _fields_ = [("request", RequestLeaseLayout), ("expected_head", SnapshotLeaseLayout)]


class ReleasedItemLayout(ctypes.Structure):
    _fields_ = [
        ("request", RequestLeaseLayout),
        ("detached_snapshot", SnapshotLeaseLayout),
        ("detached_offset", U32),
        ("detached_count", U32),
        ("reserved", U64),
    ]


class ReclamationReceiptLayout(ctypes.Structure):
    _fields_ = [
        ("reclamation", ReclamationLeaseLayout),
        ("page", PageLeaseLayout),
        ("backend_domain", U16),
        ("acknowledged", U8),
        ("reserved8", U8),
        ("reserved32", U32),
        ("backend_index", U64),
    ]


class PrefixKeyLayout(ctypes.Structure):
    _fields_ = [("namespace_bytes", U8 * 32), ("digest", U8 * 32), ("boundary", U64)]


class PrefixLookupHintLayout(ctypes.Structure):
    _fields_ = [
        ("key", PrefixKeyLayout),
        ("candidate", PrefixLeaseLayout),
        ("resident_count", U32),
        ("candidate_present", U32),
        ("reserved", U32),
        ("reserved_padding", U32),
    ]


class PrefixAttachItemLayout(ctypes.Structure):
    _fields_ = [
        ("request", RequestLeaseLayout),
        ("expected_empty_head", SnapshotLeaseLayout),
        ("hint", PrefixLookupHintLayout),
    ]


class AttachedPrefixLayout(ctypes.Structure):
    _fields_ = [
        ("prefix", PrefixLeaseLayout),
        ("target", RequestViewLayout),
        ("page_offset", U32),
        ("page_count", U32),
    ]


class PrefixPublishItemLayout(ctypes.Structure):
    _fields_ = [
        ("request", RequestLeaseLayout),
        ("expected_head", SnapshotLeaseLayout),
        ("key", PrefixKeyLayout),
    ]


class PublishedPrefixLayout(ctypes.Structure):
    _fields_ = [
        ("prefix", PrefixLeaseLayout),
        ("key", PrefixKeyLayout),
        ("resident_count", U32),
        ("reserved", U32),
    ]


class PrefixPublishReleaseLayout(ctypes.Structure):
    _fields_ = [
        ("publication", PublishedPrefixLayout),
        ("request", RequestLeaseLayout),
        ("detached_snapshot", SnapshotLeaseLayout),
        ("detached_offset", U32),
        ("detached_count", U32),
        ("reserved", U64),
    ]


class EvictedPrefixLayout(ctypes.Structure):
    _fields_ = [("prefix", PrefixLeaseLayout), ("key", PrefixKeyLayout)]


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


FROZEN_LAYOUTS = {
    RequestLeaseLayout: (16, 8),
    SnapshotLeaseLayout: (16, 8),
    StepLeaseLayout: (16, 8),
    SubmissionLeaseLayout: (16, 8),
    ReclamationLeaseLayout: (16, 8),
    PrefixLeaseLayout: (16, 8),
    PageLeaseLayout: (32, 8),
    BackendArenaRegistrationLayout: (24, 8),
    ManagerConfigLayout: (20, 4),
    ArenaIdentityLayout: (48, 8),
    ArenaStatsLayout: (120, 8),
    RequestViewLayout: (56, 8),
    SnapshotPageLayout: (88, 8),
    RequestForkItemLayout: (64, 8),
    ForkedItemLayout: (80, 8),
    PrepareItemLayout: (48, 8),
    PreparedItemLayout: (128, 8),
    ClassLoweringLayout: (32, 4),
    TailActionLayout: (88, 8),
    CopyIntentLayout: (104, 8),
    WriteIntentLayout: (16, 8),
    BindReceiptLayout: (64, 8),
    CopyReceiptLayout: (120, 8),
    SubmitItemLayout: (32, 8),
    SubmittedItemLayout: (48, 8),
    CompletionReceiptLayout: (32, 8),
    CompleteItemLayout: (16, 8),
    DetachedBindingLayout: (120, 8),
    ReclamationCertificateLayout: (104, 8),
    CompletedItemLayout: (96, 8),
    UnobservedReceiptLayout: (24, 8),
    ReleaseItemLayout: (32, 8),
    ReleasedItemLayout: (48, 8),
    ReclamationReceiptLayout: (64, 8),
    PrefixKeyLayout: (72, 8),
    PrefixLookupHintLayout: (104, 8),
    PrefixAttachItemLayout: (136, 8),
    AttachedPrefixLayout: (80, 8),
    PrefixPublishItemLayout: (104, 8),
    PublishedPrefixLayout: (96, 8),
    PrefixPublishReleaseLayout: (144, 8),
    EvictedPrefixLayout: (88, 8),
    ManagerStatsLayout: (136, 8),
}


def assert_frozen_layouts() -> None:
    for layout, expected in FROZEN_LAYOUTS.items():
        actual = (ctypes.sizeof(layout), ctypes.alignment(layout))
        if actual != expected:
            raise RuntimeError(
                f"ABI6 layout drift for {layout.__name__}: {actual} != {expected}"
            )


assert_frozen_layouts()


__all__ = [name for name in globals() if name.endswith("Layout")] + [
    "FROZEN_LAYOUTS",
    "assert_frozen_layouts",
]
