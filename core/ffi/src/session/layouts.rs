use crate::wire::{
    OrbitKvManagerConfig, OrbitKvPageLease, OrbitKvPrefixSemanticKey, OrbitKvRelocationPolicy,
    OrbitKvTokenLocation,
};

pub const ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE: u32 = 1;
pub const ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionCreateConfig {
    pub manager: OrbitKvManagerConfig,
    pub cache_sharing_policy: u32,
    pub reserved: u32,
}

macro_rules! session_id {
    ($name:ident) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
        pub struct $name {
            pub session_epoch: u64,
            pub sequence: u64,
        }
    };
}

session_id!(OrbitKvSessionBatchId);
session_id!(OrbitKvSessionPublicationId);
session_id!(OrbitKvSessionReleaseId);
session_id!(OrbitKvSessionPrefixId);
session_id!(OrbitKvSessionControlId);
session_id!(OrbitKvSessionRelocationId);

pub const ORBITKV_SESSION_RELEASE_COMPLETED: u32 = 1;
pub const ORBITKV_SESSION_RELEASE_RECYCLE_PENDING: u32 = 2;
pub const ORBITKV_SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING: u32 = 1;
pub const ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED: u32 = 2;
pub const ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION: u32 = 1;
pub const ORBITKV_SESSION_CONTROL_KIND_PREFIX_EVICTION: u32 = 2;
pub const ORBITKV_SESSION_CONTROL_OUTCOME_MATERIALIZED: u32 = 1;
pub const ORBITKV_SESSION_CONTROL_OUTCOME_EVICTED: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPrefixLookup {
    pub key: OrbitKvPrefixSemanticKey,
    pub candidate: OrbitKvSessionPrefixId,
    pub resident_count: u32,
    pub candidate_present: u32,
    pub reserved0: u32,
    pub reserved1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPrefixPublishItem {
    pub request_id: u64,
    pub key: OrbitKvPrefixSemanticKey,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPublishedPrefix {
    pub prefix_id: OrbitKvSessionPrefixId,
    pub key: OrbitKvPrefixSemanticKey,
    pub resident_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPublishedPrefixRelease {
    pub request_id: u64,
    pub prefix_id: OrbitKvSessionPrefixId,
    pub key: OrbitKvPrefixSemanticKey,
    pub resident_count: u32,
    pub detached_offset: u32,
    pub detached_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPrefixAttachItem {
    pub target_request_id: u64,
    pub prefix_id: OrbitKvSessionPrefixId,
    pub key: OrbitKvPrefixSemanticKey,
    pub resident_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRequestForkItem {
    pub source_request_id: u64,
    pub target_request_id: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionControlPlanInfo {
    pub id: OrbitKvSessionControlId,
    pub kind: u32,
    pub reserved: u32,
    pub request_count: u32,
    pub page_count: u32,
    pub prefix_count: u32,
    pub retirement_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionMaterializedRequest {
    pub request_id: u64,
    pub view_version: u64,
    pub boundary: u64,
    pub resident_count: u32,
    pub page_offset: u32,
    pub page_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPendingAttachCancel {
    pub control_id: OrbitKvSessionControlId,
    pub request_id: u64,
    pub prefix_id: OrbitKvSessionPrefixId,
    pub view_version: u64,
    pub boundary: u64,
    pub resident_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPendingAttachCancelOutcome {
    pub control_id: OrbitKvSessionControlId,
    pub request_id: u64,
    pub prefix_id: OrbitKvSessionPrefixId,
    pub view_version: u64,
    pub boundary: u64,
    pub resident_count: u32,
    pub disposition: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionControlEvidence {
    pub id: OrbitKvSessionControlId,
    pub mirror_updates_confirmed: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionControlOutcome {
    pub id: OrbitKvSessionControlId,
    pub disposition: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRequestView {
    pub request_id: u64,
    pub view_version: u64,
    pub boundary: u64,
    pub resident_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionTokenViewQuery {
    pub request_id: u64,
    pub expected_boundary: u64,
    pub class_id: u16,
    pub reserved16: u16,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionTokenView {
    pub request_id: u64,
    pub view_version: u64,
    pub placement_offset: u32,
    pub placement_count: u32,
    pub page_tokens: u32,
    pub class_id: u16,
    pub reserved16: u16,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionTokenDispositionBatchItem {
    pub request_id: u64,
    pub update_offset: u32,
    pub update_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPrepareRelocationItem {
    pub request_id: u64,
    pub policy: OrbitKvRelocationPolicy,
    pub class_id: u16,
    pub reserved16: u16,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRelocationPlan {
    pub request_id: u64,
    pub base_view_version: u64,
    pub target_view_version: u64,
    pub source_offset: u32,
    pub source_count: u32,
    pub destination_offset: u32,
    pub destination_count: u32,
    pub move_offset: u32,
    pub move_count: u32,
    pub projected_reclaimed_pages: u32,
    pub fragmentation_milli: u16,
    pub class_id: u16,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRelocationRequestEvidence {
    pub request_id: u64,
    pub copy_offset: u32,
    pub copy_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRelocationCopyEvidence {
    pub token_id: u64,
    pub source: OrbitKvTokenLocation,
    pub destination: OrbitKvTokenLocation,
    pub observed: u8,
    pub copied: u8,
    pub reserved16: u16,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRelocationAbortEvidence {
    pub request_id: u64,
    pub backend_unobserved: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRelocationRequestPublication {
    pub request_id: u64,
    pub view_version: u64,
    pub boundary: u64,
    pub resident_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRelocationPublicationEvidence {
    pub relocation_id: OrbitKvSessionRelocationId,
    pub mirror_cleanup_confirmed: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionAppendIntent {
    pub request_id: u64,
    pub target_boundary: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPreparedStep {
    pub request_id: u64,
    pub base_view_version: u64,
    pub target_view_version: u64,
    pub previous_boundary: u64,
    pub target_boundary: u64,
    pub class_offset: u32,
    pub class_count: u32,
    pub tail_offset: u32,
    pub tail_count: u32,
    pub copy_offset: u32,
    pub copy_count: u32,
    pub write_offset: u32,
    pub write_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionStepExecutionEvidence {
    pub request_id: u64,
    pub bind_offset: u32,
    pub bind_count: u32,
    pub copy_offset: u32,
    pub copy_count: u32,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionBindEvidence {
    pub page: OrbitKvPageLease,
    pub backend_domain: u16,
    pub mapped: u8,
    pub writable: u8,
    pub reserved: u32,
    pub backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionCopyEvidence {
    pub class_id: u16,
    pub backend_domain: u16,
    pub token_count: u32,
    pub source_token_offset: u32,
    pub destination_token_offset: u32,
    pub observed: u8,
    pub copied: u8,
    pub ordered_before_writes: u8,
    pub reserved8: u8,
    pub reserved32: u32,
    pub source: OrbitKvPageLease,
    pub destination: OrbitKvPageLease,
    pub source_backend_index: u64,
    pub destination_backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionStepAbortEvidence {
    pub request_id: u64,
    pub backend_unobserved: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionCompletionEvidence {
    pub completion_domain: u64,
    pub completion_value: u64,
    pub confirmed: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionStepPublication {
    pub request_id: u64,
    pub view_version: u64,
    pub boundary: u64,
    pub resident_count: u32,
    pub detached_offset: u32,
    pub detached_count: u32,
    pub reserved: u32,
}

/// Engine-facing retirement data. The private reclamation lease is reattached
/// from session state when the engine confirms retirement.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRetirement {
    pub page: OrbitKvPageLease,
    pub class_id: u16,
    pub backend_domain: u16,
    pub reserved32: u32,
    pub logical_ordinal: u64,
    pub backend_index: u64,
    pub token_begin: u64,
    pub token_end_exclusive: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionRetirementEvidence {
    pub page: OrbitKvPageLease,
    pub backend_domain: u16,
    pub acknowledged: u8,
    pub reserved8: u8,
    pub reserved32: u32,
    pub backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionPublicationEvidence {
    pub publication_id: OrbitKvSessionPublicationId,
    pub mirror_cleanup_confirmed: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionReleasedRequest {
    pub request_id: u64,
    pub detached_offset: u32,
    pub detached_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionReleaseEvidence {
    pub release_id: OrbitKvSessionReleaseId,
    pub mirror_cleanup_confirmed: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSessionReleaseOutcome {
    pub release_id: OrbitKvSessionReleaseId,
    pub disposition: u32,
    pub reserved: u32,
}

macro_rules! wire_layout {
    ($ty:ty, $size:expr, $align:expr; $($field:ident = $offset:expr),+ $(,)?) => {
        const _: [(); $size] = [(); std::mem::size_of::<$ty>()];
        const _: [(); $align] = [(); std::mem::align_of::<$ty>()];
        $(const _: [(); $offset] = [(); std::mem::offset_of!($ty, $field)];)+
    };
}

wire_layout!(OrbitKvSessionCreateConfig, 40, 8; manager = 0, cache_sharing_policy = 32, reserved = 36);
wire_layout!(OrbitKvSessionBatchId, 16, 8; session_epoch = 0, sequence = 8);
wire_layout!(OrbitKvSessionPublicationId, 16, 8; session_epoch = 0, sequence = 8);
wire_layout!(OrbitKvSessionReleaseId, 16, 8; session_epoch = 0, sequence = 8);
wire_layout!(OrbitKvSessionPrefixId, 16, 8; session_epoch = 0, sequence = 8);
wire_layout!(OrbitKvSessionControlId, 16, 8; session_epoch = 0, sequence = 8);
wire_layout!(OrbitKvSessionRelocationId, 16, 8; session_epoch = 0, sequence = 8);
wire_layout!(OrbitKvSessionPrefixLookup, 104, 8; key = 0, candidate = 72, resident_count = 88, candidate_present = 92, reserved0 = 96, reserved1 = 100);
wire_layout!(OrbitKvSessionPrefixPublishItem, 80, 8; request_id = 0, key = 8);
wire_layout!(OrbitKvSessionPublishedPrefix, 96, 8; prefix_id = 0, key = 16, resident_count = 88, reserved = 92);
wire_layout!(OrbitKvSessionPublishedPrefixRelease, 112, 8; request_id = 0, prefix_id = 8, key = 24, resident_count = 96, detached_offset = 100, detached_count = 104, reserved = 108);
wire_layout!(OrbitKvSessionPrefixAttachItem, 104, 8; target_request_id = 0, prefix_id = 8, key = 24, resident_count = 96, reserved = 100);
wire_layout!(OrbitKvSessionRequestForkItem, 16, 8; source_request_id = 0, target_request_id = 8);
wire_layout!(OrbitKvSessionControlPlanInfo, 40, 8; id = 0, kind = 16, reserved = 20, request_count = 24, page_count = 28, prefix_count = 32, retirement_count = 36);
wire_layout!(OrbitKvSessionMaterializedRequest, 40, 8; request_id = 0, view_version = 8, boundary = 16, resident_count = 24, page_offset = 28, page_count = 32, reserved = 36);
wire_layout!(OrbitKvSessionPendingAttachCancel, 64, 8; control_id = 0, request_id = 16, prefix_id = 24, view_version = 40, boundary = 48, resident_count = 56);
wire_layout!(OrbitKvSessionPendingAttachCancelOutcome, 64, 8; control_id = 0, request_id = 16, prefix_id = 24, view_version = 40, boundary = 48, resident_count = 56, disposition = 60);
wire_layout!(OrbitKvSessionControlEvidence, 24, 8; id = 0, mirror_updates_confirmed = 16, reserved = 20);
wire_layout!(OrbitKvSessionControlOutcome, 24, 8; id = 0, disposition = 16, reserved = 20);
wire_layout!(OrbitKvSessionRequestView, 32, 8; request_id = 0, view_version = 8, boundary = 16, resident_count = 24, reserved = 28);
wire_layout!(OrbitKvSessionTokenViewQuery, 24, 8; request_id = 0, expected_boundary = 8, class_id = 16, reserved16 = 18, reserved32 = 20);
wire_layout!(OrbitKvSessionTokenView, 40, 8; request_id = 0, view_version = 8, placement_offset = 16, placement_count = 20, page_tokens = 24, class_id = 28, reserved16 = 30, reserved32 = 32);
wire_layout!(OrbitKvSessionTokenDispositionBatchItem, 16, 8; request_id = 0, update_offset = 8, update_count = 12);
wire_layout!(OrbitKvSessionPrepareRelocationItem, 32, 8; request_id = 0, policy = 8, class_id = 24, reserved16 = 26, reserved32 = 28);
wire_layout!(OrbitKvSessionRelocationPlan, 64, 8; request_id = 0, base_view_version = 8, target_view_version = 16, source_offset = 24, source_count = 28, destination_offset = 32, destination_count = 36, move_offset = 40, move_count = 44, projected_reclaimed_pages = 48, fragmentation_milli = 52, class_id = 54, reserved32 = 56);
wire_layout!(OrbitKvSessionRelocationRequestEvidence, 16, 8; request_id = 0, copy_offset = 8, copy_count = 12);
wire_layout!(OrbitKvSessionRelocationCopyEvidence, 112, 8; token_id = 0, source = 8, destination = 56, observed = 104, copied = 105, reserved16 = 106, reserved32 = 108);
wire_layout!(OrbitKvSessionRelocationAbortEvidence, 16, 8; request_id = 0, backend_unobserved = 8, reserved = 12);
wire_layout!(OrbitKvSessionRelocationRequestPublication, 32, 8; request_id = 0, view_version = 8, boundary = 16, resident_count = 24, reserved = 28);
wire_layout!(OrbitKvSessionRelocationPublicationEvidence, 24, 8; relocation_id = 0, mirror_cleanup_confirmed = 16, reserved = 20);
wire_layout!(OrbitKvSessionAppendIntent, 16, 8; request_id = 0, target_boundary = 8);
wire_layout!(OrbitKvSessionPreparedStep, 72, 8; request_id = 0, base_view_version = 8, target_view_version = 16, previous_boundary = 24, target_boundary = 32, class_offset = 40, class_count = 44, tail_offset = 48, tail_count = 52, copy_offset = 56, copy_count = 60, write_offset = 64, write_count = 68);
wire_layout!(OrbitKvSessionStepExecutionEvidence, 32, 8; request_id = 0, bind_offset = 8, bind_count = 12, copy_offset = 16, copy_count = 20, reserved = 24);
wire_layout!(OrbitKvSessionBindEvidence, 48, 8; page = 0, backend_domain = 32, mapped = 34, writable = 35, reserved = 36, backend_index = 40);
wire_layout!(OrbitKvSessionCopyEvidence, 104, 8; class_id = 0, backend_domain = 2, token_count = 4, source_token_offset = 8, destination_token_offset = 12, observed = 16, copied = 17, ordered_before_writes = 18, reserved8 = 19, reserved32 = 20, source = 24, destination = 56, source_backend_index = 88, destination_backend_index = 96);
wire_layout!(OrbitKvSessionStepAbortEvidence, 16, 8; request_id = 0, backend_unobserved = 8, reserved = 12);
wire_layout!(OrbitKvSessionCompletionEvidence, 24, 8; completion_domain = 0, completion_value = 8, confirmed = 16, reserved = 20);
wire_layout!(OrbitKvSessionStepPublication, 40, 8; request_id = 0, view_version = 8, boundary = 16, resident_count = 24, detached_offset = 28, detached_count = 32, reserved = 36);
wire_layout!(OrbitKvSessionRetirement, 88, 8; page = 0, class_id = 32, backend_domain = 34, reserved32 = 36, logical_ordinal = 40, backend_index = 48, token_begin = 56, token_end_exclusive = 64, completion_domain = 72, completion_value = 80);
wire_layout!(OrbitKvSessionRetirementEvidence, 48, 8; page = 0, backend_domain = 32, acknowledged = 34, reserved8 = 35, reserved32 = 36, backend_index = 40);
wire_layout!(OrbitKvSessionPublicationEvidence, 24, 8; publication_id = 0, mirror_cleanup_confirmed = 16, reserved = 20);
wire_layout!(OrbitKvSessionReleasedRequest, 16, 8; request_id = 0, detached_offset = 8, detached_count = 12);
wire_layout!(OrbitKvSessionReleaseEvidence, 24, 8; release_id = 0, mirror_cleanup_confirmed = 16, reserved = 20);
wire_layout!(OrbitKvSessionReleaseOutcome, 24, 8; release_id = 0, disposition = 16, reserved = 20);
