macro_rules! lease {
    ($name:ident) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
        pub struct $name {
            pub engine_epoch: u64,
            pub slot: u32,
            pub generation: u32,
        }
    };
}

lease!(OrbitKvRequestLease);
lease!(OrbitKvSnapshotLease);
lease!(OrbitKvStepLease);
lease!(OrbitKvSubmissionLease);
lease!(OrbitKvReclamationLease);
lease!(OrbitKvPrefixLease);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct OrbitKvPageLease {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub generation: u64,
    pub page_id: u32,
    pub pool_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBackendArenaRegistration {
    pub pool_id: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub page_count: u32,
    pub reserved: u32,
    pub backend_base_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvManagerConfig {
    pub maximum_requests: u32,
    pub maximum_operations: u32,
    pub maximum_prefixes: u32,
    pub maximum_reclamations: u32,
    pub maximum_step_tokens: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvArenaIdentity {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub backend_base_index: u64,
    pub pool_id: u32,
    pub page_count: u32,
    pub page_tokens: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub first_page_id: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvArenaStats {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub class_id: u16,
    pub backend_domain: u16,
    pub pool_id: u32,
    pub page_count: u32,
    pub first_page_id: u32,
    pub reserved: u32,
    pub reserved_padding: u32,
    pub free_pages: u64,
    pub reserved_pages: u64,
    pub writing_pages: u64,
    pub active_pages: u64,
    pub retiring_pages: u64,
    pub quarantined_pages: u64,
    pub exhausted_pages: u64,
    pub request_page_refs: u64,
    pub prefix_page_refs: u64,
    pub reader_pins: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvRequestView {
    pub request: OrbitKvRequestLease,
    pub snapshot: OrbitKvSnapshotLease,
    pub view_version: u64,
    pub boundary: u64,
    pub resident_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSnapshotPage {
    pub page: OrbitKvPageLease,
    pub logical_ordinal: u64,
    pub temporal_cell_index: u64,
    pub temporal_cycle: u64,
    pub backend_index: u64,
    pub class_id: u16,
    pub backend_domain: u16,
    pub valid_token_count: u32,
    pub visible_token_offset: u32,
    pub visible_token_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvRequestForkBatchItem {
    pub source_request: OrbitKvRequestLease,
    pub expected_source_head: OrbitKvSnapshotLease,
    pub target_empty_request: OrbitKvRequestLease,
    pub expected_target_head: OrbitKvSnapshotLease,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvForkedBatchItem {
    pub source: OrbitKvRequestLease,
    pub target: OrbitKvRequestView,
    pub page_offset: u32,
    pub page_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPrepareBatchItem {
    pub request: OrbitKvRequestLease,
    pub expected_head: OrbitKvSnapshotLease,
    pub target_boundary: u64,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPreparedBatchItem {
    pub step: OrbitKvStepLease,
    pub request: OrbitKvRequestLease,
    pub base_snapshot: OrbitKvSnapshotLease,
    pub target_snapshot: OrbitKvSnapshotLease,
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
pub struct OrbitKvClassLowering {
    pub class_id: u16,
    pub flags: u16,
    pub tail_offset: u32,
    pub tail_count: u32,
    pub copy_offset: u32,
    pub copy_count: u32,
    pub write_offset: u32,
    pub write_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvTailAction {
    pub class_id: u16,
    pub kind: u16,
    pub valid_token_count: u32,
    pub logical_ordinal: u64,
    pub source: OrbitKvPageLease,
    pub destination: OrbitKvPageLease,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvCopyIntent {
    pub class_id: u16,
    pub backend_domain: u16,
    pub token_count: u32,
    pub source_token_offset: u32,
    pub destination_token_offset: u32,
    pub reserved: u32,
    pub source: OrbitKvPageLease,
    pub destination: OrbitKvPageLease,
    pub source_backend_index: u64,
    pub destination_backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvWriteIntent {
    pub page_generation: u64,
    pub page_id: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBackendBindReceipt {
    pub step: OrbitKvStepLease,
    pub page: OrbitKvPageLease,
    pub backend_domain: u16,
    pub mapped: u8,
    pub writable: u8,
    pub reserved: u32,
    pub backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBackendCopyReceipt {
    pub step: OrbitKvStepLease,
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
pub struct OrbitKvSubmitBatchItem {
    pub step: OrbitKvStepLease,
    pub receipt_offset: u32,
    pub receipt_count: u32,
    pub copy_receipt_offset: u32,
    pub copy_receipt_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSubmittedBatchItem {
    pub submission: OrbitKvSubmissionLease,
    pub request: OrbitKvRequestLease,
    pub target_snapshot: OrbitKvSnapshotLease,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBatchCompletionReceipt {
    pub engine_epoch: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
    pub confirmed: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvCompleteBatchItem {
    pub submission: OrbitKvSubmissionLease,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvDetachedBinding {
    pub old: OrbitKvPageLease,
    pub replacement: OrbitKvPageLease,
    pub logical_ordinal: u64,
    pub old_backend_index: u64,
    pub replacement_backend_index: u64,
    pub token_begin: u64,
    pub token_end_exclusive: u64,
    pub class_id: u16,
    pub backend_domain: u16,
    pub action: u16,
    pub reason: u16,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReclamationCertificate {
    pub reclamation: OrbitKvReclamationLease,
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
pub struct OrbitKvCompletedBatchItem {
    pub submission: OrbitKvSubmissionLease,
    pub request: OrbitKvRequestLease,
    pub detached_snapshot: OrbitKvSnapshotLease,
    pub published_snapshot: OrbitKvSnapshotLease,
    pub published_view_version: u64,
    pub published_boundary: u64,
    pub resident_count: u32,
    pub detached_offset: u32,
    pub detached_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBackendUnobservedReceipt {
    pub step: OrbitKvStepLease,
    pub backend_unobserved: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReleaseBatchItem {
    pub request: OrbitKvRequestLease,
    pub expected_head: OrbitKvSnapshotLease,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReleasedBatchItem {
    pub request: OrbitKvRequestLease,
    pub detached_snapshot: OrbitKvSnapshotLease,
    pub detached_offset: u32,
    pub detached_count: u32,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReclamationReceipt {
    pub reclamation: OrbitKvReclamationLease,
    pub page: OrbitKvPageLease,
    pub backend_domain: u16,
    pub acknowledged: u8,
    pub reserved8: u8,
    pub reserved32: u32,
    pub backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPrefixSemanticKey {
    pub namespace: [u8; 32],
    pub digest: [u8; 32],
    pub boundary: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPrefixLookupHint {
    pub key: OrbitKvPrefixSemanticKey,
    pub candidate: OrbitKvPrefixLease,
    pub resident_count: u32,
    pub candidate_present: u32,
    pub reserved: u32,
    pub reserved_padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPrefixAttachBatchItem {
    pub request: OrbitKvRequestLease,
    pub expected_empty_head: OrbitKvSnapshotLease,
    pub hint: OrbitKvPrefixLookupHint,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvAttachedPrefixBatchItem {
    pub prefix: OrbitKvPrefixLease,
    pub target: OrbitKvRequestView,
    pub page_offset: u32,
    pub page_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPrefixPublishBatchItem {
    pub request: OrbitKvRequestLease,
    pub expected_head: OrbitKvSnapshotLease,
    pub key: OrbitKvPrefixSemanticKey,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPublishedPrefix {
    pub prefix: OrbitKvPrefixLease,
    pub key: OrbitKvPrefixSemanticKey,
    pub resident_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPrefixPublishReleaseBatchItem {
    pub publication: OrbitKvPublishedPrefix,
    pub request: OrbitKvRequestLease,
    pub detached_snapshot: OrbitKvSnapshotLease,
    pub detached_offset: u32,
    pub detached_count: u32,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvEvictedPrefix {
    pub prefix: OrbitKvPrefixLease,
    pub key: OrbitKvPrefixSemanticKey,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvManagerStats {
    pub active_requests: u64,
    pub active_snapshots: u64,
    pub active_prefixes: u64,
    pub evicted_prefixes: u64,
    pub prepared_steps: u64,
    pub submitted_steps: u64,
    pub free_pages: u64,
    pub reserved_pages: u64,
    pub writing_pages: u64,
    pub active_pages: u64,
    pub retiring_pages: u64,
    pub quarantined_pages: u64,
    pub exhausted_pages: u64,
    pub pending_reclamations: u64,
    pub total_request_page_refs: u64,
    pub total_prefix_page_refs: u64,
    pub total_reader_pins: u64,
}

macro_rules! abi_layout {
    ($ty:ty, $size:expr, $align:expr; $($field:ident = $offset:expr),+ $(,)?) => {
        const _: [(); $size] = [(); std::mem::size_of::<$ty>()];
        const _: [(); $align] = [(); std::mem::align_of::<$ty>()];
        $(const _: [(); $offset] = [(); std::mem::offset_of!($ty, $field)];)+
    };
}

abi_layout!(OrbitKvRequestLease, 16, 8; engine_epoch = 0, slot = 8, generation = 12);
abi_layout!(OrbitKvSnapshotLease, 16, 8; engine_epoch = 0, slot = 8, generation = 12);
abi_layout!(OrbitKvStepLease, 16, 8; engine_epoch = 0, slot = 8, generation = 12);
abi_layout!(OrbitKvSubmissionLease, 16, 8; engine_epoch = 0, slot = 8, generation = 12);
abi_layout!(OrbitKvReclamationLease, 16, 8; engine_epoch = 0, slot = 8, generation = 12);
abi_layout!(OrbitKvPrefixLease, 16, 8; engine_epoch = 0, slot = 8, generation = 12);
abi_layout!(OrbitKvPageLease, 32, 8; engine_epoch = 0, pool_epoch = 8, generation = 16, page_id = 24, pool_id = 28);
abi_layout!(OrbitKvBackendArenaRegistration, 24, 8; pool_id = 0, class_id = 4, backend_domain = 6, page_count = 8, reserved = 12, backend_base_index = 16);
abi_layout!(OrbitKvManagerConfig, 20, 4; maximum_requests = 0, maximum_operations = 4, maximum_prefixes = 8, maximum_reclamations = 12, maximum_step_tokens = 16);
abi_layout!(OrbitKvArenaIdentity, 48, 8; engine_epoch = 0, pool_epoch = 8, backend_base_index = 16, pool_id = 24, page_count = 28, page_tokens = 32, class_id = 36, backend_domain = 38, first_page_id = 40, reserved = 44);
abi_layout!(OrbitKvArenaStats, 120, 8; engine_epoch = 0, pool_epoch = 8, class_id = 16, backend_domain = 18, pool_id = 20, page_count = 24, first_page_id = 28, reserved = 32, reserved_padding = 36, free_pages = 40, reserved_pages = 48, writing_pages = 56, active_pages = 64, retiring_pages = 72, quarantined_pages = 80, exhausted_pages = 88, request_page_refs = 96, prefix_page_refs = 104, reader_pins = 112);
abi_layout!(OrbitKvRequestView, 56, 8; request = 0, snapshot = 16, view_version = 32, boundary = 40, resident_count = 48, reserved = 52);
abi_layout!(OrbitKvSnapshotPage, 88, 8; page = 0, logical_ordinal = 32, temporal_cell_index = 40, temporal_cycle = 48, backend_index = 56, class_id = 64, backend_domain = 66, valid_token_count = 68, visible_token_offset = 72, visible_token_count = 76, reserved = 80);
abi_layout!(OrbitKvRequestForkBatchItem, 64, 8; source_request = 0, expected_source_head = 16, target_empty_request = 32, expected_target_head = 48);
abi_layout!(OrbitKvForkedBatchItem, 80, 8; source = 0, target = 16, page_offset = 72, page_count = 76);
abi_layout!(OrbitKvPrepareBatchItem, 48, 8; request = 0, expected_head = 16, target_boundary = 32, reserved = 40);
abi_layout!(OrbitKvPreparedBatchItem, 128, 8; step = 0, request = 16, base_snapshot = 32, target_snapshot = 48, base_view_version = 64, target_view_version = 72, previous_boundary = 80, target_boundary = 88, class_offset = 96, class_count = 100, tail_offset = 104, tail_count = 108, copy_offset = 112, copy_count = 116, write_offset = 120, write_count = 124);
abi_layout!(OrbitKvClassLowering, 32, 4; class_id = 0, flags = 2, tail_offset = 4, tail_count = 8, copy_offset = 12, copy_count = 16, write_offset = 20, write_count = 24, reserved = 28);
abi_layout!(OrbitKvTailAction, 88, 8; class_id = 0, kind = 2, valid_token_count = 4, logical_ordinal = 8, source = 16, destination = 48, reserved = 80);
abi_layout!(OrbitKvCopyIntent, 104, 8; class_id = 0, backend_domain = 2, token_count = 4, source_token_offset = 8, destination_token_offset = 12, reserved = 16, source = 24, destination = 56, source_backend_index = 88, destination_backend_index = 96);
abi_layout!(OrbitKvWriteIntent, 16, 8; page_generation = 0, page_id = 8, reserved = 12);
abi_layout!(OrbitKvBackendBindReceipt, 64, 8; step = 0, page = 16, backend_domain = 48, mapped = 50, writable = 51, reserved = 52, backend_index = 56);
abi_layout!(OrbitKvBackendCopyReceipt, 120, 8; step = 0, class_id = 16, backend_domain = 18, token_count = 20, source_token_offset = 24, destination_token_offset = 28, observed = 32, copied = 33, ordered_before_writes = 34, reserved8 = 35, reserved32 = 36, source = 40, destination = 72, source_backend_index = 104, destination_backend_index = 112);
abi_layout!(OrbitKvSubmitBatchItem, 32, 8; step = 0, receipt_offset = 16, receipt_count = 20, copy_receipt_offset = 24, copy_receipt_count = 28);
abi_layout!(OrbitKvSubmittedBatchItem, 48, 8; submission = 0, request = 16, target_snapshot = 32);
abi_layout!(OrbitKvBatchCompletionReceipt, 32, 8; engine_epoch = 0, completion_domain = 8, completion_value = 16, confirmed = 24, reserved = 28);
abi_layout!(OrbitKvCompleteBatchItem, 16, 8; submission = 0);
abi_layout!(OrbitKvDetachedBinding, 120, 8; old = 0, replacement = 32, logical_ordinal = 64, old_backend_index = 72, replacement_backend_index = 80, token_begin = 88, token_end_exclusive = 96, class_id = 104, backend_domain = 106, action = 108, reason = 110, reserved = 112);
abi_layout!(OrbitKvReclamationCertificate, 104, 8; reclamation = 0, page = 16, class_id = 48, backend_domain = 50, reserved32 = 52, logical_ordinal = 56, backend_index = 64, token_begin = 72, token_end_exclusive = 80, completion_domain = 88, completion_value = 96);
abi_layout!(OrbitKvCompletedBatchItem, 96, 8; submission = 0, request = 16, detached_snapshot = 32, published_snapshot = 48, published_view_version = 64, published_boundary = 72, resident_count = 80, detached_offset = 84, detached_count = 88, reserved = 92);
abi_layout!(OrbitKvBackendUnobservedReceipt, 24, 8; step = 0, backend_unobserved = 16, reserved = 20);
abi_layout!(OrbitKvReleaseBatchItem, 32, 8; request = 0, expected_head = 16);
abi_layout!(OrbitKvReleasedBatchItem, 48, 8; request = 0, detached_snapshot = 16, detached_offset = 32, detached_count = 36, reserved = 40);
abi_layout!(OrbitKvReclamationReceipt, 64, 8; reclamation = 0, page = 16, backend_domain = 48, acknowledged = 50, reserved8 = 51, reserved32 = 52, backend_index = 56);
abi_layout!(OrbitKvPrefixSemanticKey, 72, 8; namespace = 0, digest = 32, boundary = 64);
abi_layout!(OrbitKvPrefixLookupHint, 104, 8; key = 0, candidate = 72, resident_count = 88, candidate_present = 92, reserved = 96, reserved_padding = 100);
abi_layout!(OrbitKvPrefixAttachBatchItem, 136, 8; request = 0, expected_empty_head = 16, hint = 32);
abi_layout!(OrbitKvAttachedPrefixBatchItem, 80, 8; prefix = 0, target = 16, page_offset = 72, page_count = 76);
abi_layout!(OrbitKvPrefixPublishBatchItem, 104, 8; request = 0, expected_head = 16, key = 32);
abi_layout!(OrbitKvPublishedPrefix, 96, 8; prefix = 0, key = 16, resident_count = 88, reserved = 92);
abi_layout!(OrbitKvPrefixPublishReleaseBatchItem, 144, 8; publication = 0, request = 96, detached_snapshot = 112, detached_offset = 128, detached_count = 132, reserved = 136);
abi_layout!(OrbitKvEvictedPrefix, 88, 8; prefix = 0, key = 16);
abi_layout!(OrbitKvManagerStats, 136, 8; active_requests = 0, active_snapshots = 8, active_prefixes = 16, evicted_prefixes = 24, prepared_steps = 32, submitted_steps = 40, free_pages = 48, reserved_pages = 56, writing_pages = 64, active_pages = 72, retiring_pages = 80, quarantined_pages = 88, exhausted_pages = 96, pending_reclamations = 104, total_request_page_refs = 112, total_prefix_page_refs = 120, total_reader_pins = 128);
