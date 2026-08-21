use serde::Serialize;

use super::identity::{
    PageLease, PrefixLease, PrefixSemanticKey, ReclamationLease, RequestLease, SnapshotLease,
    StepLease, SubmissionLease, ViewVersion,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PrefixLookupHint {
    pub key: PrefixSemanticKey,
    pub candidate: Option<PrefixLease>,
    pub resident_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PrefixAttachItem {
    pub request: RequestLease,
    pub expected_empty_head: SnapshotLease,
    pub hint: PrefixLookupHint,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AttachedPrefix {
    pub prefix: PrefixLease,
    pub target: MaterializedRequestView,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PublishedPrefix {
    pub prefix: PrefixLease,
    pub key: PrefixSemanticKey,
    pub resident_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PrefixPublishItem {
    pub request: RequestLease,
    pub expected_head: SnapshotLease,
    pub key: PrefixSemanticKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EvictedPrefix {
    pub prefix: PrefixLease,
    pub key: PrefixSemanticKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrefixEvictionBatch {
    pub evicted: Box<[EvictedPrefix]>,
    pub retirements: Box<[ReclamationCertificate]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrefixPublishRelease {
    pub publication: PublishedPrefix,
    pub release: ReleaseCompletion,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct BackendArenaRegistration {
    pub pool_id: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub page_count: u32,
    pub reserved: u32,
    pub backend_base_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct ManagerConfig {
    pub maximum_requests: u32,
    pub maximum_operations: u32,
    pub maximum_prefixes: u32,
    pub maximum_reclamations: u32,
    pub maximum_step_tokens: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PreparedStep {
    pub step: StepLease,
    pub request: RequestLease,
    pub base_snapshot: SnapshotLease,
    pub target_snapshot: SnapshotLease,
    pub base_view_version: ViewVersion,
    pub target_view_version: ViewVersion,
    pub previous_boundary: u64,
    pub target_boundary: u64,
    pub class_lowerings: Box<[ClassLowering]>,
    pub tail_actions: Box<[TailAction]>,
    pub copy_intents: Box<[CopyIntent]>,
    pub write_intents: Box<[WriteIntent]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(u16)]
pub enum TailActionKind {
    None = 0,
    InPlace = 1,
    CopyOnWrite = 2,
    Fresh = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct TailAction {
    pub class_id: u16,
    pub kind: TailActionKind,
    pub valid_token_count: u32,
    pub logical_ordinal: u64,
    pub source: PageLease,
    pub destination: PageLease,
    pub reserved: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct CopyIntent {
    pub class_id: u16,
    pub backend_domain: u16,
    pub token_count: u32,
    pub source_token_offset: u32,
    pub destination_token_offset: u32,
    pub reserved: u32,
    pub source: PageLease,
    pub destination: PageLease,
    pub source_backend_index: u64,
    pub destination_backend_index: u64,
}

/// Canonical per-class spans for one prepared append.
///
/// Every class owns exactly one [`TailAction`]. Copy and fresh-write spans may
/// be empty. All three flattened arrays are partitioned without holes in
/// class-id order; physical tail identity lives only in the tail action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct ClassLowering {
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

/// One exact manager-selected page that must be bound before submission.
///
/// The owning class and logical ordinal are derived from the canonical class
/// span. Engine/pool identity, backend domain, and backend index are derived
/// from the request and registered arena identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct WriteIntent {
    pub page_generation: u64,
    pub page_id: u32,
    pub reserved: u32,
}

const _: [(); 32] = [(); std::mem::size_of::<ClassLowering>()];
const _: [(); 4] = [(); std::mem::align_of::<ClassLowering>()];
const _: [(); 16] = [(); std::mem::size_of::<WriteIntent>()];
const _: [(); 8] = [(); std::mem::align_of::<WriteIntent>()];

/// One request-bound append target in an atomic prepare transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PrepareBatchItem {
    pub request: RequestLease,
    pub expected_head: SnapshotLease,
    pub target_boundary: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ReleaseBatchItem {
    pub request: RequestLease,
    pub expected_head: SnapshotLease,
}

/// Atomically forks one immutable source snapshot into an already-acquired
/// empty target request. A batch may name the same source more than once, but
/// every target must be unique and disjoint from all sources.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct RequestForkItem {
    pub source_request: RequestLease,
    pub expected_source_head: SnapshotLease,
    pub target_empty_request: RequestLease,
    pub expected_target_head: SnapshotLease,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct BackendBindReceipt {
    pub step: StepLease,
    pub page: PageLease,
    pub backend_domain: u16,
    pub mapped: u8,
    pub writable: u8,
    pub reserved: u32,
    pub backend_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct BackendCopyReceipt {
    pub step: StepLease,
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
    pub source: PageLease,
    pub destination: PageLease,
    pub source_backend_index: u64,
    pub destination_backend_index: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SubmittedStep {
    pub submission: SubmissionLease,
    pub request: RequestLease,
    pub target_snapshot: SnapshotLease,
}

/// One prepared step and its canonical range in the flattened receipt array.
///
/// The step is authoritative: the request identity is derived by the manager
/// and returned in [`SubmittedStep`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SubmitBatchItem {
    pub step: StepLease,
    pub receipt_offset: u32,
    pub receipt_count: u32,
    pub copy_receipt_offset: u32,
    pub copy_receipt_count: u32,
}

/// One completion event shared by every submission in an atomic completion
/// batch. The engine epoch binds the event to exactly one manager instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct BatchCompletionReceipt {
    pub engine_epoch: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
    pub confirmed: u32,
    pub reserved: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct BackendUnobservedReceipt {
    pub step: StepLease,
    pub backend_unobserved: u32,
    pub reserved: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReclamationCertificate {
    pub reclamation: ReclamationLease,
    pub page: PageLease,
    pub class_id: u16,
    pub backend_domain: u16,
    pub logical_ordinal: u64,
    pub backend_index: u64,
    pub token_begin: u64,
    pub token_end_exclusive: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct ReclamationReceipt {
    pub reclamation: ReclamationLease,
    pub page: PageLease,
    pub backend_domain: u16,
    pub acknowledged: u8,
    pub reserved8: u8,
    pub reserved32: u32,
    pub backend_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(u16)]
pub enum DetachedAction {
    Clear = 1,
    Replace = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(u16)]
pub enum DetachedReason {
    Retention = 1,
    CopyOnWrite = 2,
    RequestRelease = 3,
    PrefixTransfer = 4,
}

/// One exact request-row mirror update caused by an immutable snapshot
/// transition. This is non-owning: reclamation authority remains exclusively
/// in the batch-global [`ReclamationCertificate`] array.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct DetachedBinding {
    pub old: PageLease,
    pub replacement: PageLease,
    pub logical_ordinal: u64,
    pub old_backend_index: u64,
    pub replacement_backend_index: u64,
    pub token_begin: u64,
    pub token_end_exclusive: u64,
    pub class_id: u16,
    pub backend_domain: u16,
    pub action: DetachedAction,
    pub reason: DetachedReason,
    pub reserved: u64,
}

const _: [(); 120] = [(); std::mem::size_of::<DetachedBinding>()];
const _: [(); 8] = [(); std::mem::align_of::<DetachedBinding>()];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StepCompletion {
    pub submission: SubmissionLease,
    pub request: RequestLease,
    pub detached_snapshot: SnapshotLease,
    pub publication: PublishedReceipt,
    pub detached: Box<[DetachedBinding]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompletionBatch {
    pub completions: Box<[StepCompletion]>,
    pub retirements: Box<[ReclamationCertificate]>,
}

/// Compact publication identity returned after an atomic completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PublishedReceipt {
    pub snapshot: SnapshotLease,
    pub view_version: ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RequestView {
    pub request: RequestLease,
    pub snapshot: SnapshotLease,
    pub view_version: ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
}

/// Cold-path physical materialization of one snapshot root entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SnapshotPage {
    pub class_id: u16,
    pub backend_domain: u16,
    pub logical_ordinal: u64,
    pub temporal_cell_index: u64,
    pub temporal_cycle: u64,
    pub page: PageLease,
    pub backend_index: u64,
    pub valid_token_count: u32,
    pub visible_token_offset: u32,
    pub visible_token_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MaterializedRequestView {
    pub view: RequestView,
    pub pages: Box<[SnapshotPage]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ForkedRequest {
    pub source: RequestLease,
    pub target: MaterializedRequestView,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReleaseCompletion {
    pub request: RequestLease,
    pub detached_snapshot: SnapshotLease,
    pub detached: Box<[DetachedBinding]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReleaseBatchCompletion {
    pub releases: Box<[ReleaseCompletion]>,
    pub retirements: Box<[ReclamationCertificate]>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ManagerStats {
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

/// Pure read census for one compiler-defined retention class and backend pool.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct ArenaStats {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub class_id: u16,
    pub backend_domain: u16,
    pub pool_id: u32,
    pub page_count: u32,
    /// Starts this class pool's manager-global half-open page range. Distinct
    /// class-pool ranges need not be adjacent.
    pub first_page_id: u32,
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
