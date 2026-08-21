use std::collections::BTreeMap;
use std::sync::Arc;

use super::arena::{Arena, PageCounts, PageState, RuntimeClass};
use super::identity::{
    PrefixLease, PrefixSemanticKey, RequestLease, SnapshotLease, StepLease, SubmissionLease,
    ViewVersion,
};
#[cfg(test)]
use super::persistent_snapshot::HotPathInstrumentation;
use super::persistent_snapshot::{ClassRoot, RequestSnapshot, RootEntry};
use super::protocol::{CopyIntent, ReclamationCertificate, TailActionKind};

#[derive(Debug)]
pub(super) struct PrefixState {
    pub(super) key: PrefixSemanticKey,
    pub(super) roots: Arc<[ClassRoot]>,
    pub(super) evicted: bool,
}

#[derive(Debug)]
pub(super) struct RequestState {
    pub(super) head: SnapshotLease,
    pub(super) pending_step: Option<StepLease>,
    pub(super) inflight_submission: Option<SubmissionLease>,
    pub(super) last_completion_domain: u64,
    pub(super) last_completion_value: u64,
    pub(super) released: bool,
    pub(super) quarantined: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ClassDelta {
    pub(super) class_id: u16,
    pub(super) tail_action: TailActionKind,
    pub(super) tail_source: Option<RootEntry>,
    pub(super) tail_destination: Option<RootEntry>,
    pub(super) copy_intent: Option<CopyIntent>,
    pub(super) writes: Arc<[RootEntry]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StepDelta {
    pub(super) request: RequestLease,
    pub(super) base_snapshot: SnapshotLease,
    pub(super) target_snapshot: SnapshotLease,
    pub(super) base_view_version: ViewVersion,
    pub(super) target_view_version: ViewVersion,
    pub(super) previous_boundary: u64,
    pub(super) target_boundary: u64,
    pub(super) classes: Box<[ClassDelta]>,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedState {
    pub(super) delta: Arc<StepDelta>,
}

#[derive(Clone, Debug)]
pub(super) struct SubmittedState {
    pub(super) delta: Arc<StepDelta>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ClassTransition {
    pub(super) retire_from_root: usize,
    pub(super) retire_from_writes: usize,
    pub(super) retain_first_ordinal: u64,
    pub(super) resident_count: usize,
}

#[derive(Clone, Debug)]
pub(super) enum OperationState {
    Prepared(PreparedState),
    Submitted(SubmittedState),
}

#[derive(Clone, Debug)]
pub(super) struct ReclamationState {
    pub(super) certificate: ReclamationCertificate,
}

#[derive(Debug)]
pub struct CanonicalKvManager {
    pub(super) engine_epoch: u64,
    pub(super) pool_epoch: u64,
    pub(super) page_tokens: u64,
    pub(super) classes: Box<[RuntimeClass]>,
    pub(super) maximum_step_tokens: u64,
    pub(super) requests: Arena<RequestState>,
    pub(super) snapshots: Arena<RequestSnapshot>,
    pub(super) prefixes: Arena<PrefixState>,
    pub(super) prefix_index: BTreeMap<PrefixSemanticKey, PrefixLease>,
    pub(super) operations: Arena<OperationState>,
    pub(super) reclamations: Arena<ReclamationState>,
    pub(super) pages: Vec<PageState>,
    pub(super) free_pages: Vec<Vec<u32>>,
    pub(super) page_counts: Vec<PageCounts>,
    pub(super) prepared_steps: u64,
    pub(super) submitted_steps: u64,
    pub(super) active_prefixes: u64,
    pub(super) evicted_prefixes: u64,
    #[cfg(test)]
    pub(super) hot_path: HotPathInstrumentation,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CensusWork {
    pub(super) classes: u64,
    pub(super) page_slots: u64,
    pub(super) prefix_slots: u64,
}
