use std::collections::BTreeMap;
use std::sync::Arc;

use super::arena::{Arena, PageCounts, PageState, RuntimeClass};
use super::identity::{
    PrefixLease, PrefixSemanticKey, RelocationLease, RequestLease, SnapshotLease, StepLease,
    SubmissionLease, ViewVersion,
};
#[cfg(test)]
use super::persistent_snapshot::HotPathInstrumentation;
use super::persistent_snapshot::{ClassRoot, RequestSnapshot, RootEntry};
use super::protocol::{CopyIntent, ReclamationCertificate, TailActionKind};
use super::relocation_transaction::RelocationState;

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
    pub(super) pending_relocation: Option<RelocationLease>,
    pub(super) last_completion_domain: u64,
    pub(super) last_completion_value: u64,
    pub(super) released: bool,
    pub(super) quarantined: bool,
}

impl RequestState {
    pub(super) fn busy(&self) -> bool {
        self.pending_step.is_some()
            || self.inflight_submission.is_some()
            || self.pending_relocation.is_some()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ClassDelta {
    pub(super) class_id: u16,
    pub(super) layout: super::RootLayout,
    pub(super) previous_layout_boundary: u64,
    pub(super) target_layout_boundary: u64,
    pub(super) epoch_reset: bool,
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
    pub(super) relocations: Arena<RelocationState>,
    pub(super) reclamations: Arena<ReclamationState>,
    pub(super) pages: Vec<PageState>,
    pub(super) free_pages: Vec<Vec<u32>>,
    pub(super) page_counts: Vec<PageCounts>,
    /// Last accepted completion value in each adapter-defined execution domain.
    /// Domains are independent timelines; values must advance strictly within
    /// one domain before a later completion call may publish manager state.
    pub(super) completion_high_water: BTreeMap<u64, u64>,
    pub(super) prepared_steps: u64,
    pub(super) submitted_steps: u64,
    pub(super) active_prefixes: u64,
    pub(super) evicted_prefixes: u64,
    #[cfg(test)]
    pub(super) hot_path: HotPathInstrumentation,
}

impl CanonicalKvManager {
    pub(super) fn validate_completion_frontier(
        &self,
        completion_domain: u64,
        completion_value: u64,
    ) -> Result<(), super::KvManagerError> {
        if completion_domain == 0 || completion_value == 0 {
            return Err(super::KvManagerError::InvalidCompletionFrontier);
        }
        if let Some(&previous) = self.completion_high_water.get(&completion_domain)
            && completion_value <= previous
        {
            return Err(super::KvManagerError::CompletionFrontierDidNotAdvance {
                completion_domain,
                previous,
                received: completion_value,
            });
        }
        Ok(())
    }

    pub(super) fn commit_completion_frontier(
        &mut self,
        completion_domain: u64,
        completion_value: u64,
    ) {
        let previous = self
            .completion_high_water
            .insert(completion_domain, completion_value);
        debug_assert!(previous.is_none_or(|value| value < completion_value));
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CensusWork {
    pub(super) classes: u64,
    pub(super) page_slots: u64,
    pub(super) prefix_slots: u64,
}
