use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use thiserror::Error;

use crate::plan::{
    AddressProgram, BlockDomain, ClassLayoutProgram, CompiledKvClass, CompiledKvPlan,
    RetentionKind, RetirementProgram,
};

#[cfg(test)]
const DEVICE_KV_ACCESS_READ: u32 = 1 << 0;
#[cfg(test)]
const DEVICE_KV_ACCESS_WRITE: u32 = 1 << 1;
#[cfg(test)]
const DEVICE_KV_NEEDS_BINDING: u32 = 1 << 2;
pub const CLASS_LOWERING_HAS_PREVIOUS_TAIL: u16 = 1 << 0;

const CANONICAL_PAGE_TOKENS: u64 = 16;
const FIRST_POOL_EPOCH: u64 = 1;

static NEXT_ENGINE_EPOCH: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct RequestLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct StepLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct SubmissionLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct ReclamationLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct PageLease {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub generation: u64,
    pub page_id: u32,
    pub pool_id: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(transparent)]
pub struct ViewVersion(pub u64);

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
struct DeviceKvEntry {
    pub class_id: u16,
    pub backend_domain: u16,
    pub access_flags: u32,
    pub logical_ordinal: u64,
    pub token_begin: u64,
    pub valid_token_count: u32,
    pub visible_token_offset: u32,
    pub visible_token_count: u32,
    pub pool_id: u32,
    pub temporal_cell_index: u64,
    pub temporal_cycle: u64,
    pub pool_epoch: u64,
    pub page_generation: u64,
    pub backend_index: u64,
    pub page_id: u32,
    pub reserved: u32,
}

#[cfg(test)]
const _: [(); 88] = [(); std::mem::size_of::<DeviceKvEntry>()];
#[cfg(test)]
const _: [(); 8] = [(); std::mem::align_of::<DeviceKvEntry>()];

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
    pub maximum_reclamations: u32,
    pub maximum_step_tokens: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PreparedStep {
    pub step: StepLease,
    pub request: RequestLease,
    pub base_view_version: ViewVersion,
    pub target_view_version: ViewVersion,
    pub previous_boundary: u64,
    pub target_boundary: u64,
    pub class_lowerings: Box<[ClassLowering]>,
    pub write_intents: Box<[WriteIntent]>,
}

/// Minimal per-class information required to lower one prepared append.
///
/// `write_offset..write_offset + write_count` indexes this step's
/// [`PreparedStep::write_intents`]. The manager emits classes in canonical
/// class-id order and emits each class's write intents in logical-ordinal
/// order. A previous tail is present only when `previous_boundary` is not
/// page-aligned; every other physical field is recovered from the registered
/// arena identity and the affine page geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct ClassLowering {
    pub class_id: u16,
    pub flags: u16,
    pub write_offset: u32,
    pub write_count: u32,
    pub previous_tail_page_id: u32,
    pub previous_tail_generation: u64,
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

const _: [(); 24] = [(); std::mem::size_of::<ClassLowering>()];
const _: [(); 8] = [(); std::mem::align_of::<ClassLowering>()];
const _: [(); 16] = [(); std::mem::size_of::<WriteIntent>()];
const _: [(); 8] = [(); std::mem::align_of::<WriteIntent>()];

/// One request-bound append target in an atomic prepare transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PrepareBatchItem {
    pub request: RequestLease,
    pub target_boundary: u64,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SubmittedStep {
    pub submission: SubmissionLease,
    pub request: RequestLease,
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
    pub request: RequestLease,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StepCompletion {
    pub submission: SubmissionLease,
    pub request: RequestLease,
    pub publication: PublishedReceipt,
    pub retirements: Box<[ReclamationCertificate]>,
}

/// Compact publication identity returned after an atomic completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PublishedReceipt {
    pub view_version: ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReleaseCompletion {
    pub request: RequestLease,
    pub retirements: Box<[ReclamationCertificate]>,
}

/// One canonical release and its range in the flattened reclamation array.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReleasedBatchItem {
    pub release: ReleaseCompletion,
    pub retirement_offset: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ManagerStats {
    pub active_requests: u64,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RootEntry {
    class_id: u16,
    backend_domain: u16,
    logical_ordinal: u64,
    temporal_cell_index: u64,
    temporal_cycle: u64,
    page: PageLease,
    backend_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RuntimeClass {
    class_id: u16,
    retention: RetentionKind,
    window_tokens: Option<u64>,
    period_blocks: Option<u64>,
    backend: BackendArenaRegistration,
    first_page_id: u32,
}

impl RuntimeClass {
    fn candidate_start(self, previous_boundary: u64) -> u64 {
        match self.retention {
            RetentionKind::Full => 0,
            RetentionKind::Sliding => previous_boundary.saturating_sub(self.history_tokens()),
            RetentionKind::Chunked => unreachable!("canonical profile rejects chunked retention"),
        }
    }

    fn retained_start(self, target_boundary: u64) -> u64 {
        match self.retention {
            RetentionKind::Full => 0,
            RetentionKind::Sliding => target_boundary.saturating_sub(self.history_tokens()),
            RetentionKind::Chunked => unreachable!("canonical profile rejects chunked retention"),
        }
    }

    fn history_tokens(self) -> u64 {
        self.window_tokens
            .expect("validated sliding class has a window")
            .saturating_sub(1)
    }

    fn temporal_address(self, ordinal: u64) -> (u64, u64) {
        match self.retention {
            RetentionKind::Full => (ordinal, 0),
            RetentionKind::Sliding => {
                let period = self
                    .period_blocks
                    .expect("validated sliding class has a period");
                (ordinal % period, ordinal / period)
            }
            RetentionKind::Chunked => unreachable!("canonical profile rejects chunked retention"),
        }
    }

    fn contains_page(self, page_id: u32) -> bool {
        page_id >= self.first_page_id && page_id - self.first_page_id < self.backend.page_count
    }

    fn backend_index(self, page_id: u32) -> Result<u64, KvManagerError> {
        if !self.contains_page(page_id) {
            return Err(KvManagerError::WrongPageArena);
        }
        self.backend
            .backend_base_index
            .checked_add(u64::from(page_id - self.first_page_id))
            .ok_or(KvManagerError::ArithmeticOverflow("backend page index"))
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ClassRoot {
    entries: VecDeque<RootEntry>,
}

#[derive(Debug, Eq, PartialEq)]
struct RequestSnapshot {
    boundary: u64,
    view_version: ViewVersion,
    roots: Box<[ClassRoot]>,
}

impl RequestSnapshot {
    fn resident_count(&self) -> usize {
        self.roots.iter().map(|root| root.entries.len()).sum()
    }

    fn is_empty(&self) -> bool {
        self.roots.iter().all(|root| root.entries.is_empty())
    }
}

#[derive(Debug)]
struct RequestState {
    snapshot: RequestSnapshot,
    pending_step: Option<StepLease>,
    inflight_submission: Option<SubmissionLease>,
    last_completion_domain: u64,
    last_completion_value: u64,
    pending_reclamations: u64,
    released: bool,
    quarantined: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ActiveBinding {
    request: RequestLease,
    class_id: u16,
    logical_ordinal: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PagePhase {
    Free,
    Reserved {
        step: StepLease,
    },
    Writing {
        submission: SubmissionLease,
        previous: Option<ActiveBinding>,
        target: ActiveBinding,
    },
    Active(ActiveBinding),
    Retiring {
        reclamation: ReclamationLease,
    },
    Quarantined,
    Exhausted,
}

#[derive(Clone, Copy, Debug)]
struct PageState {
    class_id: u16,
    generation: u64,
    readers: u32,
    phase: PagePhase,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PageCounts {
    free: u64,
    reserved: u64,
    writing: u64,
    active: u64,
    retiring: u64,
    quarantined: u64,
    exhausted: u64,
}

impl PageCounts {
    fn increment(&mut self, phase: PagePhase) {
        *self.counter_mut(phase) += 1;
    }

    fn decrement(&mut self, phase: PagePhase) {
        let counter = self.counter_mut(phase);
        debug_assert!(*counter > 0);
        *counter -= 1;
    }

    fn counter_mut(&mut self, phase: PagePhase) -> &mut u64 {
        match phase {
            PagePhase::Free => &mut self.free,
            PagePhase::Reserved { .. } => &mut self.reserved,
            PagePhase::Writing { .. } => &mut self.writing,
            PagePhase::Active(_) => &mut self.active,
            PagePhase::Retiring { .. } => &mut self.retiring,
            PagePhase::Quarantined => &mut self.quarantined,
            PagePhase::Exhausted => &mut self.exhausted,
        }
    }
}

impl PageState {
    const fn free(class_id: u16) -> Self {
        Self {
            class_id,
            generation: 0,
            readers: 0,
            phase: PagePhase::Free,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ClassDelta {
    class_id: u16,
    previous_tail: Option<RootEntry>,
    writes: Arc<[RootEntry]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StepDelta {
    request: RequestLease,
    base_view_version: ViewVersion,
    target_view_version: ViewVersion,
    previous_boundary: u64,
    target_boundary: u64,
    classes: Box<[ClassDelta]>,
}

#[derive(Clone, Debug)]
struct PreparedState {
    delta: Arc<StepDelta>,
}

#[derive(Clone, Debug)]
struct SubmittedState {
    delta: Arc<StepDelta>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClassTransition {
    retire_from_root: usize,
    retire_from_writes: usize,
    retain_first_ordinal: u64,
    resident_count: usize,
}

#[derive(Clone, Debug)]
enum OperationState {
    Prepared(PreparedState),
    Submitted(SubmittedState),
}

#[derive(Clone, Debug)]
struct ReclamationState {
    certificate: ReclamationCertificate,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct HotPathInstrumentation {
    hot_root_entries_visited: u64,
    device_view_entries_materialized: u64,
    snapshot_entries_cloned: u64,
    delta_entries_touched: u64,
    retirement_entries_touched: u64,
}

#[derive(Debug)]
struct ArenaSlot<T> {
    generation: u32,
    value: Option<T>,
}

#[derive(Debug)]
struct Arena<T> {
    label: &'static str,
    slots: Vec<ArenaSlot<T>>,
    free: Vec<u32>,
    active: usize,
}

impl<T> Arena<T> {
    fn new(label: &'static str, capacity: u32) -> Result<Self, KvManagerError> {
        if capacity == 0 {
            return Err(KvManagerError::ZeroCapacity(label));
        }
        let capacity = usize::try_from(capacity)
            .map_err(|_| KvManagerError::ArithmeticOverflow("arena capacity"))?;
        Ok(Self {
            label,
            slots: (0..capacity)
                .map(|_| ArenaSlot {
                    generation: 0,
                    value: None,
                })
                .collect(),
            free: (0..u32::try_from(capacity)
                .map_err(|_| KvManagerError::ArithmeticOverflow("arena capacity"))?)
                .rev()
                .collect(),
            active: 0,
        })
    }

    fn plan_many(&self, count: usize) -> Result<Vec<(u32, u32)>, KvManagerError> {
        if self.free.len() < count {
            return Err(KvManagerError::ArenaExhausted(self.label));
        }
        Ok(self
            .free
            .iter()
            .rev()
            .take(count)
            .map(|&slot| {
                let state = &self.slots[slot as usize];
                let generation = state
                    .generation
                    .checked_add(1)
                    .expect("generation-exhausted slots are not free");
                (slot, generation)
            })
            .collect())
    }

    fn insert_planned(&mut self, planned: (u32, u32), value: T) {
        let (slot, generation) = planned;
        let index = usize::try_from(slot).expect("planned arena slot fits usize");
        let popped = self.free.pop();
        assert_eq!(popped, Some(slot), "planned arena slot remains stack head");
        let state = &mut self.slots[index];
        debug_assert!(state.value.is_none());
        debug_assert_eq!(state.generation.checked_add(1), Some(generation));
        state.generation = generation;
        state.value = Some(value);
        self.active += 1;
    }

    fn get(&self, slot: u32, generation: u32) -> Result<&T, KvManagerError> {
        let state = self.slot(slot)?;
        if state.generation != generation {
            return Err(KvManagerError::StaleLease(self.label));
        }
        state
            .value
            .as_ref()
            .ok_or(KvManagerError::StaleLease(self.label))
    }

    fn get_mut(&mut self, slot: u32, generation: u32) -> Result<&mut T, KvManagerError> {
        let label = self.label;
        let state = self.slot_mut(slot)?;
        if state.generation != generation {
            return Err(KvManagerError::StaleLease(label));
        }
        state
            .value
            .as_mut()
            .ok_or(KvManagerError::StaleLease(label))
    }

    fn remove(&mut self, slot: u32, generation: u32) -> Result<T, KvManagerError> {
        let label = self.label;
        let state = self.slot_mut(slot)?;
        if state.generation != generation {
            return Err(KvManagerError::StaleLease(label));
        }
        let value = state
            .value
            .take()
            .ok_or(KvManagerError::StaleLease(label))?;
        if state.generation != u32::MAX {
            self.free.push(slot);
        }
        self.active -= 1;
        Ok(value)
    }

    fn slot(&self, slot: u32) -> Result<&ArenaSlot<T>, KvManagerError> {
        self.slots
            .get(usize::try_from(slot).map_err(|_| KvManagerError::StaleLease(self.label))?)
            .ok_or(KvManagerError::StaleLease(self.label))
    }

    fn slot_mut(&mut self, slot: u32) -> Result<&mut ArenaSlot<T>, KvManagerError> {
        self.slots
            .get_mut(usize::try_from(slot).map_err(|_| KvManagerError::StaleLease(self.label))?)
            .ok_or(KvManagerError::StaleLease(self.label))
    }

    fn active_len(&self) -> usize {
        self.active
    }
}

#[derive(Debug)]
pub struct CanonicalKvManager {
    engine_epoch: u64,
    pool_epoch: u64,
    page_tokens: u64,
    classes: Box<[RuntimeClass]>,
    maximum_step_tokens: u64,
    requests: Arena<RequestState>,
    operations: Arena<OperationState>,
    reclamations: Arena<ReclamationState>,
    pages: Vec<PageState>,
    free_pages: Vec<Vec<u32>>,
    page_counts: Vec<PageCounts>,
    prepared_steps: u64,
    submitted_steps: u64,
    #[cfg(test)]
    hot_path: HotPathInstrumentation,
}

impl CanonicalKvManager {
    /// Creates the canonical Full, sliding, or hybrid Full+sliding manager.
    ///
    /// # Errors
    ///
    /// Rejects non-page-16, chunked, region-partitioned, or otherwise
    /// unsupported profiles. Every accepted class owns one explicitly
    /// registered physical arena, and reclamation capacity must cover every
    /// registered page; unsupported profiles never fall back.
    pub fn new(
        plan: &CompiledKvPlan,
        config: ManagerConfig,
        backends: &[BackendArenaRegistration],
    ) -> Result<Self, KvManagerError> {
        let classes = compile_manager_profile(plan, config, backends)?;
        let total_pages = classes.iter().try_fold(0_usize, |total, class| {
            total
                .checked_add(
                    usize::try_from(class.backend.page_count)
                        .map_err(|_| KvManagerError::ArithmeticOverflow("page count"))?,
                )
                .ok_or(KvManagerError::ArithmeticOverflow("total page count"))
        })?;
        let mut pages = Vec::with_capacity(total_pages);
        let mut free_pages = Vec::with_capacity(classes.len());
        let mut page_counts = Vec::with_capacity(classes.len());
        for class in &classes {
            pages.extend((0..class.backend.page_count).map(|_| PageState::free(class.class_id)));
            let end = class
                .first_page_id
                .checked_add(class.backend.page_count)
                .ok_or(KvManagerError::ArithmeticOverflow("class page range"))?;
            free_pages.push((class.first_page_id..end).rev().collect());
            page_counts.push(PageCounts {
                free: u64::from(class.backend.page_count),
                ..PageCounts::default()
            });
        }
        let engine_epoch = NEXT_ENGINE_EPOCH
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current.checked_add(1)
            })
            .map_err(|_| KvManagerError::EngineEpochExhausted)?;
        Ok(Self {
            engine_epoch,
            pool_epoch: engine_epoch
                .checked_add(FIRST_POOL_EPOCH)
                .ok_or(KvManagerError::EngineEpochExhausted)?,
            page_tokens: plan.page_tokens,
            classes: classes.into_boxed_slice(),
            maximum_step_tokens: u64::from(config.maximum_step_tokens),
            requests: Arena::new("request", config.maximum_requests)?,
            operations: Arena::new("operation", config.maximum_operations)?,
            reclamations: Arena::new("reclamation", config.maximum_reclamations)?,
            pages,
            free_pages,
            page_counts,
            prepared_steps: 0,
            submitted_steps: 0,
            #[cfg(test)]
            hot_path: HotPathInstrumentation::default(),
        })
    }

    #[must_use]
    pub const fn engine_epoch(&self) -> u64 {
        self.engine_epoch
    }

    #[must_use]
    pub const fn pool_epoch(&self) -> u64 {
        self.pool_epoch
    }

    /// Acquires a non-empty batch of request identities atomically.
    ///
    /// # Errors
    ///
    /// Returns an error with no state change when the batch is empty or the
    /// fixed request arena cannot satisfy the entire batch.
    pub fn acquire_requests(
        &mut self,
        request_count: usize,
    ) -> Result<Box<[RequestLease]>, KvManagerError> {
        if request_count == 0 {
            return Err(KvManagerError::EmptyBatch);
        }
        let planned = self.requests.plan_many(request_count)?;
        let requests = planned
            .iter()
            .map(|&(slot, generation)| RequestLease {
                engine_epoch: self.engine_epoch,
                slot,
                generation,
            })
            .collect::<Vec<_>>();
        for (&slot, &request) in planned.iter().zip(&requests) {
            self.requests.insert_planned(
                slot,
                RequestState {
                    snapshot: RequestSnapshot {
                        boundary: 0,
                        view_version: ViewVersion(0),
                        roots: (0..self.classes.len())
                            .map(|_| ClassRoot {
                                entries: VecDeque::new(),
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    },
                    pending_step: None,
                    inflight_submission: None,
                    last_completion_domain: 0,
                    last_completion_value: 0,
                    pending_reclamations: 0,
                    released: false,
                    quarantined: false,
                },
            );
            debug_assert_eq!(request.engine_epoch, self.engine_epoch);
        }
        Ok(requests.into_boxed_slice())
    }

    /// Atomically reserves manager-selected pages for an ordered request batch.
    ///
    /// Every request, operation slot, target boundary, and physical page is
    /// preflighted for the entire batch. Any error leaves requests, operation
    /// generations, page generations, and free lists unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty batch, duplicate request, stale identity,
    /// invalid boundary, insufficient operation capacity, or insufficient
    /// physical capacity.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn prepare_batch(
        &mut self,
        items: &[PrepareBatchItem],
    ) -> Result<Box<[PreparedStep]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_requests = BTreeSet::new();
        for item in items {
            if !seen_requests.insert(item.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
        }
        let planned_operations = self.operations.plan_many(items.len())?;
        let mut page_cursors = vec![0_usize; self.classes.len()];
        let mut plans = Vec::with_capacity(items.len());

        #[cfg(test)]
        let mut delta_entries_touched = 0_u64;

        for (item, planned_operation) in items.iter().zip(planned_operations.iter().copied()) {
            let state = self.request(item.request)?;
            if state.released || state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if state.pending_step.is_some() || state.inflight_submission.is_some() {
                return Err(KvManagerError::RequestBusy);
            }
            if state.snapshot.roots.len() != self.classes.len() {
                return Err(KvManagerError::Invariant("snapshot class cardinality"));
            }
            if item.target_boundary <= state.snapshot.boundary {
                return Err(KvManagerError::NonMonotonicBoundary {
                    current: state.snapshot.boundary,
                    target: item.target_boundary,
                });
            }
            let step_tokens = item.target_boundary - state.snapshot.boundary;
            if step_tokens > self.maximum_step_tokens {
                return Err(KvManagerError::StepTooLarge {
                    requested: step_tokens,
                    maximum: self.maximum_step_tokens,
                });
            }
            let target_version = ViewVersion(
                state
                    .snapshot
                    .view_version
                    .0
                    .checked_add(1)
                    .ok_or(KvManagerError::ViewVersionExhausted)?,
            );
            let step = StepLease {
                engine_epoch: self.engine_epoch,
                slot: planned_operation.0,
                generation: planned_operation.1,
            };
            let previous_boundary = state.snapshot.boundary;
            let first_new_ordinal = previous_boundary.div_ceil(self.page_tokens);
            let new_end_ordinal = item.target_boundary.div_ceil(self.page_tokens);
            let previous_tail_ordinal = (previous_boundary % self.page_tokens != 0)
                .then_some(previous_boundary / self.page_tokens);
            let mut planned_pages = Vec::new();
            let mut class_lowerings = Vec::with_capacity(self.classes.len());
            let mut write_intents = Vec::new();
            let mut class_deltas = Vec::with_capacity(self.classes.len());

            for class in self.classes.iter().copied() {
                let root = state
                    .snapshot
                    .roots
                    .get(usize::from(class.class_id))
                    .ok_or(KvManagerError::Invariant("snapshot class cardinality"))?;
                let class_write_offset = u32::try_from(write_intents.len())
                    .map_err(|_| KvManagerError::ArithmeticOverflow("class write offset"))?;
                let previous_tail = if let Some(ordinal) = previous_tail_ordinal {
                    let tail = root
                        .entries
                        .back()
                        .copied()
                        .filter(|entry| {
                            entry.class_id == class.class_id && entry.logical_ordinal == ordinal
                        })
                        .ok_or(KvManagerError::Invariant("missing previous tail"))?;
                    Some(tail)
                } else {
                    None
                };
                let mut writes = Vec::with_capacity(
                    usize::try_from(new_end_ordinal.saturating_sub(first_new_ordinal))
                        .map_err(|_| KvManagerError::ArithmeticOverflow("class write count"))?,
                );
                for ordinal in first_new_ordinal..new_end_ordinal {
                    let class_index = usize::from(class.class_id);
                    let cursor = page_cursors[class_index];
                    let free = &self.free_pages[class_index];
                    let stack_index = free
                        .len()
                        .checked_sub(cursor.saturating_add(1))
                        .ok_or(KvManagerError::PageCapacityExhausted)?;
                    let page_id = free[stack_index];
                    let page_state = self.page(page_id)?;
                    if page_state.class_id != class.class_id || page_state.phase != PagePhase::Free
                    {
                        return Err(KvManagerError::Invariant("free-page stack state"));
                    }
                    let page = PageLease {
                        engine_epoch: self.engine_epoch,
                        pool_epoch: self.pool_epoch,
                        generation: page_state
                            .generation
                            .checked_add(1)
                            .ok_or(KvManagerError::PageCapacityExhausted)?,
                        page_id,
                        pool_id: class.backend.pool_id,
                    };
                    page_cursors[class_index] = cursor
                        .checked_add(1)
                        .ok_or(KvManagerError::ArithmeticOverflow("batch page cursor"))?;
                    planned_pages.push(page);
                    let entry = self.root_entry_for_page(class, ordinal, page)?;
                    write_intents.push(WriteIntent {
                        page_generation: page.generation,
                        page_id,
                        reserved: 0,
                    });
                    writes.push(entry);
                }
                let class_write_count = u32::try_from(write_intents.len())
                    .map_err(|_| KvManagerError::ArithmeticOverflow("class write count"))?
                    .checked_sub(class_write_offset)
                    .ok_or(KvManagerError::Invariant("class write range"))?;
                let (flags, previous_tail_page_id, previous_tail_generation) = previous_tail
                    .map_or((0, 0, 0), |page| {
                        (
                            CLASS_LOWERING_HAS_PREVIOUS_TAIL,
                            page.page.page_id,
                            page.page.generation,
                        )
                    });
                class_lowerings.push(ClassLowering {
                    class_id: class.class_id,
                    flags,
                    write_offset: class_write_offset,
                    write_count: class_write_count,
                    previous_tail_page_id,
                    previous_tail_generation,
                });
                #[cfg(test)]
                {
                    delta_entries_touched = delta_entries_touched
                        .checked_add(writes.len() as u64)
                        .expect("test instrumentation does not overflow");
                }
                class_deltas.push(ClassDelta {
                    class_id: class.class_id,
                    previous_tail,
                    writes: writes.into(),
                });
            }
            let output = PreparedStep {
                step,
                request: item.request,
                base_view_version: state.snapshot.view_version,
                target_view_version: target_version,
                previous_boundary,
                target_boundary: item.target_boundary,
                class_lowerings: class_lowerings.into_boxed_slice(),
                write_intents: write_intents.into_boxed_slice(),
            };
            let delta = Arc::new(StepDelta {
                request: item.request,
                base_view_version: state.snapshot.view_version,
                target_view_version: target_version,
                previous_boundary,
                target_boundary: item.target_boundary,
                classes: class_deltas.into_boxed_slice(),
            });
            plans.push((
                planned_operation,
                planned_pages,
                PreparedState { delta },
                output,
            ));
        }

        for (planned_operation, planned_pages, prepared, _) in &plans {
            let step = StepLease {
                engine_epoch: self.engine_epoch,
                slot: planned_operation.0,
                generation: planned_operation.1,
            };
            self.apply_page_reservations(step, planned_pages);
            self.operations.insert_planned(
                *planned_operation,
                OperationState::Prepared(prepared.clone()),
            );
            self.prepared_steps += 1;
            self.request_mut(prepared.delta.request)
                .expect("batch preflight retained the request")
                .pending_step = Some(step);
        }
        #[cfg(test)]
        {
            self.hot_path.delta_entries_touched += delta_entries_touched;
        }
        Ok(plans
            .into_iter()
            .map(|(_, _, _, output)| output)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Atomically validates backend bindings and pins an ordered step batch.
    ///
    /// Receipt ranges must form one canonical, gap-free partition of
    /// `receipts` in item order. The authoritative request identity is derived
    /// from each step; callers cannot substitute it.
    ///
    /// # Errors
    ///
    /// Structural identity/range failures reject the whole batch without
    /// mutation. Once all steps are resolved, any semantic bind-receipt
    /// mismatch fail-stops every candidate in the batch: all reachable pages
    /// and requests are quarantined, so they cannot be aborted or reused.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn submit_batch(
        &mut self,
        items: &[SubmitBatchItem],
        receipts: &[BackendBindReceipt],
    ) -> Result<Box<[SubmittedStep]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_steps = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut expected_receipt_offset = 0_usize;
        let mut plans = Vec::with_capacity(items.len());

        for item in items {
            if !seen_steps.insert(item.step) {
                return Err(KvManagerError::DuplicateStep);
            }
            let receipt_offset = usize::try_from(item.receipt_offset)
                .map_err(|_| KvManagerError::InvalidBatchRange)?;
            let receipt_count = usize::try_from(item.receipt_count)
                .map_err(|_| KvManagerError::InvalidBatchRange)?;
            if receipt_offset != expected_receipt_offset {
                return Err(KvManagerError::InvalidBatchRange);
            }
            let receipt_end = receipt_offset
                .checked_add(receipt_count)
                .ok_or(KvManagerError::InvalidBatchRange)?;
            receipts
                .get(receipt_offset..receipt_end)
                .ok_or(KvManagerError::InvalidBatchRange)?;
            expected_receipt_offset = receipt_end;

            self.check_step_epoch(item.step)?;
            let prepared = match self.operations.get(item.step.slot, item.step.generation)? {
                OperationState::Prepared(prepared) => prepared.clone(),
                OperationState::Submitted(_) => return Err(KvManagerError::StepAlreadySubmitted),
            };
            if !seen_requests.insert(prepared.delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            plans.push((item.step, prepared, receipt_offset, receipt_end));
        }
        if expected_receipt_offset != receipts.len() {
            return Err(KvManagerError::InvalidBatchRange);
        }

        let semantic_result = (|| {
            let mut seen_pages = BTreeSet::new();
            for (step, prepared, begin, end) in &plans {
                Self::validate_bind_receipts(*step, prepared, &receipts[*begin..*end])?;
                self.preflight_prepared_delta(prepared, *step)?;
                for class_delta in &prepared.delta.classes {
                    if class_delta
                        .previous_tail
                        .iter()
                        .chain(class_delta.writes.iter())
                        .any(|entry| !seen_pages.insert(entry.page.page_id))
                    {
                        return Err(KvManagerError::DuplicatePage);
                    }
                }
                let request_state = self.request(prepared.delta.request)?;
                if request_state.pending_step != Some(*step)
                    || request_state.snapshot.view_version != prepared.delta.base_view_version
                    || request_state.snapshot.boundary != prepared.delta.previous_boundary
                    || request_state.snapshot.roots.len() != prepared.delta.classes.len()
                {
                    return Err(KvManagerError::StaleView);
                }
            }
            Ok(())
        })();
        if let Err(error) = semantic_result {
            for (step, prepared, _, _) in &plans {
                let mut reachable = self
                    .request(prepared.delta.request)
                    .expect("prepared operation retains its request")
                    .snapshot
                    .roots
                    .iter()
                    .flat_map(|root| root.entries.iter().map(|entry| entry.page.page_id))
                    .collect::<Vec<_>>();
                reachable.extend(
                    prepared
                        .delta
                        .classes
                        .iter()
                        .flat_map(|delta| delta.writes.iter().map(|entry| entry.page.page_id)),
                );
                for page_id in reachable {
                    self.set_page_phase(page_id, PagePhase::Quarantined)
                        .expect("batch submit retained manager-produced candidate page");
                }
                self.operations
                    .remove(step.slot, step.generation)
                    .expect("batch submit preflight retained prepared operation");
                self.prepared_steps -= 1;
                let request = self
                    .request_mut(prepared.delta.request)
                    .expect("batch submit preflight retained request");
                request.pending_step = None;
                request.quarantined = true;
            }
            return Err(error);
        }

        let plans = plans
            .into_iter()
            .map(|(step, prepared, _, _)| {
                let submission = SubmissionLease {
                    engine_epoch: step.engine_epoch,
                    slot: step.slot,
                    generation: step.generation,
                };
                let output = SubmittedStep {
                    submission,
                    request: prepared.delta.request,
                };
                (step, prepared, submission, output)
            })
            .collect::<Vec<_>>();

        #[cfg(test)]
        let mut delta_entries_touched = 0_u64;
        for (step, prepared, submission, _) in &plans {
            for class_delta in &prepared.delta.classes {
                if let Some(entry) = class_delta.previous_tail {
                    let target = ActiveBinding {
                        request: prepared.delta.request,
                        class_id: entry.class_id,
                        logical_ordinal: entry.logical_ordinal,
                    };
                    let page = self
                        .page_mut(entry.page.page_id)
                        .expect("batch preflight validated previous tail");
                    page.readers += 1;
                    self.set_page_phase(
                        entry.page.page_id,
                        PagePhase::Writing {
                            submission: *submission,
                            previous: Some(target),
                            target,
                        },
                    )
                    .expect("batch preflight retained previous tail");
                }
                for entry in class_delta.writes.iter().copied() {
                    let page = self
                        .page_mut(entry.page.page_id)
                        .expect("batch preflight validated reserved write");
                    page.readers += 1;
                    self.set_page_phase(
                        entry.page.page_id,
                        PagePhase::Writing {
                            submission: *submission,
                            previous: None,
                            target: ActiveBinding {
                                request: prepared.delta.request,
                                class_id: entry.class_id,
                                logical_ordinal: entry.logical_ordinal,
                            },
                        },
                    )
                    .expect("batch preflight retained reserved write");
                }
                #[cfg(test)]
                {
                    delta_entries_touched += u64::from(class_delta.previous_tail.is_some())
                        + class_delta.writes.len() as u64;
                }
            }
            *self
                .operations
                .get_mut(step.slot, step.generation)
                .expect("batch preflight retained operation") =
                OperationState::Submitted(SubmittedState {
                    delta: Arc::clone(&prepared.delta),
                });
            self.prepared_steps -= 1;
            self.submitted_steps += 1;
            let request = self
                .request_mut(prepared.delta.request)
                .expect("batch preflight retained request");
            request.pending_step = None;
            request.inflight_submission = Some(*submission);
        }
        #[cfg(test)]
        {
            self.hot_path.delta_entries_touched += delta_entries_touched;
        }
        Ok(plans
            .into_iter()
            .map(|(_, _, _, output)| output)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Atomically publishes an ordered submission batch at one shared backend
    /// completion point.
    ///
    /// Submission identities are authoritative and derive their requests.
    /// Every root, page pin, retirement, and reclamation slot is preflighted
    /// before any publication occurs.
    ///
    /// # Errors
    ///
    /// Any invalid completion event or submission rejects the whole batch with
    /// no published view, page phase, operation, or reclamation mutation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn complete_batch(
        &mut self,
        receipt: BatchCompletionReceipt,
        submissions: &[SubmissionLease],
    ) -> Result<Box<[StepCompletion]>, KvManagerError> {
        if submissions.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        if receipt.reserved != 0 {
            return Err(KvManagerError::ReservedFieldNonZero);
        }
        if receipt.confirmed != 1 {
            return Err(KvManagerError::CompletionNotConfirmed);
        }
        if receipt.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }

        let mut seen_submissions = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut total_retirements = 0_usize;
        let mut prelim = Vec::with_capacity(submissions.len());
        for &submission in submissions {
            if !seen_submissions.insert(submission) {
                return Err(KvManagerError::DuplicateSubmission);
            }
            self.check_submission_epoch(submission)?;
            let submitted = match self
                .operations
                .get(submission.slot, submission.generation)?
            {
                OperationState::Submitted(submitted) => submitted.clone(),
                OperationState::Prepared(_) => return Err(KvManagerError::StepNotSubmitted),
            };
            let delta = &submitted.delta;
            if !seen_requests.insert(delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request_state = self.request(delta.request)?;
            if request_state.inflight_submission != Some(submission)
                || request_state.snapshot.view_version != delta.base_view_version
                || request_state.snapshot.boundary != delta.previous_boundary
                || request_state.snapshot.roots.len() != self.classes.len()
                || delta.classes.len() != self.classes.len()
            {
                return Err(KvManagerError::StaleView);
            }
            self.preflight_submitted_delta(submission, &submitted)?;
            for class_delta in &delta.classes {
                for entry in class_delta
                    .previous_tail
                    .iter()
                    .chain(class_delta.writes.iter())
                {
                    if !seen_pages.insert(entry.page.page_id) {
                        return Err(KvManagerError::DuplicatePage);
                    }
                }
            }

            let mut transitions = Vec::with_capacity(self.classes.len());
            let mut retire_entries = Vec::new();
            let mut resident_count = 0_usize;
            for ((class, root), class_delta) in self
                .classes
                .iter()
                .copied()
                .zip(request_state.snapshot.roots.iter())
                .zip(delta.classes.iter())
            {
                if class_delta.class_id != class.class_id {
                    return Err(KvManagerError::Invariant("delta class ordering"));
                }
                if let (Some(front), Some(back)) = (root.entries.front(), root.entries.back()) {
                    let expected_len = back
                        .logical_ordinal
                        .checked_sub(front.logical_ordinal)
                        .and_then(|span| span.checked_add(1))
                        .and_then(|span| usize::try_from(span).ok())
                        .ok_or(KvManagerError::Invariant("snapshot root span"))?;
                    if front.class_id != class.class_id
                        || back.class_id != class.class_id
                        || expected_len != root.entries.len()
                    {
                        return Err(KvManagerError::Invariant("snapshot root continuity"));
                    }
                }
                if let (Some(front), Some(back)) =
                    (class_delta.writes.first(), class_delta.writes.last())
                {
                    let expected_len = back
                        .logical_ordinal
                        .checked_sub(front.logical_ordinal)
                        .and_then(|span| span.checked_add(1))
                        .and_then(|span| usize::try_from(span).ok())
                        .ok_or(KvManagerError::Invariant("delta write span"))?;
                    if front.class_id != class.class_id
                        || back.class_id != class.class_id
                        || expected_len != class_delta.writes.len()
                    {
                        return Err(KvManagerError::Invariant("delta write continuity"));
                    }
                }
                if let (Some(back), Some(front)) = (root.entries.back(), class_delta.writes.first())
                    && back.logical_ordinal.checked_add(1) != Some(front.logical_ordinal)
                {
                    return Err(KvManagerError::StaleView);
                }
                let candidate_first = root
                    .entries
                    .front()
                    .or_else(|| class_delta.writes.first())
                    .map(|entry| entry.logical_ordinal)
                    .ok_or(KvManagerError::Invariant("empty append candidate"))?;
                let expected_first =
                    class.candidate_start(delta.previous_boundary) / self.page_tokens;
                let candidate_end = delta.target_boundary.div_ceil(self.page_tokens);
                let candidate_len = root
                    .entries
                    .len()
                    .checked_add(class_delta.writes.len())
                    .ok_or(KvManagerError::ArithmeticOverflow("candidate root length"))?;
                let expected_len = candidate_end
                    .checked_sub(candidate_first)
                    .and_then(|span| usize::try_from(span).ok())
                    .ok_or(KvManagerError::Invariant("candidate root span"))?;
                if candidate_first != expected_first || candidate_len != expected_len {
                    return Err(KvManagerError::StaleView);
                }
                let retain_first_ordinal = match class.retention {
                    RetentionKind::Full => candidate_first,
                    RetentionKind::Sliding => (class.retained_start(delta.target_boundary)
                        / self.page_tokens)
                        .min(candidate_end),
                    RetentionKind::Chunked => {
                        unreachable!("canonical profile rejects chunked retention")
                    }
                };
                let retire_count = usize::try_from(
                    retain_first_ordinal
                        .checked_sub(candidate_first)
                        .ok_or(KvManagerError::Invariant("retained root prefix"))?,
                )
                .map_err(|_| KvManagerError::ArithmeticOverflow("retirement count"))?;
                if retire_count > candidate_len {
                    return Err(KvManagerError::Invariant("retirement exceeds candidate"));
                }
                let retire_from_root = retire_count.min(root.entries.len());
                let retire_from_writes = retire_count - retire_from_root;
                for entry in root.entries.iter().take(retire_from_root) {
                    let is_tail = class_delta
                        .previous_tail
                        .is_some_and(|tail| tail.page == entry.page);
                    if !is_tail {
                        if !seen_pages.insert(entry.page.page_id) {
                            return Err(KvManagerError::DuplicatePage);
                        }
                        self.preflight_active_root_entry(delta.request, *entry)?;
                    }
                    retire_entries.push(*entry);
                }
                retire_entries.extend(class_delta.writes.iter().take(retire_from_writes).copied());
                let class_resident = candidate_len - retire_count;
                resident_count = resident_count.checked_add(class_resident).ok_or(
                    KvManagerError::ArithmeticOverflow("published resident count"),
                )?;
                transitions.push(ClassTransition {
                    retire_from_root,
                    retire_from_writes,
                    retain_first_ordinal,
                    resident_count: class_resident,
                });
            }
            request_state
                .pending_reclamations
                .checked_add(retire_entries.len() as u64)
                .ok_or(KvManagerError::ArithmeticOverflow(
                    "request pending reclamations",
                ))?;
            total_retirements = total_retirements
                .checked_add(retire_entries.len())
                .ok_or(KvManagerError::ArithmeticOverflow("batch retirements"))?;
            prelim.push((
                submission,
                submitted,
                transitions,
                retire_entries,
                resident_count,
            ));
        }
        let planned_reclamations = self.reclamations.plan_many(total_retirements)?;
        let mut reclamation_cursor = 0_usize;
        let mut plans = Vec::with_capacity(prelim.len());
        for (submission, submitted, transitions, retire_entries, resident_count) in prelim {
            let retirement_end = reclamation_cursor.checked_add(retire_entries.len()).ok_or(
                KvManagerError::ArithmeticOverflow("batch reclamation range"),
            )?;
            let item_slots = planned_reclamations
                .get(reclamation_cursor..retirement_end)
                .ok_or(KvManagerError::Invariant("batch reclamation cardinality"))?;
            let certificates = retire_entries
                .iter()
                .zip(item_slots.iter().copied())
                .map(|(entry, planned)| {
                    self.certificate_for_root(
                        submitted.delta.request,
                        *entry,
                        submitted.delta.target_boundary,
                        planned,
                        receipt.completion_domain,
                        receipt.completion_value,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            reclamation_cursor = retirement_end;
            let resident_count = u32::try_from(resident_count)
                .map_err(|_| KvManagerError::ArithmeticOverflow("published resident count"))?;
            let output = StepCompletion {
                submission,
                request: submitted.delta.request,
                publication: PublishedReceipt {
                    view_version: submitted.delta.target_view_version,
                    boundary: submitted.delta.target_boundary,
                    resident_count,
                },
                retirements: certificates.clone().into_boxed_slice(),
            };
            plans.push((
                submission,
                submitted,
                transitions,
                retire_entries,
                certificates,
                output,
            ));
        }
        debug_assert_eq!(reclamation_cursor, planned_reclamations.len());

        // Capacity growth is the only potentially allocating part of root
        // publication. Do it for every request after collective semantic
        // preflight and before any logical page/root mutation.
        for (_, submitted, transitions, _, _, _) in &plans {
            let request = self
                .request_mut(submitted.delta.request)
                .expect("batch completion preflight retained request");
            for ((root, class_delta), transition) in request
                .snapshot
                .roots
                .iter_mut()
                .zip(submitted.delta.classes.iter())
                .zip(transitions)
            {
                root.entries.reserve(
                    class_delta
                        .writes
                        .len()
                        .saturating_sub(transition.retire_from_writes),
                );
            }
        }

        let mut slot_cursor = 0_usize;
        #[cfg(test)]
        let mut delta_entries_touched = 0_u64;
        #[cfg(test)]
        let mut retirement_entries_touched = 0_u64;
        #[cfg(test)]
        let mut hot_root_entries_visited = 0_u64;
        for (submission, submitted, transitions, retire_entries, certificates, _) in &plans {
            for (class_delta, transition) in submitted.delta.classes.iter().zip(transitions.iter())
            {
                for entry in class_delta
                    .previous_tail
                    .iter()
                    .chain(class_delta.writes.iter())
                {
                    self.page_mut(entry.page.page_id)
                        .expect("batch preflight validated submitted write")
                        .readers -= 1;
                    if entry.logical_ordinal >= transition.retain_first_ordinal {
                        self.set_page_phase(
                            entry.page.page_id,
                            PagePhase::Active(ActiveBinding {
                                request: submitted.delta.request,
                                class_id: entry.class_id,
                                logical_ordinal: entry.logical_ordinal,
                            }),
                        )
                        .expect("batch preflight retained submitted write");
                    }
                }
                #[cfg(test)]
                {
                    delta_entries_touched += u64::from(class_delta.previous_tail.is_some())
                        + class_delta.writes.len() as u64;
                }
            }
            for (entry, certificate) in retire_entries.iter().zip(certificates) {
                debug_assert_eq!(
                    self.page(entry.page.page_id)
                        .expect("batch preflight validated retiring page")
                        .readers,
                    0
                );
                self.set_page_phase(
                    entry.page.page_id,
                    PagePhase::Retiring {
                        reclamation: certificate.reclamation,
                    },
                )
                .expect("batch preflight retained retiring page");
            }
            for certificate in certificates {
                let planned = planned_reclamations[slot_cursor];
                slot_cursor += 1;
                self.reclamations.insert_planned(
                    planned,
                    ReclamationState {
                        certificate: certificate.clone(),
                    },
                );
            }
            let request = self
                .request_mut(submitted.delta.request)
                .expect("batch preflight retained request");
            for ((root, class_delta), transition) in request
                .snapshot
                .roots
                .iter_mut()
                .zip(submitted.delta.classes.iter())
                .zip(transitions)
            {
                for _ in 0..transition.retire_from_root {
                    root.entries
                        .pop_front()
                        .expect("completion preflight validated root retirement");
                }
                root.entries.extend(
                    class_delta
                        .writes
                        .iter()
                        .skip(transition.retire_from_writes)
                        .copied(),
                );
                debug_assert_eq!(root.entries.len(), transition.resident_count);
            }
            request.snapshot.boundary = submitted.delta.target_boundary;
            request.snapshot.view_version = submitted.delta.target_view_version;
            request.inflight_submission = None;
            request.last_completion_domain = receipt.completion_domain;
            request.last_completion_value = receipt.completion_value;
            request.pending_reclamations += certificates.len() as u64;
            self.operations
                .remove(submission.slot, submission.generation)
                .expect("batch preflight retained submitted operation");
            self.submitted_steps -= 1;
            #[cfg(test)]
            {
                retirement_entries_touched += retire_entries.len() as u64;
                hot_root_entries_visited += transitions
                    .iter()
                    .map(|transition| transition.retire_from_root as u64)
                    .sum::<u64>();
            }
        }
        #[cfg(test)]
        {
            self.hot_path.delta_entries_touched += delta_entries_touched;
            self.hot_path.retirement_entries_touched += retirement_entries_touched;
            self.hot_path.hot_root_entries_visited += hot_root_entries_visited;
        }
        Ok(plans
            .into_iter()
            .map(|(_, _, _, _, _, output)| output)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Atomically aborts a non-empty prepared batch proven backend-unobserved.
    ///
    /// # Errors
    ///
    /// Any missing proof, duplicate, stale step, or stale page rejects the
    /// whole batch without mutation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn abort_steps(
        &mut self,
        receipts: &[BackendUnobservedReceipt],
    ) -> Result<(), KvManagerError> {
        if receipts.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_steps = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut plans = Vec::with_capacity(receipts.len());
        for &receipt in receipts {
            if receipt.reserved != 0 {
                return Err(KvManagerError::ReservedFieldNonZero);
            }
            if receipt.backend_unobserved != 1 {
                return Err(KvManagerError::BackendObservationUnknown);
            }
            if !seen_steps.insert(receipt.step) {
                return Err(KvManagerError::DuplicateStep);
            }
            self.check_step_epoch(receipt.step)?;
            let prepared = match self
                .operations
                .get(receipt.step.slot, receipt.step.generation)?
            {
                OperationState::Prepared(prepared) => prepared.clone(),
                OperationState::Submitted(_) => return Err(KvManagerError::StepAlreadySubmitted),
            };
            if !seen_requests.insert(prepared.delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request = self.request(prepared.delta.request)?;
            if request.pending_step != Some(receipt.step) {
                return Err(KvManagerError::StaleView);
            }
            let reserved = prepared
                .delta
                .classes
                .iter()
                .flat_map(|class| class.writes.iter())
                .map(|entry| entry.page.page_id)
                .collect::<Vec<_>>();
            for &page_id in &reserved {
                if !seen_pages.insert(page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let page = self.page(page_id)?;
                if page.phase != (PagePhase::Reserved { step: receipt.step }) || page.readers != 0 {
                    return Err(KvManagerError::StalePage);
                }
            }
            plans.push((receipt.step, prepared.delta.request, reserved));
        }
        let mut recycled_by_class = vec![Vec::<u32>::new(); self.classes.len()];
        for (step, request, reserved) in plans {
            for page_id in reserved {
                let (class_id, generation) = {
                    let page = self
                        .page(page_id)
                        .expect("batch abort preflight retained reserved page");
                    (page.class_id, page.generation)
                };
                if generation == u64::MAX {
                    self.set_page_phase(page_id, PagePhase::Exhausted)
                        .expect("batch abort preflight retained reserved page");
                } else {
                    self.set_page_phase(page_id, PagePhase::Free)
                        .expect("batch abort preflight retained reserved page");
                    recycled_by_class[usize::from(class_id)].push(page_id);
                }
            }
            self.operations
                .remove(step.slot, step.generation)
                .expect("batch abort preflight retained operation");
            self.prepared_steps -= 1;
            self.request_mut(request)
                .expect("batch abort preflight retained request")
                .pending_step = None;
        }
        for (free, mut recycled) in self.free_pages.iter_mut().zip(recycled_by_class) {
            recycled.sort_unstable_by(|left, right| right.cmp(left));
            free.extend(recycled);
        }
        Ok(())
    }

    /// Atomically fail-stops an ordered prepared batch after ambiguous backend
    /// lowering.
    ///
    /// # Errors
    ///
    /// Any duplicate, stale, or submitted identity rejects the whole call
    /// before quarantine begins.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn quarantine_steps(&mut self, steps: &[StepLease]) -> Result<(), KvManagerError> {
        if steps.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_steps = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut plans = Vec::with_capacity(steps.len());
        for &step in steps {
            if !seen_steps.insert(step) {
                return Err(KvManagerError::DuplicateStep);
            }
            self.check_step_epoch(step)?;
            let prepared = match self.operations.get(step.slot, step.generation)? {
                OperationState::Prepared(prepared) => prepared.clone(),
                OperationState::Submitted(_) => return Err(KvManagerError::StepAlreadySubmitted),
            };
            if !seen_requests.insert(prepared.delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request = self.request(prepared.delta.request)?;
            if request.pending_step != Some(step) {
                return Err(KvManagerError::StaleView);
            }
            let reserved = prepared
                .delta
                .classes
                .iter()
                .flat_map(|class| class.writes.iter())
                .map(|entry| entry.page.page_id)
                .collect::<Vec<_>>();
            for &page_id in &reserved {
                if !seen_pages.insert(page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let page = self.page(page_id)?;
                if page.phase != (PagePhase::Reserved { step }) || page.readers != 0 {
                    return Err(KvManagerError::StalePage);
                }
            }
            plans.push((step, prepared.delta.request, reserved));
        }
        for (step, request, reserved) in plans {
            for page_id in reserved {
                self.set_page_phase(page_id, PagePhase::Quarantined)
                    .expect("batch quarantine preflight retained page");
            }
            self.operations
                .remove(step.slot, step.generation)
                .expect("batch quarantine preflight retained operation");
            self.prepared_steps -= 1;
            let request = self
                .request_mut(request)
                .expect("batch quarantine preflight retained request");
            request.pending_step = None;
            request.quarantined = true;
        }
        Ok(())
    }

    /// Atomically fail-stops every page reachable by an ordered ambiguous
    /// submission batch.
    ///
    /// # Errors
    ///
    /// Any duplicate, stale, or unsubmitted identity rejects the whole call
    /// before quarantine begins.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn quarantine_submissions(
        &mut self,
        submissions: &[SubmissionLease],
    ) -> Result<(), KvManagerError> {
        if submissions.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_submissions = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut plans = Vec::with_capacity(submissions.len());
        for &submission in submissions {
            if !seen_submissions.insert(submission) {
                return Err(KvManagerError::DuplicateSubmission);
            }
            self.check_submission_epoch(submission)?;
            let submitted = match self
                .operations
                .get(submission.slot, submission.generation)?
            {
                OperationState::Submitted(submitted) => submitted.clone(),
                OperationState::Prepared(_) => return Err(KvManagerError::StepNotSubmitted),
            };
            if !seen_requests.insert(submitted.delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request = self.request(submitted.delta.request)?;
            if request.inflight_submission != Some(submission) {
                return Err(KvManagerError::StaleView);
            }
            let mut page_ids = request
                .snapshot
                .roots
                .iter()
                .flat_map(|root| root.entries.iter())
                .map(|entry| entry.page.page_id)
                .collect::<Vec<_>>();
            page_ids.extend(
                submitted
                    .delta
                    .classes
                    .iter()
                    .flat_map(|class| class.writes.iter())
                    .map(|entry| entry.page.page_id),
            );
            for &page_id in &page_ids {
                if !seen_pages.insert(page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                self.page(page_id)?;
            }
            plans.push((submission, submitted.delta.request, page_ids));
        }
        for (submission, request, page_ids) in plans {
            for page_id in page_ids {
                self.set_page_phase(page_id, PagePhase::Quarantined)
                    .expect("batch quarantine preflight retained page");
            }
            let request = self
                .request_mut(request)
                .expect("batch quarantine preflight retained request");
            request.inflight_submission = None;
            request.quarantined = true;
            self.operations
                .remove(submission.slot, submission.generation)
                .expect("batch quarantine preflight retained operation");
            self.submitted_steps -= 1;
        }
        Ok(())
    }

    /// Atomically releases an ordered batch of quiescent requests.
    ///
    /// # Errors
    ///
    /// Any duplicate, unavailable, busy, stale, or under-provisioned item
    /// rejects the entire batch without changing a root or page phase.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn release_batch(
        &mut self,
        requests: &[RequestLease],
    ) -> Result<Box<[ReleasedBatchItem]>, KvManagerError> {
        if requests.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut total_retirements = 0_usize;
        let mut states = Vec::with_capacity(requests.len());
        for &request in requests {
            if !seen_requests.insert(request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let state = self.request(request)?;
            if state.released || state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if state.pending_step.is_some() || state.inflight_submission.is_some() {
                return Err(KvManagerError::RequestBusy);
            }
            let released_version = ViewVersion(
                state
                    .snapshot
                    .view_version
                    .0
                    .checked_add(1)
                    .ok_or(KvManagerError::ViewVersionExhausted)?,
            );
            let resident_count = state.snapshot.resident_count();
            state
                .pending_reclamations
                .checked_add(resident_count as u64)
                .ok_or(KvManagerError::ArithmeticOverflow(
                    "request pending reclamations",
                ))?;
            let entries = state
                .snapshot
                .roots
                .iter()
                .flat_map(|root| root.entries.iter().copied())
                .collect::<Vec<_>>();
            for entry in &entries {
                if !seen_pages.insert(entry.page.page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let page = self.page(entry.page.page_id)?;
                if page.readers != 0
                    || page.phase
                        != PagePhase::Active(ActiveBinding {
                            request,
                            class_id: entry.class_id,
                            logical_ordinal: entry.logical_ordinal,
                        })
                {
                    return Err(KvManagerError::StalePage);
                }
            }
            total_retirements = total_retirements.checked_add(resident_count).ok_or(
                KvManagerError::ArithmeticOverflow("batch release retirements"),
            )?;
            states.push((
                request,
                state.snapshot.boundary,
                released_version,
                state.last_completion_domain,
                state.last_completion_value,
                entries,
            ));
        }
        let planned = self.reclamations.plan_many(total_retirements)?;
        let mut planned_cursor = 0_usize;
        let mut retirement_offset = 0_u32;
        let mut plans = Vec::with_capacity(states.len());
        for (request, boundary, released_version, domain, value, entries) in states {
            let end = planned_cursor
                .checked_add(entries.len())
                .ok_or(KvManagerError::ArithmeticOverflow("batch release range"))?;
            let slots = planned
                .get(planned_cursor..end)
                .ok_or(KvManagerError::Invariant("batch release cardinality"))?;
            let certificates = entries
                .iter()
                .copied()
                .zip(slots.iter().copied())
                .map(|(entry, slot)| {
                    self.certificate_for_root(request, entry, boundary, slot, domain, value)
                })
                .collect::<Result<Vec<_>, KvManagerError>>()?;
            planned_cursor = end;
            let output = ReleasedBatchItem {
                release: ReleaseCompletion {
                    request,
                    retirements: certificates.clone().into_boxed_slice(),
                },
                retirement_offset,
            };
            retirement_offset =
                retirement_offset
                    .checked_add(u32::try_from(certificates.len()).map_err(|_| {
                        KvManagerError::ArithmeticOverflow("batch release retirements")
                    })?)
                    .ok_or(KvManagerError::ArithmeticOverflow(
                        "batch release retirements",
                    ))?;
            plans.push((request, released_version, certificates, output));
        }
        debug_assert_eq!(planned_cursor, planned.len());

        let mut slot_cursor = 0_usize;
        for (request, released_version, certificates, _) in &plans {
            for certificate in certificates {
                self.set_page_phase(
                    certificate.page.page_id,
                    PagePhase::Retiring {
                        reclamation: certificate.reclamation,
                    },
                )
                .expect("batch release preflight retained page");
                let slot = planned[slot_cursor];
                slot_cursor += 1;
                self.reclamations.insert_planned(
                    slot,
                    ReclamationState {
                        certificate: certificate.clone(),
                    },
                );
            }
            let request_state = self
                .request_mut(*request)
                .expect("batch release preflight retained request");
            for root in &mut request_state.snapshot.roots {
                root.entries.clear();
            }
            request_state.released = true;
            request_state.snapshot.view_version = *released_version;
            request_state.pending_reclamations += certificates.len() as u64;
        }
        Ok(plans
            .into_iter()
            .map(|(_, _, _, output)| output)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Atomically acknowledges a complete reclamation batch.
    ///
    /// # Errors
    ///
    /// Returns an error before mutation when any receipt is stale, duplicated,
    /// unacknowledged, or does not exactly match its certificate.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn acknowledge_reclamations(
        &mut self,
        receipts: &[ReclamationReceipt],
    ) -> Result<(), KvManagerError> {
        if receipts.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        let mut states = Vec::with_capacity(receipts.len());
        let mut acknowledged_by_request = BTreeMap::<RequestLease, u64>::new();
        for receipt in receipts {
            if receipt.reserved8 != 0 || receipt.reserved32 != 0 {
                return Err(KvManagerError::ReservedFieldNonZero);
            }
            if receipt.acknowledged != 1 {
                return Err(KvManagerError::ReclamationNotAcknowledged);
            }
            self.check_reclamation_epoch(receipt.reclamation)?;
            if !seen.insert(receipt.reclamation) {
                return Err(KvManagerError::DuplicateReclamation);
            }
            let state = self
                .reclamations
                .get(receipt.reclamation.slot, receipt.reclamation.generation)?;
            let certificate = &state.certificate;
            if certificate.reclamation != receipt.reclamation
                || certificate.page != receipt.page
                || certificate.backend_domain != receipt.backend_domain
                || certificate.backend_index != receipt.backend_index
            {
                return Err(KvManagerError::ReclamationMismatch);
            }
            let page = self.page(receipt.page.page_id)?;
            if page.readers != 0
                || page.generation != receipt.page.generation
                || page.phase
                    != (PagePhase::Retiring {
                        reclamation: receipt.reclamation,
                    })
            {
                return Err(KvManagerError::StalePage);
            }
            *acknowledged_by_request
                .entry(certificate.request)
                .or_default() += 1;
            states.push(receipt.reclamation);
        }
        for (&request, &count) in &acknowledged_by_request {
            if self.request(request)?.pending_reclamations < count {
                return Err(KvManagerError::Invariant(
                    "request reclamation count underflow",
                ));
            }
        }
        let mut recycled_by_class = vec![Vec::<u32>::new(); self.classes.len()];
        for reclamation in states {
            let certificate = self
                .reclamations
                .remove(reclamation.slot, reclamation.generation)
                .expect("reclamation preflight retained certificate")
                .certificate;
            let page_id = certificate.page.page_id;
            let (generation, class_id) = {
                let page = self
                    .page(page_id)
                    .expect("reclamation preflight retained page");
                debug_assert_eq!(page.readers, 0);
                debug_assert_eq!(page.phase, PagePhase::Retiring { reclamation });
                (page.generation, page.class_id)
            };
            if generation == u64::MAX {
                self.set_page_phase(page_id, PagePhase::Exhausted)
                    .expect("reclamation preflight retained page");
            } else {
                self.set_page_phase(page_id, PagePhase::Free)
                    .expect("reclamation preflight retained page");
                recycled_by_class[usize::from(class_id)].push(page_id);
            }
        }
        for (free, mut recycled) in self.free_pages.iter_mut().zip(recycled_by_class) {
            recycled.sort_unstable_by(|left, right| right.cmp(left));
            free.extend(recycled);
        }
        for (request, count) in acknowledged_by_request {
            self.request_mut(request)
                .expect("reclamation preflight retained request")
                .pending_reclamations -= count;
        }
        Ok(())
    }

    /// Atomically recycles a non-empty batch of fully released identities.
    ///
    /// # Errors
    ///
    /// Any duplicate, stale, or non-recyclable request rejects the whole
    /// batch without advancing any request generation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn recycle_requests(&mut self, requests: &[RequestLease]) -> Result<(), KvManagerError> {
        if requests.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        for &request in requests {
            if !seen.insert(request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let state = self.request(request)?;
            if !state.released
                || state.pending_step.is_some()
                || state.inflight_submission.is_some()
                || !state.snapshot.is_empty()
                || state.pending_reclamations != 0
            {
                return Err(KvManagerError::RequestNotRecyclable);
            }
        }
        for &request in requests {
            self.requests
                .remove(request.slot, request.generation)
                .expect("batch recycle preflight retained request");
        }
        Ok(())
    }

    #[must_use]
    pub fn stats(&self) -> ManagerStats {
        self.page_counts.iter().fold(
            ManagerStats {
                active_requests: self.requests.active_len() as u64,
                prepared_steps: self.prepared_steps,
                submitted_steps: self.submitted_steps,
                pending_reclamations: self.reclamations.active_len() as u64,
                ..ManagerStats::default()
            },
            |mut stats, counts| {
                stats.free_pages += counts.free;
                stats.reserved_pages += counts.reserved;
                stats.writing_pages += counts.writing;
                stats.active_pages += counts.active;
                stats.retiring_pages += counts.retiring;
                stats.quarantined_pages += counts.quarantined;
                stats.exhausted_pages += counts.exhausted;
                stats
            },
        )
    }

    /// Returns an immutable per-class physical-page census in class-id order.
    #[must_use]
    pub fn arena_stats(&self) -> Box<[ArenaStats]> {
        self.classes
            .iter()
            .zip(&self.page_counts)
            .map(|(class, counts)| ArenaStats {
                engine_epoch: self.engine_epoch,
                pool_epoch: self.pool_epoch,
                class_id: class.class_id,
                backend_domain: class.backend.backend_domain,
                pool_id: class.backend.pool_id,
                page_count: class.backend.page_count,
                first_page_id: class.first_page_id,
                free_pages: counts.free,
                reserved_pages: counts.reserved,
                writing_pages: counts.writing,
                active_pages: counts.active,
                retiring_pages: counts.retiring,
                quarantined_pages: counts.quarantined,
                exhausted_pages: counts.exhausted,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn runtime_class(&self, class_id: u16) -> Result<RuntimeClass, KvManagerError> {
        self.classes
            .get(usize::from(class_id))
            .copied()
            .filter(|class| class.class_id == class_id)
            .ok_or(KvManagerError::InvalidClass(class_id))
    }

    fn request(&self, request: RequestLease) -> Result<&RequestState, KvManagerError> {
        self.check_request_epoch(request)?;
        self.requests.get(request.slot, request.generation)
    }

    fn request_mut(&mut self, request: RequestLease) -> Result<&mut RequestState, KvManagerError> {
        self.check_request_epoch(request)?;
        self.requests.get_mut(request.slot, request.generation)
    }

    fn check_request_epoch(&self, request: RequestLease) -> Result<(), KvManagerError> {
        if request.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    fn check_step_epoch(&self, step: StepLease) -> Result<(), KvManagerError> {
        if step.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    fn check_submission_epoch(&self, submission: SubmissionLease) -> Result<(), KvManagerError> {
        if submission.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    fn check_reclamation_epoch(&self, reclamation: ReclamationLease) -> Result<(), KvManagerError> {
        if reclamation.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    fn page(&self, page_id: u32) -> Result<&PageState, KvManagerError> {
        let index = page_id
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(KvManagerError::InvalidPage(page_id))?;
        self.pages
            .get(index)
            .ok_or(KvManagerError::InvalidPage(page_id))
    }

    fn page_mut(&mut self, page_id: u32) -> Result<&mut PageState, KvManagerError> {
        let index = page_id
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(KvManagerError::InvalidPage(page_id))?;
        self.pages
            .get_mut(index)
            .ok_or(KvManagerError::InvalidPage(page_id))
    }

    fn set_page_phase(&mut self, page_id: u32, target: PagePhase) -> Result<(), KvManagerError> {
        let (class_id, previous) = {
            let page = self.page(page_id)?;
            (page.class_id, page.phase)
        };
        if std::mem::discriminant(&previous) != std::mem::discriminant(&target) {
            let counts = &mut self.page_counts[usize::from(class_id)];
            counts.decrement(previous);
            counts.increment(target);
        }
        self.page_mut(page_id)?.phase = target;
        Ok(())
    }

    fn apply_page_reservations(&mut self, step: StepLease, pages: &[PageLease]) {
        for lease in pages {
            let class_id = self
                .page(lease.page_id)
                .expect("planned page id remains valid")
                .class_id;
            let popped = self.free_pages[usize::from(class_id)].pop();
            assert_eq!(
                popped,
                Some(lease.page_id),
                "planned page remains class stack head"
            );
            let page = self
                .page_mut(lease.page_id)
                .expect("planned page id remains valid");
            debug_assert_eq!(page.phase, PagePhase::Free);
            debug_assert_eq!(page.generation.checked_add(1), Some(lease.generation));
            page.generation = lease.generation;
            self.set_page_phase(lease.page_id, PagePhase::Reserved { step })
                .expect("planned page remains valid");
        }
    }

    fn root_entry_for_page(
        &self,
        class: RuntimeClass,
        ordinal: u64,
        page: PageLease,
    ) -> Result<RootEntry, KvManagerError> {
        self.validate_page_lease(class, page)?;
        let (cell, cycle) = class.temporal_address(ordinal);
        let backend_index = class.backend_index(page.page_id)?;
        Ok(RootEntry {
            class_id: class.class_id,
            backend_domain: class.backend.backend_domain,
            logical_ordinal: ordinal,
            temporal_cell_index: cell,
            temporal_cycle: cycle,
            page,
            backend_index,
        })
    }

    fn validate_page_lease(
        &self,
        class: RuntimeClass,
        page: PageLease,
    ) -> Result<(), KvManagerError> {
        if page.engine_epoch != self.engine_epoch
            || page.pool_epoch != self.pool_epoch
            || page.pool_id != class.backend.pool_id
            || !class.contains_page(page.page_id)
            || self.page(page.page_id)?.class_id != class.class_id
        {
            return Err(KvManagerError::WrongPageArena);
        }
        Ok(())
    }

    #[cfg(test)]
    fn device_entry(
        &self,
        root: RootEntry,
        access_flags: u32,
        valid: u64,
        visible_offset: u64,
        visible: u64,
    ) -> Result<DeviceKvEntry, KvManagerError> {
        let token_begin = root
            .logical_ordinal
            .checked_mul(self.page_tokens)
            .ok_or(KvManagerError::ArithmeticOverflow("entry token begin"))?;
        Ok(DeviceKvEntry {
            class_id: root.class_id,
            backend_domain: root.backend_domain,
            access_flags,
            logical_ordinal: root.logical_ordinal,
            token_begin,
            valid_token_count: u32::try_from(valid)
                .map_err(|_| KvManagerError::ArithmeticOverflow("valid token count"))?,
            visible_token_offset: u32::try_from(visible_offset)
                .map_err(|_| KvManagerError::ArithmeticOverflow("visible token offset"))?,
            visible_token_count: u32::try_from(visible)
                .map_err(|_| KvManagerError::ArithmeticOverflow("visible token count"))?,
            pool_id: root.page.pool_id,
            temporal_cell_index: root.temporal_cell_index,
            temporal_cycle: root.temporal_cycle,
            pool_epoch: root.page.pool_epoch,
            page_generation: root.page.generation,
            backend_index: root.backend_index,
            page_id: root.page.page_id,
            reserved: 0,
        })
    }

    fn validate_bind_receipts(
        step: StepLease,
        prepared: &PreparedState,
        receipts: &[BackendBindReceipt],
    ) -> Result<(), KvManagerError> {
        let mut expected = BTreeMap::new();
        for entry in prepared
            .delta
            .classes
            .iter()
            .flat_map(|class| class.writes.iter())
        {
            if expected.insert(entry.page.page_id, entry).is_some() {
                return Err(KvManagerError::DuplicatePage);
            }
        }
        if receipts.len() != expected.len() {
            return Err(KvManagerError::BindingReceiptMismatch);
        }
        let mut seen = BTreeSet::new();
        for receipt in receipts {
            if receipt.reserved != 0 {
                return Err(KvManagerError::ReservedFieldNonZero);
            }
            if receipt.step != step || receipt.mapped != 1 || receipt.writable != 1 {
                return Err(KvManagerError::BindingReceiptMismatch);
            }
            if !seen.insert(receipt.page.page_id) {
                return Err(KvManagerError::DuplicateBindingReceipt);
            }
            let entry = expected
                .get(&receipt.page.page_id)
                .copied()
                .ok_or(KvManagerError::BindingReceiptMismatch)?;
            if receipt.page != entry.page
                || receipt.backend_domain != entry.backend_domain
                || receipt.backend_index != entry.backend_index
            {
                return Err(KvManagerError::BindingReceiptMismatch);
            }
        }
        Ok(())
    }

    fn preflight_prepared_delta(
        &self,
        prepared: &PreparedState,
        step: StepLease,
    ) -> Result<(), KvManagerError> {
        let mut seen = BTreeSet::new();
        let delta = &prepared.delta;
        let request = self.request(delta.request)?;
        if delta.classes.len() != self.classes.len()
            || request.snapshot.roots.len() != self.classes.len()
        {
            return Err(KvManagerError::StaleView);
        }
        let first_new = delta.previous_boundary.div_ceil(self.page_tokens);
        let new_end = delta.target_boundary.div_ceil(self.page_tokens);
        for (((class, class_delta), root), class_index) in self
            .classes
            .iter()
            .copied()
            .zip(delta.classes.iter())
            .zip(request.snapshot.roots.iter())
            .zip(0_usize..)
        {
            if class_delta.class_id != class.class_id
                || class_index != usize::from(class.class_id)
                || class_delta.writes.len()
                    != usize::try_from(new_end - first_new)
                        .map_err(|_| KvManagerError::ArithmeticOverflow("class write count"))?
            {
                return Err(KvManagerError::Invariant("delta class shape"));
            }
            let expected_tail = if delta.previous_boundary.is_multiple_of(self.page_tokens) {
                None
            } else {
                root.entries.back().copied()
            };
            if class_delta.previous_tail != expected_tail
                || class_delta.previous_tail.is_some_and(|entry| {
                    entry.logical_ordinal != delta.previous_boundary / self.page_tokens
                        || entry.class_id != class.class_id
                })
            {
                return Err(KvManagerError::StaleView);
            }
            if let Some(entry) = class_delta.previous_tail {
                if !seen.insert(entry.page.page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                self.validate_page_lease(class, entry.page)?;
                let page = self.page(entry.page.page_id)?;
                let target = ActiveBinding {
                    request: delta.request,
                    class_id: entry.class_id,
                    logical_ordinal: entry.logical_ordinal,
                };
                if page.generation != entry.page.generation
                    || page.readers != 0
                    || page.phase != PagePhase::Active(target)
                {
                    return Err(KvManagerError::StalePage);
                }
                if page.readers == u32::MAX {
                    return Err(KvManagerError::ReaderCountOverflow(entry.page.page_id));
                }
            }
            for (offset, entry) in class_delta.writes.iter().enumerate() {
                if !seen.insert(entry.page.page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let expected_ordinal = first_new
                    .checked_add(offset as u64)
                    .ok_or(KvManagerError::ArithmeticOverflow("write ordinal"))?;
                if self.root_entry_for_page(class, expected_ordinal, entry.page)? != *entry {
                    return Err(KvManagerError::StaleView);
                }
                let page = self.page(entry.page.page_id)?;
                if page.generation != entry.page.generation
                    || page.readers != 0
                    || page.phase != (PagePhase::Reserved { step })
                {
                    return Err(KvManagerError::StalePage);
                }
                if page.readers == u32::MAX {
                    return Err(KvManagerError::ReaderCountOverflow(entry.page.page_id));
                }
            }
        }
        Ok(())
    }

    fn preflight_submitted_delta(
        &self,
        submission: SubmissionLease,
        submitted: &SubmittedState,
    ) -> Result<(), KvManagerError> {
        let delta = &submitted.delta;
        if delta.classes.len() != self.classes.len() {
            return Err(KvManagerError::StaleView);
        }
        for (class, class_delta) in self.classes.iter().copied().zip(delta.classes.iter()) {
            if class_delta.class_id != class.class_id {
                return Err(KvManagerError::Invariant("delta class ordering"));
            }
            if let Some(entry) = class_delta.previous_tail {
                self.validate_page_lease(class, entry.page)?;
                let page = self.page(entry.page.page_id)?;
                let expected_target = ActiveBinding {
                    request: delta.request,
                    class_id: entry.class_id,
                    logical_ordinal: entry.logical_ordinal,
                };
                if page.generation != entry.page.generation
                    || page.readers != 1
                    || !matches!(
                    page.phase,
                    PagePhase::Writing {
                        submission: owner,
                        previous,
                        target,
                    } if owner == submission
                        && previous == Some(expected_target)
                        && target == expected_target
                    )
                {
                    return Err(KvManagerError::StalePage);
                }
            }
            for entry in class_delta.writes.iter() {
                self.validate_page_lease(class, entry.page)?;
                let page = self.page(entry.page.page_id)?;
                let expected_target = ActiveBinding {
                    request: delta.request,
                    class_id: entry.class_id,
                    logical_ordinal: entry.logical_ordinal,
                };
                if page.generation != entry.page.generation
                    || page.readers != 1
                    || !matches!(
                        page.phase,
                        PagePhase::Writing {
                            submission: owner,
                            previous: None,
                            target,
                        } if owner == submission && target == expected_target
                    )
                {
                    return Err(KvManagerError::StalePage);
                }
            }
        }
        Ok(())
    }

    fn preflight_active_root_entry(
        &self,
        request: RequestLease,
        entry: RootEntry,
    ) -> Result<(), KvManagerError> {
        let class = self.runtime_class(entry.class_id)?;
        self.validate_page_lease(class, entry.page)?;
        let page = self.page(entry.page.page_id)?;
        if page.generation != entry.page.generation
            || page.readers != 0
            || page.phase
                != PagePhase::Active(ActiveBinding {
                    request,
                    class_id: entry.class_id,
                    logical_ordinal: entry.logical_ordinal,
                })
        {
            return Err(KvManagerError::StalePage);
        }
        Ok(())
    }

    fn certificate_for_root(
        &self,
        request: RequestLease,
        entry: RootEntry,
        target_boundary: u64,
        planned: (u32, u32),
        completion_domain: u64,
        completion_value: u64,
    ) -> Result<ReclamationCertificate, KvManagerError> {
        let token_begin = entry.logical_ordinal.checked_mul(self.page_tokens).ok_or(
            KvManagerError::ArithmeticOverflow("reclamation token begin"),
        )?;
        let page_end = token_begin
            .checked_add(self.page_tokens)
            .ok_or(KvManagerError::ArithmeticOverflow("reclamation token end"))?;
        let token_end_exclusive = target_boundary.min(page_end);
        if token_end_exclusive <= token_begin {
            return Err(KvManagerError::Invariant("empty reclamation token span"));
        }
        Ok(ReclamationCertificate {
            reclamation: ReclamationLease {
                engine_epoch: self.engine_epoch,
                slot: planned.0,
                generation: planned.1,
            },
            request,
            page: entry.page,
            class_id: entry.class_id,
            backend_domain: entry.backend_domain,
            logical_ordinal: entry.logical_ordinal,
            backend_index: entry.backend_index,
            token_begin,
            token_end_exclusive,
            completion_domain,
            completion_value,
        })
    }
}

fn compile_manager_profile(
    plan: &CompiledKvPlan,
    config: ManagerConfig,
    backends: &[BackendArenaRegistration],
) -> Result<Vec<RuntimeClass>, KvManagerError> {
    if plan.page_tokens != CANONICAL_PAGE_TOKENS {
        return Err(KvManagerError::UnsupportedProfile(
            "page_tokens must equal 16",
        ));
    }
    if plan.classes.is_empty() || plan.classes.len() != backends.len() {
        return Err(KvManagerError::InvalidConfiguration);
    }
    if config.maximum_requests == 0
        || config.maximum_operations == 0
        || config.maximum_reclamations == 0
        || config.maximum_step_tokens == 0
    {
        return Err(KvManagerError::InvalidConfiguration);
    }
    let layout = plan
        .layout_program()
        .map_err(|_| KvManagerError::UnsupportedProfile("invalid compiled layout"))?;
    if layout.classes.len() != plan.classes.len() {
        return Err(KvManagerError::UnsupportedProfile(
            "compiled layout class count mismatch",
        ));
    }
    let mut pool_ids = BTreeSet::new();
    let mut backend_classes = BTreeSet::new();
    let mut backend_ranges = Vec::<(u16, u64, u64)>::with_capacity(backends.len());
    for backend in backends {
        if backend.pool_id == 0
            || backend.page_count == 0
            || backend.reserved != 0
            || !pool_ids.insert(backend.pool_id)
            || !backend_classes.insert(backend.class_id)
            || backend
                .backend_base_index
                .checked_add(u64::from(backend.page_count - 1))
                .is_none()
        {
            return Err(KvManagerError::InvalidConfiguration);
        }
        let last_backend_index = backend
            .backend_base_index
            .checked_add(u64::from(backend.page_count - 1))
            .expect("backend range overflow was rejected");
        if backend_ranges.iter().any(|&(domain, first, last)| {
            domain == backend.backend_domain
                && backend.backend_base_index <= last
                && first <= last_backend_index
        }) {
            return Err(KvManagerError::InvalidConfiguration);
        }
        backend_ranges.push((
            backend.backend_domain,
            backend.backend_base_index,
            last_backend_index,
        ));
    }

    let mut next_page_id = 1_u64;
    let mut runtime = Vec::with_capacity(plan.classes.len());
    for (index, (class, class_layout)) in plan.classes.iter().zip(&layout.classes).enumerate() {
        let class_id = u16::try_from(index).map_err(|_| {
            KvManagerError::UnsupportedProfile("retention class count exceeds u16 class ids")
        })?;
        let backend = *backends
            .iter()
            .find(|backend| backend.class_id == class_id)
            .ok_or(KvManagerError::InvalidConfiguration)?;
        let (window_tokens, period_blocks, minimum_pages) =
            validate_class_program(class, class_layout)?;
        if u64::from(backend.page_count) < minimum_pages {
            return Err(KvManagerError::InvalidConfiguration);
        }
        let first_page_id = u32::try_from(next_page_id)
            .map_err(|_| KvManagerError::ArithmeticOverflow("global page id"))?;
        next_page_id = next_page_id
            .checked_add(u64::from(backend.page_count))
            .ok_or(KvManagerError::ArithmeticOverflow("global page id"))?;
        if next_page_id > u64::from(u32::MAX) + 1 {
            return Err(KvManagerError::ArithmeticOverflow("global page id"));
        }
        runtime.push(RuntimeClass {
            class_id,
            retention: class.spec.retention,
            window_tokens,
            period_blocks,
            backend,
            first_page_id,
        });
    }
    let total_pages = next_page_id - 1;
    if u64::from(config.maximum_reclamations) < total_pages {
        return Err(KvManagerError::InvalidConfiguration);
    }
    Ok(runtime)
}

fn validate_class_program(
    class: &CompiledKvClass,
    class_layout: &ClassLayoutProgram,
) -> Result<(Option<u64>, Option<u64>, u64), KvManagerError> {
    if class.block_domain != BlockDomain::all()
        || class_layout.block_domain != BlockDomain::all()
        || class.kv_head_range.is_some()
        || class.source_state.is_some()
    {
        return Err(KvManagerError::UnsupportedProfile(
            "canonical manager requires whole-domain layer classes",
        ));
    }
    match class.spec.retention {
        RetentionKind::Full => {
            if class.spec.window_tokens.is_some()
                || class.slot_count.is_some()
                || !matches!(class_layout.address, AddressProgram::AppendOnly)
                || !retirement_program_matches(RetentionKind::Full, None, &class_layout.retirement)
            {
                return Err(KvManagerError::UnsupportedProfile(
                    "full retention requires append-only addressing",
                ));
            }
            Ok((None, None, 1))
        }
        RetentionKind::Sliding => {
            let window = class
                .spec
                .window_tokens
                .filter(|window| *window > 0)
                .ok_or(KvManagerError::UnsupportedProfile(
                    "sliding retention requires a positive window",
                ))?;
            let history = window - 1;
            let expected_period = 1_u64
                .checked_add(history / CANONICAL_PAGE_TOKENS)
                .and_then(|blocks| {
                    blocks.checked_add(u64::from(history % CANONICAL_PAGE_TOKENS != 0))
                })
                .ok_or(KvManagerError::ArithmeticOverflow("periodic block count"))?;
            if class.slot_count != Some(expected_period)
                || !matches!(
                    class_layout.address,
                    AddressProgram::Periodic { period_blocks }
                        if period_blocks == expected_period
                )
                || !retirement_program_matches(
                    RetentionKind::Sliding,
                    Some(history),
                    &class_layout.retirement,
                )
            {
                return Err(KvManagerError::UnsupportedProfile(
                    "sliding capacity does not match periodic semantics",
                ));
            }
            Ok((Some(window), Some(expected_period), expected_period))
        }
        RetentionKind::Chunked => Err(KvManagerError::UnsupportedProfile(
            "chunked retention is not implemented by the canonical manager",
        )),
    }
}

fn retirement_program_matches(
    retention: RetentionKind,
    history_tokens: Option<u64>,
    program: &RetirementProgram,
) -> bool {
    match (retention, history_tokens, program) {
        (RetentionKind::Full, None, RetirementProgram::Never) => true,
        (
            RetentionKind::Sliding,
            Some(expected),
            RetirementProgram::BlockEndPlus { offset_tokens },
        ) => *offset_tokens == expected,
        _ => false,
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum KvManagerError {
    #[error("batch must contain at least one item")]
    EmptyBatch,
    #[error("batch contains a duplicate request")]
    DuplicateRequest,
    #[error("batch contains a duplicate step")]
    DuplicateStep,
    #[error("batch contains a duplicate submission")]
    DuplicateSubmission,
    #[error("batch contains an invalid or non-canonical flat-buffer range")]
    InvalidBatchRange,
    #[error("{0} capacity must be positive")]
    ZeroCapacity(&'static str),
    #[error("{0} arena is exhausted")]
    ArenaExhausted(&'static str),
    #[error("arithmetic overflow: {0}")]
    ArithmeticOverflow(&'static str),
    #[error("engine epoch space is exhausted")]
    EngineEpochExhausted,
    #[error("unsupported manager profile: {0}")]
    UnsupportedProfile(&'static str),
    #[error("manager configuration is invalid")]
    InvalidConfiguration,
    #[error("identity belongs to a different manager")]
    WrongEngine,
    #[error("page belongs to a different manager or pool")]
    WrongPageArena,
    #[error("stale {0} lease")]
    StaleLease(&'static str),
    #[error("request is busy")]
    RequestBusy,
    #[error("request is released or quarantined")]
    RequestUnavailable,
    #[error("request is not recyclable")]
    RequestNotRecyclable,
    #[error("target boundary {target} does not advance current boundary {current}")]
    NonMonotonicBoundary { current: u64, target: u64 },
    #[error("step contains {requested} tokens, exceeding maximum {maximum}")]
    StepTooLarge { requested: u64, maximum: u64 },
    #[error("view version space is exhausted")]
    ViewVersionExhausted,
    #[error("physical page capacity is exhausted")]
    PageCapacityExhausted,
    #[error("invalid physical page id {0}")]
    InvalidPage(u32),
    #[error("invalid retention class id {0}")]
    InvalidClass(u16),
    #[error("physical page is stale")]
    StalePage,
    #[error("physical page {0} is still pinned")]
    PageStillPinned(u32),
    #[error("physical page {0} reader count overflow")]
    ReaderCountOverflow(u32),
    #[error("device view is stale")]
    StaleView,
    #[error("step was already submitted")]
    StepAlreadySubmitted,
    #[error("step was not submitted")]
    StepNotSubmitted,
    #[error("binding receipt does not match the manager-selected write set")]
    BindingReceiptMismatch,
    #[error("binding receipt duplicates a page")]
    DuplicateBindingReceipt,
    #[error("device view duplicates a page")]
    DuplicatePage,
    #[error("completion receipt is not confirmed")]
    CompletionNotConfirmed,
    #[error("backend observation is unknown")]
    BackendObservationUnknown,
    #[error("reclamation receipt is not acknowledged")]
    ReclamationNotAcknowledged,
    #[error("reclamation receipt does not match its certificate")]
    ReclamationMismatch,
    #[error("reclamation receipt is duplicated")]
    DuplicateReclamation,
    #[error("reserved field must be zero")]
    ReservedFieldNonZero,
    #[error("internal manager invariant failed: {0}")]
    Invariant(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{KvClassSpec, KvPlanInput, compile_plan};
    use std::collections::BTreeMap;
    use std::time::Instant;

    type ArenaCounts = (u16, u32, u32, u64, u64, u64, u64, u64, u64, u64);

    fn sliding_plan(window_tokens: u64, page_tokens: u64) -> CompiledKvPlan {
        compile_plan(KvPlanInput {
            page_tokens,
            classes: vec![KvClassSpec {
                name: "swa".into(),
                layers: vec![0],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(window_tokens),
            }],
        })
        .expect("test plan compiles")
    }

    fn full_plan(page_tokens: u64) -> CompiledKvPlan {
        compile_plan(KvPlanInput {
            page_tokens,
            classes: vec![KvClassSpec {
                name: "full".into(),
                layers: vec![0],
                retention: RetentionKind::Full,
                bytes_per_token_per_layer: 128,
                window_tokens: None,
            }],
        })
        .expect("test full plan compiles")
    }

    fn hybrid_plan(window_tokens: u64) -> CompiledKvPlan {
        compile_plan(KvPlanInput {
            page_tokens: CANONICAL_PAGE_TOKENS,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![0],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 128,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: vec![1],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(window_tokens),
                },
            ],
        })
        .expect("test hybrid plan compiles")
    }

    const fn backend(
        class_id: u16,
        pool_id: u32,
        page_count: u32,
        backend_base_index: u64,
    ) -> BackendArenaRegistration {
        BackendArenaRegistration {
            pool_id,
            class_id,
            backend_domain: class_id + 10,
            page_count,
            reserved: 0,
            backend_base_index,
        }
    }

    fn manager_for_plan(
        plan: &CompiledKvPlan,
        backends: &[BackendArenaRegistration],
        maximum_step_tokens: u32,
        maximum_reclamations: u32,
    ) -> CanonicalKvManager {
        CanonicalKvManager::new(
            plan,
            ManagerConfig {
                maximum_requests: 4,
                maximum_operations: 4,
                maximum_reclamations,
                maximum_step_tokens,
            },
            backends,
        )
        .expect("test manager constructs")
    }

    fn manager_with(
        window_tokens: u64,
        page_count: u32,
        maximum_step_tokens: u32,
        maximum_reclamations: u32,
    ) -> CanonicalKvManager {
        manager_for_plan(
            &sliding_plan(window_tokens, CANONICAL_PAGE_TOKENS),
            &[BackendArenaRegistration {
                pool_id: 7,
                class_id: 0,
                backend_domain: 3,
                page_count,
                reserved: 0,
                backend_base_index: 100,
            }],
            maximum_step_tokens,
            maximum_reclamations,
        )
    }

    fn operation_entries(
        manager: &CanonicalKvManager,
        slot: u32,
        generation: u32,
    ) -> Vec<DeviceKvEntry> {
        let delta = match manager
            .operations
            .get(slot, generation)
            .expect("operation exists")
        {
            OperationState::Prepared(prepared) => &prepared.delta,
            OperationState::Submitted(submitted) => &submitted.delta,
        };
        let state = manager.request(delta.request).expect("operation request");
        let first_new = delta.previous_boundary.div_ceil(manager.page_tokens);
        let write_first = delta.previous_boundary / manager.page_tokens;
        manager
            .classes
            .iter()
            .copied()
            .zip(state.snapshot.roots.iter())
            .zip(delta.classes.iter())
            .flat_map(|((class, root), class_delta)| {
                root.entries
                    .iter()
                    .chain(class_delta.writes.iter())
                    .map(move |entry| {
                        let token_begin = entry.logical_ordinal * manager.page_tokens;
                        let page_end = token_begin.saturating_add(manager.page_tokens);
                        let valid_end = delta.target_boundary.min(page_end);
                        let visible_begin = class
                            .candidate_start(delta.previous_boundary)
                            .max(token_begin);
                        let visible_end = delta.target_boundary.min(page_end);
                        let mut access_flags = DEVICE_KV_ACCESS_READ;
                        if entry.logical_ordinal >= write_first {
                            access_flags |= DEVICE_KV_ACCESS_WRITE;
                        }
                        if entry.logical_ordinal >= first_new {
                            access_flags |= DEVICE_KV_NEEDS_BINDING;
                        }
                        manager
                            .device_entry(
                                *entry,
                                access_flags,
                                valid_end - token_begin,
                                visible_begin - token_begin,
                                visible_end - visible_begin,
                            )
                            .expect("operation entry")
                    })
            })
            .collect()
    }

    fn snapshot_entries(state: &RequestState) -> Vec<RootEntry> {
        state
            .snapshot
            .roots
            .iter()
            .flat_map(|root| root.entries.iter().copied())
            .collect()
    }

    fn published_entries(
        manager: &CanonicalKvManager,
        request: RequestLease,
    ) -> Vec<DeviceKvEntry> {
        let state = manager.request(request).expect("request exists");
        state
            .snapshot
            .roots
            .iter()
            .flat_map(|root| root.entries.iter().copied())
            .map(|root| {
                let class = manager
                    .runtime_class(root.class_id)
                    .expect("published class exists");
                let token_begin = root.logical_ordinal * manager.page_tokens;
                let token_end = token_begin
                    .saturating_add(manager.page_tokens)
                    .min(state.snapshot.boundary);
                let visible_begin = class
                    .retained_start(state.snapshot.boundary)
                    .max(token_begin)
                    .min(token_end);
                let visible_end = state
                    .snapshot
                    .boundary
                    .min(token_begin.saturating_add(manager.page_tokens))
                    .max(visible_begin);
                manager
                    .device_entry(
                        root,
                        u32::from(visible_end > visible_begin) * DEVICE_KV_ACCESS_READ,
                        token_end - token_begin,
                        visible_begin - token_begin,
                        visible_end - visible_begin,
                    )
                    .expect("published entry")
            })
            .collect()
    }

    fn binding_receipts(
        manager: &CanonicalKvManager,
        prepared: &PreparedStep,
    ) -> Vec<BackendBindReceipt> {
        prepared
            .class_lowerings
            .iter()
            .flat_map(|lowering| {
                let class = manager
                    .runtime_class(lowering.class_id)
                    .expect("prepared class exists");
                let begin = usize::try_from(lowering.write_offset).expect("write offset");
                let end = begin + usize::try_from(lowering.write_count).expect("write count");
                prepared.write_intents[begin..end]
                    .iter()
                    .map(move |intent| BackendBindReceipt {
                        step: prepared.step,
                        page: PageLease {
                            engine_epoch: prepared.request.engine_epoch,
                            pool_epoch: manager.pool_epoch,
                            generation: intent.page_generation,
                            page_id: intent.page_id,
                            pool_id: class.backend.pool_id,
                        },
                        backend_domain: class.backend.backend_domain,
                        mapped: 1,
                        writable: 1,
                        reserved: 0,
                        backend_index: class
                            .backend_index(intent.page_id)
                            .expect("prepared page belongs to class"),
                    })
            })
            .collect()
    }

    fn submit(manager: &mut CanonicalKvManager, prepared: &PreparedStep) -> SubmittedStep {
        let receipts = binding_receipts(manager, prepared);
        manager
            .submit_batch(
                &[SubmitBatchItem {
                    step: prepared.step,
                    receipt_offset: 0,
                    receipt_count: u32::try_from(receipts.len()).expect("receipt count"),
                }],
                &receipts,
            )
            .expect("test submit succeeds")[0]
            .clone()
    }

    fn complete(
        manager: &mut CanonicalKvManager,
        submitted: &SubmittedStep,
        domain: u64,
        value: u64,
    ) -> StepCompletion {
        manager
            .complete_batch(
                BatchCompletionReceipt {
                    engine_epoch: submitted.submission.engine_epoch,
                    completion_domain: domain,
                    completion_value: value,
                    confirmed: 1,
                    reserved: 0,
                },
                &[submitted.submission],
            )
            .expect("test completion succeeds")[0]
            .clone()
    }

    fn reclamation_receipts(certificates: &[ReclamationCertificate]) -> Vec<ReclamationReceipt> {
        certificates
            .iter()
            .map(|certificate| ReclamationReceipt {
                reclamation: certificate.reclamation,
                page: certificate.page,
                backend_domain: certificate.backend_domain,
                acknowledged: 1,
                reserved8: 0,
                reserved32: 0,
                backend_index: certificate.backend_index,
            })
            .collect()
    }

    fn arena_counts(manager: &CanonicalKvManager) -> Vec<ArenaCounts> {
        manager
            .arena_stats()
            .iter()
            .map(|stats| {
                (
                    stats.class_id,
                    stats.pool_id,
                    stats.page_count,
                    stats.free_pages,
                    stats.reserved_pages,
                    stats.writing_pages,
                    stats.active_pages,
                    stats.retiring_pages,
                    stats.quarantined_pages,
                    stats.exhausted_pages,
                )
            })
            .collect()
    }

    fn assert_incremental_census_matches_full_scan(manager: &CanonicalKvManager) {
        let mut scanned = vec![PageCounts::default(); manager.classes.len()];
        for page in &manager.pages {
            scanned[usize::from(page.class_id)].increment(page.phase);
        }
        assert_eq!(manager.page_counts, scanned);

        let mut prepared = 0_u64;
        let mut submitted = 0_u64;
        for slot in &manager.operations.slots {
            match slot.value {
                Some(OperationState::Prepared(_)) => prepared += 1,
                Some(OperationState::Submitted(_)) => submitted += 1,
                None => {}
            }
        }
        assert_eq!(manager.prepared_steps, prepared);
        assert_eq!(manager.submitted_steps, submitted);
        let request_pending = manager
            .requests
            .slots
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .map(|request| request.pending_reclamations)
            .sum::<u64>();
        assert_eq!(request_pending, manager.reclamations.active_len() as u64);
    }

    fn complete_initial_18(
        manager: &mut CanonicalKvManager,
        request: RequestLease,
    ) -> StepCompletion {
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .expect("prepare 18")[0]
            .clone();
        let submitted = submit(manager, &prepared);
        complete(manager, &submitted, 7, 9)
    }

    fn batch_submit_items(
        manager: &CanonicalKvManager,
        prepared: &[PreparedStep],
    ) -> (Vec<SubmitBatchItem>, Vec<BackendBindReceipt>) {
        let mut items = Vec::with_capacity(prepared.len());
        let mut receipts = Vec::new();
        for item in prepared {
            let item_receipts = binding_receipts(manager, item);
            items.push(SubmitBatchItem {
                step: item.step,
                receipt_offset: u32::try_from(receipts.len()).expect("receipt offset"),
                receipt_count: u32::try_from(item_receipts.len()).expect("receipt count"),
            });
            receipts.extend(item_receipts);
        }
        (items, receipts)
    }

    #[test]
    fn batch_lifecycle_preserves_order_offsets_and_shared_completion() {
        let mut manager = manager_with(18, 8, 64, 8);
        let requests = manager.acquire_requests(2).expect("batch acquire");
        assert_incremental_census_matches_full_scan(&manager);
        let prepared = manager
            .prepare_batch(&[
                PrepareBatchItem {
                    request: requests[0],
                    target_boundary: 18,
                },
                PrepareBatchItem {
                    request: requests[1],
                    target_boundary: 18,
                },
            ])
            .expect("batch prepare");
        assert_incremental_census_matches_full_scan(&manager);
        assert!(prepared.iter().all(|item| {
            item.class_lowerings.len() == 1
                && item.class_lowerings[0].write_offset == 0
                && item.class_lowerings[0].write_count == 2
                && item.write_intents.len() == 2
        }));
        let (submit_items, receipts) = batch_submit_items(&manager, &prepared);
        let submitted = manager
            .submit_batch(&submit_items, &receipts)
            .expect("batch submit");
        assert_incremental_census_matches_full_scan(&manager);
        assert_eq!(submitted[0].request, requests[0]);
        assert_eq!(submitted[1].request, requests[1]);
        let submissions = submitted
            .iter()
            .map(|item| item.submission)
            .collect::<Vec<_>>();
        let completed = manager
            .complete_batch(
                BatchCompletionReceipt {
                    engine_epoch: manager.engine_epoch(),
                    completion_domain: 77,
                    completion_value: 99,
                    confirmed: 1,
                    reserved: 0,
                },
                &submissions,
            )
            .expect("batch complete");
        assert_incremental_census_matches_full_scan(&manager);
        assert_eq!(completed[0].request, requests[0]);
        assert_eq!(completed[1].request, requests[1]);
        assert!(
            completed.iter().all(|item| {
                item.publication.resident_count == 2 && item.retirements.is_empty()
            })
        );

        let released = manager.release_batch(&requests).expect("batch release");
        assert_incremental_census_matches_full_scan(&manager);
        assert_eq!(released[0].retirement_offset, 0);
        assert_eq!(released[1].retirement_offset, 2);
        let certificates = released
            .iter()
            .flat_map(|item| item.release.retirements.iter().cloned())
            .collect::<Vec<_>>();
        assert!(certificates.iter().all(|certificate| {
            certificate.completion_domain == 77 && certificate.completion_value == 99
        }));
        manager
            .acknowledge_reclamations(&reclamation_receipts(&certificates))
            .expect("batch acknowledge");
        assert_incremental_census_matches_full_scan(&manager);
        manager.recycle_requests(&requests).expect("batch recycle");
        assert_incremental_census_matches_full_scan(&manager);
        assert_eq!(manager.stats().active_requests, 0);
    }

    #[test]
    fn large_sliding_completion_retirement_preflight_is_linear_and_exact() {
        const PAGE_COUNT: u32 = 4_096;
        const TARGET: u64 = 65_536;

        let mut manager = manager_with(
            18,
            PAGE_COUNT,
            u32::try_from(TARGET).expect("target fits maximum step"),
            PAGE_COUNT,
        );
        let request = manager.acquire_requests(1).expect("request batch")[0];
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: TARGET,
            }])
            .expect("large sliding prepare");
        assert_eq!(prepared[0].write_intents.len(), PAGE_COUNT as usize);
        let (items, receipts) = batch_submit_items(&manager, &prepared);
        let submitted = manager
            .submit_batch(&items, &receipts)
            .expect("large sliding submit");
        let completed = manager
            .complete_batch(
                BatchCompletionReceipt {
                    engine_epoch: manager.engine_epoch(),
                    completion_domain: 1,
                    completion_value: 1,
                    confirmed: 1,
                    reserved: 0,
                },
                &[submitted[0].submission],
            )
            .expect("large sliding completion");
        assert_eq!(completed[0].publication.resident_count, 2);
        assert_eq!(completed[0].retirements.len(), PAGE_COUNT as usize - 2);
    }

    #[test]
    fn full_8192_steady_hot_path_is_snapshot_delta_only() {
        const CONTEXT_PAGES: u32 = 8_192;
        const CONTEXT_TOKENS: u64 = CONTEXT_PAGES as u64 * CANONICAL_PAGE_TOKENS;
        let plan = full_plan(CANONICAL_PAGE_TOKENS);
        let mut manager = manager_for_plan(
            &plan,
            &[backend(0, 93, CONTEXT_PAGES + 2, 30_000)],
            u32::try_from(CONTEXT_TOKENS + 2).expect("maximum step"),
            CONTEXT_PAGES + 2,
        );
        let request = manager.acquire_requests(1).expect("request")[0];
        let initial = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: CONTEXT_TOKENS,
            }])
            .expect("initial prepare");
        let submitted = submit(&mut manager, &initial[0]);
        complete(&mut manager, &submitted, 1, 1);
        assert_eq!(
            manager
                .request(request)
                .expect("request")
                .snapshot
                .resident_count(),
            CONTEXT_PAGES as usize
        );

        manager.hot_path = HotPathInstrumentation::default();
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: CONTEXT_TOKENS + 1,
            }])
            .expect("steady prepare");
        assert_eq!(prepared[0].write_intents.len(), 1);
        assert_eq!(
            manager.hot_path,
            HotPathInstrumentation {
                delta_entries_touched: 1,
                ..HotPathInstrumentation::default()
            }
        );

        manager.hot_path = HotPathInstrumentation::default();
        let submitted = submit(&mut manager, &prepared[0]);
        assert_eq!(
            manager.hot_path,
            HotPathInstrumentation {
                delta_entries_touched: 1,
                ..HotPathInstrumentation::default()
            }
        );

        manager.hot_path = HotPathInstrumentation::default();
        let completion = complete(&mut manager, &submitted, 1, 2);
        assert!(completion.retirements.is_empty());
        assert_eq!(completion.publication.resident_count, CONTEXT_PAGES + 1);
        assert_eq!(
            manager.hot_path,
            HotPathInstrumentation {
                delta_entries_touched: 1,
                ..HotPathInstrumentation::default()
            }
        );

        manager.hot_path = HotPathInstrumentation::default();
        let tail = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: CONTEXT_TOKENS + 2,
            }])
            .expect("tail prepare");
        assert!(tail[0].write_intents.is_empty());
        assert_eq!(
            tail[0].class_lowerings[0].flags,
            CLASS_LOWERING_HAS_PREVIOUS_TAIL
        );
        assert_eq!(manager.hot_path, HotPathInstrumentation::default());
        let submitted = submit(&mut manager, &tail[0]);
        complete(&mut manager, &submitted, 1, 3);
        assert_eq!(manager.hot_path.hot_root_entries_visited, 0);
        assert_eq!(manager.hot_path.device_view_entries_materialized, 0);
        assert_eq!(manager.hot_path.snapshot_entries_cloned, 0);
    }

    #[test]
    fn hybrid_large_step_retires_only_the_exact_prefix() {
        const PAGE_COUNT: u32 = 4_096;
        const TARGET: u64 = PAGE_COUNT as u64 * CANONICAL_PAGE_TOKENS;
        let plan = hybrid_plan(18);
        let mut manager = manager_for_plan(
            &plan,
            &[
                backend(0, 94, PAGE_COUNT, 40_000),
                backend(1, 95, PAGE_COUNT, 50_000),
            ],
            u32::try_from(TARGET).expect("maximum step"),
            PAGE_COUNT * 2,
        );
        let request = manager.acquire_requests(1).expect("request")[0];
        let initial = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .expect("initial prepare");
        let submitted = submit(&mut manager, &initial[0]);
        complete(&mut manager, &submitted, 1, 1);

        manager.hot_path = HotPathInstrumentation::default();
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: TARGET,
            }])
            .expect("large hybrid prepare");
        assert_eq!(
            prepared[0].write_intents.len(),
            (usize::try_from(PAGE_COUNT).expect("page count") - 2) * 2
        );
        let submitted = submit(&mut manager, &prepared[0]);
        manager.hot_path = HotPathInstrumentation::default();
        let completion = complete(&mut manager, &submitted, 2, 2);
        assert_eq!(completion.publication.resident_count, PAGE_COUNT + 2);
        assert_eq!(completion.retirements.len(), PAGE_COUNT as usize - 2);
        assert!(completion.retirements.iter().all(|item| item.class_id == 1));
        assert_eq!(
            completion
                .retirements
                .first()
                .expect("first")
                .logical_ordinal,
            0
        );
        assert_eq!(
            completion.retirements.last().expect("last").logical_ordinal,
            u64::from(PAGE_COUNT) - 3
        );
        assert_eq!(manager.hot_path.hot_root_entries_visited, 2);
        assert_eq!(
            manager.hot_path.retirement_entries_touched,
            u64::from(PAGE_COUNT - 2)
        );
        assert_eq!(manager.hot_path.device_view_entries_materialized, 0);
        assert_eq!(manager.hot_path.snapshot_entries_cloned, 0);
    }

    #[test]
    fn b4_semantic_submit_fault_quarantines_every_reachable_snapshot() {
        const CONTEXT_PAGES: u32 = 8;
        let plan = full_plan(CANONICAL_PAGE_TOKENS);
        let mut manager = manager_for_plan(
            &plan,
            &[backend(0, 96, (CONTEXT_PAGES + 1) * 4, 60_000)],
            CONTEXT_PAGES * 16 + 1,
            (CONTEXT_PAGES + 1) * 4,
        );
        let requests = manager.acquire_requests(4).expect("B4 requests");
        let initial = requests
            .iter()
            .copied()
            .map(|request| PrepareBatchItem {
                request,
                target_boundary: u64::from(CONTEXT_PAGES) * CANONICAL_PAGE_TOKENS,
            })
            .collect::<Vec<_>>();
        let prepared = manager.prepare_batch(&initial).expect("initial prepare");
        let (items, receipts) = batch_submit_items(&manager, &prepared);
        let submitted = manager
            .submit_batch(&items, &receipts)
            .expect("initial submit");
        let submissions = submitted
            .iter()
            .map(|item| item.submission)
            .collect::<Vec<_>>();
        manager
            .complete_batch(
                BatchCompletionReceipt {
                    engine_epoch: manager.engine_epoch(),
                    completion_domain: 1,
                    completion_value: 1,
                    confirmed: 1,
                    reserved: 0,
                },
                &submissions,
            )
            .expect("initial completion");

        let extensions = requests
            .iter()
            .copied()
            .map(|request| PrepareBatchItem {
                request,
                target_boundary: u64::from(CONTEXT_PAGES) * CANONICAL_PAGE_TOKENS + 1,
            })
            .collect::<Vec<_>>();
        let prepared = manager
            .prepare_batch(&extensions)
            .expect("extension prepare");
        let (items, mut receipts) = batch_submit_items(&manager, &prepared);
        receipts.last_mut().expect("last receipt").backend_index += 1;
        assert_eq!(
            manager.submit_batch(&items, &receipts),
            Err(KvManagerError::BindingReceiptMismatch)
        );
        assert!(
            requests
                .iter()
                .all(|&request| { manager.request(request).expect("request").quarantined })
        );
        assert_eq!(manager.stats().prepared_steps, 0);
        assert_eq!(manager.stats().active_pages, 0);
        assert_eq!(manager.stats().reserved_pages, 0);
        assert_eq!(
            manager.stats().quarantined_pages,
            u64::from((CONTEXT_PAGES + 1) * 4)
        );
    }

    #[test]
    fn batch_prepare_capacity_failure_is_collectively_zero_mutation() {
        let mut manager = manager_with(18, 3, 64, 3);
        let requests = manager.acquire_requests(2).expect("batch acquire");
        let before = manager.stats();
        assert_eq!(
            manager.prepare_batch(&[
                PrepareBatchItem {
                    request: requests[0],
                    target_boundary: 18,
                },
                PrepareBatchItem {
                    request: requests[1],
                    target_boundary: 18,
                },
            ]),
            Err(KvManagerError::PageCapacityExhausted)
        );
        assert_eq!(manager.stats(), before);
        assert!(
            manager
                .prepare_batch(&[PrepareBatchItem {
                    request: requests[0],
                    target_boundary: 18,
                }])
                .is_ok()
        );
    }

    #[test]
    fn structural_submit_failure_is_retryable_but_semantic_failure_quarantines_all() {
        let mut manager = manager_with(18, 8, 64, 8);
        let requests = manager.acquire_requests(2).expect("batch acquire");
        let prepare_items = requests
            .iter()
            .copied()
            .map(|request| PrepareBatchItem {
                request,
                target_boundary: 18,
            })
            .collect::<Vec<_>>();
        let prepared = manager
            .prepare_batch(&prepare_items)
            .expect("batch prepare");
        let (mut items, mut receipts) = batch_submit_items(&manager, &prepared);
        let before = manager.stats();
        items[1].receipt_offset += 1;
        assert_eq!(
            manager.submit_batch(&items, &receipts),
            Err(KvManagerError::InvalidBatchRange)
        );
        assert_eq!(manager.stats(), before);

        let (items, _) = batch_submit_items(&manager, &prepared);
        receipts.last_mut().expect("receipt").backend_index += 1;
        assert_eq!(
            manager.submit_batch(&items, &receipts),
            Err(KvManagerError::BindingReceiptMismatch)
        );
        let stats = manager.stats();
        assert_eq!(stats.prepared_steps, 0);
        assert_eq!(stats.quarantined_pages, 4);
        for request in requests {
            assert!(manager.request(request).expect("request").quarantined);
        }
        assert_eq!(
            manager.abort_steps(&[
                BackendUnobservedReceipt {
                    step: prepared[0].step,
                    backend_unobserved: 1,
                    reserved: 0,
                },
                BackendUnobservedReceipt {
                    step: prepared[1].step,
                    backend_unobserved: 1,
                    reserved: 0,
                },
            ]),
            Err(KvManagerError::StaleLease("operation"))
        );
    }

    #[test]
    fn batch_complete_duplicate_is_zero_mutation_and_retryable() {
        let mut manager = manager_with(18, 8, 64, 8);
        let requests = manager.acquire_requests(2).expect("batch acquire");
        let prepare_items = requests
            .iter()
            .copied()
            .map(|request| PrepareBatchItem {
                request,
                target_boundary: 18,
            })
            .collect::<Vec<_>>();
        let prepared = manager
            .prepare_batch(&prepare_items)
            .expect("batch prepare");
        let (items, receipts) = batch_submit_items(&manager, &prepared);
        let submitted = manager
            .submit_batch(&items, &receipts)
            .expect("batch submit");
        let first = submitted[0].submission;
        let second = submitted[1].submission;
        let event = BatchCompletionReceipt {
            engine_epoch: manager.engine_epoch(),
            completion_domain: 7,
            completion_value: 8,
            confirmed: 1,
            reserved: 0,
        };
        let before = manager.stats();
        assert_eq!(
            manager.complete_batch(event, &[first, first]),
            Err(KvManagerError::DuplicateSubmission)
        );
        assert_eq!(manager.stats(), before);
        assert!(manager.complete_batch(event, &[first, second]).is_ok());
    }

    #[test]
    fn observed_submit_with_stale_candidate_quarantines_entire_batch() {
        let mut manager = manager_with(18, 8, 64, 8);
        let requests = manager.acquire_requests(2).expect("batch acquire");
        let prepare_items = requests
            .iter()
            .copied()
            .map(|request| PrepareBatchItem {
                request,
                target_boundary: 18,
            })
            .collect::<Vec<_>>();
        let prepared = manager
            .prepare_batch(&prepare_items)
            .expect("batch prepare");
        let (items, receipts) = batch_submit_items(&manager, &prepared);
        let stale_page = prepared[1].write_intents[0].page_id;
        manager.page_mut(stale_page).expect("page").generation += 1;
        assert_eq!(
            manager.submit_batch(&items, &receipts),
            Err(KvManagerError::StalePage)
        );
        assert_eq!(manager.stats().prepared_steps, 0);
        assert_eq!(manager.stats().quarantined_pages, 4);
        assert!(
            requests
                .iter()
                .all(|&request| manager.request(request).expect("request").quarantined)
        );
    }

    #[test]
    fn empty_reclamation_ack_is_rejected() {
        let mut manager = manager_with(18, 4, 64, 4);
        assert_eq!(
            manager.acknowledge_reclamations(&[]),
            Err(KvManagerError::EmptyBatch)
        );
    }

    #[test]
    #[ignore = "CPU microbenchmark; run explicitly with --ignored --nocapture"]
    #[allow(clippy::too_many_lines)]
    fn cpu_microbench_large_pool_b1_b4_control_paths() {
        const PAGE_COUNT: u32 = 65_536;
        const LIFECYCLE_ITERATIONS: u128 = 2_000;
        const CENSUS_ITERATIONS: u128 = 10_000;
        const STEADY_ITERATIONS: u64 = 64;
        for batch_size in [1_usize, 4] {
            let plan = full_plan(CANONICAL_PAGE_TOKENS);
            let mut manager =
                manager_for_plan(&plan, &[backend(0, 91, PAGE_COUNT, 10_000)], 16, PAGE_COUNT);
            let requests = manager
                .acquire_requests(batch_size)
                .expect("benchmark request batch");
            let prepare_items = requests
                .iter()
                .copied()
                .map(|request| PrepareBatchItem {
                    request,
                    target_boundary: 1,
                })
                .collect::<Vec<_>>();

            let lifecycle_start = Instant::now();
            for _ in 0..LIFECYCLE_ITERATIONS {
                let prepared = manager
                    .prepare_batch(&prepare_items)
                    .expect("benchmark prepare");
                let aborts = prepared
                    .iter()
                    .map(|item| BackendUnobservedReceipt {
                        step: item.step,
                        backend_unobserved: 1,
                        reserved: 0,
                    })
                    .collect::<Vec<_>>();
                manager.abort_steps(&aborts).expect("benchmark abort");
            }
            let lifecycle_elapsed = lifecycle_start.elapsed();

            let census_start = Instant::now();
            for _ in 0..CENSUS_ITERATIONS {
                std::hint::black_box(manager.stats());
                std::hint::black_box(manager.arena_stats());
            }
            let census_elapsed = census_start.elapsed();
            assert_incremental_census_matches_full_scan(&manager);
            eprintln!(
                "orbitkv_cpu_microbench pool_pages={PAGE_COUNT} batch={batch_size} lifecycle_ns_per_iter={} census_ns_per_iter={}",
                lifecycle_elapsed.as_nanos() / LIFECYCLE_ITERATIONS,
                census_elapsed.as_nanos() / CENSUS_ITERATIONS,
            );
        }

        for context_pages in [512_u32, 8_192] {
            for batch_size in [1_usize, 4] {
                let batch_u32 = u32::try_from(batch_size).expect("benchmark batch size");
                let extra_pages = u32::try_from(STEADY_ITERATIONS)
                    .expect("steady iterations")
                    .div_ceil(u32::try_from(CANONICAL_PAGE_TOKENS).expect("page tokens"));
                let page_count = context_pages
                    .checked_add(extra_pages)
                    .and_then(|pages| pages.checked_mul(batch_u32))
                    .expect("benchmark page capacity");
                let context_tokens = u64::from(context_pages) * CANONICAL_PAGE_TOKENS;
                let maximum_step_tokens =
                    u32::try_from(context_tokens).expect("benchmark step tokens");
                let plan = full_plan(CANONICAL_PAGE_TOKENS);
                let mut manager = manager_for_plan(
                    &plan,
                    &[backend(0, 92, page_count, 20_000)],
                    maximum_step_tokens,
                    page_count,
                );
                let requests = manager
                    .acquire_requests(batch_size)
                    .expect("steady benchmark request batch");
                let initial_items = requests
                    .iter()
                    .copied()
                    .map(|request| PrepareBatchItem {
                        request,
                        target_boundary: context_tokens,
                    })
                    .collect::<Vec<_>>();
                let prepared = manager
                    .prepare_batch(&initial_items)
                    .expect("steady benchmark initial prepare");
                let (submit_items, receipts) = batch_submit_items(&manager, &prepared);
                let submitted = manager
                    .submit_batch(&submit_items, &receipts)
                    .expect("steady benchmark initial submit");
                let submissions = submitted
                    .iter()
                    .map(|item| item.submission)
                    .collect::<Vec<_>>();
                manager
                    .complete_batch(
                        BatchCompletionReceipt {
                            engine_epoch: manager.engine_epoch(),
                            completion_domain: 1,
                            completion_value: 1,
                            confirmed: 1,
                            reserved: 0,
                        },
                        &submissions,
                    )
                    .expect("steady benchmark initial complete");

                manager.hot_path = HotPathInstrumentation::default();
                let mut prepare_ns = 0_u128;
                let mut submit_ns = 0_u128;
                let mut complete_ns = 0_u128;
                let steady_start = Instant::now();
                for iteration in 0..STEADY_ITERATIONS {
                    let extend_items = requests
                        .iter()
                        .copied()
                        .map(|request| PrepareBatchItem {
                            request,
                            target_boundary: context_tokens + iteration + 1,
                        })
                        .collect::<Vec<_>>();
                    let phase_start = Instant::now();
                    let prepared = manager
                        .prepare_batch(&extend_items)
                        .expect("steady benchmark extend prepare");
                    prepare_ns += phase_start.elapsed().as_nanos();
                    std::hint::black_box(&prepared);
                    let (submit_items, receipts) = batch_submit_items(&manager, &prepared);
                    let phase_start = Instant::now();
                    let submitted = manager
                        .submit_batch(&submit_items, &receipts)
                        .expect("steady benchmark submit");
                    submit_ns += phase_start.elapsed().as_nanos();
                    let submissions = submitted
                        .iter()
                        .map(|item| item.submission)
                        .collect::<Vec<_>>();
                    let phase_start = Instant::now();
                    manager
                        .complete_batch(
                            BatchCompletionReceipt {
                                engine_epoch: manager.engine_epoch(),
                                completion_domain: 1,
                                completion_value: iteration + 2,
                                confirmed: 1,
                                reserved: 0,
                            },
                            &submissions,
                        )
                        .expect("steady benchmark complete");
                    complete_ns += phase_start.elapsed().as_nanos();
                }
                let steady_elapsed = steady_start.elapsed();
                let iterations = u128::from(STEADY_ITERATIONS);
                eprintln!(
                    "orbitkv_steady_full context_pages={context_pages} batch={batch_size} iterations={STEADY_ITERATIONS} prepare_ns_per_iter={} submit_ns_per_iter={} complete_ns_per_iter={} phase_total_ns_per_iter={} wall_ns_per_iter={} hot_root_entries_visited={} device_view_entries_materialized={} snapshot_entries_cloned={} delta_entries_touched={} retirement_entries_touched={}",
                    prepare_ns / iterations,
                    submit_ns / iterations,
                    complete_ns / iterations,
                    (prepare_ns + submit_ns + complete_ns) / iterations,
                    steady_elapsed.as_nanos() / iterations,
                    manager.hot_path.hot_root_entries_visited,
                    manager.hot_path.device_view_entries_materialized,
                    manager.hot_path.snapshot_entries_cloned,
                    manager.hot_path.delta_entries_touched,
                    manager.hot_path.retirement_entries_touched,
                );
                assert_eq!(manager.hot_path.hot_root_entries_visited, 0);
                assert_eq!(manager.hot_path.device_view_entries_materialized, 0);
                assert_eq!(manager.hot_path.snapshot_entries_cloned, 0);
                assert_eq!(manager.hot_path.retirement_entries_touched, 0);
                assert_incremental_census_matches_full_scan(&manager);
            }
        }
    }

    #[test]
    fn rejects_noncanonical_profiles_and_periods() {
        let wrong_page = sliding_plan(18, 8);
        let result = CanonicalKvManager::new(
            &wrong_page,
            ManagerConfig {
                maximum_requests: 1,
                maximum_operations: 1,
                maximum_reclamations: 1,
                maximum_step_tokens: 1,
            },
            &[BackendArenaRegistration {
                pool_id: 1,
                class_id: 0,
                backend_domain: 0,
                page_count: 3,
                reserved: 0,
                backend_base_index: 0,
            }],
        );
        assert!(matches!(result, Err(KvManagerError::UnsupportedProfile(_))));

        let mut malformed = sliding_plan(18, 16);
        malformed.classes[0].slot_count = Some(99);
        let result = CanonicalKvManager::new(
            &malformed,
            ManagerConfig {
                maximum_requests: 1,
                maximum_operations: 1,
                maximum_reclamations: 1,
                maximum_step_tokens: 1,
            },
            &[BackendArenaRegistration {
                pool_id: 1,
                class_id: 0,
                backend_domain: 0,
                page_count: 99,
                reserved: 0,
                backend_base_index: 0,
            }],
        );
        assert!(matches!(result, Err(KvManagerError::UnsupportedProfile(_))));
    }

    #[test]
    fn rejects_malicious_retirement_programs_and_pool_aliases() {
        let plan = hybrid_plan(18);
        let mut malicious_layout = plan.layout_program().expect("layout");
        malicious_layout.classes[0].retirement =
            RetirementProgram::BlockEndPlus { offset_tokens: 17 };
        assert!(matches!(
            validate_class_program(&plan.classes[0], &malicious_layout.classes[0]),
            Err(KvManagerError::UnsupportedProfile(_))
        ));
        malicious_layout.classes[1].retirement = RetirementProgram::Never;
        assert!(matches!(
            validate_class_program(&plan.classes[1], &malicious_layout.classes[1]),
            Err(KvManagerError::UnsupportedProfile(_))
        ));
        malicious_layout.classes[1].retirement =
            RetirementProgram::BlockEndPlus { offset_tokens: 18 };
        assert!(matches!(
            validate_class_program(&plan.classes[1], &malicious_layout.classes[1]),
            Err(KvManagerError::UnsupportedProfile(_))
        ));

        let result = CanonicalKvManager::new(
            &plan,
            ManagerConfig {
                maximum_requests: 1,
                maximum_operations: 1,
                maximum_reclamations: 1,
                maximum_step_tokens: 1,
            },
            &[backend(0, 71, 1, 0), backend(1, 71, 3, 0)],
        );
        assert!(matches!(result, Err(KvManagerError::InvalidConfiguration)));
    }

    #[test]
    fn backend_ranges_are_disjoint_within_each_domain() {
        let plan = hybrid_plan(18);
        let mut backends = [backend(0, 73, 4, 100), backend(1, 74, 3, 103)];
        backends[1].backend_domain = backends[0].backend_domain;
        let settings = ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 1,
            maximum_reclamations: 7,
            maximum_step_tokens: 64,
        };
        assert!(matches!(
            CanonicalKvManager::new(&plan, settings, &backends),
            Err(KvManagerError::InvalidConfiguration)
        ));

        backends[1].backend_base_index = 104;
        assert!(CanonicalKvManager::new(&plan, settings, &backends).is_ok());

        backends[1].backend_domain += 1;
        backends[1].backend_base_index = 100;
        assert!(CanonicalKvManager::new(&plan, settings, &backends).is_ok());
    }

    #[test]
    fn full_attention_is_append_only_until_request_release() {
        let plan = full_plan(CANONICAL_PAGE_TOKENS);
        let mut manager = manager_for_plan(&plan, &[backend(0, 21, 8, 1_000)], 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");

        let first = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare full");
        let first = submit(&mut manager, &first);
        let first = complete(&mut manager, &first, 2, 3);
        assert!(first.retirements.is_empty());
        assert_eq!(first.publication.resident_count, 2);

        let second = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 35,
            }])
            .map(|items| items[0].clone())
            .expect("extend full");
        let second = submit(&mut manager, &second);
        let second = complete(&mut manager, &second, 2, 4);
        assert!(second.retirements.is_empty());
        assert_eq!(
            published_entries(&manager, request)
                .iter()
                .map(|entry| (
                    entry.class_id,
                    entry.logical_ordinal,
                    entry.pool_id,
                    entry.temporal_cell_index,
                    entry.temporal_cycle,
                    entry.visible_token_offset,
                    entry.visible_token_count,
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 21, 0, 0, 0, 16),
                (0, 1, 21, 1, 0, 0, 16),
                (0, 2, 21, 2, 0, 0, 3),
            ]
        );
        let release = manager
            .release_batch(&[request])
            .map(|items| items[0].release.clone())
            .expect("release full");
        assert_eq!(release.retirements.len(), 3);
    }

    #[test]
    fn hybrid_classes_have_independent_pools_addresses_and_retirement() {
        let plan = hybrid_plan(18);
        let mut manager = manager_for_plan(
            &plan,
            &[backend(0, 31, 8, 1_000), backend(1, 32, 4, 2_000)],
            64,
            12,
        );
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let first = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare hybrid");
        assert_eq!(
            operation_entries(&manager, first.step.slot, first.step.generation)
                .iter()
                .map(|entry| (entry.class_id, entry.pool_id, entry.backend_index))
                .collect::<Vec<_>>(),
            vec![
                (0, 31, 1_000),
                (0, 31, 1_001),
                (1, 32, 2_000),
                (1, 32, 2_001)
            ]
        );
        let first = submit(&mut manager, &first);
        assert!(complete(&mut manager, &first, 5, 6).retirements.is_empty());

        let second = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 35,
            }])
            .map(|items| items[0].clone())
            .expect("extend hybrid");
        let second = submit(&mut manager, &second);
        let completion = complete(&mut manager, &second, 5, 7);
        assert_eq!(completion.retirements.len(), 1);
        assert_eq!(completion.retirements[0].class_id, 1);
        assert_eq!(completion.retirements[0].page.pool_id, 32);
        assert_eq!(completion.retirements[0].logical_ordinal, 0);
        assert_eq!(
            published_entries(&manager, request)
                .iter()
                .map(|entry| (
                    entry.class_id,
                    entry.logical_ordinal,
                    entry.pool_id,
                    entry.temporal_cell_index,
                    entry.temporal_cycle,
                    entry.visible_token_offset,
                    entry.visible_token_count,
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 31, 0, 0, 0, 16),
                (0, 1, 31, 1, 0, 0, 16),
                (0, 2, 31, 2, 0, 0, 3),
                (1, 1, 32, 1, 0, 2, 14),
                (1, 2, 32, 2, 0, 0, 3),
            ]
        );
        assert_eq!(manager.stats().active_pages, 5);
        assert_eq!(manager.stats().retiring_pages, 1);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn hybrid_arena_census_is_pure_and_tracks_every_lifecycle_phase() {
        let plan = hybrid_plan(18);
        let mut manager = manager_for_plan(
            &plan,
            &[backend(0, 81, 8, 1_000), backend(1, 82, 4, 2_000)],
            64,
            12,
        );
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 8, 0, 0, 0, 0, 0, 0),
                (1, 82, 4, 4, 0, 0, 0, 0, 0, 0),
            ]
        );
        let first_read = manager.arena_stats();
        assert_eq!(manager.arena_stats(), first_read);
        assert!(
            first_read
                .iter()
                .all(|stats| stats.engine_epoch == manager.engine_epoch()
                    && stats.pool_epoch == manager.pool_epoch())
        );
        assert_eq!(first_read[1].first_page_id, 9);

        let first = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare first");
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 6, 2, 0, 0, 0, 0, 0),
                (1, 82, 4, 2, 2, 0, 0, 0, 0, 0),
            ]
        );
        let first = submit(&mut manager, &first);
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 6, 0, 2, 0, 0, 0, 0),
                (1, 82, 4, 2, 0, 2, 0, 0, 0, 0),
            ]
        );
        complete(&mut manager, &first, 10, 1);
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 6, 0, 0, 2, 0, 0, 0),
                (1, 82, 4, 2, 0, 0, 2, 0, 0, 0),
            ]
        );

        let second = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 35,
            }])
            .map(|items| items[0].clone())
            .expect("prepare second");
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 5, 1, 0, 2, 0, 0, 0),
                (1, 82, 4, 1, 1, 0, 2, 0, 0, 0),
            ]
        );
        let second = submit(&mut manager, &second);
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 5, 0, 2, 1, 0, 0, 0),
                (1, 82, 4, 1, 0, 2, 1, 0, 0, 0),
            ]
        );
        let second = complete(&mut manager, &second, 10, 2);
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 5, 0, 0, 3, 0, 0, 0),
                (1, 82, 4, 1, 0, 0, 2, 1, 0, 0),
            ]
        );
        manager
            .acknowledge_reclamations(&reclamation_receipts(&second.retirements))
            .expect("ack sliding retirement");
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 5, 0, 0, 3, 0, 0, 0),
                (1, 82, 4, 2, 0, 0, 2, 0, 0, 0),
            ]
        );

        let release = manager
            .release_batch(&[request])
            .map(|items| items[0].release.clone())
            .expect("release");
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 5, 0, 0, 0, 3, 0, 0),
                (1, 82, 4, 2, 0, 0, 0, 2, 0, 0),
            ]
        );
        manager
            .acknowledge_reclamations(&reclamation_receipts(&release.retirements))
            .expect("ack release");
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 81, 8, 8, 0, 0, 0, 0, 0, 0),
                (1, 82, 4, 4, 0, 0, 0, 0, 0, 0),
            ]
        );
    }

    #[test]
    fn hybrid_prepare_oom_is_atomic_across_class_pools() {
        let plan = hybrid_plan(18);
        let mut manager =
            manager_for_plan(&plan, &[backend(0, 41, 1, 0), backend(1, 42, 3, 0)], 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let before = manager.stats();
        assert_eq!(
            manager
                .prepare_batch(&[PrepareBatchItem {
                    request,
                    target_boundary: 18
                }])
                .map(|items| items[0].clone()),
            Err(KvManagerError::PageCapacityExhausted)
        );
        assert_eq!(manager.stats(), before);
        assert!(
            manager
                .request(request)
                .expect("request")
                .snapshot
                .is_empty()
        );
        assert!(
            manager
                .prepare_batch(&[PrepareBatchItem {
                    request,
                    target_boundary: 1
                }])
                .map(|items| items[0].clone())
                .is_ok()
        );
    }

    #[test]
    fn reclamation_capacity_must_cover_every_registered_page() {
        let plan = hybrid_plan(18);
        let backends = [backend(0, 51, 4, 0), backend(1, 52, 4, 0)];
        assert!(matches!(
            CanonicalKvManager::new(
                &plan,
                ManagerConfig {
                    maximum_requests: 4,
                    maximum_operations: 4,
                    maximum_reclamations: 7,
                    maximum_step_tokens: 64,
                },
                &backends,
            ),
            Err(KvManagerError::InvalidConfiguration)
        ));

        manager_for_plan(&plan, &backends, 64, 8);
    }

    #[test]
    fn minimum_reclamation_capacity_releases_a_full_arena_root() {
        let plan = full_plan(CANONICAL_PAGE_TOKENS);
        let mut manager = manager_for_plan(&plan, &[backend(0, 53, 4, 0)], 64, 4);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 64,
            }])
            .map(|items| items[0].clone())
            .expect("prepare full root");
        let submitted = submit(&mut manager, &prepared);
        complete(&mut manager, &submitted, 8, 1);

        let release = manager
            .release_batch(&[request])
            .map(|items| items[0].release.clone())
            .expect("release full root");
        assert_eq!(release.retirements.len(), 4);
        manager
            .acknowledge_reclamations(&reclamation_receipts(&release.retirements))
            .expect("ack full root");
        manager
            .recycle_requests(&[request])
            .expect("recycle request");
        assert_eq!(manager.stats().free_pages, 4);
    }

    #[test]
    fn hybrid_lifecycle_property_holds_across_irregular_boundaries() {
        let plan = hybrid_plan(18);
        let mut manager = manager_for_plan(
            &plan,
            &[backend(0, 61, 8, 10_000), backend(1, 62, 4, 20_000)],
            64,
            12,
        );
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let mut full_pages = BTreeMap::new();
        for boundary in [
            1_u64, 2, 15, 16, 17, 18, 31, 32, 33, 35, 48, 49, 64, 79, 80, 81, 95,
        ] {
            let published_before =
                snapshot_entries(manager.request(request).expect("published before"));
            let prepared = manager
                .prepare_batch(&[PrepareBatchItem {
                    request,
                    target_boundary: boundary,
                }])
                .map(|items| items[0].clone())
                .expect("prepare property step");
            assert_eq!(
                snapshot_entries(manager.request(request).expect("not published")),
                published_before
            );
            let submitted = submit(&mut manager, &prepared);
            assert_eq!(
                snapshot_entries(manager.request(request).expect("not published")),
                published_before
            );
            let completion = complete(&mut manager, &submitted, 9, boundary);
            if !completion.retirements.is_empty() {
                manager
                    .acknowledge_reclamations(&reclamation_receipts(&completion.retirements))
                    .expect("ack property retirements");
            }

            let entries = published_entries(&manager, request);
            let full = entries
                .iter()
                .filter(|entry| entry.class_id == 0)
                .collect::<Vec<_>>();
            let sliding = entries
                .iter()
                .filter(|entry| entry.class_id == 1)
                .collect::<Vec<_>>();
            assert_eq!(full.len() as u64, boundary.div_ceil(16));
            let retain_start = boundary.saturating_sub(17);
            let last = (boundary - 1) / 16;
            let expected_sliding = (0..=last)
                .filter(|ordinal| {
                    let token_begin = ordinal * 16;
                    let token_end = boundary.min(token_begin + 16);
                    token_end > retain_start || (!boundary.is_multiple_of(16) && *ordinal == last)
                })
                .count();
            assert_eq!(sliding.len(), expected_sliding);
            for entry in full {
                assert_eq!(entry.pool_id, 61);
                assert_eq!(entry.temporal_cell_index, entry.logical_ordinal);
                assert_eq!(entry.temporal_cycle, 0);
                if let Some(page_id) = full_pages.insert(entry.logical_ordinal, entry.page_id) {
                    assert_eq!(entry.page_id, page_id);
                }
            }
            for entry in sliding {
                assert_eq!(entry.pool_id, 62);
                assert_eq!(entry.temporal_cell_index, entry.logical_ordinal % 3);
                assert_eq!(entry.temporal_cycle, entry.logical_ordinal / 3);
            }
            let unique_pages = entries
                .iter()
                .map(|entry| entry.page_id)
                .collect::<BTreeSet<_>>();
            assert_eq!(unique_pages.len(), entries.len());
            assert_eq!(manager.stats().active_pages, entries.len() as u64);
            assert_eq!(manager.stats().retiring_pages, 0);
            assert_eq!(manager.stats().pending_reclamations, 0);
        }
        let release = manager
            .release_batch(&[request])
            .map(|items| items[0].release.clone())
            .expect("release property request");
        manager
            .acknowledge_reclamations(&reclamation_receipts(&release.retirements))
            .expect("ack property release");
        manager
            .recycle_requests(&[request])
            .expect("recycle property request");
        assert_eq!(manager.stats().active_requests, 0);
        assert_eq!(manager.stats().free_pages, 12);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn randomized_hybrid_snapshot_delta_matches_reference_model() {
        let plan = hybrid_plan(18);
        let mut manager = manager_for_plan(
            &plan,
            &[backend(0, 63, 300, 30_000), backend(1, 64, 32, 40_000)],
            64,
            332,
        );
        let request = manager.acquire_requests(1).expect("request")[0];
        let mut reference = BTreeMap::<(u16, u64), (u32, u64)>::new();
        let mut boundary = 0_u64;
        let mut seed = 0x5eed_cafe_f00d_beef_u64;

        for step_index in 0..128_u64 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let increment = ((seed >> 32) % 64) + 1;
            let target = boundary + increment;
            let prepared = manager
                .prepare_batch(&[PrepareBatchItem {
                    request,
                    target_boundary: target,
                }])
                .expect("random prepare");
            let expected_new = usize::try_from(
                target.div_ceil(CANONICAL_PAGE_TOKENS) - boundary.div_ceil(CANONICAL_PAGE_TOKENS),
            )
            .expect("new pages");
            assert!(
                prepared[0]
                    .class_lowerings
                    .iter()
                    .all(|lowering| lowering.write_count as usize == expected_new)
            );
            assert_eq!(prepared[0].write_intents.len(), expected_new * 2);
            assert!(prepared[0].class_lowerings.iter().all(|lowering| {
                (lowering.flags & CLASS_LOWERING_HAS_PREVIOUS_TAIL != 0)
                    != boundary.is_multiple_of(CANONICAL_PAGE_TOKENS)
            }));

            let candidate =
                operation_entries(&manager, prepared[0].step.slot, prepared[0].step.generation);
            let mut candidate_model = BTreeMap::new();
            for entry in &candidate {
                let key = (entry.class_id, entry.logical_ordinal);
                let identity = (entry.page_id, entry.page_generation);
                assert!(candidate_model.insert(key, identity).is_none());
                if let Some(previous) = reference.get(&key) {
                    assert_eq!(
                        *previous, identity,
                        "stable page identity at step {step_index}"
                    );
                }
            }
            for class_id in [0_u16, 1] {
                let first = if class_id == 0 {
                    0
                } else {
                    boundary.saturating_sub(17) / CANONICAL_PAGE_TOKENS
                };
                let end = target.div_ceil(CANONICAL_PAGE_TOKENS);
                let ordinals = candidate_model
                    .keys()
                    .filter(|(candidate_class, _)| *candidate_class == class_id)
                    .map(|(_, ordinal)| *ordinal)
                    .collect::<Vec<_>>();
                assert_eq!(ordinals, (first..end).collect::<Vec<_>>());
            }

            let submitted = submit(&mut manager, &prepared[0]);
            let completion = complete(&mut manager, &submitted, 9, step_index + 1);
            let sliding_retain_first = target.saturating_sub(17) / CANONICAL_PAGE_TOKENS;
            let expected_retirements = candidate_model
                .iter()
                .filter(|((class_id, ordinal), _)| {
                    *class_id == 1 && *ordinal < sliding_retain_first
                })
                .map(|(&(class_id, ordinal), &(page_id, generation))| {
                    (class_id, ordinal, page_id, generation)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                completion
                    .retirements
                    .iter()
                    .map(|item| {
                        (
                            item.class_id,
                            item.logical_ordinal,
                            item.page.page_id,
                            item.page.generation,
                        )
                    })
                    .collect::<Vec<_>>(),
                expected_retirements
            );
            reference = candidate_model
                .into_iter()
                .filter(|((class_id, ordinal), _)| {
                    *class_id == 0 || *ordinal >= sliding_retain_first
                })
                .collect();
            assert_eq!(
                published_entries(&manager, request)
                    .iter()
                    .map(|entry| {
                        (
                            (entry.class_id, entry.logical_ordinal),
                            (entry.page_id, entry.page_generation),
                        )
                    })
                    .collect::<BTreeMap<_, _>>(),
                reference
            );
            if !completion.retirements.is_empty() {
                manager
                    .acknowledge_reclamations(&reclamation_receipts(&completion.retirements))
                    .expect("random retirement acknowledgement");
            }
            assert_incremental_census_matches_full_scan(&manager);
            boundary = target;
        }
    }

    #[test]
    fn window_one_exact_boundaries_can_publish_an_empty_snapshot() {
        let mut manager = manager_with(1, 4, 32, 4);
        let request = manager.acquire_requests(1).expect("request")[0];
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 16,
            }])
            .expect("aligned prepare");
        let submitted = submit(&mut manager, &prepared[0]);
        let completion = complete(&mut manager, &submitted, 1, 1);
        assert_eq!(completion.publication.resident_count, 0);
        assert_eq!(completion.retirements.len(), 1);
        manager
            .acknowledge_reclamations(&reclamation_receipts(&completion.retirements))
            .expect("aligned retirement");
        assert!(
            manager
                .request(request)
                .expect("request")
                .snapshot
                .is_empty()
        );

        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 17,
            }])
            .expect("post-empty prepare");
        assert_eq!(prepared[0].write_intents.len(), 1);
        assert_eq!(prepared[0].class_lowerings[0].flags, 0);
        let submitted = submit(&mut manager, &prepared[0]);
        let completion = complete(&mut manager, &submitted, 1, 2);
        assert_eq!(completion.publication.resident_count, 1);
        assert!(completion.retirements.is_empty());
        assert_eq!(published_entries(&manager, request)[0].logical_ordinal, 1);
    }

    #[test]
    fn prepare_oom_is_zero_mutation() {
        let mut manager = manager_with(18, 3, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let before = manager.stats();
        assert_eq!(
            manager
                .prepare_batch(&[PrepareBatchItem {
                    request,
                    target_boundary: 51
                }])
                .map(|items| items[0].clone()),
            Err(KvManagerError::PageCapacityExhausted)
        );
        assert_eq!(manager.stats(), before);
        assert!(
            manager
                .request(request)
                .expect("request")
                .snapshot
                .is_empty()
        );
        assert!(
            manager
                .prepare_batch(&[PrepareBatchItem {
                    request,
                    target_boundary: 1
                }])
                .map(|items| items[0].clone())
                .is_ok()
        );
    }

    #[test]
    fn publishes_only_after_confirmed_completion() {
        let mut manager = manager_with(18, 4, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare");
        assert_eq!(
            manager
                .request(request)
                .expect("current")
                .snapshot
                .view_version,
            ViewVersion(0)
        );
        let submitted = submit(&mut manager, &prepared);
        assert_eq!(
            manager
                .request(request)
                .expect("current")
                .snapshot
                .view_version,
            ViewVersion(0)
        );
        let completion = complete(&mut manager, &submitted, 4, 12);
        assert_eq!(completion.publication.view_version, ViewVersion(1));
        let state = manager.request(request).expect("current");
        assert_eq!(
            state.snapshot.view_version,
            completion.publication.view_version
        );
        assert_eq!(state.snapshot.boundary, completion.publication.boundary);
        assert_eq!(
            state.snapshot.resident_count(),
            completion.publication.resident_count as usize
        );
        let before = manager.stats();
        assert!(
            manager
                .complete_batch(
                    BatchCompletionReceipt {
                        engine_epoch: submitted.submission.engine_epoch,
                        completion_domain: 4,
                        completion_value: 12,
                        confirmed: 1,
                        reserved: 0,
                    },
                    &[submitted.submission],
                )
                .is_err()
        );
        assert_eq!(manager.stats(), before);
    }

    #[test]
    fn w18_spans_are_exact_across_wrap() {
        let mut manager = manager_with(18, 5, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        assert!(
            complete_initial_18(&mut manager, request)
                .retirements
                .is_empty()
        );

        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 35,
            }])
            .map(|items| items[0].clone())
            .expect("prepare wrap");
        let entries = operation_entries(&manager, prepared.step.slot, prepared.step.generation);
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries
                .iter()
                .map(|entry| (
                    entry.logical_ordinal,
                    entry.visible_token_offset,
                    entry.visible_token_count,
                    entry.access_flags,
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 1, 15, DEVICE_KV_ACCESS_READ),
                (1, 0, 16, DEVICE_KV_ACCESS_READ | DEVICE_KV_ACCESS_WRITE),
                (
                    2,
                    0,
                    3,
                    DEVICE_KV_ACCESS_READ | DEVICE_KV_ACCESS_WRITE | DEVICE_KV_NEEDS_BINDING,
                ),
            ]
        );
        let submitted = submit(&mut manager, &prepared);
        let completion = complete(&mut manager, &submitted, 11, 22);
        assert_eq!(completion.retirements.len(), 1);
        assert_eq!(completion.retirements[0].logical_ordinal, 0);
        assert_eq!(completion.retirements[0].completion_domain, 11);
        assert_eq!(completion.retirements[0].completion_value, 22);
        assert_eq!(
            published_entries(&manager, request)
                .iter()
                .map(|entry| (
                    entry.logical_ordinal,
                    entry.visible_token_offset,
                    entry.visible_token_count,
                ))
                .collect::<Vec<_>>(),
            vec![(1, 2, 14), (2, 0, 3)]
        );
    }

    #[test]
    fn abort_requires_pre_submit_unobserved_proof() {
        let mut manager = manager_with(18, 4, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare");
        let before = manager.stats();
        assert_eq!(
            manager.abort_steps(&[BackendUnobservedReceipt {
                step: prepared.step,
                backend_unobserved: 0,
                reserved: 0,
            }]),
            Err(KvManagerError::BackendObservationUnknown)
        );
        assert_eq!(manager.stats(), before);
        manager
            .abort_steps(&[BackendUnobservedReceipt {
                step: prepared.step,
                backend_unobserved: 1,
                reserved: 0,
            }])
            .expect("safe abort");
        assert_eq!(manager.stats().free_pages, 4);

        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare retry");
        let _submitted = submit(&mut manager, &prepared);
        assert_eq!(
            manager.abort_steps(&[BackendUnobservedReceipt {
                step: prepared.step,
                backend_unobserved: 1,
                reserved: 0,
            }]),
            Err(KvManagerError::StepAlreadySubmitted)
        );
        assert_eq!(manager.stats().writing_pages, 2);
    }

    #[test]
    fn forged_binding_quarantines_pages_without_reuse() {
        let mut manager = manager_with(18, 3, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare");
        let reserved_ids = prepared
            .write_intents
            .iter()
            .map(|intent| intent.page_id)
            .collect::<BTreeSet<_>>();
        let mut receipts = binding_receipts(&manager, &prepared);
        receipts[0].backend_index += 1;
        assert_eq!(
            manager.submit_batch(
                &[SubmitBatchItem {
                    step: prepared.step,
                    receipt_offset: 0,
                    receipt_count: u32::try_from(receipts.len()).expect("receipt count"),
                }],
                &receipts,
            ),
            Err(KvManagerError::BindingReceiptMismatch)
        );
        assert_eq!(manager.stats().quarantined_pages, 2);
        assert!(manager.request(request).expect("request").quarantined);

        let second = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("second request");
        let second_prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request: second,
                target_boundary: 1,
            }])
            .map(|items| items[0].clone())
            .expect("second prepare");
        assert!(
            second_prepared
                .write_intents
                .iter()
                .all(|intent| !reserved_ids.contains(&intent.page_id))
        );
    }

    #[test]
    fn ambiguous_pre_submit_lowering_has_an_explicit_quarantine_path() {
        let mut manager = manager_with(18, 4, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare");
        manager
            .quarantine_steps(&[prepared.step])
            .expect("quarantine prepared lowering");
        assert_eq!(manager.stats().prepared_steps, 0);
        assert_eq!(manager.stats().quarantined_pages, 2);
        assert!(manager.request(request).expect("request").quarantined);
        assert_eq!(
            manager.abort_steps(&[BackendUnobservedReceipt {
                step: prepared.step,
                backend_unobserved: 1,
                reserved: 0,
            }]),
            Err(KvManagerError::StaleLease("operation"))
        );
    }

    #[test]
    fn completion_preflight_failure_is_zero_mutation() {
        let mut manager = manager_with(18, 4, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare");
        let submitted = submit(&mut manager, &prepared);
        let last_page = operation_entries(
            &manager,
            submitted.submission.slot,
            submitted.submission.generation,
        )
        .last()
        .expect("page")
        .page_id;
        manager.page_mut(last_page).expect("page").readers = 2;
        let before = snapshot_entries(manager.request(request).expect("current"));
        assert_eq!(
            manager.complete_batch(
                BatchCompletionReceipt {
                    engine_epoch: submitted.submission.engine_epoch,
                    completion_domain: 1,
                    completion_value: 2,
                    confirmed: 1,
                    reserved: 0,
                },
                &[submitted.submission],
            ),
            Err(KvManagerError::StalePage)
        );
        assert_eq!(
            snapshot_entries(manager.request(request).expect("current")),
            before
        );
        assert_eq!(manager.stats().submitted_steps, 1);
        assert_eq!(manager.stats().pending_reclamations, 0);
        assert_eq!(manager.page(last_page).expect("page").readers, 2);
        manager.page_mut(last_page).expect("page").readers = 1;
        assert_eq!(
            complete(&mut manager, &submitted, 1, 2)
                .publication
                .view_version,
            ViewVersion(1)
        );
    }

    #[test]
    fn reclamation_batch_is_atomic_and_release_inherits_completion() {
        let mut manager = manager_with(18, 8, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        complete_initial_18(&mut manager, request);
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 51,
            }])
            .map(|items| items[0].clone())
            .expect("prepare long step");
        let submitted = submit(&mut manager, &prepared);
        let completion = complete(&mut manager, &submitted, 17, 29);
        assert_eq!(completion.retirements.len(), 2);
        let mut receipts = reclamation_receipts(&completion.retirements);
        receipts[1].backend_index += 1;
        let before = manager.stats();
        assert_eq!(
            manager.acknowledge_reclamations(&receipts),
            Err(KvManagerError::ReclamationMismatch)
        );
        assert_eq!(manager.stats(), before);
        manager
            .acknowledge_reclamations(&reclamation_receipts(&completion.retirements))
            .expect("atomic acknowledgement");

        let release = manager
            .release_batch(&[request])
            .map(|items| items[0].release.clone())
            .expect("release");
        assert!(
            release
                .retirements
                .iter()
                .all(|certificate| certificate.completion_domain == 17
                    && certificate.completion_value == 29)
        );
    }

    #[test]
    fn hybrid_release_acknowledgement_is_atomic_across_pools() {
        let plan = hybrid_plan(18);
        let mut manager = manager_for_plan(
            &plan,
            &[backend(0, 91, 4, 1_000), backend(1, 92, 3, 2_000)],
            64,
            7,
        );
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare hybrid");
        let submitted = submit(&mut manager, &prepared);
        complete(&mut manager, &submitted, 17, 29);
        let release = manager
            .release_batch(&[request])
            .map(|items| items[0].release.clone())
            .expect("release hybrid");
        assert_eq!(release.retirements.len(), 4);
        assert!(release.retirements.iter().any(|item| item.class_id == 0));
        assert!(release.retirements.iter().any(|item| item.class_id == 1));

        let mut forged = reclamation_receipts(&release.retirements);
        let second_pool = forged
            .iter_mut()
            .find(|item| item.page.pool_id == 92)
            .expect("second-pool receipt");
        second_pool.backend_index += 1;
        let before = arena_counts(&manager);
        assert_eq!(
            manager.acknowledge_reclamations(&forged),
            Err(KvManagerError::ReclamationMismatch)
        );
        assert_eq!(arena_counts(&manager), before);

        manager
            .acknowledge_reclamations(&reclamation_receipts(&release.retirements))
            .expect("retry exact hybrid acknowledgement");
        manager
            .recycle_requests(&[request])
            .expect("recycle hybrid request");
        assert_eq!(manager.stats().free_pages, 7);
    }

    #[test]
    fn release_ack_recycle_advances_request_and_page_generations() {
        let mut manager = manager_with(18, 4, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        complete_initial_18(&mut manager, request);
        let release = manager
            .release_batch(&[request])
            .map(|items| items[0].release.clone())
            .expect("release");
        let first_page = release.retirements[0].page;
        assert_eq!(
            manager.recycle_requests(&[request]),
            Err(KvManagerError::RequestNotRecyclable)
        );
        manager
            .acknowledge_reclamations(&reclamation_receipts(&release.retirements))
            .expect("ack release");
        manager
            .recycle_requests(&[request])
            .expect("recycle request");
        let next = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("next request");
        assert_eq!(next.slot, request.slot);
        assert_eq!(next.generation, request.generation + 1);
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request: next,
                target_boundary: 1,
            }])
            .map(|items| items[0].clone())
            .expect("prepare next");
        assert_eq!(prepared.write_intents[0].page_id, first_page.page_id);
        assert_eq!(
            prepared.write_intents[0].page_generation,
            first_page.generation + 1
        );
        assert!(manager.request(request).is_err());
    }

    #[test]
    fn reclamation_arena_backpressure_is_zero_mutation_and_retryable() {
        let mut arena = Arena::new("reclamation", 1).expect("arena");
        let occupied = arena.plan_many(1).expect("initial allocation")[0];
        arena.insert_planned(occupied, 7_u8);
        assert_eq!(
            arena.plan_many(1),
            Err(KvManagerError::ArenaExhausted("reclamation"))
        );
        assert_eq!(arena.get(occupied.0, occupied.1), Ok(&7));
        assert_eq!(arena.remove(occupied.0, occupied.1), Ok(7));
        assert!(arena.plan_many(1).is_ok());
    }

    #[test]
    fn ambiguous_submission_and_cross_manager_identities_fail_closed() {
        let mut first = manager_with(18, 4, 64, 8);
        let second = manager_with(18, 4, 64, 8);
        let request = first
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        assert!(matches!(
            second.request(request),
            Err(KvManagerError::WrongEngine)
        ));
        let prepared = first
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 18,
            }])
            .map(|items| items[0].clone())
            .expect("prepare");
        let submitted = submit(&mut first, &prepared);
        first
            .quarantine_submissions(&[submitted.submission])
            .expect("quarantine");
        assert_eq!(first.stats().quarantined_pages, 2);
        assert!(first.request(request).expect("request").quarantined);
    }

    #[test]
    fn release_version_overflow_and_page_generation_exhaustion_are_safe() {
        let mut manager = manager_with(18, 3, 64, 8);
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        complete_initial_18(&mut manager, request);
        manager
            .request_mut(request)
            .expect("request")
            .snapshot
            .view_version = ViewVersion(u64::MAX);
        let before = manager.stats();
        let page_table = snapshot_entries(manager.request(request).expect("request"));
        assert_eq!(
            manager
                .release_batch(&[request])
                .map(|items| items[0].release.clone()),
            Err(KvManagerError::ViewVersionExhausted)
        );
        assert_eq!(manager.stats(), before);
        assert_eq!(
            snapshot_entries(manager.request(request).expect("request")),
            page_table
        );

        let mut generation_manager = manager_with(18, 3, 64, 8);
        let request = generation_manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        generation_manager.pages[0].generation = u64::MAX - 1;
        let prepared = generation_manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 1,
            }])
            .map(|items| items[0].clone())
            .expect("prepare max gen");
        assert_eq!(prepared.write_intents[0].page_generation, u64::MAX);
        generation_manager
            .abort_steps(&[BackendUnobservedReceipt {
                step: prepared.step,
                backend_unobserved: 1,
                reserved: 0,
            }])
            .expect("abort max gen");
        assert_eq!(generation_manager.stats().exhausted_pages, 1);
        assert_eq!(generation_manager.stats().free_pages, 2);
    }

    #[test]
    fn generation_exhaustion_is_isolated_to_one_hybrid_arena_page() {
        let plan = hybrid_plan(18);
        let mut manager = manager_for_plan(
            &plan,
            &[backend(0, 101, 4, 1_000), backend(1, 102, 3, 2_000)],
            64,
            7,
        );
        manager.pages[4].generation = u64::MAX - 1;
        let request = manager
            .acquire_requests(1)
            .map(|requests| requests[0])
            .expect("request");
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                target_boundary: 1,
            }])
            .map(|items| items[0].clone())
            .expect("prepare max gen");
        let second_class = prepared
            .class_lowerings
            .iter()
            .find(|lowering| lowering.class_id == 1)
            .expect("second-arena class");
        let second = prepared.write_intents
            [usize::try_from(second_class.write_offset).expect("write offset")];
        assert_eq!(second.page_id, 5);
        assert_eq!(second.page_generation, u64::MAX);
        manager
            .abort_steps(&[BackendUnobservedReceipt {
                step: prepared.step,
                backend_unobserved: 1,
                reserved: 0,
            }])
            .expect("abort max generation");
        assert_eq!(
            arena_counts(&manager),
            vec![
                (0, 101, 4, 4, 0, 0, 0, 0, 0, 0),
                (1, 102, 3, 2, 0, 0, 0, 0, 0, 1)
            ]
        );
    }
}
