use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;

use crate::{CompiledKvClass, CompiledKvPlan, PlanError, RetentionKind};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct BlockKey {
    pub request_id: String,
    pub class_name: String,
    pub ordinal: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct BlockHandle {
    pub class_name: String,
    pub slot: u64,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ViewBlock {
    pub logical: BlockKey,
    pub physical: BlockHandle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct KvView {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub submission_id: u64,
    pub request_id: String,
    pub semantic_frontier: u64,
    pub blocks: Vec<ViewBlock>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SemanticProof {
    SlidingWindow {
        semantic_frontier: u64,
        death_boundary: u64,
    },
    RequestReleased,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionProof {
    pub completed_through: u64,
    pub outstanding_readers: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RetirementCertificate {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub certificate_id: u64,
    pub logical: BlockKey,
    pub physical: BlockHandle,
    pub token_start: u64,
    pub token_end_exclusive: u64,
    pub semantic_proof: SemanticProof,
    pub execution_proof: ExecutionProof,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClassPoolConfig {
    pub class_name: String,
    pub slot_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockManagerConfig {
    pub pools: Vec<ClassPoolConfig>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ManagerStats {
    pub requests: u64,
    pub active_submissions: u64,
    pub pending_certificates: u64,
    pub resident_blocks: u64,
    pub retiring_blocks: u64,
    pub free_slots: BTreeMap<String, u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SlotPhase {
    #[default]
    Free,
    Active,
    Retiring,
    Certified,
}

#[derive(Clone, Debug, Default)]
struct SlotState {
    generation: u64,
    occupant: Option<BlockKey>,
    readers: BTreeSet<u64>,
    phase: SlotPhase,
    pending_certificate: Option<u64>,
}

#[derive(Clone, Debug)]
struct ClassPool {
    slots: Vec<SlotState>,
    free: BTreeSet<usize>,
}

#[derive(Clone, Debug, Default)]
struct RequestState {
    semantic_frontier: u64,
    materialized_boundary: u64,
    released: bool,
    blocks: BTreeSet<BlockKey>,
}

#[derive(Clone, Debug)]
struct SubmissionState {
    blocks: Vec<ViewBlock>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ManagerError {
    #[error("unknown KV class {0:?}")]
    UnknownClass(String),
    #[error("missing physical pool configuration for KV class {0:?}")]
    MissingPool(String),
    #[error("duplicate physical pool configuration for KV class {0:?}")]
    DuplicatePool(String),
    #[error("physical pool {class:?} must contain at least one slot")]
    EmptyPool { class: String },
    #[error("physical slot count does not fit the host address space for {0:?}")]
    SlotCountTooLarge(String),
    #[error("request {0:?} already exists")]
    DuplicateRequest(String),
    #[error("unknown request {0:?}")]
    UnknownRequest(String),
    #[error("request {0:?} has already been released")]
    RequestReleased(String),
    #[error("semantic frontier for {request:?} cannot move backwards from {current} to {next}")]
    SemanticFrontierMovedBackwards {
        request: String,
        current: u64,
        next: u64,
    },
    #[error(
        "semantic frontier {frontier} for {request:?} exceeds materialized boundary {materialized}"
    )]
    SemanticFrontierBeyondMaterialized {
        request: String,
        frontier: u64,
        materialized: u64,
    },
    #[error("materialized boundary for {request:?} cannot move backwards from {current} to {next}")]
    MaterializedBoundaryMovedBackwards {
        request: String,
        current: u64,
        next: u64,
    },
    #[error("logical block is already dead at the request frontier: {0:?}")]
    CannotMaterializeDead(BlockKey),
    #[error("physical pool {class:?} has no committed free slot")]
    PoolExhausted { class: String },
    #[error("logical block is not resident: {0:?}")]
    BlockNotResident(BlockKey),
    #[error("logical block handle is stale: {0:?}")]
    StaleHandle(BlockHandle),
    #[error("unknown submission {0}")]
    UnknownSubmission(u64),
    #[error("unknown retirement certificate {0}")]
    UnknownCertificate(u64),
    #[error("retirement certificate {0} no longer matches its physical slot")]
    StaleCertificate(u64),
    #[error("submission generation exhausted")]
    SubmissionGenerationExhausted,
    #[error("certificate generation exhausted")]
    CertificateGenerationExhausted,
    #[error("block generation exhausted for {0:?}")]
    BlockGenerationExhausted(BlockKey),
    #[error("integer overflow while calculating {0}")]
    ArithmeticOverflow(&'static str),
    #[error(transparent)]
    Plan(#[from] PlanError),
}

pub struct KvBlockManager {
    plan: CompiledKvPlan,
    fingerprint: String,
    classes: BTreeMap<String, CompiledKvClass>,
    pools: BTreeMap<String, ClassPool>,
    requests: BTreeMap<String, RequestState>,
    block_handles: BTreeMap<BlockKey, BlockHandle>,
    submissions: BTreeMap<u64, SubmissionState>,
    completed_out_of_order: BTreeSet<u64>,
    completed_through: u64,
    next_submission_id: u64,
    pending_certificates: BTreeMap<u64, RetirementCertificate>,
    next_certificate_id: u64,
}

impl KvBlockManager {
    /// Creates a block manager whose physical slot budgets are explicit.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, empty, unknown, or
    /// host-unrepresentable pool configurations.
    pub fn new(plan: CompiledKvPlan, config: BlockManagerConfig) -> Result<Self, ManagerError> {
        let classes = plan
            .classes
            .iter()
            .map(|class| (class.spec.name.clone(), class.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut configured = BTreeMap::new();
        for pool in config.pools {
            if !classes.contains_key(&pool.class_name) {
                return Err(ManagerError::UnknownClass(pool.class_name));
            }
            if pool.slot_count == 0 {
                return Err(ManagerError::EmptyPool {
                    class: pool.class_name,
                });
            }
            let class_name = pool.class_name.clone();
            if configured.insert(class_name.clone(), pool).is_some() {
                return Err(ManagerError::DuplicatePool(class_name));
            }
        }
        let mut pools = BTreeMap::new();
        for class_name in classes.keys() {
            let config = configured
                .remove(class_name)
                .ok_or_else(|| ManagerError::MissingPool(class_name.clone()))?;
            let count = usize::try_from(config.slot_count)
                .map_err(|_| ManagerError::SlotCountTooLarge(class_name.clone()))?;
            pools.insert(
                class_name.clone(),
                ClassPool {
                    slots: vec![SlotState::default(); count],
                    free: (0..count).collect(),
                },
            );
        }
        let fingerprint = plan.fingerprint();
        Ok(Self {
            plan,
            fingerprint,
            classes,
            pools,
            requests: BTreeMap::new(),
            block_handles: BTreeMap::new(),
            submissions: BTreeMap::new(),
            completed_out_of_order: BTreeSet::new(),
            completed_through: 0,
            next_submission_id: 1,
            pending_certificates: BTreeMap::new(),
            next_certificate_id: 1,
        })
    }

    #[must_use]
    pub fn plan_fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Registers one independent logical request.
    ///
    /// # Errors
    ///
    /// Returns an error if the request already exists.
    pub fn register_request(&mut self, request_id: impl Into<String>) -> Result<(), ManagerError> {
        let request_id = request_id.into();
        if self.requests.contains_key(&request_id) {
            return Err(ManagerError::DuplicateRequest(request_id));
        }
        self.requests.insert(request_id, RequestState::default());
        Ok(())
    }

    /// Records that a request has materialized tokens through `boundary`.
    ///
    /// This method allocates every newly intersected logical block. It does not
    /// advance semantic time or reclaim blocks.
    ///
    /// # Errors
    ///
    /// Returns an error for backward movement, a released request, a dead
    /// logical block, pool exhaustion, or checked arithmetic failure.
    pub fn materialize_to(
        &mut self,
        request_id: &str,
        boundary: u64,
    ) -> Result<Vec<ViewBlock>, ManagerError> {
        let request = self
            .requests
            .get(request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?;
        if request.released {
            return Err(ManagerError::RequestReleased(request_id.to_owned()));
        }
        if boundary < request.materialized_boundary {
            return Err(ManagerError::MaterializedBoundaryMovedBackwards {
                request: request_id.to_owned(),
                current: request.materialized_boundary,
                next: boundary,
            });
        }
        let old_boundary = request.materialized_boundary;
        if old_boundary == boundary {
            return Ok(Vec::new());
        }
        let first_block = old_boundary / self.plan.page_tokens;
        let last_block = (boundary - 1) / self.plan.page_tokens;
        let class_names = self.classes.keys().cloned().collect::<Vec<_>>();
        let mut required = BTreeMap::<String, u64>::new();
        for ordinal in first_block..=last_block {
            for class_name in &class_names {
                let logical = BlockKey {
                    request_id: request_id.to_owned(),
                    class_name: class_name.clone(),
                    ordinal,
                };
                if self.block_handles.contains_key(&logical) {
                    continue;
                }
                if self.is_semantically_dead(&logical)? {
                    return Err(ManagerError::CannotMaterializeDead(logical));
                }
                let count = required.entry(class_name.clone()).or_default();
                *count = count
                    .checked_add(1)
                    .ok_or(ManagerError::ArithmeticOverflow(
                        "materialization block count",
                    ))?;
            }
        }
        for (class_name, required_slots) in required {
            let available_slots = self
                .pools
                .get(&class_name)
                .ok_or_else(|| ManagerError::UnknownClass(class_name.clone()))?
                .free
                .len() as u64;
            if required_slots > available_slots {
                return Err(ManagerError::PoolExhausted { class: class_name });
            }
        }
        let mut allocated = Vec::new();
        for ordinal in first_block..=last_block {
            for class_name in &class_names {
                let logical = BlockKey {
                    request_id: request_id.to_owned(),
                    class_name: class_name.clone(),
                    ordinal,
                };
                if let Some(handle) = self.block_handles.get(&logical) {
                    allocated.push(ViewBlock {
                        logical,
                        physical: handle.clone(),
                    });
                    continue;
                }
                let handle = self.allocate(logical.clone())?;
                self.requests
                    .get_mut(request_id)
                    .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?
                    .blocks
                    .insert(logical.clone());
                self.block_handles.insert(logical.clone(), handle.clone());
                allocated.push(ViewBlock {
                    logical,
                    physical: handle,
                });
            }
        }
        self.requests
            .get_mut(request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?
            .materialized_boundary = boundary;
        Ok(allocated)
    }

    /// Advances the semantic frontier and emits certificates for blocks that
    /// have no future legal readers and no outstanding GPU readers.
    ///
    /// # Errors
    ///
    /// Returns an error if the frontier moves backward or beyond materialized
    /// state, or if checked lifetime arithmetic overflows.
    pub fn advance_semantic_frontier(
        &mut self,
        request_id: &str,
        boundary: u64,
    ) -> Result<Vec<RetirementCertificate>, ManagerError> {
        let request = self
            .requests
            .get(request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?;
        if request.released {
            return Err(ManagerError::RequestReleased(request_id.to_owned()));
        }
        if boundary < request.semantic_frontier {
            return Err(ManagerError::SemanticFrontierMovedBackwards {
                request: request_id.to_owned(),
                current: request.semantic_frontier,
                next: boundary,
            });
        }
        if boundary > request.materialized_boundary {
            return Err(ManagerError::SemanticFrontierBeyondMaterialized {
                request: request_id.to_owned(),
                frontier: boundary,
                materialized: request.materialized_boundary,
            });
        }
        self.requests
            .get_mut(request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?
            .semantic_frontier = boundary;
        self.mark_request_retirements(request_id)
    }

    /// Pins an immutable generation-checked view for one GPU submission.
    ///
    /// # Errors
    ///
    /// Returns an error if required live blocks are missing, a handle is stale,
    /// the request was released, or the submission id overflows.
    pub fn submit_view(&mut self, request_id: &str) -> Result<KvView, ManagerError> {
        let request = self
            .requests
            .get(request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?;
        if request.released {
            return Err(ManagerError::RequestReleased(request_id.to_owned()));
        }
        let semantic_frontier = request.semantic_frontier;
        let continuation = self.plan.continuation_blocks(semantic_frontier)?;
        let mut blocks = Vec::new();
        for class in &self.plan.classes {
            for &ordinal in &continuation[&class.spec.name] {
                let logical = BlockKey {
                    request_id: request_id.to_owned(),
                    class_name: class.spec.name.clone(),
                    ordinal,
                };
                let physical = self
                    .block_handles
                    .get(&logical)
                    .cloned()
                    .ok_or_else(|| ManagerError::BlockNotResident(logical.clone()))?;
                self.validate_handle(&logical, &physical)?;
                blocks.push(ViewBlock { logical, physical });
            }
        }
        let submission_id = self.next_submission_id;
        self.next_submission_id = self
            .next_submission_id
            .checked_add(1)
            .ok_or(ManagerError::SubmissionGenerationExhausted)?;
        for block in &blocks {
            self.slot_mut(&block.physical)?
                .readers
                .insert(submission_id);
        }
        self.submissions.insert(
            submission_id,
            SubmissionState {
                blocks: blocks.clone(),
            },
        );
        Ok(KvView {
            schema: "orbitkv.kv-view.v1",
            plan_fingerprint: self.fingerprint.clone(),
            submission_id,
            request_id: request_id.to_owned(),
            semantic_frontier,
            blocks,
        })
    }

    /// Completes a GPU submission and emits newly unblocked retirement
    /// certificates. Completion may arrive out of order.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown submission or stale generation handle.
    pub fn complete_submission(
        &mut self,
        submission_id: u64,
    ) -> Result<Vec<RetirementCertificate>, ManagerError> {
        let submission = self
            .submissions
            .remove(&submission_id)
            .ok_or(ManagerError::UnknownSubmission(submission_id))?;
        let mut affected = BTreeSet::new();
        for block in submission.blocks {
            self.validate_handle(&block.logical, &block.physical)?;
            self.slot_mut(&block.physical)?
                .readers
                .remove(&submission_id);
            affected.insert(block.logical);
        }
        self.record_completion(submission_id);
        let mut certificates = Vec::new();
        for logical in affected {
            if let Some(certificate) = self.try_certify(&logical)? {
                certificates.push(certificate);
            }
        }
        Ok(certificates)
    }

    /// Marks every block of a finished or cancelled request semantically dead.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown or already released request.
    pub fn release_request(
        &mut self,
        request_id: &str,
    ) -> Result<Vec<RetirementCertificate>, ManagerError> {
        let request = self
            .requests
            .get_mut(request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?;
        if request.released {
            return Err(ManagerError::RequestReleased(request_id.to_owned()));
        }
        request.released = true;
        self.mark_request_retirements(request_id)
    }

    /// Commits a certificate after the physical backend has completed the
    /// corresponding free/unmap operation.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown, stale, pinned, or mismatched
    /// certificate. Failure leaves the slot unavailable.
    pub fn commit_reclamation(&mut self, certificate_id: u64) -> Result<(), ManagerError> {
        let certificate = self
            .pending_certificates
            .get(&certificate_id)
            .cloned()
            .ok_or(ManagerError::UnknownCertificate(certificate_id))?;
        self.validate_handle(&certificate.logical, &certificate.physical)?;
        {
            let slot = self.slot_mut(&certificate.physical)?;
            if slot.pending_certificate != Some(certificate_id)
                || slot.phase != SlotPhase::Certified
                || !slot.readers.is_empty()
            {
                return Err(ManagerError::StaleCertificate(certificate_id));
            }
            slot.occupant = None;
            slot.phase = SlotPhase::Free;
            slot.pending_certificate = None;
        }
        let slot_index = usize::try_from(certificate.physical.slot)
            .map_err(|_| ManagerError::StaleCertificate(certificate_id))?;
        self.pools
            .get_mut(&certificate.physical.class_name)
            .ok_or_else(|| ManagerError::UnknownClass(certificate.physical.class_name.clone()))?
            .free
            .insert(slot_index);
        self.block_handles.remove(&certificate.logical);
        if let Some(request) = self.requests.get_mut(&certificate.logical.request_id) {
            request.blocks.remove(&certificate.logical);
        }
        self.pending_certificates.remove(&certificate_id);
        Ok(())
    }

    #[must_use]
    pub fn stats(&self) -> ManagerStats {
        let mut free_slots = BTreeMap::new();
        let mut resident_blocks = 0_u64;
        let mut retiring_blocks = 0_u64;
        for (name, pool) in &self.pools {
            free_slots.insert(name.clone(), pool.free.len() as u64);
            for slot in &pool.slots {
                if slot.occupant.is_some() {
                    resident_blocks += 1;
                }
                if matches!(slot.phase, SlotPhase::Retiring | SlotPhase::Certified) {
                    retiring_blocks += 1;
                }
            }
        }
        ManagerStats {
            requests: self.requests.len() as u64,
            active_submissions: self.submissions.len() as u64,
            pending_certificates: self.pending_certificates.len() as u64,
            resident_blocks,
            retiring_blocks,
            free_slots,
        }
    }

    fn allocate(&mut self, logical: BlockKey) -> Result<BlockHandle, ManagerError> {
        let pool = self
            .pools
            .get_mut(&logical.class_name)
            .ok_or_else(|| ManagerError::UnknownClass(logical.class_name.clone()))?;
        let slot_index = pool
            .free
            .pop_first()
            .ok_or_else(|| ManagerError::PoolExhausted {
                class: logical.class_name.clone(),
            })?;
        let slot = &mut pool.slots[slot_index];
        debug_assert_eq!(slot.phase, SlotPhase::Free);
        debug_assert!(slot.occupant.is_none());
        slot.generation = slot
            .generation
            .checked_add(1)
            .ok_or_else(|| ManagerError::BlockGenerationExhausted(logical.clone()))?;
        slot.occupant = Some(logical.clone());
        slot.phase = SlotPhase::Active;
        Ok(BlockHandle {
            class_name: logical.class_name,
            slot: slot_index as u64,
            generation: slot.generation,
        })
    }

    fn mark_request_retirements(
        &mut self,
        request_id: &str,
    ) -> Result<Vec<RetirementCertificate>, ManagerError> {
        let blocks = self
            .requests
            .get(request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?
            .blocks
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut certificates = Vec::new();
        for logical in blocks {
            if !self.is_semantically_dead(&logical)? {
                continue;
            }
            let handle = self
                .block_handles
                .get(&logical)
                .cloned()
                .ok_or_else(|| ManagerError::BlockNotResident(logical.clone()))?;
            let slot = self.slot_mut(&handle)?;
            if slot.phase == SlotPhase::Active {
                slot.phase = SlotPhase::Retiring;
            }
            if let Some(certificate) = self.try_certify(&logical)? {
                certificates.push(certificate);
            }
        }
        Ok(certificates)
    }

    fn try_certify(
        &mut self,
        logical: &BlockKey,
    ) -> Result<Option<RetirementCertificate>, ManagerError> {
        let handle = self
            .block_handles
            .get(logical)
            .cloned()
            .ok_or_else(|| ManagerError::BlockNotResident(logical.clone()))?;
        {
            let slot = self.slot(&handle)?;
            if slot.phase != SlotPhase::Retiring || !slot.readers.is_empty() {
                return Ok(None);
            }
        }
        let certificate_id = self.next_certificate_id;
        self.next_certificate_id = self
            .next_certificate_id
            .checked_add(1)
            .ok_or(ManagerError::CertificateGenerationExhausted)?;
        let token_start = logical
            .ordinal
            .checked_mul(self.plan.page_tokens)
            .ok_or(ManagerError::ArithmeticOverflow("block token start"))?;
        let token_end_exclusive = token_start
            .checked_add(self.plan.page_tokens)
            .ok_or(ManagerError::ArithmeticOverflow("block token end"))?;
        let request = self
            .requests
            .get(&logical.request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(logical.request_id.clone()))?;
        let semantic_proof = if request.released {
            SemanticProof::RequestReleased
        } else {
            SemanticProof::SlidingWindow {
                semantic_frontier: request.semantic_frontier,
                death_boundary: self
                    .death_boundary(logical)?
                    .ok_or_else(|| ManagerError::CannotMaterializeDead(logical.clone()))?,
            }
        };
        let certificate = RetirementCertificate {
            schema: "orbitkv.retirement-certificate.v1",
            plan_fingerprint: self.fingerprint.clone(),
            certificate_id,
            logical: logical.clone(),
            physical: handle.clone(),
            token_start,
            token_end_exclusive,
            semantic_proof,
            execution_proof: ExecutionProof {
                completed_through: self.completed_through,
                outstanding_readers: 0,
            },
        };
        let slot = self.slot_mut(&handle)?;
        slot.phase = SlotPhase::Certified;
        slot.pending_certificate = Some(certificate_id);
        self.pending_certificates
            .insert(certificate_id, certificate.clone());
        Ok(Some(certificate))
    }

    fn is_semantically_dead(&self, logical: &BlockKey) -> Result<bool, ManagerError> {
        let request = self
            .requests
            .get(&logical.request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(logical.request_id.clone()))?;
        if request.released {
            return Ok(true);
        }
        Ok(self
            .death_boundary(logical)?
            .is_some_and(|death| request.semantic_frontier >= death))
    }

    fn death_boundary(&self, logical: &BlockKey) -> Result<Option<u64>, ManagerError> {
        let class = self
            .classes
            .get(&logical.class_name)
            .ok_or_else(|| ManagerError::UnknownClass(logical.class_name.clone()))?;
        match class.spec.retention {
            RetentionKind::Full => Ok(None),
            RetentionKind::Sliding => {
                let window = class.spec.window_tokens.ok_or_else(|| {
                    ManagerError::Plan(PlanError::InvalidCompiledClass {
                        class: class.spec.name.clone(),
                    })
                })?;
                let block_end = logical
                    .ordinal
                    .checked_add(1)
                    .and_then(|ordinal| ordinal.checked_mul(self.plan.page_tokens))
                    .ok_or(ManagerError::ArithmeticOverflow("block death boundary"))?;
                let death = block_end
                    .checked_add(window - 1)
                    .ok_or(ManagerError::ArithmeticOverflow("block death boundary"))?;
                Ok(Some(death))
            }
        }
    }

    fn validate_handle(
        &self,
        logical: &BlockKey,
        handle: &BlockHandle,
    ) -> Result<(), ManagerError> {
        let slot = self.slot(handle)?;
        if slot.generation != handle.generation || slot.occupant.as_ref() != Some(logical) {
            return Err(ManagerError::StaleHandle(handle.clone()));
        }
        Ok(())
    }

    fn slot(&self, handle: &BlockHandle) -> Result<&SlotState, ManagerError> {
        let index =
            usize::try_from(handle.slot).map_err(|_| ManagerError::StaleHandle(handle.clone()))?;
        self.pools
            .get(&handle.class_name)
            .ok_or_else(|| ManagerError::UnknownClass(handle.class_name.clone()))?
            .slots
            .get(index)
            .ok_or_else(|| ManagerError::StaleHandle(handle.clone()))
    }

    fn slot_mut(&mut self, handle: &BlockHandle) -> Result<&mut SlotState, ManagerError> {
        let index =
            usize::try_from(handle.slot).map_err(|_| ManagerError::StaleHandle(handle.clone()))?;
        self.pools
            .get_mut(&handle.class_name)
            .ok_or_else(|| ManagerError::UnknownClass(handle.class_name.clone()))?
            .slots
            .get_mut(index)
            .ok_or_else(|| ManagerError::StaleHandle(handle.clone()))
    }

    fn record_completion(&mut self, submission_id: u64) {
        if submission_id == self.completed_through + 1 {
            self.completed_through = submission_id;
            while self
                .completed_out_of_order
                .remove(&(self.completed_through + 1))
            {
                self.completed_through += 1;
            }
        } else if submission_id > self.completed_through + 1 {
            self.completed_out_of_order.insert(submission_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{KvClassSpec, KvPlanInput, compile_plan};

    use super::*;

    fn hybrid_plan() -> CompiledKvPlan {
        compile_plan(KvPlanInput {
            page_tokens: 16,
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
                    window_tokens: Some(32),
                },
            ],
        })
        .unwrap()
    }

    fn manager(full_slots: u64, swa_slots: u64) -> KvBlockManager {
        KvBlockManager::new(
            hybrid_plan(),
            BlockManagerConfig {
                pools: vec![
                    ClassPoolConfig {
                        class_name: "full".into(),
                        slot_count: full_slots,
                    },
                    ClassPoolConfig {
                        class_name: "swa".into(),
                        slot_count: swa_slots,
                    },
                ],
            },
        )
        .unwrap()
    }

    #[test]
    fn certificate_commit_is_required_before_reuse() {
        let mut manager = manager(8, 3);
        manager.register_request("r0").unwrap();
        manager.materialize_to("r0", 48).unwrap();
        let certificates = manager.advance_semantic_frontier("r0", 47).unwrap();
        assert_eq!(certificates.len(), 1);
        assert_eq!(certificates[0].logical.ordinal, 0);
        assert_eq!(
            certificates[0].semantic_proof,
            SemanticProof::SlidingWindow {
                semantic_frontier: 47,
                death_boundary: 47,
            }
        );
        assert!(matches!(
            manager.materialize_to("r0", 64),
            Err(ManagerError::PoolExhausted { .. })
        ));
        let old = certificates[0].physical.clone();
        manager
            .commit_reclamation(certificates[0].certificate_id)
            .unwrap();
        let allocated = manager.materialize_to("r0", 64).unwrap();
        let new = allocated
            .iter()
            .find(|block| block.logical.class_name == "swa")
            .unwrap();
        assert_eq!(old.slot, new.physical.slot);
        assert_eq!(old.generation + 1, new.physical.generation);
    }

    #[test]
    fn execution_pin_delays_certificate() {
        let mut manager = manager(8, 3);
        manager.register_request("r0").unwrap();
        manager.materialize_to("r0", 48).unwrap();
        manager.advance_semantic_frontier("r0", 32).unwrap();
        let view = manager.submit_view("r0").unwrap();
        assert!(
            manager
                .advance_semantic_frontier("r0", 48)
                .unwrap()
                .is_empty()
        );
        let certificates = manager.complete_submission(view.submission_id).unwrap();
        assert_eq!(certificates.len(), 1);
        assert_eq!(certificates[0].execution_proof.outstanding_readers, 0);
    }

    #[test]
    fn out_of_order_completion_advances_contiguous_frontier() {
        let mut manager = manager(8, 3);
        manager.register_request("r0").unwrap();
        manager.materialize_to("r0", 32).unwrap();
        manager.advance_semantic_frontier("r0", 32).unwrap();
        let first = manager.submit_view("r0").unwrap();
        let second = manager.submit_view("r0").unwrap();
        manager.complete_submission(second.submission_id).unwrap();
        assert_eq!(manager.completed_through, 0);
        manager.complete_submission(first.submission_id).unwrap();
        assert_eq!(manager.completed_through, 2);
    }

    #[test]
    fn request_release_retires_full_and_sliding_blocks() {
        let mut manager = manager(2, 2);
        manager.register_request("r0").unwrap();
        manager.materialize_to("r0", 16).unwrap();
        manager.advance_semantic_frontier("r0", 16).unwrap();
        let certificates = manager.release_request("r0").unwrap();
        assert_eq!(certificates.len(), 2);
        assert!(
            certificates.iter().all(|certificate| {
                certificate.semantic_proof == SemanticProof::RequestReleased
            })
        );
        for certificate in certificates {
            manager
                .commit_reclamation(certificate.certificate_id)
                .unwrap();
        }
        assert_eq!(manager.stats().resident_blocks, 0);
    }

    #[test]
    fn requests_with_same_ordinal_have_distinct_physical_identity() {
        let mut manager = manager(4, 4);
        manager.register_request("r0").unwrap();
        manager.register_request("r1").unwrap();
        let first = manager.materialize_to("r0", 16).unwrap();
        let second = manager.materialize_to("r1", 16).unwrap();
        for class_name in ["full", "swa"] {
            let first_handle = &first
                .iter()
                .find(|block| block.logical.class_name == class_name)
                .unwrap()
                .physical;
            let second_handle = &second
                .iter()
                .find(|block| block.logical.class_name == class_name)
                .unwrap()
                .physical;
            assert_ne!(first_handle.slot, second_handle.slot);
        }
    }
}
