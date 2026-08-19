use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::{
    AddressProgram, BlockDomain, CellVersion, CompiledKvPlan, PlanError, RetirementProgram,
};

const DENSE_RUNTIME_SCHEMA: &str = "orbitkv.dense-runtime-artifact.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ClassId(pub u16);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct RequestLease {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct DenseLogicalBlock {
    pub request: RequestLease,
    pub class_id: ClassId,
    pub ordinal: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DenseTemporalAddress {
    pub class_id: ClassId,
    pub cell_index: u64,
    pub version: CellVersion,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DensePhysicalHandle {
    pub class_id: ClassId,
    pub slot: u64,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DenseBindingBlock {
    pub logical: DenseLogicalBlock,
    pub temporal: DenseTemporalAddress,
    pub physical: DensePhysicalHandle,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct DenseBackendHandle {
    pub domain: ClassId,
    pub index: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DenseViewBlock {
    pub logical: DenseLogicalBlock,
    pub temporal: DenseTemporalAddress,
    pub physical: DensePhysicalHandle,
    pub backend: DenseBackendHandle,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DenseClassProgram {
    pub class_id: ClassId,
    pub name: String,
    pub address: AddressProgram,
    pub retirement: RetirementProgram,
    pub block_domain: BlockDomain,
    pub cell_origin: u64,
    pub cells_per_request: u64,
    pub physical_slots: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DenseRuntimeArtifact {
    pub schema: String,
    pub artifact_fingerprint: String,
    pub plan_fingerprint: String,
    pub page_tokens: u64,
    pub maximum_requests: u32,
    pub maximum_inflight_submissions: u32,
    pub maximum_blocks_per_request: u64,
    pub classes: Vec<DenseClassProgram>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseBindingIntent {
    pub schema: &'static str,
    pub binding_id: u64,
    pub request: RequestLease,
    pub previous_boundary: u64,
    pub target_boundary: u64,
    pub resident_blocks: Vec<DenseViewBlock>,
    pub pending_blocks: Vec<DenseBindingBlock>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DensePhysicalBindingBlockReceipt {
    pub logical: DenseLogicalBlock,
    pub physical: DensePhysicalHandle,
    pub backend: DenseBackendHandle,
    pub payload_ready: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DensePhysicalBindingReceipt {
    pub schema: String,
    pub artifact_fingerprint: String,
    pub binding_id: u64,
    pub backend_transaction_id: String,
    pub blocks: Vec<DensePhysicalBindingBlockReceipt>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseView {
    pub schema: &'static str,
    pub artifact_fingerprint: String,
    pub submission_id: u64,
    pub submission_sequence: u64,
    pub request: RequestLease,
    pub semantic_frontier: u64,
    pub blocks: Vec<DenseViewBlock>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseSemanticProof {
    DeathBoundary {
        semantic_frontier: u64,
        death_boundary: u64,
    },
    RequestReleased,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DenseRetirementCertificate {
    pub schema: &'static str,
    pub artifact_fingerprint: String,
    pub certificate_id: u64,
    pub logical: DenseLogicalBlock,
    pub temporal: DenseTemporalAddress,
    pub physical: DensePhysicalHandle,
    pub backend: DenseBackendHandle,
    pub token_start: u64,
    pub token_end_exclusive: u64,
    pub semantic_proof: DenseSemanticProof,
    pub completed_through: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DensePhysicalReclamationReceipt {
    pub schema: String,
    pub artifact_fingerprint: String,
    pub certificate_id: u64,
    pub physical: DensePhysicalHandle,
    pub backend: DenseBackendHandle,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct DenseRuntimeStats {
    pub active_requests: u64,
    pub active_submissions: u64,
    pub pending_bindings: u64,
    pub pending_certificates: u64,
    pub reserved_blocks: u64,
    pub resident_blocks: u64,
    pub retiring_blocks: u64,
    pub free_request_slots: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DenseSlotPhase {
    #[default]
    Free,
    Reserved,
    Active,
    Retiring,
    Certified,
}

#[derive(Clone, Debug, Default)]
struct DenseSlot {
    generation: u64,
    occupant: Option<DenseLogicalBlock>,
    version: Option<CellVersion>,
    backend: Option<DenseBackendHandle>,
    readers: u32,
    phase: DenseSlotPhase,
    pending_binding: Option<u64>,
    pending_certificate: Option<u64>,
}

#[derive(Clone, Debug)]
struct DenseClassRuntime {
    program: DenseClassProgram,
    slots: Vec<DenseSlot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DenseBackendBinding {
    backend: DenseBackendHandle,
    physical: DensePhysicalHandle,
}

#[derive(Clone, Debug, Default)]
struct DenseRequest {
    generation: u32,
    active: bool,
    released: bool,
    semantic_frontier: u64,
    materialized_boundary: u64,
    pending_binding: Option<u64>,
}

#[derive(Clone, Debug)]
struct DenseSubmission {
    sequence: u64,
    blocks: Vec<DenseViewBlock>,
}

struct DenseBindingPreflight {
    previous_boundary: u64,
    resident_blocks: Vec<DenseViewBlock>,
    pending_blocks: Vec<DenseBindingBlock>,
}

#[derive(Clone, Debug)]
struct DenseArenaSlot<T> {
    generation: u32,
    value: Option<T>,
}

#[derive(Clone, Debug)]
struct DenseArena<T> {
    label: &'static str,
    slots: Vec<DenseArenaSlot<T>>,
    free: Vec<u32>,
    active: usize,
}

#[derive(Clone, Debug)]
struct CompletionWindow {
    completed_through: u64,
    completed_out_of_order: Vec<Option<u64>>,
}

impl<T> DenseArena<T> {
    fn new(label: &'static str, capacity: u32) -> Result<Self, DenseRuntimeError> {
        if capacity == 0 {
            return Err(DenseRuntimeError::ArenaExhausted(label));
        }
        let capacity = usize::try_from(capacity)
            .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense arena capacity"))?;
        Ok(Self {
            label,
            slots: (0..capacity)
                .map(|_| DenseArenaSlot {
                    generation: 0,
                    value: None,
                })
                .collect(),
            free: (0..u32::try_from(capacity)
                .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense arena capacity"))?)
                .rev()
                .collect(),
            active: 0,
        })
    }

    fn insert_with(&mut self, make_value: impl FnOnce(u64) -> T) -> Result<u64, DenseRuntimeError> {
        let slot = self
            .free
            .pop()
            .ok_or(DenseRuntimeError::ArenaExhausted(self.label))?;
        let state = self
            .slots
            .get_mut(slot as usize)
            .ok_or(DenseRuntimeError::ArenaExhausted(self.label))?;
        state.generation = state
            .generation
            .checked_add(1)
            .ok_or(DenseRuntimeError::ArenaGenerationExhausted(self.label))?;
        let id = encode_arena_id(slot, state.generation);
        state.value = Some(make_value(id));
        self.active += 1;
        Ok(id)
    }

    fn get(&self, id: u64) -> Option<&T> {
        let (slot, generation) = decode_arena_id(id);
        self.slots
            .get(slot as usize)
            .filter(|state| state.generation == generation)
            .and_then(|state| state.value.as_ref())
    }

    fn remove(&mut self, id: u64) -> Option<T> {
        let (slot, generation) = decode_arena_id(id);
        let state = self
            .slots
            .get_mut(slot as usize)
            .filter(|state| state.generation == generation)?;
        let value = state.value.take()?;
        self.free.push(slot);
        self.active -= 1;
        Some(value)
    }

    const fn active(&self) -> usize {
        self.active
    }
}

impl CompletionWindow {
    fn new(capacity: u32) -> Result<Self, DenseRuntimeError> {
        let capacity =
            usize::try_from(capacity).map_err(|_| DenseRuntimeError::CompletionWindowExhausted)?;
        if capacity == 0 {
            return Err(DenseRuntimeError::CompletionWindowExhausted);
        }
        Ok(Self {
            completed_through: 0,
            completed_out_of_order: vec![None; capacity],
        })
    }

    fn can_record(&self, sequence: u64) -> Result<(), DenseRuntimeError> {
        if sequence <= self.completed_through {
            return Ok(());
        }
        let capacity = u64::try_from(self.completed_out_of_order.len())
            .map_err(|_| DenseRuntimeError::CompletionWindowExhausted)?;
        if sequence > self.completed_through.saturating_add(capacity) {
            return Err(DenseRuntimeError::CompletionWindowExhausted);
        }
        if sequence > self.completed_through + 1 {
            let index = usize::try_from(sequence % capacity)
                .map_err(|_| DenseRuntimeError::CompletionWindowExhausted)?;
            if self.completed_out_of_order[index].is_some() {
                return Err(DenseRuntimeError::CompletionWindowExhausted);
            }
        }
        Ok(())
    }

    fn record(&mut self, sequence: u64) -> Result<(), DenseRuntimeError> {
        self.can_record(sequence)?;
        if sequence <= self.completed_through {
            return Ok(());
        }
        let capacity = u64::try_from(self.completed_out_of_order.len())
            .map_err(|_| DenseRuntimeError::CompletionWindowExhausted)?;
        if sequence == self.completed_through + 1 {
            self.completed_through = sequence;
            loop {
                let next = self
                    .completed_through
                    .checked_add(1)
                    .ok_or(DenseRuntimeError::SubmissionGenerationExhausted)?;
                let index = usize::try_from(next % capacity)
                    .map_err(|_| DenseRuntimeError::CompletionWindowExhausted)?;
                if self.completed_out_of_order[index] != Some(next) {
                    break;
                }
                self.completed_out_of_order[index] = None;
                self.completed_through = next;
            }
            return Ok(());
        }
        let index = usize::try_from(sequence % capacity)
            .map_err(|_| DenseRuntimeError::CompletionWindowExhausted)?;
        self.completed_out_of_order[index] = Some(sequence);
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DenseKvRuntime {
    artifact: DenseRuntimeArtifact,
    classes: Vec<DenseClassRuntime>,
    requests: Vec<DenseRequest>,
    free_requests: Vec<u32>,
    submissions: DenseArena<DenseSubmission>,
    completion: CompletionWindow,
    next_submission_sequence: u64,
    bindings: DenseArena<DenseBindingIntent>,
    certificates: DenseArena<DenseRetirementCertificate>,
    backend_bindings: Vec<Vec<Option<DenseBackendBinding>>>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DenseRuntimeError {
    #[error("dense runtime maximum_requests must be positive")]
    ZeroMaximumRequests,
    #[error("dense runtime maximum_inflight_submissions must be positive")]
    ZeroMaximumInflight,
    #[error("dense runtime maximum_blocks_per_request must be positive")]
    ZeroMaximumBlocks,
    #[error("dense runtime contains too many classes for ClassId")]
    TooManyClasses,
    #[error("dense class {0:?} has no addressable cells")]
    ZeroClassCells(String),
    #[error("dense class {0:?} physical slot count does not fit the host")]
    SlotCountTooLarge(String),
    #[error("dense request capacity is exhausted")]
    RequestCapacityExhausted,
    #[error("dense request lease is stale: {0:?}")]
    StaleRequest(RequestLease),
    #[error("dense request has already been released: {0:?}")]
    RequestReleased(RequestLease),
    #[error("dense hydration requires a fresh request lease")]
    HydrationRequiresFreshRequest,
    #[error("dense hydration boundary must be positive")]
    ZeroHydrationBoundary,
    #[error("dense request generation exhausted at slot {0}")]
    RequestGenerationExhausted(u32),
    #[error("dense materialized boundary moved backwards")]
    MaterializedBoundaryMovedBackwards,
    #[error("dense semantic frontier moved backwards")]
    SemanticFrontierMovedBackwards,
    #[error("dense semantic frontier exceeds materialized boundary")]
    SemanticFrontierBeyondMaterialized,
    #[error("dense resident frontier cannot advance before any block is bound")]
    ResidentFrontierWithoutBinding,
    #[error("dense request already has pending binding {0}")]
    PendingBinding(u64),
    #[error("dense binding generation exhausted")]
    BindingGenerationExhausted,
    #[error("unknown dense binding {0}")]
    UnknownBinding(u64),
    #[error("dense binding {0} is stale")]
    StaleBinding(u64),
    #[error("dense binding receipt does not match intent {0}")]
    MismatchedBindingReceipt(u64),
    #[error("dense backend handle is outside the compiled domain: {0:?}")]
    BackendHandleOutOfRange(DenseBackendHandle),
    #[error("dense backend binding table is exhausted for domain {0:?}")]
    BackendBindingCapacityExhausted(ClassId),
    #[error("dense backend handle is already bound: {0:?}")]
    BackendHandleCollision(DenseBackendHandle),
    #[error("dense physical handle has no committed backend binding: {0:?}")]
    MissingBackendBinding(DensePhysicalHandle),
    #[error("dense binding receipt {0} does not prove payload readiness")]
    PayloadNotReady(u64),
    #[error("dense binding backend transaction id must not be empty")]
    EmptyBackendTransactionId,
    #[error("dense logical cell is not reclaimable: {0:?}")]
    CellCollision(DenseLogicalBlock),
    #[error("dense block is not resident: {0:?}")]
    BlockNotResident(DenseLogicalBlock),
    #[error("dense physical handle is stale: {0:?}")]
    StaleHandle(DensePhysicalHandle),
    #[error("dense submission generation exhausted")]
    SubmissionGenerationExhausted,
    #[error("unknown dense submission {0}")]
    UnknownSubmission(u64),
    #[error("dense reader count overflow")]
    ReaderCountOverflow,
    #[error("dense retirement certificate generation exhausted")]
    CertificateGenerationExhausted,
    #[error("unknown dense retirement certificate {0}")]
    UnknownCertificate(u64),
    #[error("dense reclamation receipt does not match certificate {0}")]
    MismatchedReclamationReceipt(u64),
    #[error("dense request still owns resident state")]
    RequestStillResident,
    #[error("dense arithmetic overflow while calculating {0}")]
    ArithmeticOverflow(&'static str),
    #[error("dense {0} arena is exhausted")]
    ArenaExhausted(&'static str),
    #[error("dense {0} arena generation exhausted")]
    ArenaGenerationExhausted(&'static str),
    #[error("dense completion window is exhausted")]
    CompletionWindowExhausted,
    #[error("dense artifact schema is unsupported")]
    UnsupportedSchema,
    #[error("dense artifact fingerprint does not match its contents")]
    FingerprintMismatch,
    #[error("dense artifact geometry is invalid: {0}")]
    InvalidArtifactGeometry(&'static str),
    #[error("dense artifact contains duplicate class name {0:?}")]
    DuplicateArtifactClassName(String),
    #[error("dense artifact does not match plan {0:?}")]
    PlanMismatch(String),
    #[error(transparent)]
    Plan(#[from] PlanError),
}

impl DenseRuntimeArtifact {
    /// Compiles one fixed-capacity dense ownership artifact.
    ///
    /// # Errors
    ///
    /// Returns an error for zero limits, oversized IDs, invalid address
    /// geometry, or checked arithmetic overflow.
    pub fn compile(
        plan: &CompiledKvPlan,
        maximum_requests: u32,
        maximum_inflight_submissions: u32,
        maximum_blocks_per_request: u64,
    ) -> Result<Self, DenseRuntimeError> {
        if maximum_requests == 0 {
            return Err(DenseRuntimeError::ZeroMaximumRequests);
        }
        if maximum_inflight_submissions == 0 {
            return Err(DenseRuntimeError::ZeroMaximumInflight);
        }
        if maximum_blocks_per_request == 0 {
            return Err(DenseRuntimeError::ZeroMaximumBlocks);
        }
        if plan.classes.len() > usize::from(u16::MAX) {
            return Err(DenseRuntimeError::TooManyClasses);
        }
        let layout = plan.layout_program()?;
        let mut classes = Vec::with_capacity(layout.classes.len());
        for (index, class) in layout.classes.into_iter().enumerate() {
            let class_id =
                ClassId(u16::try_from(index).map_err(|_| DenseRuntimeError::TooManyClasses)?);
            let (cell_origin, cells_per_request) = addressable_cells(
                &class.address,
                &class.block_domain,
                maximum_blocks_per_request,
            );
            if cells_per_request == 0 {
                return Err(DenseRuntimeError::ZeroClassCells(class.name));
            }
            let physical_slots = cells_per_request
                .checked_mul(u64::from(maximum_requests))
                .ok_or(DenseRuntimeError::ArithmeticOverflow(
                    "dense physical slot count",
                ))?;
            classes.push(DenseClassProgram {
                class_id,
                name: class.name,
                address: class.address,
                retirement: class.retirement,
                block_domain: class.block_domain,
                cell_origin,
                cells_per_request,
                physical_slots,
            });
        }
        let total_physical_slots =
            classes.iter().try_fold(0_u64, |total, class| {
                total.checked_add(class.physical_slots).ok_or(
                    DenseRuntimeError::ArithmeticOverflow("dense total physical slots"),
                )
            })?;
        u32::try_from(total_physical_slots).map_err(|_| {
            DenseRuntimeError::ArithmeticOverflow("dense certificate arena capacity")
        })?;
        usize::try_from(total_physical_slots).map_err(|_| {
            DenseRuntimeError::ArithmeticOverflow("dense host physical slot capacity")
        })?;
        let mut artifact = Self {
            schema: DENSE_RUNTIME_SCHEMA.into(),
            artifact_fingerprint: String::new(),
            plan_fingerprint: plan.fingerprint(),
            page_tokens: plan.page_tokens,
            maximum_requests,
            maximum_inflight_submissions,
            maximum_blocks_per_request,
            classes,
        };
        artifact.artifact_fingerprint = artifact.compute_fingerprint()?;
        Ok(artifact)
    }

    /// Validates this artifact against its own fingerprint and one compiled
    /// semantic plan.
    ///
    /// # Errors
    ///
    /// Returns an error for schema, fingerprint, or plan mismatch.
    pub fn validate(&self, plan: &CompiledKvPlan) -> Result<(), DenseRuntimeError> {
        if self.schema != DENSE_RUNTIME_SCHEMA {
            return Err(DenseRuntimeError::UnsupportedSchema);
        }
        if self.plan_fingerprint != plan.fingerprint() || self.page_tokens != plan.page_tokens {
            return Err(DenseRuntimeError::PlanMismatch(
                self.plan_fingerprint.clone(),
            ));
        }
        let expected = Self::compile(
            plan,
            self.maximum_requests,
            self.maximum_inflight_submissions,
            self.maximum_blocks_per_request,
        )?;
        if expected.classes != self.classes {
            return Err(DenseRuntimeError::PlanMismatch(
                self.plan_fingerprint.clone(),
            ));
        }
        if self.compute_fingerprint()? != self.artifact_fingerprint {
            return Err(DenseRuntimeError::FingerprintMismatch);
        }
        Ok(())
    }

    fn compute_fingerprint(&self) -> Result<String, DenseRuntimeError> {
        let payload = serde_json::json!({
            "schema": self.schema,
            "plan_fingerprint": self.plan_fingerprint,
            "page_tokens": self.page_tokens,
            "maximum_requests": self.maximum_requests,
            "maximum_inflight_submissions": self.maximum_inflight_submissions,
            "maximum_blocks_per_request": self.maximum_blocks_per_request,
            "classes": self.classes,
        });
        let bytes = serde_json::to_vec(&payload)
            .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense fingerprint JSON"))?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }
}

impl DenseKvRuntime {
    /// Creates the array-backed runtime from a checked dense artifact.
    ///
    /// # Errors
    ///
    /// Returns an error if artifact geometry does not fit host arrays.
    pub fn new(artifact: DenseRuntimeArtifact) -> Result<Self, DenseRuntimeError> {
        if artifact.schema != DENSE_RUNTIME_SCHEMA {
            return Err(DenseRuntimeError::UnsupportedSchema);
        }
        if artifact.compute_fingerprint()? != artifact.artifact_fingerprint {
            return Err(DenseRuntimeError::FingerprintMismatch);
        }
        validate_artifact_geometry(&artifact)?;
        let mut classes = Vec::with_capacity(artifact.classes.len());
        for program in &artifact.classes {
            let slot_count = usize::try_from(program.physical_slots)
                .map_err(|_| DenseRuntimeError::SlotCountTooLarge(program.name.clone()))?;
            classes.push(DenseClassRuntime {
                program: program.clone(),
                slots: vec![DenseSlot::default(); slot_count],
            });
        }
        let request_count = usize::try_from(artifact.maximum_requests)
            .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense request count"))?;
        let free_requests = (0..artifact.maximum_requests).rev().collect();
        let certificate_capacity = classes.iter().try_fold(0_u64, |total, class| {
            total.checked_add(class.program.physical_slots).ok_or(
                DenseRuntimeError::ArithmeticOverflow("dense certificate capacity"),
            )
        })?;
        let certificate_capacity = u32::try_from(certificate_capacity)
            .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense certificate capacity"))?;
        let backend_bindings = classes
            .iter()
            .map(|class| vec![None; class.slots.len()])
            .collect();
        Ok(Self {
            submissions: DenseArena::new("submission", artifact.maximum_inflight_submissions)?,
            completion: CompletionWindow::new(artifact.maximum_inflight_submissions)?,
            next_submission_sequence: 1,
            bindings: DenseArena::new("binding", artifact.maximum_requests)?,
            certificates: DenseArena::new("certificate", certificate_capacity)?,
            backend_bindings,
            artifact,
            classes,
            requests: vec![DenseRequest::default(); request_count],
            free_requests,
        })
    }

    #[must_use]
    pub fn artifact(&self) -> &DenseRuntimeArtifact {
        &self.artifact
    }

    /// Acquires one generation-checked request slot.
    ///
    /// # Errors
    ///
    /// Returns an error when capacity or generation is exhausted.
    pub fn acquire_request(&mut self) -> Result<RequestLease, DenseRuntimeError> {
        let slot = self
            .free_requests
            .pop()
            .ok_or(DenseRuntimeError::RequestCapacityExhausted)?;
        let request = self
            .requests
            .get_mut(slot as usize)
            .ok_or(DenseRuntimeError::RequestCapacityExhausted)?;
        request.generation = request
            .generation
            .checked_add(1)
            .ok_or(DenseRuntimeError::RequestGenerationExhausted(slot))?;
        request.active = true;
        request.released = false;
        request.semantic_frontier = 0;
        request.materialized_boundary = 0;
        request.pending_binding = None;
        Ok(RequestLease {
            slot,
            generation: request.generation,
        })
    }

    /// Executes prepare/commit with an in-memory ready receipt.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid geometry or unsafe cell reuse.
    pub fn materialize_to(
        &mut self,
        request: RequestLease,
        boundary: u64,
    ) -> Result<Vec<DenseViewBlock>, DenseRuntimeError> {
        let intent = self.prepare_binding_to(request, boundary)?;
        let receipt = DensePhysicalBindingReceipt {
            schema: "orbitkv.dense-physical-binding-receipt.v1".into(),
            artifact_fingerprint: self.artifact.artifact_fingerprint.clone(),
            binding_id: intent.binding_id,
            backend_transaction_id: format!("dense-reference:{}", intent.binding_id),
            blocks: intent
                .pending_blocks
                .iter()
                .map(|block| DensePhysicalBindingBlockReceipt {
                    logical: block.logical,
                    physical: block.physical,
                    backend: DenseBackendHandle {
                        domain: block.physical.class_id,
                        index: block.physical.slot,
                    },
                    payload_ready: true,
                })
                .collect(),
        };
        self.commit_binding(&receipt)
    }

    /// Reserves all new cells without publishing them.
    ///
    /// # Errors
    ///
    /// Returns an error without mutation if preflight fails.
    pub fn prepare_binding_to(
        &mut self,
        lease: RequestLease,
        boundary: u64,
    ) -> Result<DenseBindingIntent, DenseRuntimeError> {
        let DenseBindingPreflight {
            previous_boundary,
            resident_blocks,
            pending_blocks,
        } = self.preflight_binding(lease, boundary)?;
        let binding_id = self.bindings.insert_with(|binding_id| DenseBindingIntent {
            schema: "orbitkv.dense-binding-intent.v1",
            binding_id,
            request: lease,
            previous_boundary,
            target_boundary: boundary,
            resident_blocks,
            pending_blocks,
        })?;
        let intent = self
            .bindings
            .get(binding_id)
            .cloned()
            .ok_or(DenseRuntimeError::UnknownBinding(binding_id))?;
        if let Err(error) = self.reserve_binding_slots(binding_id, &intent.pending_blocks) {
            self.bindings.remove(binding_id);
            return Err(error);
        }
        self.request_mut(lease)?.pending_binding = Some(binding_id);
        Ok(intent)
    }

    /// Reserves only the compiler-proven live set at one continuation boundary.
    ///
    /// This entry point is for Capsule or disaggregated restore into a fresh
    /// request lease. It skips semantically dead history instead of replaying
    /// every prior allocation through a finite periodic address machine.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-fresh request, invalid boundary, or unsafe
    /// live-cell collision.
    pub fn prepare_hydration_to(
        &mut self,
        lease: RequestLease,
        boundary: u64,
    ) -> Result<DenseBindingIntent, DenseRuntimeError> {
        let request = self.request(lease)?;
        if request.released {
            return Err(DenseRuntimeError::RequestReleased(lease));
        }
        if request.materialized_boundary != 0 || request.semantic_frontier != 0 {
            return Err(DenseRuntimeError::HydrationRequiresFreshRequest);
        }
        if let Some(binding_id) = request.pending_binding {
            return Err(DenseRuntimeError::PendingBinding(binding_id));
        }
        if boundary == 0 {
            return Err(DenseRuntimeError::ZeroHydrationBoundary);
        }
        let pending_blocks = self.hydration_blocks(lease, boundary)?;
        let binding_id = self.bindings.insert_with(|binding_id| DenseBindingIntent {
            schema: "orbitkv.dense-binding-intent.v1",
            binding_id,
            request: lease,
            previous_boundary: 0,
            target_boundary: boundary,
            resident_blocks: Vec::new(),
            pending_blocks,
        })?;
        let intent = self
            .bindings
            .get(binding_id)
            .cloned()
            .ok_or(DenseRuntimeError::UnknownBinding(binding_id))?;
        if let Err(error) = self.reserve_binding_slots(binding_id, &intent.pending_blocks) {
            self.bindings.remove(binding_id);
            return Err(error);
        }
        self.request_mut(lease)?.pending_binding = Some(binding_id);
        Ok(intent)
    }

    fn hydration_blocks(
        &self,
        lease: RequestLease,
        boundary: u64,
    ) -> Result<Vec<DenseBindingBlock>, DenseRuntimeError> {
        let last = (boundary - 1) / self.artifact.page_tokens;
        if last >= self.artifact.maximum_blocks_per_request {
            return Err(DenseRuntimeError::ArithmeticOverflow(
                "dense maximum hydration boundary",
            ));
        }
        let mut pending = Vec::new();
        for class_index in 0..self.classes.len() {
            let program = &self.classes[class_index].program;
            for ordinal in program.block_domain.start_block..=last {
                if !program.block_domain.contains(ordinal)
                    || program
                        .retirement
                        .death_boundary(self.artifact.page_tokens, ordinal)?
                        .is_some_and(|death| death <= boundary)
                {
                    continue;
                }
                let (logical, temporal, _, slot_index) =
                    self.locate_cell(lease, class_index, ordinal)?;
                let slot = self.classes[class_index].slots.get(slot_index).ok_or(
                    DenseRuntimeError::ArithmeticOverflow("dense hydration slot index"),
                )?;
                if slot.phase != DenseSlotPhase::Free
                    || slot.occupant.is_some()
                    || slot.backend.is_some()
                {
                    return Err(DenseRuntimeError::CellCollision(logical));
                }
                pending.push(DenseBindingBlock {
                    logical,
                    temporal,
                    physical: DensePhysicalHandle {
                        class_id: program.class_id,
                        slot: slot_index as u64,
                        generation: slot.generation.checked_add(1).ok_or(
                            DenseRuntimeError::ArithmeticOverflow("dense slot generation"),
                        )?,
                    },
                });
            }
        }
        Ok(pending)
    }

    fn preflight_binding(
        &self,
        lease: RequestLease,
        boundary: u64,
    ) -> Result<DenseBindingPreflight, DenseRuntimeError> {
        let request = self.request(lease)?;
        if request.released {
            return Err(DenseRuntimeError::RequestReleased(lease));
        }
        if let Some(binding_id) = request.pending_binding {
            return Err(DenseRuntimeError::PendingBinding(binding_id));
        }
        if boundary < request.materialized_boundary {
            return Err(DenseRuntimeError::MaterializedBoundaryMovedBackwards);
        }
        let previous_boundary = request.materialized_boundary;
        let mut resident_blocks = Vec::new();
        let mut pending_blocks = Vec::new();
        if previous_boundary < boundary {
            let first = previous_boundary / self.artifact.page_tokens;
            let last = (boundary - 1) / self.artifact.page_tokens;
            for ordinal in first..=last {
                for class_index in 0..self.classes.len() {
                    let program = &self.classes[class_index].program;
                    if !program.block_domain.contains(ordinal) {
                        continue;
                    }
                    if ordinal >= self.artifact.maximum_blocks_per_request {
                        return Err(DenseRuntimeError::ArithmeticOverflow(
                            "dense maximum block boundary",
                        ));
                    }
                    let (logical, temporal, class_id, slot_index) =
                        self.locate_cell(lease, class_index, ordinal)?;
                    let slot = self
                        .classes
                        .get(class_index)
                        .and_then(|class| class.slots.get(slot_index))
                        .ok_or(DenseRuntimeError::ArithmeticOverflow(
                            "dense physical slot index",
                        ))?;
                    if slot.occupant == Some(logical)
                        && slot.version == Some(temporal.version)
                        && matches!(
                            slot.phase,
                            DenseSlotPhase::Active
                                | DenseSlotPhase::Retiring
                                | DenseSlotPhase::Certified
                        )
                    {
                        let backend =
                            slot.backend
                                .ok_or(DenseRuntimeError::MissingBackendBinding(
                                    DensePhysicalHandle {
                                        class_id,
                                        slot: slot_index as u64,
                                        generation: slot.generation,
                                    },
                                ))?;
                        resident_blocks.push(DenseViewBlock {
                            logical,
                            temporal,
                            physical: DensePhysicalHandle {
                                class_id,
                                slot: slot_index as u64,
                                generation: slot.generation,
                            },
                            backend,
                        });
                        continue;
                    }
                    if slot.phase != DenseSlotPhase::Free
                        || slot.occupant.is_some()
                        || slot.backend.is_some()
                    {
                        return Err(DenseRuntimeError::CellCollision(logical));
                    }
                    let generation = slot.generation.checked_add(1).ok_or(
                        DenseRuntimeError::ArithmeticOverflow("dense slot generation"),
                    )?;
                    pending_blocks.push(DenseBindingBlock {
                        logical,
                        temporal,
                        physical: DensePhysicalHandle {
                            class_id,
                            slot: slot_index as u64,
                            generation,
                        },
                    });
                }
            }
        }
        Ok(DenseBindingPreflight {
            previous_boundary,
            resident_blocks,
            pending_blocks,
        })
    }

    fn reserve_binding_slots(
        &mut self,
        binding_id: u64,
        pending_blocks: &[DenseBindingBlock],
    ) -> Result<(), DenseRuntimeError> {
        for block in pending_blocks {
            let class = self
                .classes
                .get(usize::from(block.physical.class_id.0))
                .filter(|class| class.program.class_id == block.physical.class_id)
                .ok_or(DenseRuntimeError::StaleHandle(block.physical))?;
            let slot_index = usize::try_from(block.physical.slot)
                .map_err(|_| DenseRuntimeError::SlotCountTooLarge(class.program.name.clone()))?;
            let slot = class
                .slots
                .get(slot_index)
                .ok_or(DenseRuntimeError::StaleHandle(block.physical))?;
            if slot.phase != DenseSlotPhase::Free
                || slot.occupant.is_some()
                || slot.backend.is_some()
                || slot
                    .generation
                    .checked_add(1)
                    .is_none_or(|generation| generation != block.physical.generation)
            {
                return Err(DenseRuntimeError::StaleBinding(binding_id));
            }
        }
        for block in pending_blocks {
            let class = self
                .classes
                .get_mut(usize::from(block.physical.class_id.0))
                .filter(|class| class.program.class_id == block.physical.class_id)
                .ok_or(DenseRuntimeError::StaleHandle(block.physical))?;
            let slot_index = usize::try_from(block.physical.slot)
                .map_err(|_| DenseRuntimeError::SlotCountTooLarge(class.program.name.clone()))?;
            let slot = class
                .slots
                .get_mut(slot_index)
                .ok_or(DenseRuntimeError::StaleHandle(block.physical))?;
            if slot.phase != DenseSlotPhase::Free
                || slot.occupant.is_some()
                || slot.backend.is_some()
                || slot
                    .generation
                    .checked_add(1)
                    .is_none_or(|generation| generation != block.physical.generation)
            {
                return Err(DenseRuntimeError::StaleBinding(binding_id));
            }
            slot.generation = block.physical.generation;
            slot.occupant = Some(block.logical);
            slot.version = Some(block.temporal.version);
            slot.phase = DenseSlotPhase::Reserved;
            slot.pending_binding = Some(binding_id);
        }
        Ok(())
    }

    fn backend_binding(
        &self,
        backend: DenseBackendHandle,
    ) -> Result<Option<DensePhysicalHandle>, DenseRuntimeError> {
        self.class(backend.domain)?;
        self.backend_bindings
            .get(usize::from(backend.domain.0))
            .ok_or(DenseRuntimeError::BackendHandleOutOfRange(backend))
            .map(|bindings| {
                bindings
                    .iter()
                    .flatten()
                    .find(|binding| binding.backend == backend)
                    .map(|binding| binding.physical)
            })
    }

    fn insert_backend_binding(
        &mut self,
        backend: DenseBackendHandle,
        physical: DensePhysicalHandle,
    ) -> Result<(), DenseRuntimeError> {
        self.class(backend.domain)?;
        let bindings = self
            .backend_bindings
            .get_mut(usize::from(backend.domain.0))
            .ok_or(DenseRuntimeError::BackendHandleOutOfRange(backend))?;
        if bindings
            .iter()
            .flatten()
            .any(|binding| binding.backend == backend)
        {
            return Err(DenseRuntimeError::BackendHandleCollision(backend));
        }
        let binding = bindings
            .iter_mut()
            .find(|binding| binding.is_none())
            .ok_or(DenseRuntimeError::BackendBindingCapacityExhausted(
                backend.domain,
            ))?;
        *binding = Some(DenseBackendBinding { backend, physical });
        Ok(())
    }

    fn remove_backend_binding(
        &mut self,
        backend: DenseBackendHandle,
        physical: DensePhysicalHandle,
    ) -> Result<(), DenseRuntimeError> {
        self.class(backend.domain)?;
        let binding = self
            .backend_bindings
            .get_mut(usize::from(backend.domain.0))
            .and_then(|bindings| {
                bindings.iter_mut().find(|binding| {
                    binding.is_some_and(|binding| {
                        binding.backend == backend && binding.physical == physical
                    })
                })
            })
            .ok_or(DenseRuntimeError::BackendHandleOutOfRange(backend))?;
        *binding = None;
        Ok(())
    }

    /// Atomically publishes a complete ready receipt.
    ///
    /// # Errors
    ///
    /// Returns an error while leaving the binding pending on any mismatch.
    pub fn commit_binding(
        &mut self,
        receipt: &DensePhysicalBindingReceipt,
    ) -> Result<Vec<DenseViewBlock>, DenseRuntimeError> {
        let intent = self
            .bindings
            .get(receipt.binding_id)
            .cloned()
            .ok_or(DenseRuntimeError::UnknownBinding(receipt.binding_id))?;
        self.validate_binding_receipt(&intent, receipt)?;

        let mut published = intent.resident_blocks;
        for actual in &receipt.blocks {
            self.insert_backend_binding(actual.backend, actual.physical)?;
            let slot = self.slot_mut(actual.physical)?;
            slot.phase = DenseSlotPhase::Active;
            slot.pending_binding = None;
            slot.backend = Some(actual.backend);
            let temporal = intent
                .pending_blocks
                .iter()
                .find(|block| block.logical == actual.logical)
                .map(|block| block.temporal)
                .ok_or(DenseRuntimeError::MismatchedBindingReceipt(
                    receipt.binding_id,
                ))?;
            published.push(DenseViewBlock {
                logical: actual.logical,
                temporal,
                physical: actual.physical,
                backend: actual.backend,
            });
        }
        let request = self.request_mut(intent.request)?;
        request.materialized_boundary = intent.target_boundary;
        request.pending_binding = None;
        self.bindings
            .remove(receipt.binding_id)
            .ok_or(DenseRuntimeError::UnknownBinding(receipt.binding_id))?;
        published.sort_by_key(|block| (block.logical.class_id, block.logical.ordinal));
        Ok(published)
    }

    fn validate_binding_receipt(
        &self,
        intent: &DenseBindingIntent,
        receipt: &DensePhysicalBindingReceipt,
    ) -> Result<(), DenseRuntimeError> {
        if receipt.schema != "orbitkv.dense-physical-binding-receipt.v1"
            || receipt.artifact_fingerprint != self.artifact.artifact_fingerprint
            || receipt.backend_transaction_id.is_empty()
        {
            return if receipt.backend_transaction_id.is_empty() {
                Err(DenseRuntimeError::EmptyBackendTransactionId)
            } else {
                Err(DenseRuntimeError::MismatchedBindingReceipt(
                    receipt.binding_id,
                ))
            };
        }
        if receipt.blocks.len() != intent.pending_blocks.len() {
            return Err(DenseRuntimeError::MismatchedBindingReceipt(
                receipt.binding_id,
            ));
        }
        let mut backends = receipt
            .blocks
            .iter()
            .map(|block| block.backend)
            .collect::<Vec<_>>();
        backends.sort();
        if backends.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(DenseRuntimeError::MismatchedBindingReceipt(
                receipt.binding_id,
            ));
        }
        for expected in &intent.pending_blocks {
            let mut matches = receipt.blocks.iter().filter(|actual| {
                actual.logical == expected.logical && actual.physical == expected.physical
            });
            let Some(actual) = matches.next() else {
                return Err(DenseRuntimeError::MismatchedBindingReceipt(
                    receipt.binding_id,
                ));
            };
            if matches.next().is_some() {
                return Err(DenseRuntimeError::MismatchedBindingReceipt(
                    receipt.binding_id,
                ));
            }
            if actual.backend.domain != expected.logical.class_id {
                return Err(DenseRuntimeError::MismatchedBindingReceipt(
                    receipt.binding_id,
                ));
            }
            if self.backend_binding(actual.backend)?.is_some() {
                return Err(DenseRuntimeError::BackendHandleCollision(actual.backend));
            }
        }
        if receipt.blocks.iter().any(|block| !block.payload_ready) {
            return Err(DenseRuntimeError::PayloadNotReady(receipt.binding_id));
        }
        let request = self.request(intent.request)?;
        if request.released
            || request.materialized_boundary != intent.previous_boundary
            || request.pending_binding != Some(receipt.binding_id)
        {
            return Err(DenseRuntimeError::StaleBinding(receipt.binding_id));
        }
        for block in &intent.pending_blocks {
            let slot = self.slot(block.physical)?;
            if slot.phase != DenseSlotPhase::Reserved
                || slot.pending_binding != Some(receipt.binding_id)
                || slot.occupant != Some(block.logical)
                || slot.version != Some(block.temporal.version)
                || slot.backend.is_some()
            {
                return Err(DenseRuntimeError::StaleBinding(receipt.binding_id));
            }
        }
        Ok(())
    }

    /// Aborts an invisible binding while consuming slot generations.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown or stale binding.
    pub fn abort_binding(&mut self, binding_id: u64) -> Result<(), DenseRuntimeError> {
        let intent = self
            .bindings
            .get(binding_id)
            .cloned()
            .ok_or(DenseRuntimeError::UnknownBinding(binding_id))?;
        for block in &intent.pending_blocks {
            let slot = self.slot(block.physical)?;
            if slot.phase != DenseSlotPhase::Reserved || slot.pending_binding != Some(binding_id) {
                return Err(DenseRuntimeError::StaleBinding(binding_id));
            }
        }
        for block in &intent.pending_blocks {
            let slot = self.slot_mut(block.physical)?;
            slot.occupant = None;
            slot.version = None;
            slot.backend = None;
            slot.phase = DenseSlotPhase::Free;
            slot.pending_binding = None;
        }
        self.request_mut(intent.request)?.pending_binding = None;
        self.bindings
            .remove(binding_id)
            .ok_or(DenseRuntimeError::UnknownBinding(binding_id))?;
        Ok(())
    }

    /// Advances semantic time and returns newly safe retirement certificates.
    ///
    /// # Errors
    ///
    /// Returns an error for stale requests or invalid frontier movement.
    pub fn advance_semantic_frontier(
        &mut self,
        lease: RequestLease,
        boundary: u64,
    ) -> Result<Vec<DenseRetirementCertificate>, DenseRuntimeError> {
        let request = self.request(lease)?;
        if request.released {
            return Err(DenseRuntimeError::RequestReleased(lease));
        }
        if boundary < request.semantic_frontier {
            return Err(DenseRuntimeError::SemanticFrontierMovedBackwards);
        }
        if boundary > request.materialized_boundary {
            return Err(DenseRuntimeError::SemanticFrontierBeyondMaterialized);
        }
        self.request_mut(lease)?.semantic_frontier = boundary;
        self.mark_retirements(lease)
    }

    /// Advances logical time within an already committed physical block.
    ///
    /// # Errors
    ///
    /// Returns an error if the boundary crosses into an unbound logical block
    /// or if either frontier would move backwards.
    pub fn advance_resident_frontier(
        &mut self,
        lease: RequestLease,
        boundary: u64,
    ) -> Result<Vec<DenseRetirementCertificate>, DenseRuntimeError> {
        let request = self.request(lease)?;
        if request.released {
            return Err(DenseRuntimeError::RequestReleased(lease));
        }
        if let Some(binding_id) = request.pending_binding {
            return Err(DenseRuntimeError::PendingBinding(binding_id));
        }
        if boundary < request.materialized_boundary {
            return Err(DenseRuntimeError::MaterializedBoundaryMovedBackwards);
        }
        if boundary < request.semantic_frontier {
            return Err(DenseRuntimeError::SemanticFrontierMovedBackwards);
        }
        if boundary == request.materialized_boundary {
            self.request_mut(lease)?.semantic_frontier = boundary;
            return self.mark_retirements(lease);
        }
        if request.materialized_boundary == 0 {
            return Err(DenseRuntimeError::ResidentFrontierWithoutBinding);
        }
        let previous_ordinal = (request.materialized_boundary - 1) / self.artifact.page_tokens;
        let target_ordinal = (boundary - 1) / self.artifact.page_tokens;
        if previous_ordinal != target_ordinal {
            return Err(DenseRuntimeError::SemanticFrontierBeyondMaterialized);
        }
        for class in &self.classes {
            if class.program.block_domain.contains(target_ordinal) {
                self.block(DenseLogicalBlock {
                    request: lease,
                    class_id: class.program.class_id,
                    ordinal: target_ordinal,
                })?;
            }
        }
        let request = self.request_mut(lease)?;
        request.materialized_boundary = boundary;
        request.semantic_frontier = boundary;
        self.mark_retirements(lease)
    }

    /// Pins the dense live-set view for one submission.
    ///
    /// # Errors
    ///
    /// Returns an error for stale state or exhausted IDs.
    pub fn submit_view(&mut self, lease: RequestLease) -> Result<DenseView, DenseRuntimeError> {
        let request = self.request(lease)?;
        if request.released {
            return Err(DenseRuntimeError::RequestReleased(lease));
        }
        let frontier = request.semantic_frontier;
        let mut blocks = self.live_blocks(lease, frontier)?;
        blocks.sort_by_key(|block| (block.logical.class_id, block.logical.ordinal));
        let submission_sequence = self.next_submission_sequence;
        let next_submission_sequence = submission_sequence
            .checked_add(1)
            .ok_or(DenseRuntimeError::SubmissionGenerationExhausted)?;
        for block in &blocks {
            let slot = self.slot(block.physical)?;
            if slot.phase != DenseSlotPhase::Active || slot.occupant != Some(block.logical) {
                return Err(DenseRuntimeError::BlockNotResident(block.logical));
            }
            if slot.readers == u32::MAX {
                return Err(DenseRuntimeError::ReaderCountOverflow);
            }
        }
        for block in &blocks {
            self.slot_mut(block.physical)?.readers += 1;
        }
        let submission_id = match self.submissions.insert_with(|_| DenseSubmission {
            sequence: submission_sequence,
            blocks: blocks.clone(),
        }) {
            Ok(submission_id) => submission_id,
            Err(error) => {
                for block in &blocks {
                    self.slot_mut(block.physical)?.readers -= 1;
                }
                return Err(error);
            }
        };
        self.next_submission_sequence = next_submission_sequence;
        Ok(DenseView {
            schema: "orbitkv.dense-view.v1",
            artifact_fingerprint: self.artifact.artifact_fingerprint.clone(),
            submission_id,
            submission_sequence,
            request: lease,
            semantic_frontier: frontier,
            blocks,
        })
    }

    /// Completes one submission and certifies newly unblocked retired blocks.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown submissions or stale handles.
    pub fn complete_submission(
        &mut self,
        submission_id: u64,
    ) -> Result<Vec<DenseRetirementCertificate>, DenseRuntimeError> {
        let submission = self
            .submissions
            .get(submission_id)
            .cloned()
            .ok_or(DenseRuntimeError::UnknownSubmission(submission_id))?;
        for block in &submission.blocks {
            let slot = self.slot(block.physical)?;
            if slot.occupant != Some(block.logical)
                || slot.generation != block.physical.generation
                || slot.backend != Some(block.backend)
                || self.backend_binding(block.backend)? != Some(block.physical)
                || slot.readers == 0
            {
                return Err(DenseRuntimeError::StaleHandle(block.physical));
            }
        }
        self.completion.can_record(submission.sequence)?;
        self.submissions
            .remove(submission_id)
            .ok_or(DenseRuntimeError::UnknownSubmission(submission_id))?;
        let DenseSubmission { sequence, blocks } = submission;
        let mut affected = Vec::new();
        for block in blocks {
            self.slot_mut(block.physical)?.readers -= 1;
            affected.push(block.logical);
        }
        self.completion.record(sequence)?;
        affected.sort();
        affected.dedup();
        let mut certificates = Vec::new();
        for logical in affected {
            if let Some(certificate) = self.try_certify(logical)? {
                certificates.push(certificate);
            }
        }
        Ok(certificates)
    }

    /// Marks all request blocks semantically dead.
    ///
    /// # Errors
    ///
    /// Returns an error for stale or already released requests.
    pub fn release_request(
        &mut self,
        lease: RequestLease,
    ) -> Result<Vec<DenseRetirementCertificate>, DenseRuntimeError> {
        let request = self.request(lease)?;
        if request.released {
            return Err(DenseRuntimeError::RequestReleased(lease));
        }
        if let Some(binding_id) = request.pending_binding {
            return Err(DenseRuntimeError::PendingBinding(binding_id));
        }
        self.request_mut(lease)?.released = true;
        self.mark_retirements(lease)
    }

    /// Commits physical reclamation for one certified slot.
    ///
    /// # Errors
    ///
    /// Returns an error for stale or mismatched receipts.
    pub fn commit_reclamation(
        &mut self,
        receipt: &DensePhysicalReclamationReceipt,
    ) -> Result<(), DenseRuntimeError> {
        self.commit_reclamations(std::slice::from_ref(receipt))
    }

    /// Atomically commits physical reclamation for a complete receipt batch.
    ///
    /// # Errors
    ///
    /// Returns an error without freeing any slot if a receipt is duplicated,
    /// stale, or mismatched.
    pub fn commit_reclamations(
        &mut self,
        receipts: &[DensePhysicalReclamationReceipt],
    ) -> Result<(), DenseRuntimeError> {
        let mut certificate_ids = receipts
            .iter()
            .map(|receipt| receipt.certificate_id)
            .collect::<Vec<_>>();
        certificate_ids.sort_unstable();
        if let Some(duplicate) = certificate_ids
            .windows(2)
            .find(|pair| pair[0] == pair[1])
            .map(|pair| pair[0])
        {
            return Err(DenseRuntimeError::MismatchedReclamationReceipt(duplicate));
        }
        let mut certificates = Vec::with_capacity(receipts.len());
        for receipt in receipts {
            let certificate = self
                .certificates
                .get(receipt.certificate_id)
                .cloned()
                .ok_or(DenseRuntimeError::UnknownCertificate(
                    receipt.certificate_id,
                ))?;
            if receipt.schema != "orbitkv.dense-physical-reclamation-receipt.v1"
                || receipt.artifact_fingerprint != self.artifact.artifact_fingerprint
                || receipt.physical != certificate.physical
                || receipt.backend != certificate.backend
            {
                return Err(DenseRuntimeError::MismatchedReclamationReceipt(
                    receipt.certificate_id,
                ));
            }
            let slot = self.slot(certificate.physical)?;
            if slot.phase != DenseSlotPhase::Certified
                || slot.pending_certificate != Some(receipt.certificate_id)
                || slot.readers != 0
                || slot.occupant != Some(certificate.logical)
                || slot.backend != Some(certificate.backend)
                || self.backend_binding(certificate.backend)? != Some(certificate.physical)
            {
                return Err(DenseRuntimeError::MismatchedReclamationReceipt(
                    receipt.certificate_id,
                ));
            }
            certificates.push(certificate);
        }
        for certificate in certificates {
            let slot = self.slot_mut(certificate.physical)?;
            slot.occupant = None;
            slot.version = None;
            slot.backend = None;
            slot.phase = DenseSlotPhase::Free;
            slot.pending_certificate = None;
            self.remove_backend_binding(certificate.backend, certificate.physical)?;
            self.certificates.remove(certificate.certificate_id).ok_or(
                DenseRuntimeError::UnknownCertificate(certificate.certificate_id),
            )?;
        }
        Ok(())
    }

    /// Returns a request slot to the dense lease pool after all physical state
    /// and readers are gone.
    ///
    /// # Errors
    ///
    /// Returns an error while request state is still resident.
    pub fn recycle_request(&mut self, lease: RequestLease) -> Result<(), DenseRuntimeError> {
        let request = self.request(lease)?;
        if !request.released {
            return Err(DenseRuntimeError::RequestStillResident);
        }
        for class in &self.classes {
            let start = request_slot_start(&class.program, lease)?;
            let end = start
                + usize::try_from(class.program.cells_per_request).map_err(|_| {
                    DenseRuntimeError::SlotCountTooLarge(class.program.name.clone())
                })?;
            if class.slots[start..end].iter().any(|slot| {
                slot.phase != DenseSlotPhase::Free
                    || slot.occupant.is_some()
                    || slot.backend.is_some()
                    || slot.readers != 0
            }) {
                return Err(DenseRuntimeError::RequestStillResident);
            }
        }
        let request = self.request_mut(lease)?;
        request.active = false;
        request.released = false;
        self.free_requests.push(lease.slot);
        Ok(())
    }

    #[must_use]
    pub fn stats(&self) -> DenseRuntimeStats {
        let mut stats = DenseRuntimeStats {
            active_requests: self
                .requests
                .iter()
                .filter(|request| request.active)
                .count() as u64,
            active_submissions: self.submissions.active() as u64,
            pending_bindings: self.bindings.active() as u64,
            pending_certificates: self.certificates.active() as u64,
            free_request_slots: self.free_requests.len() as u64,
            ..DenseRuntimeStats::default()
        };
        for slot in self.classes.iter().flat_map(|class| &class.slots) {
            match slot.phase {
                DenseSlotPhase::Reserved => stats.reserved_blocks += 1,
                DenseSlotPhase::Active => stats.resident_blocks += 1,
                DenseSlotPhase::Retiring | DenseSlotPhase::Certified => {
                    stats.resident_blocks += 1;
                    stats.retiring_blocks += 1;
                }
                DenseSlotPhase::Free => {}
            }
        }
        stats
    }

    #[must_use]
    pub fn resident_blocks(&self, lease: RequestLease) -> Vec<DenseViewBlock> {
        if self.request(lease).is_err() {
            return Vec::new();
        }
        let mut blocks = Vec::new();
        for class in &self.classes {
            let Ok(start) = request_slot_start(&class.program, lease) else {
                continue;
            };
            let Ok(count) = usize::try_from(class.program.cells_per_request) else {
                continue;
            };
            for (offset, slot) in class.slots[start..start + count].iter().enumerate() {
                let Some(logical) = slot.occupant else {
                    continue;
                };
                let Some(backend) = slot.backend else {
                    continue;
                };
                blocks.push(DenseViewBlock {
                    logical,
                    temporal: DenseTemporalAddress {
                        class_id: class.program.class_id,
                        cell_index: class.program.cell_origin + offset as u64,
                        version: slot.version.unwrap_or(CellVersion { cycle: 0 }),
                    },
                    physical: DensePhysicalHandle {
                        class_id: class.program.class_id,
                        slot: (start + offset) as u64,
                        generation: slot.generation,
                    },
                    backend,
                });
            }
        }
        blocks.sort_by_key(|block| (block.logical.class_id, block.logical.ordinal));
        blocks
    }

    fn mark_retirements(
        &mut self,
        lease: RequestLease,
    ) -> Result<Vec<DenseRetirementCertificate>, DenseRuntimeError> {
        let released = self.request(lease)?.released;
        let frontier = self.request(lease)?.semantic_frontier;
        let blocks = self.resident_blocks(lease);
        let mut certificates = Vec::new();
        for block in blocks {
            let dead = released || self.is_dead(block.logical, frontier)?;
            if !dead {
                continue;
            }
            let slot = self.slot_mut(block.physical)?;
            if slot.phase == DenseSlotPhase::Active {
                slot.phase = DenseSlotPhase::Retiring;
            }
            if let Some(certificate) = self.try_certify(block.logical)? {
                certificates.push(certificate);
            }
        }
        Ok(certificates)
    }

    fn try_certify(
        &mut self,
        logical: DenseLogicalBlock,
    ) -> Result<Option<DenseRetirementCertificate>, DenseRuntimeError> {
        let block = self.block(logical)?;
        let slot = self.slot(block.physical)?;
        if slot.phase != DenseSlotPhase::Retiring || slot.readers != 0 {
            return Ok(None);
        }
        let request = self.request(logical.request)?;
        let semantic_proof = if request.released {
            DenseSemanticProof::RequestReleased
        } else {
            let death_boundary = self
                .class(logical.class_id)?
                .program
                .retirement
                .death_boundary(self.artifact.page_tokens, logical.ordinal)?
                .ok_or(DenseRuntimeError::BlockNotResident(logical))?;
            DenseSemanticProof::DeathBoundary {
                semantic_frontier: request.semantic_frontier,
                death_boundary,
            }
        };
        let token_start = logical
            .ordinal
            .checked_mul(self.artifact.page_tokens)
            .ok_or(DenseRuntimeError::ArithmeticOverflow("dense token start"))?;
        let token_end_exclusive = token_start
            .checked_add(self.artifact.page_tokens)
            .ok_or(DenseRuntimeError::ArithmeticOverflow("dense token end"))?;
        let artifact_fingerprint = self.artifact.artifact_fingerprint.clone();
        let completed_through = self.completion.completed_through;
        let certificate_id =
            self.certificates
                .insert_with(|certificate_id| DenseRetirementCertificate {
                    schema: "orbitkv.dense-retirement-certificate.v1",
                    artifact_fingerprint,
                    certificate_id,
                    logical,
                    temporal: block.temporal,
                    physical: block.physical,
                    backend: block.backend,
                    token_start,
                    token_end_exclusive,
                    semantic_proof,
                    completed_through,
                })?;
        let certificate = self
            .certificates
            .get(certificate_id)
            .cloned()
            .ok_or(DenseRuntimeError::UnknownCertificate(certificate_id))?;
        let slot = self.slot_mut(block.physical)?;
        slot.phase = DenseSlotPhase::Certified;
        slot.pending_certificate = Some(certificate_id);
        Ok(Some(certificate))
    }

    fn live_blocks(
        &self,
        lease: RequestLease,
        frontier: u64,
    ) -> Result<Vec<DenseViewBlock>, DenseRuntimeError> {
        let mut blocks = Vec::new();
        for block in self.resident_blocks(lease) {
            if !self.is_dead(block.logical, frontier)? {
                blocks.push(block);
            }
        }
        Ok(blocks)
    }

    fn is_dead(
        &self,
        logical: DenseLogicalBlock,
        frontier: u64,
    ) -> Result<bool, DenseRuntimeError> {
        Ok(self
            .class(logical.class_id)?
            .program
            .retirement
            .death_boundary(self.artifact.page_tokens, logical.ordinal)?
            .is_some_and(|death| frontier >= death))
    }

    fn locate_cell(
        &self,
        lease: RequestLease,
        class_index: usize,
        ordinal: u64,
    ) -> Result<(DenseLogicalBlock, DenseTemporalAddress, ClassId, usize), DenseRuntimeError> {
        let class = self
            .classes
            .get(class_index)
            .ok_or(DenseRuntimeError::ArithmeticOverflow("dense class index"))?;
        let (cell_index, version) = class.program.address.evaluate_dense(ordinal)?;
        let local_cell = cell_index.checked_sub(class.program.cell_origin).ok_or(
            DenseRuntimeError::ArithmeticOverflow("dense cell origin subtraction"),
        )?;
        if local_cell >= class.program.cells_per_request {
            return Err(DenseRuntimeError::ArithmeticOverflow("dense cell index"));
        }
        let start = request_slot_start(&class.program, lease)?;
        let cell = usize::try_from(local_cell)
            .map_err(|_| DenseRuntimeError::SlotCountTooLarge(class.program.name.clone()))?;
        let slot_index = start
            .checked_add(cell)
            .ok_or(DenseRuntimeError::ArithmeticOverflow(
                "dense physical slot index",
            ))?;
        Ok((
            DenseLogicalBlock {
                request: lease,
                class_id: class.program.class_id,
                ordinal,
            },
            DenseTemporalAddress {
                class_id: class.program.class_id,
                cell_index,
                version,
            },
            class.program.class_id,
            slot_index,
        ))
    }

    fn block(&self, logical: DenseLogicalBlock) -> Result<DenseViewBlock, DenseRuntimeError> {
        let class = self.class(logical.class_id)?;
        let (cell_index, version) = class.program.address.evaluate_dense(logical.ordinal)?;
        let local_cell = cell_index.checked_sub(class.program.cell_origin).ok_or(
            DenseRuntimeError::ArithmeticOverflow("dense cell origin subtraction"),
        )?;
        let start = request_slot_start(&class.program, logical.request)?;
        let cell = usize::try_from(local_cell)
            .map_err(|_| DenseRuntimeError::SlotCountTooLarge(class.program.name.clone()))?;
        let slot_index = start
            .checked_add(cell)
            .ok_or(DenseRuntimeError::ArithmeticOverflow(
                "dense physical slot index",
            ))?;
        let slot = class
            .slots
            .get(slot_index)
            .ok_or_else(|| DenseRuntimeError::SlotCountTooLarge(class.program.name.clone()))?;
        if slot.occupant != Some(logical) || slot.version != Some(version) {
            return Err(DenseRuntimeError::BlockNotResident(logical));
        }
        Ok(DenseViewBlock {
            logical,
            temporal: DenseTemporalAddress {
                class_id: logical.class_id,
                cell_index,
                version,
            },
            physical: DensePhysicalHandle {
                class_id: logical.class_id,
                slot: slot_index as u64,
                generation: slot.generation,
            },
            backend: slot
                .backend
                .ok_or(DenseRuntimeError::MissingBackendBinding(
                    DensePhysicalHandle {
                        class_id: logical.class_id,
                        slot: slot_index as u64,
                        generation: slot.generation,
                    },
                ))?,
        })
    }

    fn request(&self, lease: RequestLease) -> Result<&DenseRequest, DenseRuntimeError> {
        self.requests
            .get(lease.slot as usize)
            .filter(|request| request.active && request.generation == lease.generation)
            .ok_or(DenseRuntimeError::StaleRequest(lease))
    }

    fn request_mut(&mut self, lease: RequestLease) -> Result<&mut DenseRequest, DenseRuntimeError> {
        self.requests
            .get_mut(lease.slot as usize)
            .filter(|request| request.active && request.generation == lease.generation)
            .ok_or(DenseRuntimeError::StaleRequest(lease))
    }

    fn class(&self, class_id: ClassId) -> Result<&DenseClassRuntime, DenseRuntimeError> {
        self.classes
            .get(usize::from(class_id.0))
            .filter(|class| class.program.class_id == class_id)
            .ok_or(DenseRuntimeError::ArithmeticOverflow("dense class id"))
    }

    fn slot(&self, handle: DensePhysicalHandle) -> Result<&DenseSlot, DenseRuntimeError> {
        let class = self.class(handle.class_id)?;
        let index = usize::try_from(handle.slot)
            .map_err(|_| DenseRuntimeError::SlotCountTooLarge(class.program.name.clone()))?;
        class
            .slots
            .get(index)
            .filter(|slot| slot.generation == handle.generation)
            .ok_or(DenseRuntimeError::StaleHandle(handle))
    }

    fn slot_mut(
        &mut self,
        handle: DensePhysicalHandle,
    ) -> Result<&mut DenseSlot, DenseRuntimeError> {
        let class = self
            .classes
            .get_mut(usize::from(handle.class_id.0))
            .filter(|class| class.program.class_id == handle.class_id)
            .ok_or(DenseRuntimeError::StaleHandle(handle))?;
        let index = usize::try_from(handle.slot)
            .map_err(|_| DenseRuntimeError::SlotCountTooLarge(class.program.name.clone()))?;
        class
            .slots
            .get_mut(index)
            .filter(|slot| slot.generation == handle.generation)
            .ok_or(DenseRuntimeError::StaleHandle(handle))
    }
}

fn addressable_cells(
    address: &AddressProgram,
    domain: &BlockDomain,
    maximum_blocks: u64,
) -> (u64, u64) {
    match *address {
        AddressProgram::Periodic { period_blocks }
        | AddressProgram::PeriodicFrom { period_blocks, .. } => (0, period_blocks),
        AddressProgram::ResettableArena { blocks_per_epoch } => (0, blocks_per_epoch),
        AddressProgram::AppendOnly | AddressProgram::Pinned => {
            (domain.start_block, domain.blocks_before(maximum_blocks))
        }
    }
}

fn validate_artifact_geometry(artifact: &DenseRuntimeArtifact) -> Result<(), DenseRuntimeError> {
    validate_artifact_limits(artifact)?;
    let mut total_physical_slots = 0_u64;
    for (index, class) in artifact.classes.iter().enumerate() {
        validate_class_geometry(artifact, index, class)?;
        total_physical_slots = total_physical_slots
            .checked_add(class.physical_slots)
            .ok_or(DenseRuntimeError::ArithmeticOverflow(
                "dense total physical slots",
            ))?;
    }
    u32::try_from(total_physical_slots)
        .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense certificate arena capacity"))?;
    usize::try_from(total_physical_slots)
        .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense host physical slot capacity"))?;
    Ok(())
}

fn validate_artifact_limits(artifact: &DenseRuntimeArtifact) -> Result<(), DenseRuntimeError> {
    if artifact.maximum_requests == 0 {
        return Err(DenseRuntimeError::ZeroMaximumRequests);
    }
    if artifact.maximum_inflight_submissions == 0 {
        return Err(DenseRuntimeError::ZeroMaximumInflight);
    }
    if artifact.maximum_blocks_per_request == 0 {
        return Err(DenseRuntimeError::ZeroMaximumBlocks);
    }
    if artifact.page_tokens == 0 {
        return Err(DenseRuntimeError::Plan(PlanError::ZeroPageTokens));
    }
    if artifact.classes.is_empty() {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "artifact has no classes",
        ));
    }
    if artifact.classes.len() > usize::from(u16::MAX) {
        return Err(DenseRuntimeError::TooManyClasses);
    }
    usize::try_from(artifact.maximum_requests)
        .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense request count"))?;
    usize::try_from(artifact.maximum_inflight_submissions)
        .map_err(|_| DenseRuntimeError::ArithmeticOverflow("dense submission capacity"))?;
    Ok(())
}

fn validate_class_geometry(
    artifact: &DenseRuntimeArtifact,
    index: usize,
    class: &DenseClassProgram,
) -> Result<(), DenseRuntimeError> {
    let expected_class_id =
        ClassId(u16::try_from(index).map_err(|_| DenseRuntimeError::TooManyClasses)?);
    if class.class_id != expected_class_id {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "class IDs are not sequential",
        ));
    }
    if class.name.is_empty() {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "class name is empty",
        ));
    }
    if artifact.classes[..index]
        .iter()
        .any(|previous| previous.name == class.name)
    {
        return Err(DenseRuntimeError::DuplicateArtifactClassName(
            class.name.clone(),
        ));
    }
    if class
        .block_domain
        .end_block_exclusive
        .is_some_and(|end| end <= class.block_domain.start_block)
    {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "block domain is empty",
        ));
    }
    let period = match class.address {
        AddressProgram::Periodic { period_blocks }
        | AddressProgram::PeriodicFrom { period_blocks, .. } => Some(period_blocks),
        AddressProgram::ResettableArena { blocks_per_epoch } => Some(blocks_per_epoch),
        AddressProgram::AppendOnly | AddressProgram::Pinned => None,
    };
    if period == Some(0) {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "address period is zero",
        ));
    }
    if let AddressProgram::PeriodicFrom { origin_block, .. } = class.address
        && origin_block != class.block_domain.start_block
    {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "periodic origin does not match block domain",
        ));
    }
    if matches!(
        class.retirement,
        RetirementProgram::EpochEnd {
            blocks_per_epoch: 0
        }
    ) {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "retirement epoch is zero",
        ));
    }
    let (expected_origin, expected_cells) = addressable_cells(
        &class.address,
        &class.block_domain,
        artifact.maximum_blocks_per_request,
    );
    if expected_cells == 0 {
        return Err(DenseRuntimeError::ZeroClassCells(class.name.clone()));
    }
    if class.cell_origin != expected_origin || class.cells_per_request != expected_cells {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "class cell geometry does not match its address program",
        ));
    }
    class
        .cell_origin
        .checked_add(class.cells_per_request)
        .ok_or(DenseRuntimeError::ArithmeticOverflow(
            "dense class cell range",
        ))?;
    let expected_slots = class
        .cells_per_request
        .checked_mul(u64::from(artifact.maximum_requests))
        .ok_or(DenseRuntimeError::ArithmeticOverflow(
            "dense physical slot count",
        ))?;
    if class.physical_slots != expected_slots {
        return Err(DenseRuntimeError::InvalidArtifactGeometry(
            "physical slots do not match request stripes",
        ));
    }
    usize::try_from(class.physical_slots)
        .map_err(|_| DenseRuntimeError::SlotCountTooLarge(class.name.clone()))?;
    Ok(())
}

fn request_slot_start(
    program: &DenseClassProgram,
    lease: RequestLease,
) -> Result<usize, DenseRuntimeError> {
    u64::from(lease.slot)
        .checked_mul(program.cells_per_request)
        .and_then(|start| usize::try_from(start).ok())
        .ok_or_else(|| DenseRuntimeError::SlotCountTooLarge(program.name.clone()))
}

fn encode_arena_id(slot: u32, generation: u32) -> u64 {
    (u64::from(generation) << 32) | u64::from(slot)
}

fn decode_arena_id(id: u64) -> (u32, u32) {
    let bytes = id.to_le_bytes();
    (
        u32::from_le_bytes(bytes[..4].try_into().expect("four-byte slot ID")),
        u32::from_le_bytes(bytes[4..].try_into().expect("four-byte generation")),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{
        BlockManagerConfig, ClassPoolConfig, IntExpr, KvBlockManager, KvClassSpec, KvPlanInput,
        Predicate, RetentionKind, RetentionProgramInput, RetentionStateDecl, compile_plan,
        compile_retention_program,
    };

    use super::*;

    fn hybrid_plan() -> CompiledKvPlan {
        compile_plan(KvPlanInput {
            page_tokens: 4,
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
                    window_tokens: Some(9),
                },
            ],
        })
        .unwrap()
    }

    fn sink_sliding_plan() -> CompiledKvPlan {
        compile_retention_program(RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 4,
            states: vec![RetentionStateDecl {
                name: "attention".into(),
                layers: vec![0],
                kv_head_range: None,
                bytes_per_token_per_layer: 128,
                may_read: Predicate::Or {
                    terms: vec![
                        Predicate::LessThan {
                            lhs: IntExpr::KeyPosition,
                            rhs: IntExpr::Constant { value: 4 },
                        },
                        Predicate::LessThan {
                            lhs: IntExpr::Sub {
                                lhs: Box::new(IntExpr::QueryPosition),
                                rhs: Box::new(IntExpr::KeyPosition),
                            },
                            rhs: IntExpr::Constant { value: 8 },
                        },
                    ],
                },
            }],
        })
        .unwrap()
    }

    fn commit_dense(runtime: &mut DenseKvRuntime, certificates: Vec<DenseRetirementCertificate>) {
        let artifact_fingerprint = runtime.artifact().artifact_fingerprint.clone();
        for certificate in certificates {
            runtime
                .commit_reclamation(&DensePhysicalReclamationReceipt {
                    schema: "orbitkv.dense-physical-reclamation-receipt.v1".into(),
                    artifact_fingerprint: artifact_fingerprint.clone(),
                    certificate_id: certificate.certificate_id,
                    physical: certificate.physical,
                    backend: certificate.backend,
                })
                .unwrap();
        }
    }

    fn class_id_for(plan: &CompiledKvPlan, name: &str) -> ClassId {
        ClassId(
            u16::try_from(
                plan.classes
                    .iter()
                    .position(|class| class.spec.name == name)
                    .unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn artifact_uses_stable_dense_ids_and_direct_stripes() {
        let plan = hybrid_plan();
        let artifact = DenseRuntimeArtifact::compile(&plan, 2, 16, 64).unwrap();
        artifact.validate(&plan).unwrap();
        assert_eq!(artifact.classes[0].class_id, ClassId(0));
        assert_eq!(artifact.classes[1].class_id, ClassId(1));
        assert_eq!(artifact.classes[0].cells_per_request, 64);
        assert_eq!(artifact.classes[1].cells_per_request, 3);
        assert_eq!(artifact.classes[1].physical_slots, 6);
    }

    #[test]
    fn runtime_rejects_self_consistent_invalid_artifact_geometry() {
        let plan = hybrid_plan();
        let mut artifact = DenseRuntimeArtifact::compile(&plan, 2, 16, 64).unwrap();
        artifact.classes[0].physical_slots += 1;
        artifact.artifact_fingerprint = artifact.compute_fingerprint().unwrap();
        assert!(matches!(
            DenseKvRuntime::new(artifact),
            Err(DenseRuntimeError::InvalidArtifactGeometry(
                "physical slots do not match request stripes"
            ))
        ));
    }

    #[test]
    fn binding_publishes_dynamic_backend_pages_in_immutable_views() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 1, 4, 16).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let lease = runtime.acquire_request().unwrap();
        let intent = runtime.prepare_binding_to(lease, 4).unwrap();
        let backend_indices = [7, 1];
        let receipt = DensePhysicalBindingReceipt {
            schema: "orbitkv.dense-physical-binding-receipt.v1".into(),
            artifact_fingerprint: runtime.artifact().artifact_fingerprint.clone(),
            binding_id: intent.binding_id,
            backend_transaction_id: "sglang:test-binding".into(),
            blocks: intent
                .pending_blocks
                .iter()
                .zip(backend_indices)
                .map(|(block, index)| DensePhysicalBindingBlockReceipt {
                    logical: block.logical,
                    physical: block.physical,
                    backend: DenseBackendHandle {
                        domain: block.logical.class_id,
                        index,
                    },
                    payload_ready: true,
                })
                .collect(),
        };
        let published = runtime.commit_binding(&receipt).unwrap();
        assert_eq!(
            published
                .iter()
                .map(|block| block.backend.index)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(backend_indices)
        );
        assert!(
            published
                .iter()
                .any(|block| block.backend.index != block.physical.slot)
        );
        runtime.advance_semantic_frontier(lease, 4).unwrap();
        let view = runtime.submit_view(lease).unwrap();
        assert_eq!(
            view.blocks
                .iter()
                .map(|block| block.backend)
                .collect::<BTreeSet<_>>(),
            published
                .iter()
                .map(|block| block.backend)
                .collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn hydration_materializes_only_the_compiler_proven_live_set() {
        let plan = compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![1],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 128,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: vec![0],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(1024),
                },
            ],
        })
        .unwrap();
        let artifact = DenseRuntimeArtifact::compile(&plan, 1, 4, 1024).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let lease = runtime.acquire_request().unwrap();
        let intent = runtime.prepare_hydration_to(lease, 16_384).unwrap();
        let full = class_id_for(&plan, "full");
        let swa = class_id_for(&plan, "swa");
        assert_eq!(
            intent
                .pending_blocks
                .iter()
                .filter(|block| block.logical.class_id == full)
                .count(),
            1024
        );
        let swa_blocks = intent
            .pending_blocks
            .iter()
            .filter(|block| block.logical.class_id == swa)
            .collect::<Vec<_>>();
        assert_eq!(swa_blocks.len(), 64);
        assert_eq!(swa_blocks.first().unwrap().logical.ordinal, 960);
        assert_eq!(swa_blocks.last().unwrap().logical.ordinal, 1023);
        assert_eq!(
            swa_blocks
                .iter()
                .map(|block| block.temporal.cell_index)
                .collect::<BTreeSet<_>>()
                .len(),
            64
        );
    }

    #[test]
    fn resident_frontier_advances_only_inside_a_bound_page() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 1, 4, 16).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let lease = runtime.acquire_request().unwrap();
        runtime.materialize_to(lease, 1).unwrap();
        runtime.advance_resident_frontier(lease, 4).unwrap();
        assert_eq!(runtime.request(lease).unwrap().materialized_boundary, 4);
        assert_eq!(runtime.request(lease).unwrap().semantic_frontier, 4);
        assert_eq!(
            runtime.advance_resident_frontier(lease, 5),
            Err(DenseRuntimeError::SemanticFrontierBeyondMaterialized)
        );
    }

    #[test]
    fn duplicate_backend_page_fails_before_any_binding_is_published() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 1, 4, 16).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let lease = runtime.acquire_request().unwrap();
        let intent = runtime.prepare_binding_to(lease, 8).unwrap();
        let receipt = DensePhysicalBindingReceipt {
            schema: "orbitkv.dense-physical-binding-receipt.v1".into(),
            artifact_fingerprint: runtime.artifact().artifact_fingerprint.clone(),
            binding_id: intent.binding_id,
            backend_transaction_id: "sglang:duplicate".into(),
            blocks: intent
                .pending_blocks
                .iter()
                .map(|block| DensePhysicalBindingBlockReceipt {
                    logical: block.logical,
                    physical: block.physical,
                    backend: DenseBackendHandle {
                        domain: block.logical.class_id,
                        index: 0,
                    },
                    payload_ready: true,
                })
                .collect(),
        };
        assert_eq!(
            runtime.commit_binding(&receipt),
            Err(DenseRuntimeError::MismatchedBindingReceipt(
                intent.binding_id
            ))
        );
        assert_eq!(runtime.stats().reserved_blocks, 4);
        assert_eq!(runtime.stats().resident_blocks, 0);
        assert!(
            runtime
                .backend_bindings
                .iter()
                .flatten()
                .all(Option::is_none)
        );
        runtime.abort_binding(intent.binding_id).unwrap();
    }

    #[test]
    fn reclamation_receipt_must_name_the_committed_backend_page() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 1, 4, 16).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let lease = runtime.acquire_request().unwrap();
        runtime.materialize_to(lease, 4).unwrap();
        runtime.advance_semantic_frontier(lease, 4).unwrap();
        let certificates = runtime.release_request(lease).unwrap();
        let certificate = certificates[0].clone();
        let wrong = DensePhysicalReclamationReceipt {
            schema: "orbitkv.dense-physical-reclamation-receipt.v1".into(),
            artifact_fingerprint: runtime.artifact().artifact_fingerprint.clone(),
            certificate_id: certificate.certificate_id,
            physical: certificate.physical,
            backend: DenseBackendHandle {
                domain: certificate.backend.domain,
                index: (certificate.backend.index + 1)
                    % runtime
                        .class(certificate.backend.domain)
                        .unwrap()
                        .slots
                        .len() as u64,
            },
        };
        assert_eq!(
            runtime.commit_reclamation(&wrong),
            Err(DenseRuntimeError::MismatchedReclamationReceipt(
                certificate.certificate_id
            ))
        );
        commit_dense(&mut runtime, certificates);
    }

    #[test]
    fn reclamation_batch_is_atomic_on_receipt_mismatch() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 1, 4, 16).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let lease = runtime.acquire_request().unwrap();
        runtime.materialize_to(lease, 4).unwrap();
        runtime.advance_semantic_frontier(lease, 4).unwrap();
        let certificates = runtime.release_request(lease).unwrap();
        let artifact_fingerprint = runtime.artifact().artifact_fingerprint.clone();
        let mut receipts = certificates
            .iter()
            .map(|certificate| DensePhysicalReclamationReceipt {
                schema: "orbitkv.dense-physical-reclamation-receipt.v1".into(),
                artifact_fingerprint: artifact_fingerprint.clone(),
                certificate_id: certificate.certificate_id,
                physical: certificate.physical,
                backend: certificate.backend,
            })
            .collect::<Vec<_>>();
        receipts[1].backend.index = u64::MAX;
        assert_eq!(
            runtime.commit_reclamations(&receipts),
            Err(DenseRuntimeError::MismatchedReclamationReceipt(
                receipts[1].certificate_id
            ))
        );
        assert_eq!(runtime.stats().resident_blocks, 2);
        receipts[1].backend = certificates[1].backend;
        runtime.commit_reclamations(&receipts).unwrap();
        assert_eq!(runtime.stats().resident_blocks, 0);
    }

    #[test]
    fn partitioned_pinned_cells_preserve_nonzero_origin() {
        let plan = sink_sliding_plan();
        let artifact = DenseRuntimeArtifact::compile(&plan, 1, 16, 16).unwrap();
        let sink = artifact
            .classes
            .iter()
            .find(|class| class.name.ends_with("::sink"))
            .unwrap();
        assert_eq!(sink.cell_origin, 0);
        assert_eq!(sink.cells_per_request, 1);
        let sink_class_id = sink.class_id;
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let lease = runtime.acquire_request().unwrap();
        runtime.materialize_to(lease, 16).unwrap();
        runtime.advance_semantic_frontier(lease, 16).unwrap();
        let live = runtime.submit_view(lease).unwrap();
        assert!(live.blocks.iter().any(|block| {
            block.logical.class_id == sink_class_id && block.temporal.cell_index == 0
        }));
    }

    #[test]
    fn request_generation_prevents_stale_reuse() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 1, 16, 16).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let first = runtime.acquire_request().unwrap();
        runtime.materialize_to(first, 4).unwrap();
        runtime.advance_semantic_frontier(first, 4).unwrap();
        let certificates = runtime.release_request(first).unwrap();
        commit_dense(&mut runtime, certificates);
        runtime.recycle_request(first).unwrap();
        let second = runtime.acquire_request().unwrap();
        assert_eq!(first.slot, second.slot);
        assert_eq!(first.generation + 1, second.generation);
        assert_eq!(
            runtime.materialize_to(first, 4),
            Err(DenseRuntimeError::StaleRequest(first))
        );
    }

    #[test]
    fn fixed_submission_arena_reuses_slots_with_new_generation() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 1, 1, 16).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let lease = runtime.acquire_request().unwrap();
        runtime.materialize_to(lease, 4).unwrap();
        runtime.advance_semantic_frontier(lease, 4).unwrap();
        let first = runtime.submit_view(lease).unwrap();
        runtime.complete_submission(first.submission_id).unwrap();
        let second = runtime.submit_view(lease).unwrap();
        assert_ne!(first.submission_id, second.submission_id);
        assert_eq!(
            decode_arena_id(first.submission_id).0,
            decode_arena_id(second.submission_id).0
        );
        assert_eq!(
            runtime.complete_submission(first.submission_id),
            Err(DenseRuntimeError::UnknownSubmission(first.submission_id))
        );
        runtime.complete_submission(second.submission_id).unwrap();
        assert_eq!(runtime.submissions.slots.len(), 1);
    }

    #[test]
    fn all_dense_metadata_arenas_remain_bounded_over_long_execution() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 1, 1, 1).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let binding_capacity = runtime.bindings.slots.len();
        let submission_capacity = runtime.submissions.slots.len();
        let certificate_capacity = runtime.certificates.slots.len();
        let completion_capacity = runtime.completion.completed_out_of_order.len();

        for _ in 0..1000 {
            let lease = runtime.acquire_request().unwrap();
            runtime.materialize_to(lease, 4).unwrap();
            runtime.advance_semantic_frontier(lease, 4).unwrap();
            let view = runtime.submit_view(lease).unwrap();
            runtime.complete_submission(view.submission_id).unwrap();
            let certificates = runtime.release_request(lease).unwrap();
            commit_dense(&mut runtime, certificates);
            runtime.recycle_request(lease).unwrap();
        }

        assert_eq!(runtime.bindings.slots.len(), binding_capacity);
        assert_eq!(runtime.submissions.slots.len(), submission_capacity);
        assert_eq!(runtime.certificates.slots.len(), certificate_capacity);
        assert_eq!(
            runtime.completion.completed_out_of_order.len(),
            completion_capacity
        );
        assert_eq!(runtime.bindings.active(), 0);
        assert_eq!(runtime.submissions.active(), 0);
        assert_eq!(runtime.certificates.active(), 0);
        assert_eq!(runtime.stats().active_requests, 0);
    }

    #[test]
    fn request_stripes_isolate_identical_logical_ordinals() {
        let artifact = DenseRuntimeArtifact::compile(&hybrid_plan(), 2, 2, 16).unwrap();
        let mut runtime = DenseKvRuntime::new(artifact).unwrap();
        let first = runtime.acquire_request().unwrap();
        let second = runtime.acquire_request().unwrap();
        let first_blocks = runtime.materialize_to(first, 4).unwrap();
        let second_blocks = runtime.materialize_to(second, 4).unwrap();
        for class_id in [ClassId(0), ClassId(1)] {
            let first_handle = first_blocks
                .iter()
                .find(|block| block.logical.class_id == class_id)
                .unwrap()
                .physical;
            let second_handle = second_blocks
                .iter()
                .find(|block| block.logical.class_id == class_id)
                .unwrap()
                .physical;
            assert_ne!(first_handle.slot, second_handle.slot);
        }
    }

    #[test]
    fn dense_and_reference_managers_match_lifetime_events() {
        let plan = hybrid_plan();
        let artifact = DenseRuntimeArtifact::compile(&plan, 1, 16, 128).unwrap();
        let mut dense = DenseKvRuntime::new(artifact).unwrap();
        let lease = dense.acquire_request().unwrap();
        let mut reference = KvBlockManager::new(
            plan.clone(),
            BlockManagerConfig {
                pools: vec![
                    ClassPoolConfig {
                        class_name: "full".into(),
                        slot_count: 128,
                    },
                    ClassPoolConfig {
                        class_name: "swa".into(),
                        slot_count: 3,
                    },
                ],
            },
        )
        .unwrap();
        reference.register_request("r0").unwrap();

        for boundary in (4..=128).step_by(4) {
            dense.materialize_to(lease, boundary).unwrap();
            reference.materialize_to("r0", boundary).unwrap();
            let dense_certificates = dense.advance_semantic_frontier(lease, boundary).unwrap();
            let reference_certificates =
                reference.advance_semantic_frontier("r0", boundary).unwrap();
            let dense_retired = dense_certificates
                .iter()
                .map(|certificate| (certificate.logical.class_id, certificate.logical.ordinal))
                .collect::<BTreeSet<_>>();
            let reference_retired = reference_certificates
                .iter()
                .map(|certificate| {
                    (
                        class_id_for(&plan, &certificate.logical.class_name),
                        certificate.logical.ordinal,
                    )
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(dense_retired, reference_retired);
            commit_dense(&mut dense, dense_certificates);
            for certificate in reference_certificates {
                reference
                    .commit_reclamation(&crate::PhysicalReclamationReceipt {
                        schema: "orbitkv.physical-reclamation-receipt.v1",
                        certificate_id: certificate.certificate_id,
                        physical: certificate.physical,
                    })
                    .unwrap();
            }

            let dense_view = dense.submit_view(lease).unwrap();
            let dense_submission_id = dense_view.submission_id;
            let dense_live = dense_view
                .blocks
                .into_iter()
                .map(|block| (block.logical.class_id, block.logical.ordinal))
                .collect::<BTreeSet<_>>();
            let reference_view = reference.submit_view("r0").unwrap();
            let reference_live = reference_view
                .blocks
                .iter()
                .map(|block| {
                    (
                        class_id_for(&plan, &block.logical.class_name),
                        block.logical.ordinal,
                    )
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(dense_live, reference_live);
            dense.complete_submission(dense_submission_id).unwrap();
            reference
                .complete_submission(reference_view.submission_id)
                .unwrap();
        }
    }

    #[test]
    fn deterministic_random_event_stream_matches_reference_manager() {
        let plan = hybrid_plan();
        let artifact = DenseRuntimeArtifact::compile(&plan, 1, 16, 512).unwrap();
        let mut dense = DenseKvRuntime::new(artifact).unwrap();
        let lease = dense.acquire_request().unwrap();
        let mut reference = KvBlockManager::new(
            plan.clone(),
            BlockManagerConfig {
                pools: vec![
                    ClassPoolConfig {
                        class_name: "full".into(),
                        slot_count: 512,
                    },
                    ClassPoolConfig {
                        class_name: "swa".into(),
                        slot_count: 3,
                    },
                ],
            },
        )
        .unwrap();
        reference.register_request("r0").unwrap();

        let mut seed = 0x8d26_49c7_31ab_f005_u64;
        let mut boundary = 0_u64;
        let mut submissions = Vec::<(u64, u64)>::new();
        for _ in 0..1000 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let action = (seed >> 61) as u8;
            if action <= 4 && boundary < 2048 {
                let step = ((seed >> 16) % 4 + 1) * plan.page_tokens;
                let next = boundary.saturating_add(step).min(2048);
                let dense_result = dense.materialize_to(lease, next);
                let reference_result = reference.materialize_to("r0", next);
                assert_eq!(dense_result.is_ok(), reference_result.is_ok());
                if dense_result.is_err() {
                    complete_one_random_submission(
                        &mut dense,
                        &mut reference,
                        &mut submissions,
                        seed,
                    );
                    continue;
                }
                boundary = next;
                let dense_certificates = dense.advance_semantic_frontier(lease, boundary).unwrap();
                let reference_certificates =
                    reference.advance_semantic_frontier("r0", boundary).unwrap();
                assert_retired_equivalent(&plan, &dense_certificates, &reference_certificates);
                commit_dense(&mut dense, dense_certificates);
                commit_reference(&mut reference, reference_certificates);
            } else if action <= 6 && boundary > 0 {
                let dense_view = dense.submit_view(lease).unwrap();
                let reference_view = reference.submit_view("r0").unwrap();
                assert_live_equivalent(&plan, &dense_view.blocks, &reference_view.blocks);
                submissions.push((dense_view.submission_id, reference_view.submission_id));
            } else {
                complete_one_random_submission(&mut dense, &mut reference, &mut submissions, seed);
            }
            assert_eq!(
                dense.stats().resident_blocks,
                reference.stats().resident_blocks
            );
            assert_eq!(
                dense.stats().active_submissions,
                reference.stats().active_submissions
            );
        }
        while !submissions.is_empty() {
            complete_one_random_submission(&mut dense, &mut reference, &mut submissions, seed);
            seed = seed.rotate_left(7);
        }
    }

    fn complete_one_random_submission(
        dense: &mut DenseKvRuntime,
        reference: &mut KvBlockManager,
        submissions: &mut Vec<(u64, u64)>,
        seed: u64,
    ) {
        if submissions.is_empty() {
            return;
        }
        let index = usize::try_from(seed % submissions.len() as u64).unwrap();
        let (dense_submission, reference_submission) = submissions.swap_remove(index);
        let dense_certificates = dense.complete_submission(dense_submission).unwrap();
        let reference_certificates = reference.complete_submission(reference_submission).unwrap();
        commit_dense(dense, dense_certificates);
        commit_reference(reference, reference_certificates);
    }

    fn assert_retired_equivalent(
        plan: &CompiledKvPlan,
        dense: &[DenseRetirementCertificate],
        reference: &[crate::RetirementCertificate],
    ) {
        let dense_retired = dense
            .iter()
            .map(|certificate| (certificate.logical.class_id, certificate.logical.ordinal))
            .collect::<BTreeSet<_>>();
        let reference_retired = reference
            .iter()
            .map(|certificate| {
                (
                    class_id_for(plan, &certificate.logical.class_name),
                    certificate.logical.ordinal,
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(dense_retired, reference_retired);
    }

    fn assert_live_equivalent(
        plan: &CompiledKvPlan,
        dense: &[DenseViewBlock],
        reference: &[crate::ViewBlock],
    ) {
        let dense_live = dense
            .iter()
            .map(|block| (block.logical.class_id, block.logical.ordinal))
            .collect::<BTreeSet<_>>();
        let reference_live = reference
            .iter()
            .map(|block| {
                (
                    class_id_for(plan, &block.logical.class_name),
                    block.logical.ordinal,
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(dense_live, reference_live);
    }

    fn commit_reference(
        reference: &mut KvBlockManager,
        certificates: Vec<crate::RetirementCertificate>,
    ) {
        for certificate in certificates {
            reference
                .commit_reclamation(&crate::PhysicalReclamationReceipt {
                    schema: "orbitkv.physical-reclamation-receipt.v1",
                    certificate_id: certificate.certificate_id,
                    physical: certificate.physical,
                })
                .unwrap();
        }
    }
}
