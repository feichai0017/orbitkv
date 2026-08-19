use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;

use crate::{
    CompiledKvClass, CompiledKvPlan, LayoutProgram, LogicalCellId, PlanError, TemporalAddress,
};

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
    pub temporal: TemporalAddress,
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
    ChunkEpoch {
        semantic_frontier: u64,
        death_boundary: u64,
        chunk_tokens: u64,
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
    pub temporal: TemporalAddress,
    pub physical: BlockHandle,
    pub token_start: u64,
    pub token_end_exclusive: u64,
    pub semantic_proof: SemanticProof,
    pub execution_proof: ExecutionProof,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalReclamationReceipt {
    pub schema: &'static str,
    pub certificate_id: u64,
    pub physical: BlockHandle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BindingIntent {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub binding_id: u64,
    pub request_id: String,
    pub previous_boundary: u64,
    pub target_boundary: u64,
    pub resident_blocks: Vec<ViewBlock>,
    pub pending_blocks: Vec<ViewBlock>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalBindingBlockReceipt {
    pub logical: BlockKey,
    pub physical: BlockHandle,
    pub payload_ready: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PhysicalBindingReceipt {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub binding_id: u64,
    pub backend_transaction_id: String,
    pub blocks: Vec<PhysicalBindingBlockReceipt>,
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
    pub pending_bindings: u64,
    pub pending_certificates: u64,
    pub reserved_blocks: u64,
    pub resident_blocks: u64,
    pub retiring_blocks: u64,
    pub free_slots: BTreeMap<String, u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SlotPhase {
    #[default]
    Free,
    Reserved,
    Active,
    Retiring,
    Certified,
}

#[derive(Clone, Debug, Default)]
struct SlotState {
    generation: u64,
    occupant: Option<BlockKey>,
    temporal: Option<TemporalAddress>,
    readers: BTreeSet<u64>,
    phase: SlotPhase,
    pending_binding: Option<u64>,
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

#[derive(Clone, Debug)]
struct CellBinding {
    logical: BlockKey,
    version: crate::CellVersion,
    physical: BlockHandle,
}

struct BindingPreflight {
    previous_boundary: u64,
    resident_blocks: Vec<ViewBlock>,
    pending_logical: Vec<(BlockKey, TemporalAddress)>,
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
    #[error("physical pool {class:?} has {configured} slots, below compiled minimum {minimum}")]
    InsufficientPoolSlots {
        class: String,
        configured: u64,
        minimum: u64,
    },
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
    #[error("request {request:?} already has pending binding {binding_id}")]
    PendingBinding { request: String, binding_id: u64 },
    #[error("unknown binding transaction {0}")]
    UnknownBinding(u64),
    #[error("binding transaction {0} is stale")]
    StaleBinding(u64),
    #[error("physical binding receipt does not match transaction {0}")]
    MismatchedBindingReceipt(u64),
    #[error("physical binding receipt {0} does not prove payload readiness")]
    PayloadNotReady(u64),
    #[error("physical binding backend transaction id must not be empty")]
    EmptyBackendTransactionId,
    #[error("logical block is not resident: {0:?}")]
    BlockNotResident(BlockKey),
    #[error("logical block handle is stale: {0:?}")]
    StaleHandle(BlockHandle),
    #[error("logical cell is still bound to another version: {0:?}")]
    LogicalCellCollision(LogicalCellId),
    #[error("unknown submission {0}")]
    UnknownSubmission(u64),
    #[error("unknown retirement certificate {0}")]
    UnknownCertificate(u64),
    #[error("retirement certificate {0} no longer matches its physical slot")]
    StaleCertificate(u64),
    #[error("physical reclamation receipt does not match certificate {0}")]
    MismatchedReclamationReceipt(u64),
    #[error("submission generation exhausted")]
    SubmissionGenerationExhausted,
    #[error("certificate generation exhausted")]
    CertificateGenerationExhausted,
    #[error("binding generation exhausted")]
    BindingGenerationExhausted,
    #[error("block generation exhausted for {0:?}")]
    BlockGenerationExhausted(BlockKey),
    #[error("integer overflow while calculating {0}")]
    ArithmeticOverflow(&'static str),
    #[error(transparent)]
    Plan(#[from] PlanError),
}

pub struct KvBlockManager {
    plan: CompiledKvPlan,
    layout: LayoutProgram,
    fingerprint: String,
    classes: BTreeMap<String, CompiledKvClass>,
    pools: BTreeMap<String, ClassPool>,
    requests: BTreeMap<String, RequestState>,
    block_handles: BTreeMap<BlockKey, BlockHandle>,
    cell_bindings: BTreeMap<LogicalCellId, CellBinding>,
    submissions: BTreeMap<u64, SubmissionState>,
    completed_out_of_order: BTreeSet<u64>,
    completed_through: u64,
    next_submission_id: u64,
    pending_certificates: BTreeMap<u64, RetirementCertificate>,
    next_certificate_id: u64,
    pending_bindings: BTreeMap<u64, BindingIntent>,
    pending_request_bindings: BTreeMap<String, u64>,
    pending_block_bindings: BTreeMap<BlockKey, u64>,
    pending_cell_bindings: BTreeMap<LogicalCellId, u64>,
    next_binding_id: u64,
}

impl KvBlockManager {
    /// Creates a block manager whose physical slot budgets are explicit.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, empty, unknown, or
    /// host-unrepresentable pool configurations.
    pub fn new(plan: CompiledKvPlan, config: BlockManagerConfig) -> Result<Self, ManagerError> {
        let layout = plan.layout_program()?;
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
            if let Some(minimum) = classes[class_name].slot_count
                && config.slot_count < minimum
            {
                return Err(ManagerError::InsufficientPoolSlots {
                    class: class_name.clone(),
                    configured: config.slot_count,
                    minimum,
                });
            }
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
            layout,
            fingerprint,
            classes,
            pools,
            requests: BTreeMap::new(),
            block_handles: BTreeMap::new(),
            cell_bindings: BTreeMap::new(),
            submissions: BTreeMap::new(),
            completed_out_of_order: BTreeSet::new(),
            completed_through: 0,
            next_submission_id: 1,
            pending_certificates: BTreeMap::new(),
            next_certificate_id: 1,
            pending_bindings: BTreeMap::new(),
            pending_request_bindings: BTreeMap::new(),
            pending_block_bindings: BTreeMap::new(),
            pending_cell_bindings: BTreeMap::new(),
            next_binding_id: 1,
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
    /// This compatibility helper executes the same two-phase binding protocol
    /// as external backends and immediately commits an in-memory receipt.
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
        let intent = self.prepare_binding_to(request_id, boundary)?;
        let receipt = PhysicalBindingReceipt {
            schema: "orbitkv.physical-binding-receipt.v1",
            plan_fingerprint: self.fingerprint.clone(),
            binding_id: intent.binding_id,
            backend_transaction_id: format!("reference:{}", intent.binding_id),
            blocks: intent
                .pending_blocks
                .iter()
                .map(|block| PhysicalBindingBlockReceipt {
                    logical: block.logical.clone(),
                    physical: block.physical.clone(),
                    payload_ready: true,
                })
                .collect(),
        };
        self.commit_binding(&receipt)
    }

    /// Reserves physical slots for all blocks intersecting `boundary` without
    /// publishing them to request state or immutable views.
    ///
    /// # Errors
    ///
    /// Returns an error without reserving any slot if the request, address
    /// program, generation, or pool preflight fails.
    pub fn prepare_binding_to(
        &mut self,
        request_id: &str,
        boundary: u64,
    ) -> Result<BindingIntent, ManagerError> {
        let BindingPreflight {
            previous_boundary,
            resident_blocks,
            pending_logical,
        } = self.preflight_binding(request_id, boundary)?;
        let binding_id = self.next_binding_id;
        let pending_blocks = self.plan_binding_slots(pending_logical)?;
        self.next_binding_id = self
            .next_binding_id
            .checked_add(1)
            .ok_or(ManagerError::BindingGenerationExhausted)?;
        self.reserve_binding_slots(binding_id, &pending_blocks)?;
        let intent = BindingIntent {
            schema: "orbitkv.binding-intent.v1",
            plan_fingerprint: self.fingerprint.clone(),
            binding_id,
            request_id: request_id.to_owned(),
            previous_boundary,
            target_boundary: boundary,
            resident_blocks,
            pending_blocks,
        };
        self.pending_request_bindings
            .insert(request_id.to_owned(), binding_id);
        self.pending_bindings.insert(binding_id, intent.clone());
        Ok(intent)
    }

    fn preflight_binding(
        &self,
        request_id: &str,
        boundary: u64,
    ) -> Result<BindingPreflight, ManagerError> {
        let request = self
            .requests
            .get(request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(request_id.to_owned()))?;
        if request.released {
            return Err(ManagerError::RequestReleased(request_id.to_owned()));
        }
        if let Some(binding_id) = self.pending_request_bindings.get(request_id) {
            return Err(ManagerError::PendingBinding {
                request: request_id.to_owned(),
                binding_id: *binding_id,
            });
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
            return Ok(BindingPreflight {
                previous_boundary: old_boundary,
                resident_blocks: Vec::new(),
                pending_logical: Vec::new(),
            });
        }
        let first_block = old_boundary / self.plan.page_tokens;
        let last_block = (boundary - 1) / self.plan.page_tokens;
        let class_names = self.classes.keys().cloned().collect::<Vec<_>>();
        let mut required = BTreeMap::<String, u64>::new();
        let mut resident_blocks = Vec::new();
        let mut pending_logical = Vec::new();
        let mut pending_cells = BTreeSet::new();
        for ordinal in first_block..=last_block {
            for class_name in &class_names {
                if !self.class_contains_block(class_name, ordinal)? {
                    continue;
                }
                let logical = BlockKey {
                    request_id: request_id.to_owned(),
                    class_name: class_name.clone(),
                    ordinal,
                };
                if let Some(handle) = self.block_handles.get(&logical) {
                    let temporal = self
                        .layout
                        .temporal_address(request_id, class_name, ordinal)?;
                    resident_blocks.push(ViewBlock {
                        logical,
                        temporal,
                        physical: handle.clone(),
                    });
                    continue;
                }
                if self.pending_block_bindings.contains_key(&logical) {
                    return Err(ManagerError::StaleBinding(
                        self.pending_block_bindings[&logical],
                    ));
                }
                if self.is_semantically_dead(&logical)? {
                    return Err(ManagerError::CannotMaterializeDead(logical));
                }
                let temporal = self
                    .layout
                    .temporal_address(request_id, class_name, ordinal)?;
                if self.cell_bindings.contains_key(&temporal.cell)
                    || self.pending_cell_bindings.contains_key(&temporal.cell)
                    || !pending_cells.insert(temporal.cell.clone())
                {
                    return Err(ManagerError::LogicalCellCollision(temporal.cell));
                }
                let count = required.entry(class_name.clone()).or_default();
                *count = count
                    .checked_add(1)
                    .ok_or(ManagerError::ArithmeticOverflow(
                        "materialization block count",
                    ))?;
                pending_logical.push((logical, temporal));
            }
        }
        for (class_name, required_slots) in &required {
            let available_slots = self
                .pools
                .get(class_name)
                .ok_or_else(|| ManagerError::UnknownClass(class_name.clone()))?
                .free
                .len() as u64;
            if *required_slots > available_slots {
                return Err(ManagerError::PoolExhausted {
                    class: class_name.clone(),
                });
            }
        }
        Ok(BindingPreflight {
            previous_boundary: old_boundary,
            resident_blocks,
            pending_logical,
        })
    }

    fn plan_binding_slots(
        &self,
        pending_logical: Vec<(BlockKey, TemporalAddress)>,
    ) -> Result<Vec<ViewBlock>, ManagerError> {
        let mut offsets = BTreeMap::<String, usize>::new();
        let mut pending_blocks = Vec::with_capacity(pending_logical.len());
        for (logical, temporal) in pending_logical {
            let offset = offsets.entry(logical.class_name.clone()).or_default();
            let pool = self
                .pools
                .get(&logical.class_name)
                .ok_or_else(|| ManagerError::UnknownClass(logical.class_name.clone()))?;
            let slot_index =
                *pool
                    .free
                    .iter()
                    .nth(*offset)
                    .ok_or_else(|| ManagerError::PoolExhausted {
                        class: logical.class_name.clone(),
                    })?;
            let generation = pool.slots[slot_index]
                .generation
                .checked_add(1)
                .ok_or_else(|| ManagerError::BlockGenerationExhausted(logical.clone()))?;
            *offset += 1;
            pending_blocks.push(ViewBlock {
                logical: logical.clone(),
                temporal,
                physical: BlockHandle {
                    class_name: logical.class_name,
                    slot: slot_index as u64,
                    generation,
                },
            });
        }
        Ok(pending_blocks)
    }

    fn reserve_binding_slots(
        &mut self,
        binding_id: u64,
        pending_blocks: &[ViewBlock],
    ) -> Result<(), ManagerError> {
        for block in pending_blocks {
            let slot_index = usize::try_from(block.physical.slot)
                .map_err(|_| ManagerError::StaleHandle(block.physical.clone()))?;
            let pool = self
                .pools
                .get(&block.physical.class_name)
                .ok_or_else(|| ManagerError::UnknownClass(block.physical.class_name.clone()))?;
            let slot = pool
                .slots
                .get(slot_index)
                .ok_or_else(|| ManagerError::StaleHandle(block.physical.clone()))?;
            if !pool.free.contains(&slot_index)
                || slot.phase != SlotPhase::Free
                || slot.occupant.is_some()
                || slot.temporal.is_some()
                || slot.generation.checked_add(1) != Some(block.physical.generation)
            {
                return Err(ManagerError::StaleBinding(binding_id));
            }
        }
        for block in pending_blocks {
            let slot_index = usize::try_from(block.physical.slot)
                .map_err(|_| ManagerError::StaleHandle(block.physical.clone()))?;
            let pool = self
                .pools
                .get_mut(&block.physical.class_name)
                .ok_or_else(|| ManagerError::UnknownClass(block.physical.class_name.clone()))?;
            if !pool.free.remove(&slot_index) {
                return Err(ManagerError::StaleBinding(binding_id));
            }
            let slot = &mut pool.slots[slot_index];
            debug_assert_eq!(slot.phase, SlotPhase::Free);
            slot.generation = block.physical.generation;
            slot.occupant = Some(block.logical.clone());
            slot.temporal = Some(block.temporal.clone());
            slot.phase = SlotPhase::Reserved;
            slot.pending_binding = Some(binding_id);
            self.pending_block_bindings
                .insert(block.logical.clone(), binding_id);
            self.pending_cell_bindings
                .insert(block.temporal.cell.clone(), binding_id);
        }
        Ok(())
    }

    /// Publishes one prepared binding after the physical backend proves every
    /// reserved block is bound and its payload is ready.
    ///
    /// Receipt validation is completed for the full batch before any logical
    /// block becomes visible.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale, partial, duplicate, mismatched, or
    /// payload-incomplete receipt. Failure leaves the intent pending.
    pub fn commit_binding(
        &mut self,
        receipt: &PhysicalBindingReceipt,
    ) -> Result<Vec<ViewBlock>, ManagerError> {
        let binding_id = receipt.binding_id;
        let intent = self
            .pending_bindings
            .get(&binding_id)
            .cloned()
            .ok_or(ManagerError::UnknownBinding(binding_id))?;
        if receipt.schema != "orbitkv.physical-binding-receipt.v1"
            || receipt.plan_fingerprint != self.fingerprint
            || receipt.backend_transaction_id.is_empty()
        {
            return if receipt.backend_transaction_id.is_empty() {
                Err(ManagerError::EmptyBackendTransactionId)
            } else {
                Err(ManagerError::MismatchedBindingReceipt(binding_id))
            };
        }
        let mut receipt_blocks = BTreeMap::new();
        for block in &receipt.blocks {
            if !block.payload_ready {
                return Err(ManagerError::PayloadNotReady(binding_id));
            }
            if receipt_blocks
                .insert(block.logical.clone(), block.physical.clone())
                .is_some()
            {
                return Err(ManagerError::MismatchedBindingReceipt(binding_id));
            }
        }
        if receipt_blocks.len() != intent.pending_blocks.len() {
            return Err(ManagerError::MismatchedBindingReceipt(binding_id));
        }
        for block in &intent.pending_blocks {
            if receipt_blocks.get(&block.logical) != Some(&block.physical)
                || self.pending_block_bindings.get(&block.logical) != Some(&binding_id)
                || self.pending_cell_bindings.get(&block.temporal.cell) != Some(&binding_id)
                || self.block_handles.contains_key(&block.logical)
                || self.cell_bindings.contains_key(&block.temporal.cell)
            {
                return Err(ManagerError::MismatchedBindingReceipt(binding_id));
            }
            let slot = self.slot(&block.physical)?;
            if slot.phase != SlotPhase::Reserved
                || slot.pending_binding != Some(binding_id)
                || slot.occupant.as_ref() != Some(&block.logical)
                || slot.temporal.as_ref() != Some(&block.temporal)
            {
                return Err(ManagerError::StaleBinding(binding_id));
            }
        }
        let request = self
            .requests
            .get(&intent.request_id)
            .ok_or_else(|| ManagerError::UnknownRequest(intent.request_id.clone()))?;
        if request.released || request.materialized_boundary != intent.previous_boundary {
            return Err(ManagerError::StaleBinding(binding_id));
        }

        for block in &intent.pending_blocks {
            let slot = self.slot_mut(&block.physical)?;
            slot.phase = SlotPhase::Active;
            slot.pending_binding = None;
        }
        {
            let request = self
                .requests
                .get_mut(&intent.request_id)
                .ok_or_else(|| ManagerError::UnknownRequest(intent.request_id.clone()))?;
            for block in &intent.pending_blocks {
                request.blocks.insert(block.logical.clone());
            }
            request.materialized_boundary = intent.target_boundary;
        }
        for block in &intent.pending_blocks {
            self.block_handles
                .insert(block.logical.clone(), block.physical.clone());
            self.cell_bindings.insert(
                block.temporal.cell.clone(),
                CellBinding {
                    logical: block.logical.clone(),
                    version: block.temporal.version,
                    physical: block.physical.clone(),
                },
            );
            self.pending_block_bindings.remove(&block.logical);
            self.pending_cell_bindings.remove(&block.temporal.cell);
        }
        self.pending_request_bindings.remove(&intent.request_id);
        self.pending_bindings.remove(&binding_id);
        let mut published = intent.resident_blocks;
        published.extend(intent.pending_blocks);
        Ok(published)
    }

    /// Aborts a prepared binding and returns all reserved slots to their pools.
    ///
    /// Generation numbers remain consumed, so stale backend receipts cannot
    /// become valid after a later reservation.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown or internally stale intent.
    pub fn abort_binding(&mut self, binding_id: u64) -> Result<(), ManagerError> {
        let intent = self
            .pending_bindings
            .get(&binding_id)
            .cloned()
            .ok_or(ManagerError::UnknownBinding(binding_id))?;
        for block in &intent.pending_blocks {
            let slot = self.slot(&block.physical)?;
            if slot.phase != SlotPhase::Reserved
                || slot.pending_binding != Some(binding_id)
                || slot.occupant.as_ref() != Some(&block.logical)
                || slot.temporal.as_ref() != Some(&block.temporal)
            {
                return Err(ManagerError::StaleBinding(binding_id));
            }
        }
        for block in &intent.pending_blocks {
            {
                let slot = self.slot_mut(&block.physical)?;
                slot.occupant = None;
                slot.temporal = None;
                slot.phase = SlotPhase::Free;
                slot.pending_binding = None;
            }
            let slot_index = usize::try_from(block.physical.slot)
                .map_err(|_| ManagerError::StaleBinding(binding_id))?;
            self.pools
                .get_mut(&block.physical.class_name)
                .ok_or_else(|| ManagerError::UnknownClass(block.physical.class_name.clone()))?
                .free
                .insert(slot_index);
            self.pending_block_bindings.remove(&block.logical);
            self.pending_cell_bindings.remove(&block.temporal.cell);
        }
        self.pending_request_bindings.remove(&intent.request_id);
        self.pending_bindings.remove(&binding_id);
        Ok(())
    }

    fn class_contains_block(&self, class_name: &str, ordinal: u64) -> Result<bool, ManagerError> {
        Ok(self
            .classes
            .get(class_name)
            .ok_or_else(|| ManagerError::UnknownClass(class_name.to_owned()))?
            .block_domain
            .contains(ordinal))
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
        if let Some(binding_id) = self.pending_request_bindings.get(request_id) {
            return Err(ManagerError::PendingBinding {
                request: request_id.to_owned(),
                binding_id: *binding_id,
            });
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
                let temporal = self.layout.temporal_address(
                    request_id,
                    &logical.class_name,
                    logical.ordinal,
                )?;
                self.validate_temporal_binding(&logical, &temporal, &physical)?;
                blocks.push(ViewBlock {
                    logical,
                    temporal,
                    physical,
                });
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
        if let Some(binding_id) = self.pending_request_bindings.get(request_id) {
            return Err(ManagerError::PendingBinding {
                request: request_id.to_owned(),
                binding_id: *binding_id,
            });
        }
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
    pub fn commit_reclamation(
        &mut self,
        receipt: &PhysicalReclamationReceipt,
    ) -> Result<(), ManagerError> {
        let certificate_id = receipt.certificate_id;
        let certificate = self
            .pending_certificates
            .get(&certificate_id)
            .cloned()
            .ok_or(ManagerError::UnknownCertificate(certificate_id))?;
        if receipt.schema != "orbitkv.physical-reclamation-receipt.v1"
            || receipt.physical != certificate.physical
        {
            return Err(ManagerError::MismatchedReclamationReceipt(certificate_id));
        }
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
            slot.temporal = None;
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
        if self
            .cell_bindings
            .get(&certificate.temporal.cell)
            .is_some_and(|binding| {
                binding.logical == certificate.logical
                    && binding.version == certificate.temporal.version
                    && binding.physical == certificate.physical
            })
        {
            self.cell_bindings.remove(&certificate.temporal.cell);
        }
        if let Some(request) = self.requests.get_mut(&certificate.logical.request_id) {
            request.blocks.remove(&certificate.logical);
        }
        self.pending_certificates.remove(&certificate_id);
        Ok(())
    }

    #[must_use]
    pub fn stats(&self) -> ManagerStats {
        let mut free_slots = BTreeMap::new();
        let mut reserved_blocks = 0_u64;
        let mut resident_blocks = 0_u64;
        let mut retiring_blocks = 0_u64;
        for (name, pool) in &self.pools {
            free_slots.insert(name.clone(), pool.free.len() as u64);
            for slot in &pool.slots {
                if slot.occupant.is_some() {
                    if slot.phase == SlotPhase::Reserved {
                        reserved_blocks += 1;
                    } else {
                        resident_blocks += 1;
                    }
                }
                if matches!(slot.phase, SlotPhase::Retiring | SlotPhase::Certified) {
                    retiring_blocks += 1;
                }
            }
        }
        ManagerStats {
            requests: self.requests.len() as u64,
            active_submissions: self.submissions.len() as u64,
            pending_bindings: self.pending_bindings.len() as u64,
            pending_certificates: self.pending_certificates.len() as u64,
            reserved_blocks,
            resident_blocks,
            retiring_blocks,
            free_slots,
        }
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
            let death_boundary = self
                .death_boundary(logical)?
                .ok_or_else(|| ManagerError::CannotMaterializeDead(logical.clone()))?;
            let class = self
                .classes
                .get(&logical.class_name)
                .ok_or_else(|| ManagerError::UnknownClass(logical.class_name.clone()))?;
            match class.spec.retention {
                crate::RetentionKind::Sliding => SemanticProof::SlidingWindow {
                    semantic_frontier: request.semantic_frontier,
                    death_boundary,
                },
                crate::RetentionKind::Chunked => SemanticProof::ChunkEpoch {
                    semantic_frontier: request.semantic_frontier,
                    death_boundary,
                    chunk_tokens: class.chunk_tokens.ok_or_else(|| {
                        ManagerError::Plan(PlanError::InvalidCompiledChunk {
                            class: logical.class_name.clone(),
                        })
                    })?,
                },
                crate::RetentionKind::Full => {
                    return Err(ManagerError::CannotMaterializeDead(logical.clone()));
                }
            }
        };
        let certificate = RetirementCertificate {
            schema: "orbitkv.retirement-certificate.v1",
            plan_fingerprint: self.fingerprint.clone(),
            certificate_id,
            logical: logical.clone(),
            temporal: self.layout.temporal_address(
                &logical.request_id,
                &logical.class_name,
                logical.ordinal,
            )?,
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
        Ok(self
            .layout
            .class(&logical.class_name)?
            .retirement
            .death_boundary(self.layout.page_tokens, logical.ordinal)?)
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
        let temporal = self.layout.temporal_address(
            &logical.request_id,
            &logical.class_name,
            logical.ordinal,
        )?;
        if slot.temporal.as_ref() != Some(&temporal) {
            return Err(ManagerError::StaleHandle(handle.clone()));
        }
        self.validate_temporal_binding(logical, &temporal, handle)?;
        Ok(())
    }

    fn validate_temporal_binding(
        &self,
        logical: &BlockKey,
        temporal: &TemporalAddress,
        handle: &BlockHandle,
    ) -> Result<(), ManagerError> {
        let binding = self
            .cell_bindings
            .get(&temporal.cell)
            .ok_or_else(|| ManagerError::StaleHandle(handle.clone()))?;
        if binding.logical != *logical
            || binding.version != temporal.version
            || binding.physical != *handle
        {
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
    use crate::{
        IntExpr, KvClassSpec, KvPlanInput, KvRuntimeSimulator, Predicate, ResidentTemporalBlock,
        RetentionKind, RetentionProgramInput, RetentionStateDecl, compile_plan,
        compile_retention_program,
    };

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

    fn ready_receipt(intent: &BindingIntent) -> PhysicalBindingReceipt {
        PhysicalBindingReceipt {
            schema: "orbitkv.physical-binding-receipt.v1",
            plan_fingerprint: intent.plan_fingerprint.clone(),
            binding_id: intent.binding_id,
            backend_transaction_id: format!("test:{}", intent.binding_id),
            blocks: intent
                .pending_blocks
                .iter()
                .map(|block| PhysicalBindingBlockReceipt {
                    logical: block.logical.clone(),
                    physical: block.physical.clone(),
                    payload_ready: true,
                })
                .collect(),
        }
    }

    #[test]
    fn prepared_binding_is_invisible_until_backend_commit() {
        let mut manager = manager(8, 3);
        manager.register_request("r0").unwrap();
        let intent = manager.prepare_binding_to("r0", 16).unwrap();
        assert_eq!(manager.stats().pending_bindings, 1);
        assert_eq!(manager.stats().reserved_blocks, 2);
        assert_eq!(manager.stats().resident_blocks, 0);
        assert!(manager.submit_view("r0").unwrap().blocks.is_empty());
        assert!(matches!(
            manager.advance_semantic_frontier("r0", 1),
            Err(ManagerError::PendingBinding { .. })
        ));
        let published = manager.commit_binding(&ready_receipt(&intent)).unwrap();
        assert_eq!(published.len(), 2);
        assert_eq!(manager.stats().pending_bindings, 0);
        manager.advance_semantic_frontier("r0", 1).unwrap();
        assert_eq!(manager.submit_view("r0").unwrap().blocks.len(), 2);
    }

    #[test]
    fn binding_receipt_preflight_is_atomic_and_abortable() {
        let mut manager = manager(8, 3);
        manager.register_request("r0").unwrap();
        let intent = manager.prepare_binding_to("r0", 16).unwrap();
        let mut incomplete = ready_receipt(&intent);
        incomplete.blocks[1].payload_ready = false;
        assert_eq!(
            manager.commit_binding(&incomplete),
            Err(ManagerError::PayloadNotReady(intent.binding_id))
        );
        assert_eq!(manager.stats().pending_bindings, 1);
        assert!(manager.submit_view("r0").unwrap().blocks.is_empty());
        manager.abort_binding(intent.binding_id).unwrap();
        assert_eq!(manager.stats().pending_bindings, 0);
        assert_eq!(manager.stats().free_slots["full"], 8);
        assert_eq!(manager.stats().free_slots["swa"], 3);

        let retried = manager.prepare_binding_to("r0", 16).unwrap();
        for (first, second) in intent.pending_blocks.iter().zip(&retried.pending_blocks) {
            assert_eq!(first.physical.slot, second.physical.slot);
            assert_eq!(first.physical.generation + 1, second.physical.generation);
        }
        manager.commit_binding(&ready_receipt(&retried)).unwrap();
    }

    #[test]
    fn mismatched_binding_receipt_cannot_publish_a_partial_batch() {
        let mut manager = manager(8, 3);
        manager.register_request("r0").unwrap();
        let intent = manager.prepare_binding_to("r0", 16).unwrap();
        let mut receipt = ready_receipt(&intent);
        receipt.blocks.pop();
        assert_eq!(
            manager.commit_binding(&receipt),
            Err(ManagerError::MismatchedBindingReceipt(intent.binding_id))
        );
        assert_eq!(manager.stats().pending_bindings, 1);
        assert!(manager.submit_view("r0").unwrap().blocks.is_empty());
        manager.abort_binding(intent.binding_id).unwrap();
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
            Err(ManagerError::PoolExhausted { .. } | ManagerError::LogicalCellCollision(_))
        ));
        let old = certificates[0].physical.clone();
        let receipt = PhysicalReclamationReceipt {
            schema: "orbitkv.physical-reclamation-receipt.v1",
            certificate_id: certificates[0].certificate_id,
            physical: certificates[0].physical.clone(),
        };
        manager.commit_reclamation(&receipt).unwrap();
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
        let mut manager = manager(2, 3);
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
            let receipt = PhysicalReclamationReceipt {
                schema: "orbitkv.physical-reclamation-receipt.v1",
                certificate_id: certificate.certificate_id,
                physical: certificate.physical,
            };
            manager.commit_reclamation(&receipt).unwrap();
        }
        assert_eq!(manager.stats().resident_blocks, 0);
    }

    #[test]
    fn mismatched_physical_receipt_fails_closed() {
        let mut manager = manager(8, 3);
        manager.register_request("r0").unwrap();
        manager.materialize_to("r0", 48).unwrap();
        let certificate = manager
            .advance_semantic_frontier("r0", 47)
            .unwrap()
            .remove(0);
        let mut wrong = certificate.physical.clone();
        wrong.generation += 1;
        assert!(matches!(
            manager.commit_reclamation(&PhysicalReclamationReceipt {
                schema: "orbitkv.physical-reclamation-receipt.v1",
                certificate_id: certificate.certificate_id,
                physical: wrong,
            }),
            Err(ManagerError::MismatchedReclamationReceipt(_))
        ));
        assert_eq!(manager.stats().pending_certificates, 1);
        assert_eq!(manager.stats().free_slots["swa"], 0);
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

    #[test]
    fn physical_pool_cannot_undercut_compiled_live_cell_bound() {
        assert!(matches!(
            KvBlockManager::new(
                hybrid_plan(),
                BlockManagerConfig {
                    pools: vec![
                        ClassPoolConfig {
                            class_name: "full".into(),
                            slot_count: 1,
                        },
                        ClassPoolConfig {
                            class_name: "swa".into(),
                            slot_count: 2,
                        },
                    ],
                },
            ),
            Err(ManagerError::InsufficientPoolSlots {
                class,
                configured: 2,
                minimum: 3,
            }) if class == "swa"
        ));
    }

    #[test]
    fn compiler_simulator_and_manager_agree_on_temporal_addresses() {
        let plan = compile_plan(KvPlanInput {
            page_tokens: 4,
            classes: vec![KvClassSpec {
                name: "swa".into(),
                layers: vec![0],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(9),
            }],
        })
        .unwrap();
        let layout = plan.layout_program().unwrap();
        let slot_count = plan.classes[0].slot_count.unwrap();
        let mut simulator = KvRuntimeSimulator::new(plan.clone()).unwrap();
        let mut manager = KvBlockManager::new(
            plan,
            BlockManagerConfig {
                pools: vec![ClassPoolConfig {
                    class_name: "swa".into(),
                    slot_count,
                }],
            },
        )
        .unwrap();
        manager.register_request("r0").unwrap();

        for boundary in 1..=48 {
            let allocated = manager.materialize_to("r0", boundary).unwrap();
            for block in allocated {
                assert_eq!(
                    block.temporal,
                    layout
                        .temporal_address("r0", &block.logical.class_name, block.logical.ordinal)
                        .unwrap()
                );
            }
            let certificates = manager.advance_semantic_frontier("r0", boundary).unwrap();
            for certificate in certificates {
                let receipt = PhysicalReclamationReceipt {
                    schema: "orbitkv.physical-reclamation-receipt.v1",
                    certificate_id: certificate.certificate_id,
                    physical: certificate.physical,
                };
                manager.commit_reclamation(&receipt).unwrap();
            }
            simulator.append_to(boundary).unwrap();

            let manager_live = manager
                .submit_view("r0")
                .unwrap()
                .blocks
                .into_iter()
                .map(|block| ResidentTemporalBlock {
                    logical: crate::LogicalBlock {
                        class_name: block.logical.class_name,
                        ordinal: block.logical.ordinal,
                    },
                    temporal: block.temporal,
                })
                .collect::<BTreeSet<_>>();
            let simulator_live = simulator
                .resident_temporal_blocks("r0")
                .unwrap()
                .into_iter()
                .collect::<BTreeSet<_>>();
            assert_eq!(manager_live, simulator_live);
            let submission_id = manager.next_submission_id - 1;
            let certificates = manager.complete_submission(submission_id).unwrap();
            for certificate in certificates {
                let receipt = PhysicalReclamationReceipt {
                    schema: "orbitkv.physical-reclamation-receipt.v1",
                    certificate_id: certificate.certificate_id,
                    physical: certificate.physical,
                };
                manager.commit_reclamation(&receipt).unwrap();
            }
        }
    }

    #[test]
    fn sink_sliding_regions_share_layers_but_not_lifetimes() {
        let plan = compile_retention_program(RetentionProgramInput {
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
        .unwrap();
        let mut manager = KvBlockManager::new(
            plan,
            BlockManagerConfig {
                pools: vec![
                    ClassPoolConfig {
                        class_name: "attention::sink".into(),
                        slot_count: 1,
                    },
                    ClassPoolConfig {
                        class_name: "attention::local".into(),
                        slot_count: 3,
                    },
                ],
            },
        )
        .unwrap();
        manager.register_request("r0").unwrap();
        for boundary in 1..=20 {
            manager.materialize_to("r0", boundary).unwrap();
            let certificates = manager.advance_semantic_frontier("r0", boundary).unwrap();
            for certificate in certificates {
                manager
                    .commit_reclamation(&PhysicalReclamationReceipt {
                        schema: "orbitkv.physical-reclamation-receipt.v1",
                        certificate_id: certificate.certificate_id,
                        physical: certificate.physical,
                    })
                    .unwrap();
            }
        }
        let view = manager.submit_view("r0").unwrap();
        let logical = view
            .blocks
            .iter()
            .map(|block| (block.logical.class_name.as_str(), block.logical.ordinal))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            logical,
            BTreeSet::from([
                ("attention::sink", 0),
                ("attention::local", 3),
                ("attention::local", 4),
            ])
        );
    }

    #[test]
    fn chunk_epoch_certificate_gates_resettable_arena_reuse() {
        let plan = compile_retention_program(RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 4,
            states: vec![RetentionStateDecl {
                name: "chunked".into(),
                layers: vec![0],
                kv_head_range: None,
                bytes_per_token_per_layer: 128,
                may_read: Predicate::Equal {
                    lhs: IntExpr::FloorDiv {
                        value: Box::new(IntExpr::QueryPosition),
                        divisor: 16,
                    },
                    rhs: IntExpr::FloorDiv {
                        value: Box::new(IntExpr::KeyPosition),
                        divisor: 16,
                    },
                },
            }],
        })
        .unwrap();
        let mut manager = KvBlockManager::new(
            plan,
            BlockManagerConfig {
                pools: vec![ClassPoolConfig {
                    class_name: "chunked".into(),
                    slot_count: 4,
                }],
            },
        )
        .unwrap();
        manager.register_request("r0").unwrap();
        for boundary in 1..=16 {
            manager.materialize_to("r0", boundary).unwrap();
            let certificates = manager.advance_semantic_frontier("r0", boundary).unwrap();
            if boundary < 16 {
                assert!(certificates.is_empty());
                continue;
            }
            assert_eq!(certificates.len(), 4);
            for certificate in &certificates {
                assert_eq!(
                    certificate.semantic_proof,
                    SemanticProof::ChunkEpoch {
                        semantic_frontier: 16,
                        death_boundary: 16,
                        chunk_tokens: 16,
                    }
                );
            }
            for certificate in certificates {
                manager
                    .commit_reclamation(&PhysicalReclamationReceipt {
                        schema: "orbitkv.physical-reclamation-receipt.v1",
                        certificate_id: certificate.certificate_id,
                        physical: certificate.physical,
                    })
                    .unwrap();
            }
        }
        let next = manager.materialize_to("r0", 17).unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].logical.ordinal, 4);
        assert_eq!(next[0].temporal.cell.cell_index, 0);
        assert_eq!(next[0].temporal.version.cycle, 1);
        assert_eq!(next[0].physical.generation, 2);
    }

    #[test]
    fn manager_materializes_disjoint_head_lifetime_stripes() {
        let state =
            |name: &str, start: u32, end_exclusive: u32, window: i64| -> RetentionStateDecl {
                RetentionStateDecl {
                    name: name.into(),
                    layers: vec![0],
                    kv_head_range: Some(crate::KvHeadRange {
                        start,
                        end_exclusive,
                    }),
                    bytes_per_token_per_layer: u64::from(end_exclusive - start) * 512,
                    may_read: Predicate::LessThan {
                        lhs: IntExpr::Sub {
                            lhs: Box::new(IntExpr::QueryPosition),
                            rhs: Box::new(IntExpr::KeyPosition),
                        },
                        rhs: IntExpr::Constant { value: window },
                    },
                }
            };
        let plan = compile_retention_program(RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 16,
            states: vec![
                state("w512", 0, 8, 512),
                state("w2048", 8, 16, 2048),
                state("w8192", 16, 32, 8192),
            ],
        })
        .unwrap();
        let pools = plan
            .classes
            .iter()
            .map(|class| ClassPoolConfig {
                class_name: class.spec.name.clone(),
                slot_count: class.slot_count.unwrap(),
            })
            .collect();
        let mut manager = KvBlockManager::new(plan, BlockManagerConfig { pools }).unwrap();
        manager.register_request("r0").unwrap();
        for boundary in 1..=64 {
            manager.materialize_to("r0", boundary).unwrap();
            for certificate in manager.advance_semantic_frontier("r0", boundary).unwrap() {
                manager
                    .commit_reclamation(&PhysicalReclamationReceipt {
                        schema: "orbitkv.physical-reclamation-receipt.v1",
                        certificate_id: certificate.certificate_id,
                        physical: certificate.physical,
                    })
                    .unwrap();
            }
        }
        let view = manager.submit_view("r0").unwrap();
        let by_class = view.blocks.into_iter().fold(
            BTreeMap::<String, Vec<u64>>::new(),
            |mut classes, block| {
                classes
                    .entry(block.logical.class_name)
                    .or_default()
                    .push(block.logical.ordinal);
                classes
            },
        );
        assert_eq!(by_class["w512"], vec![0, 1, 2, 3]);
        assert_eq!(by_class["w2048"], vec![0, 1, 2, 3]);
        assert_eq!(by_class["w8192"], vec![0, 1, 2, 3]);
        assert_eq!(manager.stats().resident_blocks, 12);
    }
}
