#![forbid(unsafe_code)]

use std::collections::BTreeSet;

mod compiler_facts;
pub use compiler_facts::{LuminalCompilerFacts, LuminalStateClassFacts};

#[cfg(feature = "cuda")]
pub mod cuda;
mod external_tier;
#[cfg(feature = "cuda")]
pub mod model;
pub use external_tier::{
    ExternalRestoreBatch, ExternalRestoreSpan, ExternalTransferBatch, ExternalTransferSpan,
    KvComponent,
};
pub mod transport;
pub use transport::{
    ExternalCopyChecksum, ExternalKvTransport, ExternalTransferOperation, ExternalTransferOutcome,
    ExternalTransportError, HostFaultPoint, HostMemoryTransport, HostTensorRegion,
    HostTransportFault, TransferObservation,
};

use orbitkv::{
    AttentionStateBackend, EngineBatchPlan, EngineBindEvidence, EngineCopyEvidence,
    EnginePreparedBatchView, EngineStepExecutionEvidence, ExecutionEvidence, RuntimeManifest,
    RuntimeManifestError, RuntimeManifestSource, TokenStorageKind,
    kv_manager::{ArenaStats, BackendArenaRegistration, PageLease, TailActionKind, WriteIntent},
    plan::{AddressProgram, RetentionKind, RetirementProgram},
};
use thiserror::Error;

/// A model-independent attention class accepted by the Luminal executor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionClass {
    pub class_id: u16,
    pub name: String,
    pub layers: Box<[u32]>,
    pub page_tokens: u32,
    pub key_bytes_per_token_per_layer: u64,
    pub value_bytes_per_token_per_layer: u64,
    pub visibility: AttentionVisibility,
}

/// Runtime visibility rule compiled from the `OrbitKV` retention program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionVisibility {
    Full,
    Sliding { window_tokens: u64 },
    Chunked { blocks_per_epoch: u64 },
}

/// Immutable part of the joint OrbitKV/Luminal execution contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutorPlan {
    pub manifest_fingerprint: String,
    pub page_tokens: u32,
    pub classes: Box<[AttentionClass]>,
    state_layout_facts: orbitkv::StateLayoutFacts,
}

/// Physical pool registration retained by the composition layer that created
/// the `OrbitKV` session and the Luminal KV buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutorArena {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub pool_id: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub first_page_id: u32,
    pub page_count: u32,
    pub backend_base_index: u64,
}

/// Device metadata consumed directly by Luminal paged attention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionBatch {
    pub class_id: u16,
    pub query_indptr: Box<[i32]>,
    pub page_indptr: Box<[i32]>,
    pub page_indices: Box<[i32]>,
    pub last_page_len: Box<[i32]>,
}

/// Physical writes and copies that must precede one Luminal forward.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedBatch {
    source: EngineBatchPlan,
    steps: Box<[PreparedStep]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedStep {
    pub request_id: u64,
    pub classes: Box<[PreparedClass]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedClass {
    pub class_id: u16,
    pub write_slots: Box<[u64]>,
    pub copies: Box<[TokenCopy]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenCopy {
    pub source_slot: u64,
    pub destination_slot: u64,
    pub token_count: u32,
}

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error(transparent)]
    Manifest(#[from] RuntimeManifestError),
    #[error("Luminal currently accepts token KV only")]
    UnsupportedStateStorage,
    #[error("Luminal fixed-state execution is not implemented")]
    UnsupportedFixedState,
    #[error("compiled attention classes do not match their layout classes")]
    ClassMismatch,
    #[error("attention batch is empty")]
    EmptyBatch,
    #[error("attention request ids must be unique")]
    DuplicateRequest,
    #[error("attention request geometry is invalid")]
    InvalidRequestGeometry,
    #[error("physical page index exceeds Luminal's i32 metadata range")]
    PageIndexOverflow,
    #[error("prepared batch refers to an unknown class")]
    UnknownClass,
    #[error("prepared batch and compiled executor geometry differ")]
    PreparedGeometryMismatch,
    #[error("manager arena and executor buffer registration differ")]
    ArenaRegistrationMismatch,
    #[error("physical token-slot arithmetic overflowed")]
    SlotOverflow,
    #[error("attention kernel geometry is invalid")]
    InvalidKernelGeometry,
    #[error("external transfer plan and executor geometry differ")]
    ExternalTransferMismatch,
    #[error("compiler state/layout facts do not match the executor plan")]
    CompilerFactsMismatch,
}

impl ExecutorArena {
    /// Binds a manager-owned page arena to its executor buffer range.
    ///
    /// # Errors
    ///
    /// Returns an error unless every shared identity and geometry field agrees.
    pub fn bind(
        stats: ArenaStats,
        registration: BackendArenaRegistration,
    ) -> Result<Self, ExecutorError> {
        if registration.reserved != 0
            || stats.class_id != registration.class_id
            || stats.backend_domain != registration.backend_domain
            || stats.pool_id != registration.pool_id
            || stats.page_count != registration.page_count
        {
            return Err(ExecutorError::ArenaRegistrationMismatch);
        }
        registration
            .backend_base_index
            .checked_add(u64::from(registration.page_count))
            .ok_or(ExecutorError::ArenaRegistrationMismatch)?;
        Ok(Self {
            engine_epoch: stats.engine_epoch,
            pool_epoch: stats.pool_epoch,
            pool_id: stats.pool_id,
            class_id: stats.class_id,
            backend_domain: stats.backend_domain,
            first_page_id: stats.first_page_id,
            page_count: stats.page_count,
            backend_base_index: registration.backend_base_index,
        })
    }
}

impl ExecutorPlan {
    /// Compiles the engine-neutral `OrbitKV` manifest into the subset currently
    /// executable by the forked Luminal backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest cannot be projected to token KV
    /// classes supported by the current executor.
    pub fn compile(manifest: &RuntimeManifest) -> Result<Self, ExecutorError> {
        manifest.validate()?;
        let state_layout_facts = manifest
            .state_layout_facts()
            .map_err(|_| ExecutorError::CompilerFactsMismatch)?;
        let manager = manifest
            .token_manager_plan
            .as_ref()
            .ok_or(ExecutorError::UnsupportedStateStorage)?;
        let page_tokens = u32::try_from(manager.layout.page_tokens)
            .map_err(|_| ExecutorError::PreparedGeometryMismatch)?;
        let state_plan = manifest.attention_state_plan.as_ref();
        if state_plan.is_some_and(|plan| {
            plan.states
                .iter()
                .any(|state| !matches!(state.backend, AttentionStateBackend::TokenSlots { .. }))
        }) {
            return Err(ExecutorError::UnsupportedFixedState);
        }
        let classes = manager
            .layout
            .classes
            .iter()
            .enumerate()
            .map(|(class_id, layout)| {
                compile_attention_class(
                    class_id,
                    page_tokens,
                    layout,
                    state_plan,
                    &manifest.source,
                    manager.layout.classes.len(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if classes.is_empty() {
            return Err(ExecutorError::UnsupportedStateStorage);
        }
        Ok(Self {
            manifest_fingerprint: manifest.fingerprint.clone(),
            page_tokens,
            classes: classes.into_boxed_slice(),
            state_layout_facts,
        })
    }

    /// Builds the explicit CSR page-table metadata consumed by Luminal.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown class, duplicate request, invalid token
    /// geometry, or a page index that cannot be represented by the backend.
    pub fn attention_batch(
        &self,
        class_id: u16,
        prepared: &EnginePreparedBatchView,
    ) -> Result<AttentionBatch, ExecutorError> {
        let class = self
            .classes
            .get(usize::from(class_id))
            .filter(|class| class.class_id == class_id)
            .ok_or(ExecutorError::UnknownClass)?;
        if prepared.requests.is_empty() {
            return Err(ExecutorError::EmptyBatch);
        }
        let mut ids = BTreeSet::new();
        let mut query_indptr = vec![0_i32];
        let mut page_indptr = vec![0_i32];
        let mut page_indices = Vec::new();
        let mut last_page_len = Vec::with_capacity(prepared.requests.len());
        for request in &prepared.requests {
            let query_tokens = request
                .target_boundary
                .checked_sub(request.previous_boundary)
                .ok_or(ExecutorError::InvalidRequestGeometry)?;
            if query_tokens == 0 || !ids.insert(request.request_id) {
                return Err(if query_tokens == 0 {
                    ExecutorError::InvalidRequestGeometry
                } else {
                    ExecutorError::DuplicateRequest
                });
            }
            let pages = request
                .pages
                .iter()
                .filter(|page| page.class_id == class_id)
                .collect::<Vec<_>>();
            if pages.is_empty() {
                return Err(ExecutorError::InvalidRequestGeometry);
            }
            if pages.iter().enumerate().any(|(index, page)| {
                page.logical_ordinal != pages[0].logical_ordinal.saturating_add(index as u64)
                    || page.valid_token_count == 0
                    || page.valid_token_count > class.page_tokens
                    || page.visible_token_offset > page.valid_token_count
                    || page.visible_token_count
                        != page.valid_token_count - page.visible_token_offset
            }) {
                return Err(ExecutorError::InvalidRequestGeometry);
            }
            push_indptr(&mut query_indptr, query_tokens)?;
            push_indptr(&mut page_indptr, pages.len() as u64)?;
            page_indices.extend(
                pages
                    .iter()
                    .map(|page| {
                        i32::try_from(page.backend_index)
                            .map_err(|_| ExecutorError::PageIndexOverflow)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let final_page = pages.last().ok_or(ExecutorError::InvalidRequestGeometry)?;
            last_page_len.push(
                i32::try_from(final_page.valid_token_count)
                    .map_err(|_| ExecutorError::InvalidRequestGeometry)?,
            );
        }
        Ok(AttentionBatch {
            class_id,
            query_indptr: query_indptr.into_boxed_slice(),
            page_indptr: page_indptr.into_boxed_slice(),
            page_indices: page_indices.into_boxed_slice(),
            last_page_len: last_page_len.into_boxed_slice(),
        })
    }

    /// Builds one ordered attention batch for every compiled token class.
    ///
    /// The returned order is the canonical class-id order shared by
    /// `PreparedStep::classes` and `DecoderStep::classes`.
    ///
    /// # Errors
    ///
    /// Propagates validation failures from any individual class without
    /// returning a partial list.
    pub fn attention_batches(
        &self,
        prepared: &EnginePreparedBatchView,
    ) -> Result<Box<[AttentionBatch]>, ExecutorError> {
        self.classes
            .iter()
            .map(|class| self.attention_batch(class.class_id, prepared))
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }

    /// Lowers manager-selected write and COW destinations to flat token slots.
    /// No page-table state is inferred or retained here.
    ///
    /// # Errors
    ///
    /// Returns an error if the prepared plan disagrees with registered arena
    /// geometry, contains invalid spans, or overflows physical slot indexing.
    #[allow(clippy::too_many_lines)]
    pub fn lower_prepared(
        &self,
        source: EngineBatchPlan,
        arenas: &[ExecutorArena],
    ) -> Result<PreparedBatch, ExecutorError> {
        validate_arenas(arenas, self.classes.len())?;
        if source.steps.is_empty()
            || source
                .steps
                .iter()
                .any(|step| step.class_lowerings.len() != self.classes.len())
        {
            return Err(ExecutorError::PreparedGeometryMismatch);
        }
        let page_tokens = u64::from(self.page_tokens);
        let steps = source
            .steps
            .iter()
            .map(|step| {
                let classes = step
                    .class_lowerings
                    .iter()
                    .map(|lowering| lower_class(step, lowering, arenas, page_tokens))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PreparedStep {
                    request_id: step.request_id.0,
                    classes: classes.into_boxed_slice(),
                })
            })
            .collect::<Result<Vec<_>, ExecutorError>>()?;
        Ok(PreparedBatch {
            source,
            steps: steps.into_boxed_slice(),
        })
    }
}

fn compile_attention_class(
    class_id: usize,
    page_tokens: u32,
    layout: &orbitkv::plan::ClassLayoutProgram,
    state_plan: Option<&orbitkv::CompiledAttentionStatePlan>,
    source: &RuntimeManifestSource,
    class_count: usize,
) -> Result<AttentionClass, ExecutorError> {
    let (storage, retention, window_tokens, key_bytes, value_bytes) = match source {
        RuntimeManifestSource::AttentionState { .. } => {
            let state = state_plan
                .and_then(|plan| {
                    plan.states
                        .iter()
                        .find(|state| state.name == layout.name && state.layers == layout.layers)
                })
                .ok_or(ExecutorError::ClassMismatch)?;
            let AttentionStateBackend::TokenSlots {
                storage,
                components,
                retention,
                window_tokens,
                ..
            } = &state.backend
            else {
                return Err(ExecutorError::UnsupportedFixedState);
            };
            let component_bytes = |name| {
                components
                    .iter()
                    .find(|component| component.name == name)
                    .map(|component| component.bytes_per_token_per_layer)
                    .ok_or(ExecutorError::ClassMismatch)
            };
            (
                *storage,
                *retention,
                *window_tokens,
                component_bytes("key")?,
                component_bytes("value")?,
            )
        }
        RuntimeManifestSource::RetentionIr { program } => {
            if program.states.len() != 1 || class_count != 1 {
                return Err(ExecutorError::ClassMismatch);
            }
            let key_bytes = layout.bytes_per_token_per_layer / 2;
            (
                TokenStorageKind::TokenKv,
                RetentionKind::Chunked,
                None,
                key_bytes,
                layout.bytes_per_token_per_layer - key_bytes,
            )
        }
    };
    if storage != TokenStorageKind::TokenKv || key_bytes == 0 || value_bytes == 0 {
        return Err(ExecutorError::UnsupportedStateStorage);
    }
    let visibility = match (&layout.address, &layout.retirement, retention) {
        (AddressProgram::AppendOnly, RetirementProgram::Never, RetentionKind::Full) => {
            AttentionVisibility::Full
        }
        (
            AddressProgram::Periodic { .. } | AddressProgram::PeriodicFrom { .. },
            RetirementProgram::BlockEndPlus { .. },
            RetentionKind::Sliding,
        ) => AttentionVisibility::Sliding {
            window_tokens: window_tokens.ok_or(ExecutorError::ClassMismatch)?,
        },
        (
            AddressProgram::ResettableArena { blocks_per_epoch },
            RetirementProgram::EpochEnd {
                blocks_per_epoch: retirement_blocks,
            },
            RetentionKind::Chunked,
        ) if blocks_per_epoch == retirement_blocks => AttentionVisibility::Chunked {
            blocks_per_epoch: *blocks_per_epoch,
        },
        _ => return Err(ExecutorError::ClassMismatch),
    };
    Ok(AttentionClass {
        class_id: u16::try_from(class_id).map_err(|_| ExecutorError::PreparedGeometryMismatch)?,
        name: layout.name.clone(),
        layers: layout.layers.clone().into_boxed_slice(),
        page_tokens,
        key_bytes_per_token_per_layer: key_bytes,
        value_bytes_per_token_per_layer: value_bytes,
        visibility,
    })
}

impl PreparedBatch {
    #[must_use]
    pub const fn batch_id(&self) -> orbitkv::EngineBatchId {
        self.source.batch_id
    }

    #[must_use]
    pub fn steps(&self) -> &[PreparedStep] {
        &self.steps
    }

    /// Builds exact success evidence after the executor has completed every
    /// bind and copy in this prepared batch.
    ///
    /// Calling this method is an assertion by the device executor: it must be
    /// done only after the listed mappings exist and every copy completed
    /// before its dependent writes. `RuntimeSession` independently validates
    /// the returned evidence against its private prepared transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if arena identity, class ordering, page placement, or
    /// the lowered batch differs from the manager-authored source plan.
    pub fn execution_evidence_after_success(
        &self,
        arenas: &[ExecutorArena],
    ) -> Result<ExecutionEvidence, ExecutorError> {
        validate_arenas(
            arenas,
            self.source
                .steps
                .first()
                .map_or(0, |step| step.class_lowerings.len()),
        )?;
        if self.steps.len() != self.source.steps.len() {
            return Err(ExecutorError::PreparedGeometryMismatch);
        }
        let steps = self
            .source
            .steps
            .iter()
            .zip(&self.steps)
            .map(|(source, lowered)| {
                if lowered.request_id != source.request_id.0
                    || lowered.classes.len() != source.class_lowerings.len()
                {
                    return Err(ExecutorError::PreparedGeometryMismatch);
                }
                let mut binds = Vec::new();
                for lowering in &source.class_lowerings {
                    let arena = arena_for(arenas, lowering.class_id)?;
                    for action in checked_span(
                        &source.tail_actions,
                        lowering.tail_offset,
                        lowering.tail_count,
                    )?
                    .iter()
                    .filter(|action| {
                        matches!(
                            action.kind,
                            TailActionKind::CopyOnWrite | TailActionKind::Fresh
                        )
                    }) {
                        binds.push(bind_evidence(action.destination, arena)?);
                    }
                    for write in checked_span(
                        &source.write_intents,
                        lowering.write_offset,
                        lowering.write_count,
                    )? {
                        binds.push(bind_evidence(write_lease(*write, arena), arena)?);
                    }
                }
                let copies = source
                    .copy_intents
                    .iter()
                    .map(|copy| {
                        let arena = arena_for(arenas, copy.class_id)?;
                        validate_copy(copy, arena)?;
                        Ok(EngineCopyEvidence {
                            class_id: copy.class_id,
                            backend_domain: copy.backend_domain,
                            token_count: copy.token_count,
                            source_token_offset: copy.source_token_offset,
                            destination_token_offset: copy.destination_token_offset,
                            observed: true,
                            copied: true,
                            ordered_before_writes: true,
                            source: copy.source,
                            destination: copy.destination,
                            source_backend_index: copy.source_backend_index,
                            destination_backend_index: copy.destination_backend_index,
                        })
                    })
                    .collect::<Result<Vec<_>, ExecutorError>>()?;
                Ok(EngineStepExecutionEvidence {
                    request_id: source.request_id,
                    bind_receipts: binds.into_boxed_slice(),
                    copy_receipts: copies.into_boxed_slice(),
                })
            })
            .collect::<Result<Vec<_>, ExecutorError>>()?;
        Ok(ExecutionEvidence {
            batch_id: self.source.batch_id,
            steps: steps.into_boxed_slice(),
        })
    }
}

fn lower_class(
    step: &orbitkv::EngineStepPlan,
    lowering: &orbitkv::kv_manager::ClassLowering,
    arenas: &[ExecutorArena],
    page_tokens: u64,
) -> Result<PreparedClass, ExecutorError> {
    let arena = arena_for(arenas, lowering.class_id)?;
    let tail = checked_span(
        &step.tail_actions,
        lowering.tail_offset,
        lowering.tail_count,
    )?;
    let writes = checked_span(
        &step.write_intents,
        lowering.write_offset,
        lowering.write_count,
    )?;
    let copies = checked_span(
        &step.copy_intents,
        lowering.copy_offset,
        lowering.copy_count,
    )?;
    let [tail] = tail else {
        return Err(ExecutorError::PreparedGeometryMismatch);
    };
    let first = lowering.previous_layout_boundary;
    let end = lowering.target_layout_boundary;
    if end < first {
        return Err(ExecutorError::PreparedGeometryMismatch);
    }
    let mut write_slots = Vec::with_capacity(
        usize::try_from(end - first).map_err(|_| ExecutorError::PreparedGeometryMismatch)?,
    );
    for physical_token in first..end {
        let ordinal = physical_token / page_tokens;
        let offset = physical_token % page_tokens;
        let physical_page = if ordinal == tail.logical_ordinal && tail.kind != TailActionKind::None
        {
            let page = if tail.kind == TailActionKind::InPlace {
                tail.source
            } else {
                tail.destination
            };
            backend_page_for_lease(page, arena)?
        } else {
            let first_write = first.div_ceil(page_tokens);
            let index = usize::try_from(
                ordinal
                    .checked_sub(first_write)
                    .ok_or(ExecutorError::PreparedGeometryMismatch)?,
            )
            .map_err(|_| ExecutorError::PreparedGeometryMismatch)?;
            backend_page_for_write(
                *writes
                    .get(index)
                    .ok_or(ExecutorError::PreparedGeometryMismatch)?,
                arena,
            )?
        };
        write_slots.push(
            physical_page
                .checked_mul(page_tokens)
                .and_then(|base| base.checked_add(offset))
                .ok_or(ExecutorError::SlotOverflow)?,
        );
    }
    let copies = copies
        .iter()
        .map(|copy| {
            validate_copy(copy, arena)?;
            Ok(TokenCopy {
                source_slot: token_slot(
                    backend_page_for_lease(copy.source, arena)?,
                    copy.source_token_offset,
                    page_tokens,
                )?,
                destination_slot: token_slot(
                    backend_page_for_lease(copy.destination, arena)?,
                    copy.destination_token_offset,
                    page_tokens,
                )?,
                token_count: copy.token_count,
            })
        })
        .collect::<Result<Vec<_>, ExecutorError>>()?;
    Ok(PreparedClass {
        class_id: lowering.class_id,
        write_slots: write_slots.into_boxed_slice(),
        copies: copies.into_boxed_slice(),
    })
}

pub(crate) fn validate_arenas(
    arenas: &[ExecutorArena],
    class_count: usize,
) -> Result<(), ExecutorError> {
    if arenas.len() != class_count
        || arenas.iter().enumerate().any(|(class_id, arena)| {
            usize::from(arena.class_id) != class_id
                || arena.engine_epoch == 0
                || arena.pool_epoch == 0
                || arena.pool_id == 0
                || arena.page_count == 0
                || arena.first_page_id.checked_add(arena.page_count).is_none()
        })
    {
        return Err(ExecutorError::PreparedGeometryMismatch);
    }
    Ok(())
}

pub(crate) fn arena_for(
    arenas: &[ExecutorArena],
    class_id: u16,
) -> Result<ExecutorArena, ExecutorError> {
    arenas
        .get(usize::from(class_id))
        .copied()
        .filter(|arena| arena.class_id == class_id)
        .ok_or(ExecutorError::UnknownClass)
}

fn write_lease(write: WriteIntent, arena: ExecutorArena) -> PageLease {
    PageLease {
        engine_epoch: arena.engine_epoch,
        pool_epoch: arena.pool_epoch,
        generation: write.page_generation,
        page_id: write.page_id,
        pool_id: arena.pool_id,
    }
}

fn bind_evidence(
    page: PageLease,
    arena: ExecutorArena,
) -> Result<EngineBindEvidence, ExecutorError> {
    Ok(EngineBindEvidence {
        page,
        backend_domain: arena.backend_domain,
        mapped: true,
        writable: true,
        backend_index: backend_page_for_lease(page, arena)?,
    })
}

fn validate_copy(
    copy: &orbitkv::kv_manager::CopyIntent,
    arena: ExecutorArena,
) -> Result<(), ExecutorError> {
    if copy.backend_domain != arena.backend_domain
        || copy.source_backend_index != backend_page_for_lease(copy.source, arena)?
        || copy.destination_backend_index != backend_page_for_lease(copy.destination, arena)?
    {
        return Err(ExecutorError::PreparedGeometryMismatch);
    }
    Ok(())
}

pub(crate) fn token_slot(page: u64, offset: u32, page_tokens: u64) -> Result<u64, ExecutorError> {
    page.checked_mul(page_tokens)
        .and_then(|base| base.checked_add(u64::from(offset)))
        .ok_or(ExecutorError::SlotOverflow)
}

fn backend_page_for_write(write: WriteIntent, arena: ExecutorArena) -> Result<u64, ExecutorError> {
    backend_page_for_lease(write_lease(write, arena), arena)
}

pub(crate) fn backend_page_for_lease(
    page: PageLease,
    arena: ExecutorArena,
) -> Result<u64, ExecutorError> {
    if page.engine_epoch != arena.engine_epoch
        || page.pool_epoch != arena.pool_epoch
        || page.pool_id != arena.pool_id
        || page.generation == 0
    {
        return Err(ExecutorError::PreparedGeometryMismatch);
    }
    let relative = page
        .page_id
        .checked_sub(arena.first_page_id)
        .filter(|relative| *relative < arena.page_count)
        .ok_or(ExecutorError::PreparedGeometryMismatch)?;
    arena
        .backend_base_index
        .checked_add(u64::from(relative))
        .ok_or(ExecutorError::SlotOverflow)
}

fn checked_span<T>(values: &[T], offset: u32, count: u32) -> Result<&[T], ExecutorError> {
    let begin = usize::try_from(offset).map_err(|_| ExecutorError::PreparedGeometryMismatch)?;
    let end = begin
        .checked_add(usize::try_from(count).map_err(|_| ExecutorError::PreparedGeometryMismatch)?)
        .ok_or(ExecutorError::PreparedGeometryMismatch)?;
    values
        .get(begin..end)
        .ok_or(ExecutorError::PreparedGeometryMismatch)
}

fn push_indptr(values: &mut Vec<i32>, amount: u64) -> Result<(), ExecutorError> {
    let previous = i64::from(*values.last().expect("indptr always has an origin"));
    let amount = i64::try_from(amount).map_err(|_| ExecutorError::InvalidRequestGeometry)?;
    values.push(
        i32::try_from(
            previous
                .checked_add(amount)
                .ok_or(ExecutorError::InvalidRequestGeometry)?,
        )
        .map_err(|_| ExecutorError::InvalidRequestGeometry)?,
    );
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_executor_plan(
    manifest_fingerprint: &str,
    page_tokens: u32,
    classes: Vec<AttentionClass>,
) -> ExecutorPlan {
    use orbitkv::{
        StateClassLayoutFacts, StateComponentFact, StateLayoutFacts, StateStorageFacts,
        StateStorageKind,
        plan::{AddressProgram, BlockDomain, RetentionKind, RetirementProgram},
    };

    let state_classes = classes
        .iter()
        .map(|class| {
            let (retention, window_tokens, address, retirement) = match class.visibility {
                AttentionVisibility::Full => (
                    RetentionKind::Full,
                    None,
                    AddressProgram::AppendOnly,
                    RetirementProgram::Never,
                ),
                AttentionVisibility::Sliding { window_tokens } => {
                    let period_blocks = 1 + (window_tokens - 1).div_ceil(u64::from(page_tokens));
                    (
                        RetentionKind::Sliding,
                        Some(window_tokens),
                        AddressProgram::Periodic { period_blocks },
                        RetirementProgram::BlockEndPlus {
                            offset_tokens: window_tokens - 1,
                        },
                    )
                }
                AttentionVisibility::Chunked { blocks_per_epoch } => (
                    RetentionKind::Chunked,
                    None,
                    AddressProgram::ResettableArena { blocks_per_epoch },
                    RetirementProgram::EpochEnd { blocks_per_epoch },
                ),
            };
            let bytes_per_token_per_layer = class
                .key_bytes_per_token_per_layer
                .checked_add(class.value_bytes_per_token_per_layer)
                .unwrap();
            StateClassLayoutFacts {
                manager_class_id: Some(class.class_id),
                name: class.name.clone(),
                layers: class.layers.clone(),
                storage: StateStorageFacts::TokenSlots {
                    storage: StateStorageKind::TokenKv,
                    components: vec![
                        StateComponentFact {
                            name: "key".into(),
                            bytes_per_token_per_layer: class.key_bytes_per_token_per_layer,
                        },
                        StateComponentFact {
                            name: "value".into(),
                            bytes_per_token_per_layer: class.value_bytes_per_token_per_layer,
                        },
                    ]
                    .into_boxed_slice(),
                    bytes_per_token_per_layer,
                    page_bytes_per_layer: bytes_per_token_per_layer * u64::from(page_tokens),
                },
                retention: Some(retention),
                window_tokens,
                address: Some(address),
                retirement: Some(retirement),
                block_domain: Some(BlockDomain::all()),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    ExecutorPlan {
        manifest_fingerprint: manifest_fingerprint.into(),
        page_tokens,
        state_layout_facts: StateLayoutFacts {
            manifest_fingerprint: manifest_fingerprint.into(),
            page_tokens: u64::from(page_tokens),
            classes: state_classes,
        },
        classes: classes.into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbitkv::{
        AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
        EngineAppendIntent, EngineBatchId, EngineCompletionEvidence, EnginePreparedBatchView,
        EnginePreparedRequestView, EnginePublicationEvidence, EngineRequestId,
        EngineRetirementEvidence, EngineStepPlan, RuntimeSession, compile_plan,
        compile_runtime_manifest,
        kv_manager::{
            BackendArenaRegistration, CanonicalKvManager, ClassLowering, ManagerConfig, PageLease,
            PhysicalResidencePolicy, SnapshotPage, TailAction, TailActionKind, ViewVersion,
            WriteIntent,
        },
        plan::RetentionKind,
    };

    fn manifest(states: Vec<AttentionStateSpec>) -> RuntimeManifest {
        compile_runtime_manifest(AttentionStatePlanInput {
            page_tokens: 16,
            states,
        })
        .unwrap()
    }

    fn token_state(
        name: &str,
        layers: Vec<u32>,
        retention: RetentionKind,
        window_tokens: Option<u64>,
    ) -> AttentionStateSpec {
        AttentionStateSpec {
            name: name.into(),
            layers,
            storage: AttentionStateStorage::TokenKv {
                key_bytes_per_token_per_layer: 128,
                value_bytes_per_token_per_layer: 128,
                retention,
                window_tokens,
            },
        }
    }

    fn retirement_evidence(
        retirements: &[orbitkv::EngineRetirement],
    ) -> Box<[EngineRetirementEvidence]> {
        retirements
            .iter()
            .map(|retirement| EngineRetirementEvidence {
                page: retirement.page,
                backend_domain: retirement.backend_domain,
                acknowledged: true,
                backend_index: retirement.backend_index,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[test]
    fn compiles_full_and_sliding_classes() {
        let plan = ExecutorPlan::compile(&manifest(vec![
            token_state("full", vec![0, 2], RetentionKind::Full, None),
            token_state("sliding", vec![1, 3], RetentionKind::Sliding, Some(64)),
        ]))
        .unwrap();
        assert_eq!(plan.page_tokens, 16);
        assert_eq!(plan.classes.len(), 2);
        assert_eq!(plan.classes[0].visibility, AttentionVisibility::Full);
        assert_eq!(
            plan.classes[1].visibility,
            AttentionVisibility::Sliding { window_tokens: 64 }
        );
    }

    #[test]
    fn builds_flashinfer_csr_without_owning_page_state() {
        let plan = ExecutorPlan::compile(&manifest(vec![token_state(
            "full",
            vec![0, 1],
            RetentionKind::Full,
            None,
        )]))
        .unwrap();
        let batch = plan
            .attention_batch(
                0,
                &EnginePreparedBatchView {
                    batch_id: EngineBatchId::from_parts(1, 1),
                    requests: vec![
                        prepared_request(7, 17, 18, &[(0, 4, 16), (1, 9, 2)]),
                        prepared_request(8, 13, 16, &[(0, 2, 16)]),
                    ]
                    .into_boxed_slice(),
                },
            )
            .unwrap();
        assert_eq!(&*batch.query_indptr, &[0, 1, 4]);
        assert_eq!(&*batch.page_indptr, &[0, 2, 3]);
        assert_eq!(&*batch.page_indices, &[4, 9, 2]);
        assert_eq!(&*batch.last_page_len, &[2, 16]);
    }

    #[test]
    fn builds_prefill_csr_from_pages_retained_for_earlier_queries() {
        let plan = ExecutorPlan::compile(&manifest(vec![token_state(
            "sliding",
            vec![0],
            RetentionKind::Sliding,
            Some(18),
        )]))
        .unwrap();
        let mut request = prepared_request(7, 18, 35, &[(0, 4, 16), (1, 9, 16), (2, 6, 3)]);
        request.pages[0].visible_token_offset = 16;
        request.pages[0].visible_token_count = 0;
        request.pages[1].visible_token_offset = 1;
        request.pages[1].visible_token_count = 15;
        let batch = plan
            .attention_batch(
                0,
                &EnginePreparedBatchView {
                    batch_id: EngineBatchId::from_parts(1, 1),
                    requests: vec![request].into_boxed_slice(),
                },
            )
            .unwrap();
        assert_eq!(&*batch.query_indptr, &[0, 17]);
        assert_eq!(&*batch.page_indptr, &[0, 3]);
        assert_eq!(&*batch.page_indices, &[4, 9, 6]);
        assert_eq!(&*batch.last_page_len, &[3]);
    }

    fn prepared_request(
        request_id: u64,
        previous_boundary: u64,
        target_boundary: u64,
        pages: &[(u64, u64, u32)],
    ) -> EnginePreparedRequestView {
        EnginePreparedRequestView {
            request_id: EngineRequestId(request_id),
            previous_boundary,
            target_boundary,
            pages: pages
                .iter()
                .map(
                    |&(logical_ordinal, backend_index, valid_token_count)| SnapshotPage {
                        class_id: 0,
                        backend_domain: 0,
                        logical_ordinal,
                        temporal_cell_index: logical_ordinal,
                        temporal_cycle: 0,
                        page: PageLease {
                            engine_epoch: 1,
                            pool_epoch: 1,
                            generation: 1,
                            page_id: u32::try_from(logical_ordinal + 1).unwrap(),
                            pool_id: 7,
                        },
                        backend_index,
                        valid_token_count,
                        visible_token_offset: 0,
                        visible_token_count: valid_token_count,
                    },
                )
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    #[test]
    fn lowers_manager_pages_into_executor_token_slots() {
        let plan = ExecutorPlan::compile(&manifest(vec![token_state(
            "full",
            vec![0, 1],
            RetentionKind::Full,
            None,
        )]))
        .unwrap();
        let page = |page_id| PageLease {
            engine_epoch: 1,
            pool_epoch: 1,
            generation: 1,
            page_id,
            pool_id: 7,
        };
        let source = EngineBatchPlan {
            batch_id: EngineBatchId::from_parts(1, 1),
            steps: vec![EngineStepPlan {
                request_id: EngineRequestId(9),
                base_view_version: ViewVersion(1),
                target_view_version: ViewVersion(2),
                previous_boundary: 0,
                target_boundary: 18,
                class_lowerings: vec![ClassLowering {
                    class_id: 0,
                    flags: 0,
                    tail_offset: 0,
                    tail_count: 1,
                    copy_offset: 0,
                    copy_count: 0,
                    write_offset: 0,
                    write_count: 2,
                    reserved: 0,
                    previous_layout_boundary: 0,
                    target_layout_boundary: 18,
                }]
                .into_boxed_slice(),
                tail_actions: vec![TailAction {
                    class_id: 0,
                    kind: TailActionKind::None,
                    valid_token_count: 0,
                    logical_ordinal: 0,
                    source: PageLease::default(),
                    destination: PageLease::default(),
                    reserved: 0,
                }]
                .into_boxed_slice(),
                copy_intents: Box::default(),
                write_intents: vec![
                    WriteIntent {
                        page_generation: 1,
                        page_id: page(10).page_id,
                        reserved: 0,
                    },
                    WriteIntent {
                        page_generation: 1,
                        page_id: page(11).page_id,
                        reserved: 0,
                    },
                ]
                .into_boxed_slice(),
            }]
            .into_boxed_slice(),
        };
        let lowered = plan
            .lower_prepared(
                source,
                &[ExecutorArena {
                    engine_epoch: 1,
                    pool_epoch: 1,
                    pool_id: 7,
                    class_id: 0,
                    backend_domain: 0,
                    first_page_id: 10,
                    page_count: 8,
                    backend_base_index: 4,
                }],
            )
            .unwrap();
        assert_eq!(lowered.steps()[0].request_id, 9);
        assert_eq!(
            &*lowered.steps()[0].classes[0].write_slots,
            &(64_u64..82).collect::<Vec<_>>()
        );
    }

    #[test]
    fn successful_lowering_produces_session_accepted_execution_evidence() {
        let state = token_state("full", vec![0, 1], RetentionKind::Full, None);
        let manifest = manifest(vec![state.clone()]);
        let manager_plan = compile_plan(
            orbitkv::compile_attention_state_plan(AttentionStatePlanInput {
                page_tokens: 16,
                states: vec![state],
            })
            .unwrap()
            .token_manager_plan()
            .unwrap(),
        )
        .unwrap();
        let registration = BackendArenaRegistration {
            pool_id: 7,
            class_id: 0,
            backend_domain: 11,
            page_count: 8,
            reserved: 0,
            backend_base_index: 4,
        };
        let manager = CanonicalKvManager::new(
            &manager_plan,
            ManagerConfig {
                maximum_requests: 2,
                maximum_operations: 4,
                maximum_prefixes: 1,
                maximum_reclamations: 8,
                maximum_step_tokens: 64,
            },
            &[registration],
        )
        .unwrap();
        let mut session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
        let request_id = EngineRequestId(9);
        session.acquire_requests(&[request_id]).unwrap();
        let source = session
            .prepare_append_batch(&[EngineAppendIntent {
                request_id,
                target_boundary: 18,
            }])
            .unwrap();
        let device_view = session.prepared_execution_view(source.batch_id).unwrap();
        let arena_stats = session.arena_stats()[0];
        let arenas = [ExecutorArena::bind(arena_stats, registration).unwrap()];
        let executor_plan = ExecutorPlan::compile(&manifest).unwrap();
        let attention = executor_plan.attention_batch(0, &device_view).unwrap();
        assert_eq!(&*attention.query_indptr, &[0, 18]);
        assert_eq!(&*attention.page_indptr, &[0, 2]);
        assert_eq!(&*attention.page_indices, &[4, 5]);
        assert_eq!(&*attention.last_page_len, &[2]);
        let prepared = executor_plan.lower_prepared(source, &arenas).unwrap();
        let evidence = prepared.execution_evidence_after_success(&arenas).unwrap();
        let ticket = session.submit_execution(&evidence).unwrap();
        assert_eq!(ticket.batch_id(), prepared.batch_id());
        let publication = session
            .complete_execution_by_batch(
                ticket.batch_id(),
                EngineCompletionEvidence {
                    completion_domain: 1,
                    completion_value: 1,
                    confirmed: true,
                },
            )
            .unwrap();
        session
            .confirm_publication(&EnginePublicationEvidence {
                publication_id: publication.publication_id,
                mirror_cleanup_confirmed: true,
                reclamation_receipts: Box::default(),
            })
            .unwrap();
        assert_eq!(session.stats().active_requests, 1);
    }

    #[test]
    fn compiled_and_request_lifetime_residence_lower_to_equivalent_csr_geometry() {
        let state = token_state("sliding", vec![0], RetentionKind::Sliding, Some(18));
        let manifest = manifest(vec![state.clone()]);
        let manager_plan = compile_plan(
            orbitkv::compile_attention_state_plan(AttentionStatePlanInput {
                page_tokens: 16,
                states: vec![state],
            })
            .unwrap()
            .token_manager_plan()
            .unwrap(),
        )
        .unwrap();
        let executor_plan = ExecutorPlan::compile(&manifest).unwrap();
        let request_id = EngineRequestId(19);
        let mut batches = Vec::new();

        for (pool_id, policy) in [
            (31, PhysicalResidencePolicy::Compiled),
            (32, PhysicalResidencePolicy::RequestLifetime),
        ] {
            let registration = BackendArenaRegistration {
                pool_id,
                class_id: 0,
                backend_domain: 11,
                page_count: 8,
                reserved: 0,
                backend_base_index: 4,
            };
            let manager = CanonicalKvManager::new_with_residence(
                &manager_plan,
                ManagerConfig {
                    maximum_requests: 1,
                    maximum_operations: 4,
                    maximum_prefixes: 1,
                    maximum_reclamations: 8,
                    maximum_step_tokens: 64,
                },
                &[registration],
                policy,
            )
            .unwrap();
            let mut session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
            session.acquire_requests(&[request_id]).unwrap();
            let arena = ExecutorArena::bind(session.arena_stats()[0], registration).unwrap();
            let initial = session
                .prepare_append_batch(&[EngineAppendIntent {
                    request_id,
                    target_boundary: 35,
                }])
                .unwrap();
            let lowered = executor_plan.lower_prepared(initial, &[arena]).unwrap();
            let evidence = lowered.execution_evidence_after_success(&[arena]).unwrap();
            let ticket = session.submit_execution(&evidence).unwrap();
            let publication = session
                .complete_execution_by_batch(
                    ticket.batch_id(),
                    EngineCompletionEvidence {
                        completion_domain: 7,
                        completion_value: 1,
                        confirmed: true,
                    },
                )
                .unwrap();
            session
                .confirm_publication(&EnginePublicationEvidence {
                    publication_id: publication.publication_id,
                    mirror_cleanup_confirmed: true,
                    reclamation_receipts: retirement_evidence(&publication.retirements),
                })
                .unwrap();

            let next = session
                .prepare_append_batch(&[EngineAppendIntent {
                    request_id,
                    target_boundary: 52,
                }])
                .unwrap();
            let view = session.prepared_execution_view(next.batch_id).unwrap();
            batches.push(executor_plan.attention_batch(0, &view).unwrap());
            session
                .abort_prepared_execution(
                    next.batch_id,
                    &[orbitkv::EngineStepAbortEvidence {
                        request_id,
                        backend_unobserved: true,
                    }],
                )
                .unwrap();
        }

        assert_eq!(batches[0].query_indptr, batches[1].query_indptr);
        assert_eq!(batches[0].page_indptr, batches[1].page_indptr);
        assert_eq!(batches[0].last_page_len, batches[1].last_page_len);
        assert_eq!(batches[0].page_indices.len(), batches[1].page_indices.len());
        assert_ne!(batches[0].page_indices, batches[1].page_indices);
    }
}
