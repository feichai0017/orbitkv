#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use orbitkv::{
    EngineBatchPlan, ExecutionAddressProgram, ExecutionRetirementProgram, ExecutionTokenBackend,
    RuntimeManifest, TargetAdmissionError, TokenStorageKind, derive_execution_signature,
    kv_manager::TailActionKind,
};
use thiserror::Error;

/// A model-independent attention class accepted by the Luminal executor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionClass {
    pub class_id: u16,
    pub name: String,
    pub layers: Box<[u32]>,
    pub page_tokens: u32,
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
}

/// Physical pool registration retained by the composition layer that created
/// the `OrbitKV` session and the Luminal KV buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutorArena {
    pub class_id: u16,
    pub first_page_id: u32,
    pub page_count: u32,
    pub backend_base_index: u64,
}

/// One request's already-authoritative physical page view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestPageView {
    pub request_id: u64,
    pub query_tokens: u32,
    pub visible_tokens: u64,
    pub page_indices: Box<[u32]>,
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
    pub source: EngineBatchPlan,
    pub steps: Box<[PreparedStep]>,
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
    Admission(#[from] TargetAdmissionError),
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
    #[error("physical token-slot arithmetic overflowed")]
    SlotOverflow,
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
        let signature = derive_execution_signature(manifest)?;
        if !signature.fixed_states.is_empty() {
            return Err(ExecutorError::UnsupportedFixedState);
        }
        let page_tokens = u32::try_from(signature.page_tokens)
            .map_err(|_| ExecutorError::PreparedGeometryMismatch)?;
        let mut classes = Vec::with_capacity(signature.token_classes.len());
        for (class_id, (layout, state)) in signature
            .token_classes
            .iter()
            .zip(&signature.token_states)
            .enumerate()
        {
            if layout.name != state.name || layout.layers != state.layers {
                return Err(ExecutorError::ClassMismatch);
            }
            let ExecutionTokenBackend::TokenSlots {
                storage,
                retention,
                window_tokens,
                ..
            } = state.backend;
            if storage != TokenStorageKind::TokenKv {
                return Err(ExecutorError::UnsupportedStateStorage);
            }
            let visibility = match (&layout.address, &layout.retirement, retention) {
                (
                    ExecutionAddressProgram::AppendOnly,
                    ExecutionRetirementProgram::Never,
                    orbitkv::plan::RetentionKind::Full,
                ) => AttentionVisibility::Full,
                (
                    ExecutionAddressProgram::Periodic { .. }
                    | ExecutionAddressProgram::PeriodicFrom { .. },
                    ExecutionRetirementProgram::BlockEndPlus { .. },
                    orbitkv::plan::RetentionKind::Sliding,
                ) => AttentionVisibility::Sliding {
                    window_tokens: window_tokens.ok_or(ExecutorError::ClassMismatch)?,
                },
                (
                    ExecutionAddressProgram::ResettableArena { blocks_per_epoch },
                    ExecutionRetirementProgram::EpochEnd {
                        blocks_per_epoch: retirement_blocks,
                    },
                    orbitkv::plan::RetentionKind::Chunked,
                ) if blocks_per_epoch == retirement_blocks => AttentionVisibility::Chunked {
                    blocks_per_epoch: *blocks_per_epoch,
                },
                _ => return Err(ExecutorError::ClassMismatch),
            };
            classes.push(AttentionClass {
                class_id: u16::try_from(class_id)
                    .map_err(|_| ExecutorError::PreparedGeometryMismatch)?,
                name: state.name.clone(),
                layers: state.layers.clone().into_boxed_slice(),
                page_tokens,
                visibility,
            });
        }
        if classes.is_empty() {
            return Err(ExecutorError::UnsupportedStateStorage);
        }
        Ok(Self {
            manifest_fingerprint: manifest.fingerprint.clone(),
            page_tokens,
            classes: classes.into_boxed_slice(),
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
        requests: &[RequestPageView],
    ) -> Result<AttentionBatch, ExecutorError> {
        let class = self
            .classes
            .get(usize::from(class_id))
            .filter(|class| class.class_id == class_id)
            .ok_or(ExecutorError::UnknownClass)?;
        if requests.is_empty() {
            return Err(ExecutorError::EmptyBatch);
        }
        let mut ids = BTreeSet::new();
        let mut query_indptr = vec![0_i32];
        let mut page_indptr = vec![0_i32];
        let mut page_indices = Vec::new();
        let mut last_page_len = Vec::with_capacity(requests.len());
        for request in requests {
            if request.query_tokens == 0 || request.visible_tokens == 0 {
                return Err(ExecutorError::InvalidRequestGeometry);
            }
            if !ids.insert(request.request_id) {
                return Err(ExecutorError::DuplicateRequest);
            }
            let expected_pages = request
                .visible_tokens
                .div_ceil(u64::from(class.page_tokens));
            if usize::try_from(expected_pages).ok() != Some(request.page_indices.len()) {
                return Err(ExecutorError::InvalidRequestGeometry);
            }
            push_indptr(&mut query_indptr, u64::from(request.query_tokens))?;
            push_indptr(&mut page_indptr, expected_pages)?;
            page_indices.extend(
                request
                    .page_indices
                    .iter()
                    .copied()
                    .map(|page| i32::try_from(page).map_err(|_| ExecutorError::PageIndexOverflow))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let remainder = request.visible_tokens % u64::from(class.page_tokens);
            last_page_len.push(
                i32::try_from(if remainder == 0 {
                    u64::from(class.page_tokens)
                } else {
                    remainder
                })
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
        if arenas.len() != self.classes.len()
            || arenas
                .iter()
                .enumerate()
                .any(|(class_id, arena)| usize::from(arena.class_id) != class_id)
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

fn lower_class(
    step: &orbitkv::EngineStepPlan,
    lowering: &orbitkv::kv_manager::ClassLowering,
    arenas: &[ExecutorArena],
    page_tokens: u64,
) -> Result<PreparedClass, ExecutorError> {
    let arena = *arenas
        .get(usize::from(lowering.class_id))
        .filter(|arena| arena.class_id == lowering.class_id)
        .ok_or(ExecutorError::UnknownClass)?;
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
            backend_page(page.page_id, arena)?
        } else {
            let first_write = first.div_ceil(page_tokens);
            let index = usize::try_from(
                ordinal
                    .checked_sub(first_write)
                    .ok_or(ExecutorError::PreparedGeometryMismatch)?,
            )
            .map_err(|_| ExecutorError::PreparedGeometryMismatch)?;
            backend_page(
                writes
                    .get(index)
                    .ok_or(ExecutorError::PreparedGeometryMismatch)?
                    .page_id,
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
            Ok(TokenCopy {
                source_slot: token_slot(
                    copy.source_backend_index,
                    copy.source_token_offset,
                    page_tokens,
                )?,
                destination_slot: token_slot(
                    copy.destination_backend_index,
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

fn token_slot(page: u64, offset: u32, page_tokens: u64) -> Result<u64, ExecutorError> {
    page.checked_mul(page_tokens)
        .and_then(|base| base.checked_add(u64::from(offset)))
        .ok_or(ExecutorError::SlotOverflow)
}

fn backend_page(page_id: u32, arena: ExecutorArena) -> Result<u64, ExecutorError> {
    let relative = page_id
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
mod tests {
    use super::*;
    use orbitkv::{
        AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, EngineBatchId,
        EngineRequestId, EngineStepPlan, compile_runtime_manifest,
        kv_manager::{
            ClassLowering, PageLease, TailAction, TailActionKind, ViewVersion, WriteIntent,
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
                &[
                    RequestPageView {
                        request_id: 7,
                        query_tokens: 1,
                        visible_tokens: 18,
                        page_indices: vec![4, 9].into_boxed_slice(),
                    },
                    RequestPageView {
                        request_id: 8,
                        query_tokens: 3,
                        visible_tokens: 16,
                        page_indices: vec![2].into_boxed_slice(),
                    },
                ],
            )
            .unwrap();
        assert_eq!(&*batch.query_indptr, &[0, 1, 4]);
        assert_eq!(&*batch.page_indptr, &[0, 2, 3]);
        assert_eq!(&*batch.page_indices, &[4, 9, 2]);
        assert_eq!(&*batch.last_page_len, &[2, 16]);
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
                    class_id: 0,
                    first_page_id: 10,
                    page_count: 8,
                    backend_base_index: 4,
                }],
            )
            .unwrap();
        assert_eq!(lowered.steps[0].request_id, 9);
        assert_eq!(
            &*lowered.steps[0].classes[0].write_slots,
            &(64_u64..82).collect::<Vec<_>>()
        );
    }
}
