use std::collections::BTreeSet;

use orbitkv::{
    EnginePreparedRelocation, EngineRelocationId, EngineRequestId, kv_manager::TokenLocation,
};
#[cfg(any(feature = "cuda", test))]
use orbitkv::{
    EngineRelocationCopyEvidence, EngineRelocationExecutionEvidence,
    EngineRelocationRequestEvidence,
};

use crate::{
    AttentionClass, ExecutorArena, ExecutorError, ExecutorPlan, arena_for, backend_page_for_lease,
    token_slot, validate_arenas,
};

/// One manager-authored token movement lowered to physical executor slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelocationCopy {
    pub token_id: u64,
    pub source_slot: u64,
    pub destination_slot: u64,
    source: TokenLocation,
    destination: TokenLocation,
}

/// One contiguous component copy in a device cache tensor.
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RelocationByteRange {
    pub source_offset: usize,
    pub destination_offset: usize,
    pub bytes: usize,
}

/// One request/class relocation with immutable component geometry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelocationRequest {
    pub request_id: EngineRequestId,
    pub class_id: u16,
    pub layers: Box<[u32]>,
    pub key_bytes_per_token_per_layer: u64,
    pub value_bytes_per_token_per_layer: u64,
    pub copies: Box<[RelocationCopy]>,
}

/// Validated relocation work whose page choices remain owned by `OrbitKV`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelocationBatch {
    relocation_id: EngineRelocationId,
    requests: Box<[RelocationRequest]>,
}

impl ExecutorPlan {
    /// Lowers a canonical relocation plan to physical token slots.
    ///
    /// # Errors
    ///
    /// Rejects mismatched arenas, classes, page identities, backend indices,
    /// token offsets, or overlapping source/destination slots before device
    /// work is submitted.
    pub fn lower_relocation(
        &self,
        source: &EnginePreparedRelocation,
        arenas: &[ExecutorArena],
    ) -> Result<RelocationBatch, ExecutorError> {
        validate_arenas(arenas, self.classes.len())?;
        if source.plans.is_empty() {
            return Err(ExecutorError::EmptyBatch);
        }
        let mut request_ids = BTreeSet::new();
        let mut page_ids = BTreeSet::new();
        let requests = source
            .plans
            .iter()
            .map(|plan| {
                if !request_ids.insert(plan.request_id) {
                    return Err(ExecutorError::DuplicateRequest);
                }
                if plan
                    .source_pages
                    .iter()
                    .chain(&plan.destination_pages)
                    .any(|page| !page_ids.insert(page.page_id))
                {
                    return Err(ExecutorError::PreparedGeometryMismatch);
                }
                let class = self
                    .classes
                    .get(usize::from(plan.class_id))
                    .filter(|class| class.class_id == plan.class_id)
                    .ok_or(ExecutorError::UnknownClass)?;
                lower_relocation_request(plan, class, arena_for(arenas, plan.class_id)?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RelocationBatch {
            relocation_id: source.relocation_id,
            requests: requests.into_boxed_slice(),
        })
    }
}

impl RelocationBatch {
    #[must_use]
    pub const fn relocation_id(&self) -> EngineRelocationId {
        self.relocation_id
    }

    #[must_use]
    pub fn requests(&self) -> &[RelocationRequest] {
        &self.requests
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn execution_evidence_after_success(&self) -> EngineRelocationExecutionEvidence {
        EngineRelocationExecutionEvidence {
            relocation_id: self.relocation_id,
            requests: self
                .requests
                .iter()
                .map(|request| EngineRelocationRequestEvidence {
                    request_id: request.request_id,
                    copies: request
                        .copies
                        .iter()
                        .map(|copy| EngineRelocationCopyEvidence {
                            token_id: copy.token_id,
                            source: copy.source,
                            destination: copy.destination,
                            observed: true,
                            copied: true,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

#[cfg(any(feature = "cuda", test))]
impl RelocationRequest {
    pub(crate) fn key_ranges(&self) -> Result<Box<[RelocationByteRange]>, ExecutorError> {
        component_ranges(&self.copies, self.key_bytes_per_token_per_layer)
    }

    pub(crate) fn value_ranges(&self) -> Result<Box<[RelocationByteRange]>, ExecutorError> {
        component_ranges(&self.copies, self.value_bytes_per_token_per_layer)
    }
}

fn lower_relocation_request(
    plan: &orbitkv::EngineRelocationPlan,
    class: &AttentionClass,
    arena: ExecutorArena,
) -> Result<RelocationRequest, ExecutorError> {
    if class.layers.is_empty()
        || !class.token_relocatable
        || class.key_bytes_per_token_per_layer == 0
        || class.value_bytes_per_token_per_layer == 0
    {
        return Err(ExecutorError::PreparedGeometryMismatch);
    }
    let sources = plan.source_pages.iter().copied().collect::<BTreeSet<_>>();
    let destinations = plan
        .destination_pages
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if sources.len() != plan.source_pages.len()
        || destinations.len() != plan.destination_pages.len()
        || sources.is_empty()
        || !sources.is_disjoint(&destinations)
    {
        return Err(ExecutorError::PreparedGeometryMismatch);
    }
    let expected_destinations = plan.moves.len().div_ceil(class.page_tokens as usize);
    if plan.destination_pages.len() != expected_destinations
        || usize::try_from(plan.projected_reclaimed_pages).ok()
            != plan
                .source_pages
                .len()
                .checked_sub(plan.destination_pages.len())
    {
        return Err(ExecutorError::PreparedGeometryMismatch);
    }
    for page in sources.iter().chain(&destinations) {
        backend_page_for_lease(*page, arena)?;
    }
    let page_tokens = u64::from(class.page_tokens);
    let mut tokens = BTreeSet::new();
    let mut source_slots = BTreeSet::new();
    let mut destination_slots = BTreeSet::new();
    let copies = plan
        .moves
        .iter()
        .map(|movement| {
            if movement.source.reserved != 0
                || movement.destination.reserved != 0
                || movement.source.offset >= class.page_tokens
                || movement.destination.offset >= class.page_tokens
                || !sources.contains(&movement.source.page)
                || !destinations.contains(&movement.destination.page)
                || movement.source.backend_index
                    != backend_page_for_lease(movement.source.page, arena)?
                || movement.destination.backend_index
                    != backend_page_for_lease(movement.destination.page, arena)?
            {
                return Err(ExecutorError::PreparedGeometryMismatch);
            }
            let source_slot = token_slot(
                movement.source.backend_index,
                movement.source.offset,
                page_tokens,
            )?;
            let destination_slot = token_slot(
                movement.destination.backend_index,
                movement.destination.offset,
                page_tokens,
            )?;
            if !tokens.insert(movement.token_id)
                || !source_slots.insert(source_slot)
                || !destination_slots.insert(destination_slot)
            {
                return Err(ExecutorError::PreparedGeometryMismatch);
            }
            Ok(RelocationCopy {
                token_id: movement.token_id,
                source_slot,
                destination_slot,
                source: movement.source,
                destination: movement.destination,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RelocationRequest {
        request_id: plan.request_id,
        class_id: plan.class_id,
        layers: class.layers.clone(),
        key_bytes_per_token_per_layer: class.key_bytes_per_token_per_layer,
        value_bytes_per_token_per_layer: class.value_bytes_per_token_per_layer,
        copies: copies.into_boxed_slice(),
    })
}

#[cfg(any(feature = "cuda", test))]
fn component_ranges(
    copies: &[RelocationCopy],
    bytes_per_token: u64,
) -> Result<Box<[RelocationByteRange]>, ExecutorError> {
    copies
        .iter()
        .map(|copy| {
            Ok(RelocationByteRange {
                source_offset: byte_offset(copy.source_slot, bytes_per_token)?,
                destination_offset: byte_offset(copy.destination_slot, bytes_per_token)?,
                bytes: usize::try_from(bytes_per_token).map_err(|_| ExecutorError::SlotOverflow)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

#[cfg(any(feature = "cuda", test))]
fn byte_offset(slot: u64, bytes_per_token: u64) -> Result<usize, ExecutorError> {
    usize::try_from(
        slot.checked_mul(bytes_per_token)
            .ok_or(ExecutorError::SlotOverflow)?,
    )
    .map_err(|_| ExecutorError::SlotOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbitkv::{
        AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
        EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence,
        EngineRelocationPlan, EngineRelocationPublicationEvidence, EngineRetirementEvidence,
        EngineTokenDispositionBatchItem, EngineTokenDispositionUpdate, RuntimeSession,
        compile_attention_state_plan, compile_plan, compile_runtime_manifest,
        kv_manager::{
            BackendArenaRegistration, CanonicalKvManager, ManagerConfig, PageLease,
            RelocationPolicy, TokenDisposition, TokenMove, ViewVersion,
        },
        plan::RetentionKind,
    };

    fn page(id: u32) -> PageLease {
        PageLease {
            engine_epoch: 1,
            pool_epoch: 1,
            generation: 1,
            page_id: id,
            pool_id: 7,
        }
    }

    fn location(id: u32, backend_index: u64, offset: u32) -> TokenLocation {
        TokenLocation {
            page: page(id),
            backend_index,
            offset,
            reserved: 0,
        }
    }

    fn plan() -> ExecutorPlan {
        crate::test_executor_plan(
            "test",
            16,
            vec![AttentionClass {
                class_id: 0,
                name: "attention".into(),
                layers: vec![1, 3].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 8,
                value_bytes_per_token_per_layer: 12,
                token_relocatable: true,
                visibility: crate::AttentionVisibility::Full,
            }],
        )
    }

    fn arena() -> ExecutorArena {
        ExecutorArena {
            engine_epoch: 1,
            pool_epoch: 1,
            pool_id: 7,
            class_id: 0,
            backend_domain: 9,
            first_page_id: 10,
            page_count: 8,
            backend_base_index: 20,
        }
    }

    fn prepared() -> EnginePreparedRelocation {
        EnginePreparedRelocation {
            relocation_id: EngineRelocationId::from_parts(1, 4),
            plans: vec![EngineRelocationPlan {
                request_id: EngineRequestId(5),
                class_id: 0,
                base_version: ViewVersion(2),
                target_version: ViewVersion(3),
                fragmentation_milli: 500,
                source_pages: vec![page(10), page(11)].into_boxed_slice(),
                destination_pages: vec![page(12)].into_boxed_slice(),
                moves: vec![
                    TokenMove {
                        token_id: 2,
                        source: location(10, 20, 2),
                        destination: location(12, 22, 0),
                    },
                    TokenMove {
                        token_id: 17,
                        source: location(11, 21, 1),
                        destination: location(12, 22, 1),
                    },
                ]
                .into_boxed_slice(),
                projected_reclaimed_pages: 1,
            }]
            .into_boxed_slice(),
        }
    }

    fn lifecycle_fixture() -> (RuntimeSession, ExecutorPlan, ExecutorArena, EngineRequestId) {
        let state = AttentionStateSpec {
            name: "attention".into(),
            layers: vec![0],
            storage: AttentionStateStorage::TokenKv {
                key_bytes_per_token_per_layer: 8,
                value_bytes_per_token_per_layer: 12,
                retention: RetentionKind::Full,
                window_tokens: None,
            },
        };
        let input = AttentionStatePlanInput {
            page_tokens: 16,
            states: vec![state],
        };
        let manifest = compile_runtime_manifest(input.clone()).unwrap();
        let manager_plan = compile_plan(
            compile_attention_state_plan(input)
                .unwrap()
                .token_manager_plan()
                .unwrap(),
        )
        .unwrap();
        let registration = BackendArenaRegistration {
            pool_id: 7,
            class_id: 0,
            backend_domain: 9,
            page_count: 8,
            reserved: 0,
            backend_base_index: 20,
        };
        let manager = CanonicalKvManager::new(
            &manager_plan,
            ManagerConfig {
                maximum_requests: 1,
                maximum_operations: 2,
                maximum_prefixes: 1,
                maximum_reclamations: 8,
                maximum_step_tokens: 64,
            },
            &[registration],
        )
        .unwrap();
        let mut session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
        let request_id = EngineRequestId(5);
        session.acquire_requests(&[request_id]).unwrap();
        let arena = ExecutorArena::bind(session.arena_stats()[0], registration).unwrap();
        (
            session,
            ExecutorPlan::compile(&manifest).unwrap(),
            arena,
            request_id,
        )
    }

    fn append_initial(
        session: &mut RuntimeSession,
        executor: &ExecutorPlan,
        arena: ExecutorArena,
        request_id: EngineRequestId,
    ) {
        let append = session
            .prepare_append_batch(&[EngineAppendIntent {
                request_id,
                target_boundary: 48,
            }])
            .unwrap();
        let lowered = executor.lower_prepared(append, &[arena]).unwrap();
        let ticket = session
            .submit_execution(&lowered.execution_evidence_after_success(&[arena]).unwrap())
            .unwrap();
        let publication = session
            .complete_execution_by_batch(
                ticket.batch_id(),
                EngineCompletionEvidence {
                    completion_domain: 9,
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
    }

    fn mark_and_prepare_relocation(
        session: &mut RuntimeSession,
        request_id: EngineRequestId,
    ) -> EnginePreparedRelocation {
        let updates = (0..48)
            .filter(|token| token % 16 >= 8)
            .map(|token_id| EngineTokenDispositionUpdate {
                class_id: 0,
                token_id,
                disposition: TokenDisposition::policy_evicted(1, 1, 1),
            })
            .collect::<Vec<_>>();
        session
            .mark_token_dispositions_batch(&[EngineTokenDispositionBatchItem {
                request_id,
                updates: updates.into_boxed_slice(),
            }])
            .unwrap();
        session
            .prepare_relocation_batch(&[orbitkv::EnginePrepareRelocationItem {
                request_id,
                class_id: 0,
                policy: RelocationPolicy::static_fragmentation(250, 8, 2, true),
            }])
            .unwrap()
    }

    fn confirm_relocation(
        session: &mut RuntimeSession,
        batch: &RelocationBatch,
    ) -> Box<[PageLease]> {
        let ticket = session
            .submit_relocation(&batch.execution_evidence_after_success())
            .unwrap();
        let publication = session
            .complete_relocation(
                ticket.relocation_id(),
                EngineCompletionEvidence {
                    completion_domain: 9,
                    completion_value: 2,
                    confirmed: true,
                },
            )
            .unwrap();
        let retired = publication
            .retirements
            .iter()
            .map(|retirement| EngineRetirementEvidence {
                page: retirement.page,
                backend_domain: retirement.backend_domain,
                acknowledged: true,
                backend_index: retirement.backend_index,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        session
            .confirm_relocation_publication(&EngineRelocationPublicationEvidence {
                relocation_id: publication.relocation_id,
                mirror_cleanup_confirmed: true,
                reclamation_receipts: retired,
            })
            .unwrap();
        publication
            .retirements
            .iter()
            .map(|retirement| retirement.page)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[test]
    fn lowers_manager_relocation_to_component_byte_ranges() {
        let batch = plan().lower_relocation(&prepared(), &[arena()]).unwrap();
        let request = &batch.requests()[0];
        assert_eq!(&*request.layers, &[1, 3]);
        assert_eq!(
            &*request.key_ranges().unwrap(),
            &[
                RelocationByteRange {
                    source_offset: 2_576,
                    destination_offset: 2_816,
                    bytes: 8,
                },
                RelocationByteRange {
                    source_offset: 2_696,
                    destination_offset: 2_824,
                    bytes: 8,
                },
            ]
        );
        assert_eq!(request.value_ranges().unwrap()[0].bytes, 12);
        assert_eq!(
            batch.execution_evidence_after_success().requests[0]
                .copies
                .len(),
            2
        );
    }

    #[test]
    fn rejects_forged_backend_index_before_copy() {
        let mut prepared = prepared();
        prepared.plans[0].moves[0].destination.backend_index += 1;
        assert!(matches!(
            plan().lower_relocation(&prepared, &[arena()]),
            Err(ExecutorError::PreparedGeometryMismatch)
        ));
    }

    #[test]
    fn lowered_relocation_evidence_completes_the_canonical_lifecycle() {
        let (mut session, executor, arena, request_id) = lifecycle_fixture();
        append_initial(&mut session, &executor, arena, request_id);
        let prepared = mark_and_prepare_relocation(&mut session, request_id);
        let destination_pages = prepared.plans[0].destination_pages.clone();
        let lowered = executor.lower_relocation(&prepared, &[arena]).unwrap();
        let retired_pages = confirm_relocation(&mut session, &lowered);
        assert!(
            retired_pages
                .iter()
                .all(|page| !destination_pages.contains(page))
        );
        let view = session
            .token_views_batch(&[orbitkv::EngineTokenViewQuery {
                request_id,
                class_id: 0,
                expected_boundary: 48,
            }])
            .unwrap();
        assert_eq!(
            view[0]
                .placements
                .iter()
                .filter(|placement| placement.disposition.retained())
                .count(),
            24
        );
        assert!(view[0].placements.iter().all(|placement| {
            placement
                .location
                .is_none_or(|location| destination_pages.contains(&location.page))
        }));
    }
}
