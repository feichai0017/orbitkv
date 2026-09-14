pub(crate) mod support;

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
fn compiles_hybrid_fixed_state_geometry_without_claiming_execution() {
    let plan = ExecutorPlan::compile(&manifest(vec![
        token_state("full", vec![3], RetentionKind::Full, None),
        AttentionStateSpec {
            name: "recurrent".into(),
            layers: vec![0, 1, 2],
            storage: AttentionStateStorage::Recurrent {
                family: orbitkv::RecurrentFamily::Gdn,
                state_bytes_per_layer: 1_048_576,
                checkpoint_slots_per_request: 2,
            },
        },
        AttentionStateSpec {
            name: "convolution".into(),
            layers: vec![0, 1, 2],
            storage: AttentionStateStorage::Convolution {
                state_bytes_per_layer: 36_864,
                kernel_width: 4,
                checkpoint_slots_per_request: 2,
            },
        },
    ]))
    .unwrap();
    assert_eq!(plan.classes.len(), 1);
    assert_eq!(plan.fixed_states.len(), 2);
    assert_eq!(plan.fixed_states[0].state_id, 1);
    assert!(matches!(
        plan.fixed_states[0].storage,
        FixedStateStorage::Recurrent {
            family: orbitkv::RecurrentFamily::Gdn,
            bytes_per_layer: 1_048_576,
            slots_per_request: 2,
            bytes_per_request: 6_291_456,
        }
    ));
    assert!(matches!(
        plan.fixed_states[1].storage,
        FixedStateStorage::Convolution {
            bytes_per_layer: 36_864,
            kernel_width: 4,
            slots_per_request: 2,
            bytes_per_request: 221_184,
        }
    ));
}

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR containing an external hybrid-state checkpoint"]
fn external_hybrid_checkpoint_compiles_executor_state_contract() {
    let directory = std::env::var_os("ORBITKV_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("ORBITKV_MODEL_DIR is required");
    let bytes = std::fs::read(directory.join("config.json")).unwrap();
    let manifest = orbitkv::compile_hf_runtime_manifest(
        &bytes,
        orbitkv::HfRetentionOptions {
            page_tokens: 16,
            kv_dtype_bytes: 2,
        },
    )
    .unwrap();
    let plan = ExecutorPlan::compile(&manifest).unwrap();
    assert!(!plan.classes.is_empty());
    assert_eq!(plan.fixed_states.len(), 2);
    assert!(matches!(
        plan.fixed_states[0].storage,
        FixedStateStorage::Recurrent { .. }
    ));
    assert!(matches!(
        plan.fixed_states[1].storage,
        FixedStateStorage::Convolution { .. }
    ));
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
            fixed_states: Box::default(),
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
