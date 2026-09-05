#![forbid(unsafe_code)]

use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence, EngineRequestId,
    ExternalExportAbortEvidence, ExternalObjectKey, ExternalReplica, ExternalReplicaTarget,
    RuntimeSession, RuntimeSessionError, compile_attention_state_plan, compile_plan,
    compile_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
    plan::{CompiledKvPlan, RetentionKind},
};
use orbitkv_executor::{
    ExecutorArena, ExecutorPlan, ExternalKvTransport, ExternalRestoreBatch, ExternalTransferBatch,
    ExternalTransferOperation, HostFaultPoint, HostMemoryTransport, HostTensorRegion,
    HostTransportFault, KvComponent, TransferObservation,
};

const PAGE_TOKENS: u64 = 16;
const PAGE_COUNT: u32 = 4;
const KEY_TOKEN_BYTES: u64 = 2;
const VALUE_TOKEN_BYTES: u64 = 3;
const LAYERS: [u32; 2] = [0, 1];
const STORAGE_DOMAIN: u64 = 71;
const OBJECT_INDEX: u64 = 91;
const OBJECT_BASE: u64 = 1_000;

struct SessionFixture {
    session: RuntimeSession,
    executor: ExecutorPlan,
    arena: ExecutorArena,
}

fn plans() -> (CompiledKvPlan, ExecutorPlan) {
    let input = AttentionStatePlanInput {
        page_tokens: PAGE_TOKENS,
        states: vec![AttentionStateSpec {
            name: "attention".into(),
            layers: LAYERS.to_vec(),
            storage: AttentionStateStorage::TokenKv {
                key_bytes_per_token_per_layer: KEY_TOKEN_BYTES,
                value_bytes_per_token_per_layer: VALUE_TOKEN_BYTES,
                retention: RetentionKind::Full,
                window_tokens: None,
            },
        }],
    };
    let manifest = compile_runtime_manifest(input.clone()).expect("manifest");
    let manager = compile_plan(
        compile_attention_state_plan(input)
            .expect("attention plan")
            .token_manager_plan()
            .expect("manager input"),
    )
    .expect("manager plan");
    (
        manager,
        ExecutorPlan::compile(&manifest).expect("executor plan"),
    )
}

fn fixture(
    plan: &CompiledKvPlan,
    executor: &ExecutorPlan,
    backend_domain: u16,
    pool_id: u32,
    backend_base_index: u64,
    maximum_requests: u32,
) -> SessionFixture {
    let registration = BackendArenaRegistration {
        pool_id,
        class_id: 0,
        backend_domain,
        page_count: PAGE_COUNT,
        reserved: 0,
        backend_base_index,
    };
    let manager = CanonicalKvManager::new(
        plan,
        ManagerConfig {
            maximum_requests,
            maximum_operations: 8,
            maximum_prefixes: 1,
            maximum_reclamations: PAGE_COUNT,
            maximum_step_tokens: 32,
        },
        &[registration],
    )
    .expect("manager");
    let session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
    let arena = ExecutorArena::bind(session.arena_stats()[0], registration).expect("arena");
    SessionFixture {
        session,
        executor: executor.clone(),
        arena,
    }
}

fn append(
    fixture: &mut SessionFixture,
    request_id: EngineRequestId,
    boundary: u64,
    completion_value: u64,
) {
    fixture
        .session
        .acquire_requests(&[request_id])
        .expect("acquire");
    let plan = fixture
        .session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: boundary,
        }])
        .expect("prepare append");
    let prepared = fixture
        .executor
        .lower_prepared(plan, &[fixture.arena])
        .expect("lower append");
    let evidence = prepared
        .execution_evidence_after_success(&[fixture.arena])
        .expect("execution evidence");
    let ticket = fixture
        .session
        .submit_execution(&evidence)
        .expect("submit append");
    let publication = fixture
        .session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 1,
                completion_value,
                confirmed: true,
            },
        )
        .expect("complete append");
    fixture
        .session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::default(),
        })
        .expect("confirm append");
}

fn tensor_regions(
    backend_domain: u16,
    backend_pages: u64,
    fill: Option<u8>,
) -> Vec<HostTensorRegion> {
    LAYERS
        .into_iter()
        .flat_map(|layer| {
            [
                (KvComponent::Key, KEY_TOKEN_BYTES),
                (KvComponent::Value, VALUE_TOKEN_BYTES),
            ]
            .into_iter()
            .map(move |(component, token_bytes)| {
                let len = usize::try_from(backend_pages * PAGE_TOKENS * token_bytes).unwrap();
                let bytes = match fill {
                    Some(byte) => vec![byte; len],
                    None => (0..len)
                        .map(|offset| {
                            let component_tag = match component {
                                KvComponent::Key => 17,
                                KvComponent::Value => 83,
                            };
                            u8::try_from(
                                (usize::try_from(layer).unwrap() * 41 + component_tag + offset)
                                    % 251,
                            )
                            .unwrap()
                        })
                        .collect(),
                };
                HostTensorRegion::new(backend_domain, layer, component, 0, bytes)
            })
        })
        .collect()
}

fn register(transport: &HostMemoryTransport, regions: &[HostTensorRegion]) {
    for region in regions {
        transport
            .register_region(region.clone())
            .expect("register region");
    }
}

fn region(regions: &[HostTensorRegion], layer: u32, component: KvComponent) -> &HostTensorRegion {
    regions
        .iter()
        .find(|region| region.layer() == layer && region.component() == component)
        .expect("registered tensor")
}

fn expected_object(batch: &ExternalTransferBatch, regions: &[HostTensorRegion]) -> Vec<u8> {
    let mut output = Vec::new();
    for span in &batch.spans {
        let tensor = region(regions, span.layer, span.component).snapshot();
        let begin = usize::try_from(span.source_tensor_offset).unwrap();
        let end = begin + usize::try_from(span.bytes).unwrap();
        output.extend_from_slice(&tensor[begin..end]);
    }
    output
}

fn assert_restored_bytes(
    batch: &ExternalRestoreBatch,
    object: &[u8],
    regions: &[HostTensorRegion],
) {
    for tensor in regions {
        let bytes = tensor.snapshot();
        let mut written = vec![false; bytes.len()];
        for span in batch
            .spans
            .iter()
            .filter(|span| span.layer == tensor.layer() && span.component == tensor.component())
        {
            let destination = usize::try_from(span.destination_tensor_offset).unwrap();
            let count = usize::try_from(span.bytes).unwrap();
            let source = usize::try_from(span.source_offset - OBJECT_BASE).unwrap();
            assert_eq!(
                &bytes[destination..destination + count],
                &object[source..source + count]
            );
            written[destination..destination + count].fill(true);
        }
        assert!(
            bytes
                .iter()
                .zip(written)
                .all(|(&byte, written)| written || byte == 0xEE)
        );
    }
}

async fn export_replica(
    fixture: &mut SessionFixture,
    transport: &dyn ExternalKvTransport,
    plan: &CompiledKvPlan,
    object_index: u64,
) -> (ExternalReplica, ExternalTransferBatch) {
    let request_id = EngineRequestId(1);
    append(fixture, request_id, 17, 1);
    let export = fixture
        .session
        .prepare_external_export(
            request_id,
            ExternalObjectKey {
                namespace: [11; 32],
                digest: [29; 32],
                plan_fingerprint: plan.fingerprint_digest(),
                boundary: 17,
            },
            ExternalReplicaTarget {
                storage_domain: STORAGE_DOMAIN,
                object_index,
                base_offset: OBJECT_BASE,
            },
        )
        .expect("prepare export");
    let batch = fixture
        .executor
        .lower_external_export(&export, &[fixture.arena])
        .expect("lower export");
    let outcome = transport.export(&batch).await.expect("export bytes");
    let receipts = outcome.export_receipts(&export).expect("export receipts");
    let replica = fixture
        .session
        .complete_external_export(outcome.completion(), &receipts)
        .expect("complete export");
    (replica, batch)
}

#[tokio::test]
async fn real_bytes_round_trip_across_sessions_and_delete_exact_object() {
    let (manager_plan, executor) = plans();
    let mut source = fixture(&manager_plan, &executor, 11, 1, 0, 1);
    let source_regions = tensor_regions(11, u64::from(PAGE_COUNT), None);
    let target_regions = tensor_regions(12, 8, Some(0xEE));
    let transport = HostMemoryTransport::new(77);
    register(&transport, &source_regions);
    register(&transport, &target_regions);

    let (replica, export_batch) =
        export_replica(&mut source, &transport, &manager_plan, OBJECT_INDEX).await;
    let object = transport
        .object_bytes(STORAGE_DOMAIN, OBJECT_INDEX)
        .expect("published object");
    assert_eq!(object, expected_object(&export_batch, &source_regions));
    assert_eq!(object.len(), 170);

    let mut target = fixture(&manager_plan, &executor, 12, 2, 4, 1);
    let request_id = EngineRequestId(2);
    target.session.acquire_requests(&[request_id]).unwrap();
    target
        .session
        .admit_external_replica(replica.clone())
        .unwrap();
    let restore = target
        .session
        .prepare_external_restore(request_id, replica.key)
        .expect("prepare restore");
    let restore_batch = target
        .executor
        .lower_external_restore(&restore, &[target.arena])
        .expect("lower restore");
    let outcome = transport
        .restore(&restore_batch)
        .await
        .expect("restore bytes");
    let receipts = outcome
        .restore_receipts(&restore)
        .expect("restore receipts");
    target
        .session
        .submit_external_restore(restore.transfer_id, &receipts)
        .expect("submit restore");
    let publication = target
        .session
        .complete_external_restore(outcome.completion())
        .expect("complete restore");
    target
        .session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::default(),
        })
        .expect("confirm restore");
    assert_restored_bytes(&restore_batch, &object, &target_regions);

    let deletion = transport.delete(&replica).await.expect("delete object");
    target
        .session
        .confirm_external_replica_deletion(deletion)
        .expect("confirm replica deletion");
    assert!(
        transport
            .object_bytes(STORAGE_DOMAIN, OBJECT_INDEX)
            .is_none()
    );
}

#[tokio::test]
async fn injected_failures_map_to_abort_or_quarantine_without_false_publication() {
    let (manager_plan, executor) = plans();
    let mut source = fixture(&manager_plan, &executor, 21, 3, 0, 1);
    let transport = HostMemoryTransport::new(88);
    register(&transport, &tensor_regions(21, u64::from(PAGE_COUNT), None));
    append(&mut source, EngineRequestId(3), 17, 1);
    let key = ExternalObjectKey {
        namespace: [12; 32],
        digest: [30; 32],
        plan_fingerprint: manager_plan.fingerprint_digest(),
        boundary: 17,
    };
    let export = source
        .session
        .prepare_external_export(
            EngineRequestId(3),
            key,
            ExternalReplicaTarget {
                storage_domain: STORAGE_DOMAIN,
                object_index: OBJECT_INDEX + 1,
                base_offset: OBJECT_BASE,
            },
        )
        .unwrap();
    let batch = source
        .executor
        .lower_external_export(&export, &[source.arena])
        .unwrap();
    transport.inject_fault(HostTransportFault {
        operation: ExternalTransferOperation::Export,
        point: HostFaultPoint::BeforeMutation,
    });
    let error = transport.export(&batch).await.unwrap_err();
    assert_eq!(error.observation(), TransferObservation::Unobserved);
    source
        .session
        .abort_external_export(ExternalExportAbortEvidence {
            transfer_id: export.transfer_id,
            backend_unobserved: true,
        })
        .unwrap();
    assert_eq!(source.session.stats().total_reader_pins, 0);
    assert!(
        transport
            .object_bytes(STORAGE_DOMAIN, OBJECT_INDEX + 1)
            .is_none()
    );

    let ambiguous = source
        .session
        .prepare_external_export(
            EngineRequestId(3),
            key,
            ExternalReplicaTarget {
                storage_domain: STORAGE_DOMAIN,
                object_index: OBJECT_INDEX + 2,
                base_offset: OBJECT_BASE,
            },
        )
        .unwrap();
    let ambiguous_batch = source
        .executor
        .lower_external_export(&ambiguous, &[source.arena])
        .unwrap();
    transport.inject_fault(HostTransportFault {
        operation: ExternalTransferOperation::Export,
        point: HostFaultPoint::AfterMutation,
    });
    let error = transport.export(&ambiguous_batch).await.unwrap_err();
    assert_eq!(error.observation(), TransferObservation::Ambiguous);
    assert!(
        transport
            .object_bytes(STORAGE_DOMAIN, OBJECT_INDEX + 2)
            .is_some()
    );
    source
        .session
        .quarantine_external_export(ambiguous.transfer_id)
        .unwrap();
    assert_eq!(source.session.external_tier_stats().quarantined_exports, 1);
    assert!(matches!(
        source.session.prepare_append_batch(&[EngineAppendIntent {
            request_id: EngineRequestId(3),
            target_boundary: 18,
        }]),
        Err(RuntimeSessionError::RequestNotReady { .. })
    ));
}

#[tokio::test]
async fn ambiguous_restore_writes_bytes_but_never_publishes_manager_state() {
    let (manager_plan, executor) = plans();
    let transport = HostMemoryTransport::new(99);
    let mut source = fixture(&manager_plan, &executor, 31, 4, 0, 1);
    register(&transport, &tensor_regions(31, u64::from(PAGE_COUNT), None));
    let (replica, _) =
        export_replica(&mut source, &transport, &manager_plan, OBJECT_INDEX + 3).await;

    let target_regions = tensor_regions(32, 8, Some(0xEE));
    register(&transport, &target_regions);
    let mut target = fixture(&manager_plan, &executor, 32, 5, 4, 1);
    let request_id = EngineRequestId(4);
    target.session.acquire_requests(&[request_id]).unwrap();
    target
        .session
        .admit_external_replica(replica.clone())
        .unwrap();
    let restore = target
        .session
        .prepare_external_restore(request_id, replica.key)
        .unwrap();
    let batch = target
        .executor
        .lower_external_restore(&restore, &[target.arena])
        .unwrap();
    transport.inject_fault(HostTransportFault {
        operation: ExternalTransferOperation::Restore,
        point: HostFaultPoint::AfterMutation,
    });
    let error = transport.restore(&batch).await.unwrap_err();
    assert_eq!(error.observation(), TransferObservation::Ambiguous);
    assert!(
        target_regions
            .iter()
            .any(|region| { region.snapshot().iter().any(|&byte| byte != 0xEE) })
    );
    target
        .session
        .quarantine_external_restore(restore.transfer_id)
        .unwrap();
    assert_eq!(target.session.external_tier_stats().quarantined_restores, 1);
    assert!(matches!(
        target.session.prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 1,
        }]),
        Err(RuntimeSessionError::RequestNotReady { .. })
    ));
}
