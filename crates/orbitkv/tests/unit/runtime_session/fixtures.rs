use super::*;

pub(super) const PAGE_TOKENS: u64 = 16;

pub(super) fn full_plan() -> CompiledKvPlan {
    compile_plan(KvPlanInput {
        page_tokens: PAGE_TOKENS,
        classes: vec![KvClassSpec {
            name: "full".into(),
            layers: vec![0],
            retention: RetentionKind::Full,
            bytes_per_token_per_layer: 128,
            window_tokens: None,
            storage: TokenStorageKind::TokenKv,
            components: Vec::new(),
        }],
    })
    .expect("full plan")
}

pub(super) fn hybrid_plan() -> CompiledKvPlan {
    compile_plan(KvPlanInput {
        page_tokens: PAGE_TOKENS,
        classes: vec![
            KvClassSpec {
                name: "full".into(),
                layers: vec![0],
                retention: RetentionKind::Full,
                bytes_per_token_per_layer: 128,
                window_tokens: None,
                storage: TokenStorageKind::TokenKv,
                components: Vec::new(),
            },
            KvClassSpec {
                name: "swa".into(),
                layers: vec![1],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(18),
                storage: TokenStorageKind::TokenKv,
                components: Vec::new(),
            },
        ],
    })
    .expect("hybrid plan")
}

pub(super) fn backend(
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

pub(super) fn session(
    plan: &CompiledKvPlan,
    backends: &[BackendArenaRegistration],
    maximum_requests: u32,
) -> RuntimeSession {
    RuntimeSession::new(
        CanonicalKvManager::new(
            plan,
            ManagerConfig {
                maximum_requests,
                maximum_operations: 8,
                maximum_prefixes: 1,
                maximum_reclamations: backends.iter().map(|item| item.page_count).sum(),
                maximum_step_tokens: 64,
            },
            backends,
        )
        .expect("manager"),
        CacheSharingPolicy::SharedPrefix,
    )
}

pub(super) fn session_with_config(
    plan: &CompiledKvPlan,
    backends: &[BackendArenaRegistration],
    config: ManagerConfig,
) -> RuntimeSession {
    RuntimeSession::new(
        CanonicalKvManager::new(plan, config, backends).expect("manager"),
        CacheSharingPolicy::SharedPrefix,
    )
}

pub(super) fn prefix_key(tag: u8, boundary: u64) -> PrefixSemanticKey {
    PrefixSemanticKey {
        namespace: [0xA5; 32],
        digest: [tag; 32],
        boundary,
    }
}

pub(super) fn execution_evidence(
    session: &RuntimeSession,
    plan: &EngineBatchPlan,
    backends: &[BackendArenaRegistration],
) -> ExecutionEvidence {
    let arenas = session.arena_stats();
    let steps = plan
        .steps
        .iter()
        .map(|step| {
            let mut binds = Vec::new();
            for lowering in &step.class_lowerings {
                let arena = arenas
                    .iter()
                    .find(|arena| arena.class_id == lowering.class_id)
                    .expect("class arena");
                let registration = backends
                    .iter()
                    .find(|backend| backend.class_id == lowering.class_id)
                    .expect("class registration");
                let backend_index = |page_id: u32| {
                    registration.backend_base_index + u64::from(page_id - arena.first_page_id)
                };
                let tail_begin = lowering.tail_offset as usize;
                let tail_end = tail_begin + lowering.tail_count as usize;
                for action in step.tail_actions[tail_begin..tail_end]
                    .iter()
                    .filter(|action| {
                        matches!(
                            action.kind,
                            TailActionKind::CopyOnWrite | TailActionKind::Fresh
                        )
                    })
                {
                    binds.push(EngineBindEvidence {
                        page: action.destination,
                        backend_domain: arena.backend_domain,
                        mapped: true,
                        writable: true,
                        backend_index: backend_index(action.destination.page_id),
                    });
                }
                let write_begin = lowering.write_offset as usize;
                let write_end = write_begin + lowering.write_count as usize;
                for intent in &step.write_intents[write_begin..write_end] {
                    binds.push(EngineBindEvidence {
                        page: PageLease {
                            engine_epoch: arena.engine_epoch,
                            pool_epoch: arena.pool_epoch,
                            generation: intent.page_generation,
                            page_id: intent.page_id,
                            pool_id: arena.pool_id,
                        },
                        backend_domain: arena.backend_domain,
                        mapped: true,
                        writable: true,
                        backend_index: backend_index(intent.page_id),
                    });
                }
            }
            let copies = step
                .copy_intents
                .iter()
                .map(|intent| EngineCopyEvidence {
                    class_id: intent.class_id,
                    backend_domain: intent.backend_domain,
                    token_count: intent.token_count,
                    source_token_offset: intent.source_token_offset,
                    destination_token_offset: intent.destination_token_offset,
                    observed: true,
                    copied: true,
                    ordered_before_writes: true,
                    source: intent.source,
                    destination: intent.destination,
                    source_backend_index: intent.source_backend_index,
                    destination_backend_index: intent.destination_backend_index,
                })
                .collect::<Vec<_>>();
            EngineStepExecutionEvidence {
                request_id: step.request_id,
                bind_receipts: binds.into_boxed_slice(),
                copy_receipts: copies.into_boxed_slice(),
                fixed_states: Box::default(),
            }
        })
        .collect::<Vec<_>>();
    ExecutionEvidence {
        batch_id: plan.batch_id,
        steps: steps.into_boxed_slice(),
    }
}

pub(super) fn control_reclamation_evidence(
    retirements: &[EngineRetirement],
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

pub(super) fn materialization(plan: &EngineControlPlan) -> &EngineMaterializationPlan {
    let EngineControlPlan::Materialization(materialization) = plan else {
        panic!("expected materialization plan");
    };
    materialization
}

pub(super) fn eviction(plan: &EngineControlPlan) -> &EnginePrefixEvictionPlan {
    let EngineControlPlan::PrefixEviction(eviction) = plan else {
        panic!("expected prefix eviction plan");
    };
    eviction
}

pub(super) fn append(
    session: &mut RuntimeSession,
    backends: &[BackendArenaRegistration],
    intents: &[EngineAppendIntent],
    completion_domain: u64,
    completion_value: u64,
) -> (EngineBatchPlan, EngineBatchPublication) {
    let plan = session
        .prepare_append_batch(intents)
        .expect("prepare append");
    let evidence = execution_evidence(session, &plan, backends);
    let ticket = session
        .submit_execution(&evidence)
        .expect("submit execution");
    let publication = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain,
                completion_value,
                confirmed: true,
            },
        )
        .expect("complete execution");
    (plan, publication)
}

pub(super) fn confirm_publication(
    session: &mut RuntimeSession,
    publication: &EngineBatchPublication,
) {
    session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&publication.retirements),
        })
        .expect("confirm publication");
}

pub(super) fn publish_ready_prefix(
    session: &mut RuntimeSession,
    backends: &[BackendArenaRegistration],
    request_id: EngineRequestId,
    tag: u8,
    boundary: u64,
) -> (PrefixSemanticKey, EnginePublishedPrefix) {
    session
        .acquire_requests(&[request_id])
        .expect("acquire prefix source");
    let (_, publication) = append(
        session,
        backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: boundary,
        }],
        u64::from(tag) + 1,
        1,
    );
    confirm_publication(session, &publication);
    let key = prefix_key(tag, boundary);
    let published = session
        .publish_prefix_batch(&[(request_id, key)])
        .expect("publish prefix")[0];
    (key, published)
}

pub(super) fn release_ready_request(session: &mut RuntimeSession, request_id: EngineRequestId) {
    let release = session
        .prepare_release_batch(&[request_id])
        .expect("prepare request release");
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&release.retirements),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
}

pub(super) fn assert_no_manager_capability(value: &serde_json::Value) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                assert_no_manager_capability(item);
            }
        }
        serde_json::Value::Object(fields) => {
            assert!(
                !fields.contains_key("reclamation"),
                "public DTO leaked a reclamation capability: {value}"
            );
            assert!(
                !(fields.contains_key("engine_epoch")
                    && fields.contains_key("slot")
                    && fields.contains_key("generation")),
                "public DTO leaked a manager capability lease: {value}"
            );
            for item in fields.values() {
                assert_no_manager_capability(item);
            }
        }
        _ => {}
    }
}
