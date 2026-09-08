use crate::kv_manager::{BackendArenaRegistration, ManagerConfig, PageLease, PrefixSemanticKey};
use crate::plan::{
    CompiledKvPlan, KvClassSpec, KvPlanInput, RetentionKind, TokenStorageKind, compile_plan,
};

const PAGE_TOKENS: u64 = 16;

fn full_plan() -> CompiledKvPlan {
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

fn hybrid_plan() -> CompiledKvPlan {
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

fn backend(
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

fn session(
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

fn session_with_config(
    plan: &CompiledKvPlan,
    backends: &[BackendArenaRegistration],
    config: ManagerConfig,
) -> RuntimeSession {
    RuntimeSession::new(
        CanonicalKvManager::new(plan, config, backends).expect("manager"),
        CacheSharingPolicy::SharedPrefix,
    )
}

fn prefix_key(tag: u8, boundary: u64) -> PrefixSemanticKey {
    PrefixSemanticKey {
        namespace: [0xA5; 32],
        digest: [tag; 32],
        boundary,
    }
}

fn execution_evidence(
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

fn control_reclamation_evidence(
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

fn materialization(plan: &EngineControlPlan) -> &EngineMaterializationPlan {
    let EngineControlPlan::Materialization(materialization) = plan else {
        panic!("expected materialization plan");
    };
    materialization
}

fn eviction(plan: &EngineControlPlan) -> &EnginePrefixEvictionPlan {
    let EngineControlPlan::PrefixEviction(eviction) = plan else {
        panic!("expected prefix eviction plan");
    };
    eviction
}

fn append(
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

fn confirm_publication(session: &mut RuntimeSession, publication: &EngineBatchPublication) {
    session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&publication.retirements),
        })
        .expect("confirm publication");
}

fn publish_ready_prefix(
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

fn release_ready_request(session: &mut RuntimeSession, request_id: EngineRequestId) {
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

#[test]
#[allow(clippy::too_many_lines)]
fn prefix_publish_lookup_and_attach_b1_b4_are_transactional() {
    for (pool_id, tag, target_count) in [(70, 10, 1_usize), (71, 11, 4)] {
        let backends = [backend(0, pool_id, 16, 20_000)];
        let mut session = session_with_config(
            &full_plan(),
            &backends,
            ManagerConfig {
                maximum_requests: 6,
                maximum_operations: 8,
                maximum_prefixes: 2,
                maximum_reclamations: 16,
                maximum_step_tokens: 64,
            },
        );
        let source = EngineRequestId(1);
        let (key, published) = publish_ready_prefix(&mut session, &backends, source, tag, 32);
        let lookup = session
            .lookup_prefix_batch(&[key])
            .expect("lookup published prefix")[0];
        assert_eq!(lookup.candidate, Some(published.prefix_id));
        assert_eq!(lookup.resident_count, published.resident_count);

        let targets = (0..target_count)
            .map(|offset| EngineRequestId(10 + offset as u64))
            .collect::<Vec<_>>();
        session
            .acquire_requests(&targets)
            .expect("acquire attach targets");
        let items = targets
            .iter()
            .copied()
            .map(|target| (target, lookup))
            .collect::<Vec<_>>();
        let before_prepare = session.stats();
        let first = session
            .prepare_prefix_attach(&items)
            .expect("prepare attach");
        assert_eq!(session.stats(), before_prepare);
        session
            .abort_control(first)
            .expect("abort attach reservation");
        assert_eq!(session.stats(), before_prepare);
        assert_eq!(
            session
                .lookup_prefix_batch(&[key])
                .expect("lookup after abort")[0],
            lookup
        );

        let control_id = session
            .prepare_prefix_attach(&items)
            .expect("prepare attach retry");
        let committed = session.commit_control(control_id).expect("commit attach");
        let after_commit = session.stats();
        assert_eq!(session.commit_control(control_id), Ok(committed.clone()));
        assert_eq!(session.stats(), after_commit);
        assert_eq!(
            session.abort_control(control_id),
            Err(RuntimeSessionError::ControlAlreadyCommitted(control_id))
        );
        let requests = &materialization(&committed).requests;
        assert_eq!(requests.len(), target_count);
        assert!(requests.iter().zip(&targets).all(|(request, target)| {
            request.request_id == *target
                && request.boundary == key.boundary
                && usize::try_from(request.resident_count).ok() == Some(request.pages.len())
        }));
        assert_eq!(
            session.confirm_control(&EngineControlEvidence {
                control_id,
                mirror_updates_confirmed: false,
                reclamation_receipts: Box::new([]),
            }),
            Err(RuntimeSessionError::MirrorUpdatesNotConfirmed)
        );
        assert_eq!(
            session.confirm_control(&EngineControlEvidence {
                control_id,
                mirror_updates_confirmed: true,
                reclamation_receipts: Box::new([]),
            }),
            Ok(EngineControlOutcome::Materialized)
        );
        assert_eq!(
            session.commit_control(control_id),
            Err(RuntimeSessionError::StaleControl(control_id))
        );
        session
            .prepare_append_batch(&[EngineAppendIntent {
                request_id: targets[0],
                target_boundary: 48,
            }])
            .expect("confirmed attach target is ready");
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn request_fork_b1_b4_abort_replay_confirm_and_quarantine() {
    for (pool_id, target_count) in [(80, 1_usize), (81, 4)] {
        let backends = [backend(0, pool_id, 16, 30_000)];
        let mut session = session(&full_plan(), &backends, 6);
        let source = EngineRequestId(100);
        session
            .acquire_requests(&[source])
            .expect("acquire fork source");
        let (_, publication) = append(
            &mut session,
            &backends,
            &[EngineAppendIntent {
                request_id: source,
                target_boundary: 32,
            }],
            31,
            1,
        );
        confirm_publication(&mut session, &publication);
        let targets = (0..target_count)
            .map(|offset| EngineRequestId(110 + offset as u64))
            .collect::<Vec<_>>();
        session
            .acquire_requests(&targets)
            .expect("acquire fork targets");
        let items = targets
            .iter()
            .copied()
            .map(|target| (source, target))
            .collect::<Vec<_>>();
        let baseline = session.stats();
        let aborted = session.prepare_request_fork(&items).expect("prepare fork");
        assert_eq!(session.stats(), baseline);
        session.abort_control(aborted).expect("abort fork");
        assert_eq!(session.stats(), baseline);

        let control_id = session
            .prepare_request_fork(&items)
            .expect("prepare fork retry");
        let committed = session.commit_control(control_id).expect("commit fork");
        let after_commit = session.stats();
        assert_eq!(session.commit_control(control_id), Ok(committed.clone()));
        assert_eq!(session.stats(), after_commit);
        assert_eq!(
            session.abort_control(control_id),
            Err(RuntimeSessionError::ControlAlreadyCommitted(control_id))
        );
        let requests = &materialization(&committed).requests;
        assert_eq!(requests.len(), target_count);
        assert!(requests.iter().zip(&targets).all(|(request, target)| {
            request.request_id == *target
                && request.boundary == 32
                && usize::try_from(request.resident_count).ok() == Some(request.pages.len())
        }));
        if target_count == 1 {
            session
                .quarantine_control(control_id)
                .expect("quarantine fork");
            assert!(matches!(
                session.prepare_append_batch(&[EngineAppendIntent {
                    request_id: targets[0],
                    target_boundary: 48,
                }]),
                Err(RuntimeSessionError::RequestNotReady {
                    state: "quarantined",
                    ..
                })
            ));
            session
                .prepare_append_batch(&[EngineAppendIntent {
                    request_id: source,
                    target_boundary: 48,
                }])
                .expect("fork source unlocked after quarantine");
        } else {
            assert_eq!(
                session.confirm_control(&EngineControlEvidence {
                    control_id,
                    mirror_updates_confirmed: true,
                    reclamation_receipts: Box::new([]),
                }),
                Ok(EngineControlOutcome::Materialized)
            );
            session
                .prepare_append_batch(&[EngineAppendIntent {
                    request_id: targets[0],
                    target_boundary: 48,
                }])
                .expect("fork target ready after confirmation");
        }
    }
}

#[test]
fn prepared_execution_view_uses_the_manager_selected_cow_tail() {
    let backends = [backend(0, 84, 8, 35_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let source = EngineRequestId(220);
    let sibling = EngineRequestId(221);
    session
        .acquire_requests(&[source, sibling])
        .expect("acquire fork pair");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 18,
        }],
        34,
        1,
    );
    confirm_publication(&mut session, &publication);
    let fork_id = session
        .prepare_request_fork(&[(source, sibling)])
        .expect("prepare fork");
    let fork = session.commit_control(fork_id).expect("commit fork");
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id: fork_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineControlOutcome::Materialized)
    );
    assert_eq!(materialization(&fork).requests[0].request_id, sibling);

    let plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: source,
            target_boundary: 19,
        }])
        .expect("prepare shared-tail append");
    let action = plan.steps[0].tail_actions[0];
    assert_eq!(action.kind, TailActionKind::CopyOnWrite);
    let view = session
        .prepared_execution_view(plan.batch_id)
        .expect("materialize COW candidate");
    let tail = view.requests[0].pages.last().expect("candidate tail");
    assert_eq!(tail.page, action.destination);
    assert_ne!(tail.page, action.source);
    assert_eq!(tail.valid_token_count, 3);

    session
        .abort_prepared_execution(
            plan.batch_id,
            &[EngineStepAbortEvidence {
                request_id: source,
                backend_unobserved: true,
            }],
        )
        .expect("abort COW probe");
}

#[test]
fn attach_and_fork_reject_a_late_bad_output_without_partial_session_view_updates() {
    let backends = [backend(0, 82, 16, 32_000)];
    let mut attach_session = session_with_config(
        &full_plan(),
        &backends,
        ManagerConfig {
            maximum_requests: 3,
            maximum_operations: 8,
            maximum_prefixes: 2,
            maximum_reclamations: 16,
            maximum_step_tokens: 64,
        },
    );
    let source = EngineRequestId(200);
    let (key, _) = publish_ready_prefix(&mut attach_session, &backends, source, 20, 16);
    let hint = attach_session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    let targets = [EngineRequestId(201), EngineRequestId(202)];
    attach_session
        .acquire_requests(&targets)
        .expect("attach targets");
    let control_id = attach_session
        .prepare_prefix_attach(&[(targets[0], hint), (targets[1], hint)])
        .expect("prepare B2 attach");
    let old_views = targets
        .iter()
        .map(|target| attach_session.requests[target].view)
        .collect::<Vec<_>>();
    attach_session.inject_test_fault(RuntimeSessionTestFault::AttachSecondOutput);
    let poisoned = RuntimeSessionError::SessionPoisoned("prefix attach result changed");
    assert_eq!(
        attach_session.commit_control(control_id),
        Err(poisoned.clone())
    );
    assert!(
        targets
            .iter()
            .zip(old_views)
            .all(|(target, old)| attach_session.requests[target].view == old)
    );
    assert_eq!(attach_session.abort_control(control_id), Err(poisoned));

    let mut fork_session = session(&full_plan(), &backends, 3);
    let source = EngineRequestId(210);
    fork_session
        .acquire_requests(&[source])
        .expect("fork source");
    let (_, publication) = append(
        &mut fork_session,
        &backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 16,
        }],
        33,
        1,
    );
    confirm_publication(&mut fork_session, &publication);
    let targets = [EngineRequestId(211), EngineRequestId(212)];
    fork_session
        .acquire_requests(&targets)
        .expect("fork targets");
    let control_id = fork_session
        .prepare_request_fork(&[(source, targets[0]), (source, targets[1])])
        .expect("prepare B2 fork");
    let old_views = targets
        .iter()
        .map(|target| fork_session.requests[target].view)
        .collect::<Vec<_>>();
    fork_session.inject_test_fault(RuntimeSessionTestFault::ForkSecondOutput);
    let poisoned = RuntimeSessionError::SessionPoisoned("request fork result changed");
    assert_eq!(
        fork_session.commit_control(control_id),
        Err(poisoned.clone())
    );
    assert!(
        targets
            .iter()
            .zip(old_views)
            .all(|(target, old)| fork_session.requests[target].view == old)
    );
    assert_eq!(fork_session.quarantine_control(control_id), Err(poisoned));
}

#[test]
#[allow(clippy::too_many_lines)]
fn prefix_evict_requires_exact_ack_then_recycles_page_and_identity() {
    let backends = [backend(0, 83, 2, 34_000)];
    let mut session = session_with_config(
        &full_plan(),
        &backends,
        ManagerConfig {
            maximum_requests: 2,
            maximum_operations: 8,
            maximum_prefixes: 1,
            maximum_reclamations: 2,
            maximum_step_tokens: 64,
        },
    );
    let source = EngineRequestId(220);
    let (key, published) = publish_ready_prefix(&mut session, &backends, source, 21, 16);
    release_ready_request(&mut session, source);
    let baseline = session.stats();
    let aborted = session
        .prepare_prefix_evict(&[published.prefix_id])
        .expect("prepare eviction");
    assert_eq!(session.stats(), baseline);
    session.abort_control(aborted).expect("abort eviction");
    assert_eq!(session.stats(), baseline);
    assert_eq!(
        session
            .lookup_prefix_batch(&[key])
            .expect("lookup after abort")[0]
            .candidate,
        Some(published.prefix_id)
    );

    let control_id = session
        .prepare_prefix_evict(&[published.prefix_id])
        .expect("prepare eviction retry");
    let committed = session.commit_control(control_id).expect("commit eviction");
    assert_eq!(session.commit_control(control_id), Ok(committed.clone()));
    assert_eq!(
        session.abort_control(control_id),
        Err(RuntimeSessionError::ControlAlreadyCommitted(control_id))
    );
    let retirement = eviction(&committed).retirements[0];
    let pending = session.stats();
    for evidence in [
        Box::new([]) as Box<[EngineRetirementEvidence]>,
        Box::new([EngineRetirementEvidence {
            page: PageLease {
                page_id: retirement.page.page_id + 1,
                ..retirement.page
            },
            backend_domain: retirement.backend_domain,
            acknowledged: true,
            backend_index: retirement.backend_index,
        }]),
        Box::new([EngineRetirementEvidence {
            page: retirement.page,
            backend_domain: retirement.backend_domain + 1,
            acknowledged: true,
            backend_index: retirement.backend_index,
        }]),
        Box::new([EngineRetirementEvidence {
            page: retirement.page,
            backend_domain: retirement.backend_domain,
            acknowledged: false,
            backend_index: retirement.backend_index,
        }]),
        Box::new([EngineRetirementEvidence {
            page: retirement.page,
            backend_domain: retirement.backend_domain,
            acknowledged: true,
            backend_index: retirement.backend_index + 1,
        }]),
    ] {
        assert_eq!(
            session.confirm_control(&EngineControlEvidence {
                control_id,
                mirror_updates_confirmed: true,
                reclamation_receipts: evidence,
            }),
            Err(RuntimeSessionError::ReclamationReceiptMismatch)
        );
        assert_eq!(session.stats(), pending);
    }
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&eviction(&committed).retirements),
        }),
        Ok(EngineControlOutcome::Evicted)
    );
    assert_eq!(session.stats().pending_reclamations, 0);
    assert_eq!(
        session.lookup_prefix_batch(&[key]).expect("evicted lookup")[0].candidate,
        None
    );
    assert_eq!(
        session.prepare_prefix_evict(&[published.prefix_id]),
        Err(RuntimeSessionError::StalePrefix(published.prefix_id))
    );

    let replacement = EngineRequestId(221);
    session
        .acquire_requests(&[replacement])
        .expect("replacement request");
    let plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: replacement,
            target_boundary: 16,
        }])
        .expect("reuse evicted page");
    let reused = plan.steps[0].write_intents[0];
    assert_eq!(reused.page_id, retirement.page.page_id);
    assert_eq!(reused.page_generation, retirement.page.generation + 1);
}

#[test]
fn prefix_recycle_failure_after_ack_is_sticky_poison() {
    let backends = [backend(0, 84, 1, 35_000)];
    let mut session = session(&full_plan(), &backends, 1);
    let source = EngineRequestId(230);
    let (_, published) = publish_ready_prefix(&mut session, &backends, source, 22, 16);
    release_ready_request(&mut session, source);
    let control_id = session
        .prepare_prefix_evict(&[published.prefix_id])
        .expect("prepare");
    let plan = session.commit_control(control_id).expect("commit");
    session.inject_test_fault(RuntimeSessionTestFault::PrefixRecycleFatalOnce);
    let poisoned = RuntimeSessionError::SessionPoisoned(
        "unexpected prefix recycle failure after acknowledgement",
    );
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&eviction(&plan).retirements),
        }),
        Err(poisoned.clone())
    );
    assert_eq!(session.stats().pending_reclamations, 0);
    assert_eq!(session.commit_control(control_id), Err(poisoned));
}

#[test]
fn operation_sequence_overflow_is_sticky_and_never_issues_maximum() {
    for kind in ["batch", "publication", "release", "prefix", "control"] {
        let mut next = u64::MAX - 1;
        assert_eq!(allocate_sequence(&mut next, kind), Ok(u64::MAX - 1));
        assert_eq!(next, u64::MAX);
        assert_eq!(
            allocate_sequence(&mut next, kind),
            Err(RuntimeSessionError::IdentityExhausted(kind))
        );
        assert_eq!(next, u64::MAX);
        assert_eq!(
            allocate_sequence(&mut next, kind),
            Err(RuntimeSessionError::IdentityExhausted(kind))
        );
        assert!(was_issued(u64::MAX - 1, next));
        assert!(!was_issued(u64::MAX, next));
    }
}

#[test]
fn operation_sequence_exhaustion_fails_closed_through_session_entrypoints() {
    let backends = [backend(0, 18, 3, 700)];

    let mut batch_session = session(&full_plan(), &backends, 1);
    let batch_request = EngineRequestId(1);
    batch_session
        .acquire_requests(&[batch_request])
        .expect("acquire batch request");
    batch_session.next_batch_sequence = u64::MAX;
    let batch_intents = [EngineAppendIntent {
        request_id: batch_request,
        target_boundary: 16,
    }];
    for _ in 0..2 {
        assert_eq!(
            batch_session.prepare_append_batch(&batch_intents),
            Err(RuntimeSessionError::IdentityExhausted("batch"))
        );
    }
    assert_eq!(batch_session.stats().prepared_steps, 0);

    let mut publication_session = session(&full_plan(), &backends, 1);
    let publication_request = EngineRequestId(2);
    publication_session
        .acquire_requests(&[publication_request])
        .expect("acquire publication request");
    let plan = publication_session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: publication_request,
            target_boundary: 16,
        }])
        .expect("prepare publication request");
    let execution = execution_evidence(&publication_session, &plan, &backends);
    let ticket = publication_session
        .submit_execution(&execution)
        .expect("submit publication request");
    publication_session.next_publication_sequence = u64::MAX;
    let completion = EngineCompletionEvidence {
        completion_domain: 3,
        completion_value: 1,
        confirmed: true,
    };
    for _ in 0..2 {
        assert_eq!(
            publication_session.complete_execution_by_batch(ticket.batch_id(), completion),
            Err(RuntimeSessionError::IdentityExhausted("publication"))
        );
    }
    assert_eq!(publication_session.stats().submitted_steps, 1);

    let mut release_session = session(&full_plan(), &backends, 1);
    let release_request = EngineRequestId(3);
    release_session
        .acquire_requests(&[release_request])
        .expect("acquire release request");
    let (_, publication) = append(
        &mut release_session,
        &backends,
        &[EngineAppendIntent {
            request_id: release_request,
            target_boundary: 16,
        }],
        3,
        1,
    );
    confirm_publication(&mut release_session, &publication);
    release_session.next_release_sequence = u64::MAX;
    for _ in 0..2 {
        assert_eq!(
            release_session.prepare_release_batch(&[release_request]),
            Err(RuntimeSessionError::IdentityExhausted("release"))
        );
    }
    assert_eq!(release_session.stats().active_requests, 1);
}

#[test]
#[allow(clippy::too_many_lines)]
fn operation_high_water_preserves_out_of_order_pending_ids() {
    let backends = [backend(0, 19, 2, 800)];
    let mut session = session(&full_plan(), &backends, 2);
    let requests = [EngineRequestId(5), EngineRequestId(6)];
    session.acquire_requests(&requests).expect("acquire");

    let first_plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: requests[0],
            target_boundary: 16,
        }])
        .expect("prepare first");
    let second_plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: requests[1],
            target_boundary: 16,
        }])
        .expect("prepare second");
    assert_eq!(first_plan.batch_id.sequence(), 1);
    assert_eq!(second_plan.batch_id.sequence(), 2);

    let epoch = first_plan.batch_id.session_epoch();
    let zero_batch = EngineBatchId::from_parts(epoch, 0);
    let future_batch = EngineBatchId::from_parts(epoch, u64::MAX);
    assert_eq!(
        session.quarantine_prepared_execution(zero_batch),
        Err(RuntimeSessionError::UnknownBatch(zero_batch))
    );
    assert_eq!(
        session.quarantine_prepared_execution(future_batch),
        Err(RuntimeSessionError::UnknownBatch(future_batch))
    );

    let first_execution = execution_evidence(&session, &first_plan, &backends);
    let first_ticket = session
        .submit_execution(&first_execution)
        .expect("submit first");
    let second_execution = execution_evidence(&session, &second_plan, &backends);
    let second_ticket = session
        .submit_execution(&second_execution)
        .expect("submit second");

    let second_publication = session
        .complete_execution_by_batch(
            second_ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 4,
                completion_value: 1,
                confirmed: true,
            },
        )
        .expect("complete newer batch first");
    assert_eq!(
        session.complete_execution_by_batch(
            second_plan.batch_id,
            EngineCompletionEvidence {
                completion_domain: 4,
                completion_value: 2,
                confirmed: true,
            },
        ),
        Err(RuntimeSessionError::StaleBatch(second_plan.batch_id))
    );
    let first_publication = session
        .complete_execution_by_batch(
            first_ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 4,
                completion_value: 2,
                confirmed: true,
            },
        )
        .expect("older batch remains pending");

    assert_eq!(second_publication.publication_id.sequence(), 1);
    assert_eq!(first_publication.publication_id.sequence(), 2);
    let zero_publication = EnginePublicationId::from_parts(epoch, 0);
    let future_publication = EnginePublicationId::from_parts(epoch, u64::MAX);
    for publication_id in [zero_publication, future_publication] {
        assert_eq!(
            session.confirm_publication(&EnginePublicationEvidence {
                publication_id,
                mirror_cleanup_confirmed: true,
                reclamation_receipts: Box::new([]),
            }),
            Err(RuntimeSessionError::UnknownPublication(publication_id))
        );
    }

    confirm_publication(&mut session, &first_publication);
    assert_eq!(
        session.confirm_publication(&EnginePublicationEvidence {
            publication_id: first_publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::StalePublication(
            first_publication.publication_id
        ))
    );
    confirm_publication(&mut session, &second_publication);

    let first_release = session
        .prepare_release_batch(&[requests[0]])
        .expect("prepare first release");
    let second_release = session
        .prepare_release_batch(&[requests[1]])
        .expect("prepare second release");
    assert_eq!(first_release.release_id.sequence(), 1);
    assert_eq!(second_release.release_id.sequence(), 2);
    let zero_release = EngineReleaseId::from_parts(epoch, 0);
    let future_release = EngineReleaseId::from_parts(epoch, u64::MAX);
    for release_id in [zero_release, future_release] {
        assert_eq!(
            session.confirm_release(&EngineReleaseEvidence {
                release_id,
                mirror_cleanup_confirmed: true,
                reclamation_receipts: Box::new([]),
            }),
            Err(RuntimeSessionError::UnknownRelease(release_id))
        );
    }

    let second_evidence = EngineReleaseEvidence {
        release_id: second_release.release_id,
        mirror_cleanup_confirmed: true,
        reclamation_receipts: control_reclamation_evidence(&second_release.retirements),
    };
    assert_eq!(
        session.confirm_release(&second_evidence),
        Ok(EngineReleaseOutcome::Completed)
    );
    assert_eq!(
        session.confirm_release(&second_evidence),
        Err(RuntimeSessionError::StaleRelease(second_release.release_id))
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: first_release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&first_release.retirements),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );

    assert!(session.batches.is_empty());
    assert!(session.publications.is_empty());
    assert!(session.releases.is_empty());
}

#[test]
fn batch_id_completion_uses_session_owned_submission() {
    let backends = [backend(0, 20, 1, 900)];
    let mut session = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(7);
    session.acquire_requests(&[request_id]).expect("acquire");
    let plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }])
        .expect("prepare");
    let execution = execution_evidence(&session, &plan, &backends);
    session.submit_execution(&execution).expect("submit");

    let completion = EngineCompletionEvidence {
        completion_domain: 5,
        completion_value: 1,
        confirmed: true,
    };
    let publication = session
        .complete_execution_by_batch(plan.batch_id, completion)
        .expect("complete by batch id");
    assert_eq!(publication.batch_id, plan.batch_id);
    assert_eq!(publication.steps[0].request_id, request_id);
    assert_eq!(publication.steps[0].boundary, 16);
    assert_eq!(
        session.complete_execution_by_batch(plan.batch_id, completion),
        Err(RuntimeSessionError::StaleBatch(plan.batch_id))
    );
    confirm_publication(&mut session, &publication);
}

#[test]
fn full_release_ack_gates_physical_reuse_and_recycles_engine_id() {
    let backends = [backend(0, 21, 2, 1_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let first = EngineRequestId(101);
    let second = EngineRequestId(202);
    let acquired = session.acquire_requests(&[first]).expect("acquire first");
    assert_eq!(acquired[0].boundary, 0);

    let (first_plan, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: first,
            target_boundary: 32,
        }],
        7,
        1,
    );
    assert!(publication.retirements.is_empty());
    assert!(matches!(
        session.prepare_release_batch(&[first]),
        Err(RuntimeSessionError::RequestNotReady {
            request_id,
            state: "publication confirmation pending",
        }) if request_id == first
    ));
    confirm_publication(&mut session, &publication);

    let release = session
        .prepare_release_batch(&[first])
        .expect("prepare release");
    assert_eq!(release.retirements.len(), 2);
    session.acquire_requests(&[second]).expect("acquire second");
    let before = session.stats();
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: second,
            target_boundary: 32,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    assert_eq!(session.stats(), before);

    let exact = control_reclamation_evidence(&release.retirements);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: false,
            reclamation_receipts: exact.clone(),
        }),
        Err(RuntimeSessionError::MirrorCleanupNotConfirmed)
    );
    assert_eq!(session.stats(), before);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: exact,
        }),
        Ok(EngineReleaseOutcome::Completed)
    );

    let reused = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: second,
            target_boundary: 32,
        }])
        .expect("reuse pages after ACK");
    let mut old_pages = first_plan.steps[0]
        .write_intents
        .iter()
        .map(|intent| (intent.page_id, intent.page_generation))
        .collect::<Vec<_>>();
    let mut new_pages = reused.steps[0]
        .write_intents
        .iter()
        .map(|intent| (intent.page_id, intent.page_generation))
        .collect::<Vec<_>>();
    old_pages.sort_unstable();
    new_pages.sort_unstable();
    assert_eq!(
        new_pages,
        old_pages
            .iter()
            .map(|&(page_id, generation)| (page_id, generation + 1))
            .collect::<Vec<_>>()
    );
    session
        .acquire_requests(&[first])
        .expect("engine id recycled");
}

#[test]
#[allow(clippy::too_many_lines)]
fn wrong_batch_and_grouped_cardinality_are_retryable_and_atomic() {
    let backends = [backend(0, 31, 4, 3_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let requests = [EngineRequestId(1), EngineRequestId(2)];
    session.acquire_requests(&requests).expect("acquire");
    let plan = session
        .prepare_append_batch(&[
            EngineAppendIntent {
                request_id: requests[0],
                target_boundary: 16,
            },
            EngineAppendIntent {
                request_id: requests[1],
                target_boundary: 16,
            },
        ])
        .expect("prepare");
    let exact = execution_evidence(&session, &plan, &backends);
    let prepared_stats = session.stats();

    let unknown_request = EngineRequestId(99);
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: unknown_request,
            target_boundary: 16,
        }]),
        Err(RuntimeSessionError::UnknownRequest(unknown_request))
    );
    assert_eq!(session.stats(), prepared_stats);

    let mut wrong_batch = exact.clone();
    let unknown_batch = EngineBatchId {
        session_epoch: plan.batch_id.session_epoch(),
        sequence: u64::MAX,
    };
    wrong_batch.batch_id = unknown_batch;
    assert_eq!(
        session.submit_execution(&wrong_batch),
        Err(RuntimeSessionError::UnknownBatch(unknown_batch))
    );
    assert_eq!(session.stats(), prepared_stats);

    let mut short = exact.clone();
    short.steps = short.steps[..1].to_vec().into_boxed_slice();
    assert!(matches!(
        session.submit_execution(&short),
        Err(RuntimeSessionError::EvidenceCardinality {
            field: "steps",
            expected: 2,
            actual: 1,
        })
    ));
    assert_eq!(session.stats(), prepared_stats);

    let ticket = session.submit_execution(&exact).expect("retry submit");
    let submitted_stats = session.stats();
    assert_eq!(ticket.batch_id(), plan.batch_id);
    assert_eq!(
        session.complete_execution_by_batch(
            plan.batch_id,
            EngineCompletionEvidence {
                completion_domain: 9,
                completion_value: 1,
                confirmed: false,
            },
        ),
        Err(RuntimeSessionError::Manager(
            KvManagerError::CompletionNotConfirmed
        ))
    );
    assert_eq!(session.stats(), submitted_stats);
    let publication = session
        .complete_execution_by_batch(
            plan.batch_id,
            EngineCompletionEvidence {
                completion_domain: 9,
                completion_value: 1,
                confirmed: true,
            },
        )
        .expect("retry completion");
    assert_eq!(publication.steps.len(), 2);
    assert_eq!(publication.steps[0].boundary, 16);
    assert_eq!(
        session.complete_execution_by_batch(
            plan.batch_id,
            EngineCompletionEvidence {
                completion_domain: 9,
                completion_value: 2,
                confirmed: true,
            },
        ),
        Err(RuntimeSessionError::StaleBatch(plan.batch_id))
    );
    confirm_publication(&mut session, &publication);
}

#[test]
fn engine_plan_and_evidence_hide_manager_transaction_leases() {
    let backends = [backend(0, 32, 2, 3_500)];
    let mut session = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(3);
    session.acquire_requests(&[request_id]).expect("acquire");
    let plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }])
        .expect("prepare");
    let evidence = execution_evidence(&session, &plan, &backends);

    let plan_wire = serde_json::to_string(&plan).expect("serialize plan");
    assert!(!plan_wire.contains("\"step\":"));
    assert!(!plan_wire.contains("snapshot"));
    let evidence_wire = serde_json::to_string(&evidence).expect("serialize evidence");
    assert!(!evidence_wire.contains("\"step\":"));
    assert!(!evidence_wire.contains("snapshot"));

    session
        .submit_execution(&evidence)
        .expect("private canonical step is injected");
}

#[test]
#[allow(clippy::too_many_lines)]
fn hybrid_publication_ack_is_separate_and_exact_before_reuse() {
    let backends = [backend(0, 41, 6, 4_000), backend(1, 42, 3, 5_000)];
    let mut session = session(&hybrid_plan(), &backends, 2);
    let first = EngineRequestId(11);
    let second = EngineRequestId(22);
    session
        .acquire_requests(&[first, second])
        .expect("acquire hybrid requests");

    let (_, initial) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: first,
            target_boundary: 18,
        }],
        17,
        1,
    );
    assert!(initial.retirements.is_empty());
    confirm_publication(&mut session, &initial);

    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: first,
            target_boundary: 35,
        }],
        17,
        2,
    );
    assert_eq!(publication.retirements.len(), 1);
    assert_eq!(publication.retirements[0].class_id, 1);
    let pending_stats = session.stats();
    assert_eq!(pending_stats.retiring_pages, 1);

    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: second,
            target_boundary: 16,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    assert_eq!(session.stats(), pending_stats);
    assert!(matches!(
        session.prepare_release_batch(&[first]),
        Err(RuntimeSessionError::RequestNotReady {
            request_id,
            state: "publication confirmation pending",
        }) if request_id == first
    ));

    assert_eq!(
        session.confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::ReclamationReceiptMismatch)
    );
    assert_eq!(session.stats(), pending_stats);
    let exact = control_reclamation_evidence(&publication.retirements);
    let mut forged = exact.clone();
    forged[0].backend_index += 1;
    assert_eq!(
        session.confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: forged,
        }),
        Err(RuntimeSessionError::ReclamationReceiptMismatch)
    );
    assert_eq!(session.stats(), pending_stats);
    session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: exact,
        })
        .expect("confirm hybrid publication");

    let reused = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: second,
            target_boundary: 16,
        }])
        .expect("reuse SWA retirement after publication ACK");
    let swa = reused.steps[0]
        .class_lowerings
        .iter()
        .find(|lowering| lowering.class_id == 1)
        .expect("SWA lowering");
    let reused_swa = reused.steps[0].write_intents[swa.write_offset as usize];
    assert_eq!(reused_swa.page_id, publication.retirements[0].page.page_id);
    assert_eq!(
        reused_swa.page_generation,
        publication.retirements[0].page.generation + 1
    );

    let release = session
        .prepare_release_batch(&[first])
        .expect("release hybrid request");
    assert_eq!(release.releases.len(), 1);
    assert_eq!(release.retirements.len(), 5);
    let release_stats = session.stats();
    let exact = control_reclamation_evidence(&release.retirements);
    let mut forged = exact.clone();
    forged[0].acknowledged = false;
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: forged,
        }),
        Err(RuntimeSessionError::ReclamationReceiptMismatch)
    );
    assert_eq!(session.stats(), release_stats);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: exact,
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::StaleRelease(release.release_id))
    );
}

#[test]
fn prepared_abort_releases_private_snapshot_and_reuses_pages() {
    let backends = [backend(0, 51, 2, 6_000)];
    let mut session = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(301);
    session.acquire_requests(&[request_id]).expect("acquire");
    let baseline = session.stats();
    assert_eq!(baseline.active_snapshots, 1);

    let first = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 32,
        }])
        .expect("first prepare");
    let first_pages = first.steps[0]
        .write_intents
        .iter()
        .map(|intent| (intent.page_id, intent.page_generation))
        .collect::<BTreeSet<_>>();
    assert_eq!(session.stats().active_snapshots, 2);
    session
        .abort_prepared_execution(
            first.batch_id,
            &[EngineStepAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        )
        .expect("abort prepared batch");
    assert_eq!(session.stats(), baseline);
    assert_eq!(
        session.abort_prepared_execution(
            first.batch_id,
            &[EngineStepAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        ),
        Err(RuntimeSessionError::StaleBatch(first.batch_id))
    );

    let retry = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 32,
        }])
        .expect("prepare after abort");
    let retry_pages = retry.steps[0]
        .write_intents
        .iter()
        .map(|intent| (intent.page_id, intent.page_generation))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        retry_pages,
        first_pages
            .iter()
            .map(|&(page_id, generation)| (page_id, generation + 1))
            .collect()
    );
}

#[test]
fn invalid_abort_is_batch_atomic_and_retryable() {
    let backends = [backend(0, 52, 2, 7_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let requests = [EngineRequestId(401), EngineRequestId(402)];
    session.acquire_requests(&requests).expect("acquire");
    let plan = session
        .prepare_append_batch(&[
            EngineAppendIntent {
                request_id: requests[0],
                target_boundary: 16,
            },
            EngineAppendIntent {
                request_id: requests[1],
                target_boundary: 16,
            },
        ])
        .expect("prepare");
    let prepared = session.stats();

    assert_eq!(
        session.abort_prepared_execution(
            plan.batch_id,
            &[
                EngineStepAbortEvidence {
                    request_id: requests[0],
                    backend_unobserved: true,
                },
                EngineStepAbortEvidence {
                    request_id: requests[1],
                    backend_unobserved: false,
                },
            ],
        ),
        Err(RuntimeSessionError::Manager(
            KvManagerError::BackendObservationUnknown
        ))
    );
    assert_eq!(session.stats(), prepared);
    assert!(matches!(
        session.prepare_release_batch(&[requests[0]]),
        Err(RuntimeSessionError::RequestNotReady {
            request_id,
            state: "append prepared",
        }) if request_id == requests[0]
    ));

    assert!(matches!(
        session.abort_prepared_execution(
            plan.batch_id,
            &[
                EngineStepAbortEvidence {
                    request_id: requests[1],
                    backend_unobserved: true,
                },
                EngineStepAbortEvidence {
                    request_id: requests[0],
                    backend_unobserved: true,
                },
            ],
        ),
        Err(RuntimeSessionError::EvidenceRequest { index: 0, .. })
    ));
    assert_eq!(session.stats(), prepared);

    session
        .abort_prepared_execution(
            plan.batch_id,
            &[
                EngineStepAbortEvidence {
                    request_id: requests[0],
                    backend_unobserved: true,
                },
                EngineStepAbortEvidence {
                    request_id: requests[1],
                    backend_unobserved: true,
                },
            ],
        )
        .expect("retry exact abort");
    assert_eq!(session.stats().prepared_steps, 0);
    session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: requests[0],
            target_boundary: 16,
        }])
        .expect("request ready after abort");
}

#[test]
fn explicit_prepared_and_submitted_quarantine_are_terminal() {
    let backends = [backend(0, 53, 6, 8_000)];
    let mut session = session(&full_plan(), &backends, 3);
    let prepared_request = EngineRequestId(501);
    let submitted_request = EngineRequestId(502);
    let survivor = EngineRequestId(503);
    session
        .acquire_requests(&[prepared_request, submitted_request, survivor])
        .expect("acquire");

    let prepared = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: prepared_request,
            target_boundary: 18,
        }])
        .expect("prepared batch");
    let prepared_pages = prepared.steps[0]
        .write_intents
        .iter()
        .map(|intent| intent.page_id)
        .collect::<BTreeSet<_>>();
    session
        .quarantine_prepared_execution(prepared.batch_id)
        .expect("quarantine prepared");
    assert_eq!(session.stats().prepared_steps, 0);
    assert_eq!(session.stats().quarantined_pages, 2);
    assert!(matches!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: prepared_request,
            target_boundary: 18,
        }]),
        Err(RuntimeSessionError::RequestNotReady {
            request_id,
            state: "quarantined",
        }) if request_id == prepared_request
    ));

    let submitted = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: submitted_request,
            target_boundary: 18,
        }])
        .expect("prepare submitted batch");
    let submitted_pages = submitted.steps[0]
        .write_intents
        .iter()
        .map(|intent| intent.page_id)
        .collect::<BTreeSet<_>>();
    let evidence = execution_evidence(&session, &submitted, &backends);
    session.submit_execution(&evidence).expect("submit");
    session
        .quarantine_submitted_execution(submitted.batch_id)
        .expect("quarantine submitted");
    assert_eq!(session.stats().submitted_steps, 0);
    assert_eq!(session.stats().quarantined_pages, 4);
    assert_eq!(
        session.quarantine_submitted_execution(submitted.batch_id),
        Err(RuntimeSessionError::StaleBatch(submitted.batch_id))
    );

    let survivor_plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: survivor,
            target_boundary: 18,
        }])
        .expect("remaining pages stay usable");
    assert!(
        survivor_plan.steps[0]
            .write_intents
            .iter()
            .all(|intent| !prepared_pages.contains(&intent.page_id)
                && !submitted_pages.contains(&intent.page_id))
    );
}

#[test]
fn foreign_session_objects_cannot_address_local_operations() {
    let backends = [backend(0, 54, 4, 9_000)];
    let mut first = session(&full_plan(), &backends, 1);
    let mut second = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(601);
    first
        .acquire_requests(&[request_id])
        .expect("first acquire");
    second
        .acquire_requests(&[request_id])
        .expect("second acquire");
    let first_plan = first
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }])
        .expect("first prepare");
    let second_plan = second
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }])
        .expect("second prepare");
    assert_eq!(
        first_plan.batch_id.sequence(),
        second_plan.batch_id.sequence()
    );
    assert_ne!(
        first_plan.batch_id.session_epoch(),
        second_plan.batch_id.session_epoch()
    );
    let first_prepared = first.stats();

    assert_eq!(
        first.abort_prepared_execution(
            second_plan.batch_id,
            &[EngineStepAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        ),
        Err(RuntimeSessionError::ForeignBatch(second_plan.batch_id))
    );
    assert_eq!(first.stats(), first_prepared);

    let mut foreign_evidence = execution_evidence(&second, &second_plan, &backends);
    foreign_evidence.batch_id = first_plan.batch_id;
    assert_eq!(
        second.submit_execution(&foreign_evidence),
        Err(RuntimeSessionError::ForeignBatch(first_plan.batch_id))
    );
    let second_evidence = execution_evidence(&second, &second_plan, &backends);
    let second_ticket = second
        .submit_execution(&second_evidence)
        .expect("second submit");
    assert_eq!(
        first.complete_execution_by_batch(
            second_ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 61,
                completion_value: 1,
                confirmed: true,
            },
        ),
        Err(RuntimeSessionError::ForeignBatch(second_plan.batch_id))
    );

    first
        .abort_prepared_execution(
            first_plan.batch_id,
            &[EngineStepAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        )
        .expect("local operation remains recoverable");
}

#[test]
fn foreign_publication_and_release_ids_are_rejected_before_lookup() {
    let backends = [backend(0, 55, 4, 10_000)];
    let mut first = session(&full_plan(), &backends, 1);
    let mut second = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(701);
    first
        .acquire_requests(&[request_id])
        .expect("first acquire");
    second
        .acquire_requests(&[request_id])
        .expect("second acquire");
    let (_, first_publication) = append(
        &mut first,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }],
        71,
        1,
    );
    let (_, second_publication) = append(
        &mut second,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }],
        72,
        1,
    );
    assert_eq!(
        first.confirm_publication(&EnginePublicationEvidence {
            publication_id: second_publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::ForeignPublication(
            second_publication.publication_id
        ))
    );
    confirm_publication(&mut first, &first_publication);
    confirm_publication(&mut second, &second_publication);

    let first_release = first
        .prepare_release_batch(&[request_id])
        .expect("first release");
    let second_release = second
        .prepare_release_batch(&[request_id])
        .expect("second release");
    assert_eq!(
        first.confirm_release(&EngineReleaseEvidence {
            release_id: second_release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&second_release.retirements),
        }),
        Err(RuntimeSessionError::ForeignRelease(
            second_release.release_id
        ))
    );
    assert_eq!(
        first.confirm_release(&EngineReleaseEvidence {
            release_id: first_release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&first_release.retirements),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
}

#[test]
fn impossible_post_manager_output_poisons_every_mutating_path() {
    let backends = [backend(0, 56, 2, 11_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let request_id = EngineRequestId(801);
    session.acquire_requests(&[request_id]).expect("acquire");
    let plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }])
        .expect("prepare");
    let evidence = execution_evidence(&session, &plan, &backends);
    let ticket = session.submit_execution(&evidence).expect("submit");
    session.inject_test_fault(RuntimeSessionTestFault::CompletionCardinality);
    let poisoned = RuntimeSessionError::SessionPoisoned("completion result cardinality");
    assert_eq!(
        session.complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 81,
                completion_value: 1,
                confirmed: true,
            },
        ),
        Err(poisoned.clone())
    );
    assert_eq!(
        session.acquire_requests(&[EngineRequestId(802)]),
        Err(poisoned.clone())
    );
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 32,
        }]),
        Err(poisoned.clone())
    );
    assert_eq!(
        session.quarantine_submitted_execution(ticket.batch_id()),
        Err(poisoned)
    );
    assert_eq!(session.stats().active_snapshots, 1);
}

#[test]
fn acknowledged_release_retries_by_id_without_replaying_receipts() {
    let backends = [backend(0, 57, 1, 12_000)];
    let mut session = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(901);
    session.acquire_requests(&[request_id]).expect("acquire");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }],
        91,
        1,
    );
    confirm_publication(&mut session, &publication);
    let release = session
        .prepare_release_batch(&[request_id])
        .expect("release");
    session.inject_test_fault(RuntimeSessionTestFault::ReleaseRecycleOnce);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&release.retirements),
        }),
        Ok(EngineReleaseOutcome::RecyclePending)
    );
    assert_eq!(session.stats().pending_reclamations, 0);
    assert!(matches!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 32,
        }]),
        Err(RuntimeSessionError::RequestNotReady {
            request_id: pending,
            state: "release confirmation pending",
        }) if pending == request_id
    ));
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&release.retirements),
        }),
        Err(RuntimeSessionError::ReleaseRetryNotIdOnly)
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::ReleaseRetryNotIdOnly)
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: false,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
    assert_eq!(session.stats().active_requests, 0);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: false,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::StaleRelease(release.release_id))
    );
}

#[test]
fn unexpected_post_ack_recycle_failure_poison_is_sticky() {
    let backends = [backend(0, 58, 1, 13_000)];
    let mut session = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(902);
    session.acquire_requests(&[request_id]).expect("acquire");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }],
        92,
        1,
    );
    confirm_publication(&mut session, &publication);
    let release = session
        .prepare_release_batch(&[request_id])
        .expect("release");
    session.inject_test_fault(RuntimeSessionTestFault::ReleaseRecycleFatalOnce);
    let poisoned = RuntimeSessionError::SessionPoisoned(
        "unexpected release recycle failure after acknowledgement",
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&release.retirements),
        }),
        Err(poisoned.clone())
    );
    assert_eq!(session.stats().pending_reclamations, 0);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: false,
            reclamation_receipts: Box::new([]),
        }),
        Err(poisoned.clone())
    );
    assert_eq!(
        session.acquire_requests(&[EngineRequestId(903)]),
        Err(poisoned)
    );
}
