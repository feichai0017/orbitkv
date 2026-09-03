use super::*;
use crate::kv_manager::{
    CLASS_LOWERING_EPOCH_START, CLASS_LOWERING_RESETTABLE, DetachedAction, DetachedReason,
};
use crate::{
    IntExpr, Predicate, RetentionProgramInput, RetentionStateDecl, compile_retention_program,
};

const CHUNK_TOKENS: u64 = 32;
const CHUNK_TOKENS_I64: i64 = 32;
const CHUNK_TOKENS_U32: u32 = 32;

fn exact_chunked_plan() -> CompiledKvPlan {
    compile_retention_program(RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: PAGE_TOKENS,
        states: vec![RetentionStateDecl {
            name: "chunked".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: Predicate::Equal {
                lhs: IntExpr::FloorDiv {
                    value: Box::new(IntExpr::QueryPosition),
                    divisor: CHUNK_TOKENS_I64,
                },
                rhs: IntExpr::FloorDiv {
                    value: Box::new(IntExpr::KeyPosition),
                    divisor: CHUNK_TOKENS_I64,
                },
            },
        }],
    })
    .expect("exact Chunked plan")
}

fn chunked_session(
    page_count: u32,
    maximum_requests: u32,
) -> (RuntimeSession, [BackendArenaRegistration; 1]) {
    let backends = [backend(0, 121, page_count, 70_000)];
    let manager = CanonicalKvManager::new(
        &exact_chunked_plan(),
        ManagerConfig {
            maximum_requests,
            maximum_operations: 8,
            maximum_prefixes: 2,
            maximum_reclamations: page_count,
            maximum_step_tokens: CHUNK_TOKENS_U32,
        },
        &backends,
    )
    .expect("Chunked manager");
    (
        RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate),
        backends,
    )
}

#[test]
#[allow(clippy::too_many_lines)]
fn exact_chunked_epoch_end_ack_gates_reset_page_generation_reuse() {
    let (mut session, backends) = chunked_session(2, 2);
    let request = EngineRequestId(51);
    let competing = EngineRequestId(52);
    session
        .acquire_requests(&[request, competing])
        .expect("acquire Chunked requests");

    let (to_31, at_31) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: request,
            target_boundary: 31,
        }],
        17,
        1,
    );
    let lowering = to_31.steps[0].class_lowerings[0];
    assert_eq!(
        lowering.flags,
        CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START
    );
    assert_eq!(
        (
            lowering.previous_layout_boundary,
            lowering.target_layout_boundary,
        ),
        (0, 31)
    );
    assert_eq!(lowering.write_count, 2);
    assert_eq!(at_31.steps[0].boundary, 31);
    assert_eq!(at_31.steps[0].resident_count, 2);
    assert!(at_31.steps[0].detached.is_empty());
    assert!(at_31.retirements.is_empty());
    confirm_publication(&mut session, &at_31);

    let (to_32, at_32) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: request,
            target_boundary: 32,
        }],
        17,
        2,
    );
    let lowering = to_32.steps[0].class_lowerings[0];
    assert_eq!(lowering.flags, CLASS_LOWERING_RESETTABLE);
    assert_eq!(
        (
            lowering.previous_layout_boundary,
            lowering.target_layout_boundary,
        ),
        (31, 32)
    );
    assert_eq!(to_32.steps[0].tail_actions.len(), 1);
    assert_eq!(to_32.steps[0].tail_actions[0].kind, TailActionKind::InPlace);
    assert_eq!(to_32.steps[0].tail_actions[0].valid_token_count, 15);
    assert!(to_32.steps[0].copy_intents.is_empty());
    assert!(to_32.steps[0].write_intents.is_empty());
    assert_eq!(
        (at_32.steps[0].boundary, at_32.steps[0].resident_count),
        (32, 0)
    );
    assert_eq!(at_32.steps[0].detached.len(), 2);
    assert_eq!(at_32.retirements.len(), 2);
    for (ordinal, (detached, retirement)) in at_32.steps[0]
        .detached
        .iter()
        .zip(at_32.retirements.iter())
        .enumerate()
    {
        let ordinal = u64::try_from(ordinal).expect("retirement ordinal fits u64");
        assert_eq!(detached.class_id, 0);
        assert_eq!(detached.logical_ordinal, ordinal);
        assert_eq!(detached.action, DetachedAction::Clear);
        assert_eq!(detached.reason, DetachedReason::Retention);
        assert_eq!(detached.old, retirement.page);
        assert_eq!(detached.replacement, PageLease::default());
        assert_eq!(detached.old_backend_index, retirement.backend_index);
        assert_eq!(detached.replacement_backend_index, 0);
        assert_eq!(
            (detached.token_begin, detached.token_end_exclusive),
            (ordinal * PAGE_TOKENS, (ordinal + 1) * PAGE_TOKENS)
        );
        assert_eq!(
            (retirement.class_id, retirement.logical_ordinal),
            (0, ordinal)
        );
        assert_eq!(
            (retirement.token_begin, retirement.token_end_exclusive),
            (ordinal * PAGE_TOKENS, (ordinal + 1) * PAGE_TOKENS)
        );
        assert_eq!(
            (retirement.completion_domain, retirement.completion_value),
            (17, 2)
        );
    }

    let pending = session.stats();
    let pending_arenas = session.arena_stats();
    assert_eq!((pending.free_pages, pending.retiring_pages), (0, 2));
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: competing,
            target_boundary: 1,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    let mut bad_ack = control_reclamation_evidence(&at_32.retirements);
    bad_ack[1].backend_index += 1;
    assert_eq!(
        session.confirm_publication(&EnginePublicationEvidence {
            publication_id: at_32.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: bad_ack,
        }),
        Err(RuntimeSessionError::ReclamationReceiptMismatch)
    );
    assert_eq!(session.stats(), pending);
    assert_eq!(session.arena_stats(), pending_arenas);
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: competing,
            target_boundary: 1,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    confirm_publication(&mut session, &at_32);

    let to_33 = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: request,
            target_boundary: 33,
        }])
        .expect("prepare first token in next epoch");
    let lowering = to_33.steps[0].class_lowerings[0];
    assert_eq!(
        lowering.flags,
        CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START
    );
    assert_eq!(
        (
            lowering.previous_layout_boundary,
            lowering.target_layout_boundary,
        ),
        (32, 33)
    );
    assert_eq!(to_33.steps[0].tail_actions[0].kind, TailActionKind::None);
    assert!(to_33.steps[0].copy_intents.is_empty());
    assert_eq!(to_33.steps[0].write_intents.len(), 1);
    let reused = to_33.steps[0].write_intents[0];
    let retired = at_32
        .retirements
        .iter()
        .find(|retirement| retirement.page.page_id == reused.page_id)
        .expect("ACKed epoch page reused");
    assert_eq!(reused.page_generation, retired.page.generation + 1);

    let evidence = execution_evidence(&session, &to_33, &backends);
    let ticket = session
        .submit_execution(&evidence)
        .expect("submit 32 to 33");
    let at_33 = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 17,
                completion_value: 3,
                confirmed: true,
            },
        )
        .expect("complete 32 to 33");
    assert_eq!(
        (at_33.steps[0].boundary, at_33.steps[0].resident_count),
        (33, 1)
    );
    assert!(at_33.steps[0].detached.is_empty());
    assert!(at_33.retirements.is_empty());
    confirm_publication(&mut session, &at_33);

    release_ready_request(&mut session, request);
    release_ready_request(&mut session, competing);
    let drained = session.stats();
    assert_eq!(drained.active_requests, 0);
    assert_eq!(drained.active_snapshots, 0);
    assert_eq!(drained.active_pages, 0);
    assert_eq!(drained.retiring_pages, 0);
    assert_eq!(drained.pending_reclamations, 0);
    assert_eq!(drained.free_pages, 2);
}

#[test]
fn chunk_boundary_failure_from_empty_request_is_atomic() {
    let (mut session, _) = chunked_session(8, 1);
    let request_id = EngineRequestId(61);
    session
        .acquire_requests(&[request_id])
        .expect("acquire empty boundary request");
    let before = session.stats();
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 33,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::ChunkBoundaryCrossed {
                previous: 0,
                target: 33,
                chunk_tokens: CHUNK_TOKENS,
            }
        ))
    );
    assert_eq!(session.stats(), before);
}

#[test]
fn chunk_boundary_batch_failure_preserves_every_request() {
    let (mut session, backends) = chunked_session(8, 2);
    let requests = [EngineRequestId(62), EngineRequestId(63)];
    session
        .acquire_requests(&requests)
        .expect("acquire boundary requests");
    for (index, request_id) in requests.iter().copied().enumerate() {
        let (_, publication) = append(
            &mut session,
            &backends,
            &[EngineAppendIntent {
                request_id,
                target_boundary: 31,
            }],
            19,
            u64::try_from(index).expect("completion index fits u64") + 1,
        );
        confirm_publication(&mut session, &publication);
    }

    let before = session.stats();
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: requests[0],
            target_boundary: 33,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::ChunkBoundaryCrossed {
                previous: 31,
                target: 33,
                chunk_tokens: CHUNK_TOKENS,
            }
        ))
    );
    assert_eq!(session.stats(), before);
    assert_eq!(
        session.prepare_append_batch(&[
            EngineAppendIntent {
                request_id: requests[0],
                target_boundary: 32,
            },
            EngineAppendIntent {
                request_id: requests[1],
                target_boundary: 33,
            },
        ]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::ChunkBoundaryCrossed {
                previous: 31,
                target: 33,
                chunk_tokens: CHUNK_TOKENS,
            }
        ))
    );
    assert_eq!(session.stats(), before);

    let valid = session
        .prepare_append_batch(&[
            EngineAppendIntent {
                request_id: requests[0],
                target_boundary: 32,
            },
            EngineAppendIntent {
                request_id: requests[1],
                target_boundary: 32,
            },
        ])
        .expect("both requests remained ready at boundary 31");
    assert!(valid.steps.iter().all(|step| step.previous_boundary == 31
        && step.target_boundary == 32
        && step.class_lowerings[0].flags == CLASS_LOWERING_RESETTABLE));
    session
        .abort_prepared_execution(
            valid.batch_id,
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
        .expect("abort valid boundary probe");
}

#[test]
fn chunked_runtime_session_rejects_every_prefix_entrypoint() {
    let (mut session, _) = chunked_session(4, 2);
    let source = EngineRequestId(71);
    let target = EngineRequestId(72);
    session
        .acquire_requests(&[source, target])
        .expect("acquire Prefix probes");
    let key = prefix_key(0xCE, CHUNK_TOKENS);
    let prefix_id = EnginePrefixId::from_parts(session.arena_stats()[0].engine_epoch, 1);
    let before = session.stats();
    assert_eq!(
        session.lookup_prefix_batch(&[key]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.publish_prefix_batch(&[(source, key)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.prepare_prefix_attach(&[(
            target,
            EnginePrefixLookup {
                key,
                candidate: Some(prefix_id),
                resident_count: 0,
            },
        )]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.prepare_prefix_evict(&[prefix_id]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.publish_prefix_and_release_batch(&[(source, key)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.prepare_request_fork(&[(source, target)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(session.stats(), before);
}
