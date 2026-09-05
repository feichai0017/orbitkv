use super::*;

use crate::kv_manager::PhysicalResidencePolicy;
use crate::plan::compile_retention_program;

fn residence_session(
    policy: PhysicalResidencePolicy,
    page_count: u32,
    maximum_requests: u32,
) -> (RuntimeSession, [BackendArenaRegistration; 1]) {
    let backends = [backend(0, 97, page_count, 8_000)];
    let plan = compile_plan(KvPlanInput {
        page_tokens: PAGE_TOKENS,
        classes: vec![KvClassSpec {
            name: "sliding".into(),
            layers: vec![0],
            retention: RetentionKind::Sliding,
            bytes_per_token_per_layer: 128,
            window_tokens: Some(18),
            storage: TokenStorageKind::TokenKv,
            components: Vec::new(),
        }],
    })
    .expect("sliding plan");
    let manager = CanonicalKvManager::new_with_residence(
        &plan,
        ManagerConfig {
            maximum_requests,
            maximum_operations: 8,
            maximum_prefixes: 1,
            maximum_reclamations: page_count,
            maximum_step_tokens: 64,
        },
        &backends,
        policy,
    )
    .expect("manager");
    (
        RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate),
        backends,
    )
}

fn append_with_view(
    session: &mut RuntimeSession,
    backends: &[BackendArenaRegistration],
    request_id: EngineRequestId,
    target_boundary: u64,
    completion_value: u64,
) -> (EnginePreparedRequestView, EngineBatchPublication) {
    let plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary,
        }])
        .expect("prepare append");
    let mut views = session
        .prepared_execution_view(plan.batch_id)
        .expect("prepared attention view")
        .requests
        .into_vec();
    let view = views.pop().expect("single request view");
    assert!(views.is_empty());
    let evidence = execution_evidence(session, &plan, backends);
    let ticket = session.submit_execution(&evidence).expect("submit append");
    let publication = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 71,
                completion_value,
                confirmed: true,
            },
        )
        .expect("complete append");
    (view, publication)
}

fn semantic_page_shape(view: &EnginePreparedRequestView) -> Vec<(u64, u32, u32, u32)> {
    view.pages
        .iter()
        .map(|page| {
            (
                page.logical_ordinal,
                page.valid_token_count,
                page.visible_token_offset,
                page.visible_token_count,
            )
        })
        .collect()
}

fn token_semantics(
    session: &mut RuntimeSession,
    request_id: EngineRequestId,
    boundary: u64,
) -> Vec<(u64, crate::kv_manager::TokenDisposition)> {
    session
        .token_views_batch(&[EngineTokenViewQuery {
            request_id,
            class_id: 0,
            expected_boundary: boundary,
        }])
        .expect("token view")[0]
        .placements
        .iter()
        .map(|placement| (placement.token_id, placement.disposition))
        .collect()
}

#[test]
fn compiled_and_request_lifetime_residence_preserve_attention_semantics() {
    let request_id = EngineRequestId(801);
    let (mut compiled, compiled_backends) =
        residence_session(PhysicalResidencePolicy::Compiled, 8, 1);
    let (mut conservative, conservative_backends) =
        residence_session(PhysicalResidencePolicy::RequestLifetime, 8, 1);
    compiled.acquire_requests(&[request_id]).expect("acquire");
    conservative
        .acquire_requests(&[request_id])
        .expect("acquire");

    for (session, backends) in [
        (&mut compiled, &compiled_backends[..]),
        (&mut conservative, &conservative_backends[..]),
    ] {
        let (_, publication) = append(
            session,
            backends,
            &[EngineAppendIntent {
                request_id,
                target_boundary: 35,
            }],
            71,
            1,
        );
        confirm_publication(session, &publication);
    }

    let (compiled_view, compiled_publication) =
        append_with_view(&mut compiled, &compiled_backends, request_id, 52, 2);
    let (conservative_view, conservative_publication) =
        append_with_view(&mut conservative, &conservative_backends, request_id, 52, 2);

    assert_eq!(
        semantic_page_shape(&compiled_view),
        semantic_page_shape(&conservative_view)
    );
    assert_eq!(semantic_page_shape(&compiled_view).len(), 3);
    assert_eq!(compiled_publication.retirements.len(), 1);
    assert!(conservative_publication.retirements.is_empty());
    confirm_publication(&mut compiled, &compiled_publication);
    confirm_publication(&mut conservative, &conservative_publication);
    assert_eq!(
        token_semantics(&mut compiled, request_id, 52),
        token_semantics(&mut conservative, request_id, 52)
    );

    let compiled_arena = compiled.arena_stats()[0];
    let conservative_arena = conservative.arena_stats()[0];
    assert_eq!(compiled_arena.active_pages, 2);
    assert_eq!(conservative_arena.active_pages, 4);
    assert_eq!(compiled_arena.page_payload_bytes, 2_048);
    assert_eq!(compiled_arena.resident_pages, 2);
    assert_eq!(compiled_arena.resident_bytes, 4_096);
    assert_eq!(conservative_arena.resident_pages, 4);
    assert_eq!(conservative_arena.resident_bytes, 8_192);

    for (session, expected_retirements) in [(&mut compiled, 2), (&mut conservative, 4)] {
        let release = session
            .prepare_release_batch(&[request_id])
            .expect("prepare release");
        assert_eq!(release.retirements.len(), expected_retirements);
        assert_eq!(
            session.confirm_release(&EngineReleaseEvidence {
                release_id: release.release_id,
                mirror_cleanup_confirmed: true,
                reclamation_receipts: control_reclamation_evidence(&release.retirements),
            }),
            Ok(EngineReleaseOutcome::Completed)
        );
        assert_eq!(session.stats().active_pages, 0);
        assert_eq!(session.stats().free_pages, 8);
    }
}

#[test]
fn compiled_residence_increases_admission_at_fixed_capacity() {
    let first = EngineRequestId(811);
    let second = EngineRequestId(812);
    let (mut compiled, compiled_backends) =
        residence_session(PhysicalResidencePolicy::Compiled, 3, 2);
    let (mut conservative, conservative_backends) =
        residence_session(PhysicalResidencePolicy::RequestLifetime, 3, 2);

    for (session, backends) in [
        (&mut compiled, &compiled_backends[..]),
        (&mut conservative, &conservative_backends[..]),
    ] {
        session.acquire_requests(&[first, second]).expect("acquire");
        let (_, initial) = append(
            session,
            backends,
            &[EngineAppendIntent {
                request_id: first,
                target_boundary: 18,
            }],
            72,
            1,
        );
        confirm_publication(session, &initial);
        let (_, wrapped) = append(
            session,
            backends,
            &[EngineAppendIntent {
                request_id: first,
                target_boundary: 35,
            }],
            72,
            2,
        );
        confirm_publication(session, &wrapped);
    }

    let admitted = compiled
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: second,
            target_boundary: 16,
        }])
        .expect("compiled reclamation admits the second request");
    assert_eq!(
        conservative.prepare_append_batch(&[EngineAppendIntent {
            request_id: second,
            target_boundary: 16,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    compiled
        .abort_prepared_execution(
            admitted.batch_id,
            &[EngineStepAbortEvidence {
                request_id: second,
                backend_unobserved: true,
            }],
        )
        .expect("abort admission probe");
}

#[test]
fn request_lifetime_residence_rejects_resettable_address_programs() {
    let plan = compile_retention_program(crate::retention::RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: PAGE_TOKENS,
        states: vec![crate::retention::RetentionStateDecl {
            name: "chunk".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: crate::retention::Predicate::Equal {
                lhs: crate::retention::IntExpr::FloorDiv {
                    value: Box::new(crate::retention::IntExpr::QueryPosition),
                    divisor: 32,
                },
                rhs: crate::retention::IntExpr::FloorDiv {
                    value: Box::new(crate::retention::IntExpr::KeyPosition),
                    divisor: 32,
                },
            },
        }],
    })
    .expect("chunked plan");
    let result = CanonicalKvManager::new_with_residence(
        &plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 2,
            maximum_step_tokens: 32,
        },
        &[backend(0, 98, 2, 9_000)],
        PhysicalResidencePolicy::RequestLifetime,
    );
    assert!(matches!(
        result,
        Err(KvManagerError::UnsupportedProfile(
            "request-lifetime residence does not support resettable classes"
        ))
    ));
}
