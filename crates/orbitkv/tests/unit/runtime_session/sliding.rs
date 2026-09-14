use super::*;

fn pure_sliding_plan() -> CompiledKvPlan {
    compile_plan(KvPlanInput {
        page_tokens: PAGE_TOKENS,
        classes: vec![KvClassSpec {
            name: "swa".into(),
            layers: vec![0],
            retention: RetentionKind::Sliding,
            bytes_per_token_per_layer: 128,
            window_tokens: Some(18),
            storage: TokenStorageKind::TokenKv,
            components: Vec::new(),
        }],
    })
    .expect("pure sliding plan")
}

fn pure_sliding_session(
    page_count: u32,
    maximum_requests: u32,
) -> (RuntimeSession, [BackendArenaRegistration; 1]) {
    let backends = [backend(0, 43, page_count, 5_500)];
    let manager = CanonicalKvManager::new(
        &pure_sliding_plan(),
        ManagerConfig {
            maximum_requests,
            maximum_operations: 8,
            maximum_prefixes: 1,
            maximum_reclamations: page_count,
            maximum_step_tokens: 64,
        },
        &backends,
    )
    .expect("manager");
    (
        RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate),
        backends,
    )
}

#[test]
#[allow(clippy::too_many_lines)]
fn pure_sliding_wrap_retires_detached_pages_and_ack_gates_generation_reuse() {
    let (mut session, backends) = pure_sliding_session(3, 2);
    let first = EngineRequestId(31);
    let second = EngineRequestId(32);
    session
        .acquire_requests(&[first, second])
        .expect("acquire pure sliding requests");

    let (_, initial) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: first,
            target_boundary: 18,
        }],
        23,
        1,
    );
    assert!(initial.retirements.is_empty());
    confirm_publication(&mut session, &initial);

    let (_, wrapped) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: first,
            target_boundary: 35,
        }],
        23,
        2,
    );
    assert_eq!(wrapped.retirements.len(), 1);
    assert_eq!(wrapped.retirements[0].class_id, 0);
    let retirement = wrapped.retirements[0];
    let detached = wrapped.steps[0]
        .detached
        .iter()
        .find(|item| item.old == retirement.page)
        .expect("periodic wrap detaches the retiring page");
    assert_eq!(detached.class_id, 0);
    assert_eq!(detached.old_backend_index, retirement.backend_index);
    assert_eq!(session.stats().retiring_pages, 1);

    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: second,
            target_boundary: 16,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    confirm_publication(&mut session, &wrapped);

    let reused = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: second,
            target_boundary: 16,
        }])
        .expect("reuse periodic page after ACK");
    let write = reused.steps[0].write_intents[0];
    assert_eq!(write.page_id, retirement.page.page_id);
    assert_eq!(write.page_generation, retirement.page.generation + 1);
    session
        .abort_prepared_execution(
            reused.batch_id,
            &[EngineStepAbortEvidence {
                request_id: second,
                backend_unobserved: true,
            }],
        )
        .expect("abort reuse probe");

    for request_id in [first, second] {
        let release = session
            .prepare_release_batch(&[request_id])
            .expect("prepare final release");
        assert_eq!(
            session.confirm_release(&EngineReleaseEvidence {
                release_id: release.release_id,
                mirror_cleanup_confirmed: true,
                reclamation_receipts: control_reclamation_evidence(&release.retirements),
            }),
            Ok(EngineReleaseOutcome::Completed)
        );
    }
    let stats = session.stats();
    assert_eq!(stats.active_requests, 0);
    assert_eq!(stats.active_snapshots, 0);
    assert_eq!(stats.active_pages, 0);
    assert_eq!(stats.retiring_pages, 0);
    assert_eq!(stats.pending_reclamations, 0);
    assert_eq!(stats.free_pages, 3);
}

#[test]
fn prepared_sliding_view_keeps_pages_needed_by_the_earliest_query() {
    let (mut session, backends) = pure_sliding_session(4, 1);
    let request_id = EngineRequestId(35);
    session.acquire_requests(&[request_id]).expect("acquire");

    let (_, initial) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: 18,
        }],
        24,
        1,
    );
    confirm_publication(&mut session, &initial);

    let plan = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 35,
        }])
        .expect("prepare window-crossing append");
    let view = session
        .prepared_execution_view(plan.batch_id)
        .expect("materialize prepared execution view");
    let pages = &view.requests[0].pages;
    assert_eq!(
        (
            view.requests[0].previous_boundary,
            view.requests[0].target_boundary
        ),
        (18, 35)
    );
    assert_eq!(
        pages
            .iter()
            .map(|page| page.logical_ordinal)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(pages.last().expect("last page").valid_token_count, 3);
    assert_eq!(
        pages
            .iter()
            .map(|page| (page.visible_token_offset, page.visible_token_count))
            .collect::<Vec<_>>(),
        vec![(16, 0), (2, 14), (0, 3)]
    );

    session
        .abort_prepared_execution(
            plan.batch_id,
            &[EngineStepAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        )
        .expect("abort read-only probe");
}

#[test]
fn pure_sliding_runtime_session_rejects_every_prefix_entrypoint() {
    let (mut session, _) = pure_sliding_session(3, 2);
    let source = EngineRequestId(41);
    let target = EngineRequestId(42);
    session
        .acquire_requests(&[source, target])
        .expect("acquire requests");
    let key = prefix_key(91, 16);
    assert!(matches!(
        session.lookup_prefix_batch(&[key]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    ));
    assert!(matches!(
        session.publish_prefix_batch(&[(source, key)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    ));
    assert!(matches!(
        session.prepare_prefix_attach(&[(
            target,
            EnginePrefixLookup {
                key,
                candidate: None,
                resident_count: 0,
            },
        )]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    ));
    assert!(matches!(
        session.prepare_prefix_evict(&[EnginePrefixId::from_parts(1, 1)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    ));
    assert!(matches!(
        session.publish_prefix_and_release_batch(&[(source, key)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    ));
    assert!(matches!(
        session.prepare_request_fork(&[(source, target)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    ));
    assert_eq!(session.stats().active_prefixes, 0);
}
