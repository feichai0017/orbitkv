fn ready_prefix_release_sources(
    session: &mut RuntimeSession,
    backends: &[BackendArenaRegistration],
    request_ids: &[EngineRequestId],
) {
    session
        .acquire_requests(request_ids)
        .expect("acquire prefix-release sources");
    let intents = request_ids
        .iter()
        .copied()
        .map(|request_id| EngineAppendIntent {
            request_id,
            target_boundary: 16,
        })
        .collect::<Vec<_>>();
    let (_, publication) = append(session, backends, &intents, 101, 1);
    confirm_publication(session, &publication);
}

#[test]
fn prefix_publish_release_manager_error_is_batch_atomic() {
    let backends = [backend(0, 92, 4, 43_000)];
    let mut session = session_with_config(
        &full_plan(),
        &backends,
        ManagerConfig {
            maximum_requests: 2,
            maximum_operations: 8,
            maximum_prefixes: 2,
            maximum_reclamations: 4,
            maximum_step_tokens: 64,
        },
    );
    let requests = [EngineRequestId(310), EngineRequestId(311)];
    ready_prefix_release_sources(&mut session, &backends, &requests);
    let before = session.stats();
    let keys = [prefix_key(32, 16), prefix_key(33, 32)];

    assert_eq!(
        session
            .publish_prefix_and_release_batch(&[(requests[0], keys[0]), (requests[1], keys[1]),]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PrefixBoundaryMismatch
        ))
    );
    assert_eq!(session.stats(), before);
    assert_eq!(session.next_prefix_sequence, 1);
    assert_eq!(session.next_release_sequence, 1);
    assert!(session.prefixes.is_empty());
    assert!(session.releases.is_empty());
    assert!(
        requests
            .iter()
            .all(|request_id| session.requests[request_id].phase == RequestPhase::Ready)
    );
    assert_eq!(
        session
            .lookup_prefix_batch(&keys)
            .expect("lookup after rejected transfer")
            .iter()
            .map(|hint| hint.candidate)
            .collect::<Vec<_>>(),
        vec![None, None]
    );

    let retry = session
        .publish_prefix_and_release_batch(&[
            (requests[0], keys[0]),
            (requests[1], prefix_key(33, 16)),
        ])
        .expect("retry valid transfer");
    assert_eq!(retry.release_id.sequence(), 1);
    assert_eq!(retry.items[0].prefix_id.sequence(), 1);
    assert_eq!(retry.items[1].prefix_id.sequence(), 2);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: retry.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
    assert_eq!(session.stats().active_requests, 0);
    assert_eq!(session.stats().active_prefixes, 2);
    assert_eq!(
        session
            .lookup_prefix_batch(&[keys[0], prefix_key(33, 16)])
            .expect("lookup transferred prefixes")
            .iter()
            .map(|hint| hint.candidate)
            .collect::<Vec<_>>(),
        retry
            .items
            .iter()
            .map(|item| Some(item.prefix_id))
            .collect::<Vec<_>>()
    );
}

#[test]
fn prefix_publish_release_reserves_both_identity_ranges_before_commit() {
    for exhausted in ["prefix", "release"] {
        let backends = [backend(
            0,
            if exhausted == "prefix" { 93 } else { 94 },
            2,
            44_000,
        )];
        let request_ids: Box<[EngineRequestId]> = if exhausted == "prefix" {
            Box::new([EngineRequestId(320), EngineRequestId(321)])
        } else {
            Box::new([EngineRequestId(322)])
        };
        let request_count = u32::try_from(request_ids.len()).expect("small test batch");
        let mut session = session_with_config(
            &full_plan(),
            &backends,
            ManagerConfig {
                maximum_requests: request_count,
                maximum_operations: 8,
                maximum_prefixes: request_count,
                maximum_reclamations: 2,
                maximum_step_tokens: 64,
            },
        );
        ready_prefix_release_sources(&mut session, &backends, &request_ids);
        if exhausted == "prefix" {
            session.next_prefix_sequence = u64::MAX - 1;
        } else {
            session.next_release_sequence = u64::MAX;
        }
        let before = session.stats();
        let before_prefix = session.next_prefix_sequence;
        let before_release = session.next_release_sequence;
        let items = request_ids
            .iter()
            .enumerate()
            .map(|(index, request_id)| {
                let tag = 34 + u8::try_from(index).expect("small test batch");
                (*request_id, prefix_key(tag, 16))
            })
            .collect::<Vec<_>>();

        assert_eq!(
            session.publish_prefix_and_release_batch(&items),
            Err(RuntimeSessionError::IdentityExhausted(exhausted))
        );
        assert_eq!(session.stats(), before);
        assert_eq!(session.next_prefix_sequence, before_prefix);
        assert_eq!(session.next_release_sequence, before_release);
        assert!(
            request_ids
                .iter()
                .all(|request_id| session.requests[request_id].phase == RequestPhase::Ready)
        );
        assert!(session.prefixes.is_empty());
        assert!(session.releases.is_empty());
    }
}

#[test]
fn prefix_publish_release_stays_pending_until_confirm_then_keeps_prefix() {
    let backends = [backend(0, 95, 1, 45_000)];
    let mut session = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(330);
    let key = prefix_key(35, 16);
    ready_prefix_release_sources(&mut session, &backends, &[request_id]);

    let plan = session
        .publish_prefix_and_release_batch(&[(request_id, key)])
        .expect("atomic prefix publish-release");
    assert_eq!(plan.items.len(), 1);
    assert_eq!(plan.items[0].request_id, request_id);
    assert_eq!(plan.items[0].key, key);
    assert_eq!(plan.items[0].resident_count, 1);
    assert_eq!(plan.items[0].detached.len(), 1);
    assert_eq!(
        session.lookup_prefix_batch(&[key]).expect("lookup")[0].candidate,
        Some(plan.items[0].prefix_id)
    );
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
    assert!(matches!(
        session.prepare_release_batch(&[request_id]),
        Err(RuntimeSessionError::RequestNotReady {
            request_id: pending,
            state: "release confirmation pending",
        }) if pending == request_id
    ));
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: plan.release_id,
            mirror_cleanup_confirmed: false,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::MirrorCleanupNotConfirmed)
    );

    session.inject_test_fault(RuntimeSessionTestFault::ReleaseRecycleOnce);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: plan.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineReleaseOutcome::RecyclePending)
    );
    assert_eq!(
        session.lookup_prefix_batch(&[key]).expect("pending lookup")[0].candidate,
        Some(plan.items[0].prefix_id)
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: plan.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::ReleaseRetryNotIdOnly)
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: plan.release_id,
            mirror_cleanup_confirmed: false,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
    assert_eq!(session.stats().active_requests, 0);
    assert_eq!(session.stats().active_prefixes, 1);
    assert_eq!(
        session
            .lookup_prefix_batch(&[key])
            .expect("resident lookup")[0]
            .candidate,
        Some(plan.items[0].prefix_id)
    );
    session
        .acquire_requests(&[request_id])
        .expect("recycle engine request identity");
}

#[test]
fn prefix_publish_release_wrong_detached_sticky_poisons_session() {
    let backends = [backend(0, 96, 1, 46_000)];
    let mut session = session(&full_plan(), &backends, 1);
    let request_id = EngineRequestId(340);
    ready_prefix_release_sources(&mut session, &backends, &[request_id]);
    session.inject_test_fault(RuntimeSessionTestFault::PrefixPublishReleaseOutput);

    let poisoned = RuntimeSessionError::SessionPoisoned("prefix publish-release result changed");
    assert_eq!(
        session.publish_prefix_and_release_batch(&[(request_id, prefix_key(36, 16))]),
        Err(poisoned.clone())
    );
    assert_eq!(session.stats().active_prefixes, 1);
    assert_eq!(session.next_prefix_sequence, 2);
    assert_eq!(session.next_release_sequence, 2);
    assert_eq!(
        session.acquire_requests(&[EngineRequestId(341)]),
        Err(poisoned)
    );
}

#[test]
fn prefix_publish_release_public_dtos_hide_manager_capabilities() {
    let value = serde_json::to_value(EnginePrefixPublishReleasePlan {
        release_id: EngineReleaseId::from_parts(7, 1),
        items: Box::new([EnginePublishedPrefixRelease {
            request_id: EngineRequestId(2),
            prefix_id: EnginePrefixId::from_parts(7, 3),
            key: prefix_key(37, 16),
            resident_count: 0,
            detached: Box::new([]),
        }]),
    })
    .expect("serialize prefix publish-release plan");
    assert_no_manager_capability(&value);
    let wire = value.to_string();
    for forbidden in [
        "request_lease",
        "snapshot",
        "prefix_lease",
        "reclamation",
        "step",
        "submission",
    ] {
        assert!(!wire.contains(forbidden), "public DTO leaked {forbidden}");
    }
}
