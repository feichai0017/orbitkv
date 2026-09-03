#[test]
fn batch_prefix_publication_allocates_contiguous_ids_without_skipping_high_water() {
    let backends = [backend(0, 91, 4, 42_000)];
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
    let requests = [EngineRequestId(300), EngineRequestId(301)];
    session
        .acquire_requests(&requests)
        .expect("acquire sources");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[
            EngineAppendIntent {
                request_id: requests[0],
                target_boundary: 16,
            },
            EngineAppendIntent {
                request_id: requests[1],
                target_boundary: 16,
            },
        ],
        41,
        1,
    );
    confirm_publication(&mut session, &publication);
    let keys = [prefix_key(30, 16), prefix_key(31, 16)];
    let published = session
        .publish_prefix_batch(&[(requests[0], keys[0]), (requests[1], keys[1])])
        .expect("publish batch");
    let epoch = published[0].prefix_id.session_epoch();
    assert_eq!(published.len(), 2);
    assert_eq!(published[0].prefix_id, EnginePrefixId::from_parts(epoch, 1));
    assert_eq!(published[1].prefix_id, EnginePrefixId::from_parts(epoch, 2));
    assert_eq!(session.next_prefix_sequence, 3);

    let future = EnginePrefixId::from_parts(epoch, 3);
    assert_eq!(
        session.prepare_prefix_evict(&[future]),
        Err(RuntimeSessionError::UnknownPrefix(future))
    );

    let evict_id = session
        .prepare_prefix_evict(&[published[0].prefix_id])
        .expect("prepare first eviction");
    let plan = session
        .commit_control(evict_id)
        .expect("commit first eviction");
    assert!(eviction(&plan).retirements.is_empty());
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id: evict_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineControlOutcome::Evicted)
    );
    assert_eq!(
        session.prepare_prefix_evict(&[published[0].prefix_id]),
        Err(RuntimeSessionError::StalePrefix(published[0].prefix_id))
    );
    assert_eq!(
        session
            .lookup_prefix_batch(&[keys[1]])
            .expect("lookup second prefix")[0]
            .candidate,
        Some(published[1].prefix_id)
    );
}

#[test]
fn prefix_and_control_ids_reject_foreign_zero_future_and_stale_values() {
    let backends = [backend(0, 85, 4, 36_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let source = EngineRequestId(240);
    let (key, published) = publish_ready_prefix(&mut session, &backends, source, 23, 16);
    let epoch = published.prefix_id.session_epoch();
    let foreign_prefix = EnginePrefixId::from_parts(epoch + 1, published.prefix_id.sequence());
    let zero_prefix = EnginePrefixId::from_parts(epoch, 0);
    let future_prefix = EnginePrefixId::from_parts(epoch, u64::MAX);
    assert_eq!(
        session.prepare_prefix_evict(&[foreign_prefix]),
        Err(RuntimeSessionError::ForeignPrefix(foreign_prefix))
    );
    for prefix_id in [zero_prefix, future_prefix] {
        assert_eq!(
            session.prepare_prefix_evict(&[prefix_id]),
            Err(RuntimeSessionError::UnknownPrefix(prefix_id))
        );
    }

    let control_id = session
        .prepare_prefix_evict(&[published.prefix_id])
        .expect("prepare eviction");
    let foreign_control = EngineControlId::from_parts(epoch + 1, control_id.sequence());
    let zero_control = EngineControlId::from_parts(epoch, 0);
    let future_control = EngineControlId::from_parts(epoch, u64::MAX);
    assert_eq!(
        session.commit_control(foreign_control),
        Err(RuntimeSessionError::ForeignControl(foreign_control))
    );
    assert_eq!(
        session.abort_control(zero_control),
        Err(RuntimeSessionError::UnknownControl(zero_control))
    );
    assert_eq!(
        session.quarantine_control(future_control),
        Err(RuntimeSessionError::UnknownControl(future_control))
    );
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::ControlNotCommitted(control_id))
    );
    session.abort_control(control_id).expect("abort eviction");
    assert_eq!(
        session.commit_control(control_id),
        Err(RuntimeSessionError::StaleControl(control_id))
    );

    let evict_id = session
        .prepare_prefix_evict(&[published.prefix_id])
        .expect("prepare eviction retry");
    let plan = session.commit_control(evict_id).expect("commit eviction");
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id: evict_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&eviction(&plan).retirements),
        }),
        Ok(EngineControlOutcome::Evicted)
    );
    assert_eq!(
        session.prepare_prefix_evict(&[published.prefix_id]),
        Err(RuntimeSessionError::StalePrefix(published.prefix_id))
    );
    assert_eq!(
        session
            .lookup_prefix_batch(&[key])
            .expect("lookup evicted prefix")[0]
            .candidate,
        None
    );
}

#[test]
fn prefix_and_control_sequence_exhaustion_is_failure_atomic() {
    let backends = [backend(0, 86, 4, 37_000)];
    let mut prefix_session = session(&full_plan(), &backends, 1);
    let source = EngineRequestId(250);
    prefix_session
        .acquire_requests(&[source])
        .expect("acquire source");
    let (_, publication) = append(
        &mut prefix_session,
        &backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 16,
        }],
        40,
        1,
    );
    confirm_publication(&mut prefix_session, &publication);
    prefix_session.next_prefix_sequence = u64::MAX;
    let key = prefix_key(24, 16);
    let baseline = prefix_session.stats();
    for _ in 0..2 {
        assert_eq!(
            prefix_session.publish_prefix_batch(&[(source, key)]),
            Err(RuntimeSessionError::IdentityExhausted("prefix"))
        );
        assert_eq!(prefix_session.stats(), baseline);
        assert_eq!(
            prefix_session
                .lookup_prefix_batch(&[key])
                .expect("lookup unpublished")[0]
                .candidate,
            None
        );
    }

    let mut control_session = session(&full_plan(), &backends, 2);
    let source = EngineRequestId(251);
    let (key, published) = publish_ready_prefix(&mut control_session, &backends, source, 25, 16);
    let target = EngineRequestId(252);
    control_session
        .acquire_requests(&[target])
        .expect("acquire target");
    let hint = control_session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    control_session.next_control_sequence = u64::MAX;
    let baseline = control_session.stats();
    for _ in 0..2 {
        assert_eq!(
            control_session.prepare_prefix_attach(&[(target, hint)]),
            Err(RuntimeSessionError::IdentityExhausted("control"))
        );
        assert_eq!(control_session.stats(), baseline);
        assert_eq!(
            control_session.prepare_prefix_evict(&[published.prefix_id]),
            Err(RuntimeSessionError::IdentityExhausted("control"))
        );
        assert_eq!(control_session.stats(), baseline);
        assert_eq!(
            control_session.prepare_request_fork(&[(source, target)]),
            Err(RuntimeSessionError::IdentityExhausted("control"))
        );
        assert_eq!(control_session.stats(), baseline);
    }
    control_session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: target,
            target_boundary: 16,
        }])
        .expect("target remained ready");
}

#[test]
fn control_manager_error_preserves_reservation_for_abort() {
    let backends = [backend(0, 87, 4, 38_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let source = EngineRequestId(260);
    let (key, published) = publish_ready_prefix(&mut session, &backends, source, 26, 16);
    let target = EngineRequestId(261);
    session.acquire_requests(&[target]).expect("acquire target");
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup")[0];

    let attach_id = session
        .prepare_prefix_attach(&[(target, hint)])
        .expect("prepare attach");
    session.inject_test_fault(RuntimeSessionTestFault::ControlCommitManagerOnce);
    assert_eq!(
        session.commit_control(attach_id),
        Err(RuntimeSessionError::Manager(
            KvManagerError::ArenaExhausted("snapshot")
        ))
    );
    assert!(session.controls.contains_key(&attach_id));
    session
        .abort_control(attach_id)
        .expect("abort preserved attach");

    let fork_id = session
        .prepare_request_fork(&[(source, target)])
        .expect("prepare fork");
    session.inject_test_fault(RuntimeSessionTestFault::ControlCommitManagerOnce);
    assert_eq!(
        session.commit_control(fork_id),
        Err(RuntimeSessionError::Manager(
            KvManagerError::ArenaExhausted("snapshot")
        ))
    );
    assert!(session.controls.contains_key(&fork_id));
    session
        .abort_control(fork_id)
        .expect("abort preserved fork");

    let evict_id = session
        .prepare_prefix_evict(&[published.prefix_id])
        .expect("prepare eviction");
    session.inject_test_fault(RuntimeSessionTestFault::ControlCommitManagerOnce);
    assert_eq!(
        session.commit_control(evict_id),
        Err(RuntimeSessionError::Manager(
            KvManagerError::ArenaExhausted("reclamation")
        ))
    );
    assert!(session.controls.contains_key(&evict_id));
    session
        .abort_control(evict_id)
        .expect("abort preserved eviction");
    assert_eq!(
        session
            .lookup_prefix_batch(&[key])
            .expect("lookup after abort")[0]
            .candidate,
        Some(published.prefix_id)
    );
}

#[test]
fn eviction_quarantine_is_terminal_without_ack_or_recycle() {
    let backends = [backend(0, 88, 1, 39_000)];
    let mut session = session(&full_plan(), &backends, 1);
    let source = EngineRequestId(270);
    let (key, published) = publish_ready_prefix(&mut session, &backends, source, 27, 16);
    release_ready_request(&mut session, source);
    let control_id = session
        .prepare_prefix_evict(&[published.prefix_id])
        .expect("prepare eviction");
    let plan = session.commit_control(control_id).expect("commit eviction");
    assert_eq!(eviction(&plan).retirements.len(), 1);
    let pending = session.stats();
    assert_eq!(pending.pending_reclamations, 1);
    session
        .quarantine_control(control_id)
        .expect("quarantine eviction");
    assert_eq!(session.stats(), pending);
    assert_eq!(
        session
            .lookup_prefix_batch(&[key])
            .expect("lookup quarantined")[0]
            .candidate,
        None
    );
    assert!(matches!(
        session.prepare_prefix_evict(&[published.prefix_id]),
        Err(RuntimeSessionError::PrefixNotReady {
            state: "quarantined",
            ..
        })
    ));
    assert_eq!(
        session.commit_control(control_id),
        Err(RuntimeSessionError::StaleControl(control_id))
    );
    assert_eq!(session.stats().free_pages, 0);
}

#[test]
fn prefix_reservation_rejects_conflicting_attach_and_duplicate_evict() {
    let backends = [backend(0, 90, 2, 41_000)];
    let mut session = session(&full_plan(), &backends, 3);
    let source = EngineRequestId(290);
    let (key, published) = publish_ready_prefix(&mut session, &backends, source, 29, 16);
    let targets = [EngineRequestId(291), EngineRequestId(292)];
    session.acquire_requests(&targets).expect("acquire targets");
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    let attach_id = session
        .prepare_prefix_attach(&[(targets[0], hint)])
        .expect("prepare attach");
    assert!(matches!(
        session.prepare_prefix_attach(&[(targets[1], hint)]),
        Err(RuntimeSessionError::PrefixNotReady {
            prefix_id,
            state: "attach control pending",
        }) if prefix_id == published.prefix_id
    ));
    session.abort_control(attach_id).expect("abort attach");
    assert_eq!(
        session.prepare_prefix_evict(&[published.prefix_id, published.prefix_id]),
        Err(RuntimeSessionError::DuplicatePrefix(published.prefix_id))
    );
}

#[test]
fn control_public_dtos_hide_manager_capability_leases() {
    let backends = [backend(0, 89, 2, 40_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let source = EngineRequestId(280);
    let (key, published) = publish_ready_prefix(&mut session, &backends, source, 28, 16);
    let lookup = session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    let target = EngineRequestId(281);
    session.acquire_requests(&[target]).expect("acquire target");
    let attach_id = session
        .prepare_prefix_attach(&[(target, lookup)])
        .expect("prepare attach");
    let materialization = session.commit_control(attach_id).expect("commit attach");
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id: attach_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineControlOutcome::Materialized)
    );
    release_ready_request(&mut session, source);
    release_ready_request(&mut session, target);
    let control_id = session
        .prepare_prefix_evict(&[published.prefix_id])
        .expect("prepare eviction");
    let plan = session.commit_control(control_id).expect("commit eviction");
    let evidence = EngineControlEvidence {
        control_id,
        mirror_updates_confirmed: true,
        reclamation_receipts: control_reclamation_evidence(&eviction(&plan).retirements),
    };
    for value in [
        serde_json::to_value(lookup).expect("serialize lookup"),
        serde_json::to_value(published).expect("serialize publication"),
        serde_json::to_value(&materialization).expect("serialize materialization"),
        serde_json::to_value(&plan).expect("serialize control plan"),
        serde_json::to_value(&evidence).expect("serialize control evidence"),
    ] {
        let wire = value.to_string();
        assert!(!wire.contains("\"slot\""));
        assert!(!wire.contains("\"reclamation\""));
        assert!(!wire.contains("\"snapshot\""));
        assert!(!wire.contains("\"step\""));
        assert!(!wire.contains("\"submission\""));
    }
}

#[test]
fn committed_attach_cancel_replays_tombstone_until_finalize() {
    let backends = [backend(0, 92, 2, 43_000)];
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
    let source = EngineRequestId(310);
    let (key, published) = publish_ready_prefix(&mut session, &backends, source, 31, 16);
    let target = EngineRequestId(311);
    session.acquire_requests(&[target]).expect("acquire target");
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    let control_id = session
        .prepare_prefix_attach(&[(target, hint)])
        .expect("prepare attach");
    let plan = session.commit_control(control_id).expect("commit attach");
    let attached = &materialization(&plan).requests[0];
    assert_eq!(materialization(&plan).requests.len(), 1);
    let after_attach = session.stats();

    let identity = EnginePendingAttachCancel {
        control_id,
        request_id: target,
        prefix_id: published.prefix_id,
        view_version: attached.view_version,
        boundary: attached.boundary,
        resident_count: attached.resident_count,
    };
    let expected = EnginePendingAttachCancelOutcome {
        disposition: EnginePendingAttachCancelDisposition::RecyclePending,
        control_id,
        request_id: target,
        prefix_id: published.prefix_id,
        view_version: attached.view_version,
        boundary: attached.boundary,
        resident_count: attached.resident_count,
    };
    assert_eq!(
        session.cancel_pending_attach(identity),
        Ok(expected)
    );
    let after_cancel = session.stats();
    assert_eq!(after_cancel.active_requests, after_attach.active_requests);
    assert_eq!(after_cancel.active_snapshots + 1, after_attach.active_snapshots);
    assert_eq!(
        after_cancel.total_request_page_refs + u64::from(expected.resident_count),
        after_attach.total_request_page_refs
    );
    assert_eq!(
        after_cancel.total_prefix_page_refs,
        after_attach.total_prefix_page_refs
    );
    assert_eq!(after_cancel.pending_reclamations, 0);
    assert_eq!(after_cancel.retiring_pages, 0);
    assert_eq!(
        session.cancel_pending_attach(identity),
        Ok(expected)
    );
    assert_eq!(
        session.commit_control(control_id),
        Err(RuntimeSessionError::StaleControl(control_id))
    );
    assert_eq!(
        session.acquire_requests(&[EngineRequestId(312)]),
        Err(RuntimeSessionError::Manager(KvManagerError::ArenaExhausted(
            "request"
        )))
    );

    let finalized = EnginePendingAttachCancelOutcome {
        disposition: EnginePendingAttachCancelDisposition::Finalized,
        ..expected
    };
    assert_eq!(
        session.finalize_pending_attach_cancel(identity),
        Ok(finalized)
    );
    assert_eq!(
        session.finalize_pending_attach_cancel(identity),
        Ok(finalized)
    );
    assert_eq!(session.stats().active_requests + 1, after_attach.active_requests);
    session
        .acquire_requests(&[EngineRequestId(312)])
        .expect("reacquire recycled request slot");
    assert_eq!(session.cancel_pending_attach(identity), Ok(finalized));
}

#[test]
fn committed_attach_cancel_rejects_fork_materialization() {
    let backends = [backend(0, 93, 3, 44_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let source = EngineRequestId(320);
    let target = EngineRequestId(321);
    session
        .acquire_requests(&[source, target])
        .expect("acquire requests");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 16,
        }],
        51,
        1,
    );
    confirm_publication(&mut session, &publication);
    let control_id = session
        .prepare_request_fork(&[(source, target)])
        .expect("prepare fork");
    let plan = session.commit_control(control_id).expect("commit fork");
    assert_eq!(materialization(&plan).requests.len(), 1);
    assert_eq!(
        session.cancel_pending_attach(EnginePendingAttachCancel {
            control_id,
            request_id: target,
            prefix_id: EnginePrefixId::from_parts(control_id.session_epoch(), 1),
            view_version: materialization(&plan).requests[0].view_version,
            boundary: materialization(&plan).requests[0].boundary,
            resident_count: materialization(&plan).requests[0].resident_count,
        }),
        Err(RuntimeSessionError::ControlNotCancelable(control_id))
    );
    assert_eq!(
        session.finalize_pending_attach_cancel(EnginePendingAttachCancel {
            control_id,
            request_id: target,
            prefix_id: EnginePrefixId::from_parts(control_id.session_epoch(), 1),
            view_version: materialization(&plan).requests[0].view_version,
            boundary: materialization(&plan).requests[0].boundary,
            resident_count: materialization(&plan).requests[0].resident_count,
        }),
        Err(RuntimeSessionError::CanceledRequestNotPending(control_id))
    );
}

#[test]
fn committed_attach_cancel_requires_committed_singleton_control() {
    let backends = [backend(0, 94, 3, 45_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let source = EngineRequestId(330);
    let (key, _) = publish_ready_prefix(&mut session, &backends, source, 32, 16);
    let target = EngineRequestId(331);
    session.acquire_requests(&[target]).expect("acquire target");
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    let control_id = session
        .prepare_prefix_attach(&[(target, hint)])
        .expect("prepare attach");
    assert_eq!(
        session.cancel_pending_attach(EnginePendingAttachCancel {
            control_id,
            request_id: target,
            prefix_id: hint.candidate.expect("lookup hit"),
            view_version: ViewVersion(1),
            boundary: 0,
            resident_count: 0,
        }),
        Err(RuntimeSessionError::ControlNotCommitted(control_id))
    );
    assert_eq!(
        session.finalize_pending_attach_cancel(EnginePendingAttachCancel {
            control_id,
            request_id: target,
            prefix_id: hint.candidate.expect("lookup hit"),
            view_version: ViewVersion(1),
            boundary: 0,
            resident_count: 0,
        }),
        Err(RuntimeSessionError::CanceledRequestNotPending(control_id))
    );
}

#[test]
fn pending_attach_cancel_expectation_mismatch_is_failure_atomic() {
    let backends = [backend(0, 95, 2, 46_000)];
    let mut session = session(&full_plan(), &backends, 2);
    let source = EngineRequestId(340);
    let (key, published) = publish_ready_prefix(&mut session, &backends, source, 33, 16);
    let target = EngineRequestId(341);
    session.acquire_requests(&[target]).expect("acquire target");
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    let control_id = session
        .prepare_prefix_attach(&[(target, hint)])
        .expect("prepare attach");
    let plan = session.commit_control(control_id).expect("commit attach");
    let item = &materialization(&plan).requests[0];
    let exact = EnginePendingAttachCancel {
        control_id,
        request_id: target,
        prefix_id: published.prefix_id,
        view_version: item.view_version,
        boundary: item.boundary,
        resident_count: item.resident_count,
    };
    let baseline = session.stats();
    for mismatched in [
        EnginePendingAttachCancel { request_id: EngineRequestId(999), ..exact },
        EnginePendingAttachCancel {
            prefix_id: EnginePrefixId::from_parts(control_id.session_epoch(), 999),
            ..exact
        },
        EnginePendingAttachCancel {
            view_version: ViewVersion(exact.view_version.0 + 1),
            ..exact
        },
        EnginePendingAttachCancel { boundary: exact.boundary + 1, ..exact },
        EnginePendingAttachCancel { resident_count: exact.resident_count + 1, ..exact },
    ] {
        assert_eq!(
            session.cancel_pending_attach(mismatched),
            Err(RuntimeSessionError::PendingAttachCancelMismatch(control_id))
        );
        assert_eq!(session.stats(), baseline);
        assert_eq!(session.commit_control(control_id), Ok(plan.clone()));
    }
    session
        .cancel_pending_attach(exact)
        .expect("exact cancel after mismatch");
}

#[test]
fn finalized_cancel_tombstones_are_bounded_without_capacity_loss() {
    let backends = [backend(0, 96, 2, 47_000)];
    let mut session = session_with_config(
        &full_plan(),
        &backends,
        ManagerConfig {
            maximum_requests: 2,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 2,
            maximum_step_tokens: 64,
        },
    );
    let source = EngineRequestId(350);
    let (key, _) = publish_ready_prefix(&mut session, &backends, source, 34, 16);
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    for sequence in 0..3 {
        let target = EngineRequestId(351 + sequence);
        session.acquire_requests(&[target]).expect("acquire target");
        let control_id = session
            .prepare_prefix_attach(&[(target, hint)])
            .expect("finalized tombstone yielded control capacity");
        let plan = session.commit_control(control_id).expect("commit attach");
        let item = &materialization(&plan).requests[0];
        let exact = EnginePendingAttachCancel {
            control_id,
            request_id: target,
            prefix_id: hint.candidate.expect("lookup hit"),
            view_version: item.view_version,
            boundary: item.boundary,
            resident_count: item.resident_count,
        };
        session.cancel_pending_attach(exact).expect("cancel attach");
        if sequence == 0 {
            assert_eq!(
                session.prepare_prefix_evict(&[exact.prefix_id]),
                Err(RuntimeSessionError::Manager(KvManagerError::ArenaExhausted(
                    "control"
                )))
            );
        }
        session
            .finalize_pending_attach_cancel(exact)
            .expect("finalize attach cancel");
    }
}

#[test]
fn control_id_exhaustion_preserves_finalized_cancel_replay() {
    let backends = [backend(0, 97, 2, 48_000)];
    let mut session = session_with_config(
        &full_plan(),
        &backends,
        ManagerConfig {
            maximum_requests: 2,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 2,
            maximum_step_tokens: 64,
        },
    );
    let source = EngineRequestId(360);
    let (key, _) = publish_ready_prefix(&mut session, &backends, source, 35, 16);
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup")[0];
    let target = EngineRequestId(361);
    session.acquire_requests(&[target]).expect("acquire target");
    let control_id = session
        .prepare_prefix_attach(&[(target, hint)])
        .expect("prepare attach");
    let plan = session.commit_control(control_id).expect("commit attach");
    let item = &materialization(&plan).requests[0];
    let exact = EnginePendingAttachCancel {
        control_id,
        request_id: target,
        prefix_id: hint.candidate.expect("lookup hit"),
        view_version: item.view_version,
        boundary: item.boundary,
        resident_count: item.resident_count,
    };
    session.cancel_pending_attach(exact).expect("cancel");
    let finalized = session
        .finalize_pending_attach_cancel(exact)
        .expect("finalize");
    session.next_control_sequence = u64::MAX;

    assert_eq!(
        session.prepare_prefix_evict(&[exact.prefix_id]),
        Err(RuntimeSessionError::IdentityExhausted("control"))
    );
    assert_eq!(session.cancel_pending_attach(exact), Ok(finalized));
    assert_eq!(
        session.finalize_pending_attach_cancel(exact),
        Ok(finalized)
    );
}
