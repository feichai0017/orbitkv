use super::*;

#[test]
fn prepare_prefix_evict() {
    let session = Session::new();
    make_ready(&session, 40);
    let semantic_key = key(5, 16);
    let published = publish_prefix(&session, 40, semantic_key);
    let mut control_id = OrbitKvSessionControlId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let mut error = [0; 256];

    for (prefixes, prefix_count) in [
        (std::ptr::null(), 0),
        (std::ptr::null(), 1),
        (std::ptr::null(), MAXIMUM_PREFIXES + 1),
    ] {
        assert_eq!(
            unsafe {
                orbitkv_session_prepare_prefix_evict(
                    session.as_ptr(),
                    prefixes,
                    prefix_count,
                    &mut control_id,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(control_id, OrbitKvSessionControlId::default());
        control_id.sequence = u64::MAX;
    }
    for invalid in [
        OrbitKvSessionPrefixId {
            session_epoch: published.prefix_id.session_epoch + 1,
            sequence: published.prefix_id.sequence,
        },
        OrbitKvSessionPrefixId {
            session_epoch: published.prefix_id.session_epoch,
            sequence: 0,
        },
    ] {
        assert_eq!(
            unsafe {
                orbitkv_session_prepare_prefix_evict(
                    session.as_ptr(),
                    &invalid,
                    1,
                    &mut control_id,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_RETRYABLE_CONFLICT
        );
        assert_eq!(control_id, OrbitKvSessionControlId::default());
    }
    let duplicates = [published.prefix_id; 2];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_evict(
                session.as_ptr(),
                duplicates.as_ptr(),
                2,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_evict(
                session.as_ptr(),
                &published.prefix_id,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(
        unsafe {
            orbitkv_session_abort_control(
                session.as_ptr(),
                control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(lookup_prefix(&session, semantic_key).candidate_present, 1);
}

#[test]
fn abort_control() {
    let session = Session::new();
    make_ready(&session, 50);
    let semantic_key = key(6, 16);
    let published = publish_prefix(&session, 50, semantic_key);
    let control_id = prepare_evict(&session, published.prefix_id);
    let mut error = [0; 256];
    let foreign = OrbitKvSessionControlId {
        session_epoch: control_id.session_epoch + 1,
        sequence: control_id.sequence,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_abort_control(
                session.as_ptr(),
                foreign,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_abort_control(
                session.as_ptr(),
                control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(
        lookup_prefix(&session, semantic_key).candidate,
        published.prefix_id
    );
    assert_eq!(
        unsafe {
            orbitkv_session_abort_control(
                session.as_ptr(),
                control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );

    let committed_id = prepare_evict(&session, published.prefix_id);
    let info = commit(&session, committed_id);
    assert_eq!(info.kind, ORBITKV_SESSION_CONTROL_KIND_PREFIX_EVICTION);
    assert_eq!(
        unsafe {
            orbitkv_session_abort_control(
                session.as_ptr(),
                committed_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    let (_, retirements) = read_eviction_plan(&session, committed_id);
    let evidence = retirement_evidence(&retirements);
    let mut outcome = OrbitKvSessionControlOutcome::default();
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: committed_id,
                    mirror_updates_confirmed: 1,
                    reserved: 0,
                },
                evidence.as_ptr(),
                evidence.len() as u32,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn confirm_control() {
    let session = Session::new();
    make_ready(&session, 80);
    let semantic_key = key(8, 16);
    let published = publish_prefix(&session, 80, semantic_key);
    release_request(&session, 80);
    let control_id = prepare_evict(&session, published.prefix_id);
    let info = commit(&session, control_id);
    assert_eq!(info.kind, ORBITKV_SESSION_CONTROL_KIND_PREFIX_EVICTION);
    let (_, retirements) = read_eviction_plan(&session, control_id);
    assert_eq!(retirements.len(), 1);
    let evidence = retirement_evidence(&retirements);
    let sentinel = OrbitKvSessionControlOutcome {
        id: control_id,
        disposition: u32::MAX,
        reserved: u32::MAX,
    };
    let mut error = [0; 256];

    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: 1,
                    reserved: 0,
                },
                evidence.as_ptr(),
                evidence.len() as u32,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    for bad_evidence in [
        OrbitKvSessionControlEvidence {
            id: control_id,
            mirror_updates_confirmed: 1,
            reserved: 1,
        },
        OrbitKvSessionControlEvidence {
            id: control_id,
            mirror_updates_confirmed: 2,
            reserved: 0,
        },
        OrbitKvSessionControlEvidence {
            id: OrbitKvSessionControlId {
                session_epoch: control_id.session_epoch + 1,
                sequence: control_id.sequence,
            },
            mirror_updates_confirmed: 1,
            reserved: 0,
        },
    ] {
        let mut outcome = sentinel;
        let expected = if bad_evidence.id == control_id {
            ORBITKV_STATUS_INVALID_ARGUMENT
        } else {
            ORBITKV_STATUS_RETRYABLE_CONFLICT
        };
        assert_eq!(
            unsafe {
                orbitkv_session_confirm_control(
                    session.as_ptr(),
                    bad_evidence,
                    evidence.as_ptr(),
                    evidence.len() as u32,
                    &mut outcome,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            expected
        );
        assert_eq!(outcome, OrbitKvSessionControlOutcome::default());
    }
    let mut outcome = sentinel;
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: 1,
                    reserved: 0,
                },
                std::ptr::null(),
                PAGE_CAPACITY + 1,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(outcome, OrbitKvSessionControlOutcome::default());

    let mut invalid_receipt = evidence[0];
    invalid_receipt.acknowledged = 2;
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: 1,
                    reserved: 0,
                },
                &invalid_receipt,
                1,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    invalid_receipt = evidence[0];
    invalid_receipt.reserved32 = 1;
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: 1,
                    reserved: 0,
                },
                &invalid_receipt,
                1,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: 0,
                    reserved: 0,
                },
                evidence.as_ptr(),
                1,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );

    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: 1,
                    reserved: 0,
                },
                evidence.as_ptr(),
                1,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(outcome.id, control_id);
    assert_eq!(outcome.disposition, ORBITKV_SESSION_CONTROL_OUTCOME_EVICTED);
    assert_eq!(outcome.reserved, 0);
    outcome = sentinel;
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: 1,
                    reserved: 0,
                },
                evidence.as_ptr(),
                1,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(outcome, OrbitKvSessionControlOutcome::default());
    assert_eq!(lookup_prefix(&session, semantic_key).candidate_present, 0);
}

#[test]
fn pending_attach_cancel_validation_and_output_init() {
    let session = Session::new();
    make_ready(&session, 81);
    let semantic_key = key(10, 16);
    let published = publish_prefix(&session, 81, semantic_key);
    let lookup = lookup_prefix(&session, semantic_key);
    acquire(&session, &[82]);
    let item = OrbitKvSessionPrefixAttachItem {
        target_request_id: 82,
        prefix_id: lookup.candidate,
        key: lookup.key,
        resident_count: lookup.resident_count,
        reserved: 0,
    };
    let mut control_id = OrbitKvSessionControlId::default();
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_attach(
                session.as_ptr(),
                &item,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let _info = commit(&session, control_id);
    let (requests, pages) = read_materialization_plan(&session, control_id);
    assert_eq!((requests.len(), pages.len()), (1, 1));
    let expected = OrbitKvSessionPendingAttachCancel {
        control_id,
        request_id: 82,
        prefix_id: published.prefix_id,
        view_version: requests[0].view_version,
        boundary: requests[0].boundary,
        resident_count: requests[0].resident_count,
    };

    let sentinel = OrbitKvSessionPendingAttachCancelOutcome {
        control_id: OrbitKvSessionControlId {
            session_epoch: u64::MAX,
            sequence: u64::MAX,
        },
        request_id: u64::MAX,
        prefix_id: OrbitKvSessionPrefixId {
            session_epoch: u64::MAX,
            sequence: u64::MAX,
        },
        view_version: u64::MAX,
        boundary: u64::MAX,
        resident_count: u32::MAX,
        disposition: u32::MAX,
    };

    assert_eq!(
        unsafe {
            orbitkv_session_cancel_pending_attach(
                session.as_ptr(),
                expected,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );

    let foreign = OrbitKvSessionPendingAttachCancel {
        control_id: OrbitKvSessionControlId {
            session_epoch: control_id.session_epoch + 1,
            sequence: control_id.sequence,
        },
        ..expected
    };
    let mut outcome = sentinel;
    assert_eq!(
        unsafe {
            orbitkv_session_cancel_pending_attach(
                session.as_ptr(),
                foreign,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(outcome, OrbitKvSessionPendingAttachCancelOutcome::default());

    let mut mismatched = expected;
    mismatched.boundary += 1;
    outcome = sentinel;
    assert_eq!(
        unsafe {
            orbitkv_session_cancel_pending_attach(
                session.as_ptr(),
                mismatched,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(outcome, OrbitKvSessionPendingAttachCancelOutcome::default());

    let mut not_pending = sentinel;
    assert_eq!(
        unsafe {
            orbitkv_session_finalize_pending_attach_cancel(
                session.as_ptr(),
                expected,
                &mut not_pending,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(
        not_pending,
        OrbitKvSessionPendingAttachCancelOutcome::default()
    );

    assert_eq!(
        unsafe {
            orbitkv_session_cancel_pending_attach(
                session.as_ptr(),
                expected,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(
        outcome.disposition,
        ORBITKV_SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING
    );

    let mut cancel_replay = sentinel;
    assert_eq!(
        unsafe {
            orbitkv_session_cancel_pending_attach(
                session.as_ptr(),
                expected,
                &mut cancel_replay,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(cancel_replay, outcome);

    let mut finalized = sentinel;
    assert_eq!(
        unsafe {
            orbitkv_session_finalize_pending_attach_cancel(
                session.as_ptr(),
                expected,
                &mut finalized,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(
        finalized.disposition,
        ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED
    );

    let mut stale = sentinel;
    assert_eq!(
        unsafe {
            orbitkv_session_cancel_pending_attach(
                session.as_ptr(),
                expected,
                &mut stale,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(stale.control_id, control_id);
    assert_eq!(stale.request_id, 82);
    assert_eq!(stale.prefix_id, published.prefix_id);
    assert_eq!(
        stale.disposition,
        ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED
    );
}

#[test]
fn quarantine_control() {
    let session = Session::new();
    make_ready(&session, 90);
    acquire(&session, &[91, 92]);
    let control_id = prepare_fork(&session, 90, 91);
    let mut error = [0; 256];

    assert_eq!(
        unsafe {
            orbitkv_session_quarantine_control(
                session.as_ptr(),
                control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    let info = commit(&session, control_id);
    assert_eq!(info.kind, ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION);
    let foreign = OrbitKvSessionControlId {
        session_epoch: control_id.session_epoch + 1,
        sequence: control_id.sequence,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_quarantine_control(
                session.as_ptr(),
                foreign,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_quarantine_control(
                session.as_ptr(),
                control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let mut replay = OrbitKvSessionControlPlanInfo {
        reserved: u32::MAX,
        ..OrbitKvSessionControlPlanInfo::default()
    };
    assert_eq!(
        unsafe {
            orbitkv_session_commit_control(
                session.as_ptr(),
                control_id,
                &mut replay,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(replay, OrbitKvSessionControlPlanInfo::default());

    let target_intent = OrbitKvSessionAppendIntent {
        request_id: 91,
        target_boundary: 32,
    };
    let mut batch = OrbitKvSessionBatchId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let mut step = OrbitKvSessionPreparedStep::default();
    let mut class = OrbitKvClassLowering::default();
    let mut tail = OrbitKvTailAction::default();
    let mut copy = OrbitKvCopyIntent::default();
    let mut write = OrbitKvWriteIntent::default();
    let (mut step_count, mut class_count, mut tail_count, mut copy_count, mut write_count) =
        (0, 0, 0, 0, 0);
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_append(
                session.as_ptr(),
                &target_intent,
                1,
                &mut batch,
                &mut step,
                1,
                &mut step_count,
                &mut class,
                1,
                &mut class_count,
                &mut tail,
                1,
                &mut tail_count,
                &mut copy,
                1,
                &mut copy_count,
                &mut write,
                1,
                &mut write_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(batch, OrbitKvSessionBatchId::default());

    let source_append = prepare_append(&session, 90, 32);
    let abort = OrbitKvSessionStepAbortEvidence {
        request_id: 90,
        backend_unobserved: 1,
        reserved: 0,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_abort_prepared(
                session.as_ptr(),
                source_append.batch,
                &abort,
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let unrelated_append = prepare_append(&session, 92, 16);
    let abort = OrbitKvSessionStepAbortEvidence {
        request_id: 92,
        backend_unobserved: 1,
        reserved: 0,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_abort_prepared(
                session.as_ptr(),
                unrelated_append.batch,
                &abort,
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
}
