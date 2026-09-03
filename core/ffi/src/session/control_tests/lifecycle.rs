use super::*;

#[test]
fn prefix_lookup_batch() {
    let session = Session::new();
    let semantic_key = key(1, 16);
    let mut error = [0; 256];
    let mut count = u32::MAX;
    let sentinel = OrbitKvSessionPrefixLookup {
        resident_count: u32::MAX,
        ..OrbitKvSessionPrefixLookup::default()
    };
    let mut output = sentinel;

    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                std::ptr::null(),
                0,
                &mut output,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(output, sentinel);
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                std::ptr::null(),
                1,
                &mut output,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                std::ptr::null(),
                MAXIMUM_PREFIXES + 1,
                &mut output,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                &semantic_key,
                1,
                &mut output,
                0,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(count, 1);
    assert_eq!(output, sentinel);
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                &semantic_key,
                1,
                std::ptr::null_mut(),
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                &semantic_key,
                1,
                &mut output,
                1,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );

    output = OrbitKvSessionPrefixLookup::default();
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                &semantic_key,
                1,
                &mut output,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(output.key, semantic_key);
    assert_eq!(output.candidate, OrbitKvSessionPrefixId::default());
    assert_eq!(output.resident_count, 0);
    assert_eq!(output.candidate_present, 0);
    assert_eq!((output.reserved0, output.reserved1), (0, 0));

    make_ready(&session, 1);
    let published = publish_prefix(&session, 1, semantic_key);
    let hit = lookup_prefix(&session, semantic_key);
    assert_eq!(hit.candidate_present, 1);
    assert_eq!(hit.candidate, published.prefix_id);
    assert_eq!(hit.resident_count, published.resident_count);

    unsafe { session.0.as_ref() }
        .unwrap()
        .state
        .lock()
        .unwrap()
        .fail_stopped = true;
    output = sentinel;
    count = u32::MAX;
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                &semantic_key,
                1,
                &mut output,
                0,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );
    assert_eq!(count, u32::MAX);
    assert_eq!(output, sentinel);
}

#[test]
fn prefix_publish_batch() {
    let session = Session::new();
    make_ready(&session, 10);
    let first_key = key(2, 16);
    let item = OrbitKvSessionPrefixPublishItem {
        request_id: 10,
        key: first_key,
    };
    let sentinel = OrbitKvSessionPublishedPrefix {
        resident_count: u32::MAX,
        reserved: u32::MAX,
        ..OrbitKvSessionPublishedPrefix::default()
    };
    let mut output = sentinel;
    let mut count = u32::MAX;
    let mut error = [0; 256];

    for (items, item_count) in [
        (std::ptr::null(), 0),
        (std::ptr::null(), 1),
        (std::ptr::null(), MAXIMUM_PREFIXES + 1),
    ] {
        assert_eq!(
            unsafe {
                orbitkv_session_prefix_publish_batch(
                    session.as_ptr(),
                    items,
                    item_count,
                    &mut output,
                    1,
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(output, sentinel);
    }
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_batch(
                session.as_ptr(),
                &item,
                1,
                &mut output,
                0,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(count, 1);
    assert_eq!(output, sentinel);
    assert_eq!(lookup_prefix(&session, first_key).candidate_present, 0);
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_batch(
                session.as_ptr(),
                &item,
                1,
                std::ptr::null_mut(),
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(lookup_prefix(&session, first_key).candidate_present, 0);

    let duplicate_items = [
        item,
        OrbitKvSessionPrefixPublishItem {
            request_id: 10,
            key: key(3, 16),
        },
    ];
    let mut duplicate_outputs = [sentinel; 2];
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_batch(
                session.as_ptr(),
                duplicate_items.as_ptr(),
                2,
                duplicate_outputs.as_mut_ptr(),
                2,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(duplicate_outputs, [sentinel; 2]);

    output = OrbitKvSessionPublishedPrefix::default();
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_batch(
                session.as_ptr(),
                &item,
                1,
                &mut output,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_ne!(output.prefix_id, OrbitKvSessionPrefixId::default());
    assert_eq!(output.key, first_key);
    assert_eq!(output.resident_count, 1);
    assert_eq!(output.reserved, 0);

    unsafe { session.0.as_ref() }
        .unwrap()
        .state
        .lock()
        .unwrap()
        .fail_stopped = true;
    let mut stopped_output = sentinel;
    let mut stopped_count = u32::MAX;
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_batch(
                session.as_ptr(),
                &OrbitKvSessionPrefixPublishItem {
                    request_id: 11,
                    key: key(22, 16),
                },
                1,
                &mut stopped_output,
                0,
                &mut stopped_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );
    assert_eq!(stopped_count, u32::MAX);
    assert_eq!(stopped_output, sentinel);
}

#[test]
fn prepare_prefix_attach() {
    let session = Session::new();
    make_ready(&session, 20);
    let semantic_key = key(4, 16);
    let published = publish_prefix(&session, 20, semantic_key);
    let lookup = lookup_prefix(&session, semantic_key);
    acquire(&session, &[21]);
    let item = OrbitKvSessionPrefixAttachItem {
        target_request_id: 21,
        prefix_id: lookup.candidate,
        key: lookup.key,
        resident_count: lookup.resident_count,
        reserved: 0,
    };
    let mut control_id = OrbitKvSessionControlId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let mut error = [0; 256];

    for (items, item_count) in [
        (std::ptr::null(), 0),
        (std::ptr::null(), 1),
        (std::ptr::null(), MAXIMUM_PREFIXES + 1),
    ] {
        assert_eq!(
            unsafe {
                orbitkv_session_prepare_prefix_attach(
                    session.as_ptr(),
                    items,
                    item_count,
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
    let mut invalid = item;
    invalid.reserved = 1;
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_attach(
                session.as_ptr(),
                &invalid,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(control_id, OrbitKvSessionControlId::default());
    invalid = item;
    invalid.prefix_id.session_epoch += 1;
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_attach(
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
    invalid = item;
    invalid.resident_count += 1;
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_attach(
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
    assert_eq!(control_id.session_epoch, published.prefix_id.session_epoch);
    let info = commit(&session, control_id);
    assert_eq!(info.kind, ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION);
    assert_eq!((info.request_count, info.page_count), (1, 1));
    confirm_materialization(&session, control_id);
    let appended = prepare_append(&session, 21, 32);
    assert_ne!(appended.batch, OrbitKvSessionBatchId::default());
}

#[test]
fn prepare_request_fork() {
    let session = Session::new();
    make_ready(&session, 30);
    acquire(&session, &[31, 32]);
    let mut control_id = OrbitKvSessionControlId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let mut error = [0; 256];

    for (items, item_count) in [
        (std::ptr::null(), 0),
        (std::ptr::null(), 1),
        (std::ptr::null(), MAXIMUM_OPERATIONS + 1),
    ] {
        assert_eq!(
            unsafe {
                orbitkv_session_prepare_request_fork(
                    session.as_ptr(),
                    items,
                    item_count,
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
    let overlap = OrbitKvSessionRequestForkItem {
        source_request_id: 30,
        target_request_id: 30,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_request_fork(
                session.as_ptr(),
                &overlap,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    let duplicate_target = [
        OrbitKvSessionRequestForkItem {
            source_request_id: 30,
            target_request_id: 31,
        },
        OrbitKvSessionRequestForkItem {
            source_request_id: 30,
            target_request_id: 31,
        },
    ];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_request_fork(
                session.as_ptr(),
                duplicate_target.as_ptr(),
                2,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );

    let valid = [
        OrbitKvSessionRequestForkItem {
            source_request_id: 30,
            target_request_id: 31,
        },
        OrbitKvSessionRequestForkItem {
            source_request_id: 30,
            target_request_id: 32,
        },
    ];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_request_fork(
                session.as_ptr(),
                valid.as_ptr(),
                2,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_ne!(control_id, OrbitKvSessionControlId::default());
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
}

#[test]
fn commit_control() {
    let session = Session::new();
    make_ready(&session, 60);
    acquire(&session, &[61]);
    let control_id = prepare_fork(&session, 60, 61);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_commit_control(
                session.as_ptr(),
                control_id,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );

    let mut info = OrbitKvSessionControlPlanInfo {
        reserved: u32::MAX,
        ..OrbitKvSessionControlPlanInfo::default()
    };
    let foreign = OrbitKvSessionControlId {
        session_epoch: control_id.session_epoch + 1,
        sequence: control_id.sequence,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_commit_control(
                session.as_ptr(),
                foreign,
                &mut info,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(info, OrbitKvSessionControlPlanInfo::default());

    assert_eq!(
        unsafe {
            orbitkv_session_commit_control(
                session.as_ptr(),
                control_id,
                &mut info,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(std::mem::size_of::<OrbitKvSessionControlPlanInfo>(), 40);
    assert_eq!(info.id, control_id);
    assert_eq!(info.kind, ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION);
    assert_eq!(info.reserved, 0);
    assert_eq!(
        (
            info.request_count,
            info.page_count,
            info.prefix_count,
            info.retirement_count
        ),
        (1, 1, 0, 0)
    );
    let first = info;
    let after_first = stats(&session);
    info = OrbitKvSessionControlPlanInfo::default();
    assert_eq!(
        unsafe {
            orbitkv_session_commit_control(
                session.as_ptr(),
                control_id,
                &mut info,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(info, first);
    assert_eq!(stats(&session), after_first);
    confirm_materialization(&session, control_id);
}

#[test]
fn cancel_pending_attach_and_finalize_replay() {
    let session = Session::new();
    make_ready(&session, 62);
    let semantic_key = key(9, 16);
    let published = publish_prefix(&session, 62, semantic_key);
    let lookup = lookup_prefix(&session, semantic_key);
    acquire(&session, &[63]);
    let item = OrbitKvSessionPrefixAttachItem {
        target_request_id: 63,
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
    let info = commit(&session, control_id);
    assert_eq!(info.kind, ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION);
    let (requests, pages) = read_materialization_plan(&session, control_id);
    assert_eq!((requests.len(), pages.len()), (1, 1));
    let expected = OrbitKvSessionPendingAttachCancel {
        control_id,
        request_id: requests[0].request_id,
        prefix_id: published.prefix_id,
        view_version: requests[0].view_version,
        boundary: requests[0].boundary,
        resident_count: requests[0].resident_count,
    };
    assert_eq!(std::mem::size_of::<OrbitKvSessionPendingAttachCancel>(), 64);
    assert_eq!(
        std::mem::size_of::<OrbitKvSessionPendingAttachCancelOutcome>(),
        64
    );

    let sentinel = OrbitKvSessionPendingAttachCancelOutcome {
        disposition: u32::MAX,
        ..OrbitKvSessionPendingAttachCancelOutcome::default()
    };
    let mut outcome = sentinel;
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
    assert_eq!(outcome.control_id, control_id);
    assert_eq!(outcome.request_id, 63);
    assert_eq!(outcome.prefix_id, published.prefix_id);
    assert_eq!(outcome.view_version, requests[0].view_version);
    assert_eq!(outcome.boundary, requests[0].boundary);
    assert_eq!(outcome.resident_count, requests[0].resident_count);
    assert_eq!(
        outcome.disposition,
        ORBITKV_SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING
    );

    let after_cancel = stats(&session);
    assert_eq!(after_cancel.active_requests, 2);
    let mut replay = OrbitKvSessionPendingAttachCancelOutcome::default();
    assert_eq!(
        unsafe {
            orbitkv_session_cancel_pending_attach(
                session.as_ptr(),
                expected,
                &mut replay,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(replay, outcome);

    let mut finalized = OrbitKvSessionPendingAttachCancelOutcome::default();
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
    assert_eq!(finalized.control_id, control_id);
    assert_eq!(finalized.request_id, 63);
    assert_eq!(
        finalized.disposition,
        ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED
    );
    let mut finalized_replay = OrbitKvSessionPendingAttachCancelOutcome::default();
    assert_eq!(
        unsafe {
            orbitkv_session_finalize_pending_attach_cancel(
                session.as_ptr(),
                expected,
                &mut finalized_replay,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(finalized_replay, finalized);

    acquire(&session, &[64]);
    let active = stats(&session);
    assert_eq!(active.active_requests, 2);
}

#[test]
#[allow(clippy::too_many_lines)]
fn read_control_plan() {
    let session = Session::new();
    make_ready(&session, 70);
    acquire(&session, &[71]);
    let control_id = prepare_fork(&session, 70, 71);
    let mut error = [0; 256];
    let (mut request_count, mut page_count, mut prefix_count, mut retirement_count) =
        (u32::MAX, u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                std::ptr::null_mut(),
                0,
                &mut request_count,
                std::ptr::null_mut(),
                0,
                &mut page_count,
                std::ptr::null_mut(),
                0,
                &mut prefix_count,
                std::ptr::null_mut(),
                0,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(
        (request_count, page_count, prefix_count, retirement_count),
        (u32::MAX, u32::MAX, u32::MAX, u32::MAX)
    );
    let info = commit(&session, control_id);
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                std::ptr::null_mut(),
                0,
                &mut request_count,
                std::ptr::null_mut(),
                0,
                &mut page_count,
                std::ptr::null_mut(),
                0,
                &mut prefix_count,
                std::ptr::null_mut(),
                0,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(
        (request_count, page_count, prefix_count, retirement_count),
        (info.request_count, info.page_count, 0, 0)
    );
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                &mut page_count,
                std::ptr::null_mut(),
                0,
                &mut prefix_count,
                std::ptr::null_mut(),
                0,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );

    let request_sentinel = OrbitKvSessionMaterializedRequest {
        request_id: u64::MAX,
        reserved: u32::MAX,
        ..OrbitKvSessionMaterializedRequest::default()
    };
    let page_sentinel = OrbitKvSnapshotPage {
        reserved: u32::MAX,
        ..OrbitKvSnapshotPage::default()
    };
    let prefix_sentinel = OrbitKvSessionPrefixId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let retirement_sentinel = OrbitKvSessionRetirement {
        reserved32: u32::MAX,
        ..OrbitKvSessionRetirement::default()
    };
    for (request_capacity, page_capacity) in [(0, info.page_count), (info.request_count, 0)] {
        let mut requests = [request_sentinel];
        let mut pages = [page_sentinel];
        let mut prefixes = [prefix_sentinel];
        let mut retirements = [retirement_sentinel];
        assert_eq!(
            unsafe {
                orbitkv_session_read_control_plan(
                    session.as_ptr(),
                    control_id,
                    requests.as_mut_ptr(),
                    request_capacity,
                    &mut request_count,
                    pages.as_mut_ptr(),
                    page_capacity,
                    &mut page_count,
                    prefixes.as_mut_ptr(),
                    0,
                    &mut prefix_count,
                    retirements.as_mut_ptr(),
                    0,
                    &mut retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!(requests, [request_sentinel]);
        assert_eq!(pages, [page_sentinel]);
        assert_eq!(prefixes, [prefix_sentinel]);
        assert_eq!(retirements, [retirement_sentinel]);
    }

    let mut requests =
        vec![OrbitKvSessionMaterializedRequest::default(); info.request_count as usize];
    let mut pages = vec![OrbitKvSnapshotPage::default(); info.page_count as usize];
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                requests.as_mut_ptr(),
                requests.len() as u32,
                &mut request_count,
                pages.as_mut_ptr(),
                pages.len() as u32,
                &mut page_count,
                std::ptr::null_mut(),
                0,
                &mut prefix_count,
                std::ptr::null_mut(),
                0,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(requests[0].request_id, 71);
    assert_eq!(requests[0].boundary, 16);
    assert_eq!(requests[0].resident_count, 1);
    assert_eq!((requests[0].page_offset, requests[0].page_count), (0, 1));
    assert_eq!(requests[0].reserved, 0);
    let first_requests = requests.clone();
    let first_pages = pages.clone();
    requests.fill(OrbitKvSessionMaterializedRequest::default());
    pages.fill(OrbitKvSnapshotPage::default());
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                requests.as_mut_ptr(),
                requests.len() as u32,
                &mut request_count,
                pages.as_mut_ptr(),
                pages.len() as u32,
                &mut page_count,
                std::ptr::null_mut(),
                0,
                &mut prefix_count,
                std::ptr::null_mut(),
                0,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!((requests, pages), (first_requests, first_pages));
    confirm_materialization(&session, control_id);

    let eviction_session = Session::new();
    make_ready(&eviction_session, 72);
    let published = publish_prefix(&eviction_session, 72, key(7, 16));
    release_request(&eviction_session, 72);
    let eviction_id = prepare_evict(&eviction_session, published.prefix_id);
    let eviction_info = commit(&eviction_session, eviction_id);
    assert_eq!(
        (eviction_info.prefix_count, eviction_info.retirement_count),
        (1, 1)
    );
    for (prefix_capacity, retirement_capacity) in [(0, 1), (1, 0)] {
        let mut requests = [request_sentinel];
        let mut pages = [page_sentinel];
        let mut prefixes = [prefix_sentinel];
        let mut retirements = [retirement_sentinel];
        assert_eq!(
            unsafe {
                orbitkv_session_read_control_plan(
                    eviction_session.as_ptr(),
                    eviction_id,
                    requests.as_mut_ptr(),
                    0,
                    &mut request_count,
                    pages.as_mut_ptr(),
                    0,
                    &mut page_count,
                    prefixes.as_mut_ptr(),
                    prefix_capacity,
                    &mut prefix_count,
                    retirements.as_mut_ptr(),
                    retirement_capacity,
                    &mut retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!(requests, [request_sentinel]);
        assert_eq!(pages, [page_sentinel]);
        assert_eq!(prefixes, [prefix_sentinel]);
        assert_eq!(retirements, [retirement_sentinel]);
    }
    let (prefixes, retirements) = read_eviction_plan(&eviction_session, eviction_id);
    assert_eq!(prefixes, [published.prefix_id]);
    assert_eq!(retirements.len(), 1);
    let replay = read_eviction_plan(&eviction_session, eviction_id);
    assert_eq!(replay, (prefixes.clone(), retirements.clone()));
    let evidence = retirement_evidence(&retirements);
    let mut outcome = OrbitKvSessionControlOutcome::default();
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                eviction_session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: eviction_id,
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

    let corrupted = Session::new();
    make_ready(&corrupted, 73);
    acquire(&corrupted, &[74]);
    let corrupt_id = prepare_fork(&corrupted, 73, 74);
    commit(&corrupted, corrupt_id);
    {
        let mut state = unsafe { corrupted.0.as_ref() }
            .unwrap()
            .state
            .lock()
            .unwrap();
        let EngineControlPlan::Materialization(plan) = state
            .control_plans
            .get_mut(&EngineControlId::from_parts(
                corrupt_id.session_epoch,
                corrupt_id.sequence,
            ))
            .unwrap()
        else {
            panic!("fork control must cache a materialization plan");
        };
        plan.requests[0].resident_count += 1;
    }
    let (mut rc, mut pc, mut fc, mut tc) = (0, 0, 0, 0);
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                corrupted.as_ptr(),
                corrupt_id,
                std::ptr::null_mut(),
                0,
                &mut rc,
                std::ptr::null_mut(),
                0,
                &mut pc,
                std::ptr::null_mut(),
                0,
                &mut fc,
                std::ptr::null_mut(),
                0,
                &mut tc,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );
    assert!(
        unsafe { corrupted.0.as_ref() }
            .unwrap()
            .state
            .lock()
            .unwrap()
            .fail_stopped
    );
}
