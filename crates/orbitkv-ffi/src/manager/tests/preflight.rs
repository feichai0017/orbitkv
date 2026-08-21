use super::*;

#[test]
fn every_hot_flat_short_buffer_and_precommit_fault_is_zero_mutation() {
    let mut error = [0; 256];
    let handle = create(&mut error);

    let mut count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_arena_identities(
                handle,
                std::ptr::null_mut(),
                0,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(count, 2);
    assert_eq!(
        unsafe {
            orbitkv_manager_arena_stats(
                handle,
                std::ptr::null_mut(),
                0,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );

    let empty = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_request_acquire_batch(
                handle,
                1,
                std::ptr::null_mut(),
                0,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(stats(handle, &mut error), empty);
    let acquired = acquire(handle, 1, &mut error);

    let key = OrbitKvPrefixSemanticKey {
        namespace: [9; 32],
        digest: [8; 32],
        boundary: 16,
    };
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_lookup_batch(
                handle,
                &key,
                1,
                std::ptr::null_mut(),
                0,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );

    let mut prepare_input = OrbitKvPrepareBatchItem {
        request: acquired[0].request,
        expected_head: acquired[0].snapshot,
        target_boundary: 17,
        reserved: 0,
    };
    let (mut item_count, mut class_count, mut tail_count, mut copy_count, mut write_count) =
        (0, 0, 0, 0, 0);
    let before_prepare = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_prepare_batch(
                handle,
                &prepare_input,
                1,
                std::ptr::null_mut(),
                0,
                &mut item_count,
                std::ptr::null_mut(),
                0,
                &mut class_count,
                std::ptr::null_mut(),
                0,
                &mut tail_count,
                std::ptr::null_mut(),
                0,
                &mut copy_count,
                std::ptr::null_mut(),
                0,
                &mut write_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(
        (item_count, class_count, tail_count, copy_count, write_count),
        (1, 2, 2, 2, 8)
    );
    assert_eq!(stats(handle, &mut error), before_prepare);

    prepare_input.reserved = 1;
    assert_eq!(
        unsafe {
            orbitkv_manager_prepare_batch(
                handle,
                &prepare_input,
                1,
                std::ptr::null_mut(),
                0,
                &mut item_count,
                std::ptr::null_mut(),
                0,
                &mut class_count,
                std::ptr::null_mut(),
                0,
                &mut tail_count,
                std::ptr::null_mut(),
                0,
                &mut copy_count,
                std::ptr::null_mut(),
                0,
                &mut write_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    prepare_input.reserved = 0;
    prepare_input.expected_head.generation += 1;
    let mut prepared_output = OrbitKvPreparedBatchItem::default();
    let mut classes = [OrbitKvClassLowering::default(); 2];
    let mut tails = [OrbitKvTailAction::default(); 2];
    let mut copies = [OrbitKvCopyIntent::default(); 2];
    let mut writes = [OrbitKvWriteIntent::default(); 8];
    assert_eq!(
        unsafe {
            orbitkv_manager_prepare_batch(
                handle,
                &prepare_input,
                1,
                &mut prepared_output,
                1,
                &mut item_count,
                classes.as_mut_ptr(),
                2,
                &mut class_count,
                tails.as_mut_ptr(),
                2,
                &mut tail_count,
                copies.as_mut_ptr(),
                2,
                &mut copy_count,
                writes.as_mut_ptr(),
                8,
                &mut write_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(stats(handle, &mut error), before_prepare);

    let prepared = prepare(handle, &acquired, 17, &mut error);
    let (submit_items, bind_receipts, copy_receipts) =
        submission_payload(handle, &prepared, &mut error);
    let before_submit = stats(handle, &mut error);
    let mut submitted_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_submit_batch(
                handle,
                submit_items.as_ptr(),
                1,
                bind_receipts.as_ptr(),
                bind_receipts.len() as u32,
                copy_receipts.as_ptr(),
                copy_receipts.len() as u32,
                std::ptr::null_mut(),
                0,
                &mut submitted_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(stats(handle, &mut error), before_submit);
    let submitted = submit(handle, &prepared, &mut error);

    let complete_input = OrbitKvCompleteBatchItem {
        submission: submitted[0].submission,
    };
    let (mut completed_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let before_complete = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_complete_batch(
                handle,
                OrbitKvBatchCompletionReceipt {
                    engine_epoch: submitted[0].submission.engine_epoch,
                    completion_domain: 2,
                    completion_value: 3,
                    confirmed: 1,
                    reserved: 0,
                },
                &complete_input,
                1,
                std::ptr::null_mut(),
                0,
                &mut completed_count,
                std::ptr::null_mut(),
                0,
                &mut detached_count,
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
        (completed_count, detached_count, retirement_count),
        (1, 12, 12)
    );
    assert_eq!(stats(handle, &mut error), before_complete);
    let completed = complete(handle, &submitted, 3, &mut error);
    acknowledge(handle, &completed.retirements, &mut error);
    let current = published_views(&completed.items);

    let release_input = OrbitKvReleaseBatchItem {
        request: current[0].request,
        expected_head: current[0].snapshot,
    };
    let (mut released_count, mut release_detached_count, mut release_retirement_count) = (0, 0, 0);
    let before_release = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_release_batch(
                handle,
                &release_input,
                1,
                std::ptr::null_mut(),
                0,
                &mut released_count,
                std::ptr::null_mut(),
                0,
                &mut release_detached_count,
                std::ptr::null_mut(),
                0,
                &mut release_retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(
        (
            released_count,
            release_detached_count,
            release_retirement_count
        ),
        (1, current[0].resident_count, current[0].resident_count)
    );
    assert_eq!(stats(handle, &mut error), before_release);
    let certificates = release(handle, &current, &mut error);
    acknowledge(handle, &certificates, &mut error);
    recycle_requests(handle, &current, &mut error);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn canonical_span_reserved_and_retryable_conflict_statuses_are_typed() {
    let mut error = [0; 256];
    let handle = create(&mut error);
    let initial = acquire(handle, 2, &mut error);
    let prepared = prepare(handle, &initial, 32, &mut error);
    let (mut submit_items, mut bind_receipts, copy_receipts) =
        submission_payload(handle, &prepared, &mut error);
    let before_submit = stats(handle, &mut error);
    let mut submitted_outputs = [OrbitKvSubmittedBatchItem::default(); 2];
    let mut submitted_count = 0;

    submit_items[0].receipt_offset = 1;
    assert_eq!(
        unsafe {
            orbitkv_manager_submit_batch(
                handle,
                submit_items.as_ptr(),
                2,
                bind_receipts.as_ptr(),
                bind_receipts.len() as u32,
                copy_receipts.as_ptr(),
                copy_receipts.len() as u32,
                submitted_outputs.as_mut_ptr(),
                2,
                &mut submitted_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(handle, &mut error), before_submit);
    submit_items[0].receipt_offset = 0;

    bind_receipts[0].reserved = 1;
    assert_eq!(
        unsafe {
            orbitkv_manager_submit_batch(
                handle,
                submit_items.as_ptr(),
                2,
                bind_receipts.as_ptr(),
                bind_receipts.len() as u32,
                copy_receipts.as_ptr(),
                copy_receipts.len() as u32,
                submitted_outputs.as_mut_ptr(),
                2,
                &mut submitted_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(handle, &mut error), before_submit);
    let submitted = submit(handle, &prepared, &mut error);

    let complete_inputs = submitted
        .iter()
        .map(|item| OrbitKvCompleteBatchItem {
            submission: item.submission,
        })
        .collect::<Vec<_>>();
    let mut completed_outputs = [OrbitKvCompletedBatchItem::default(); 2];
    let mut detached = vec![OrbitKvDetachedBinding::default(); 64];
    let mut certificates = vec![OrbitKvReclamationCertificate::default(); 32];
    let (mut completed_count, mut detached_count, mut certificate_count) = (0, 0, 0);
    let before_complete = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_complete_batch(
                handle,
                OrbitKvBatchCompletionReceipt {
                    engine_epoch: submitted[0].submission.engine_epoch,
                    completion_domain: 1,
                    completion_value: 1,
                    confirmed: 1,
                    reserved: 1,
                },
                complete_inputs.as_ptr(),
                2,
                completed_outputs.as_mut_ptr(),
                2,
                &mut completed_count,
                detached.as_mut_ptr(),
                64,
                &mut detached_count,
                certificates.as_mut_ptr(),
                32,
                &mut certificate_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(handle, &mut error), before_complete);
    let completed = complete(handle, &submitted, 1, &mut error);
    let current = published_views(&completed.items);

    let key = OrbitKvPrefixSemanticKey {
        namespace: [4; 32],
        digest: [5; 32],
        boundary: 32,
    };
    let publish = OrbitKvPrefixPublishBatchItem {
        request: current[0].request,
        expected_head: current[0].snapshot,
        key,
    };
    let mut prefix = OrbitKvPublishedPrefix::default();
    let mut prefix_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_publish_batch(
                handle,
                &publish,
                1,
                &mut prefix,
                1,
                &mut prefix_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let duplicate = OrbitKvPrefixPublishBatchItem {
        request: current[1].request,
        expected_head: current[1].snapshot,
        key,
    };
    let before_duplicate = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_publish_batch(
                handle,
                &duplicate,
                1,
                &mut prefix,
                1,
                &mut prefix_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(stats(handle, &mut error), before_duplicate);

    let stale_release = OrbitKvReleaseBatchItem {
        request: current[1].request,
        expected_head: OrbitKvSnapshotLease {
            generation: current[1].snapshot.generation + 1,
            ..current[1].snapshot
        },
    };
    let mut release_output = OrbitKvReleasedBatchItem::default();
    let mut release_detached = vec![OrbitKvDetachedBinding::default(); 32];
    let mut release_certificates = vec![OrbitKvReclamationCertificate::default(); 32];
    let (mut release_count, mut release_detached_count, mut release_certificate_count) = (0, 0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_release_batch(
                handle,
                &stale_release,
                1,
                &mut release_output,
                1,
                &mut release_count,
                release_detached.as_mut_ptr(),
                32,
                &mut release_detached_count,
                release_certificates.as_mut_ptr(),
                32,
                &mut release_certificate_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(stats(handle, &mut error), before_duplicate);

    let release_certificates = release(handle, &current, &mut error);
    assert!(!release_certificates.is_empty());
    let mut bad_receipts = release_certificates
        .iter()
        .map(|certificate| OrbitKvReclamationReceipt {
            reclamation: certificate.reclamation,
            page: certificate.page,
            backend_domain: certificate.backend_domain,
            acknowledged: 1,
            reserved8: 0,
            reserved32: 0,
            backend_index: certificate.backend_index,
        })
        .collect::<Vec<_>>();
    bad_receipts[0].reserved32 = 1;
    let before_bad_ack = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_acknowledge_reclamations_batch(
                handle,
                bad_receipts.as_ptr(),
                bad_receipts.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(handle, &mut error), before_bad_ack);
    acknowledge(handle, &release_certificates, &mut error);
    recycle_requests(handle, &current, &mut error);

    let mut evicted = OrbitKvEvictedPrefix::default();
    let mut evict_certificates = vec![OrbitKvReclamationCertificate::default(); 32];
    let (mut evicted_count, mut evict_certificate_count) = (0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_evict_batch(
                handle,
                &prefix.prefix,
                1,
                &mut evicted,
                1,
                &mut evicted_count,
                evict_certificates.as_mut_ptr(),
                32,
                &mut evict_certificate_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    evict_certificates.truncate(evict_certificate_count as usize);
    acknowledge(handle, &evict_certificates, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_recycle_batch(
                handle,
                &prefix.prefix,
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}
