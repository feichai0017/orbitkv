use super::*;

#[test]
fn destroy_releases_an_exclusively_owned_nonquiescent_handle() {
    let mut error = [0; 256];
    let handle = create(&mut error);
    let acquired = acquire(handle, 1, &mut error);
    assert_eq!(acquired.len(), 1);
    assert_eq!(stats(handle, &mut error).active_requests, 1);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn semantic_bind_and_copy_faults_report_known_fail_stopped_quarantine() {
    assert_semantic_submit_fault_is_fail_stopped(|bind_receipts, _| {
        bind_receipts
            .first_mut()
            .expect("bind receipt")
            .backend_index += 1;
    });
    assert_semantic_submit_fault_is_fail_stopped(|_, copy_receipts| {
        copy_receipts
            .first_mut()
            .expect("copy receipt")
            .destination_backend_index += 1;
    });
}

#[test]
fn create_and_abort_reserved_fields_never_reach_core_mutation() {
    let mut error = [0; 256];
    let mut invalid_backends = backends();
    invalid_backends[0].reserved = 1;
    let mut invalid_handle = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            orbitkv_manager_create(
                HYBRID_PLAN.as_ptr(),
                HYBRID_PLAN.len(),
                &config(),
                invalid_backends.as_ptr(),
                2,
                &mut invalid_handle,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert!(invalid_handle.is_null());

    let handle = create(&mut error);
    let acquired = acquire(handle, 1, &mut error);
    let prepared = prepare(handle, &acquired, 16, &mut error);
    let mut receipt = OrbitKvBackendUnobservedReceipt {
        step: prepared.items[0].step,
        backend_unobserved: 1,
        reserved: 1,
    };
    let before = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_abort_steps_batch(handle, &receipt, 1, error.as_mut_ptr(), error.len())
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(handle, &mut error), before);
    receipt.reserved = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_abort_steps_batch(handle, &receipt, 1, error.as_mut_ptr(), error.len())
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(stats(handle, &mut error).prepared_steps, 0);
    let certificates = release(handle, &acquired, &mut error);
    assert!(certificates.is_empty());
    recycle_requests(handle, &acquired, &mut error);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn large_pool_small_step_uses_delta_complete_bound_and_exact_release_census() {
    let mut error = [0; 256];
    let large_config = OrbitKvManagerConfig {
        maximum_requests: 4,
        maximum_operations: 4,
        maximum_prefixes: 4,
        maximum_reclamations: 1024,
        maximum_step_tokens: 16,
    };
    let large_backends = [
        OrbitKvBackendArenaRegistration {
            page_count: 512,
            ..backends()[0]
        },
        OrbitKvBackendArenaRegistration {
            page_count: 512,
            ..backends()[1]
        },
    ];
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            orbitkv_manager_create(
                HYBRID_PLAN.as_ptr(),
                HYBRID_PLAN.len(),
                &large_config,
                large_backends.as_ptr(),
                2,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let acquired = acquire(handle, 1, &mut error);
    let prepared = prepare(handle, &acquired, 16, &mut error);
    let submitted = submit(handle, &prepared, &mut error);
    let complete_input = OrbitKvCompleteBatchItem {
        submission: submitted[0].submission,
    };
    let mut completed = OrbitKvCompletedBatchItem::default();
    let (mut completed_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let before_complete = stats(handle, &mut error);
    assert_eq!(
        before_complete.free_pages + before_complete.writing_pages,
        1024
    );
    assert_eq!(
        unsafe {
            orbitkv_manager_complete_batch(
                handle,
                OrbitKvBatchCompletionReceipt {
                    engine_epoch: submitted[0].submission.engine_epoch,
                    completion_domain: 1,
                    completion_value: 1,
                    confirmed: 1,
                    reserved: 0,
                },
                &complete_input,
                1,
                &mut completed,
                1,
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
    // C=2, ceil(max_step/page)=1: 2*(1+2)=6, independent of T=1024.
    assert_eq!(
        (completed_count, detached_count, retirement_count),
        (1, 6, 6)
    );
    assert_eq!(stats(handle, &mut error), before_complete);

    let completed = complete(handle, &submitted, 1, &mut error);
    let current = published_views(&completed.items);
    assert_eq!(current[0].resident_count, 2);
    let release_input = OrbitKvReleaseBatchItem {
        request: current[0].request,
        expected_head: current[0].snapshot,
    };
    let mut released = OrbitKvReleasedBatchItem::default();
    let (mut released_count, mut release_detached_count, mut release_retirement_count) = (0, 0, 0);
    let before_release = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_release_batch(
                handle,
                &release_input,
                1,
                &mut released,
                1,
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
        (1, 2, 2)
    );
    assert_eq!(stats(handle, &mut error), before_release);
    let certificates = release(handle, &current, &mut error);
    acknowledge(handle, &certificates, &mut error);
    recycle_requests(handle, &current, &mut error);
    assert_eq!(stats(handle, &mut error).free_pages, 1024);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn wire_layout_report() {
    macro_rules! report {
        ($ty:ty; $($field:ident),+ $(,)?) => {{
            print!("{} size={} align={}", stringify!($ty), size_of::<$ty>(), align_of::<$ty>());
            $(print!(" {}={}", stringify!($field), offset_of!($ty, $field));)+
            println!();
        }};
    }
    use std::mem::{align_of, offset_of, size_of};
    report!(OrbitKvRequestLease; engine_epoch, slot, generation);
    report!(OrbitKvSnapshotLease; engine_epoch, slot, generation);
    report!(OrbitKvStepLease; engine_epoch, slot, generation);
    report!(OrbitKvSubmissionLease; engine_epoch, slot, generation);
    report!(OrbitKvReclamationLease; engine_epoch, slot, generation);
    report!(OrbitKvPrefixLease; engine_epoch, slot, generation);
    report!(OrbitKvPageLease; engine_epoch, pool_epoch, generation, page_id, pool_id);
    report!(OrbitKvBackendArenaRegistration; pool_id, class_id, backend_domain, page_count, reserved, backend_base_index);
    report!(OrbitKvManagerConfig; maximum_requests, maximum_operations, maximum_prefixes, maximum_reclamations, maximum_step_tokens);
    report!(OrbitKvArenaIdentity; engine_epoch, pool_epoch, backend_base_index, pool_id, page_count, page_tokens, class_id, backend_domain, first_page_id, reserved);
    report!(OrbitKvArenaStats; engine_epoch, pool_epoch, class_id, backend_domain, pool_id, page_count, first_page_id, reserved, reserved_padding, free_pages, reserved_pages, writing_pages, active_pages, retiring_pages, quarantined_pages, exhausted_pages, request_page_refs, prefix_page_refs, reader_pins);
    report!(OrbitKvRequestView; request, snapshot, view_version, boundary, resident_count, reserved);
    report!(OrbitKvSnapshotPage; page, logical_ordinal, temporal_cell_index, temporal_cycle, backend_index, class_id, backend_domain, valid_token_count, visible_token_offset, visible_token_count, reserved);
    report!(OrbitKvRequestForkBatchItem; source_request, expected_source_head, target_empty_request, expected_target_head);
    report!(OrbitKvForkedBatchItem; source, target, page_offset, page_count);
    report!(OrbitKvPrepareBatchItem; request, expected_head, target_boundary, reserved);
    report!(OrbitKvPreparedBatchItem; step, request, base_snapshot, target_snapshot, base_view_version, target_view_version, previous_boundary, target_boundary, class_offset, class_count, tail_offset, tail_count, copy_offset, copy_count, write_offset, write_count);
    report!(OrbitKvClassLowering; class_id, flags, tail_offset, tail_count, copy_offset, copy_count, write_offset, write_count, reserved);
    report!(OrbitKvTailAction; class_id, kind, valid_token_count, logical_ordinal, source, destination, reserved);
    report!(OrbitKvCopyIntent; class_id, backend_domain, token_count, source_token_offset, destination_token_offset, reserved, source, destination, source_backend_index, destination_backend_index);
    report!(OrbitKvWriteIntent; page_generation, page_id, reserved);
    report!(OrbitKvBackendBindReceipt; step, page, backend_domain, mapped, writable, reserved, backend_index);
    report!(OrbitKvBackendCopyReceipt; step, class_id, backend_domain, token_count, source_token_offset, destination_token_offset, observed, copied, ordered_before_writes, reserved8, reserved32, source, destination, source_backend_index, destination_backend_index);
    report!(OrbitKvSubmitBatchItem; step, receipt_offset, receipt_count, copy_receipt_offset, copy_receipt_count);
    report!(OrbitKvSubmittedBatchItem; submission, request, target_snapshot);
    report!(OrbitKvBatchCompletionReceipt; engine_epoch, completion_domain, completion_value, confirmed, reserved);
    report!(OrbitKvCompleteBatchItem; submission);
    report!(OrbitKvDetachedBinding; old, replacement, logical_ordinal, old_backend_index, replacement_backend_index, token_begin, token_end_exclusive, class_id, backend_domain, action, reason, reserved);
    report!(OrbitKvReclamationCertificate; reclamation, page, class_id, backend_domain, reserved32, logical_ordinal, backend_index, token_begin, token_end_exclusive, completion_domain, completion_value);
    report!(OrbitKvCompletedBatchItem; submission, request, detached_snapshot, published_snapshot, published_view_version, published_boundary, resident_count, detached_offset, detached_count, reserved);
    report!(OrbitKvBackendUnobservedReceipt; step, backend_unobserved, reserved);
    report!(OrbitKvReleaseBatchItem; request, expected_head);
    report!(OrbitKvReleasedBatchItem; request, detached_snapshot, detached_offset, detached_count, reserved);
    report!(OrbitKvReclamationReceipt; reclamation, page, backend_domain, acknowledged, reserved8, reserved32, backend_index);
    report!(OrbitKvPrefixSemanticKey; namespace, digest, boundary);
    report!(OrbitKvPrefixLookupHint; key, candidate, resident_count, candidate_present, reserved, reserved_padding);
    report!(OrbitKvPrefixAttachBatchItem; request, expected_empty_head, hint);
    report!(OrbitKvAttachedPrefixBatchItem; prefix, target, page_offset, page_count);
    report!(OrbitKvPrefixPublishBatchItem; request, expected_head, key);
    report!(OrbitKvPublishedPrefix; prefix, key, resident_count, reserved);
    report!(OrbitKvPrefixPublishReleaseBatchItem; publication, request, detached_snapshot, detached_offset, detached_count, reserved);
    report!(OrbitKvEvictedPrefix; prefix, key);
    report!(OrbitKvManagerStats; active_requests, active_snapshots, active_prefixes, evicted_prefixes, prepared_steps, submitted_steps, free_pages, reserved_pages, writing_pages, active_pages, retiring_pages, quarantined_pages, exhausted_pages, pending_reclamations, total_request_page_refs, total_prefix_page_refs, total_reader_pins);
}
