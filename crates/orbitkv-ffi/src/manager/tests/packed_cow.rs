use super::*;

const FULL_PLAN: &[u8] = br#"{
  "page_tokens": 16,
  "classes": [
    {"name":"full","layers":[0],"retention":"full","bytes_per_token_per_layer":128}
  ]
}"#;
const ABI8_CLASS_LOWERING_PACKED: u16 = 1;
const ABI8_TOKEN_POLICY_EVICTED: u16 = 2;

fn create_full(error: &mut [c_char]) -> *mut OrbitKvManagerHandle {
    let config = OrbitKvManagerConfig {
        maximum_requests: 2,
        maximum_operations: 4,
        maximum_prefixes: 1,
        maximum_reclamations: 32,
        maximum_step_tokens: 64,
    };
    let backend = OrbitKvBackendArenaRegistration {
        pool_id: 7,
        class_id: 0,
        backend_domain: 3,
        page_count: 16,
        reserved: 0,
        backend_base_index: 100,
    };
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            orbitkv_manager_create(
                FULL_PLAN.as_ptr(),
                FULL_PLAN.len(),
                &config,
                &backend,
                1,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert!(!handle.is_null());
    handle
}

fn mark_relocation_victims(
    handle: *mut OrbitKvManagerHandle,
    source: OrbitKvRequestView,
    error: &mut [c_char],
) -> OrbitKvRequestView {
    let updates = (8..16)
        .chain(24..32)
        .chain(40..48)
        .map(|token_id| OrbitKvClassTokenDispositionUpdate {
            token_id,
            disposition: OrbitKvTokenDisposition {
                policy_or_proof_id: 17,
                version: 1,
                quality_contract: 99,
                kind: ABI8_TOKEN_POLICY_EVICTED,
                reserved16: 0,
                reserved32: 0,
            },
            class_id: 0,
            reserved16: 0,
            reserved32: 0,
        })
        .collect::<Vec<_>>();
    let input = OrbitKvTokenDispositionBatchItem {
        request: source.request,
        expected_snapshot: source.snapshot,
        update_offset: 0,
        update_count: updates.len() as u32,
    };
    let mut output = OrbitKvRequestView::default();
    let mut output_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_mark_token_dispositions_batch(
                handle,
                &input,
                1,
                updates.as_ptr(),
                updates.len() as u32,
                &mut output,
                1,
                &mut output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(output_count, 1);
    output
}

fn relocate_to_partial_packed(
    handle: *mut OrbitKvManagerHandle,
    source: OrbitKvRequestView,
    error: &mut [c_char],
) -> OrbitKvRequestView {
    let input = OrbitKvPrepareRelocationItem {
        request: source.request,
        expected_snapshot: source.snapshot,
        policy: OrbitKvRelocationPolicy {
            maximum_source_pages: 3,
            evacuation_headroom_pages: 2,
            fragmentation_threshold_milli: 250,
            full_evacuation: 1,
            reserved8: 0,
            reserved32: 0,
        },
        class_id: 0,
        reserved16: 0,
        reserved32: 0,
    };
    let mut prepared = OrbitKvPreparedRelocation::default();
    let mut sources = vec![OrbitKvPageLease::default(); 16];
    let mut destinations = vec![OrbitKvPageLease::default(); 16];
    let mut moves = vec![OrbitKvTokenMove::default(); 16 * 16];
    let (mut prepared_count, mut source_count, mut destination_count, mut move_count) =
        (0, 0, 0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_prepare_relocation_batch(
                handle,
                &input,
                1,
                &mut prepared,
                1,
                &mut prepared_count,
                sources.as_mut_ptr(),
                sources.len() as u32,
                &mut source_count,
                destinations.as_mut_ptr(),
                destinations.len() as u32,
                &mut destination_count,
                moves.as_mut_ptr(),
                moves.len() as u32,
                &mut move_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(prepared_count, 1);
    assert_eq!((source_count, destination_count, move_count), (3, 2, 24));
    assert_eq!(prepared.projected_reclaimed_pages, 1);
    moves.truncate(move_count as usize);

    let receipts = moves
        .iter()
        .map(|movement| OrbitKvRelocationCopyReceipt {
            relocation: prepared.relocation,
            token_id: movement.token_id,
            source: movement.source,
            destination: movement.destination,
            observed: 1,
            copied: 1,
            reserved16: 0,
            reserved32: 0,
        })
        .collect::<Vec<_>>();
    let mut submitted = OrbitKvSubmittedRelocation::default();
    let mut submitted_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_submit_relocation_batch(
                handle,
                &prepared.relocation,
                1,
                receipts.as_ptr(),
                receipts.len() as u32,
                &mut submitted,
                1,
                &mut submitted_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(submitted_count, 1);

    let completion = OrbitKvBatchCompletionReceipt {
        engine_epoch: source.request.engine_epoch,
        completion_domain: 2,
        completion_value: 2,
        confirmed: 1,
        reserved: 0,
    };
    let mut publication = OrbitKvRequestView::default();
    let mut retirements = vec![OrbitKvReclamationCertificate::default(); 16];
    let (mut publication_count, mut retirement_count) = (0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_complete_relocation_batch(
                handle,
                &completion,
                &submitted.relocation,
                1,
                &mut publication,
                1,
                &mut publication_count,
                retirements.as_mut_ptr(),
                retirements.len() as u32,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(publication_count, 1);
    retirements.truncate(retirement_count as usize);
    assert_eq!(retirements.len(), 3);
    acknowledge(handle, &retirements, error);
    publication
}

fn final_arena_stats(handle: *mut OrbitKvManagerHandle, error: &mut [c_char]) -> OrbitKvArenaStats {
    let mut output = OrbitKvArenaStats::default();
    let mut output_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_arena_stats(
                handle,
                &mut output,
                1,
                &mut output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(output_count, 1);
    output
}

#[test]
#[allow(clippy::too_many_lines)]
fn raw_abi8_packed_partial_fork_cow_append_is_receipt_atomic_and_reference_exact() {
    assert_eq!(crate::ORBITKV_ABI_VERSION, 8);
    assert_eq!(orbitkv_abi_version(), 8);

    let mut error = [0; 256];
    let handle = create_full(&mut error);
    let acquired = acquire(handle, 2, &mut error);
    let dense_prepared = prepare(handle, &acquired[..1], 48, &mut error);
    let dense_submitted = submit(handle, &dense_prepared, &mut error);
    let dense_completed = complete(handle, &dense_submitted, 1, &mut error);
    assert!(dense_completed.retirements.is_empty());
    let dense_source = published_views(&dense_completed.items)[0];

    let marked_source = mark_relocation_victims(handle, dense_source, &mut error);
    let source = relocate_to_partial_packed(handle, marked_source, &mut error);
    assert_eq!(source.boundary, 48);
    assert_eq!(source.resident_count, 2);

    let fork_input = OrbitKvRequestForkBatchItem {
        source_request: source.request,
        expected_source_head: source.snapshot,
        target_empty_request: acquired[1].request,
        expected_target_head: acquired[1].snapshot,
    };
    let mut forked = OrbitKvForkedBatchItem::default();
    let mut fork_pages = vec![OrbitKvSnapshotPage::default(); source.resident_count as usize];
    let (mut forked_count, mut fork_page_count) = (0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_request_fork_batch(
                handle,
                &fork_input,
                1,
                &mut forked,
                1,
                &mut forked_count,
                fork_pages.as_mut_ptr(),
                fork_pages.len() as u32,
                &mut fork_page_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!((forked_count, fork_page_count), (1, 2));
    assert_eq!(
        fork_pages
            .iter()
            .map(|page| (page.logical_ordinal, page.valid_token_count))
            .collect::<Vec<_>>(),
        vec![(0, 16), (1, 8)]
    );
    let old_packed_tail = fork_pages[1];

    let cow_prepared = prepare(handle, &[forked.target], 49, &mut error);
    assert_eq!(cow_prepared.classes.len(), 1);
    assert_eq!(cow_prepared.classes[0].flags, ABI8_CLASS_LOWERING_PACKED);
    assert_eq!(cow_prepared.tails.len(), 1);
    assert_eq!(cow_prepared.tails[0].kind, ORBITKV_TAIL_COPY_ON_WRITE);
    assert_eq!(cow_prepared.tails[0].valid_token_count, 8);
    assert_eq!(cow_prepared.tails[0].source, old_packed_tail.page);
    assert_eq!(cow_prepared.copies.len(), 1);
    assert_eq!(cow_prepared.copies[0].token_count, 8);
    assert_eq!(cow_prepared.copies[0].source_token_offset, 0);
    assert_eq!(cow_prepared.copies[0].destination_token_offset, 0);
    assert_eq!(cow_prepared.copies[0].source, old_packed_tail.page);
    assert!(cow_prepared.writes.is_empty());

    let (mut malformed_items, bind_receipts, copy_receipts) =
        submission_payload(handle, &cow_prepared, &mut error);
    assert_eq!(copy_receipts.len(), 1);
    assert_eq!(copy_receipts[0].token_count, 8);
    malformed_items[0].copy_receipt_offset = 1;
    let before_malformed = stats(handle, &mut error);
    let mut malformed_output = OrbitKvSubmittedBatchItem::default();
    let mut malformed_output_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_submit_batch(
                handle,
                malformed_items.as_ptr(),
                malformed_items.len() as u32,
                bind_receipts.as_ptr(),
                bind_receipts.len() as u32,
                copy_receipts.as_ptr(),
                copy_receipts.len() as u32,
                &mut malformed_output,
                1,
                &mut malformed_output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(handle, &mut error), before_malformed);

    let cow_submitted = submit(handle, &cow_prepared, &mut error);
    let cow_completed = complete(handle, &cow_submitted, 3, &mut error);
    assert!(cow_completed.retirements.is_empty());
    assert_eq!(cow_completed.detached.len(), 1);
    assert_eq!(cow_completed.detached[0].old, old_packed_tail.page);
    assert_eq!(cow_completed.detached[0].logical_ordinal, 1);
    assert_eq!(cow_completed.detached[0].token_begin, 16);
    assert_eq!(cow_completed.detached[0].token_end_exclusive, 24);
    let cow_target = published_views(&cow_completed.items)[0];

    let append_prepared = prepare(handle, &[cow_target], 50, &mut error);
    assert_eq!(append_prepared.classes[0].flags, ABI8_CLASS_LOWERING_PACKED);
    assert_eq!(append_prepared.tails[0].kind, ORBITKV_TAIL_IN_PLACE);
    assert_eq!(append_prepared.tails[0].valid_token_count, 9);
    assert!(append_prepared.copies.is_empty());
    let append_submitted = submit(handle, &append_prepared, &mut error);
    let append_completed = complete(handle, &append_submitted, 4, &mut error);
    assert!(append_completed.detached.is_empty());
    assert!(append_completed.retirements.is_empty());
    let appended_target = published_views(&append_completed.items)[0];
    assert_eq!(appended_target.boundary, 50);

    let source_certificates = release(handle, &[source], &mut error);
    assert_eq!(source_certificates.len(), 1);
    let old_tail_certificate = source_certificates[0];
    assert_eq!(old_tail_certificate.page, old_packed_tail.page);
    assert_eq!(old_tail_certificate.logical_ordinal, 1);
    assert_eq!(old_tail_certificate.token_begin, 16);
    assert_eq!(old_tail_certificate.token_end_exclusive, 24);
    assert_eq!(old_tail_certificate.completion_domain, 4);
    assert_eq!(old_tail_certificate.completion_value, 3);
    acknowledge(handle, &source_certificates, &mut error);
    recycle_requests(handle, &[source], &mut error);

    let target_certificates = release(handle, &[appended_target], &mut error);
    assert_eq!(
        target_certificates
            .iter()
            .map(|certificate| (
                certificate.logical_ordinal,
                certificate.token_begin,
                certificate.token_end_exclusive,
            ))
            .collect::<Vec<_>>(),
        vec![(0, 0, 16), (1, 16, 26)]
    );
    assert_ne!(
        target_certificates[0].reclamation,
        target_certificates[1].reclamation
    );
    acknowledge(handle, &target_certificates, &mut error);
    recycle_requests(handle, &[appended_target], &mut error);

    let final_stats = stats(handle, &mut error);
    assert_eq!(final_stats.active_requests, 0);
    assert_eq!(final_stats.active_snapshots, 0);
    assert_eq!(final_stats.pending_reclamations, 0);
    assert_eq!(final_stats.total_request_page_refs, 0);
    assert_eq!(final_stats.total_prefix_page_refs, 0);
    assert_eq!(final_stats.total_reader_pins, 0);
    assert_eq!(final_stats.free_pages, 16);
    let arena = final_arena_stats(handle, &mut error);
    assert_eq!(arena.free_pages, 16);
    assert_eq!(arena.reserved_pages, 0);
    assert_eq!(arena.writing_pages, 0);
    assert_eq!(arena.active_pages, 0);
    assert_eq!(arena.retiring_pages, 0);
    assert_eq!(arena.quarantined_pages, 0);
    assert_eq!(arena.request_page_refs, 0);
    assert_eq!(arena.prefix_page_refs, 0);
    assert_eq!(arena.reader_pins, 0);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}
