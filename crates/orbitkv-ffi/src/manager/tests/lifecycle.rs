use super::*;

#[test]
fn b4_full_hybrid_lifecycle_has_global_certificates_and_zero_refs() {
    let mut error = [0; 256];
    let handle = create(&mut error);
    let acquired = acquire(handle, 4, &mut error);
    assert!(acquired.iter().all(|view| view.snapshot.engine_epoch != 0));

    let prepared = prepare(handle, &acquired, 49, &mut error);
    assert_eq!(prepared.items.len(), 4);
    assert_eq!(prepared.classes.len(), 8);
    assert_eq!(prepared.tails.len(), 8);
    assert!(prepared.classes.iter().all(|class| class.tail_count == 1));
    let submitted = submit(handle, &prepared, &mut error);
    let completed = complete(handle, &submitted, 11, &mut error);
    assert_eq!(completed.items.len(), 4);
    assert!(
        completed
            .items
            .windows(2)
            .all(|items| items[0].detached_offset + items[0].detached_count
                == items[1].detached_offset)
    );
    assert_eq!(
        completed
            .items
            .iter()
            .map(|item| item.detached_count)
            .sum::<u32>(),
        completed.detached.len() as u32
    );
    acknowledge(handle, &completed.retirements, &mut error);

    let current = published_views(&completed.items);
    let release_certificates = release(handle, &current, &mut error);
    acknowledge(handle, &release_certificates, &mut error);
    recycle_requests(handle, &current, &mut error);
    let final_stats = stats(handle, &mut error);
    assert_eq!(final_stats.active_requests, 0);
    assert_eq!(final_stats.active_snapshots, 0);
    assert_eq!(final_stats.pending_reclamations, 0);
    assert_eq!(final_stats.total_request_page_refs, 0);
    assert_eq!(final_stats.total_prefix_page_refs, 0);
    assert_eq!(final_stats.total_reader_pins, 0);
    assert_eq!(final_stats.free_pages, 32);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn fork_cold_recipe_drives_cow_and_copy_receipt_fault_is_precommit() {
    let mut error = [0; 256];
    let handle = create(&mut error);
    let initial = acquire(handle, 2, &mut error);
    let source_prepared = prepare(handle, &initial[..1], 17, &mut error);
    let source_submitted = submit(handle, &source_prepared, &mut error);
    let source_completed = complete(handle, &source_submitted, 1, &mut error);
    assert!(source_completed.retirements.is_empty());
    let source = published_views(&source_completed.items)[0];

    let fork_input = OrbitKvRequestForkBatchItem {
        source_request: source.request,
        expected_source_head: source.snapshot,
        target_empty_request: initial[1].request,
        expected_target_head: initial[1].snapshot,
    };
    let mut forked = OrbitKvForkedBatchItem::default();
    let (mut forked_count, mut page_count) = (0, 0);
    let before_short = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_request_fork_batch(
                handle,
                &fork_input,
                1,
                &mut forked,
                1,
                &mut forked_count,
                std::ptr::null_mut(),
                0,
                &mut page_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(forked_count, 1);
    assert_eq!(page_count, source.resident_count);
    assert_eq!(stats(handle, &mut error), before_short);

    let mut pages = vec![OrbitKvSnapshotPage::default(); page_count as usize];
    assert_eq!(
        unsafe {
            orbitkv_manager_request_fork_batch(
                handle,
                &fork_input,
                1,
                &mut forked,
                1,
                &mut forked_count,
                pages.as_mut_ptr(),
                pages.len() as u32,
                &mut page_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(forked.page_count, source.resident_count);
    assert_eq!(forked.target.boundary, 17);
    assert_ne!(forked.target.snapshot, initial[1].snapshot);
    assert!(pages.iter().all(|page| page.valid_token_count > 0));

    let target_prepared = prepare(handle, &[forked.target], 18, &mut error);
    assert_eq!(target_prepared.copies.len(), 2);
    assert!(
        target_prepared
            .tails
            .iter()
            .all(|tail| tail.kind == ORBITKV_TAIL_COPY_ON_WRITE)
    );
    let (submit_items, bind_receipts, mut copy_receipts) =
        submission_payload(handle, &target_prepared, &mut error);
    copy_receipts[0].reserved8 = 1;
    let mut submitted = OrbitKvSubmittedBatchItem::default();
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
                &mut submitted,
                1,
                &mut submitted_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    let after_fault = stats(handle, &mut error);
    assert_eq!(after_fault.prepared_steps, 1);
    assert_eq!(after_fault.quarantined_pages, 0);

    let target_submitted = submit(handle, &target_prepared, &mut error);
    let target_completed = complete(handle, &target_submitted, 2, &mut error);
    acknowledge(handle, &target_completed.retirements, &mut error);
    let target = published_views(&target_completed.items)[0];
    let release_certificates = release(handle, &[source, target], &mut error);
    acknowledge(handle, &release_certificates, &mut error);
    recycle_requests(handle, &[source, target], &mut error);
    assert_eq!(stats(handle, &mut error).free_pages, 32);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn b4_shared_retention_and_cow_detachments_can_exceed_physical_pages() {
    const DETACHED_RETENTION: u16 = 1;
    const DETACHED_COPY_ON_WRITE: u16 = 2;
    const SLIDING_PLAN: &[u8] = br#"{
      "page_tokens": 16,
      "classes": [
        {"name":"swa","layers":[0],"retention":"sliding","bytes_per_token_per_layer":128,"window_tokens":19}
      ]
    }"#;
    let mut error = [0; 256];
    let config = OrbitKvManagerConfig {
        maximum_requests: 5,
        maximum_operations: 4,
        maximum_prefixes: 4,
        maximum_reclamations: 7,
        maximum_step_tokens: 33,
    };
    let backend = OrbitKvBackendArenaRegistration {
        pool_id: 7,
        class_id: 0,
        backend_domain: 3,
        page_count: 7,
        reserved: 0,
        backend_base_index: 100,
    };
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            orbitkv_manager_create(
                SLIDING_PLAN.as_ptr(),
                SLIDING_PLAN.len(),
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

    let initial = acquire(handle, 5, &mut error);
    let source_prepared = prepare(handle, &initial[..1], 33, &mut error);
    let source_submitted = submit(handle, &source_prepared, &mut error);
    let source_completed = complete(handle, &source_submitted, 1, &mut error);
    acknowledge(handle, &source_completed.retirements, &mut error);
    let source = published_views(&source_completed.items)[0];
    assert_eq!(source.resident_count, 3);

    let fork_inputs = initial[1..]
        .iter()
        .map(|target| OrbitKvRequestForkBatchItem {
            source_request: source.request,
            expected_source_head: source.snapshot,
            target_empty_request: target.request,
            expected_target_head: target.snapshot,
        })
        .collect::<Vec<_>>();
    let mut forked = vec![OrbitKvForkedBatchItem::default(); 4];
    let mut pages = vec![OrbitKvSnapshotPage::default(); 12];
    let (mut forked_count, mut page_count) = (0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_request_fork_batch(
                handle,
                fork_inputs.as_ptr(),
                fork_inputs.len() as u32,
                forked.as_mut_ptr(),
                forked.len() as u32,
                &mut forked_count,
                pages.as_mut_ptr(),
                pages.len() as u32,
                &mut page_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!((forked_count, page_count), (4, 12));

    let source_release = release(handle, &[source], &mut error);
    assert!(source_release.is_empty());
    recycle_requests(handle, &[source], &mut error);
    let shared = forked.iter().map(|item| item.target).collect::<Vec<_>>();
    let prepared = prepare(handle, &shared, 34, &mut error);
    assert_eq!(prepared.copies.len(), 4);
    let submitted = submit(handle, &prepared, &mut error);
    let completed = complete(handle, &submitted, 2, &mut error);

    // T=7 physical pages, yet each of the four requests detaches the same
    // retention root and COW tail: eight per-request mirror updates.
    assert_eq!(completed.detached.len(), 8);
    assert!(completed.detached.len() > backend.page_count as usize);
    assert!(completed.items.iter().all(|item| item.detached_count == 2));
    assert_eq!(
        completed
            .detached
            .iter()
            .filter(|binding| binding.reason == DETACHED_RETENTION)
            .count(),
        4
    );
    assert_eq!(
        completed
            .detached
            .iter()
            .filter(|binding| binding.reason == DETACHED_COPY_ON_WRITE)
            .count(),
        4
    );
    assert_eq!(
        completed
            .detached
            .iter()
            .map(|binding| binding.old)
            .collect::<BTreeSet<_>>()
            .len(),
        2
    );

    acknowledge(handle, &completed.retirements, &mut error);
    let current = published_views(&completed.items);
    let release_certificates = release(handle, &current, &mut error);
    acknowledge(handle, &release_certificates, &mut error);
    recycle_requests(handle, &current, &mut error);
    assert_eq!(stats(handle, &mut error).free_pages, 7);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn prefix_attach_publish_release_evict_and_recycle_are_reference_exact() {
    let mut error = [0; 256];
    let handle = create(&mut error);
    let initial = acquire(handle, 3, &mut error);
    let sources = [initial[0], initial[2]];
    let prepared = prepare(handle, &sources, 32, &mut error);
    let submitted = submit(handle, &prepared, &mut error);
    let completed = complete(handle, &submitted, 7, &mut error);
    assert!(completed.retirements.is_empty());
    let current = published_views(&completed.items);
    let key_a = OrbitKvPrefixSemanticKey {
        namespace: [1; 32],
        digest: [2; 32],
        boundary: 32,
    };
    let key_b = OrbitKvPrefixSemanticKey {
        namespace: [1; 32],
        digest: [3; 32],
        boundary: 32,
    };

    let publish_a = OrbitKvPrefixPublishBatchItem {
        request: current[0].request,
        expected_head: current[0].snapshot,
        key: key_a,
    };
    let mut prefix_a = OrbitKvPublishedPrefix::default();
    let mut published_count = 0;
    let before_publish_short = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_publish_batch(
                handle,
                &publish_a,
                1,
                std::ptr::null_mut(),
                0,
                &mut published_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(published_count, 1);
    assert_eq!(stats(handle, &mut error), before_publish_short);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_publish_batch(
                handle,
                &publish_a,
                1,
                &mut prefix_a,
                1,
                &mut published_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(prefix_a.resident_count, current[0].resident_count);
    assert_eq!(stats(handle, &mut error).total_prefix_page_refs, 4);

    let mut hint = OrbitKvPrefixLookupHint::default();
    let mut hint_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_lookup_batch(
                handle,
                &key_a,
                1,
                &mut hint,
                1,
                &mut hint_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(hint.candidate_present, 1);
    let attach_input = OrbitKvPrefixAttachBatchItem {
        request: initial[1].request,
        expected_empty_head: initial[1].snapshot,
        hint,
    };
    let mut attached = OrbitKvAttachedPrefixBatchItem::default();
    let (mut attached_count, mut page_count) = (0, 0);
    let before_short = stats(handle, &mut error);
    let mut reserved_attach = attach_input;
    reserved_attach.hint.reserved = 1;
    let mut reserved_pages = vec![OrbitKvSnapshotPage::default(); hint.resident_count as usize];
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_attach_batch(
                handle,
                &reserved_attach,
                1,
                &mut attached,
                1,
                &mut attached_count,
                reserved_pages.as_mut_ptr(),
                reserved_pages.len() as u32,
                &mut page_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(handle, &mut error), before_short);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_attach_batch(
                handle,
                &attach_input,
                1,
                &mut attached,
                1,
                &mut attached_count,
                std::ptr::null_mut(),
                0,
                &mut page_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(page_count, hint.resident_count);
    assert_eq!(stats(handle, &mut error), before_short);
    let mut pages = vec![OrbitKvSnapshotPage::default(); page_count as usize];
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_attach_batch(
                handle,
                &attach_input,
                1,
                &mut attached,
                1,
                &mut attached_count,
                pages.as_mut_ptr(),
                pages.len() as u32,
                &mut page_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(attached.target.boundary, 32);
    assert_eq!(attached.page_count, 4);

    let publish_b = OrbitKvPrefixPublishBatchItem {
        request: current[1].request,
        expected_head: current[1].snapshot,
        key: key_b,
    };
    let mut transfer = OrbitKvPrefixPublishReleaseBatchItem::default();
    let mut transfer_detached = vec![OrbitKvDetachedBinding::default(); 32];
    let (mut transfer_count, mut transfer_detached_count, mut transfer_retirement_count) =
        (0, 0, u32::MAX);
    let before_transfer_short = stats(handle, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_publish_release_batch(
                handle,
                &publish_b,
                1,
                &mut transfer,
                1,
                &mut transfer_count,
                std::ptr::null_mut(),
                0,
                &mut transfer_detached_count,
                std::ptr::null_mut(),
                0,
                &mut transfer_retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(transfer_detached_count, current[1].resident_count);
    assert_eq!(transfer_retirement_count, 0);
    assert_eq!(stats(handle, &mut error), before_transfer_short);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_publish_release_batch(
                handle,
                &publish_b,
                1,
                &mut transfer,
                1,
                &mut transfer_count,
                transfer_detached.as_mut_ptr(),
                transfer_detached.len() as u32,
                &mut transfer_detached_count,
                std::ptr::null_mut(),
                0,
                &mut transfer_retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(transfer.detached_count, current[1].resident_count);
    assert_eq!(transfer_retirement_count, 0);

    let release_certificates = release(handle, &[current[0], attached.target], &mut error);
    assert!(release_certificates.is_empty());
    recycle_requests(
        handle,
        &[current[0], attached.target, current[1]],
        &mut error,
    );
    let before_evict = stats(handle, &mut error);
    assert_eq!(before_evict.active_prefixes, 2);
    assert_eq!(before_evict.total_request_page_refs, 0);
    assert_eq!(before_evict.total_prefix_page_refs, 8);

    let prefixes = [prefix_a.prefix, transfer.publication.prefix];
    let mut evicted = [OrbitKvEvictedPrefix::default(); 2];
    let mut certificates = vec![OrbitKvReclamationCertificate::default(); 32];
    let (mut evicted_count, mut certificate_count) = (0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_evict_batch(
                handle,
                prefixes.as_ptr(),
                2,
                evicted.as_mut_ptr(),
                2,
                &mut evicted_count,
                std::ptr::null_mut(),
                0,
                &mut certificate_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(certificate_count, 32);
    assert_eq!(stats(handle, &mut error), before_evict);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_evict_batch(
                handle,
                prefixes.as_ptr(),
                2,
                evicted.as_mut_ptr(),
                2,
                &mut evicted_count,
                certificates.as_mut_ptr(),
                certificates.len() as u32,
                &mut certificate_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    certificates.truncate(certificate_count as usize);
    assert_eq!(certificates.len(), 8);
    acknowledge(handle, &certificates, &mut error);
    assert_eq!(
        unsafe {
            orbitkv_manager_prefix_recycle_batch(
                handle,
                prefixes.as_ptr(),
                2,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let final_stats = stats(handle, &mut error);
    assert_eq!(final_stats.active_requests, 0);
    assert_eq!(final_stats.active_snapshots, 0);
    assert_eq!(final_stats.active_prefixes, 0);
    assert_eq!(final_stats.evicted_prefixes, 0);
    assert_eq!(final_stats.free_pages, 32);
    assert_eq!(final_stats.total_prefix_page_refs, 0);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}
