#![allow(
    clippy::borrow_as_ptr,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use super::*;

const HYBRID_PLAN: &[u8] = br#"{
  "page_tokens": 16,
  "classes": [
    {"name":"full","layers":[0],"retention":"full","bytes_per_token_per_layer":128},
    {"name":"swa","layers":[1],"retention":"sliding","bytes_per_token_per_layer":128,"window_tokens":18}
  ]
}"#;

fn config() -> OrbitKvManagerConfig {
    OrbitKvManagerConfig {
        maximum_requests: 8,
        maximum_operations: 8,
        maximum_prefixes: 8,
        maximum_reclamations: 32,
        maximum_step_tokens: 64,
    }
}

fn backends() -> [OrbitKvBackendArenaRegistration; 2] {
    [
        OrbitKvBackendArenaRegistration {
            pool_id: 7,
            class_id: 0,
            backend_domain: 3,
            page_count: 16,
            reserved: 0,
            backend_base_index: 100,
        },
        OrbitKvBackendArenaRegistration {
            pool_id: 9,
            class_id: 1,
            backend_domain: 5,
            page_count: 16,
            reserved: 0,
            backend_base_index: 1000,
        },
    ]
}

fn create(error: &mut [c_char]) -> *mut OrbitKvManagerHandle {
    let mut handle = std::ptr::null_mut();
    let backends = backends();
    assert_eq!(
        unsafe {
            orbitkv_manager_create(
                HYBRID_PLAN.as_ptr(),
                HYBRID_PLAN.len(),
                &config(),
                backends.as_ptr(),
                backends.len() as u32,
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

fn stats(handle: *mut OrbitKvManagerHandle, error: &mut [c_char]) -> OrbitKvManagerStats {
    let mut output = OrbitKvManagerStats::default();
    assert_eq!(
        unsafe { orbitkv_manager_stats(handle, &mut output, error.as_mut_ptr(), error.len(),) },
        ORBITKV_STATUS_OK
    );
    output
}

fn acquire(
    handle: *mut OrbitKvManagerHandle,
    count: u32,
    error: &mut [c_char],
) -> Vec<OrbitKvRequestView> {
    let mut outputs = vec![OrbitKvRequestView::default(); count as usize];
    let mut output_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_request_acquire_batch(
                handle,
                count,
                outputs.as_mut_ptr(),
                count,
                &mut output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(output_count, count);
    outputs
}

struct PreparedBuffers {
    items: Vec<OrbitKvPreparedBatchItem>,
    classes: Vec<OrbitKvClassLowering>,
    tails: Vec<OrbitKvTailAction>,
    copies: Vec<OrbitKvCopyIntent>,
    writes: Vec<OrbitKvWriteIntent>,
}

fn prepare(
    handle: *mut OrbitKvManagerHandle,
    views: &[OrbitKvRequestView],
    target_boundary: u64,
    error: &mut [c_char],
) -> PreparedBuffers {
    let inputs = views
        .iter()
        .map(|view| OrbitKvPrepareBatchItem {
            request: view.request,
            expected_head: view.snapshot,
            target_boundary,
            reserved: 0,
        })
        .collect::<Vec<_>>();
    let count = inputs.len() as u32;
    let mut items = vec![OrbitKvPreparedBatchItem::default(); inputs.len()];
    let mut classes = vec![OrbitKvClassLowering::default(); inputs.len() * 2];
    let mut tails = vec![OrbitKvTailAction::default(); inputs.len() * 2];
    let mut copies = vec![OrbitKvCopyIntent::default(); inputs.len() * 2];
    let mut writes = vec![OrbitKvWriteIntent::default(); inputs.len() * 8];
    let (mut item_count, mut class_count, mut tail_count, mut copy_count, mut write_count) =
        (0, 0, 0, 0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_prepare_batch(
                handle,
                inputs.as_ptr(),
                count,
                items.as_mut_ptr(),
                items.len() as u32,
                &mut item_count,
                classes.as_mut_ptr(),
                classes.len() as u32,
                &mut class_count,
                tails.as_mut_ptr(),
                tails.len() as u32,
                &mut tail_count,
                copies.as_mut_ptr(),
                copies.len() as u32,
                &mut copy_count,
                writes.as_mut_ptr(),
                writes.len() as u32,
                &mut write_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(item_count, count);
    items.truncate(item_count as usize);
    classes.truncate(class_count as usize);
    tails.truncate(tail_count as usize);
    copies.truncate(copy_count as usize);
    writes.truncate(write_count as usize);
    PreparedBuffers {
        items,
        classes,
        tails,
        copies,
        writes,
    }
}

fn arena_identities(
    handle: *mut OrbitKvManagerHandle,
    error: &mut [c_char],
) -> Vec<OrbitKvArenaIdentity> {
    let mut output = vec![OrbitKvArenaIdentity::default(); 2];
    let mut count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_arena_identities(
                handle,
                output.as_mut_ptr(),
                output.len() as u32,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    output.truncate(count as usize);
    output
}

fn submission_payload(
    handle: *mut OrbitKvManagerHandle,
    prepared: &PreparedBuffers,
    error: &mut [c_char],
) -> (
    Vec<OrbitKvSubmitBatchItem>,
    Vec<OrbitKvBackendBindReceipt>,
    Vec<OrbitKvBackendCopyReceipt>,
) {
    let identities = arena_identities(handle, error);
    let mut receipts = Vec::new();
    let mut copy_receipts = Vec::new();
    let mut inputs = Vec::new();
    for item in &prepared.items {
        let receipt_offset = receipts.len() as u32;
        let copy_receipt_offset = copy_receipts.len() as u32;
        for lowering in &prepared.classes
            [item.class_offset as usize..(item.class_offset + item.class_count) as usize]
        {
            let identity = &identities[lowering.class_id as usize];
            for tail in &prepared.tails[lowering.tail_offset as usize
                ..(lowering.tail_offset + lowering.tail_count) as usize]
            {
                if tail.kind == ORBITKV_TAIL_COPY_ON_WRITE || tail.kind == ORBITKV_TAIL_FRESH {
                    receipts.push(OrbitKvBackendBindReceipt {
                        step: item.step,
                        page: tail.destination,
                        backend_domain: identity.backend_domain,
                        mapped: 1,
                        writable: 1,
                        reserved: 0,
                        backend_index: identity.backend_base_index
                            + u64::from(tail.destination.page_id - identity.first_page_id),
                    });
                }
            }
            for write in &prepared.writes[lowering.write_offset as usize
                ..(lowering.write_offset + lowering.write_count) as usize]
            {
                receipts.push(OrbitKvBackendBindReceipt {
                    step: item.step,
                    page: OrbitKvPageLease {
                        engine_epoch: identity.engine_epoch,
                        pool_epoch: identity.pool_epoch,
                        generation: write.page_generation,
                        page_id: write.page_id,
                        pool_id: identity.pool_id,
                    },
                    backend_domain: identity.backend_domain,
                    mapped: 1,
                    writable: 1,
                    reserved: 0,
                    backend_index: identity.backend_base_index
                        + u64::from(write.page_id - identity.first_page_id),
                });
            }
            for copy in &prepared.copies[lowering.copy_offset as usize
                ..(lowering.copy_offset + lowering.copy_count) as usize]
            {
                copy_receipts.push(OrbitKvBackendCopyReceipt {
                    step: item.step,
                    class_id: copy.class_id,
                    backend_domain: copy.backend_domain,
                    token_count: copy.token_count,
                    source_token_offset: copy.source_token_offset,
                    destination_token_offset: copy.destination_token_offset,
                    observed: 1,
                    copied: 1,
                    ordered_before_writes: 1,
                    reserved8: 0,
                    reserved32: 0,
                    source: copy.source,
                    destination: copy.destination,
                    source_backend_index: copy.source_backend_index,
                    destination_backend_index: copy.destination_backend_index,
                });
            }
        }
        inputs.push(OrbitKvSubmitBatchItem {
            step: item.step,
            receipt_offset,
            receipt_count: receipts.len() as u32 - receipt_offset,
            copy_receipt_offset,
            copy_receipt_count: copy_receipts.len() as u32 - copy_receipt_offset,
        });
    }
    (inputs, receipts, copy_receipts)
}

fn submit(
    handle: *mut OrbitKvManagerHandle,
    prepared: &PreparedBuffers,
    error: &mut [c_char],
) -> Vec<OrbitKvSubmittedBatchItem> {
    let (inputs, receipts, copy_receipts) = submission_payload(handle, prepared, error);
    let mut outputs = vec![OrbitKvSubmittedBatchItem::default(); inputs.len()];
    let mut count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_submit_batch(
                handle,
                inputs.as_ptr(),
                inputs.len() as u32,
                receipts.as_ptr(),
                receipts.len() as u32,
                copy_receipts.as_ptr(),
                copy_receipts.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(count as usize, outputs.len());
    outputs
}

fn assert_semantic_submit_fault_is_fail_stopped(
    mutate: impl FnOnce(&mut Vec<OrbitKvBackendBindReceipt>, &mut Vec<OrbitKvBackendCopyReceipt>),
) {
    let mut error = [0; 256];
    let handle = create(&mut error);
    let acquired = acquire(handle, 2, &mut error);
    let source_prepared = prepare(handle, &acquired[..1], 17, &mut error);
    let source_submitted = submit(handle, &source_prepared, &mut error);
    let source_completed = complete(handle, &source_submitted, 1, &mut error);
    assert!(source_completed.retirements.is_empty());
    let source = published_views(&source_completed.items)[0];
    let fork_input = OrbitKvRequestForkBatchItem {
        source_request: source.request,
        expected_source_head: source.snapshot,
        target_empty_request: acquired[1].request,
        expected_target_head: acquired[1].snapshot,
    };
    let mut forked = OrbitKvForkedBatchItem::default();
    let mut forked_count = 0;
    let mut page_count = 0;
    let mut pages = vec![OrbitKvSnapshotPage::default(); source.resident_count as usize];
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
    assert_eq!(forked_count, 1);
    assert_eq!(page_count, source.resident_count);
    let prepared = prepare(handle, &[forked.target], 18, &mut error);
    let (items, mut bind_receipts, mut copy_receipts) =
        submission_payload(handle, &prepared, &mut error);
    assert!(!bind_receipts.is_empty());
    assert!(!copy_receipts.is_empty());
    mutate(&mut bind_receipts, &mut copy_receipts);

    let before = stats(handle, &mut error);
    assert_eq!(before.prepared_steps, 1);
    assert_eq!(before.quarantined_pages, 0);
    let mut submitted = OrbitKvSubmittedBatchItem::default();
    let mut submitted_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_manager_submit_batch(
                handle,
                items.as_ptr(),
                items.len() as u32,
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
        ORBITKV_STATUS_FAIL_STOPPED
    );

    // FAIL_STOPPED is a known fail-closed mutation: stats remain readable
    // and prove the prepared operation was consumed into quarantine.
    let after = stats(handle, &mut error);
    assert_eq!(after.active_requests, before.active_requests);
    assert_eq!(after.prepared_steps, 0);
    assert_eq!(after.reserved_pages, 0);
    assert!(after.quarantined_pages > before.quarantined_pages);
    assert_eq!(
        unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

struct CompletedBuffers {
    items: Vec<OrbitKvCompletedBatchItem>,
    detached: Vec<OrbitKvDetachedBinding>,
    retirements: Vec<OrbitKvReclamationCertificate>,
}

fn complete(
    handle: *mut OrbitKvManagerHandle,
    submitted: &[OrbitKvSubmittedBatchItem],
    completion_value: u64,
    error: &mut [c_char],
) -> CompletedBuffers {
    let inputs = submitted
        .iter()
        .map(|item| OrbitKvCompleteBatchItem {
            submission: item.submission,
        })
        .collect::<Vec<_>>();
    let mut items = vec![OrbitKvCompletedBatchItem::default(); inputs.len()];
    let mut detached = vec![OrbitKvDetachedBinding::default(); inputs.len() * 32];
    let mut retirements = vec![OrbitKvReclamationCertificate::default(); 32];
    let (mut item_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_complete_batch(
                handle,
                OrbitKvBatchCompletionReceipt {
                    engine_epoch: submitted[0].submission.engine_epoch,
                    completion_domain: 4,
                    completion_value,
                    confirmed: 1,
                    reserved: 0,
                },
                inputs.as_ptr(),
                inputs.len() as u32,
                items.as_mut_ptr(),
                items.len() as u32,
                &mut item_count,
                detached.as_mut_ptr(),
                detached.len() as u32,
                &mut detached_count,
                retirements.as_mut_ptr(),
                retirements.len() as u32,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    items.truncate(item_count as usize);
    detached.truncate(detached_count as usize);
    retirements.truncate(retirement_count as usize);
    CompletedBuffers {
        items,
        detached,
        retirements,
    }
}

fn acknowledge(
    handle: *mut OrbitKvManagerHandle,
    certificates: &[OrbitKvReclamationCertificate],
    error: &mut [c_char],
) {
    if certificates.is_empty() {
        return;
    }
    let receipts = certificates
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
    assert_eq!(
        unsafe {
            orbitkv_manager_acknowledge_reclamations_batch(
                handle,
                receipts.as_ptr(),
                receipts.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
}

fn release(
    handle: *mut OrbitKvManagerHandle,
    views: &[OrbitKvRequestView],
    error: &mut [c_char],
) -> Vec<OrbitKvReclamationCertificate> {
    let inputs = views
        .iter()
        .map(|view| OrbitKvReleaseBatchItem {
            request: view.request,
            expected_head: view.snapshot,
        })
        .collect::<Vec<_>>();
    let mut outputs = vec![OrbitKvReleasedBatchItem::default(); inputs.len()];
    let mut detached = vec![OrbitKvDetachedBinding::default(); inputs.len() * 32];
    let mut certificates = vec![OrbitKvReclamationCertificate::default(); 32];
    let (mut output_count, mut detached_count, mut certificate_count) = (0, 0, 0);
    assert_eq!(
        unsafe {
            orbitkv_manager_release_batch(
                handle,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
                &mut output_count,
                detached.as_mut_ptr(),
                detached.len() as u32,
                &mut detached_count,
                certificates.as_mut_ptr(),
                certificates.len() as u32,
                &mut certificate_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(output_count as usize, inputs.len());
    certificates.truncate(certificate_count as usize);
    certificates
}

fn recycle_requests(
    handle: *mut OrbitKvManagerHandle,
    views: &[OrbitKvRequestView],
    error: &mut [c_char],
) {
    let requests = views.iter().map(|view| view.request).collect::<Vec<_>>();
    assert_eq!(
        unsafe {
            orbitkv_manager_recycle_requests_batch(
                handle,
                requests.as_ptr(),
                requests.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
}

fn published_views(completed: &[OrbitKvCompletedBatchItem]) -> Vec<OrbitKvRequestView> {
    completed
        .iter()
        .map(|item| OrbitKvRequestView {
            request: item.request,
            snapshot: item.published_snapshot,
            view_version: item.published_view_version,
            boundary: item.published_boundary,
            resident_count: item.resident_count,
            reserved: 0,
        })
        .collect()
}

mod failures;
mod lifecycle;
mod preflight;
