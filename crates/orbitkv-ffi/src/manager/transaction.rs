use super::{
    BackendBindReceipt, BackendCopyReceipt, BackendUnobservedReceipt,
    ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_OK, OrbitKvBackendBindReceipt,
    OrbitKvBackendCopyReceipt, OrbitKvBackendUnobservedReceipt, OrbitKvBatchCompletionReceipt,
    OrbitKvClassLowering, OrbitKvCompleteBatchItem, OrbitKvCompletedBatchItem, OrbitKvCopyIntent,
    OrbitKvDetachedBinding, OrbitKvManagerHandle, OrbitKvPrepareBatchItem,
    OrbitKvPreparedBatchItem, OrbitKvReclamationCertificate, OrbitKvStepLease,
    OrbitKvSubmissionLease, OrbitKvSubmitBatchItem, OrbitKvSubmittedBatchItem, OrbitKvTailAction,
    OrbitKvWriteIntent, PrepareBatchItem, StepLease, SubmissionLease, SubmitBatchItem, c_char,
    checked_mul, core_error, exact_len, ffi_boundary, input_slice, invalid, lock_manager,
    manager_ref, maximum_batch, preflight_output, validate_count_limit, validate_nonzero_limit,
    validate_submit_spans, write_converted,
};

/// Reserves one append batch and returns only canonical delta spans.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_manager_prepare_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvPrepareBatchItem,
    item_count: u32,
    prepared: *mut OrbitKvPreparedBatchItem,
    prepared_capacity: u32,
    out_prepared_count: *mut u32,
    class_lowerings: *mut OrbitKvClassLowering,
    class_capacity: u32,
    out_class_count: *mut u32,
    tail_actions: *mut OrbitKvTailAction,
    tail_capacity: u32,
    out_tail_count: *mut u32,
    copy_intents: *mut OrbitKvCopyIntent,
    copy_capacity: u32,
    out_copy_count: *mut u32,
    write_intents: *mut OrbitKvWriteIntent,
    write_capacity: u32,
    out_write_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(item_count, maximum_batch(handle), "prepare item")?;
        let items = unsafe { input_slice(items, item_count, "prepare item") }?;
        let core_items = items
            .iter()
            .map(|item| {
                if item.reserved != 0 {
                    return invalid("prepare item reserved field must be zero");
                }
                Ok(PrepareBatchItem {
                    request: item.request.into(),
                    expected_head: item.expected_head.into(),
                    target_boundary: item.target_boundary,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let class_bound = checked_mul(item_count, handle.class_count, "class lowering")?;
        let tail_bound = class_bound;
        let copy_bound = class_bound.min(handle.total_page_capacity);
        let write_bound = checked_mul(
            item_count,
            handle.maximum_write_intents_per_item,
            "write intent",
        )?
        .min(handle.total_page_capacity);
        let shorts = [
            unsafe {
                preflight_output(
                    prepared,
                    prepared_capacity,
                    out_prepared_count,
                    item_count,
                    "prepared item",
                )?
            },
            unsafe {
                preflight_output(
                    class_lowerings,
                    class_capacity,
                    out_class_count,
                    class_bound,
                    "class lowering",
                )?
            },
            unsafe {
                preflight_output(
                    tail_actions,
                    tail_capacity,
                    out_tail_count,
                    tail_bound,
                    "tail action",
                )?
            },
            unsafe {
                preflight_output(
                    copy_intents,
                    copy_capacity,
                    out_copy_count,
                    copy_bound,
                    "copy intent",
                )?
            },
            unsafe {
                preflight_output(
                    write_intents,
                    write_capacity,
                    out_write_count,
                    write_bound,
                    "write intent",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let outputs = lock_manager(handle)?
            .prepare_batch(&core_items)
            .map_err(core_error)?;
        assert_eq!(outputs.len(), item_count as usize);
        let mut class_offset = 0_u32;
        let mut tail_offset = 0_u32;
        let mut copy_offset = 0_u32;
        let mut write_offset = 0_u32;
        for (index, output) in outputs.iter().enumerate() {
            let class_count = exact_len(output.class_lowerings.len());
            let tail_count = exact_len(output.tail_actions.len());
            let copy_count = exact_len(output.copy_intents.len());
            let write_count = exact_len(output.write_intents.len());
            assert!(class_offset + class_count <= class_bound);
            assert!(tail_offset + tail_count <= tail_bound);
            assert!(copy_offset + copy_count <= copy_bound);
            assert!(write_offset + write_count <= write_bound);
            unsafe {
                for (class_index, lowering) in output.class_lowerings.iter().copied().enumerate() {
                    class_lowerings
                        .add(class_offset as usize + class_index)
                        .write(OrbitKvClassLowering {
                            class_id: lowering.class_id,
                            flags: lowering.flags,
                            tail_offset: lowering
                                .tail_offset
                                .checked_add(tail_offset)
                                .expect("preflighted global tail span"),
                            tail_count: lowering.tail_count,
                            copy_offset: lowering
                                .copy_offset
                                .checked_add(copy_offset)
                                .expect("preflighted global copy span"),
                            copy_count: lowering.copy_count,
                            write_offset: lowering
                                .write_offset
                                .checked_add(write_offset)
                                .expect("preflighted global write span"),
                            write_count: lowering.write_count,
                            reserved: 0,
                        });
                }
                write_converted(&output.tail_actions, tail_actions.add(tail_offset as usize));
                write_converted(&output.copy_intents, copy_intents.add(copy_offset as usize));
                write_converted(
                    &output.write_intents,
                    write_intents.add(write_offset as usize),
                );
                prepared.add(index).write(OrbitKvPreparedBatchItem {
                    step: output.step.into(),
                    request: output.request.into(),
                    base_snapshot: output.base_snapshot.into(),
                    target_snapshot: output.target_snapshot.into(),
                    base_view_version: output.base_view_version.0,
                    target_view_version: output.target_view_version.0,
                    previous_boundary: output.previous_boundary,
                    target_boundary: output.target_boundary,
                    class_offset,
                    class_count,
                    tail_offset,
                    tail_count,
                    copy_offset,
                    copy_count,
                    write_offset,
                    write_count,
                });
            }
            class_offset += class_count;
            tail_offset += tail_count;
            copy_offset += copy_count;
            write_offset += write_count;
        }
        unsafe {
            out_class_count.write(class_offset);
            out_tail_count.write(tail_offset);
            out_copy_count.write(copy_offset);
            out_write_count.write(write_offset);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Submits a canonical bind/copy receipt partition.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_submit_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvSubmitBatchItem,
    item_count: u32,
    receipts: *const OrbitKvBackendBindReceipt,
    receipt_count: u32,
    copy_receipts: *const OrbitKvBackendCopyReceipt,
    copy_receipt_count: u32,
    submitted: *mut OrbitKvSubmittedBatchItem,
    submitted_capacity: u32,
    out_submitted_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(item_count, maximum_batch(handle), "submit item")?;
        validate_count_limit(receipt_count, handle.total_page_capacity, "bind receipt")?;
        validate_count_limit(
            copy_receipt_count,
            handle.total_page_capacity,
            "copy receipt",
        )?;
        let items = unsafe { input_slice(items, item_count, "submit item") }?;
        let receipts = unsafe { input_slice(receipts, receipt_count, "bind receipt") }?;
        let copy_receipts =
            unsafe { input_slice(copy_receipts, copy_receipt_count, "copy receipt") }?;
        validate_submit_spans(items, receipt_count, copy_receipt_count)?;
        if receipts.iter().any(|receipt| receipt.reserved != 0) {
            return invalid("bind receipt reserved field must be zero");
        }
        if copy_receipts
            .iter()
            .any(|receipt| receipt.reserved8 != 0 || receipt.reserved32 != 0)
        {
            return invalid("copy receipt reserved fields must be zero");
        }
        if unsafe {
            preflight_output(
                submitted,
                submitted_capacity,
                out_submitted_count,
                item_count,
                "submitted item",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core_items = items
            .iter()
            .copied()
            .map(SubmitBatchItem::from)
            .collect::<Vec<_>>();
        let core_receipts = receipts
            .iter()
            .copied()
            .map(BackendBindReceipt::from)
            .collect::<Vec<_>>();
        let core_copy_receipts = copy_receipts
            .iter()
            .copied()
            .map(BackendCopyReceipt::from)
            .collect::<Vec<_>>();
        let outputs = lock_manager(handle)?
            .submit_batch(&core_items, &core_receipts, &core_copy_receipts)
            .map_err(core_error)?;
        assert_eq!(outputs.len(), item_count as usize);
        for (index, output) in outputs.iter().enumerate() {
            unsafe {
                submitted.add(index).write(OrbitKvSubmittedBatchItem {
                    submission: output.submission.into(),
                    request: output.request.into(),
                    target_snapshot: output.target_snapshot.into(),
                });
            }
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Publishes a submission batch and returns detached mirror updates plus one
/// batch-global reclamation-certificate array.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_manager_complete_batch(
    manager: *mut OrbitKvManagerHandle,
    receipt: OrbitKvBatchCompletionReceipt,
    items: *const OrbitKvCompleteBatchItem,
    item_count: u32,
    completed: *mut OrbitKvCompletedBatchItem,
    completed_capacity: u32,
    out_completed_count: *mut u32,
    detached: *mut OrbitKvDetachedBinding,
    detached_capacity: u32,
    out_detached_count: *mut u32,
    retirements: *mut OrbitKvReclamationCertificate,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(item_count, maximum_batch(handle), "completion item")?;
        if receipt.reserved != 0 {
            return invalid("completion receipt reserved field must be zero");
        }
        let items = unsafe { input_slice(items, item_count, "completion item") }?;
        // Detached bindings are per request reference, so shared old pages can
        // appear once per item and must not be capped by physical page count.
        let detached_bound = checked_mul(
            item_count,
            handle.maximum_completion_outputs_per_item,
            "completion output",
        )?;
        // Certificates are batch-global and page-owned, hence physically
        // unique and safely capped by total page capacity.
        let retirement_bound = detached_bound.min(handle.total_page_capacity);
        let shorts = [
            unsafe {
                preflight_output(
                    completed,
                    completed_capacity,
                    out_completed_count,
                    item_count,
                    "completed item",
                )?
            },
            unsafe {
                preflight_output(
                    detached,
                    detached_capacity,
                    out_detached_count,
                    detached_bound,
                    "detached binding",
                )?
            },
            unsafe {
                preflight_output(
                    retirements,
                    retirement_capacity,
                    out_retirement_count,
                    retirement_bound,
                    "reclamation certificate",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let submissions = items
            .iter()
            .map(|item| item.submission.into())
            .collect::<Vec<_>>();
        let output = lock_manager(handle)?
            .complete_batch(receipt.into(), &submissions)
            .map_err(core_error)?;
        assert_eq!(output.completions.len(), item_count as usize);
        let mut detached_offset = 0_u32;
        for (index, completion) in output.completions.iter().enumerate() {
            let detached_count = exact_len(completion.detached.len());
            assert!(detached_offset + detached_count <= detached_bound);
            unsafe {
                write_converted(&completion.detached, detached.add(detached_offset as usize));
                completed.add(index).write(OrbitKvCompletedBatchItem {
                    submission: completion.submission.into(),
                    request: completion.request.into(),
                    detached_snapshot: completion.detached_snapshot.into(),
                    published_snapshot: completion.publication.snapshot.into(),
                    published_view_version: completion.publication.view_version.0,
                    published_boundary: completion.publication.boundary,
                    resident_count: completion.publication.resident_count,
                    detached_offset,
                    detached_count,
                    reserved: 0,
                });
            }
            detached_offset += detached_count;
        }
        let retirement_count = exact_len(output.retirements.len());
        assert!(retirement_count <= retirement_bound);
        unsafe {
            write_converted(&output.retirements, retirements);
            out_detached_count.write(detached_offset);
            out_retirement_count.write(retirement_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Aborts a prepared batch proven backend-unobserved.
///
/// # Safety
/// The receipt pointer must be readable for `receipt_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_abort_steps_batch(
    manager: *mut OrbitKvManagerHandle,
    receipts: *const OrbitKvBackendUnobservedReceipt,
    receipt_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(receipt_count, maximum_batch(handle), "abort receipt")?;
        let receipts = unsafe { input_slice(receipts, receipt_count, "abort receipt") }?;
        if receipts.iter().any(|receipt| receipt.reserved != 0) {
            return invalid("abort receipt reserved field must be zero");
        }
        let core = receipts
            .iter()
            .copied()
            .map(BackendUnobservedReceipt::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .abort_steps_batch(&core)
            .map_err(core_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Fail-stops a prepared batch after ambiguous lowering.
///
/// # Safety
/// The step pointer must be readable for `step_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_quarantine_steps_batch(
    manager: *mut OrbitKvManagerHandle,
    steps: *const OrbitKvStepLease,
    step_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(step_count, maximum_batch(handle), "step")?;
        let steps = unsafe { input_slice(steps, step_count, "step") }?;
        let core = steps
            .iter()
            .copied()
            .map(StepLease::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .quarantine_steps_batch(&core)
            .map_err(core_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Fail-stops a submitted batch after ambiguous execution.
///
/// # Safety
/// The submission pointer must be readable for `submission_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_quarantine_submissions_batch(
    manager: *mut OrbitKvManagerHandle,
    submissions: *const OrbitKvSubmissionLease,
    submission_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(submission_count, maximum_batch(handle), "submission")?;
        let submissions = unsafe { input_slice(submissions, submission_count, "submission") }?;
        let core = submissions
            .iter()
            .copied()
            .map(SubmissionLease::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .quarantine_submissions_batch(&core)
            .map_err(core_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}
