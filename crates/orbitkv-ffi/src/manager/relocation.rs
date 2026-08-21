use orbitkv::kv_manager::{
    PrepareRelocationItem, RelocationCopyReceipt, RelocationLease, RelocationPolicy,
    RelocationUnobservedReceipt,
};

use super::{
    ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_OK, OrbitKvBatchCompletionReceipt,
    OrbitKvManagerHandle, OrbitKvPageLease, OrbitKvPrepareRelocationItem,
    OrbitKvPreparedRelocation, OrbitKvReclamationCertificate, OrbitKvRelocationCopyReceipt,
    OrbitKvRelocationLease, OrbitKvRelocationUnobservedReceipt, OrbitKvRequestView,
    OrbitKvSubmittedRelocation, OrbitKvTokenMove, c_char, checked_mul, core_error, exact_len,
    ffi_boundary, input_slice, invalid, lock_manager, manager_ref, maximum_batch, preflight_output,
    validate_count_limit, validate_nonzero_limit, write_converted,
};

/// Plans manager-owned full evacuation and returns exact flat move spans.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_manager_prepare_relocation_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvPrepareRelocationItem,
    item_count: u32,
    prepared: *mut OrbitKvPreparedRelocation,
    prepared_capacity: u32,
    out_prepared_count: *mut u32,
    sources: *mut OrbitKvPageLease,
    source_capacity: u32,
    out_source_count: *mut u32,
    destinations: *mut OrbitKvPageLease,
    destination_capacity: u32,
    out_destination_count: *mut u32,
    moves: *mut OrbitKvTokenMove,
    move_capacity: u32,
    out_move_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(item_count, maximum_batch(handle), "relocation item")?;
        let items = unsafe { input_slice(items, item_count, "relocation item") }?;
        if items.iter().any(|item| {
            item.reserved16 != 0
                || item.reserved32 != 0
                || item.policy.reserved8 != 0
                || item.policy.reserved32 != 0
                || item.policy.full_evacuation > 1
        }) {
            return invalid("relocation item has invalid reserved or boolean fields");
        }
        let page_bound = handle.total_page_capacity;
        let move_bound = checked_mul(
            handle.total_page_capacity,
            handle.page_tokens,
            "relocation move",
        )?;
        let shorts = [
            unsafe {
                preflight_output(
                    prepared,
                    prepared_capacity,
                    out_prepared_count,
                    item_count,
                    "prepared relocation",
                )?
            },
            unsafe {
                preflight_output(
                    sources,
                    source_capacity,
                    out_source_count,
                    page_bound,
                    "relocation source",
                )?
            },
            unsafe {
                preflight_output(
                    destinations,
                    destination_capacity,
                    out_destination_count,
                    page_bound,
                    "relocation destination",
                )?
            },
            unsafe {
                preflight_output(
                    moves,
                    move_capacity,
                    out_move_count,
                    move_bound,
                    "relocation move",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core = items
            .iter()
            .map(|item| PrepareRelocationItem {
                request: item.request.into(),
                expected_snapshot: item.expected_snapshot.into(),
                class_id: item.class_id,
                policy: RelocationPolicy {
                    fragmentation_threshold_milli: item.policy.fragmentation_threshold_milli,
                    maximum_source_pages: item.policy.maximum_source_pages,
                    evacuation_headroom_pages: item.policy.evacuation_headroom_pages,
                    full_evacuation: item.policy.full_evacuation == 1,
                },
            })
            .collect::<Vec<_>>();
        let outputs = lock_manager(handle)?
            .prepare_relocation_batch(&core)
            .map_err(core_error)?;
        let mut source_offset = 0_u32;
        let mut destination_offset = 0_u32;
        let mut move_offset = 0_u32;
        for (index, output) in outputs.iter().enumerate() {
            let source_count = exact_len(output.plan.source_pages.len());
            let destination_count = exact_len(output.plan.destination_pages.len());
            let move_count = exact_len(output.plan.moves.len());
            unsafe {
                write_converted(
                    &output.plan.source_pages,
                    sources.add(source_offset as usize),
                );
                write_converted(
                    &output.plan.destination_pages,
                    destinations.add(destination_offset as usize),
                );
                for (move_index, movement) in output.plan.moves.iter().enumerate() {
                    moves
                        .add(move_offset as usize + move_index)
                        .write(OrbitKvTokenMove {
                            token_id: movement.token_id,
                            source: movement.source.into(),
                            destination: movement.destination.into(),
                        });
                }
                prepared.add(index).write(OrbitKvPreparedRelocation {
                    relocation: output.relocation.into(),
                    request: output.request.into(),
                    base_snapshot: output.base_snapshot.into(),
                    target_snapshot: output.target_snapshot.into(),
                    base_view_version: output.plan.base_version.0,
                    target_view_version: output.plan.target_version.0,
                    source_offset,
                    source_count,
                    destination_offset,
                    destination_count,
                    move_offset,
                    move_count,
                    projected_reclaimed_pages: output.plan.projected_reclaimed_pages,
                    fragmentation_milli: output.plan.fragmentation_milli,
                    class_id: output.plan.class_id,
                    reserved32: 0,
                });
            }
            source_offset += source_count;
            destination_offset += destination_count;
            move_offset += move_count;
        }
        unsafe {
            out_source_count.write(source_offset);
            out_destination_count.write(destination_offset);
            out_move_count.write(move_offset);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Submits exact relocation copy receipts.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_submit_relocation_batch(
    manager: *mut OrbitKvManagerHandle,
    relocations: *const OrbitKvRelocationLease,
    relocation_count: u32,
    receipts: *const OrbitKvRelocationCopyReceipt,
    receipt_count: u32,
    submitted: *mut OrbitKvSubmittedRelocation,
    submitted_capacity: u32,
    out_submitted_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(relocation_count, maximum_batch(handle), "relocation")?;
        validate_count_limit(
            receipt_count,
            checked_mul(
                handle.total_page_capacity,
                handle.page_tokens,
                "relocation receipt",
            )?,
            "relocation receipt",
        )?;
        let relocations = unsafe { input_slice(relocations, relocation_count, "relocation") }?;
        let receipts = unsafe { input_slice(receipts, receipt_count, "relocation receipt") }?;
        if receipts
            .iter()
            .any(|receipt| receipt.reserved16 != 0 || receipt.reserved32 != 0)
        {
            return invalid("relocation receipt reserved fields must be zero");
        }
        if unsafe {
            preflight_output(
                submitted,
                submitted_capacity,
                out_submitted_count,
                relocation_count,
                "submitted relocation",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core_relocations = relocations
            .iter()
            .copied()
            .map(RelocationLease::from)
            .collect::<Vec<_>>();
        let core_receipts = receipts
            .iter()
            .map(|receipt| RelocationCopyReceipt {
                relocation: receipt.relocation.into(),
                token_id: receipt.token_id,
                source: receipt.source.into(),
                destination: receipt.destination.into(),
                observed: receipt.observed,
                copied: receipt.copied,
                reserved16: receipt.reserved16,
                reserved32: receipt.reserved32,
            })
            .collect::<Vec<_>>();
        let outputs = lock_manager(handle)?
            .submit_relocation_batch(&core_relocations, &core_receipts)
            .map_err(core_error)?;
        for (index, output) in outputs.iter().enumerate() {
            unsafe {
                submitted.add(index).write(OrbitKvSubmittedRelocation {
                    relocation: output.relocation.into(),
                    request: output.request.into(),
                    target_snapshot: output.target_snapshot.into(),
                });
            }
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Completes relocations at one confirmed CUDA completion point.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_complete_relocation_batch(
    manager: *mut OrbitKvManagerHandle,
    completion: *const OrbitKvBatchCompletionReceipt,
    relocations: *const OrbitKvRelocationLease,
    relocation_count: u32,
    publications: *mut OrbitKvRequestView,
    publication_capacity: u32,
    out_publication_count: *mut u32,
    retirements: *mut OrbitKvReclamationCertificate,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(relocation_count, maximum_batch(handle), "relocation")?;
        let completion = unsafe { super::required_ref(completion, "completion") }?;
        if completion.reserved != 0 {
            return invalid("completion reserved field must be zero");
        }
        let relocations = unsafe { input_slice(relocations, relocation_count, "relocation") }?;
        let shorts = [
            unsafe {
                preflight_output(
                    publications,
                    publication_capacity,
                    out_publication_count,
                    relocation_count,
                    "relocation publication",
                )?
            },
            unsafe {
                preflight_output(
                    retirements,
                    retirement_capacity,
                    out_retirement_count,
                    handle.total_page_capacity,
                    "relocation retirement",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core = relocations
            .iter()
            .copied()
            .map(RelocationLease::from)
            .collect::<Vec<_>>();
        let output = lock_manager(handle)?
            .complete_relocation_batch((*completion).into(), &core)
            .map_err(core_error)?;
        unsafe {
            write_converted(&output.publications, publications);
            write_converted(&output.retirements, retirements);
            out_retirement_count.write(exact_len(output.retirements.len()));
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Aborts backend-unobserved prepared relocation batches.
///
/// # Safety
/// Every pointer must reference its declared readable capacity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_abort_relocations_batch(
    manager: *mut OrbitKvManagerHandle,
    receipts: *const OrbitKvRelocationUnobservedReceipt,
    receipt_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(receipt_count, maximum_batch(handle), "relocation abort")?;
        let receipts = unsafe { input_slice(receipts, receipt_count, "relocation abort") }?;
        let core = receipts
            .iter()
            .map(|receipt| RelocationUnobservedReceipt {
                relocation: receipt.relocation.into(),
                backend_unobserved: receipt.backend_unobserved,
                reserved: receipt.reserved,
            })
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .abort_relocations_batch(&core)
            .map_err(core_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}
