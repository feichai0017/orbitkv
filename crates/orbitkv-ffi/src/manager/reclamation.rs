use super::{
    KvManagerError, ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_OK, OrbitKvDetachedBinding,
    OrbitKvManagerHandle, OrbitKvReclamationCertificate, OrbitKvReclamationReceipt,
    OrbitKvReleaseBatchItem, OrbitKvReleasedBatchItem, OrbitKvRequestLease, ReclamationReceipt,
    ReleaseBatchItem, RequestLease, c_char, core_error, exact_len, ffi_boundary, input_slice,
    invalid, invalid_pair, lock_manager, manager_ref, preflight_output, validate_nonzero_limit,
    write_converted,
};

/// Releases quiescent requests using expected-head CAS.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_release_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvReleaseBatchItem,
    item_count: u32,
    released: *mut OrbitKvReleasedBatchItem,
    released_capacity: u32,
    out_released_count: *mut u32,
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
        validate_nonzero_limit(item_count, handle.maximum_requests, "release item")?;
        let items = unsafe { input_slice(items, item_count, "release item") }?;
        let core_items = items
            .iter()
            .copied()
            .map(ReleaseBatchItem::from)
            .collect::<Vec<_>>();
        let requests = core_items
            .iter()
            .map(|item| item.request)
            .collect::<Vec<_>>();
        let mut manager = lock_manager(handle)?;
        let views = manager.request_views_batch(&requests).map_err(core_error)?;
        let mut detached_bound = 0_u32;
        for (item, view) in core_items.iter().zip(views.iter()) {
            if item.expected_head != view.snapshot {
                return Err(core_error(KvManagerError::StaleView));
            }
            detached_bound = detached_bound
                .checked_add(view.resident_count)
                .ok_or_else(|| invalid_pair("release resident count exceeds uint32_t"))?;
        }
        let retirement_bound = detached_bound.min(handle.total_page_capacity);
        let shorts = [
            unsafe {
                preflight_output(
                    released,
                    released_capacity,
                    out_released_count,
                    item_count,
                    "released item",
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
        let output = manager.release_batch(&core_items).map_err(core_error)?;
        assert_eq!(output.releases.len(), item_count as usize);
        let mut detached_offset = 0_u32;
        for (index, release) in output.releases.iter().enumerate() {
            let detached_count = exact_len(release.detached.len());
            assert!(detached_offset + detached_count <= detached_bound);
            unsafe {
                write_converted(&release.detached, detached.add(detached_offset as usize));
                released.add(index).write(OrbitKvReleasedBatchItem {
                    request: release.request.into(),
                    detached_snapshot: release.detached_snapshot.into(),
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

/// Acknowledges a complete reclamation receipt batch.
///
/// # Safety
/// The receipt pointer must be readable for `receipt_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_acknowledge_reclamations_batch(
    manager: *mut OrbitKvManagerHandle,
    receipts: *const OrbitKvReclamationReceipt,
    receipt_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(
            receipt_count,
            handle.total_page_capacity,
            "reclamation receipt",
        )?;
        let receipts = unsafe { input_slice(receipts, receipt_count, "reclamation receipt") }?;
        if receipts
            .iter()
            .any(|receipt| receipt.reserved8 != 0 || receipt.reserved32 != 0)
        {
            return invalid("reclamation receipt reserved fields must be zero");
        }
        let core = receipts
            .iter()
            .copied()
            .map(ReclamationReceipt::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .acknowledge_reclamations_batch(&core)
            .map_err(core_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Recycles fully released request identities.
///
/// # Safety
/// The request pointer must be readable for `request_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_recycle_requests_batch(
    manager: *mut OrbitKvManagerHandle,
    requests: *const OrbitKvRequestLease,
    request_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(request_count, handle.maximum_requests, "request")?;
        let requests = unsafe { input_slice(requests, request_count, "request") }?;
        let core = requests
            .iter()
            .copied()
            .map(RequestLease::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .recycle_requests_batch(&core)
            .map_err(core_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}
