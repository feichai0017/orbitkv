use super::{
    BTreeMap, BTreeSet, KvManagerError, ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_OK,
    OrbitKvForkedBatchItem, OrbitKvManagerHandle, OrbitKvRequestForkBatchItem, OrbitKvRequestView,
    OrbitKvSnapshotPage, RequestForkItem, c_char, core_error, ffi_boundary, input_slice,
    invalid_pair, lock_manager, manager_ref, preflight_output, validate_nonzero_limit,
    write_snapshot_pages,
};

/// Acquires independent empty snapshot heads for a request batch.
///
/// # Safety
/// Output pointers must be writable for their declared capacity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_request_acquire_batch(
    manager: *mut OrbitKvManagerHandle,
    request_count: u32,
    requests: *mut OrbitKvRequestView,
    request_capacity: u32,
    out_request_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(request_count, handle.maximum_requests, "request")?;
        if unsafe {
            preflight_output(
                requests,
                request_capacity,
                out_request_count,
                request_count,
                "request view",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let outputs = lock_manager(handle)?
            .acquire_requests_batch(request_count as usize)
            .map_err(core_error)?;
        assert_eq!(outputs.len(), request_count as usize);
        for (index, output) in outputs.iter().copied().enumerate() {
            unsafe { requests.add(index).write(output.into()) };
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Forks source snapshots into empty target requests and cold-materializes them.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_request_fork_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvRequestForkBatchItem,
    item_count: u32,
    forked: *mut OrbitKvForkedBatchItem,
    forked_capacity: u32,
    out_forked_count: *mut u32,
    pages: *mut OrbitKvSnapshotPage,
    page_capacity: u32,
    out_page_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(item_count, handle.maximum_requests, "fork item")?;
        let items = unsafe { input_slice(items, item_count, "fork item") }?;
        let core_items = items
            .iter()
            .copied()
            .map(RequestForkItem::from)
            .collect::<Vec<_>>();
        let mut manager = lock_manager(handle)?;
        let source_requests = core_items
            .iter()
            .map(|item| item.source_request)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let targets = core_items
            .iter()
            .map(|item| item.target_empty_request)
            .collect::<BTreeSet<_>>();
        if targets.len() != core_items.len()
            || source_requests
                .iter()
                .any(|source| targets.contains(source))
        {
            return Err(core_error(KvManagerError::DuplicateRequest));
        }
        let source_views = manager
            .request_views_batch(&source_requests)
            .map_err(core_error)?;
        let source_views = source_views
            .iter()
            .map(|view| (view.request, *view))
            .collect::<BTreeMap<_, _>>();
        let mut required_pages = 0_u32;
        for item in &core_items {
            let view = source_views
                .get(&item.source_request)
                .expect("queried every unique fork source");
            if view.snapshot != item.expected_source_head {
                return Err(core_error(KvManagerError::StaleView));
            }
            required_pages = required_pages
                .checked_add(view.resident_count)
                .ok_or_else(|| invalid_pair("fork page count overflows uint32_t"))?;
        }
        let item_short = unsafe {
            preflight_output(
                forked,
                forked_capacity,
                out_forked_count,
                item_count,
                "forked item",
            )?
        };
        let page_short = unsafe {
            preflight_output(
                pages,
                page_capacity,
                out_page_count,
                required_pages,
                "fork materialization page",
            )?
        };
        if item_short || page_short {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let outputs = manager
            .fork_requests_batch(&core_items)
            .map_err(core_error)?;
        assert_eq!(outputs.len(), item_count as usize);
        let mut page_offset = 0_u32;
        for (index, output) in outputs.iter().enumerate() {
            let page_count = u32::try_from(output.target.pages.len())
                .expect("preflighted fork page count fits ABI");
            assert!(page_offset + page_count <= required_pages);
            unsafe {
                write_snapshot_pages(&output.target.pages, pages.add(page_offset as usize));
                forked.add(index).write(OrbitKvForkedBatchItem {
                    source: output.source.into(),
                    target: output.target.view.into(),
                    page_offset,
                    page_count,
                });
            }
            page_offset += page_count;
        }
        assert_eq!(page_offset, required_pages);
        Ok(ORBITKV_STATUS_OK)
    })
}
