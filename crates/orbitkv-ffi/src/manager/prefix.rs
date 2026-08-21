use super::{
    KvManagerError, ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_OK,
    OrbitKvAttachedPrefixBatchItem, OrbitKvDetachedBinding, OrbitKvEvictedPrefix,
    OrbitKvManagerHandle, OrbitKvPrefixAttachBatchItem, OrbitKvPrefixLease,
    OrbitKvPrefixLookupHint, OrbitKvPrefixPublishBatchItem, OrbitKvPrefixPublishReleaseBatchItem,
    OrbitKvPrefixSemanticKey, OrbitKvPublishedPrefix, OrbitKvReclamationCertificate,
    OrbitKvSnapshotPage, PrefixAttachItem, PrefixLease, PrefixLookupHint, PrefixPublishItem,
    PrefixSemanticKey, c_char, core_error, exact_len, ffi_boundary, input_slice, invalid_pair,
    lock_manager, manager_ref, preflight_output, validate_hint, validate_nonzero_limit,
    write_converted, write_snapshot_pages,
};

/// Returns non-owning, generation-checked prefix lookup hints.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_prefix_lookup_batch(
    manager: *mut OrbitKvManagerHandle,
    keys: *const OrbitKvPrefixSemanticKey,
    key_count: u32,
    hints: *mut OrbitKvPrefixLookupHint,
    hint_capacity: u32,
    out_hint_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(key_count, handle.maximum_prefixes, "prefix lookup key")?;
        let keys = unsafe { input_slice(keys, key_count, "prefix lookup key") }?;
        if unsafe {
            preflight_output(
                hints,
                hint_capacity,
                out_hint_count,
                key_count,
                "prefix lookup hint",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core_keys = keys
            .iter()
            .copied()
            .map(PrefixSemanticKey::from)
            .collect::<Vec<_>>();
        let outputs = lock_manager(handle)?
            .lookup_prefix_batch(&core_keys)
            .map_err(core_error)?;
        assert_eq!(outputs.len(), key_count as usize);
        for (index, output) in outputs.iter().copied().enumerate() {
            unsafe { hints.add(index).write(output.into()) };
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Attaches revalidated hints to empty requests and cold-materializes roots.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_manager_prefix_attach_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvPrefixAttachBatchItem,
    item_count: u32,
    attached: *mut OrbitKvAttachedPrefixBatchItem,
    attached_capacity: u32,
    out_attached_count: *mut u32,
    pages: *mut OrbitKvSnapshotPage,
    page_capacity: u32,
    out_page_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(
            item_count,
            handle.maximum_requests.min(handle.maximum_prefixes),
            "prefix attach item",
        )?;
        let items = unsafe { input_slice(items, item_count, "prefix attach item") }?;
        for item in items {
            validate_hint(&item.hint, true)?;
        }
        let mut manager = lock_manager(handle)?;
        let keys = items
            .iter()
            .map(|item| item.hint.key.into())
            .collect::<Vec<_>>();
        let authoritative = manager.lookup_prefix_batch(&keys).map_err(core_error)?;
        let mut required_pages = 0_u32;
        for (input, current) in items.iter().zip(authoritative.iter()) {
            let supplied: PrefixLookupHint = input.hint.into();
            if supplied != *current {
                return Err(core_error(KvManagerError::PrefixHintStale));
            }
            required_pages = required_pages
                .checked_add(current.resident_count)
                .ok_or_else(|| invalid_pair("prefix attach page count overflows uint32_t"))?;
        }
        let item_short = unsafe {
            preflight_output(
                attached,
                attached_capacity,
                out_attached_count,
                item_count,
                "attached prefix item",
            )?
        };
        let page_short = unsafe {
            preflight_output(
                pages,
                page_capacity,
                out_page_count,
                required_pages,
                "prefix materialization page",
            )?
        };
        if item_short || page_short {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core_items = items
            .iter()
            .copied()
            .map(PrefixAttachItem::from)
            .collect::<Vec<_>>();
        let outputs = manager
            .attach_prefix_batch(&core_items)
            .map_err(core_error)?;
        assert_eq!(outputs.len(), item_count as usize);
        let mut page_offset = 0_u32;
        for (index, output) in outputs.iter().enumerate() {
            let page_count = exact_len(output.target.pages.len());
            assert!(page_offset + page_count <= required_pages);
            unsafe {
                write_snapshot_pages(&output.target.pages, pages.add(page_offset as usize));
                attached.add(index).write(OrbitKvAttachedPrefixBatchItem {
                    prefix: output.prefix.into(),
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

/// Publishes page-aligned Full/Hybrid roots under semantic keys.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_prefix_publish_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvPrefixPublishBatchItem,
    item_count: u32,
    published: *mut OrbitKvPublishedPrefix,
    published_capacity: u32,
    out_published_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(
            item_count,
            handle.maximum_prefixes.min(handle.maximum_requests),
            "prefix publication item",
        )?;
        let items = unsafe { input_slice(items, item_count, "prefix publication item") }?;
        if unsafe {
            preflight_output(
                published,
                published_capacity,
                out_published_count,
                item_count,
                "published prefix",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core = items
            .iter()
            .copied()
            .map(PrefixPublishItem::from)
            .collect::<Vec<_>>();
        let outputs = lock_manager(handle)?
            .publish_prefix_batch(&core)
            .map_err(core_error)?;
        assert_eq!(outputs.len(), item_count as usize);
        for (index, output) in outputs.iter().copied().enumerate() {
            unsafe { published.add(index).write(output.into()) };
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Publishes roots and releases source requests in one reference transaction.
/// The batch-global certificate array is necessarily empty because ownership
/// transfers from request references to prefix references without retirement.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_manager_prefix_publish_release_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvPrefixPublishBatchItem,
    item_count: u32,
    outputs: *mut OrbitKvPrefixPublishReleaseBatchItem,
    output_capacity: u32,
    out_output_count: *mut u32,
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
        validate_nonzero_limit(
            item_count,
            handle.maximum_prefixes.min(handle.maximum_requests),
            "prefix publish-release item",
        )?;
        let items = unsafe { input_slice(items, item_count, "prefix publish-release item") }?;
        let core = items
            .iter()
            .copied()
            .map(PrefixPublishItem::from)
            .collect::<Vec<_>>();
        let requests = core.iter().map(|item| item.request).collect::<Vec<_>>();
        let mut manager = lock_manager(handle)?;
        let views = manager.request_views_batch(&requests).map_err(core_error)?;
        let mut detached_bound = 0_u32;
        for (item, view) in core.iter().zip(views.iter()) {
            if item.expected_head != view.snapshot {
                return Err(core_error(KvManagerError::StaleView));
            }
            detached_bound = detached_bound
                .checked_add(view.resident_count)
                .ok_or_else(|| {
                    invalid_pair("prefix publish-release resident count exceeds uint32_t")
                })?;
        }
        let shorts = [
            unsafe {
                preflight_output(
                    outputs,
                    output_capacity,
                    out_output_count,
                    item_count,
                    "prefix publish-release output",
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
                    0,
                    "reclamation certificate",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let result = manager
            .publish_prefix_and_release_batch(&core)
            .map_err(core_error)?;
        assert_eq!(result.len(), item_count as usize);
        let mut detached_offset = 0_u32;
        for (index, output) in result.iter().enumerate() {
            let detached_count = exact_len(output.release.detached.len());
            assert!(detached_offset + detached_count <= detached_bound);
            unsafe {
                write_converted(
                    &output.release.detached,
                    detached.add(detached_offset as usize),
                );
                outputs
                    .add(index)
                    .write(OrbitKvPrefixPublishReleaseBatchItem {
                        publication: output.publication.into(),
                        request: output.release.request.into(),
                        detached_snapshot: output.release.detached_snapshot.into(),
                        detached_offset,
                        detached_count,
                        reserved: 0,
                    });
            }
            detached_offset += detached_count;
        }
        unsafe {
            out_detached_count.write(detached_offset);
            out_retirement_count.write(0);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Evicts prefixes and emits batch-global page-owned certificates.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_prefix_evict_batch(
    manager: *mut OrbitKvManagerHandle,
    prefixes: *const OrbitKvPrefixLease,
    prefix_count: u32,
    evicted: *mut OrbitKvEvictedPrefix,
    evicted_capacity: u32,
    out_evicted_count: *mut u32,
    retirements: *mut OrbitKvReclamationCertificate,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(prefix_count, handle.maximum_prefixes, "prefix")?;
        let prefixes = unsafe { input_slice(prefixes, prefix_count, "prefix") }?;
        let shorts = [
            unsafe {
                preflight_output(
                    evicted,
                    evicted_capacity,
                    out_evicted_count,
                    prefix_count,
                    "evicted prefix",
                )?
            },
            unsafe {
                preflight_output(
                    retirements,
                    retirement_capacity,
                    out_retirement_count,
                    handle.total_page_capacity,
                    "reclamation certificate",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core = prefixes
            .iter()
            .copied()
            .map(PrefixLease::from)
            .collect::<Vec<_>>();
        let output = lock_manager(handle)?
            .evict_prefix_batch(&core)
            .map_err(core_error)?;
        assert_eq!(output.evicted.len(), prefix_count as usize);
        let retirement_count = exact_len(output.retirements.len());
        assert!(retirement_count <= handle.total_page_capacity);
        unsafe {
            write_converted(&output.evicted, evicted);
            write_converted(&output.retirements, retirements);
            out_retirement_count.write(retirement_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Recycles already-evicted prefix identities.
///
/// # Safety
/// The prefix pointer must be readable for `prefix_count` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_prefix_recycle_batch(
    manager: *mut OrbitKvManagerHandle,
    prefixes: *const OrbitKvPrefixLease,
    prefix_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(prefix_count, handle.maximum_prefixes, "prefix")?;
        let prefixes = unsafe { input_slice(prefixes, prefix_count, "prefix") }?;
        let core = prefixes
            .iter()
            .copied()
            .map(PrefixLease::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .recycle_prefixes_batch(&core)
            .map_err(core_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}
