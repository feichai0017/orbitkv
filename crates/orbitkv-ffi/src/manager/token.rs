use orbitkv::kv_manager::{
    ClassTokenDispositionUpdate, TokenDispositionBatchItem, TokenDispositionKind,
};

use super::{
    ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_OK, OrbitKvClassTokenDispositionUpdate,
    OrbitKvManagerHandle, OrbitKvRequestView, OrbitKvTokenDispositionBatchItem,
    OrbitKvTokenPlacement, OrbitKvTokenView, OrbitKvTokenViewQuery, TokenViewQuery, c_char,
    core_error, exact_len, ffi_boundary, input_slice, invalid, lock_manager, manager_ref,
    maximum_batch, preflight_output, validate_nonzero_limit, write_converted,
};

/// Cold-materializes canonical token views.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_token_views_batch(
    manager: *mut OrbitKvManagerHandle,
    queries: *const OrbitKvTokenViewQuery,
    query_count: u32,
    views: *mut OrbitKvTokenView,
    view_capacity: u32,
    out_view_count: *mut u32,
    placements: *mut OrbitKvTokenPlacement,
    placement_capacity: u32,
    out_placement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(query_count, handle.maximum_requests, "token-view query")?;
        let queries = unsafe { input_slice(queries, query_count, "token-view query") }?;
        if queries
            .iter()
            .any(|query| query.reserved16 != 0 || query.reserved32 != 0)
        {
            return invalid("token-view query reserved fields must be zero");
        }
        let core_queries = queries
            .iter()
            .copied()
            .map(TokenViewQuery::from)
            .collect::<Vec<_>>();
        let outputs = lock_manager(handle)?
            .token_views_batch(&core_queries)
            .map_err(core_error)?;
        let total = outputs.iter().try_fold(0_u32, |count, view| {
            count
                .checked_add(exact_len(view.placements.len()))
                .ok_or_else(|| super::invalid_pair("token placement count overflows"))
        })?;
        let shorts = [
            unsafe {
                preflight_output(
                    views,
                    view_capacity,
                    out_view_count,
                    query_count,
                    "token view",
                )?
            },
            unsafe {
                preflight_output(
                    placements,
                    placement_capacity,
                    out_placement_count,
                    total,
                    "token placement",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let mut offset = 0_u32;
        for (index, output) in outputs.iter().enumerate() {
            let count = exact_len(output.placements.len());
            unsafe {
                write_converted(&output.placements, placements.add(offset as usize));
                views.add(index).write(OrbitKvTokenView {
                    view_version: output.version.0,
                    placement_offset: offset,
                    placement_count: count,
                    page_tokens: output.page_tokens,
                    class_id: output.class_id,
                    reserved16: 0,
                    reserved32: 0,
                });
            }
            offset += count;
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Publishes a canonical logical victim/proof batch without physical movement.
///
/// # Safety
/// Every pointer must reference its declared readable or writable capacity.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_mark_token_dispositions_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvTokenDispositionBatchItem,
    item_count: u32,
    updates: *const OrbitKvClassTokenDispositionUpdate,
    update_count: u32,
    outputs: *mut OrbitKvRequestView,
    output_capacity: u32,
    out_output_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        validate_nonzero_limit(item_count, maximum_batch(handle), "disposition item")?;
        let items = unsafe { input_slice(items, item_count, "disposition item") }?;
        let updates = unsafe { input_slice(updates, update_count, "disposition update") }?;
        let mut expected = 0_u32;
        for item in items {
            if item.update_offset != expected {
                return invalid("disposition spans must be canonical and gap-free");
            }
            expected = expected
                .checked_add(item.update_count)
                .ok_or_else(|| super::invalid_pair("disposition span overflows"))?;
        }
        if expected != update_count {
            return invalid("disposition spans must cover the flat update buffer");
        }
        if updates.iter().any(|update| {
            update.reserved16 != 0
                || update.reserved32 != 0
                || update.disposition.reserved16 != 0
                || update.disposition.reserved32 != 0
                || update.disposition.kind == TokenDispositionKind::Retained as u16
                || update.disposition.kind > TokenDispositionKind::PolicyEvicted as u16
        }) {
            return invalid("disposition update has an invalid kind or reserved field");
        }
        if unsafe {
            preflight_output(
                outputs,
                output_capacity,
                out_output_count,
                item_count,
                "disposition output",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core = items
            .iter()
            .map(|item| {
                let begin = item.update_offset as usize;
                let end = begin + item.update_count as usize;
                TokenDispositionBatchItem {
                    request: item.request.into(),
                    expected_snapshot: item.expected_snapshot.into(),
                    updates: updates[begin..end]
                        .iter()
                        .map(|update| ClassTokenDispositionUpdate {
                            class_id: update.class_id,
                            token_id: update.token_id,
                            disposition: update.disposition.into(),
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                }
            })
            .collect::<Vec<_>>();
        let values = lock_manager(handle)?
            .mark_token_dispositions_batch(&core)
            .map_err(core_error)?;
        unsafe { write_converted(&values, outputs) };
        Ok(ORBITKV_STATUS_OK)
    })
}
