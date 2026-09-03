use std::collections::BTreeSet;
use std::ffi::c_char;

use orbitkv::kv_manager::PrefixSemanticKey;
use orbitkv::{EngineRequestId, RuntimeSessionError};

use super::{
    ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_FAIL_STOPPED, ORBITKV_STATUS_OK,
    OrbitKvSessionHandle, OrbitKvSessionPrefixPublishItem, OrbitKvSessionPublishedPrefixRelease,
    OrbitKvSessionReleaseId, PendingReleaseWire, copy_input, ensure_running, ensure_shared_cache,
    ffi_boundary, invalid, lock_state, preflight_output, session_error, session_ref,
    wire_prefix_id, wire_release_id,
};
use crate::wire::OrbitKvDetachedBinding;

fn mark_poisoned(state: &mut super::SessionState, error: RuntimeSessionError) -> (i32, String) {
    if matches!(error, RuntimeSessionError::SessionPoisoned(_)) {
        state.fail_stopped = true;
    }
    session_error(error)
}

/// Atomically publishes request roots as prefixes and starts request release.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_prefix_publish_release_batch(
    session: *mut OrbitKvSessionHandle,
    items: *const OrbitKvSessionPrefixPublishItem,
    item_count: u32,
    out_release_id: *mut OrbitKvSessionReleaseId,
    outputs: *mut OrbitKvSessionPublishedPrefixRelease,
    output_capacity: u32,
    out_output_count: *mut u32,
    detached: *mut OrbitKvDetachedBinding,
    detached_capacity: u32,
    out_detached_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_release_id.is_null() {
            return invalid("release id output is required");
        }
        unsafe { out_release_id.write(OrbitKvSessionReleaseId::default()) };
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        super::validate_nonzero_limit(
            item_count,
            handle.maximum_requests.min(handle.maximum_prefixes),
            "prefix publish-release item",
        )?;
        let items = unsafe { copy_input(items, item_count, "prefix publish-release item") }?;
        let core = items
            .iter()
            .map(|item| {
                (
                    EngineRequestId(item.request_id),
                    PrefixSemanticKey::from(item.key),
                )
            })
            .collect::<Vec<_>>();
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let resident_counts = core
            .iter()
            .map(|(request_id, _)| {
                state
                    .resident_counts
                    .get(request_id)
                    .copied()
                    .ok_or_else(|| {
                        (
                            super::ORBITKV_STATUS_RETRYABLE_CONFLICT,
                            format!("unknown engine request id {request_id:?}"),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let detached_bound = resident_counts
            .iter()
            .try_fold(0_u32, |sum, resident_count| {
                sum.checked_add(*resident_count).ok_or_else(|| {
                    (
                        super::ORBITKV_STATUS_INVALID_ARGUMENT,
                        "prefix publish-release detached bound overflows".to_owned(),
                    )
                })
            })?;
        let shorts = [
            unsafe {
                preflight_output(
                    outputs,
                    output_capacity,
                    out_output_count,
                    item_count,
                    "published prefix-release item",
                )?
            },
            unsafe {
                preflight_output(
                    detached,
                    detached_capacity,
                    out_detached_count,
                    detached_bound,
                    "prefix publish-release detached binding",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let plan = state
            .runtime
            .publish_prefix_and_release_batch(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        let release_id = plan.release_id;
        if release_id.session_epoch() != handle.session_epoch
            || release_id.sequence() == 0
            || state.releases.contains_key(&release_id)
            || plan.items.len() != item_count as usize
        {
            state.fail_stopped = true;
            return Err((
                ORBITKV_STATUS_FAIL_STOPPED,
                "prefix publish-release result identity or cardinality changed".into(),
            ));
        }
        let mut detached_offset = 0_u32;
        let mut prefix_ids = BTreeSet::new();
        let mut wire_outputs = Vec::with_capacity(plan.items.len());
        let mut wire_detached = Vec::with_capacity(detached_bound as usize);
        for (index, item) in plan.items.iter().enumerate() {
            let Ok(detached_count) = u32::try_from(item.detached.len()) else {
                state.fail_stopped = true;
                return Err((
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "prefix publish-release detached count exceeds uint32_t".into(),
                ));
            };
            let Some(detached_end) = detached_offset.checked_add(detached_count) else {
                state.fail_stopped = true;
                return Err((
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "prefix publish-release detached count overflowed".into(),
                ));
            };
            if detached_end > detached_bound
                || item.request_id != core[index].0
                || item.key != core[index].1
                || item.resident_count != resident_counts[index]
                || item.resident_count != detached_count
                || item.prefix_id.session_epoch() != handle.session_epoch
                || item.prefix_id.sequence() == 0
                || !prefix_ids.insert(item.prefix_id)
            {
                state.fail_stopped = true;
                return Err((
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "prefix publish-release result changed".into(),
                ));
            }
            wire_detached.extend(
                item.detached
                    .iter()
                    .copied()
                    .map(OrbitKvDetachedBinding::from),
            );
            wire_outputs.push(OrbitKvSessionPublishedPrefixRelease {
                request_id: item.request_id.0,
                prefix_id: wire_prefix_id(item.prefix_id),
                key: item.key.into(),
                resident_count: item.resident_count,
                detached_offset,
                detached_count,
                reserved: 0,
            });
            detached_offset = detached_end;
        }
        if detached_offset != detached_bound {
            state.fail_stopped = true;
            return Err((
                ORBITKV_STATUS_FAIL_STOPPED,
                "prefix publish-release detached count changed".into(),
            ));
        }
        let requests = core
            .iter()
            .map(|(request_id, _)| *request_id)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let previous = state.releases.insert(
            release_id,
            PendingReleaseWire {
                requests: requests.clone(),
                retirements: Box::new([]),
            },
        );
        if let Some(previous) = previous {
            state.releases.insert(release_id, previous);
            state.fail_stopped = true;
            return Err((
                ORBITKV_STATUS_FAIL_STOPPED,
                "prefix publish-release reused a live release id".into(),
            ));
        }
        for (index, value) in wire_outputs.iter().copied().enumerate() {
            unsafe { outputs.add(index).write(value) };
        }
        unsafe {
            if !wire_detached.is_empty() {
                std::ptr::copy_nonoverlapping(
                    wire_detached.as_ptr(),
                    detached,
                    wire_detached.len(),
                );
            }
            out_release_id.write(wire_release_id(release_id));
            out_detached_count.write(detached_offset);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}
