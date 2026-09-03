use std::ffi::c_char;

use orbitkv::kv_manager::PrefixSemanticKey;
use orbitkv::{
    EngineControlEvidence, EngineControlId, EngineControlOutcome, EngineControlPlan,
    EnginePendingAttachCancel, EnginePendingAttachCancelDisposition,
    EnginePendingAttachCancelOutcome, EnginePrefixId, EnginePrefixLookup, EngineRequestId,
    RuntimeSessionError,
};

use super::{
    ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION, ORBITKV_SESSION_CONTROL_KIND_PREFIX_EVICTION,
    ORBITKV_SESSION_CONTROL_OUTCOME_EVICTED, ORBITKV_SESSION_CONTROL_OUTCOME_MATERIALIZED,
    ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED,
    ORBITKV_SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING, ORBITKV_STATUS_BUFFER_TOO_SMALL,
    ORBITKV_STATUS_FAIL_STOPPED, ORBITKV_STATUS_OK, OrbitKvSessionControlEvidence,
    OrbitKvSessionControlId, OrbitKvSessionControlOutcome, OrbitKvSessionControlPlanInfo,
    OrbitKvSessionHandle, OrbitKvSessionMaterializedRequest, OrbitKvSessionPendingAttachCancel,
    OrbitKvSessionPendingAttachCancelOutcome, OrbitKvSessionPrefixAttachItem,
    OrbitKvSessionPrefixId, OrbitKvSessionPrefixLookup, OrbitKvSessionPrefixPublishItem,
    OrbitKvSessionPublishedPrefix, OrbitKvSessionRequestForkItem, OrbitKvSessionRetirement,
    OrbitKvSessionRetirementEvidence, copy_input, ensure_running, ensure_shared_cache, exact_len,
    ffi_boundary, invalid, lock_state, retryable, session_error, session_ref, validate_bool,
    validate_count_limit, validate_nonzero_limit, validate_retirement_evidence,
    validate_session_epoch, wire_retirement,
};
use crate::wire::{OrbitKvPrefixSemanticKey, OrbitKvSnapshotPage};

fn prefix_id(value: OrbitKvSessionPrefixId) -> EnginePrefixId {
    EnginePrefixId::from_parts(value.session_epoch, value.sequence)
}

fn control_id(value: OrbitKvSessionControlId) -> EngineControlId {
    EngineControlId::from_parts(value.session_epoch, value.sequence)
}

pub(super) fn wire_prefix_id(value: EnginePrefixId) -> OrbitKvSessionPrefixId {
    OrbitKvSessionPrefixId {
        session_epoch: value.session_epoch(),
        sequence: value.sequence(),
    }
}

fn wire_control_id(value: EngineControlId) -> OrbitKvSessionControlId {
    OrbitKvSessionControlId {
        session_epoch: value.session_epoch(),
        sequence: value.sequence(),
    }
}

fn pending_attach_cancel(value: OrbitKvSessionPendingAttachCancel) -> EnginePendingAttachCancel {
    EnginePendingAttachCancel {
        control_id: control_id(value.control_id),
        request_id: EngineRequestId(value.request_id),
        prefix_id: prefix_id(value.prefix_id),
        view_version: orbitkv::kv_manager::ViewVersion(value.view_version),
        boundary: value.boundary,
        resident_count: value.resident_count,
    }
}

fn pending_attach_cancel_outcome_disposition(value: EnginePendingAttachCancelDisposition) -> u32 {
    match value {
        EnginePendingAttachCancelDisposition::RecyclePending => {
            ORBITKV_SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING
        }
        EnginePendingAttachCancelDisposition::Finalized => {
            ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED
        }
    }
}

fn wire_pending_attach_cancel_outcome(
    value: EnginePendingAttachCancelOutcome,
) -> OrbitKvSessionPendingAttachCancelOutcome {
    OrbitKvSessionPendingAttachCancelOutcome {
        control_id: wire_control_id(value.control_id),
        request_id: value.request_id.0,
        prefix_id: wire_prefix_id(value.prefix_id),
        view_version: value.view_version.0,
        boundary: value.boundary,
        resident_count: value.resident_count,
        disposition: pending_attach_cancel_outcome_disposition(value.disposition),
    }
}

fn validate_operation_id(
    session_epoch: u64,
    sequence: u64,
    handle: &OrbitKvSessionHandle,
    label: &str,
) -> Result<(), (i32, String)> {
    validate_session_epoch(handle, session_epoch, label)?;
    if sequence == 0 {
        return retryable(&format!("{label} is unknown"));
    }
    Ok(())
}

fn mark_poisoned(state: &mut super::SessionState, error: RuntimeSessionError) -> (i32, String) {
    if matches!(error, RuntimeSessionError::SessionPoisoned(_)) {
        state.fail_stopped = true;
    }
    session_error(error)
}

fn cached_control_error(
    state: &mut super::SessionState,
    error: RuntimeSessionError,
) -> (i32, String) {
    if matches!(
        error,
        RuntimeSessionError::UnknownControl(_)
            | RuntimeSessionError::ForeignControl(_)
            | RuntimeSessionError::StaleControl(_)
            | RuntimeSessionError::ControlNotCommitted(_)
    ) {
        state.fail_stopped = true;
        return (
            ORBITKV_STATUS_FAIL_STOPPED,
            "cached control plan disagrees with runtime session state".into(),
        );
    }
    mark_poisoned(state, error)
}

fn plan_info(
    expected: EngineControlId,
    plan: &EngineControlPlan,
) -> Result<OrbitKvSessionControlPlanInfo, (i32, String)> {
    let (actual, kind, request_count, page_count, prefix_count, retirement_count) = match plan {
        EngineControlPlan::Materialization(plan) => {
            let request_count = u32::try_from(plan.requests.len()).map_err(|_| {
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "control request count exceeds uint32_t".into(),
                )
            })?;
            let mut page_count = 0_u32;
            for request in &plan.requests {
                let pages = u32::try_from(request.pages.len()).map_err(|_| {
                    (
                        ORBITKV_STATUS_FAIL_STOPPED,
                        "control page count exceeds uint32_t".into(),
                    )
                })?;
                if pages != request.resident_count {
                    return Err((
                        ORBITKV_STATUS_FAIL_STOPPED,
                        "materialization resident count does not match page count".into(),
                    ));
                }
                page_count = page_count.checked_add(pages).ok_or_else(|| {
                    (
                        ORBITKV_STATUS_FAIL_STOPPED,
                        "control page count exceeds uint32_t".into(),
                    )
                })?;
            }
            (
                plan.control_id,
                ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION,
                request_count,
                page_count,
                0,
                0,
            )
        }
        EngineControlPlan::PrefixEviction(plan) => (
            plan.control_id,
            ORBITKV_SESSION_CONTROL_KIND_PREFIX_EVICTION,
            0,
            0,
            u32::try_from(plan.prefixes.len()).map_err(|_| {
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "control prefix count exceeds uint32_t".into(),
                )
            })?,
            u32::try_from(plan.retirements.len()).map_err(|_| {
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "control retirement count exceeds uint32_t".into(),
                )
            })?,
        ),
    };
    if actual != expected {
        return Err((
            ORBITKV_STATUS_FAIL_STOPPED,
            "committed control plan identity changed".into(),
        ));
    }
    Ok(OrbitKvSessionControlPlanInfo {
        id: wire_control_id(actual),
        kind,
        reserved: 0,
        request_count,
        page_count,
        prefix_count,
        retirement_count,
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn preflight_plan_outputs(
    requests: *mut OrbitKvSessionMaterializedRequest,
    request_capacity: u32,
    out_request_count: *mut u32,
    request_count: u32,
    pages: *mut OrbitKvSnapshotPage,
    page_capacity: u32,
    out_page_count: *mut u32,
    page_count: u32,
    prefixes: *mut OrbitKvSessionPrefixId,
    prefix_capacity: u32,
    out_prefix_count: *mut u32,
    prefix_count: u32,
    retirements: *mut OrbitKvSessionRetirement,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    retirement_count: u32,
) -> Result<bool, (i32, String)> {
    for (pointer, label) in [
        (out_request_count, "materialized request"),
        (out_page_count, "materialization page"),
        (out_prefix_count, "evicted prefix"),
        (out_retirement_count, "control retirement"),
    ] {
        if pointer.is_null() {
            return invalid(&format!("{label} count output is required"));
        }
    }
    unsafe {
        out_request_count.write(request_count);
        out_page_count.write(page_count);
        out_prefix_count.write(prefix_count);
        out_retirement_count.write(retirement_count);
    }
    if request_capacity < request_count
        || page_capacity < page_count
        || prefix_capacity < prefix_count
        || retirement_capacity < retirement_count
    {
        return Ok(true);
    }
    for (missing, label) in [
        (
            request_count != 0 && requests.is_null(),
            "materialized request",
        ),
        (page_count != 0 && pages.is_null(), "materialization page"),
        (prefix_count != 0 && prefixes.is_null(), "evicted prefix"),
        (
            retirement_count != 0 && retirements.is_null(),
            "control retirement",
        ),
    ] {
        if missing {
            return invalid(&format!("{label} output buffer is required"));
        }
    }
    Ok(false)
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_session_prefix_lookup_batch(
    session: *mut OrbitKvSessionHandle,
    keys: *const OrbitKvPrefixSemanticKey,
    key_count: u32,
    lookups: *mut OrbitKvSessionPrefixLookup,
    lookup_capacity: u32,
    out_lookup_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_nonzero_limit(key_count, handle.maximum_prefixes, "prefix lookup key")?;
        let keys = unsafe { copy_input(keys, key_count, "prefix lookup key") }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        if unsafe {
            super::preflight_output(
                lookups,
                lookup_capacity,
                out_lookup_count,
                key_count,
                "prefix lookup",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core = keys
            .iter()
            .copied()
            .map(PrefixSemanticKey::from)
            .collect::<Vec<_>>();
        let output = state
            .runtime
            .lookup_prefix_batch(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        assert_eq!(output.len(), key_count as usize);
        for (index, value) in output.iter().copied().enumerate() {
            let candidate = value
                .candidate
                .map_or_else(OrbitKvSessionPrefixId::default, wire_prefix_id);
            unsafe {
                lookups.add(index).write(OrbitKvSessionPrefixLookup {
                    key: value.key.into(),
                    candidate,
                    resident_count: value.resident_count,
                    candidate_present: u32::from(value.candidate.is_some()),
                    reserved0: 0,
                    reserved1: 0,
                });
            }
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_session_prefix_publish_batch(
    session: *mut OrbitKvSessionHandle,
    items: *const OrbitKvSessionPrefixPublishItem,
    item_count: u32,
    published: *mut OrbitKvSessionPublishedPrefix,
    published_capacity: u32,
    out_published_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_nonzero_limit(
            item_count,
            handle.maximum_requests.min(handle.maximum_prefixes),
            "prefix publication item",
        )?;
        let items = unsafe { copy_input(items, item_count, "prefix publication item") }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        if unsafe {
            super::preflight_output(
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
            .map(|item| {
                (
                    EngineRequestId(item.request_id),
                    PrefixSemanticKey::from(item.key),
                )
            })
            .collect::<Vec<_>>();
        let output = state
            .runtime
            .publish_prefix_batch(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        assert_eq!(output.len(), item_count as usize);
        for (index, value) in output.iter().copied().enumerate() {
            unsafe {
                published.add(index).write(OrbitKvSessionPublishedPrefix {
                    prefix_id: wire_prefix_id(value.prefix_id),
                    key: value.key.into(),
                    resident_count: value.resident_count,
                    reserved: 0,
                });
            }
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_prepare_prefix_attach(
    session: *mut OrbitKvSessionHandle,
    items: *const OrbitKvSessionPrefixAttachItem,
    item_count: u32,
    out_control_id: *mut OrbitKvSessionControlId,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_control_id.is_null() {
            return invalid("control id output is required");
        }
        unsafe { out_control_id.write(OrbitKvSessionControlId::default()) };
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_nonzero_limit(
            item_count,
            handle
                .maximum_requests
                .min(handle.maximum_operations)
                .min(handle.maximum_prefixes),
            "prefix attach item",
        )?;
        let items = unsafe { copy_input(items, item_count, "prefix attach item") }?;
        let mut core = Vec::with_capacity(items.len());
        for item in &items {
            if item.reserved != 0 {
                return invalid("prefix attach item reserved field must be zero");
            }
            validate_operation_id(
                item.prefix_id.session_epoch,
                item.prefix_id.sequence,
                handle,
                "prefix id",
            )?;
            core.push((
                EngineRequestId(item.target_request_id),
                EnginePrefixLookup {
                    key: item.key.into(),
                    candidate: Some(prefix_id(item.prefix_id)),
                    resident_count: item.resident_count,
                },
            ));
        }
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let id = state
            .runtime
            .prepare_prefix_attach(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        unsafe { out_control_id.write(wire_control_id(id)) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_prepare_request_fork(
    session: *mut OrbitKvSessionHandle,
    items: *const OrbitKvSessionRequestForkItem,
    item_count: u32,
    out_control_id: *mut OrbitKvSessionControlId,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_control_id.is_null() {
            return invalid("control id output is required");
        }
        unsafe { out_control_id.write(OrbitKvSessionControlId::default()) };
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_nonzero_limit(
            item_count,
            super::maximum_batch(handle),
            "request fork item",
        )?;
        let items = unsafe { copy_input(items, item_count, "request fork item") }?;
        let core = items
            .iter()
            .map(|item| {
                (
                    EngineRequestId(item.source_request_id),
                    EngineRequestId(item.target_request_id),
                )
            })
            .collect::<Vec<_>>();
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let id = state
            .runtime
            .prepare_request_fork(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        unsafe { out_control_id.write(wire_control_id(id)) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_prepare_prefix_evict(
    session: *mut OrbitKvSessionHandle,
    prefixes: *const OrbitKvSessionPrefixId,
    prefix_count: u32,
    out_control_id: *mut OrbitKvSessionControlId,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_control_id.is_null() {
            return invalid("control id output is required");
        }
        unsafe { out_control_id.write(OrbitKvSessionControlId::default()) };
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_nonzero_limit(
            prefix_count,
            handle.maximum_prefixes.min(handle.maximum_operations),
            "prefix eviction item",
        )?;
        let prefixes = unsafe { copy_input(prefixes, prefix_count, "prefix eviction item") }?;
        for value in &prefixes {
            validate_operation_id(value.session_epoch, value.sequence, handle, "prefix id")?;
        }
        let core = prefixes.iter().copied().map(prefix_id).collect::<Vec<_>>();
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let id = state
            .runtime
            .prepare_prefix_evict(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        unsafe { out_control_id.write(wire_control_id(id)) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_abort_control(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionControlId,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_operation_id(id.session_epoch, id.sequence, handle, "control id")?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = control_id(id);
        state
            .runtime
            .abort_control(core_id)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        state.control_plans.remove(&core_id);
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_commit_control(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionControlId,
    out_plan: *mut OrbitKvSessionControlPlanInfo,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_plan.is_null() {
            return invalid("control plan info output is required");
        }
        unsafe { out_plan.write(OrbitKvSessionControlPlanInfo::default()) };
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_operation_id(id.session_epoch, id.sequence, handle, "control id")?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = control_id(id);
        let cached = state.control_plans.contains_key(&core_id);
        let plan = state.runtime.commit_control(core_id).map_err(|error| {
            if cached {
                cached_control_error(&mut state, error)
            } else {
                mark_poisoned(&mut state, error)
            }
        })?;
        let info = match plan_info(core_id, &plan) {
            Ok(info) => info,
            Err(error) => {
                state.fail_stopped = true;
                return Err(error);
            }
        };
        if let Some(previous) = state.control_plans.get(&core_id) {
            if previous != &plan {
                state.fail_stopped = true;
                return Err((
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "committed control replay changed its plan".into(),
                ));
            }
        } else {
            state.control_plans.insert(core_id, plan);
        }
        unsafe { out_plan.write(info) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_read_control_plan(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionControlId,
    requests: *mut OrbitKvSessionMaterializedRequest,
    request_capacity: u32,
    out_request_count: *mut u32,
    pages: *mut OrbitKvSnapshotPage,
    page_capacity: u32,
    out_page_count: *mut u32,
    prefixes: *mut OrbitKvSessionPrefixId,
    prefix_capacity: u32,
    out_prefix_count: *mut u32,
    retirements: *mut OrbitKvSessionRetirement,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_operation_id(id.session_epoch, id.sequence, handle, "control id")?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = control_id(id);
        let plan = state.control_plans.get(&core_id).cloned().ok_or_else(|| {
            (
                super::ORBITKV_STATUS_RETRYABLE_CONFLICT,
                "control id is not committed or is stale".to_owned(),
            )
        })?;
        let info = match plan_info(core_id, &plan) {
            Ok(info) => info,
            Err(error) => {
                state.fail_stopped = true;
                return Err(error);
            }
        };
        if unsafe {
            preflight_plan_outputs(
                requests,
                request_capacity,
                out_request_count,
                info.request_count,
                pages,
                page_capacity,
                out_page_count,
                info.page_count,
                prefixes,
                prefix_capacity,
                out_prefix_count,
                info.prefix_count,
                retirements,
                retirement_capacity,
                out_retirement_count,
                info.retirement_count,
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        match &plan {
            EngineControlPlan::Materialization(plan) => {
                let mut page_offset = 0_u32;
                for (index, request) in plan.requests.iter().enumerate() {
                    let page_count = exact_len(request.pages.len());
                    for (page_index, page) in request.pages.iter().copied().enumerate() {
                        unsafe {
                            pages
                                .add(page_offset as usize + page_index)
                                .write(page.into());
                        }
                    }
                    unsafe {
                        requests
                            .add(index)
                            .write(OrbitKvSessionMaterializedRequest {
                                request_id: request.request_id.0,
                                view_version: request.view_version.0,
                                boundary: request.boundary,
                                resident_count: request.resident_count,
                                page_offset,
                                page_count,
                                reserved: 0,
                            });
                    }
                    page_offset += page_count;
                }
                debug_assert_eq!(page_offset, info.page_count);
            }
            EngineControlPlan::PrefixEviction(plan) => {
                for (index, prefix) in plan.prefixes.iter().copied().enumerate() {
                    unsafe { prefixes.add(index).write(wire_prefix_id(prefix)) };
                }
                for (index, retirement) in plan.retirements.iter().enumerate() {
                    unsafe { retirements.add(index).write(wire_retirement(retirement)) };
                }
            }
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_cancel_pending_attach(
    session: *mut OrbitKvSessionHandle,
    expected: OrbitKvSessionPendingAttachCancel,
    out_outcome: *mut OrbitKvSessionPendingAttachCancelOutcome,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_outcome.is_null() {
            return invalid("pending attach cancel outcome output is required");
        }
        unsafe { out_outcome.write(OrbitKvSessionPendingAttachCancelOutcome::default()) };
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_operation_id(
            expected.control_id.session_epoch,
            expected.control_id.sequence,
            handle,
            "control id",
        )?;
        validate_operation_id(
            expected.prefix_id.session_epoch,
            expected.prefix_id.sequence,
            handle,
            "prefix id",
        )?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let expected_core = pending_attach_cancel(expected);
        let outcome = state
            .runtime
            .cancel_pending_attach(expected_core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        let core_id = outcome.control_id;
        state.control_plans.remove(&core_id);
        state.resident_counts.remove(&outcome.request_id);
        unsafe { out_outcome.write(wire_pending_attach_cancel_outcome(outcome)) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_finalize_pending_attach_cancel(
    session: *mut OrbitKvSessionHandle,
    expected: OrbitKvSessionPendingAttachCancel,
    out_outcome: *mut OrbitKvSessionPendingAttachCancelOutcome,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_outcome.is_null() {
            return invalid("pending attach cancel outcome output is required");
        }
        unsafe { out_outcome.write(OrbitKvSessionPendingAttachCancelOutcome::default()) };
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_operation_id(
            expected.control_id.session_epoch,
            expected.control_id.sequence,
            handle,
            "control id",
        )?;
        validate_operation_id(
            expected.prefix_id.session_epoch,
            expected.prefix_id.sequence,
            handle,
            "prefix id",
        )?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let outcome = state
            .runtime
            .finalize_pending_attach_cancel(pending_attach_cancel(expected))
            .map_err(|error| mark_poisoned(&mut state, error))?;
        unsafe { out_outcome.write(wire_pending_attach_cancel_outcome(outcome)) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_confirm_control(
    session: *mut OrbitKvSessionHandle,
    evidence: OrbitKvSessionControlEvidence,
    retirements: *const OrbitKvSessionRetirementEvidence,
    retirement_count: u32,
    out_outcome: *mut OrbitKvSessionControlOutcome,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_outcome.is_null() {
            return invalid("control outcome output is required");
        }
        unsafe { out_outcome.write(OrbitKvSessionControlOutcome::default()) };
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_operation_id(
            evidence.id.session_epoch,
            evidence.id.sequence,
            handle,
            "control id",
        )?;
        if evidence.reserved != 0 {
            return invalid("control evidence reserved field must be zero");
        }
        let mirror_updates_confirmed = validate_bool(
            evidence.mirror_updates_confirmed,
            "control mirror update confirmation",
        )?;
        validate_count_limit(
            retirement_count,
            handle.total_page_capacity,
            "control retirement evidence",
        )?;
        let retirements =
            unsafe { copy_input(retirements, retirement_count, "control retirement evidence") }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = control_id(evidence.id);
        let plan = state.control_plans.get(&core_id).cloned().ok_or_else(|| {
            (
                super::ORBITKV_STATUS_RETRYABLE_CONFLICT,
                "control id is not committed or is stale".to_owned(),
            )
        })?;
        let expected = match &plan {
            EngineControlPlan::Materialization(_) => &[][..],
            EngineControlPlan::PrefixEviction(plan) => &plan.retirements,
        };
        let receipts = validate_retirement_evidence(expected, &retirements)?;
        let outcome = state
            .runtime
            .confirm_control(&EngineControlEvidence {
                control_id: core_id,
                mirror_updates_confirmed,
                reclamation_receipts: receipts,
            })
            .map_err(|error| cached_control_error(&mut state, error))?;
        let materialized_residents = match &plan {
            EngineControlPlan::Materialization(plan) => Some(
                plan.requests
                    .iter()
                    .map(|request| (request.request_id, request.resident_count))
                    .collect::<Vec<_>>(),
            ),
            EngineControlPlan::PrefixEviction(_) => None,
        };
        let disposition = match (&plan, outcome) {
            (EngineControlPlan::Materialization(_), EngineControlOutcome::Materialized) => {
                ORBITKV_SESSION_CONTROL_OUTCOME_MATERIALIZED
            }
            (EngineControlPlan::PrefixEviction(_), EngineControlOutcome::Evicted) => {
                ORBITKV_SESSION_CONTROL_OUTCOME_EVICTED
            }
            _ => {
                state.fail_stopped = true;
                return Err((
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "control outcome disagreed with committed plan".into(),
                ));
            }
        };
        if let Some(residents) = materialized_residents {
            for (request_id, resident_count) in residents {
                state.resident_counts.insert(request_id, resident_count);
            }
        }
        state.control_plans.remove(&core_id);
        unsafe {
            out_outcome.write(OrbitKvSessionControlOutcome {
                id: wire_control_id(core_id),
                disposition,
                reserved: 0,
            });
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_quarantine_control(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionControlId,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        ensure_shared_cache(handle)?;
        validate_operation_id(id.session_epoch, id.sequence, handle, "control id")?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = control_id(id);
        if !state.control_plans.contains_key(&core_id) {
            return retryable("control id is not committed or is stale");
        }
        state
            .runtime
            .quarantine_control(core_id)
            .map_err(|error| cached_control_error(&mut state, error))?;
        state.control_plans.remove(&core_id);
        Ok(ORBITKV_STATUS_OK)
    })
}
