use std::ffi::c_char;

use orbitkv::kv_manager::{KvManagerError, RelocationPolicy, TokenDispositionKind};
use orbitkv::{
    EngineCompletionEvidence, EnginePrepareRelocationItem, EngineRelocationAbortEvidence,
    EngineRelocationCopyEvidence, EngineRelocationExecutionEvidence, EngineRelocationId,
    EngineRelocationPublicationEvidence, EngineRelocationRequestEvidence, EngineRequestId,
    EngineTokenDispositionBatchItem, EngineTokenDispositionUpdate, EngineTokenViewQuery,
    RuntimeSessionError,
};

use super::{
    ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_FAIL_STOPPED, ORBITKV_STATUS_OK,
    OrbitKvSessionCompletionEvidence, OrbitKvSessionHandle, OrbitKvSessionPrepareRelocationItem,
    OrbitKvSessionRelocationAbortEvidence, OrbitKvSessionRelocationCopyEvidence,
    OrbitKvSessionRelocationId, OrbitKvSessionRelocationPlan,
    OrbitKvSessionRelocationPublicationEvidence, OrbitKvSessionRelocationRequestEvidence,
    OrbitKvSessionRelocationRequestPublication, OrbitKvSessionRequestView,
    OrbitKvSessionRetirement, OrbitKvSessionRetirementEvidence,
    OrbitKvSessionTokenDispositionBatchItem, OrbitKvSessionTokenView, OrbitKvSessionTokenViewQuery,
    SessionState, checked_mul, copy_input, ensure_running, exact_len, ffi_boundary, invalid,
    lock_state, maximum_batch, preflight_output, retryable, session_error, session_ref,
    validate_bool, validate_count_limit, validate_nonzero_limit, validate_retirement_evidence,
    validate_session_epoch, wire_retirement,
};
use crate::wire::{
    OrbitKvClassTokenDispositionUpdate, OrbitKvPageLease, OrbitKvTokenMove, OrbitKvTokenPlacement,
};

#[derive(Clone, Debug)]
pub(super) enum PendingRelocationWire {
    Prepared {
        requests: Box<[EngineRequestId]>,
        copy_counts: Box<[u32]>,
    },
    Submitted {
        requests: Box<[EngineRequestId]>,
    },
    PublicationPending {
        requests: Box<[EngineRequestId]>,
        retirements: Box<[orbitkv::EngineRetirement]>,
    },
}

fn relocation_id(value: OrbitKvSessionRelocationId) -> EngineRelocationId {
    EngineRelocationId::from_parts(value.session_epoch, value.sequence)
}

fn wire_relocation_id(value: EngineRelocationId) -> OrbitKvSessionRelocationId {
    OrbitKvSessionRelocationId {
        session_epoch: value.session_epoch(),
        sequence: value.sequence(),
    }
}

fn validate_relocation_id(
    handle: &OrbitKvSessionHandle,
    value: OrbitKvSessionRelocationId,
) -> Result<EngineRelocationId, (i32, String)> {
    validate_session_epoch(handle, value.session_epoch, "relocation id")?;
    if value.sequence == 0 {
        return retryable("relocation id is unknown");
    }
    Ok(relocation_id(value))
}

fn maximum_token_items(handle: &OrbitKvSessionHandle, label: &str) -> Result<u32, (i32, String)> {
    let page_tokens = handle
        .arena_identities
        .first()
        .expect("session has at least one arena")
        .page_tokens;
    checked_mul(handle.total_page_capacity, page_tokens, label)
}

fn validate_spans<T>(
    items: &[T],
    total: u32,
    offset: impl Fn(&T) -> u32,
    count: impl Fn(&T) -> u32,
    label: &str,
) -> Result<(), (i32, String)> {
    let mut expected = 0_u32;
    for item in items {
        if offset(item) != expected {
            return invalid(&format!("{label} spans must be canonical and gap-free"));
        }
        expected = expected.checked_add(count(item)).ok_or_else(|| {
            (
                super::ORBITKV_STATUS_INVALID_ARGUMENT,
                format!("{label} span overflows"),
            )
        })?;
    }
    if expected != total {
        return invalid(&format!("{label} spans must cover the flat buffer"));
    }
    Ok(())
}

fn mark_poisoned(state: &mut SessionState, error: RuntimeSessionError) -> (i32, String) {
    if matches!(error, RuntimeSessionError::SessionPoisoned(_)) {
        state.fail_stopped = true;
    }
    session_error(error)
}

fn cached_relocation_error(state: &mut SessionState, error: RuntimeSessionError) -> (i32, String) {
    if matches!(
        error,
        RuntimeSessionError::UnknownRelocation(_)
            | RuntimeSessionError::ForeignRelocation(_)
            | RuntimeSessionError::StaleRelocation(_)
            | RuntimeSessionError::RelocationNotPrepared(_)
            | RuntimeSessionError::RelocationNotSubmitted(_)
            | RuntimeSessionError::RelocationPublicationNotPending(_)
            | RuntimeSessionError::SessionPoisoned(_)
    ) {
        state.fail_stopped = true;
        return (
            ORBITKV_STATUS_FAIL_STOPPED,
            format!("cached relocation state disagrees with runtime session: {error}"),
        );
    }
    session_error(error)
}

fn wire_invariant<T>(state: &mut SessionState, message: &str) -> Result<T, (i32, String)> {
    state.fail_stopped = true;
    Err((ORBITKV_STATUS_FAIL_STOPPED, message.to_owned()))
}

/// Cold-materializes session-owned logical token views.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_token_views_batch(
    session: *mut OrbitKvSessionHandle,
    queries: *const OrbitKvSessionTokenViewQuery,
    query_count: u32,
    views: *mut OrbitKvSessionTokenView,
    view_capacity: u32,
    out_view_count: *mut u32,
    placements: *mut OrbitKvTokenPlacement,
    placement_capacity: u32,
    out_placement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        validate_nonzero_limit(query_count, handle.maximum_requests, "token-view query")?;
        let queries = unsafe { copy_input(queries, query_count, "token-view query") }?;
        if queries
            .iter()
            .any(|query| query.reserved16 != 0 || query.reserved32 != 0)
        {
            return invalid("token-view query reserved fields must be zero");
        }
        let core = queries
            .iter()
            .map(|query| EngineTokenViewQuery {
                request_id: EngineRequestId(query.request_id),
                class_id: query.class_id,
                expected_boundary: query.expected_boundary,
            })
            .collect::<Vec<_>>();
        let declared_placement_count = queries.iter().try_fold(0_u32, |total, query| {
            let count = u32::try_from(query.expected_boundary).map_err(|_| {
                (
                    super::ORBITKV_STATUS_INVALID_ARGUMENT,
                    "token-view expected boundary exceeds uint32_t".to_owned(),
                )
            })?;
            total.checked_add(count).ok_or_else(|| {
                (
                    super::ORBITKV_STATUS_INVALID_ARGUMENT,
                    "token-view placement count exceeds uint32_t".to_owned(),
                )
            })
        })?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let outputs = state
            .runtime
            .token_views_batch(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        let placement_count = outputs.iter().try_fold(0_u32, |total, view| {
            let count = u32::try_from(view.placements.len()).map_err(|_| {
                (
                    super::ORBITKV_STATUS_MANAGER_ERROR,
                    "token placement count exceeds uint32_t".to_owned(),
                )
            })?;
            total.checked_add(count).ok_or_else(|| {
                (
                    super::ORBITKV_STATUS_MANAGER_ERROR,
                    "token placement count exceeds uint32_t".to_owned(),
                )
            })
        })?;
        if placement_count != declared_placement_count {
            return wire_invariant(
                &mut state,
                "token-view result count changed from declared boundaries",
            );
        }
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
                    placement_count,
                    "token placement",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        if outputs.len() != queries.len() {
            return wire_invariant(&mut state, "token-view result cardinality changed");
        }
        let mut placement_offset = 0_u32;
        for (index, output) in outputs.iter().enumerate() {
            if output.request_id.0 != queries[index].request_id
                || output.class_id != queries[index].class_id
                || output.placements.len() as u64 != queries[index].expected_boundary
            {
                return wire_invariant(&mut state, "token-view result ordering changed");
            }
            let count = exact_len(output.placements.len());
            for (placement_index, placement) in output.placements.iter().copied().enumerate() {
                unsafe {
                    placements
                        .add(placement_offset as usize + placement_index)
                        .write(placement.into());
                }
            }
            unsafe {
                views.add(index).write(OrbitKvSessionTokenView {
                    request_id: output.request_id.0,
                    view_version: output.version.0,
                    placement_offset,
                    placement_count: count,
                    page_tokens: output.page_tokens,
                    class_id: output.class_id,
                    reserved16: 0,
                    reserved32: 0,
                });
            }
            placement_offset += count;
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Atomically publishes session-owned logical token dispositions.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_session_mark_token_dispositions_batch(
    session: *mut OrbitKvSessionHandle,
    items: *const OrbitKvSessionTokenDispositionBatchItem,
    item_count: u32,
    updates: *const OrbitKvClassTokenDispositionUpdate,
    update_count: u32,
    outputs: *mut OrbitKvSessionRequestView,
    output_capacity: u32,
    out_output_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_nonzero_limit(item_count, maximum_batch(handle), "disposition item")?;
        validate_count_limit(
            update_count,
            maximum_token_items(handle, "disposition update")?,
            "disposition update",
        )?;
        let items = unsafe { copy_input(items, item_count, "disposition item") }?;
        let updates = unsafe { copy_input(updates, update_count, "disposition update") }?;
        validate_spans(
            &items,
            update_count,
            |item| item.update_offset,
            |item| item.update_count,
            "disposition",
        )?;
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
                EngineTokenDispositionBatchItem {
                    request_id: EngineRequestId(item.request_id),
                    updates: updates[begin..end]
                        .iter()
                        .map(|update| EngineTokenDispositionUpdate {
                            class_id: update.class_id,
                            token_id: update.token_id,
                            disposition: update.disposition.into(),
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                }
            })
            .collect::<Vec<_>>();
        let values = state
            .runtime
            .mark_token_dispositions_batch(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        if values.len() != items.len()
            || values
                .iter()
                .zip(&items)
                .any(|(value, item)| value.request_id.0 != item.request_id)
        {
            return wire_invariant(&mut state, "disposition result ordering changed");
        }
        for (index, value) in values.iter().copied().enumerate() {
            state
                .resident_counts
                .insert(value.request_id, value.resident_count);
            unsafe {
                outputs.add(index).write(OrbitKvSessionRequestView {
                    request_id: value.request_id.0,
                    view_version: value.view_version.0,
                    boundary: value.boundary,
                    resident_count: value.resident_count,
                    reserved: 0,
                });
            }
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Prepares one session-owned full-evacuation relocation batch.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_prepare_relocation_batch(
    session: *mut OrbitKvSessionHandle,
    items: *const OrbitKvSessionPrepareRelocationItem,
    item_count: u32,
    out_relocation_id: *mut OrbitKvSessionRelocationId,
    plans: *mut OrbitKvSessionRelocationPlan,
    plan_capacity: u32,
    out_plan_count: *mut u32,
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
        if out_relocation_id.is_null() {
            return invalid("relocation id output is required");
        }
        unsafe { out_relocation_id.write(OrbitKvSessionRelocationId::default()) };
        let handle = unsafe { session_ref(session) }?;
        validate_nonzero_limit(item_count, maximum_batch(handle), "relocation item")?;
        let items = unsafe { copy_input(items, item_count, "relocation item") }?;
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
        let move_bound = maximum_token_items(handle, "relocation move")?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let shorts = [
            unsafe {
                preflight_output(
                    plans,
                    plan_capacity,
                    out_plan_count,
                    item_count,
                    "relocation plan",
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
            .map(|item| EnginePrepareRelocationItem {
                request_id: EngineRequestId(item.request_id),
                class_id: item.class_id,
                policy: RelocationPolicy {
                    fragmentation_threshold_milli: item.policy.fragmentation_threshold_milli,
                    maximum_source_pages: item.policy.maximum_source_pages,
                    evacuation_headroom_pages: item.policy.evacuation_headroom_pages,
                    full_evacuation: item.policy.full_evacuation == 1,
                },
            })
            .collect::<Vec<_>>();
        let output = state
            .runtime
            .prepare_relocation_batch(&core)
            .map_err(|error| mark_poisoned(&mut state, error))?;
        if output.relocation_id.session_epoch() != handle.session_epoch
            || output.relocation_id.sequence() == 0
            || output.plans.len() != items.len()
            || state.relocations.contains_key(&output.relocation_id)
        {
            return wire_invariant(
                &mut state,
                "relocation prepare identity or cardinality changed",
            );
        }
        let mut wire_plans = Vec::with_capacity(output.plans.len());
        let mut wire_sources: Vec<OrbitKvPageLease> = Vec::new();
        let mut wire_destinations: Vec<OrbitKvPageLease> = Vec::new();
        let mut wire_moves: Vec<OrbitKvTokenMove> = Vec::new();
        let mut copy_counts = Vec::with_capacity(output.plans.len());
        for (index, plan) in output.plans.iter().enumerate() {
            if plan.request_id.0 != items[index].request_id
                || plan.class_id != items[index].class_id
            {
                return wire_invariant(&mut state, "relocation prepare result ordering changed");
            }
            let source_offset = u32::try_from(wire_sources.len()).map_err(|_| {
                state.fail_stopped = true;
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "relocation source offset exceeds uint32_t".into(),
                )
            })?;
            let destination_offset = u32::try_from(wire_destinations.len()).map_err(|_| {
                state.fail_stopped = true;
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "relocation destination offset exceeds uint32_t".into(),
                )
            })?;
            let move_offset = u32::try_from(wire_moves.len()).map_err(|_| {
                state.fail_stopped = true;
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "relocation move offset exceeds uint32_t".into(),
                )
            })?;
            let source_count = u32::try_from(plan.source_pages.len()).map_err(|_| {
                state.fail_stopped = true;
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "relocation source count exceeds uint32_t".into(),
                )
            })?;
            let destination_count = u32::try_from(plan.destination_pages.len()).map_err(|_| {
                state.fail_stopped = true;
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "relocation destination count exceeds uint32_t".into(),
                )
            })?;
            let move_count = u32::try_from(plan.moves.len()).map_err(|_| {
                state.fail_stopped = true;
                (
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "relocation move count exceeds uint32_t".into(),
                )
            })?;
            if source_offset
                .checked_add(source_count)
                .is_none_or(|end| end > page_bound)
                || destination_offset
                    .checked_add(destination_count)
                    .is_none_or(|end| end > page_bound)
                || move_offset
                    .checked_add(move_count)
                    .is_none_or(|end| end > move_bound)
            {
                return wire_invariant(
                    &mut state,
                    "relocation prepare output exceeded its preflight bound",
                );
            }
            wire_sources.extend(
                plan.source_pages
                    .iter()
                    .copied()
                    .map(OrbitKvPageLease::from),
            );
            wire_destinations.extend(
                plan.destination_pages
                    .iter()
                    .copied()
                    .map(OrbitKvPageLease::from),
            );
            wire_moves.extend(plan.moves.iter().map(|movement| OrbitKvTokenMove {
                token_id: movement.token_id,
                source: movement.source.into(),
                destination: movement.destination.into(),
            }));
            copy_counts.push(move_count);
            wire_plans.push(OrbitKvSessionRelocationPlan {
                request_id: plan.request_id.0,
                base_view_version: plan.base_version.0,
                target_view_version: plan.target_version.0,
                source_offset,
                source_count,
                destination_offset,
                destination_count,
                move_offset,
                move_count,
                projected_reclaimed_pages: plan.projected_reclaimed_pages,
                fragmentation_milli: plan.fragmentation_milli,
                class_id: plan.class_id,
                reserved32: 0,
            });
        }
        let requests = output
            .plans
            .iter()
            .map(|plan| plan.request_id)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let previous = state.relocations.insert(
            output.relocation_id,
            PendingRelocationWire::Prepared {
                requests,
                copy_counts: copy_counts.into_boxed_slice(),
            },
        );
        if previous.is_some() {
            return wire_invariant(&mut state, "relocation prepare reused a live id");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(wire_plans.as_ptr(), plans, wire_plans.len());
            if !wire_sources.is_empty() {
                std::ptr::copy_nonoverlapping(wire_sources.as_ptr(), sources, wire_sources.len());
            }
            if !wire_destinations.is_empty() {
                std::ptr::copy_nonoverlapping(
                    wire_destinations.as_ptr(),
                    destinations,
                    wire_destinations.len(),
                );
            }
            if !wire_moves.is_empty() {
                std::ptr::copy_nonoverlapping(wire_moves.as_ptr(), moves, wire_moves.len());
            }
            out_relocation_id.write(wire_relocation_id(output.relocation_id));
            out_plan_count.write(exact_len(wire_plans.len()));
            out_source_count.write(exact_len(wire_sources.len()));
            out_destination_count.write(exact_len(wire_destinations.len()));
            out_move_count.write(exact_len(wire_moves.len()));
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Aborts a prepared relocation with exact ordered unobserved evidence.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_abort_prepared_relocation(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionRelocationId,
    evidence: *const OrbitKvSessionRelocationAbortEvidence,
    evidence_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = validate_relocation_id(handle, id)?;
        validate_nonzero_limit(
            evidence_count,
            maximum_batch(handle),
            "relocation abort evidence",
        )?;
        let evidence =
            unsafe { copy_input(evidence, evidence_count, "relocation abort evidence") }?;
        if evidence
            .iter()
            .any(|item| item.reserved != 0 || item.backend_unobserved > 1)
        {
            return invalid("relocation abort evidence has invalid boolean or reserved fields");
        }
        let core = evidence
            .iter()
            .map(|item| EngineRelocationAbortEvidence {
                request_id: EngineRequestId(item.request_id),
                backend_unobserved: item.backend_unobserved == 1,
            })
            .collect::<Vec<_>>();
        let Some(PendingRelocationWire::Prepared { requests, .. }) =
            state.relocations.get(&core_id)
        else {
            return retryable("relocation id is unknown, stale, or not prepared");
        };
        if requests.len() != evidence.len()
            || requests
                .iter()
                .zip(&evidence)
                .any(|(request, item)| request.0 != item.request_id)
        {
            return invalid("relocation abort evidence must exactly match prepared request order");
        }
        state
            .runtime
            .abort_prepared_relocation(core_id, &core)
            .map_err(|error| cached_relocation_error(&mut state, error))?;
        if state.relocations.remove(&core_id).is_none() {
            return wire_invariant(&mut state, "relocation abort lost cached state");
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Permanently quarantines one relocation and fail-stops this FFI handle.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_quarantine_relocation(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionRelocationId,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = validate_relocation_id(handle, id)?;
        if !state.relocations.contains_key(&core_id) {
            return retryable("relocation id is unknown or stale");
        }
        match state.runtime.quarantine_relocation(core_id) {
            Ok(()) | Err(RuntimeSessionError::SessionPoisoned(_)) => {
                state.relocations.remove(&core_id);
                state.fail_stopped = true;
                Err((
                    ORBITKV_STATUS_FAIL_STOPPED,
                    "relocation was quarantined; the session must be fail-stopped".to_owned(),
                ))
            }
            Err(error) => Err(cached_relocation_error(&mut state, error)),
        }
    })
}

/// Submits exact ordered copy evidence for one opaque relocation id.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_submit_relocation(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionRelocationId,
    requests: *const OrbitKvSessionRelocationRequestEvidence,
    request_count: u32,
    copies: *const OrbitKvSessionRelocationCopyEvidence,
    copy_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = validate_relocation_id(handle, id)?;
        validate_nonzero_limit(
            request_count,
            maximum_batch(handle),
            "relocation request evidence",
        )?;
        validate_count_limit(
            copy_count,
            maximum_token_items(handle, "relocation copy evidence")?,
            "relocation copy evidence",
        )?;
        let requests =
            unsafe { copy_input(requests, request_count, "relocation request evidence") }?;
        let copies = unsafe { copy_input(copies, copy_count, "relocation copy evidence") }?;
        validate_spans(
            &requests,
            copy_count,
            |item| item.copy_offset,
            |item| item.copy_count,
            "relocation copy",
        )?;
        if copies.iter().any(|copy| {
            copy.reserved16 != 0
                || copy.reserved32 != 0
                || copy.source.reserved != 0
                || copy.destination.reserved != 0
                || copy.observed > 1
                || copy.copied > 1
        }) {
            return invalid("relocation copy evidence has invalid boolean or reserved fields");
        }
        let Some(PendingRelocationWire::Prepared {
            requests: expected_requests,
            copy_counts,
        }) = state.relocations.get(&core_id).cloned()
        else {
            return retryable("relocation id is unknown, stale, or not prepared");
        };
        if expected_requests.len() != requests.len()
            || copy_counts.len() != requests.len()
            || expected_requests
                .iter()
                .zip(requests.iter().zip(copy_counts.iter()))
                .any(|(expected, (request, count))| {
                    expected.0 != request.request_id || *count != request.copy_count
                })
        {
            return invalid("relocation copy evidence must exactly match prepared request spans");
        }
        let grouped = requests
            .iter()
            .map(|request| {
                let begin = request.copy_offset as usize;
                let end = begin + request.copy_count as usize;
                EngineRelocationRequestEvidence {
                    request_id: EngineRequestId(request.request_id),
                    copies: copies[begin..end]
                        .iter()
                        .map(|copy| EngineRelocationCopyEvidence {
                            token_id: copy.token_id,
                            source: copy.source.into(),
                            destination: copy.destination.into(),
                            observed: copy.observed == 1,
                            copied: copy.copied == 1,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let result = state
            .runtime
            .submit_relocation(&EngineRelocationExecutionEvidence {
                relocation_id: core_id,
                requests: grouped,
            });
        let ticket = match result {
            Ok(ticket) => ticket,
            Err(error @ RuntimeSessionError::Manager(KvManagerError::BatchQuarantined(_))) => {
                state.relocations.remove(&core_id);
                state.fail_stopped = true;
                return Err(session_error(error));
            }
            Err(error) => return Err(cached_relocation_error(&mut state, error)),
        };
        if ticket.relocation_id() != core_id {
            return wire_invariant(&mut state, "relocation submit identity changed");
        }
        state.relocations.insert(
            core_id,
            PendingRelocationWire::Submitted {
                requests: expected_requests,
            },
        );
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Publishes relocation results at one confirmed GPU completion frontier.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_complete_relocation(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionRelocationId,
    completion: OrbitKvSessionCompletionEvidence,
    publications: *mut OrbitKvSessionRelocationRequestPublication,
    publication_capacity: u32,
    out_publication_count: *mut u32,
    retirements: *mut OrbitKvSessionRetirement,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = validate_relocation_id(handle, id)?;
        if completion.reserved != 0 {
            return invalid("relocation completion reserved field must be zero");
        }
        let confirmed = validate_bool(completion.confirmed, "relocation completion confirmed")?;
        let Some(PendingRelocationWire::Submitted { requests }) =
            state.relocations.get(&core_id).cloned()
        else {
            return retryable("relocation id is unknown, stale, or not submitted");
        };
        let publication_bound = u32::try_from(requests.len()).map_err(|_| {
            (
                ORBITKV_STATUS_FAIL_STOPPED,
                "cached relocation request count exceeds uint32_t".to_owned(),
            )
        })?;
        let shorts = [
            unsafe {
                preflight_output(
                    publications,
                    publication_capacity,
                    out_publication_count,
                    publication_bound,
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
        let output = state
            .runtime
            .complete_relocation(
                core_id,
                EngineCompletionEvidence {
                    completion_domain: completion.completion_domain,
                    completion_value: completion.completion_value,
                    confirmed,
                },
            )
            .map_err(|error| cached_relocation_error(&mut state, error))?;
        if output.relocation_id != core_id
            || output.requests.len() != requests.len()
            || output
                .requests
                .iter()
                .zip(&requests)
                .any(|(publication, request)| publication.request_id != *request)
            || output.retirements.len() > handle.total_page_capacity as usize
        {
            return wire_invariant(
                &mut state,
                "relocation completion output changed or exceeded its bound",
            );
        }
        let wire_publications = output
            .requests
            .iter()
            .map(|publication| OrbitKvSessionRelocationRequestPublication {
                request_id: publication.request_id.0,
                view_version: publication.view_version.0,
                boundary: publication.boundary,
                resident_count: publication.resident_count,
                reserved: 0,
            })
            .collect::<Vec<_>>();
        let wire_retirements = output
            .retirements
            .iter()
            .map(wire_retirement)
            .collect::<Vec<_>>();
        for publication in &output.requests {
            state
                .resident_counts
                .insert(publication.request_id, publication.resident_count);
        }
        state.relocations.insert(
            core_id,
            PendingRelocationWire::PublicationPending {
                requests,
                retirements: output.retirements.clone(),
            },
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                wire_publications.as_ptr(),
                publications,
                wire_publications.len(),
            );
            if !wire_retirements.is_empty() {
                std::ptr::copy_nonoverlapping(
                    wire_retirements.as_ptr(),
                    retirements,
                    wire_retirements.len(),
                );
            }
            out_publication_count.write(exact_len(wire_publications.len()));
            out_retirement_count.write(exact_len(wire_retirements.len()));
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Confirms mirror cleanup and exact retirement ACKs for a relocation.
///
/// # Safety
/// All pointers must satisfy the public header's pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_confirm_relocation_publication(
    session: *mut OrbitKvSessionHandle,
    evidence: OrbitKvSessionRelocationPublicationEvidence,
    retirements: *const OrbitKvSessionRetirementEvidence,
    retirement_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        let core_id = validate_relocation_id(handle, evidence.relocation_id)?;
        if evidence.reserved != 0 {
            return invalid("relocation publication evidence reserved field must be zero");
        }
        let mirror_cleanup_confirmed = validate_bool(
            evidence.mirror_cleanup_confirmed,
            "relocation mirror cleanup confirmation",
        )?;
        validate_count_limit(
            retirement_count,
            handle.total_page_capacity,
            "relocation retirement evidence",
        )?;
        let retirements = unsafe {
            copy_input(
                retirements,
                retirement_count,
                "relocation retirement evidence",
            )
        }?;
        let Some(PendingRelocationWire::PublicationPending {
            requests,
            retirements: expected,
        }) = state.relocations.get(&core_id).cloned()
        else {
            return retryable("relocation id is unknown, stale, or has no publication pending");
        };
        if requests.is_empty() {
            return wire_invariant(&mut state, "cached relocation publication has no requests");
        }
        let receipts = validate_retirement_evidence(&expected, &retirements)?;
        state
            .runtime
            .confirm_relocation_publication(&EngineRelocationPublicationEvidence {
                relocation_id: core_id,
                mirror_cleanup_confirmed,
                reclamation_receipts: receipts,
            })
            .map_err(|error| cached_relocation_error(&mut state, error))?;
        if state.relocations.remove(&core_id).is_none() {
            return wire_invariant(&mut state, "relocation confirmation lost cached state");
        }
        Ok(ORBITKV_STATUS_OK)
    })
}
