#![allow(clippy::missing_panics_doc, clippy::missing_safety_doc)]

use std::collections::BTreeMap;
use std::ffi::c_char;
use std::slice;
use std::sync::{Mutex, MutexGuard};

#[cfg(feature = "test-support")]
use orbitkv::RuntimeSessionTestFault;
use orbitkv::kv_manager::{KvManagerError, PageLease};
use orbitkv::{
    CacheSharingPolicy, EngineAppendIntent, EngineBatchId, EngineBindEvidence,
    EngineCompletionEvidence, EngineControlId, EngineControlPlan, EngineCopyEvidence,
    EnginePublicationEvidence, EnginePublicationId, EngineReleaseEvidence, EngineReleaseId,
    EngineReleaseOutcome, EngineRelocationId, EngineRequestId, EngineRetirement,
    EngineStepAbortEvidence, EngineStepExecutionEvidence, ExecutionEvidence, RuntimeSession,
    RuntimeSessionError,
};

#[cfg(test)]
use crate::wire::OrbitKvManagerConfig;
use crate::wire::{
    OrbitKvArenaIdentity, OrbitKvArenaStats, OrbitKvBackendArenaRegistration, OrbitKvClassLowering,
    OrbitKvCopyIntent, OrbitKvDetachedBinding, OrbitKvManagerStats, OrbitKvTailAction,
    OrbitKvWriteIntent, checked_mul, copy_input, exact_len, preflight_output, validate_bool,
    validate_count_limit, validate_nonzero_limit,
};
use crate::{
    ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_FAIL_STOPPED, ORBITKV_STATUS_INVALID_ARGUMENT,
    ORBITKV_STATUS_MANAGER_ERROR, ORBITKV_STATUS_OK, ORBITKV_STATUS_RETRYABLE_CONFLICT,
    ffi_boundary,
};

mod control;
mod create;
mod layouts;
mod prefix_release;
mod relocation;
pub use control::*;
use create::compile_session_manager;
pub use layouts::*;
pub use prefix_release::*;
pub use relocation::*;

#[cfg(test)]
mod chunked_tests;
#[cfg(test)]
mod control_tests;
#[cfg(test)]
mod prefix_release_tests;
#[cfg(test)]
mod relocation_tests;
#[cfg(test)]
mod tests;

const MAX_PLAN_JSON_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
struct PendingReleaseWire {
    requests: Box<[EngineRequestId]>,
    retirements: Box<[EngineRetirement]>,
}

#[derive(Debug)]
struct SessionState {
    runtime: RuntimeSession,
    fail_stopped: bool,
    batch_sizes: BTreeMap<EngineBatchId, u32>,
    publications: BTreeMap<EnginePublicationId, Box<[EngineRetirement]>>,
    releases: BTreeMap<EngineReleaseId, PendingReleaseWire>,
    control_plans: BTreeMap<EngineControlId, EngineControlPlan>,
    relocations: BTreeMap<EngineRelocationId, relocation::PendingRelocationWire>,
    resident_counts: BTreeMap<EngineRequestId, u32>,
}

pub struct OrbitKvSessionHandle {
    state: Mutex<SessionState>,
    cache_sharing_policy: CacheSharingPolicy,
    session_epoch: u64,
    total_page_capacity: u32,
    maximum_requests: u32,
    maximum_operations: u32,
    maximum_prefixes: u32,
    class_count: u32,
    maximum_write_intents_per_item: u32,
    maximum_completion_outputs_per_item: u32,
    arena_identities: Box<[OrbitKvArenaIdentity]>,
}

fn invalid<T>(message: &str) -> Result<T, (i32, String)> {
    Err((ORBITKV_STATUS_INVALID_ARGUMENT, message.to_owned()))
}

fn retryable<T>(message: &str) -> Result<T, (i32, String)> {
    Err((ORBITKV_STATUS_RETRYABLE_CONFLICT, message.to_owned()))
}

fn core_status(error: &KvManagerError) -> i32 {
    match error {
        KvManagerError::StaleLease(_)
        | KvManagerError::StaleView
        | KvManagerError::PrefixMiss
        | KvManagerError::PrefixHintStale
        | KvManagerError::DuplicatePrefixKey => ORBITKV_STATUS_RETRYABLE_CONFLICT,
        KvManagerError::BatchQuarantined(_) => ORBITKV_STATUS_FAIL_STOPPED,
        _ => ORBITKV_STATUS_MANAGER_ERROR,
    }
}

#[allow(clippy::needless_pass_by_value)]
fn session_error(error: RuntimeSessionError) -> (i32, String) {
    let status = match &error {
        RuntimeSessionError::Manager(error) => core_status(error),
        RuntimeSessionError::SessionPoisoned(_) => ORBITKV_STATUS_FAIL_STOPPED,
        RuntimeSessionError::RequestAlreadyAcquired(_)
        | RuntimeSessionError::UnknownRequest(_)
        | RuntimeSessionError::RequestNotReady { .. }
        | RuntimeSessionError::UnknownBatch(_)
        | RuntimeSessionError::ForeignBatch(_)
        | RuntimeSessionError::StaleBatch(_)
        | RuntimeSessionError::BatchNotPrepared(_)
        | RuntimeSessionError::BatchNotSubmitted(_)
        | RuntimeSessionError::UnknownPublication(_)
        | RuntimeSessionError::ForeignPublication(_)
        | RuntimeSessionError::StalePublication(_)
        | RuntimeSessionError::UnknownRelease(_)
        | RuntimeSessionError::ForeignRelease(_)
        | RuntimeSessionError::StaleRelease(_)
        | RuntimeSessionError::UnknownPrefix(_)
        | RuntimeSessionError::ForeignPrefix(_)
        | RuntimeSessionError::StalePrefix(_)
        | RuntimeSessionError::PrefixNotReady { .. }
        | RuntimeSessionError::UnknownControl(_)
        | RuntimeSessionError::ForeignControl(_)
        | RuntimeSessionError::StaleControl(_)
        | RuntimeSessionError::ControlAlreadyCommitted(_)
        | RuntimeSessionError::ControlNotCommitted(_)
        | RuntimeSessionError::UnknownRelocation(_)
        | RuntimeSessionError::ForeignRelocation(_)
        | RuntimeSessionError::StaleRelocation(_)
        | RuntimeSessionError::RelocationNotPrepared(_)
        | RuntimeSessionError::RelocationNotSubmitted(_)
        | RuntimeSessionError::RelocationPublicationNotPending(_)
        | RuntimeSessionError::TokenViewBoundary { .. }
        | RuntimeSessionError::CanceledRequestNotPending(_) => ORBITKV_STATUS_RETRYABLE_CONFLICT,
        RuntimeSessionError::EmptyBatch
        | RuntimeSessionError::DuplicateRequest(_)
        | RuntimeSessionError::DuplicatePrefix(_)
        | RuntimeSessionError::ControlNotCancelable(_)
        | RuntimeSessionError::PendingAttachCancelMismatch(_)
        | RuntimeSessionError::EvidenceCardinality { .. }
        | RuntimeSessionError::EvidenceRequest { .. }
        | RuntimeSessionError::MirrorUpdatesNotConfirmed
        | RuntimeSessionError::MirrorCleanupNotConfirmed
        | RuntimeSessionError::ReclamationReceiptMismatch
        | RuntimeSessionError::PrefixOperationsUnsupported
        | RuntimeSessionError::ReleaseRetryNotIdOnly => ORBITKV_STATUS_INVALID_ARGUMENT,
        RuntimeSessionError::EvidenceTooLarge | RuntimeSessionError::IdentityExhausted(_) => {
            ORBITKV_STATUS_MANAGER_ERROR
        }
    };
    (status, error.to_string())
}

unsafe fn required_ref<'a, T>(value: *const T, label: &str) -> Result<&'a T, (i32, String)> {
    if value.is_null() {
        return invalid(&format!("{label} pointer is required"));
    }
    Ok(unsafe { &*value })
}

unsafe fn session_ref<'a>(
    session: *mut OrbitKvSessionHandle,
) -> Result<&'a OrbitKvSessionHandle, (i32, String)> {
    if session.is_null() {
        return invalid("session pointer is required");
    }
    Ok(unsafe { &*session })
}

fn lock_state(
    handle: &OrbitKvSessionHandle,
) -> Result<MutexGuard<'_, SessionState>, (i32, String)> {
    handle.state.lock().map_err(|_| {
        (
            ORBITKV_STATUS_FAIL_STOPPED,
            "runtime session lock is poisoned".to_owned(),
        )
    })
}

fn ensure_running(state: &SessionState) -> Result<(), (i32, String)> {
    if state.fail_stopped {
        return Err((
            ORBITKV_STATUS_FAIL_STOPPED,
            "runtime session is fail-stopped".to_owned(),
        ));
    }
    Ok(())
}

fn ensure_shared_cache(handle: &OrbitKvSessionHandle) -> Result<(), (i32, String)> {
    if handle.cache_sharing_policy == CacheSharingPolicy::RequestPrivate {
        return invalid(
            "runtime session cache-sharing policy does not support Prefix/share operations",
        );
    }
    Ok(())
}

fn maximum_batch(handle: &OrbitKvSessionHandle) -> u32 {
    handle.maximum_requests.min(handle.maximum_operations)
}

fn validate_session_epoch(
    handle: &OrbitKvSessionHandle,
    session_epoch: u64,
    label: &str,
) -> Result<(), (i32, String)> {
    if session_epoch != handle.session_epoch {
        return retryable(&format!("{label} belongs to a different runtime session"));
    }
    Ok(())
}

fn batch_id(value: OrbitKvSessionBatchId) -> EngineBatchId {
    EngineBatchId::from_parts(value.session_epoch, value.sequence)
}

fn publication_id(value: OrbitKvSessionPublicationId) -> EnginePublicationId {
    EnginePublicationId::from_parts(value.session_epoch, value.sequence)
}

fn release_id(value: OrbitKvSessionReleaseId) -> EngineReleaseId {
    EngineReleaseId::from_parts(value.session_epoch, value.sequence)
}

fn wire_batch_id(value: EngineBatchId) -> OrbitKvSessionBatchId {
    OrbitKvSessionBatchId {
        session_epoch: value.session_epoch(),
        sequence: value.sequence(),
    }
}

fn wire_publication_id(value: EnginePublicationId) -> OrbitKvSessionPublicationId {
    OrbitKvSessionPublicationId {
        session_epoch: value.session_epoch(),
        sequence: value.sequence(),
    }
}

fn wire_release_id(value: EngineReleaseId) -> OrbitKvSessionReleaseId {
    OrbitKvSessionReleaseId {
        session_epoch: value.session_epoch(),
        sequence: value.sequence(),
    }
}

fn wire_retirement(value: &EngineRetirement) -> OrbitKvSessionRetirement {
    OrbitKvSessionRetirement {
        page: value.page.into(),
        class_id: value.class_id,
        backend_domain: value.backend_domain,
        reserved32: 0,
        logical_ordinal: value.logical_ordinal,
        backend_index: value.backend_index,
        token_begin: value.token_begin,
        token_end_exclusive: value.token_end_exclusive,
        completion_domain: value.completion_domain,
        completion_value: value.completion_value,
    }
}

fn validate_retirement_evidence(
    certificates: &[EngineRetirement],
    evidence: &[OrbitKvSessionRetirementEvidence],
) -> Result<Box<[orbitkv::EngineRetirementEvidence]>, (i32, String)> {
    if evidence.len() != certificates.len() {
        return invalid("retirement evidence count must exactly match pending retirements");
    }
    certificates
        .iter()
        .zip(evidence)
        .map(|(certificate, item)| {
            if item.reserved8 != 0 || item.reserved32 != 0 {
                return invalid("retirement evidence reserved fields must be zero");
            }
            if item.acknowledged != 1 {
                return invalid("retirement evidence acknowledged field must be one");
            }
            let page: PageLease = item.page.into();
            if page != certificate.page
                || item.backend_domain != certificate.backend_domain
                || item.backend_index != certificate.backend_index
            {
                return invalid("retirement evidence does not match pending retirement");
            }
            Ok(orbitkv::EngineRetirementEvidence {
                page,
                backend_domain: item.backend_domain,
                acknowledged: true,
                backend_index: item.backend_index,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

fn validate_execution_spans(
    steps: &[OrbitKvSessionStepExecutionEvidence],
    bind_count: u32,
    copy_count: u32,
) -> Result<(), (i32, String)> {
    let mut next_bind = 0_u32;
    let mut next_copy = 0_u32;
    for step in steps {
        if step.reserved != 0 {
            return invalid("execution step reserved field must be zero");
        }
        if step.bind_offset != next_bind || step.copy_offset != next_copy {
            return invalid("execution evidence spans must be canonical and gap-free");
        }
        next_bind = next_bind.checked_add(step.bind_count).ok_or_else(|| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "bind span overflows".to_owned(),
            )
        })?;
        next_copy = next_copy.checked_add(step.copy_count).ok_or_else(|| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "copy span overflows".to_owned(),
            )
        })?;
    }
    if next_bind != bind_count || next_copy != copy_count {
        return invalid("execution evidence spans must cover flat buffers exactly");
    }
    Ok(())
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_session_create(
    plan_json: *const u8,
    plan_json_len: usize,
    config: *const layouts::OrbitKvSessionCreateConfig,
    backends: *const OrbitKvBackendArenaRegistration,
    backend_count: u32,
    out_session: *mut *mut OrbitKvSessionHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_session.is_null() {
            return invalid("session output pointer is required");
        }
        unsafe { out_session.write(std::ptr::null_mut()) };
        let config = *unsafe { required_ref(config, "session create config") }?;
        if plan_json.is_null() || plan_json_len == 0 || plan_json_len > MAX_PLAN_JSON_BYTES {
            return invalid("selected plan JSON is missing or exceeds 1 MiB");
        }
        let backends = unsafe { copy_input(backends, backend_count, "backend registration") }?;
        let plan_json = unsafe { slice::from_raw_parts(plan_json, plan_json_len) };
        let (manager, metadata) = compile_session_manager(plan_json, config, &backends)?;
        let cache_sharing_policy = metadata.cache_sharing_policy;
        let handle = Box::new(OrbitKvSessionHandle {
            state: Mutex::new(SessionState {
                runtime: RuntimeSession::new(manager, metadata.cache_sharing_policy),
                fail_stopped: false,
                batch_sizes: BTreeMap::new(),
                publications: BTreeMap::new(),
                releases: BTreeMap::new(),
                control_plans: BTreeMap::new(),
                relocations: BTreeMap::new(),
                resident_counts: BTreeMap::new(),
            }),
            cache_sharing_policy,
            session_epoch: metadata.session_epoch,
            total_page_capacity: metadata.total_page_capacity,
            maximum_requests: metadata.maximum_requests,
            maximum_operations: metadata.maximum_operations,
            maximum_prefixes: metadata.maximum_prefixes,
            class_count: metadata.class_count,
            maximum_write_intents_per_item: metadata.maximum_write_intents_per_item,
            maximum_completion_outputs_per_item: metadata.maximum_completion_outputs_per_item,
            arena_identities: metadata.arena_identities,
        });
        unsafe { out_session.write(Box::into_raw(handle)) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_arena_identities(
    session: *mut OrbitKvSessionHandle,
    identities: *mut OrbitKvArenaIdentity,
    identity_capacity: u32,
    out_identity_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let required = exact_len(handle.arena_identities.len());
        if unsafe {
            preflight_output(
                identities,
                identity_capacity,
                out_identity_count,
                required,
                "arena identity",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                handle.arena_identities.as_ptr(),
                identities,
                required as usize,
            );
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_arena_stats(
    session: *mut OrbitKvSessionHandle,
    stats: *mut OrbitKvArenaStats,
    stats_capacity: u32,
    out_stats_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let required = exact_len(handle.arena_identities.len());
        if unsafe {
            preflight_output(
                stats,
                stats_capacity,
                out_stats_count,
                required,
                "arena stats",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let values = lock_state(handle)?.runtime.arena_stats();
        assert_eq!(values.len(), required as usize);
        for (index, value) in values.iter().copied().enumerate() {
            unsafe { stats.add(index).write(value.into()) };
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_stats(
    session: *mut OrbitKvSessionHandle,
    out_stats: *mut OrbitKvManagerStats,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        if out_stats.is_null() {
            return invalid("session stats output pointer is required");
        }
        let stats = lock_state(handle)?.runtime.stats();
        unsafe { out_stats.write(stats.into()) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_session_acquire_requests(
    session: *mut OrbitKvSessionHandle,
    request_ids: *const u64,
    request_count: u32,
    views: *mut OrbitKvSessionRequestView,
    view_capacity: u32,
    out_view_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_nonzero_limit(request_count, handle.maximum_requests, "request")?;
        let request_ids = unsafe { copy_input(request_ids, request_count, "request id") }?;
        if unsafe {
            preflight_output(
                views,
                view_capacity,
                out_view_count,
                request_count,
                "request view",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let core_ids = request_ids
            .iter()
            .copied()
            .map(EngineRequestId)
            .collect::<Vec<_>>();
        let output = state
            .runtime
            .acquire_requests(&core_ids)
            .map_err(session_error)?;
        assert_eq!(output.len(), request_count as usize);
        for (index, value) in output.iter().copied().enumerate() {
            state
                .resident_counts
                .insert(value.request_id, value.resident_count);
            unsafe {
                views.add(index).write(OrbitKvSessionRequestView {
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

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_prepare_append(
    session: *mut OrbitKvSessionHandle,
    intents: *const OrbitKvSessionAppendIntent,
    intent_count: u32,
    out_batch_id: *mut OrbitKvSessionBatchId,
    steps: *mut OrbitKvSessionPreparedStep,
    step_capacity: u32,
    out_step_count: *mut u32,
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
        let handle = unsafe { session_ref(session) }?;
        if out_batch_id.is_null() {
            return invalid("batch id output is required");
        }
        unsafe { out_batch_id.write(OrbitKvSessionBatchId::default()) };
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_nonzero_limit(intent_count, maximum_batch(handle), "append intent")?;
        let intents = unsafe { copy_input(intents, intent_count, "append intent") }?;
        let class_bound = checked_mul(intent_count, handle.class_count, "class lowering")?;
        let tail_bound = class_bound;
        let copy_bound = class_bound.min(handle.total_page_capacity);
        let write_bound = checked_mul(
            intent_count,
            handle.maximum_write_intents_per_item,
            "write intent",
        )?
        .min(handle.total_page_capacity);
        let shorts = [
            unsafe {
                preflight_output(
                    steps,
                    step_capacity,
                    out_step_count,
                    intent_count,
                    "prepared step",
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
        let core = intents
            .iter()
            .map(|intent| EngineAppendIntent {
                request_id: EngineRequestId(intent.request_id),
                target_boundary: intent.target_boundary,
            })
            .collect::<Vec<_>>();
        let output = state
            .runtime
            .prepare_append_batch(&core)
            .map_err(session_error)?;
        let mut class_offset = 0_u32;
        let mut tail_offset = 0_u32;
        let mut copy_offset = 0_u32;
        let mut write_offset = 0_u32;
        for (index, step) in output.steps.iter().enumerate() {
            let class_count = exact_len(step.class_lowerings.len());
            let tail_count = exact_len(step.tail_actions.len());
            let copy_count = exact_len(step.copy_intents.len());
            let write_count = exact_len(step.write_intents.len());
            assert!(class_offset + class_count <= class_bound);
            assert!(tail_offset + tail_count <= tail_bound);
            assert!(copy_offset + copy_count <= copy_bound);
            assert!(write_offset + write_count <= write_bound);
            for (class_index, lowering) in step.class_lowerings.iter().copied().enumerate() {
                unsafe {
                    class_lowerings
                        .add(class_offset as usize + class_index)
                        .write(OrbitKvClassLowering {
                            class_id: lowering.class_id,
                            flags: lowering.flags,
                            tail_offset: lowering.tail_offset + tail_offset,
                            tail_count: lowering.tail_count,
                            copy_offset: lowering.copy_offset + copy_offset,
                            copy_count: lowering.copy_count,
                            write_offset: lowering.write_offset + write_offset,
                            write_count: lowering.write_count,
                            reserved: 0,
                            previous_layout_boundary: lowering.previous_layout_boundary,
                            target_layout_boundary: lowering.target_layout_boundary,
                        });
                }
            }
            for (item_index, item) in step.tail_actions.iter().copied().enumerate() {
                unsafe {
                    tail_actions
                        .add(tail_offset as usize + item_index)
                        .write(item.into());
                }
            }
            for (item_index, item) in step.copy_intents.iter().copied().enumerate() {
                unsafe {
                    copy_intents
                        .add(copy_offset as usize + item_index)
                        .write(item.into());
                }
            }
            for (item_index, item) in step.write_intents.iter().copied().enumerate() {
                unsafe {
                    write_intents
                        .add(write_offset as usize + item_index)
                        .write(item.into());
                }
            }
            unsafe {
                steps.add(index).write(OrbitKvSessionPreparedStep {
                    request_id: step.request_id.0,
                    base_view_version: step.base_view_version.0,
                    target_view_version: step.target_view_version.0,
                    previous_boundary: step.previous_boundary,
                    target_boundary: step.target_boundary,
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
        state.batch_sizes.insert(output.batch_id, intent_count);
        unsafe {
            out_batch_id.write(wire_batch_id(output.batch_id));
            out_class_count.write(class_offset);
            out_tail_count.write(tail_offset);
            out_copy_count.write(copy_offset);
            out_write_count.write(write_offset);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_submit_execution(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionBatchId,
    steps: *const OrbitKvSessionStepExecutionEvidence,
    step_count: u32,
    binds: *const OrbitKvSessionBindEvidence,
    bind_count: u32,
    copies: *const OrbitKvSessionCopyEvidence,
    copy_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_session_epoch(handle, id.session_epoch, "batch id")?;
        validate_nonzero_limit(step_count, maximum_batch(handle), "execution step")?;
        let bind_bound = checked_mul(
            step_count,
            handle
                .maximum_write_intents_per_item
                .checked_add(handle.class_count)
                .ok_or_else(|| {
                    (
                        ORBITKV_STATUS_INVALID_ARGUMENT,
                        "bind evidence bound overflows".to_owned(),
                    )
                })?,
            "bind evidence",
        )?
        .min(handle.total_page_capacity);
        let copy_bound = checked_mul(step_count, handle.class_count, "copy evidence")?
            .min(handle.total_page_capacity);
        validate_count_limit(bind_count, bind_bound, "bind evidence")?;
        validate_count_limit(copy_count, copy_bound, "copy evidence")?;
        let steps = unsafe { copy_input(steps, step_count, "execution step") }?;
        let binds = unsafe { copy_input(binds, bind_count, "bind evidence") }?;
        let copies = unsafe { copy_input(copies, copy_count, "copy evidence") }?;
        validate_execution_spans(&steps, bind_count, copy_count)?;
        if binds
            .iter()
            .any(|item| item.reserved != 0 || item.mapped > 1 || item.writable > 1)
        {
            return invalid("bind evidence has invalid boolean or reserved fields");
        }
        if copies.iter().any(|item| {
            item.reserved8 != 0
                || item.reserved32 != 0
                || item.observed > 1
                || item.copied > 1
                || item.ordered_before_writes > 1
        }) {
            return invalid("copy evidence has invalid boolean or reserved fields");
        }
        let mut grouped = Vec::with_capacity(steps.len());
        for step in &steps {
            let bind_begin = step.bind_offset as usize;
            let bind_end = bind_begin + step.bind_count as usize;
            let copy_begin = step.copy_offset as usize;
            let copy_end = copy_begin + step.copy_count as usize;
            let bind_receipts = binds[bind_begin..bind_end]
                .iter()
                .map(|item| EngineBindEvidence {
                    page: item.page.into(),
                    backend_domain: item.backend_domain,
                    mapped: item.mapped == 1,
                    writable: item.writable == 1,
                    backend_index: item.backend_index,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let copy_receipts = copies[copy_begin..copy_end]
                .iter()
                .map(|item| EngineCopyEvidence {
                    class_id: item.class_id,
                    backend_domain: item.backend_domain,
                    token_count: item.token_count,
                    source_token_offset: item.source_token_offset,
                    destination_token_offset: item.destination_token_offset,
                    observed: item.observed == 1,
                    copied: item.copied == 1,
                    ordered_before_writes: item.ordered_before_writes == 1,
                    source: item.source.into(),
                    destination: item.destination.into(),
                    source_backend_index: item.source_backend_index,
                    destination_backend_index: item.destination_backend_index,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            grouped.push(EngineStepExecutionEvidence {
                request_id: EngineRequestId(step.request_id),
                bind_receipts,
                copy_receipts,
            });
        }
        let core_id = batch_id(id);
        if state.batch_sizes.get(&core_id).copied() != Some(step_count) {
            return retryable("batch id is unknown, stale, or has a different step count");
        }
        let result = state.runtime.submit_execution(&ExecutionEvidence {
            batch_id: core_id,
            steps: grouped.into_boxed_slice(),
        });
        match result {
            Ok(_) => Ok(ORBITKV_STATUS_OK),
            Err(error @ RuntimeSessionError::Manager(KvManagerError::BatchQuarantined(_))) => {
                state.batch_sizes.remove(&core_id);
                state.fail_stopped = true;
                Err(session_error(error))
            }
            Err(error) => Err(session_error(error)),
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_abort_prepared(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionBatchId,
    evidence: *const OrbitKvSessionStepAbortEvidence,
    evidence_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_session_epoch(handle, id.session_epoch, "batch id")?;
        validate_nonzero_limit(evidence_count, maximum_batch(handle), "abort evidence")?;
        let evidence = unsafe { copy_input(evidence, evidence_count, "abort evidence") }?;
        if evidence
            .iter()
            .any(|item| item.reserved != 0 || item.backend_unobserved > 1)
        {
            return invalid("abort evidence has invalid boolean or reserved fields");
        }
        let core = evidence
            .iter()
            .map(|item| EngineStepAbortEvidence {
                request_id: EngineRequestId(item.request_id),
                backend_unobserved: item.backend_unobserved == 1,
            })
            .collect::<Vec<_>>();
        let core_id = batch_id(id);
        if state.batch_sizes.get(&core_id).copied() != Some(evidence_count) {
            return retryable("batch id is unknown, stale, or has a different step count");
        }
        state
            .runtime
            .abort_prepared_execution(core_id, &core)
            .map_err(session_error)?;
        state.batch_sizes.remove(&core_id);
        Ok(ORBITKV_STATUS_OK)
    })
}

fn quarantine_result(
    result: Result<(), RuntimeSessionError>,
    state: &mut SessionState,
    id: EngineBatchId,
    phase: &str,
) -> Result<i32, (i32, String)> {
    result.map_err(session_error)?;
    state.batch_sizes.remove(&id);
    state.fail_stopped = true;
    Err((
        ORBITKV_STATUS_FAIL_STOPPED,
        format!("{phase} batch was quarantined; the session must be fail-stopped"),
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_quarantine_prepared(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionBatchId,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_session_epoch(handle, id.session_epoch, "batch id")?;
        let core_id = batch_id(id);
        if !state.batch_sizes.contains_key(&core_id) {
            return retryable("batch id is unknown or stale");
        }
        let result = state.runtime.quarantine_prepared_execution(core_id);
        quarantine_result(result, &mut state, core_id, "prepared")
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_quarantine_submitted(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionBatchId,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_session_epoch(handle, id.session_epoch, "batch id")?;
        let core_id = batch_id(id);
        if !state.batch_sizes.contains_key(&core_id) {
            return retryable("batch id is unknown or stale");
        }
        let result = state.runtime.quarantine_submitted_execution(core_id);
        quarantine_result(result, &mut state, core_id, "submitted")
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_complete_execution(
    session: *mut OrbitKvSessionHandle,
    id: OrbitKvSessionBatchId,
    evidence: OrbitKvSessionCompletionEvidence,
    out_publication_id: *mut OrbitKvSessionPublicationId,
    steps: *mut OrbitKvSessionStepPublication,
    step_capacity: u32,
    out_step_count: *mut u32,
    detached: *mut OrbitKvDetachedBinding,
    detached_capacity: u32,
    out_detached_count: *mut u32,
    retirements: *mut OrbitKvSessionRetirement,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        if out_publication_id.is_null() {
            return invalid("publication id output is required");
        }
        unsafe { out_publication_id.write(OrbitKvSessionPublicationId::default()) };
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_session_epoch(handle, id.session_epoch, "batch id")?;
        if evidence.reserved != 0 {
            return invalid("completion evidence reserved field must be zero");
        }
        let confirmed = validate_bool(evidence.confirmed, "completion confirmed")?;
        let core_id = batch_id(id);
        let step_bound = state.batch_sizes.get(&core_id).copied().ok_or_else(|| {
            (
                ORBITKV_STATUS_RETRYABLE_CONFLICT,
                "batch id is unknown or stale".to_owned(),
            )
        })?;
        let detached_bound = checked_mul(
            step_bound,
            handle.maximum_completion_outputs_per_item,
            "completion output",
        )?;
        let retirement_bound = detached_bound.min(handle.total_page_capacity);
        let shorts = [
            unsafe {
                preflight_output(
                    steps,
                    step_capacity,
                    out_step_count,
                    step_bound,
                    "publication step",
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
                    "retirement",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let output = state
            .runtime
            .complete_execution_by_batch(
                core_id,
                EngineCompletionEvidence {
                    completion_domain: evidence.completion_domain,
                    completion_value: evidence.completion_value,
                    confirmed,
                },
            )
            .map_err(session_error)?;
        assert_eq!(output.steps.len(), step_bound as usize);
        let mut detached_offset = 0_u32;
        for (index, step) in output.steps.iter().enumerate() {
            let detached_count = exact_len(step.detached.len());
            assert!(detached_offset + detached_count <= detached_bound);
            for (item_index, item) in step.detached.iter().copied().enumerate() {
                unsafe {
                    detached
                        .add(detached_offset as usize + item_index)
                        .write(item.into());
                }
            }
            unsafe {
                steps.add(index).write(OrbitKvSessionStepPublication {
                    request_id: step.request_id.0,
                    view_version: step.view_version.0,
                    boundary: step.boundary,
                    resident_count: step.resident_count,
                    detached_offset,
                    detached_count,
                    reserved: 0,
                });
            }
            state
                .resident_counts
                .insert(step.request_id, step.resident_count);
            detached_offset += detached_count;
        }
        let retirement_count = exact_len(output.retirements.len());
        assert!(retirement_count <= retirement_bound);
        for (index, item) in output.retirements.iter().enumerate() {
            unsafe { retirements.add(index).write(wire_retirement(item)) };
        }
        state.batch_sizes.remove(&core_id);
        state
            .publications
            .insert(output.publication_id, output.retirements.clone());
        unsafe {
            out_publication_id.write(wire_publication_id(output.publication_id));
            out_detached_count.write(detached_offset);
            out_retirement_count.write(retirement_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_confirm_publication(
    session: *mut OrbitKvSessionHandle,
    evidence: OrbitKvSessionPublicationEvidence,
    retirements: *const OrbitKvSessionRetirementEvidence,
    retirement_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_session_epoch(
            handle,
            evidence.publication_id.session_epoch,
            "publication id",
        )?;
        if evidence.reserved != 0 {
            return invalid("publication evidence reserved field must be zero");
        }
        let mirror_cleanup_confirmed = validate_bool(
            evidence.mirror_cleanup_confirmed,
            "publication mirror cleanup confirmation",
        )?;
        validate_count_limit(
            retirement_count,
            handle.total_page_capacity,
            "retirement evidence",
        )?;
        let retirements =
            unsafe { copy_input(retirements, retirement_count, "retirement evidence") }?;
        let core_id = publication_id(evidence.publication_id);
        let pending = state.publications.get(&core_id).ok_or_else(|| {
            (
                ORBITKV_STATUS_RETRYABLE_CONFLICT,
                "publication id is unknown or stale".to_owned(),
            )
        })?;
        let receipts = validate_retirement_evidence(pending, &retirements)?;
        state
            .runtime
            .confirm_publication(&EnginePublicationEvidence {
                publication_id: core_id,
                mirror_cleanup_confirmed,
                reclamation_receipts: receipts,
            })
            .map_err(session_error)?;
        state.publications.remove(&core_id);
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_session_prepare_release(
    session: *mut OrbitKvSessionHandle,
    request_ids: *const u64,
    request_count: u32,
    out_release_id: *mut OrbitKvSessionReleaseId,
    releases: *mut OrbitKvSessionReleasedRequest,
    release_capacity: u32,
    out_release_count: *mut u32,
    detached: *mut OrbitKvDetachedBinding,
    detached_capacity: u32,
    out_detached_count: *mut u32,
    retirements: *mut OrbitKvSessionRetirement,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        if out_release_id.is_null() {
            return invalid("release id output is required");
        }
        unsafe { out_release_id.write(OrbitKvSessionReleaseId::default()) };
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_nonzero_limit(request_count, handle.maximum_requests, "release request")?;
        let request_ids = unsafe { copy_input(request_ids, request_count, "request id") }?;
        let core_ids = request_ids
            .iter()
            .copied()
            .map(EngineRequestId)
            .collect::<Vec<_>>();
        let detached_bound = core_ids.iter().try_fold(0_u32, |sum, request_id| {
            let count = state.resident_counts.get(request_id).ok_or_else(|| {
                (
                    ORBITKV_STATUS_RETRYABLE_CONFLICT,
                    format!("unknown engine request id {request_id:?}"),
                )
            })?;
            sum.checked_add(*count).ok_or_else(|| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    "release detached bound overflows".to_owned(),
                )
            })
        })?;
        let retirement_bound = detached_bound.min(handle.total_page_capacity);
        let shorts = [
            unsafe {
                preflight_output(
                    releases,
                    release_capacity,
                    out_release_count,
                    request_count,
                    "released request",
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
                    "retirement",
                )?
            },
        ];
        if shorts.into_iter().any(|short| short) {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let output = state
            .runtime
            .prepare_release_batch(&core_ids)
            .map_err(session_error)?;
        assert_eq!(output.releases.len(), request_count as usize);
        let mut detached_offset = 0_u32;
        for (index, release) in output.releases.iter().enumerate() {
            let detached_count = exact_len(release.detached.len());
            assert!(detached_offset + detached_count <= detached_bound);
            for (item_index, item) in release.detached.iter().copied().enumerate() {
                unsafe {
                    detached
                        .add(detached_offset as usize + item_index)
                        .write(item.into());
                }
            }
            unsafe {
                releases.add(index).write(OrbitKvSessionReleasedRequest {
                    request_id: release.request_id.0,
                    detached_offset,
                    detached_count,
                });
            }
            detached_offset += detached_count;
        }
        let retirement_count = exact_len(output.retirements.len());
        assert!(retirement_count <= retirement_bound);
        for (index, item) in output.retirements.iter().enumerate() {
            unsafe { retirements.add(index).write(wire_retirement(item)) };
        }
        state.releases.insert(
            output.release_id,
            PendingReleaseWire {
                requests: core_ids.into_boxed_slice(),
                retirements: output.retirements.clone(),
            },
        );
        unsafe {
            out_release_id.write(wire_release_id(output.release_id));
            out_detached_count.write(detached_offset);
            out_retirement_count.write(retirement_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_confirm_release(
    session: *mut OrbitKvSessionHandle,
    evidence: OrbitKvSessionReleaseEvidence,
    retirements: *const OrbitKvSessionRetirementEvidence,
    retirement_count: u32,
    out_outcome: *mut OrbitKvSessionReleaseOutcome,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_outcome.is_null() {
            return invalid("release outcome output is required");
        }
        unsafe { out_outcome.write(OrbitKvSessionReleaseOutcome::default()) };
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        validate_session_epoch(handle, evidence.release_id.session_epoch, "release id")?;
        if evidence.reserved != 0 {
            return invalid("release evidence reserved field must be zero");
        }
        let mirror_cleanup_confirmed = validate_bool(
            evidence.mirror_cleanup_confirmed,
            "release mirror cleanup confirmation",
        )?;
        validate_count_limit(
            retirement_count,
            handle.total_page_capacity,
            "retirement evidence",
        )?;
        let retirements =
            unsafe { copy_input(retirements, retirement_count, "retirement evidence") }?;
        let core_id = release_id(evidence.release_id);
        let pending = state.releases.get(&core_id).ok_or_else(|| {
            (
                ORBITKV_STATUS_RETRYABLE_CONFLICT,
                "release id is unknown or stale".to_owned(),
            )
        })?;
        let receipts = if retirements.is_empty() {
            Box::new([])
        } else {
            validate_retirement_evidence(&pending.retirements, &retirements)?
        };
        let requests = pending.requests.clone();
        let result = state.runtime.confirm_release(&EngineReleaseEvidence {
            release_id: core_id,
            mirror_cleanup_confirmed,
            reclamation_receipts: receipts,
        });
        let disposition = match result {
            Ok(EngineReleaseOutcome::Completed) => {
                for request_id in &requests {
                    state.resident_counts.remove(request_id);
                }
                state.releases.remove(&core_id);
                ORBITKV_SESSION_RELEASE_COMPLETED
            }
            Ok(EngineReleaseOutcome::RecyclePending) => ORBITKV_SESSION_RELEASE_RECYCLE_PENDING,
            Err(error @ RuntimeSessionError::SessionPoisoned(_)) => {
                state.fail_stopped = true;
                return Err(session_error(error));
            }
            Err(error) => return Err(session_error(error)),
        };
        unsafe {
            out_outcome.write(OrbitKvSessionReleaseOutcome {
                release_id: wire_release_id(core_id),
                disposition,
                reserved: 0,
            });
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Injects one post-ACK release recycle failure into a test-support build.
///
/// This symbol is intentionally absent from default production builds and the
/// public production header.
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_test_inject_release_recycle_once(
    session: *mut OrbitKvSessionHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { session_ref(session) }?;
        let mut state = lock_state(handle)?;
        ensure_running(&state)?;
        state
            .runtime
            .inject_test_fault(RuntimeSessionTestFault::ReleaseRecycleOnce);
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Destroys an exclusively owned runtime session. Null is a successful no-op.
///
/// # Safety
/// A non-null handle must be live and exclusively owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_session_destroy(
    session: *mut OrbitKvSessionHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if session.is_null() {
            return Ok(ORBITKV_STATUS_OK);
        }
        unsafe { drop(Box::from_raw(session)) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[cfg(test)]
mod error_mapping_tests {
    use super::*;
    use orbitkv::{EngineControlId, EnginePrefixId};

    #[test]
    fn prefix_and_control_errors_have_stable_status_classes() {
        let prefix = EnginePrefixId::from_parts(1, 1);
        let control = EngineControlId::from_parts(1, 1);
        for error in [
            RuntimeSessionError::UnknownPrefix(prefix),
            RuntimeSessionError::ForeignPrefix(prefix),
            RuntimeSessionError::StalePrefix(prefix),
            RuntimeSessionError::PrefixNotReady {
                prefix_id: prefix,
                state: "reserved",
            },
            RuntimeSessionError::UnknownControl(control),
            RuntimeSessionError::ForeignControl(control),
            RuntimeSessionError::StaleControl(control),
            RuntimeSessionError::ControlAlreadyCommitted(control),
            RuntimeSessionError::ControlNotCommitted(control),
            RuntimeSessionError::CanceledRequestNotPending(control),
        ] {
            assert_eq!(session_error(error).0, ORBITKV_STATUS_RETRYABLE_CONFLICT);
        }
        for error in [
            RuntimeSessionError::DuplicatePrefix(prefix),
            RuntimeSessionError::PrefixOperationsUnsupported,
            RuntimeSessionError::MirrorUpdatesNotConfirmed,
            RuntimeSessionError::ControlNotCancelable(control),
            RuntimeSessionError::PendingAttachCancelMismatch(control),
        ] {
            assert_eq!(session_error(error).0, ORBITKV_STATUS_INVALID_ARGUMENT);
        }
    }
}
