use std::ffi::c_char;
use std::slice;
use std::sync::Mutex;

use orbitkv::kv_manager::{
    ArenaStats, BackendArenaRegistration, BackendBindReceipt, BackendUnobservedReceipt,
    BatchCompletionReceipt, CanonicalKvManager, ManagerConfig, ManagerStats, PageLease,
    PrepareBatchItem, ReclamationCertificate, ReclamationLease, ReclamationReceipt, RequestLease,
    StepLease, SubmissionLease, SubmitBatchItem,
};
use orbitkv::plan::RetentionKind;
use orbitkv::{KvPlanInput, compile_plan};

use crate::{
    ORBITKV_ABI_VERSION, ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_INVALID_ARGUMENT,
    ORBITKV_STATUS_MANAGER_ERROR, ORBITKV_STATUS_OK, ffi_boundary,
};

const MAX_PLAN_JSON_BYTES: usize = 1024 * 1024;
pub const ORBITKV_CLASS_LOWERING_HAS_PREVIOUS_TAIL: u16 = 1 << 0;

pub struct OrbitKvManagerHandle {
    manager: Mutex<CanonicalKvManager>,
    total_page_capacity: u32,
    maximum_requests: u32,
    maximum_operations: u32,
    class_count: u32,
    maximum_write_intents_per_item: u32,
    maximum_retirements_per_item: u32,
    arena_identities: Box<[OrbitKvArenaIdentity]>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvRequestLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStepLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSubmissionLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReclamationLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPageLease {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub generation: u64,
    pub page_id: u32,
    pub pool_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBackendArenaRegistration {
    pub pool_id: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub page_count: u32,
    pub reserved: u32,
    pub backend_base_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvManagerConfig {
    pub maximum_requests: u32,
    pub maximum_operations: u32,
    pub maximum_reclamations: u32,
    pub maximum_step_tokens: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvArenaIdentity {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub backend_base_index: u64,
    pub pool_id: u32,
    pub page_count: u32,
    pub page_tokens: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub first_page_id: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvArenaStats {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub pool_id: u32,
    pub page_count: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub first_page_id: u32,
    pub reserved: u32,
    pub reserved_padding: u32,
    pub free_pages: u64,
    pub reserved_pages: u64,
    pub writing_pages: u64,
    pub active_pages: u64,
    pub retiring_pages: u64,
    pub quarantined_pages: u64,
    pub exhausted_pages: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPrepareBatchItem {
    pub request: OrbitKvRequestLease,
    pub target_boundary: u64,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPreparedBatchItem {
    pub step: OrbitKvStepLease,
    pub request: OrbitKvRequestLease,
    pub base_view_version: u64,
    pub target_view_version: u64,
    pub previous_boundary: u64,
    pub target_boundary: u64,
    pub class_offset: u32,
    pub class_count: u32,
    pub write_offset: u32,
    pub write_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvClassLowering {
    pub class_id: u16,
    pub flags: u16,
    pub write_offset: u32,
    pub write_count: u32,
    pub previous_tail_page_id: u32,
    pub previous_tail_generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvWriteIntent {
    pub page_generation: u64,
    pub page_id: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBackendBindReceipt {
    pub step: OrbitKvStepLease,
    pub page: OrbitKvPageLease,
    pub backend_domain: u16,
    pub mapped: u8,
    pub writable: u8,
    pub reserved: u32,
    pub backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSubmitBatchItem {
    pub step: OrbitKvStepLease,
    pub receipt_offset: u32,
    pub receipt_count: u32,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSubmittedBatchItem {
    pub submission: OrbitKvSubmissionLease,
    pub request: OrbitKvRequestLease,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBatchCompletionReceipt {
    pub engine_epoch: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
    pub confirmed: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvCompleteBatchItem {
    pub submission: OrbitKvSubmissionLease,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReclamationCertificate {
    pub reclamation: OrbitKvReclamationLease,
    pub request: OrbitKvRequestLease,
    pub page: OrbitKvPageLease,
    pub class_id: u16,
    pub backend_domain: u16,
    pub reserved32: u32,
    pub logical_ordinal: u64,
    pub backend_index: u64,
    pub token_begin: u64,
    pub token_end_exclusive: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvCompletedBatchItem {
    pub submission: OrbitKvSubmissionLease,
    pub request: OrbitKvRequestLease,
    pub published_view_version: u64,
    pub published_boundary: u64,
    pub resident_count: u32,
    pub retirement_offset: u32,
    pub retirement_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReleaseBatchItem {
    pub request: OrbitKvRequestLease,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReleasedBatchItem {
    pub request: OrbitKvRequestLease,
    pub retirement_offset: u32,
    pub retirement_count: u32,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBackendUnobservedReceipt {
    pub step: OrbitKvStepLease,
    pub backend_unobserved: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvReclamationReceipt {
    pub reclamation: OrbitKvReclamationLease,
    pub page: OrbitKvPageLease,
    pub backend_domain: u16,
    pub acknowledged: u8,
    pub reserved8: u8,
    pub reserved32: u32,
    pub backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvManagerStats {
    pub active_requests: u64,
    pub prepared_steps: u64,
    pub submitted_steps: u64,
    pub free_pages: u64,
    pub reserved_pages: u64,
    pub writing_pages: u64,
    pub active_pages: u64,
    pub retiring_pages: u64,
    pub quarantined_pages: u64,
    pub exhausted_pages: u64,
    pub pending_reclamations: u64,
}

#[unsafe(no_mangle)]
pub extern "C" fn orbitkv_abi_version() -> u32 {
    ORBITKV_ABI_VERSION
}

/// Creates the canonical manager from one strict `KvPlanInput` JSON and one
/// physical backend arena per compiled attention class.
///
/// # Safety
///
/// The JSON pointer must reference `plan_json_len` readable bytes. Config and
/// output pointers must reference their complete objects. `backends` must
/// reference `backend_count` readable registrations.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_lines)]
pub unsafe extern "C" fn orbitkv_manager_create(
    plan_json: *const u8,
    plan_json_len: usize,
    config: *const OrbitKvManagerConfig,
    backends: *const OrbitKvBackendArenaRegistration,
    backend_count: u32,
    out_manager: *mut *mut OrbitKvManagerHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if out_manager.is_null() {
            return invalid("manager output pointer is required");
        }
        unsafe {
            out_manager.write(std::ptr::null_mut());
        }
        if plan_json.is_null() || plan_json_len == 0 || plan_json_len > MAX_PLAN_JSON_BYTES {
            return invalid("canonical KvPlanInput JSON is missing or exceeds 1 MiB");
        }
        let config = unsafe { required_ref(config, "manager config") }?;
        if backend_count == 0 {
            return invalid("at least one backend arena registration is required");
        }
        let bytes = unsafe { slice::from_raw_parts(plan_json, plan_json_len) };
        let input = serde_json::from_slice::<KvPlanInput>(bytes).map_err(|error| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                format!("invalid canonical KvPlanInput JSON: {error}"),
            )
        })?;
        let plan = compile_plan(input).map_err(|error| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                format!("invalid canonical KV plan: {error}"),
            )
        })?;
        let expected_backend_count = u32::try_from(plan.classes.len()).map_err(|_| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "compiled attention-class count does not fit the canonical ABI".to_owned(),
            )
        })?;
        if backend_count != expected_backend_count {
            return invalid("backend arena count must exactly match compiled attention classes");
        }
        let maximum_new_pages_per_class =
            u64::from(config.maximum_step_tokens).div_ceil(plan.page_tokens);
        let sliding_class_count = u64::try_from(
            plan.classes
                .iter()
                .filter(|class| class.spec.retention == RetentionKind::Sliding)
                .count(),
        )
        .map_err(|_| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "sliding attention-class count does not fit the canonical ABI".to_owned(),
            )
        })?;
        let maximum_write_intents_per_item = u32::try_from(
            u64::from(expected_backend_count)
                .checked_mul(maximum_new_pages_per_class)
                .ok_or_else(|| {
                    (
                        ORBITKV_STATUS_INVALID_ARGUMENT,
                        "hot prepare output bound exceeds uint64_t".to_owned(),
                    )
                })?,
        )
        .map_err(|_| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "hot prepare output bound exceeds uint32_t".to_owned(),
            )
        })?;
        let maximum_retirements_per_item = u32::try_from(
            sliding_class_count
                .checked_mul(maximum_new_pages_per_class.checked_add(1).ok_or_else(|| {
                    (
                        ORBITKV_STATUS_INVALID_ARGUMENT,
                        "hot completion output bound exceeds uint64_t".to_owned(),
                    )
                })?)
                .ok_or_else(|| {
                    (
                        ORBITKV_STATUS_INVALID_ARGUMENT,
                        "hot completion output bound exceeds uint64_t".to_owned(),
                    )
                })?,
        )
        .map_err(|_| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "hot completion output bound exceeds uint32_t".to_owned(),
            )
        })?;
        let backends =
            unsafe { input_slice(backends, backend_count, "backend arena registration")? };
        let total_page_capacity = backends.iter().try_fold(0_u32, |total, backend| {
            total.checked_add(backend.page_count).ok_or_else(|| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    "total backend page capacity exceeds uint32_t".to_owned(),
                )
            })
        })?;
        let page_tokens = u32::try_from(plan.page_tokens).map_err(|_| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "canonical plan page_tokens does not fit the ABI".to_owned(),
            )
        })?;
        let core_backends = backends
            .iter()
            .copied()
            .map(BackendArenaRegistration::from)
            .collect::<Vec<_>>();
        let manager = CanonicalKvManager::new(&plan, (*config).into(), &core_backends)
            .map_err(manager_error)?;
        let arena_identities = manager
            .arena_stats()
            .iter()
            .map(|stats| {
                let backend = backends
                    .iter()
                    .find(|backend| backend.class_id == stats.class_id)
                    .ok_or_else(|| {
                        (
                            ORBITKV_STATUS_MANAGER_ERROR,
                            "core arena census lost a registered class".to_owned(),
                        )
                    })?;
                if stats.pool_id != backend.pool_id
                    || stats.backend_domain != backend.backend_domain
                    || stats.page_count != backend.page_count
                {
                    return Err((
                        ORBITKV_STATUS_MANAGER_ERROR,
                        "core arena census differs from its registration".to_owned(),
                    ));
                }
                Ok(OrbitKvArenaIdentity {
                    engine_epoch: stats.engine_epoch,
                    pool_epoch: stats.pool_epoch,
                    backend_base_index: backend.backend_base_index,
                    pool_id: stats.pool_id,
                    page_count: stats.page_count,
                    page_tokens,
                    class_id: stats.class_id,
                    backend_domain: stats.backend_domain,
                    first_page_id: stats.first_page_id,
                    reserved: 0,
                })
            })
            .collect::<Result<Vec<_>, (i32, String)>>()?;
        let handle = Box::new(OrbitKvManagerHandle {
            manager: Mutex::new(manager),
            total_page_capacity,
            maximum_requests: config.maximum_requests,
            maximum_operations: config.maximum_operations,
            class_count: expected_backend_count,
            maximum_write_intents_per_item,
            maximum_retirements_per_item,
            arena_identities: arena_identities.into_boxed_slice(),
        });
        unsafe {
            out_manager.write(Box::into_raw(handle));
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Copies all immutable manager/backend-arena identities in ascending class-id
/// order. A short buffer is left untouched and receives the required count.
///
/// # Safety
///
/// `manager` must be live. A non-null output buffer must reference
/// `identity_capacity` writable elements. `out_identity_count` must be
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_arena_identities(
    manager: *mut OrbitKvManagerHandle,
    identities: *mut OrbitKvArenaIdentity,
    identity_capacity: u32,
    out_identity_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        if out_identity_count.is_null() {
            return invalid("arena-identity count output is required");
        }
        let required = u32_len(handle.arena_identities.len(), "arena-identity count")?;
        unsafe {
            out_identity_count.write(required);
        }
        if identity_capacity < required {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        if required != 0 && identities.is_null() {
            return invalid("arena-identity output buffer is required");
        }
        for (index, identity) in handle.arena_identities.iter().copied().enumerate() {
            unsafe {
                identities.add(index).write(identity);
            }
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Copies a self-contained per-arena page-phase census in ascending class-id
/// order. A short buffer is left untouched and receives the required count.
/// This is the class-specific source for admission and terminal-state checks;
/// aggregate manager stats are telemetry only.
///
/// # Safety
///
/// `manager` must be live. A non-null output buffer must reference
/// `stats_capacity` writable elements. `out_stats_count` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_arena_stats(
    manager: *mut OrbitKvManagerHandle,
    stats: *mut OrbitKvArenaStats,
    stats_capacity: u32,
    out_stats_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        if out_stats_count.is_null() {
            return invalid("arena-stats count output is required");
        }
        let required = u32_len(handle.arena_identities.len(), "arena-stats count")?;
        unsafe {
            out_stats_count.write(required);
        }
        if stats_capacity < required {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        if required != 0 && stats.is_null() {
            return invalid("arena-stats output buffer is required");
        }
        let core_stats = lock_manager(handle)?.arena_stats();
        if core_stats.len() != handle.arena_identities.len()
            || !core_stats
                .iter()
                .zip(handle.arena_identities.iter())
                .all(|(stats, identity)| arena_stats_match_identity(stats, identity))
        {
            return Err((
                ORBITKV_STATUS_MANAGER_ERROR,
                "core arena census does not match registered arena identities".to_owned(),
            ));
        }
        for (index, value) in core_stats.iter().copied().enumerate() {
            unsafe {
                stats.add(index).write(value.into());
            }
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Acquires an ordered batch of generation-bearing request identities.
///
/// # Safety
///
/// Every non-null pointer must reference its declared writable capacity.
///
/// # Panics
///
/// An impossible core output cardinality panics while the manager lock is
/// held. The ABI boundary catches it and the poisoned handle fails closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_request_acquire_batch(
    manager: *mut OrbitKvManagerHandle,
    request_count: u32,
    requests: *mut OrbitKvRequestLease,
    request_capacity: u32,
    out_request_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        unsafe { clear_output_count(out_request_count, "request lease")? };
        validate_count_limit(request_count, handle.maximum_requests, "request")?;
        if unsafe {
            preflight_mutating_output(
                requests,
                request_capacity,
                out_request_count,
                request_count,
                "request lease",
            )?
        } {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let mut manager = lock_manager(handle)?;
        let acquired = manager
            .acquire_requests(usize::try_from(request_count).map_err(|_| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    "request count is too large".to_owned(),
                )
            })?)
            .map_err(manager_error)?;
        assert_eq!(acquired.len(), usize::try_from(request_count).unwrap());
        unsafe {
            for (index, request) in acquired.iter().copied().enumerate() {
                requests.add(index).write(request.into());
            }
            out_request_count.write(request_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Reserves exact pages for an ordered request batch.
///
/// Class outputs require one record per item and registered class. Write
/// intents are bounded by `item_count * class_count * ceil(max_step / P)`;
/// neither hot output exposes nor reserves space for the canonical root.
///
/// # Safety
///
/// Every pointer must reference its declared readable or writable capacity.
///
/// # Panics
///
/// An impossible core flat-span or cardinality invariant panics while the
/// manager lock is held. The ABI boundary catches it and the poisoned handle
/// fails closed.
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
    write_intents: *mut OrbitKvWriteIntent,
    write_capacity: u32,
    out_write_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        unsafe {
            clear_output_count(out_prepared_count, "prepared item")?;
            clear_output_count(out_class_count, "class lowering")?;
            clear_output_count(out_write_count, "write intent")?;
        }
        validate_count_limit(item_count, maximum_batch_items(handle), "prepare item")?;
        let required_classes =
            checked_output_bound(item_count, handle.class_count, "class lowering")?;
        let required_writes = hot_page_output_bound(
            handle,
            item_count,
            handle.maximum_write_intents_per_item,
            "write intent",
        )?;
        let item_short = unsafe {
            preflight_mutating_output(
                prepared,
                prepared_capacity,
                out_prepared_count,
                item_count,
                "prepared item",
            )?
        };
        let class_short = unsafe {
            preflight_mutating_output(
                class_lowerings,
                class_capacity,
                out_class_count,
                required_classes,
                "class lowering",
            )?
        };
        let write_short = unsafe {
            preflight_mutating_output(
                write_intents,
                write_capacity,
                out_write_count,
                required_writes,
                "write intent",
            )?
        };
        if item_short || class_short || write_short {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let items = unsafe { input_slice(items, item_count, "prepare item")? };
        let core_items = items
            .iter()
            .map(|item| {
                if item.reserved != 0 {
                    return invalid("prepare item reserved field must be zero");
                }
                Ok(PrepareBatchItem {
                    request: item.request.into(),
                    target_boundary: item.target_boundary,
                })
            })
            .collect::<Result<Vec<_>, (i32, String)>>()?;
        let mut manager = lock_manager(handle)?;
        let outputs = manager.prepare_batch(&core_items).map_err(manager_error)?;
        assert_eq!(outputs.len(), core_items.len());
        let mut flat_class_count = 0_u32;
        let mut flat_write_count = 0_u32;
        for output in &outputs {
            let item_class_count =
                u32::try_from(output.class_lowerings.len()).expect("core class count fits ABI");
            let item_write_count =
                u32::try_from(output.write_intents.len()).expect("core write count fits ABI");
            assert_eq!(item_class_count, handle.class_count);
            let mut local_write_offset = 0_u32;
            for (lowering, identity) in output
                .class_lowerings
                .iter()
                .zip(handle.arena_identities.iter())
            {
                assert_eq!(lowering.class_id, identity.class_id);
                assert_eq!(lowering.write_offset, local_write_offset);
                assert_eq!(
                    lowering.flags & !ORBITKV_CLASS_LOWERING_HAS_PREVIOUS_TAIL,
                    0
                );
                if lowering.flags & ORBITKV_CLASS_LOWERING_HAS_PREVIOUS_TAIL == 0 {
                    assert_eq!(lowering.previous_tail_page_id, 0);
                    assert_eq!(lowering.previous_tail_generation, 0);
                }
                local_write_offset = local_write_offset
                    .checked_add(lowering.write_count)
                    .expect("core class write range fits ABI");
            }
            assert_eq!(local_write_offset, item_write_count);
            assert!(
                output
                    .write_intents
                    .iter()
                    .all(|intent| intent.reserved == 0)
            );
            flat_class_count = flat_class_count
                .checked_add(item_class_count)
                .expect("core class count fits capacity");
            flat_write_count = flat_write_count
                .checked_add(item_write_count)
                .expect("core write count fits capacity");
        }
        assert_eq!(flat_class_count, required_classes);
        assert!(flat_write_count <= required_writes);
        let mut class_offset = 0_u32;
        let mut write_offset = 0_u32;
        for (index, output) in outputs.iter().enumerate() {
            let item_class_count = u32::try_from(output.class_lowerings.len())
                .expect("validated class count fits ABI");
            let item_write_count =
                u32::try_from(output.write_intents.len()).expect("validated write count fits ABI");
            unsafe {
                for (local_index, lowering) in output.class_lowerings.iter().enumerate() {
                    class_lowerings
                        .add(usize::try_from(class_offset).unwrap() + local_index)
                        .write(OrbitKvClassLowering {
                            class_id: lowering.class_id,
                            flags: lowering.flags,
                            write_offset: write_offset
                                .checked_add(lowering.write_offset)
                                .expect("validated global write offset"),
                            write_count: lowering.write_count,
                            previous_tail_page_id: lowering.previous_tail_page_id,
                            previous_tail_generation: lowering.previous_tail_generation,
                        });
                }
                for (local_index, intent) in output.write_intents.iter().enumerate() {
                    write_intents
                        .add(usize::try_from(write_offset).unwrap() + local_index)
                        .write(OrbitKvWriteIntent {
                            page_generation: intent.page_generation,
                            page_id: intent.page_id,
                            reserved: 0,
                        });
                }
                prepared.add(index).write(OrbitKvPreparedBatchItem {
                    step: output.step.into(),
                    request: output.request.into(),
                    base_view_version: output.base_view_version.0,
                    target_view_version: output.target_view_version.0,
                    previous_boundary: output.previous_boundary,
                    target_boundary: output.target_boundary,
                    class_offset,
                    class_count: item_class_count,
                    write_offset,
                    write_count: item_write_count,
                });
            }
            class_offset = class_offset
                .checked_add(item_class_count)
                .expect("validated class range");
            write_offset = write_offset
                .checked_add(item_write_count)
                .expect("validated write range");
        }
        unsafe {
            out_prepared_count.write(item_count);
            out_class_count.write(flat_class_count);
            out_write_count.write(flat_write_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Validates one canonical flat receipt partition and submits the whole batch.
///
/// # Safety
///
/// Every pointer must reference its declared readable or writable capacity.
///
/// # Panics
///
/// An impossible core flat-span or cardinality invariant panics while the
/// manager lock is held. The ABI boundary catches it and the poisoned handle
/// fails closed.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_submit_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvSubmitBatchItem,
    item_count: u32,
    receipts: *const OrbitKvBackendBindReceipt,
    receipt_count: u32,
    submitted: *mut OrbitKvSubmittedBatchItem,
    submitted_capacity: u32,
    out_submitted_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        unsafe { clear_output_count(out_submitted_count, "submitted item")? };
        validate_count_limit(item_count, maximum_batch_items(handle), "submit item")?;
        validate_count_limit(
            receipt_count,
            hot_page_output_bound(
                handle,
                item_count,
                handle.maximum_write_intents_per_item,
                "binding receipt",
            )?,
            "binding receipt",
        )?;
        let item_short = unsafe {
            preflight_mutating_output(
                submitted,
                submitted_capacity,
                out_submitted_count,
                item_count,
                "submitted item",
            )?
        };
        if item_short {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let items = unsafe { input_slice(items, item_count, "submit item")? };
        let receipts = unsafe { input_slice(receipts, receipt_count, "binding receipt")? };
        let core_items = items
            .iter()
            .map(|item| {
                if item.reserved != 0 {
                    return invalid("submit item reserved field must be zero");
                }
                Ok(SubmitBatchItem {
                    step: item.step.into(),
                    receipt_offset: item.receipt_offset,
                    receipt_count: item.receipt_count,
                })
            })
            .collect::<Result<Vec<_>, (i32, String)>>()?;
        let core_receipts = receipts
            .iter()
            .copied()
            .map(BackendBindReceipt::from)
            .collect::<Vec<_>>();
        let mut manager = lock_manager(handle)?;
        let outputs = manager
            .submit_batch(&core_items, &core_receipts)
            .map_err(manager_error)?;
        assert_eq!(outputs.len(), core_items.len());
        for (index, output) in outputs.iter().enumerate() {
            unsafe {
                submitted.add(index).write(OrbitKvSubmittedBatchItem {
                    submission: output.submission.into(),
                    request: output.request.into(),
                });
            }
        }
        unsafe {
            out_submitted_count.write(item_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Publishes an ordered batch at one shared completion point.
///
/// # Safety
///
/// Every pointer must reference its declared readable or writable capacity.
///
/// # Panics
///
/// An impossible core flat-span or cardinality invariant panics while the
/// manager lock is held. The ABI boundary catches it and the poisoned handle
/// fails closed.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_complete_batch(
    manager: *mut OrbitKvManagerHandle,
    receipt: OrbitKvBatchCompletionReceipt,
    items: *const OrbitKvCompleteBatchItem,
    item_count: u32,
    completed: *mut OrbitKvCompletedBatchItem,
    completed_capacity: u32,
    out_completed_count: *mut u32,
    retirements: *mut OrbitKvReclamationCertificate,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        unsafe {
            clear_output_count(out_completed_count, "completed item")?;
            clear_output_count(out_retirement_count, "reclamation certificate")?;
        }
        validate_count_limit(item_count, maximum_batch_items(handle), "completion item")?;
        let required_retirements = hot_page_output_bound(
            handle,
            item_count,
            handle.maximum_retirements_per_item,
            "reclamation certificate",
        )?;
        let item_short = unsafe {
            preflight_mutating_output(
                completed,
                completed_capacity,
                out_completed_count,
                item_count,
                "completed item",
            )?
        };
        let retirement_short = unsafe {
            preflight_mutating_output(
                retirements,
                retirement_capacity,
                out_retirement_count,
                required_retirements,
                "reclamation certificate",
            )?
        };
        if item_short || retirement_short {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let items = unsafe { input_slice(items, item_count, "completion item")? };
        let submissions = items
            .iter()
            .map(|item| item.submission.into())
            .collect::<Vec<_>>();
        let mut manager = lock_manager(handle)?;
        let outputs = manager
            .complete_batch(receipt.into(), &submissions)
            .map_err(manager_error)?;
        assert_eq!(outputs.len(), submissions.len());
        let mut retirement_count = 0_u32;
        for output in &outputs {
            let item_retirements = u32::try_from(output.retirements.len())
                .expect("core retirement count fits the canonical ABI");
            retirement_count = retirement_count
                .checked_add(item_retirements)
                .expect("core retirement count fits capacity");
            assert!(retirement_count <= required_retirements);
        }
        let mut retirement_offset = 0_u32;
        for (index, output) in outputs.iter().enumerate() {
            let item_retirements = u32::try_from(output.retirements.len())
                .expect("validated core retirement count fits the canonical ABI");
            unsafe {
                write_reclamation_certificates(
                    &output.retirements,
                    retirements.add(usize::try_from(retirement_offset).unwrap()),
                );
                completed.add(index).write(OrbitKvCompletedBatchItem {
                    submission: output.submission.into(),
                    request: output.request.into(),
                    published_view_version: output.publication.view_version.0,
                    published_boundary: output.publication.boundary,
                    resident_count: output.publication.resident_count,
                    retirement_offset,
                    retirement_count: item_retirements,
                    reserved: 0,
                });
            }
            retirement_offset = retirement_offset
                .checked_add(item_retirements)
                .expect("validated retirement range");
        }
        unsafe {
            out_completed_count.write(item_count);
            out_retirement_count.write(retirement_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Aborts a complete prepared batch only with backend-unobserved proofs.
///
/// # Safety
///
/// `receipts` must reference `receipt_count` readable elements when nonzero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_abort_steps(
    manager: *mut OrbitKvManagerHandle,
    receipts: *const OrbitKvBackendUnobservedReceipt,
    receipt_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        validate_count_limit(
            receipt_count,
            maximum_batch_items(handle),
            "unobserved receipt",
        )?;
        let receipts = unsafe { input_slice(receipts, receipt_count, "unobserved receipt")? };
        let core = receipts
            .iter()
            .copied()
            .map(BackendUnobservedReceipt::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .abort_steps(&core)
            .map_err(manager_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Quarantines an ordered prepared batch after ambiguous backend lowering.
///
/// # Safety
///
/// `steps` must reference `step_count` readable elements when nonzero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_quarantine_steps(
    manager: *mut OrbitKvManagerHandle,
    steps: *const OrbitKvStepLease,
    step_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        validate_count_limit(step_count, maximum_batch_items(handle), "step")?;
        let steps = unsafe { input_slice(steps, step_count, "step lease")? };
        let core = steps
            .iter()
            .copied()
            .map(StepLease::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .quarantine_steps(&core)
            .map_err(manager_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Quarantines an ordered submitted batch after ambiguous GPU execution.
///
/// # Safety
///
/// `submissions` must reference `submission_count` readable elements when
/// nonzero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_quarantine_submissions(
    manager: *mut OrbitKvManagerHandle,
    submissions: *const OrbitKvSubmissionLease,
    submission_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        validate_count_limit(submission_count, maximum_batch_items(handle), "submission")?;
        let submissions =
            unsafe { input_slice(submissions, submission_count, "submission lease")? };
        let core = submissions
            .iter()
            .copied()
            .map(SubmissionLease::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .quarantine_submissions(&core)
            .map_err(manager_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Releases an ordered batch of quiescent requests.
///
/// # Safety
///
/// Every pointer must reference its declared readable or writable capacity.
///
/// # Panics
///
/// An impossible core flat-span or cardinality invariant panics while the
/// manager lock is held. The ABI boundary catches it and the poisoned handle
/// fails closed.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn orbitkv_manager_release_batch(
    manager: *mut OrbitKvManagerHandle,
    items: *const OrbitKvReleaseBatchItem,
    item_count: u32,
    released: *mut OrbitKvReleasedBatchItem,
    released_capacity: u32,
    out_released_count: *mut u32,
    retirements: *mut OrbitKvReclamationCertificate,
    retirement_capacity: u32,
    out_retirement_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        unsafe {
            clear_output_count(out_released_count, "released item")?;
            clear_output_count(out_retirement_count, "reclamation certificate")?;
        }
        validate_count_limit(item_count, handle.maximum_requests, "release item")?;
        let item_short = unsafe {
            preflight_mutating_output(
                released,
                released_capacity,
                out_released_count,
                item_count,
                "released item",
            )?
        };
        let retirement_short = unsafe {
            preflight_mutating_output(
                retirements,
                retirement_capacity,
                out_retirement_count,
                handle.total_page_capacity,
                "reclamation certificate",
            )?
        };
        if item_short || retirement_short {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        }
        let items = unsafe { input_slice(items, item_count, "release item")? };
        let requests = items
            .iter()
            .map(|item| {
                if item.reserved != 0 {
                    return invalid("release item reserved field must be zero");
                }
                Ok(item.request.into())
            })
            .collect::<Result<Vec<RequestLease>, (i32, String)>>()?;
        let mut manager = lock_manager(handle)?;
        let outputs = manager.release_batch(&requests).map_err(manager_error)?;
        assert_eq!(outputs.len(), requests.len());
        let mut retirement_count = 0_u32;
        for output in &outputs {
            assert_eq!(output.retirement_offset, retirement_count);
            let item_count = u32::try_from(output.release.retirements.len())
                .expect("core retirement count fits the canonical ABI");
            retirement_count = retirement_count
                .checked_add(item_count)
                .expect("core retirement count fits capacity");
            assert!(retirement_count <= handle.total_page_capacity);
        }
        for (index, output) in outputs.iter().enumerate() {
            let item_retirement_count = u32::try_from(output.release.retirements.len())
                .expect("validated core retirement count fits the canonical ABI");
            unsafe {
                write_reclamation_certificates(
                    &output.release.retirements,
                    retirements.add(usize::try_from(output.retirement_offset).unwrap()),
                );
                released.add(index).write(OrbitKvReleasedBatchItem {
                    request: output.release.request.into(),
                    retirement_offset: output.retirement_offset,
                    retirement_count: item_retirement_count,
                    reserved: 0,
                });
            }
        }
        unsafe {
            out_released_count.write(item_count);
            out_retirement_count.write(retirement_count);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Atomically acknowledges an exact reclamation batch.
///
/// # Safety
///
/// `receipts` must reference `receipt_count` readable elements when nonzero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_acknowledge_reclamations(
    manager: *mut OrbitKvManagerHandle,
    receipts: *const OrbitKvReclamationReceipt,
    receipt_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        validate_count_limit(
            receipt_count,
            handle.total_page_capacity,
            "reclamation receipt",
        )?;
        let receipts = unsafe { input_slice(receipts, receipt_count, "reclamation receipt")? };
        let core_receipts = receipts
            .iter()
            .copied()
            .map(ReclamationReceipt::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .acknowledge_reclamations(&core_receipts)
            .map_err(manager_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Recycles an ordered batch after all reclamations are acknowledged.
///
/// # Safety
///
/// `requests` must reference `request_count` readable elements when nonzero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_recycle_requests(
    manager: *mut OrbitKvManagerHandle,
    requests: *const OrbitKvRequestLease,
    request_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager)? };
        validate_count_limit(request_count, handle.maximum_requests, "request")?;
        let requests = unsafe { input_slice(requests, request_count, "request lease")? };
        let core = requests
            .iter()
            .copied()
            .map(RequestLease::from)
            .collect::<Vec<_>>();
        lock_manager(handle)?
            .recycle_requests(&core)
            .map_err(manager_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Returns one fixed-width manager state snapshot.
///
/// # Safety
///
/// `manager` must be live and `out_stats` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_stats(
    manager: *mut OrbitKvManagerHandle,
    out_stats: *mut OrbitKvManagerStats,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        clear_fixed_output(out_stats, "manager-stats output")?;
        let handle = unsafe { manager_ref(manager)? };
        let stats = lock_manager(handle)?.stats();
        unsafe {
            out_stats.write(stats.into());
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Destroys one quiescent manager handle. Null is a successful no-op.
///
/// # Safety
///
/// A non-null pointer must have been returned by `orbitkv_manager_create` and
/// must not have been destroyed before. No other thread may use the handle
/// concurrently with this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_destroy(
    manager: *mut OrbitKvManagerHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if manager.is_null() {
            return Ok(ORBITKV_STATUS_OK);
        }
        let handle = unsafe { manager_ref(manager)? };
        let stats = lock_manager(handle)?.stats();
        if stats.active_requests != 0
            || stats.prepared_steps != 0
            || stats.submitted_steps != 0
            || stats.reserved_pages != 0
            || stats.writing_pages != 0
            || stats.active_pages != 0
            || stats.retiring_pages != 0
            || stats.quarantined_pages != 0
            || stats.pending_reclamations != 0
        {
            return Err((
                ORBITKV_STATUS_MANAGER_ERROR,
                "manager is not quiescent; handle was not destroyed".to_owned(),
            ));
        }
        unsafe {
            drop(Box::from_raw(manager));
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

impl From<OrbitKvRequestLease> for RequestLease {
    fn from(value: OrbitKvRequestLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<RequestLease> for OrbitKvRequestLease {
    fn from(value: RequestLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<OrbitKvStepLease> for StepLease {
    fn from(value: OrbitKvStepLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<StepLease> for OrbitKvStepLease {
    fn from(value: StepLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<OrbitKvSubmissionLease> for SubmissionLease {
    fn from(value: OrbitKvSubmissionLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<SubmissionLease> for OrbitKvSubmissionLease {
    fn from(value: SubmissionLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<OrbitKvReclamationLease> for ReclamationLease {
    fn from(value: OrbitKvReclamationLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<ReclamationLease> for OrbitKvReclamationLease {
    fn from(value: ReclamationLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<OrbitKvPageLease> for PageLease {
    fn from(value: OrbitKvPageLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            generation: value.generation,
            page_id: value.page_id,
            pool_id: value.pool_id,
        }
    }
}

impl From<PageLease> for OrbitKvPageLease {
    fn from(value: PageLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            generation: value.generation,
            page_id: value.page_id,
            pool_id: value.pool_id,
        }
    }
}

impl From<OrbitKvBackendArenaRegistration> for BackendArenaRegistration {
    fn from(value: OrbitKvBackendArenaRegistration) -> Self {
        Self {
            pool_id: value.pool_id,
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            page_count: value.page_count,
            reserved: value.reserved,
            backend_base_index: value.backend_base_index,
        }
    }
}

impl From<OrbitKvManagerConfig> for ManagerConfig {
    fn from(value: OrbitKvManagerConfig) -> Self {
        Self {
            maximum_requests: value.maximum_requests,
            maximum_operations: value.maximum_operations,
            maximum_reclamations: value.maximum_reclamations,
            maximum_step_tokens: value.maximum_step_tokens,
        }
    }
}

impl From<OrbitKvBackendBindReceipt> for BackendBindReceipt {
    fn from(value: OrbitKvBackendBindReceipt) -> Self {
        Self {
            step: value.step.into(),
            page: value.page.into(),
            backend_domain: value.backend_domain,
            mapped: value.mapped,
            writable: value.writable,
            reserved: value.reserved,
            backend_index: value.backend_index,
        }
    }
}

impl From<OrbitKvBatchCompletionReceipt> for BatchCompletionReceipt {
    fn from(value: OrbitKvBatchCompletionReceipt) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            completion_domain: value.completion_domain,
            completion_value: value.completion_value,
            confirmed: value.confirmed,
            reserved: value.reserved,
        }
    }
}

impl From<OrbitKvBackendUnobservedReceipt> for BackendUnobservedReceipt {
    fn from(value: OrbitKvBackendUnobservedReceipt) -> Self {
        Self {
            step: value.step.into(),
            backend_unobserved: value.backend_unobserved,
            reserved: value.reserved,
        }
    }
}

impl From<ReclamationCertificate> for OrbitKvReclamationCertificate {
    fn from(value: ReclamationCertificate) -> Self {
        Self {
            reclamation: value.reclamation.into(),
            request: value.request.into(),
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
}

impl From<OrbitKvReclamationReceipt> for ReclamationReceipt {
    fn from(value: OrbitKvReclamationReceipt) -> Self {
        Self {
            reclamation: value.reclamation.into(),
            page: value.page.into(),
            backend_domain: value.backend_domain,
            acknowledged: value.acknowledged,
            reserved8: value.reserved8,
            reserved32: value.reserved32,
            backend_index: value.backend_index,
        }
    }
}

impl From<ArenaStats> for OrbitKvArenaStats {
    fn from(value: ArenaStats) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            pool_id: value.pool_id,
            page_count: value.page_count,
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            first_page_id: value.first_page_id,
            reserved: 0,
            reserved_padding: 0,
            free_pages: value.free_pages,
            reserved_pages: value.reserved_pages,
            writing_pages: value.writing_pages,
            active_pages: value.active_pages,
            retiring_pages: value.retiring_pages,
            quarantined_pages: value.quarantined_pages,
            exhausted_pages: value.exhausted_pages,
        }
    }
}

impl From<ManagerStats> for OrbitKvManagerStats {
    fn from(value: ManagerStats) -> Self {
        Self {
            active_requests: value.active_requests,
            prepared_steps: value.prepared_steps,
            submitted_steps: value.submitted_steps,
            free_pages: value.free_pages,
            reserved_pages: value.reserved_pages,
            writing_pages: value.writing_pages,
            active_pages: value.active_pages,
            retiring_pages: value.retiring_pages,
            quarantined_pages: value.quarantined_pages,
            exhausted_pages: value.exhausted_pages,
            pending_reclamations: value.pending_reclamations,
        }
    }
}

const _: [(); std::mem::size_of::<OrbitKvRequestLease>()] =
    [(); std::mem::size_of::<RequestLease>()];
const _: [(); std::mem::align_of::<OrbitKvRequestLease>()] =
    [(); std::mem::align_of::<RequestLease>()];
const _: [(); std::mem::size_of::<OrbitKvStepLease>()] = [(); std::mem::size_of::<StepLease>()];
const _: [(); std::mem::size_of::<OrbitKvSubmissionLease>()] =
    [(); std::mem::size_of::<SubmissionLease>()];
const _: [(); std::mem::size_of::<OrbitKvReclamationLease>()] =
    [(); std::mem::size_of::<ReclamationLease>()];
const _: [(); std::mem::size_of::<OrbitKvPageLease>()] = [(); std::mem::size_of::<PageLease>()];
const _: [(); std::mem::align_of::<OrbitKvPageLease>()] = [(); std::mem::align_of::<PageLease>()];
const _: [(); std::mem::size_of::<OrbitKvBackendArenaRegistration>()] =
    [(); std::mem::size_of::<BackendArenaRegistration>()];
const _: [(); std::mem::size_of::<OrbitKvManagerConfig>()] =
    [(); std::mem::size_of::<ManagerConfig>()];
const _: [(); std::mem::size_of::<OrbitKvBackendBindReceipt>()] =
    [(); std::mem::size_of::<BackendBindReceipt>()];
const _: [(); std::mem::size_of::<OrbitKvBatchCompletionReceipt>()] =
    [(); std::mem::size_of::<BatchCompletionReceipt>()];
const _: [(); std::mem::size_of::<OrbitKvBackendUnobservedReceipt>()] =
    [(); std::mem::size_of::<BackendUnobservedReceipt>()];
const _: [(); std::mem::size_of::<OrbitKvReclamationReceipt>()] =
    [(); std::mem::size_of::<ReclamationReceipt>()];
const _: [(); 16] = [(); std::mem::size_of::<OrbitKvRequestLease>()];
const _: [(); 16] = [(); std::mem::size_of::<OrbitKvStepLease>()];
const _: [(); 16] = [(); std::mem::size_of::<OrbitKvSubmissionLease>()];
const _: [(); 16] = [(); std::mem::size_of::<OrbitKvReclamationLease>()];
const _: [(); 32] = [(); std::mem::size_of::<OrbitKvPageLease>()];
const _: [(); 24] = [(); std::mem::size_of::<OrbitKvBackendArenaRegistration>()];
const _: [(); 16] = [(); std::mem::size_of::<OrbitKvManagerConfig>()];
const _: [(); 48] = [(); std::mem::size_of::<OrbitKvArenaIdentity>()];
const _: [(); 96] = [(); std::mem::size_of::<OrbitKvArenaStats>()];
const _: [(); 32] = [(); std::mem::size_of::<OrbitKvPrepareBatchItem>()];
const _: [(); 80] = [(); std::mem::size_of::<OrbitKvPreparedBatchItem>()];
const _: [(); 24] = [(); std::mem::size_of::<OrbitKvClassLowering>()];
const _: [(); 16] = [(); std::mem::size_of::<OrbitKvWriteIntent>()];
const _: [(); 64] = [(); std::mem::size_of::<OrbitKvBackendBindReceipt>()];
const _: [(); 32] = [(); std::mem::size_of::<OrbitKvSubmitBatchItem>()];
const _: [(); 32] = [(); std::mem::size_of::<OrbitKvSubmittedBatchItem>()];
const _: [(); 32] = [(); std::mem::size_of::<OrbitKvBatchCompletionReceipt>()];
const _: [(); 16] = [(); std::mem::size_of::<OrbitKvCompleteBatchItem>()];
const _: [(); 24] = [(); std::mem::size_of::<OrbitKvBackendUnobservedReceipt>()];
const _: [(); 120] = [(); std::mem::size_of::<OrbitKvReclamationCertificate>()];
const _: [(); 64] = [(); std::mem::size_of::<OrbitKvReclamationReceipt>()];
const _: [(); 64] = [(); std::mem::size_of::<OrbitKvCompletedBatchItem>()];
const _: [(); 24] = [(); std::mem::size_of::<OrbitKvReleaseBatchItem>()];
const _: [(); 32] = [(); std::mem::size_of::<OrbitKvReleasedBatchItem>()];
const _: [(); 88] = [(); std::mem::size_of::<OrbitKvManagerStats>()];

fn invalid<T>(message: &str) -> Result<T, (i32, String)> {
    Err((ORBITKV_STATUS_INVALID_ARGUMENT, message.to_owned()))
}

fn manager_error(error: impl std::fmt::Display) -> (i32, String) {
    (ORBITKV_STATUS_MANAGER_ERROR, error.to_string())
}

fn arena_stats_match_identity(stats: &ArenaStats, identity: &OrbitKvArenaIdentity) -> bool {
    let phase_total = [
        stats.free_pages,
        stats.reserved_pages,
        stats.writing_pages,
        stats.active_pages,
        stats.retiring_pages,
        stats.quarantined_pages,
        stats.exhausted_pages,
    ]
    .into_iter()
    .try_fold(0_u64, u64::checked_add);
    stats.engine_epoch == identity.engine_epoch
        && stats.pool_epoch == identity.pool_epoch
        && stats.class_id == identity.class_id
        && stats.backend_domain == identity.backend_domain
        && stats.pool_id == identity.pool_id
        && stats.page_count == identity.page_count
        && stats.first_page_id == identity.first_page_id
        && phase_total == Some(u64::from(stats.page_count))
}

unsafe fn required_ref<'a, T>(value: *const T, label: &str) -> Result<&'a T, (i32, String)> {
    if value.is_null() {
        return invalid(&format!("{label} pointer is required"));
    }
    Ok(unsafe { &*value })
}

unsafe fn manager_ref<'a>(
    manager: *mut OrbitKvManagerHandle,
) -> Result<&'a OrbitKvManagerHandle, (i32, String)> {
    if manager.is_null() {
        return invalid("manager pointer is required");
    }
    Ok(unsafe { &*manager })
}

fn lock_manager(
    handle: &OrbitKvManagerHandle,
) -> Result<std::sync::MutexGuard<'_, CanonicalKvManager>, (i32, String)> {
    handle.manager.lock().map_err(|_| {
        (
            ORBITKV_STATUS_MANAGER_ERROR,
            "manager lock is poisoned".to_owned(),
        )
    })
}

fn maximum_batch_items(handle: &OrbitKvManagerHandle) -> u32 {
    handle.maximum_requests.min(handle.maximum_operations)
}

fn checked_output_bound(
    item_count: u32,
    maximum_per_item: u32,
    label: &str,
) -> Result<u32, (i32, String)> {
    item_count.checked_mul(maximum_per_item).ok_or_else(|| {
        (
            ORBITKV_STATUS_INVALID_ARGUMENT,
            format!("{label} batch bound exceeds uint32_t"),
        )
    })
}

fn hot_page_output_bound(
    handle: &OrbitKvManagerHandle,
    item_count: u32,
    maximum_per_item: u32,
    label: &str,
) -> Result<u32, (i32, String)> {
    Ok(checked_output_bound(item_count, maximum_per_item, label)?.min(handle.total_page_capacity))
}

fn validate_count_limit(count: u32, maximum: u32, label: &str) -> Result<(), (i32, String)> {
    if count > maximum {
        return invalid(&format!(
            "{label} count {count} exceeds configured maximum {maximum}"
        ));
    }
    Ok(())
}

unsafe fn clear_output_count(out_count: *mut u32, label: &str) -> Result<(), (i32, String)> {
    if out_count.is_null() {
        return invalid(&format!("{label} count output is required"));
    }
    unsafe {
        out_count.write(0);
    }
    Ok(())
}

unsafe fn input_slice<'a, T>(
    input: *const T,
    count: u32,
    label: &str,
) -> Result<&'a [T], (i32, String)> {
    if count == 0 {
        return Ok(&[]);
    }
    if input.is_null() {
        return invalid(&format!("{label} buffer is required"));
    }
    let count = usize::try_from(count).map_err(|_| {
        (
            ORBITKV_STATUS_INVALID_ARGUMENT,
            format!("{label} count is too large"),
        )
    })?;
    Ok(unsafe { slice::from_raw_parts(input, count) })
}

fn clear_fixed_output<T: Default>(output: *mut T, label: &str) -> Result<(), (i32, String)> {
    if output.is_null() {
        return invalid(&format!("{label} pointer is required"));
    }
    unsafe {
        output.write(T::default());
    }
    Ok(())
}

unsafe fn preflight_mutating_output<T>(
    output: *mut T,
    capacity: u32,
    out_count: *mut u32,
    required_capacity: u32,
    label: &str,
) -> Result<bool, (i32, String)> {
    unsafe { clear_output_count(out_count, label)? };
    if capacity < required_capacity {
        unsafe {
            out_count.write(required_capacity);
        }
        return Ok(true);
    }
    if required_capacity != 0 && output.is_null() {
        return invalid(&format!("{label} output buffer is required"));
    }
    Ok(false)
}

unsafe fn write_reclamation_certificates(
    certificates: &[ReclamationCertificate],
    output: *mut OrbitKvReclamationCertificate,
) {
    for (index, certificate) in certificates.iter().cloned().enumerate() {
        unsafe {
            output.add(index).write(certificate.into());
        }
    }
}

fn u32_len(length: usize, label: &str) -> Result<u32, (i32, String)> {
    u32::try_from(length).map_err(|_| {
        (
            ORBITKV_STATUS_MANAGER_ERROR,
            format!("{label} does not fit the canonical ABI"),
        )
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::borrow_as_ptr, clippy::too_many_lines)]

    use super::*;

    const PLAN: &[u8] = br#"{
      "page_tokens": 16,
      "classes": [{
        "name": "swa",
        "layers": [0],
        "retention": "sliding",
        "bytes_per_token_per_layer": 128,
        "window_tokens": 18
      }]
    }"#;

    fn config() -> OrbitKvManagerConfig {
        OrbitKvManagerConfig {
            maximum_requests: 4,
            maximum_operations: 4,
            maximum_reclamations: 8,
            maximum_step_tokens: 64,
        }
    }

    fn backend() -> OrbitKvBackendArenaRegistration {
        OrbitKvBackendArenaRegistration {
            pool_id: 7,
            class_id: 0,
            backend_domain: 3,
            page_count: 8,
            reserved: 0,
            backend_base_index: 100,
        }
    }

    fn create(error: &mut [c_char]) -> *mut OrbitKvManagerHandle {
        let mut handle = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                orbitkv_manager_create(
                    PLAN.as_ptr(),
                    PLAN.len(),
                    &config(),
                    &backend(),
                    1,
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert!(!handle.is_null());
        handle
    }

    fn stats(handle: *mut OrbitKvManagerHandle, error: &mut [c_char]) -> OrbitKvManagerStats {
        let mut value = OrbitKvManagerStats::default();
        assert_eq!(
            unsafe { orbitkv_manager_stats(handle, &mut value, error.as_mut_ptr(), error.len()) },
            ORBITKV_STATUS_OK
        );
        value
    }

    fn identity(handle: *mut OrbitKvManagerHandle, error: &mut [c_char]) -> OrbitKvArenaIdentity {
        let mut value = OrbitKvArenaIdentity::default();
        let mut count = 0;
        assert_eq!(
            unsafe {
                orbitkv_manager_arena_identities(
                    handle,
                    &mut value,
                    1,
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(count, 1);
        value
    }

    fn acquire_two(
        handle: *mut OrbitKvManagerHandle,
        error: &mut [c_char],
    ) -> [OrbitKvRequestLease; 2] {
        let mut short = [OrbitKvRequestLease::default(); 1];
        let mut count = 99;
        assert_eq!(
            unsafe {
                orbitkv_manager_request_acquire_batch(
                    handle,
                    2,
                    short.as_mut_ptr(),
                    1,
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!(count, 2);
        assert_eq!(stats(handle, error).active_requests, 0);

        let mut requests = [OrbitKvRequestLease::default(); 2];
        assert_eq!(
            unsafe {
                orbitkv_manager_request_acquire_batch(
                    handle,
                    2,
                    requests.as_mut_ptr(),
                    2,
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(count, 2);
        assert_eq!(requests[0].engine_epoch, requests[1].engine_epoch);
        assert_ne!(requests[0].slot, requests[1].slot);
        requests
    }

    fn prepare_two(
        handle: *mut OrbitKvManagerHandle,
        requests: [OrbitKvRequestLease; 2],
        target: u64,
        error: &mut [c_char],
    ) -> (
        [OrbitKvPreparedBatchItem; 2],
        [OrbitKvClassLowering; 2],
        [OrbitKvWriteIntent; 8],
        u32,
    ) {
        let items = requests.map(|request| OrbitKvPrepareBatchItem {
            request,
            target_boundary: target,
            reserved: 0,
        });
        let mut prepared = [OrbitKvPreparedBatchItem::default(); 2];
        let mut classes = [OrbitKvClassLowering::default(); 2];
        let mut writes = [OrbitKvWriteIntent::default(); 8];
        let mut prepared_count = 0;
        let mut class_count = 0;
        let mut write_count = 0;
        assert_eq!(
            unsafe {
                orbitkv_manager_prepare_batch(
                    handle,
                    items.as_ptr(),
                    2,
                    prepared.as_mut_ptr(),
                    2,
                    &mut prepared_count,
                    classes.as_mut_ptr(),
                    2,
                    &mut class_count,
                    writes.as_mut_ptr(),
                    8,
                    &mut write_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(prepared_count, 2);
        assert_eq!(class_count, 2);
        (prepared, classes, writes, write_count)
    }

    fn bind_receipts(
        prepared: &[OrbitKvPreparedBatchItem; 2],
        classes: &[OrbitKvClassLowering; 2],
        writes: &[OrbitKvWriteIntent; 8],
        identity: OrbitKvArenaIdentity,
    ) -> (Vec<OrbitKvSubmitBatchItem>, Vec<OrbitKvBackendBindReceipt>) {
        let mut items = Vec::new();
        let mut receipts = Vec::new();
        for value in prepared {
            let receipt_offset = u32::try_from(receipts.len()).unwrap();
            let class_begin = usize::try_from(value.class_offset).unwrap();
            let class_end = class_begin + usize::try_from(value.class_count).unwrap();
            for class in &classes[class_begin..class_end] {
                assert_eq!(class.class_id, identity.class_id);
                let write_begin = usize::try_from(class.write_offset).unwrap();
                let write_end = write_begin + usize::try_from(class.write_count).unwrap();
                for intent in &writes[write_begin..write_end] {
                    receipts.push(OrbitKvBackendBindReceipt {
                        step: value.step,
                        page: OrbitKvPageLease {
                            engine_epoch: value.step.engine_epoch,
                            pool_epoch: identity.pool_epoch,
                            generation: intent.page_generation,
                            page_id: intent.page_id,
                            pool_id: identity.pool_id,
                        },
                        backend_domain: identity.backend_domain,
                        mapped: 1,
                        writable: 1,
                        reserved: 0,
                        backend_index: identity.backend_base_index
                            + u64::from(intent.page_id - identity.first_page_id),
                    });
                }
            }
            items.push(OrbitKvSubmitBatchItem {
                step: value.step,
                receipt_offset,
                receipt_count: u32::try_from(receipts.len()).unwrap() - receipt_offset,
                reserved: 0,
            });
        }
        (items, receipts)
    }

    fn ack(
        handle: *mut OrbitKvManagerHandle,
        certificates: &[OrbitKvReclamationCertificate],
        error: &mut [c_char],
    ) {
        let receipts = certificates
            .iter()
            .map(|certificate| {
                assert_eq!(certificate.reserved32, 0);
                OrbitKvReclamationReceipt {
                    reclamation: certificate.reclamation,
                    page: certificate.page,
                    backend_domain: certificate.backend_domain,
                    acknowledged: 1,
                    reserved8: 0,
                    reserved32: 0,
                    backend_index: certificate.backend_index,
                }
            })
            .collect::<Vec<_>>();
        assert!(!receipts.is_empty());
        assert_eq!(
            unsafe {
                orbitkv_manager_acknowledge_reclamations(
                    handle,
                    receipts.as_ptr(),
                    u32::try_from(receipts.len()).unwrap(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
    }

    #[test]
    fn abi5_layout_is_closed_and_padding_is_explicit() {
        assert_eq!(orbitkv_abi_version(), 5);
        assert_eq!(std::mem::align_of::<OrbitKvManagerConfig>(), 4);
        assert_eq!(std::mem::size_of::<OrbitKvPrepareBatchItem>(), 32);
        assert_eq!(std::mem::align_of::<OrbitKvPrepareBatchItem>(), 8);
        assert_eq!(std::mem::offset_of!(OrbitKvPrepareBatchItem, reserved), 24);
        assert_eq!(std::mem::size_of::<OrbitKvPreparedBatchItem>(), 80);
        assert_eq!(std::mem::align_of::<OrbitKvPreparedBatchItem>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvPreparedBatchItem, class_offset),
            64
        );
        assert_eq!(
            std::mem::offset_of!(OrbitKvPreparedBatchItem, write_offset),
            72
        );
        assert_eq!(std::mem::size_of::<OrbitKvClassLowering>(), 24);
        assert_eq!(std::mem::align_of::<OrbitKvClassLowering>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvClassLowering, previous_tail_generation),
            16
        );
        assert_eq!(std::mem::size_of::<OrbitKvWriteIntent>(), 16);
        assert_eq!(std::mem::align_of::<OrbitKvWriteIntent>(), 8);
        assert_eq!(std::mem::offset_of!(OrbitKvWriteIntent, reserved), 12);
        assert_eq!(std::mem::align_of::<OrbitKvBackendBindReceipt>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvBackendBindReceipt, reserved),
            52
        );
        assert_eq!(std::mem::align_of::<OrbitKvSubmitBatchItem>(), 8);
        assert_eq!(std::mem::offset_of!(OrbitKvSubmitBatchItem, reserved), 24);
        assert_eq!(std::mem::size_of::<OrbitKvSubmittedBatchItem>(), 32);
        assert_eq!(std::mem::align_of::<OrbitKvSubmittedBatchItem>(), 8);
        assert_eq!(std::mem::align_of::<OrbitKvBatchCompletionReceipt>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvBatchCompletionReceipt, reserved),
            28
        );
        assert_eq!(std::mem::align_of::<OrbitKvCompleteBatchItem>(), 8);
        assert_eq!(std::mem::size_of::<OrbitKvCompletedBatchItem>(), 64);
        assert_eq!(std::mem::align_of::<OrbitKvCompletedBatchItem>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvCompletedBatchItem, reserved),
            60
        );
        assert_eq!(std::mem::size_of::<OrbitKvReclamationCertificate>(), 120);
        assert_eq!(std::mem::align_of::<OrbitKvReclamationCertificate>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvReclamationCertificate, reserved32),
            68
        );
        assert_eq!(std::mem::align_of::<OrbitKvBackendUnobservedReceipt>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvBackendUnobservedReceipt, reserved),
            20
        );
        assert_eq!(std::mem::align_of::<OrbitKvReclamationReceipt>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvReclamationReceipt, reserved8),
            51
        );
        assert_eq!(
            std::mem::offset_of!(OrbitKvReclamationReceipt, reserved32),
            52
        );
        assert_eq!(std::mem::align_of::<OrbitKvReleaseBatchItem>(), 8);
        assert_eq!(std::mem::offset_of!(OrbitKvReleaseBatchItem, reserved), 16);
        assert_eq!(std::mem::align_of::<OrbitKvReleasedBatchItem>(), 8);
        assert_eq!(std::mem::offset_of!(OrbitKvReleasedBatchItem, reserved), 24);
    }

    #[test]
    fn configured_count_envelopes_precede_pointer_and_buffer_interpretation() {
        let mut error = vec![0; 2048];
        let handle = create(&mut error);

        let mut request_count = 99;
        assert_eq!(
            unsafe {
                orbitkv_manager_request_acquire_batch(
                    handle,
                    5,
                    std::ptr::null_mut(),
                    0,
                    &mut request_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(request_count, 0);

        let mut prepared_count = 99;
        let mut class_count = 99;
        let mut write_count = 99;
        assert_eq!(
            unsafe {
                orbitkv_manager_prepare_batch(
                    handle,
                    std::ptr::null(),
                    5,
                    std::ptr::null_mut(),
                    0,
                    &mut prepared_count,
                    std::ptr::null_mut(),
                    0,
                    &mut class_count,
                    std::ptr::null_mut(),
                    0,
                    &mut write_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_INVALID_ARGUMENT
        );
        assert_eq!((prepared_count, class_count, write_count), (0, 0, 0));

        assert_eq!(
            unsafe {
                orbitkv_manager_acknowledge_reclamations(
                    handle,
                    std::ptr::null(),
                    9,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_INVALID_ARGUMENT
        );
        let snapshot = stats(handle, &mut error);
        assert_eq!(snapshot.active_requests, 0);
        assert_eq!(snapshot.prepared_steps, 0);
        assert_eq!(snapshot.pending_reclamations, 0);
        assert_eq!(
            unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
            ORBITKV_STATUS_OK
        );
    }

    #[test]
    fn batch_ffi_lifecycle_is_atomic_and_uses_flat_canonical_spans() {
        let mut error = vec![0; 2048];
        let handle = create(&mut error);
        let requests = acquire_two(handle, &mut error);
        let (prepared, classes, writes, write_count) =
            prepare_two(handle, requests, 35, &mut error);
        assert_eq!(write_count, 6);
        assert_eq!(prepared[0].class_offset, 0);
        assert_eq!(prepared[1].class_offset, 1);
        assert_eq!(prepared[0].write_offset, 0);
        assert_eq!(prepared[1].write_offset, 3);
        assert_eq!(classes[0].write_offset, 0);
        assert_eq!(classes[1].write_offset, 3);

        let (submit_items, receipts) =
            bind_receipts(&prepared, &classes, &writes, identity(handle, &mut error));
        let mut submitted = [OrbitKvSubmittedBatchItem::default(); 2];
        let mut submitted_count = 0;
        assert_eq!(
            unsafe {
                orbitkv_manager_submit_batch(
                    handle,
                    submit_items.as_ptr(),
                    2,
                    receipts.as_ptr(),
                    u32::try_from(receipts.len()).unwrap(),
                    submitted.as_mut_ptr(),
                    1,
                    &mut submitted_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!(submitted_count, 2);
        let before_submit = stats(handle, &mut error);
        assert_eq!(before_submit.prepared_steps, 2);
        assert_eq!(before_submit.submitted_steps, 0);

        assert_eq!(
            unsafe {
                orbitkv_manager_submit_batch(
                    handle,
                    submit_items.as_ptr(),
                    2,
                    receipts.as_ptr(),
                    u32::try_from(receipts.len()).unwrap(),
                    submitted.as_mut_ptr(),
                    2,
                    &mut submitted_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(submitted_count, 2);

        let complete_items = submitted.map(|value| OrbitKvCompleteBatchItem {
            submission: value.submission,
        });
        let completion = OrbitKvBatchCompletionReceipt {
            engine_epoch: requests[0].engine_epoch,
            completion_domain: 9,
            completion_value: 1,
            confirmed: 1,
            reserved: 0,
        };
        let mut completed = [OrbitKvCompletedBatchItem::default(); 2];
        let mut retirements = [OrbitKvReclamationCertificate::default(); 8];
        let mut completed_count = 0;
        let mut retirement_count = 0;
        assert_eq!(
            unsafe {
                orbitkv_manager_complete_batch(
                    handle,
                    completion,
                    complete_items.as_ptr(),
                    2,
                    completed.as_mut_ptr(),
                    1,
                    &mut completed_count,
                    retirements.as_mut_ptr(),
                    8,
                    &mut retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!((completed_count, retirement_count), (2, 0));
        assert_eq!(stats(handle, &mut error).submitted_steps, 2);

        assert_eq!(
            unsafe {
                orbitkv_manager_complete_batch(
                    handle,
                    completion,
                    complete_items.as_ptr(),
                    2,
                    completed.as_mut_ptr(),
                    2,
                    &mut completed_count,
                    retirements.as_mut_ptr(),
                    7,
                    &mut retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!(completed_count, 0);
        assert_eq!(retirement_count, 8);
        assert_eq!(stats(handle, &mut error).submitted_steps, 2);

        assert_eq!(
            unsafe {
                orbitkv_manager_complete_batch(
                    handle,
                    completion,
                    complete_items.as_ptr(),
                    2,
                    completed.as_mut_ptr(),
                    2,
                    &mut completed_count,
                    retirements.as_mut_ptr(),
                    8,
                    &mut retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!((completed_count, retirement_count), (2, 2));
        assert_eq!(completed[0].resident_count, 2);
        assert_eq!(completed[1].resident_count, 2);
        assert_eq!(completed[0].published_boundary, 35);
        assert_eq!(completed[1].published_boundary, 35);
        assert_eq!(completed[0].retirement_offset, 0);
        assert_eq!(completed[1].retirement_offset, 1);
        ack(handle, &retirements[..2], &mut error);

        let release_items = requests.map(|request| OrbitKvReleaseBatchItem {
            request,
            reserved: 0,
        });
        let mut released = [OrbitKvReleasedBatchItem::default(); 2];
        let mut release_retirements = [OrbitKvReclamationCertificate::default(); 8];
        let mut released_count = 0;
        let mut release_retirement_count = 0;
        assert_eq!(
            unsafe {
                orbitkv_manager_release_batch(
                    handle,
                    release_items.as_ptr(),
                    2,
                    released.as_mut_ptr(),
                    1,
                    &mut released_count,
                    release_retirements.as_mut_ptr(),
                    8,
                    &mut release_retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!((released_count, release_retirement_count), (2, 0));
        assert_eq!(stats(handle, &mut error).active_requests, 2);

        assert_eq!(
            unsafe {
                orbitkv_manager_release_batch(
                    handle,
                    release_items.as_ptr(),
                    2,
                    released.as_mut_ptr(),
                    2,
                    &mut released_count,
                    release_retirements.as_mut_ptr(),
                    7,
                    &mut release_retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!((released_count, release_retirement_count), (0, 8));
        assert_eq!(stats(handle, &mut error).active_requests, 2);

        assert_eq!(
            unsafe {
                orbitkv_manager_release_batch(
                    handle,
                    release_items.as_ptr(),
                    2,
                    released.as_mut_ptr(),
                    2,
                    &mut released_count,
                    release_retirements.as_mut_ptr(),
                    8,
                    &mut release_retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!((released_count, release_retirement_count), (2, 4));
        ack(handle, &release_retirements[..4], &mut error);
        assert_eq!(
            unsafe {
                orbitkv_manager_recycle_requests(
                    handle,
                    requests.as_ptr(),
                    2,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        let final_stats = stats(handle, &mut error);
        assert_eq!(final_stats.active_requests, 0);
        assert_eq!(final_stats.free_pages, 8);
        assert_eq!(final_stats.pending_reclamations, 0);
        assert_eq!(
            unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
            ORBITKV_STATUS_OK
        );
    }

    #[test]
    fn short_prepare_buffer_is_zero_mutation() {
        let mut error = vec![0; 2048];
        let handle = create(&mut error);
        let requests = acquire_two(handle, &mut error);
        let items = requests.map(|request| OrbitKvPrepareBatchItem {
            request,
            target_boundary: 16,
            reserved: 0,
        });
        let mut prepared = [OrbitKvPreparedBatchItem::default(); 2];
        let mut classes = [OrbitKvClassLowering::default(); 2];
        let mut writes = [OrbitKvWriteIntent::default(); 8];
        let mut prepared_count = 99;
        let mut class_count = 99;
        let mut write_count = 99;
        assert_eq!(
            unsafe {
                orbitkv_manager_prepare_batch(
                    handle,
                    items.as_ptr(),
                    2,
                    prepared.as_mut_ptr(),
                    2,
                    &mut prepared_count,
                    classes.as_mut_ptr(),
                    1,
                    &mut class_count,
                    writes.as_mut_ptr(),
                    8,
                    &mut write_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!((prepared_count, class_count, write_count), (0, 2, 0));
        assert_eq!(stats(handle, &mut error).prepared_steps, 0);
        assert_eq!(
            unsafe {
                orbitkv_manager_prepare_batch(
                    handle,
                    items.as_ptr(),
                    2,
                    prepared.as_mut_ptr(),
                    2,
                    &mut prepared_count,
                    classes.as_mut_ptr(),
                    2,
                    &mut class_count,
                    writes.as_mut_ptr(),
                    7,
                    &mut write_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!((prepared_count, class_count, write_count), (0, 0, 8));
        let snapshot = stats(handle, &mut error);
        assert_eq!(snapshot.prepared_steps, 0);
        assert_eq!(snapshot.reserved_pages, 0);
        assert_eq!(snapshot.free_pages, 8);

        let release_items = requests.map(|request| OrbitKvReleaseBatchItem {
            request,
            reserved: 0,
        });
        let mut released = [OrbitKvReleasedBatchItem::default(); 2];
        let mut retirements = [OrbitKvReclamationCertificate::default(); 8];
        let mut released_count = 0;
        let mut retirement_count = 0;
        assert_eq!(
            unsafe {
                orbitkv_manager_release_batch(
                    handle,
                    release_items.as_ptr(),
                    2,
                    released.as_mut_ptr(),
                    2,
                    &mut released_count,
                    retirements.as_mut_ptr(),
                    8,
                    &mut retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(retirement_count, 0);
        assert_eq!(
            unsafe {
                orbitkv_manager_recycle_requests(
                    handle,
                    requests.as_ptr(),
                    2,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(
            unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
            ORBITKV_STATUS_OK
        );
    }

    #[test]
    fn semantic_submit_mismatch_quarantines_the_whole_batch() {
        let mut error = vec![0; 2048];
        let handle = create(&mut error);
        let requests = acquire_two(handle, &mut error);
        let (prepared, classes, writes, count) = prepare_two(handle, requests, 16, &mut error);
        assert_eq!(count, 2);
        let (items, mut receipts) =
            bind_receipts(&prepared, &classes, &writes, identity(handle, &mut error));
        receipts[1].backend_index += 1;
        let mut submitted = [OrbitKvSubmittedBatchItem::default(); 2];
        let mut submitted_count = 99;
        assert_eq!(
            unsafe {
                orbitkv_manager_submit_batch(
                    handle,
                    items.as_ptr(),
                    2,
                    receipts.as_ptr(),
                    2,
                    submitted.as_mut_ptr(),
                    2,
                    &mut submitted_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_MANAGER_ERROR
        );
        assert_eq!(submitted_count, 0);
        let snapshot = stats(handle, &mut error);
        assert_eq!(snapshot.prepared_steps, 0);
        assert_eq!(snapshot.quarantined_pages, 2);
        let aborts = prepared.map(|value| OrbitKvBackendUnobservedReceipt {
            step: value.step,
            backend_unobserved: 1,
            reserved: 0,
        });
        assert_eq!(
            unsafe {
                orbitkv_manager_abort_steps(
                    handle,
                    aborts.as_ptr(),
                    2,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_MANAGER_ERROR
        );
        assert_eq!(
            unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
            ORBITKV_STATUS_MANAGER_ERROR
        );
        unsafe {
            drop(Box::from_raw(handle));
        }
    }

    #[test]
    fn structural_and_middle_item_failures_are_atomic_and_retryable() {
        let mut error = vec![0; 2048];
        let handle = create(&mut error);
        let requests = acquire_two(handle, &mut error);
        let (prepared, classes, writes, count) = prepare_two(handle, requests, 16, &mut error);
        assert_eq!(count, 2);
        let (items, receipts) =
            bind_receipts(&prepared, &classes, &writes, identity(handle, &mut error));

        let mut malformed_items = items.clone();
        malformed_items[1].receipt_offset += 1;
        let mut submitted = [OrbitKvSubmittedBatchItem::default(); 2];
        let mut submitted_count = 99;
        assert_eq!(
            unsafe {
                orbitkv_manager_submit_batch(
                    handle,
                    malformed_items.as_ptr(),
                    2,
                    receipts.as_ptr(),
                    2,
                    submitted.as_mut_ptr(),
                    2,
                    &mut submitted_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_MANAGER_ERROR
        );
        assert_eq!(submitted_count, 0);
        let after_bad_range = stats(handle, &mut error);
        assert_eq!(after_bad_range.prepared_steps, 2);
        assert_eq!(after_bad_range.submitted_steps, 0);
        assert_eq!(after_bad_range.quarantined_pages, 0);

        assert_eq!(
            unsafe {
                orbitkv_manager_submit_batch(
                    handle,
                    items.as_ptr(),
                    2,
                    receipts.as_ptr(),
                    2,
                    submitted.as_mut_ptr(),
                    2,
                    &mut submitted_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );

        let completion = OrbitKvBatchCompletionReceipt {
            engine_epoch: requests[0].engine_epoch,
            completion_domain: 9,
            completion_value: 1,
            confirmed: 1,
            reserved: 0,
        };
        let mut complete_items = submitted.map(|value| OrbitKvCompleteBatchItem {
            submission: value.submission,
        });
        complete_items[1].submission.generation += 1;
        let mut completed = [OrbitKvCompletedBatchItem::default(); 2];
        let mut completion_retirements = [OrbitKvReclamationCertificate::default(); 8];
        let mut completed_count = 99;
        let mut completion_retirement_count = 99;
        assert_eq!(
            unsafe {
                orbitkv_manager_complete_batch(
                    handle,
                    completion,
                    complete_items.as_ptr(),
                    2,
                    completed.as_mut_ptr(),
                    2,
                    &mut completed_count,
                    completion_retirements.as_mut_ptr(),
                    8,
                    &mut completion_retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_MANAGER_ERROR
        );
        assert_eq!((completed_count, completion_retirement_count), (0, 0));
        let after_stale_completion = stats(handle, &mut error);
        assert_eq!(after_stale_completion.submitted_steps, 2);
        assert_eq!(after_stale_completion.writing_pages, 2);

        complete_items[1].submission.generation -= 1;
        assert_eq!(
            unsafe {
                orbitkv_manager_complete_batch(
                    handle,
                    completion,
                    complete_items.as_ptr(),
                    2,
                    completed.as_mut_ptr(),
                    2,
                    &mut completed_count,
                    completion_retirements.as_mut_ptr(),
                    8,
                    &mut completion_retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!((completed_count, completion_retirement_count), (2, 0));

        let mut release_items = requests.map(|request| OrbitKvReleaseBatchItem {
            request,
            reserved: 0,
        });
        release_items[1].request.generation += 1;
        let mut released = [OrbitKvReleasedBatchItem::default(); 2];
        let mut retirements = [OrbitKvReclamationCertificate::default(); 8];
        let mut released_count = 99;
        let mut retirement_count = 99;
        assert_eq!(
            unsafe {
                orbitkv_manager_release_batch(
                    handle,
                    release_items.as_ptr(),
                    2,
                    released.as_mut_ptr(),
                    2,
                    &mut released_count,
                    retirements.as_mut_ptr(),
                    8,
                    &mut retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_MANAGER_ERROR
        );
        assert_eq!((released_count, retirement_count), (0, 0));
        let after_stale_release = stats(handle, &mut error);
        assert_eq!(after_stale_release.active_requests, 2);
        assert_eq!(after_stale_release.active_pages, 2);

        release_items[1].request.generation -= 1;
        assert_eq!(
            unsafe {
                orbitkv_manager_release_batch(
                    handle,
                    release_items.as_ptr(),
                    2,
                    released.as_mut_ptr(),
                    2,
                    &mut released_count,
                    retirements.as_mut_ptr(),
                    8,
                    &mut retirement_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(retirement_count, 2);

        let mut reclamation_receipts = retirements[..2]
            .iter()
            .map(|certificate| OrbitKvReclamationReceipt {
                reclamation: certificate.reclamation,
                page: certificate.page,
                backend_domain: certificate.backend_domain,
                acknowledged: 1,
                reserved8: 0,
                reserved32: 0,
                backend_index: certificate.backend_index,
            })
            .collect::<Vec<_>>();
        reclamation_receipts[1].backend_index += 1;
        assert_eq!(
            unsafe {
                orbitkv_manager_acknowledge_reclamations(
                    handle,
                    reclamation_receipts.as_ptr(),
                    2,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_MANAGER_ERROR
        );
        assert_eq!(stats(handle, &mut error).pending_reclamations, 2);
        reclamation_receipts[1].backend_index -= 1;
        assert_eq!(
            unsafe {
                orbitkv_manager_acknowledge_reclamations(
                    handle,
                    reclamation_receipts.as_ptr(),
                    2,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );

        let mut stale_requests = requests;
        stale_requests[1].generation += 1;
        assert_eq!(
            unsafe {
                orbitkv_manager_recycle_requests(
                    handle,
                    stale_requests.as_ptr(),
                    2,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_MANAGER_ERROR
        );
        assert_eq!(stats(handle, &mut error).active_requests, 2);
        assert_eq!(
            unsafe {
                orbitkv_manager_recycle_requests(
                    handle,
                    requests.as_ptr(),
                    2,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(
            unsafe { orbitkv_manager_destroy(handle, error.as_mut_ptr(), error.len()) },
            ORBITKV_STATUS_OK
        );
    }
}
