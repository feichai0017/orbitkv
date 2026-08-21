use super::{
    BackendArenaRegistration, CanonicalKvManager, KvPlanInput, MAX_PLAN_JSON_BYTES, Mutex,
    ORBITKV_ABI_VERSION, ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_INVALID_ARGUMENT,
    ORBITKV_STATUS_MANAGER_ERROR, ORBITKV_STATUS_OK, OrbitKvArenaIdentity, OrbitKvArenaStats,
    OrbitKvBackendArenaRegistration, OrbitKvManagerConfig, OrbitKvManagerHandle,
    OrbitKvManagerStats, c_char, compile_plan, core_error, ffi_boundary, input_slice, invalid,
    invalid_pair, lock_manager, manager_ref, preflight_output, required_ref, slice, u32_len,
    write_copy_slice,
};

#[unsafe(no_mangle)]
pub extern "C" fn orbitkv_abi_version() -> u32 {
    ORBITKV_ABI_VERSION
}

/// Creates a canonical ABI6 manager.
///
/// # Safety
/// All pointers must name readable/writable storage for their declared sizes.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
        unsafe { out_manager.write(std::ptr::null_mut()) };
        if plan_json.is_null() || plan_json_len == 0 || plan_json_len > MAX_PLAN_JSON_BYTES {
            return invalid("canonical KvPlanInput JSON is missing or exceeds 1 MiB");
        }
        let config = unsafe { required_ref(config, "manager config") }?;
        if backend_count == 0 {
            return invalid("at least one backend arena registration is required");
        }
        let backends = unsafe { input_slice(backends, backend_count, "backend registration") }?;
        if backends.iter().any(|backend| backend.reserved != 0) {
            return invalid("backend registration reserved field must be zero");
        }
        let input = serde_json::from_slice::<KvPlanInput>(unsafe {
            slice::from_raw_parts(plan_json, plan_json_len)
        })
        .map_err(|error| {
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
        let class_count = u32_len(plan.classes.len(), "compiled class count")?;
        if class_count != backend_count {
            return invalid("backend arena count must match compiled attention classes");
        }
        let page_tokens = u32::try_from(plan.page_tokens)
            .map_err(|_| invalid_pair("page token count exceeds uint32_t"))?;
        let pages_per_step = u64::from(config.maximum_step_tokens).div_ceil(plan.page_tokens);
        let maximum_write_intents_per_item = u32::try_from(
            u64::from(class_count)
                .checked_mul(pages_per_step)
                .ok_or_else(|| invalid_pair("prepare output bound overflows"))?,
        )
        .map_err(|_| invalid_pair("prepare output bound exceeds uint32_t"))?;
        // One completion can detach/certify at most the step's newly crossed
        // pages plus a previous partial tail and one retention boundary page
        // per class. This is a delta bound; it never scales with resident KV.
        let maximum_completion_outputs_per_item = u32::try_from(
            u64::from(class_count)
                .checked_mul(
                    pages_per_step
                        .checked_add(2)
                        .ok_or_else(|| invalid_pair("completion output bound overflows"))?,
                )
                .ok_or_else(|| invalid_pair("completion output bound overflows"))?,
        )
        .map_err(|_| invalid_pair("completion output bound exceeds uint32_t"))?;
        let total_page_capacity = backends.iter().try_fold(0_u32, |sum, backend| {
            sum.checked_add(backend.page_count)
                .ok_or_else(|| invalid_pair("total page capacity exceeds uint32_t"))
        })?;
        let core_backends = backends
            .iter()
            .copied()
            .map(BackendArenaRegistration::from)
            .collect::<Vec<_>>();
        let manager =
            CanonicalKvManager::new(&plan, (*config).into(), &core_backends).map_err(core_error)?;
        let core_stats = manager.arena_stats();
        let arena_identities = core_stats
            .iter()
            .map(|stats| {
                let backend = backends
                    .iter()
                    .find(|backend| backend.class_id == stats.class_id)
                    .expect("core class originated from validated registration");
                OrbitKvArenaIdentity {
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
                }
            })
            .collect::<Vec<_>>();
        let handle = Box::new(OrbitKvManagerHandle {
            manager: Mutex::new(manager),
            total_page_capacity,
            maximum_requests: config.maximum_requests,
            maximum_operations: config.maximum_operations,
            maximum_prefixes: config.maximum_prefixes,
            class_count,
            maximum_write_intents_per_item,
            maximum_completion_outputs_per_item,
            arena_identities: arena_identities.into_boxed_slice(),
        });
        unsafe { out_manager.write(Box::into_raw(handle)) };
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Copies immutable arena identities.
///
/// # Safety
/// Output pointers must be writable for their declared capacity.
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
        let handle = unsafe { manager_ref(manager) }?;
        let required = u32_len(handle.arena_identities.len(), "arena identity count")?;
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
        unsafe { write_copy_slice(&handle.arena_identities, identities) };
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Copies one reference-complete arena census per class.
///
/// # Safety
/// Output pointers must be writable for their declared capacity.
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
        let handle = unsafe { manager_ref(manager) }?;
        let required = u32_len(handle.arena_identities.len(), "arena stats count")?;
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
        let values = lock_manager(handle)?.arena_stats();
        assert_eq!(values.len(), handle.arena_identities.len());
        for (index, value) in values.iter().copied().enumerate() {
            unsafe { stats.add(index).write(value.into()) };
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Returns one fixed-width manager snapshot including snapshot/prefix/ref state.
///
/// # Safety
/// `out_stats` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_manager_stats(
    manager: *mut OrbitKvManagerHandle,
    out_stats: *mut OrbitKvManagerStats,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { manager_ref(manager) }?;
        if out_stats.is_null() {
            return invalid("manager stats output pointer is required");
        }
        let stats = lock_manager(handle)?.stats();
        unsafe { out_stats.write(stats.into()) };
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Destroys a quiescent manager. Null is a successful no-op.
///
/// # Safety
/// A non-null handle must be live and exclusively owned by the caller.
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
        let handle = unsafe { manager_ref(manager) }?;
        let stats = lock_manager(handle)?.stats();
        if stats.active_requests != 0
            || stats.active_snapshots != 0
            || stats.active_prefixes != 0
            || stats.evicted_prefixes != 0
            || stats.prepared_steps != 0
            || stats.submitted_steps != 0
            || stats.reserved_pages != 0
            || stats.writing_pages != 0
            || stats.active_pages != 0
            || stats.retiring_pages != 0
            || stats.quarantined_pages != 0
            || stats.pending_reclamations != 0
            || stats.total_request_page_refs != 0
            || stats.total_prefix_page_refs != 0
            || stats.total_reader_pins != 0
        {
            return Err((
                ORBITKV_STATUS_MANAGER_ERROR,
                "manager is not quiescent; handle was not destroyed".to_owned(),
            ));
        }
        unsafe { drop(Box::from_raw(manager)) };
        Ok(ORBITKV_STATUS_OK)
    })
}
