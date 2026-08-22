#![allow(
    clippy::missing_panics_doc,
    clippy::missing_safety_doc,
    clippy::needless_pass_by_value
)]

use std::ffi::c_char;
use std::slice;
use std::sync::{Mutex, MutexGuard};

use orbitkv::{
    StateCheckpointError, StateCheckpointPool, StateCompletionReceipt, StateCopyIntent,
    StateCopyReceipt, StatePoolIdentity, StatePoolStats, StatePublication,
    StateRetirementCertificate, StateRetirementLease, StateSlotLease, StateTransitionLease,
};

use crate::{
    ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_FAIL_STOPPED, ORBITKV_STATUS_INVALID_ARGUMENT,
    ORBITKV_STATUS_MANAGER_ERROR, ORBITKV_STATUS_OK, ORBITKV_STATUS_RETRYABLE_CONFLICT,
    ffi_boundary,
};

pub struct OrbitKvStatePoolHandle {
    pool: Mutex<StateCheckpointPool>,
    slot_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStatePoolConfig {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub byte_count: u64,
    pub pool_id: u32,
    pub slot_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct OrbitKvStateSlotLease {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub generation: u64,
    pub slot_id: u32,
    pub pool_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct OrbitKvStateTransitionLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct OrbitKvStateRetirementLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStatePoolIdentity {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub byte_count: u64,
    pub pool_id: u32,
    pub slot_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStatePoolStats {
    pub identity: OrbitKvStatePoolIdentity,
    pub free_slots: u64,
    pub reserved_slots: u64,
    pub relocating_slots: u64,
    pub live_slots: u64,
    pub retiring_slots: u64,
    pub quarantined_slots: u64,
    pub active_owners: u64,
    pub pending_transitions: u64,
    pub pending_retirements: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStatePrepareItem {
    pub owner_id: u64,
    pub expected: OrbitKvStateSlotLease,
    pub expected_present: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStateCopyIntent {
    pub transition: OrbitKvStateTransitionLease,
    pub owner_id: u64,
    pub source: OrbitKvStateSlotLease,
    pub destination: OrbitKvStateSlotLease,
    pub byte_count: u64,
    pub source_present: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStateCopyReceipt {
    pub transition: OrbitKvStateTransitionLease,
    pub source: OrbitKvStateSlotLease,
    pub destination: OrbitKvStateSlotLease,
    pub byte_count: u64,
    pub source_present: u8,
    pub observed: u8,
    pub written: u8,
    pub reserved8: u8,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStateCompletionReceipt {
    pub engine_epoch: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
    pub confirmed: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStateRetirementCertificate {
    pub retirement: OrbitKvStateRetirementLease,
    pub slot: OrbitKvStateSlotLease,
    pub byte_count: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStatePublication {
    pub owner_id: u64,
    pub slot: OrbitKvStateSlotLease,
    pub retirement: OrbitKvStateRetirementCertificate,
    pub retirement_present: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStateAbortItem {
    pub transition: OrbitKvStateTransitionLease,
    pub backend_unobserved: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStateRetireOwnerItem {
    pub owner_id: u64,
    pub expected: OrbitKvStateSlotLease,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvStateCurrent {
    pub owner_id: u64,
    pub slot: OrbitKvStateSlotLease,
    pub present: u32,
    pub reserved: u32,
}

macro_rules! abi_layout {
    ($ty:ty, $size:expr, $align:expr; $($field:ident = $offset:expr),+ $(,)?) => {
        const _: [(); $size] = [(); std::mem::size_of::<$ty>()];
        const _: [(); $align] = [(); std::mem::align_of::<$ty>()];
        $(const _: [(); $offset] = [(); std::mem::offset_of!($ty, $field)];)+
    };
}

abi_layout!(OrbitKvStatePoolConfig, 32, 8; engine_epoch = 0, pool_epoch = 8, byte_count = 16, pool_id = 24, slot_count = 28);
abi_layout!(OrbitKvStateSlotLease, 32, 8; engine_epoch = 0, pool_epoch = 8, generation = 16, slot_id = 24, pool_id = 28);
abi_layout!(OrbitKvStateTransitionLease, 16, 8; engine_epoch = 0, slot = 8, generation = 12);
abi_layout!(OrbitKvStateRetirementLease, 16, 8; engine_epoch = 0, slot = 8, generation = 12);
abi_layout!(OrbitKvStatePoolIdentity, 32, 8; engine_epoch = 0, pool_epoch = 8, byte_count = 16, pool_id = 24, slot_count = 28);
abi_layout!(OrbitKvStatePoolStats, 104, 8; identity = 0, free_slots = 32, reserved_slots = 40, relocating_slots = 48, live_slots = 56, retiring_slots = 64, quarantined_slots = 72, active_owners = 80, pending_transitions = 88, pending_retirements = 96);
abi_layout!(OrbitKvStatePrepareItem, 48, 8; owner_id = 0, expected = 8, expected_present = 40, reserved = 44);
abi_layout!(OrbitKvStateCopyIntent, 104, 8; transition = 0, owner_id = 16, source = 24, destination = 56, byte_count = 88, source_present = 96, reserved = 100);
abi_layout!(OrbitKvStateCopyReceipt, 96, 8; transition = 0, source = 16, destination = 48, byte_count = 80, source_present = 88, observed = 89, written = 90, reserved8 = 91, reserved32 = 92);
abi_layout!(OrbitKvStateCompletionReceipt, 32, 8; engine_epoch = 0, completion_domain = 8, completion_value = 16, confirmed = 24, reserved = 28);
abi_layout!(OrbitKvStateRetirementCertificate, 72, 8; retirement = 0, slot = 16, byte_count = 48, completion_domain = 56, completion_value = 64);
abi_layout!(OrbitKvStatePublication, 120, 8; owner_id = 0, slot = 8, retirement = 40, retirement_present = 112, reserved = 116);
abi_layout!(OrbitKvStateAbortItem, 24, 8; transition = 0, backend_unobserved = 16, reserved = 20);
abi_layout!(OrbitKvStateRetireOwnerItem, 40, 8; owner_id = 0, expected = 8);
abi_layout!(OrbitKvStateCurrent, 48, 8; owner_id = 0, slot = 8, present = 40, reserved = 44);

impl From<StateSlotLease> for OrbitKvStateSlotLease {
    fn from(value: StateSlotLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            generation: value.generation,
            slot_id: value.slot_id,
            pool_id: value.pool_id,
        }
    }
}

impl From<OrbitKvStateSlotLease> for StateSlotLease {
    fn from(value: OrbitKvStateSlotLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            generation: value.generation,
            slot_id: value.slot_id,
            pool_id: value.pool_id,
        }
    }
}

impl From<StateTransitionLease> for OrbitKvStateTransitionLease {
    fn from(value: StateTransitionLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<OrbitKvStateTransitionLease> for StateTransitionLease {
    fn from(value: OrbitKvStateTransitionLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<StateRetirementLease> for OrbitKvStateRetirementLease {
    fn from(value: StateRetirementLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<OrbitKvStateRetirementLease> for StateRetirementLease {
    fn from(value: OrbitKvStateRetirementLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl From<StatePoolIdentity> for OrbitKvStatePoolIdentity {
    fn from(value: StatePoolIdentity) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            byte_count: value.byte_count,
            pool_id: value.pool_id,
            slot_count: value.slot_count,
        }
    }
}

impl From<StatePoolStats> for OrbitKvStatePoolStats {
    fn from(value: StatePoolStats) -> Self {
        Self {
            identity: value.identity.into(),
            free_slots: value.free_slots,
            reserved_slots: value.reserved_slots,
            relocating_slots: value.relocating_slots,
            live_slots: value.live_slots,
            retiring_slots: value.retiring_slots,
            quarantined_slots: value.quarantined_slots,
            active_owners: value.active_owners,
            pending_transitions: value.pending_transitions,
            pending_retirements: value.pending_retirements,
        }
    }
}

impl From<StateCopyIntent> for OrbitKvStateCopyIntent {
    fn from(value: StateCopyIntent) -> Self {
        Self {
            transition: value.transition.into(),
            owner_id: value.owner_id,
            source: value.source.unwrap_or_default().into(),
            destination: value.destination.into(),
            byte_count: value.byte_count,
            source_present: u32::from(value.source.is_some()),
            reserved: 0,
        }
    }
}

impl From<StateRetirementCertificate> for OrbitKvStateRetirementCertificate {
    fn from(value: StateRetirementCertificate) -> Self {
        Self {
            retirement: value.retirement.into(),
            slot: value.slot.into(),
            byte_count: value.byte_count,
            completion_domain: value.completion_domain,
            completion_value: value.completion_value,
        }
    }
}

impl From<StatePublication> for OrbitKvStatePublication {
    fn from(value: StatePublication) -> Self {
        Self {
            owner_id: value.owner_id,
            slot: value.slot.into(),
            retirement: value.retirement.map_or_else(Default::default, Into::into),
            retirement_present: u32::from(value.retirement.is_some()),
            reserved: 0,
        }
    }
}

fn invalid<T>(message: &str) -> Result<T, (i32, String)> {
    Err((ORBITKV_STATUS_INVALID_ARGUMENT, message.to_owned()))
}

fn state_error(error: StateCheckpointError) -> (i32, String) {
    let status = match error {
        StateCheckpointError::CopyReceiptMismatch
        | StateCheckpointError::CopyObservationUnknown
        | StateCheckpointError::OwnerQuarantined => ORBITKV_STATUS_FAIL_STOPPED,
        StateCheckpointError::StaleLease
        | StateCheckpointError::StaleTransition
        | StateCheckpointError::OwnerBusy
        | StateCheckpointError::AlreadySubmitted
        | StateCheckpointError::NotSubmitted
        | StateCheckpointError::RetirementMismatch => ORBITKV_STATUS_RETRYABLE_CONFLICT,
        StateCheckpointError::InvalidGeometry
        | StateCheckpointError::PoolExhausted
        | StateCheckpointError::CompletionNotConfirmed
        | StateCheckpointError::GenerationExhausted => ORBITKV_STATUS_MANAGER_ERROR,
    };
    (status, error.to_string())
}

unsafe fn handle_ref<'a>(
    handle: *mut OrbitKvStatePoolHandle,
) -> Result<&'a OrbitKvStatePoolHandle, (i32, String)> {
    if handle.is_null() {
        return invalid("state pool handle is required");
    }
    Ok(unsafe { &*handle })
}

fn lock_pool(
    handle: &OrbitKvStatePoolHandle,
) -> Result<MutexGuard<'_, StateCheckpointPool>, (i32, String)> {
    handle.pool.lock().map_err(|_| {
        (
            ORBITKV_STATUS_MANAGER_ERROR,
            "state pool lock is poisoned".to_owned(),
        )
    })
}

unsafe fn input_slice<'a, T>(
    input: *const T,
    count: u32,
    label: &str,
) -> Result<&'a [T], (i32, String)> {
    if count == 0 {
        return invalid(&format!("{label} batch must not be empty"));
    }
    if input.is_null() {
        return invalid(&format!("{label} input is required"));
    }
    Ok(unsafe { slice::from_raw_parts(input, count as usize) })
}

unsafe fn output_slice<'a, T>(
    output: *mut T,
    capacity: u32,
    out_count: *mut u32,
    required: u32,
    label: &str,
) -> Result<Option<&'a mut [T]>, (i32, String)> {
    if out_count.is_null() {
        return invalid(&format!("{label} output count is required"));
    }
    unsafe { out_count.write(required) };
    if capacity < required {
        return Ok(None);
    }
    if output.is_null() {
        return invalid(&format!("{label} output is required"));
    }
    Ok(Some(unsafe {
        slice::from_raw_parts_mut(output, required as usize)
    }))
}

fn completion(
    value: OrbitKvStateCompletionReceipt,
) -> Result<StateCompletionReceipt, (i32, String)> {
    if value.reserved != 0 || value.confirmed > 1 {
        return invalid("state completion reserved/confirmed fields are invalid");
    }
    Ok(StateCompletionReceipt {
        engine_epoch: value.engine_epoch,
        completion_domain: value.completion_domain,
        completion_value: value.completion_value,
        confirmed: value.confirmed == 1,
    })
}

fn retirement(value: OrbitKvStateRetirementCertificate) -> StateRetirementCertificate {
    StateRetirementCertificate {
        retirement: value.retirement.into(),
        slot: value.slot.into(),
        byte_count: value.byte_count,
        completion_domain: value.completion_domain,
        completion_value: value.completion_value,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_create(
    config: *const OrbitKvStatePoolConfig,
    out_handle: *mut *mut OrbitKvStatePoolHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if config.is_null() || out_handle.is_null() {
            return invalid("state pool config and output handle are required");
        }
        unsafe { out_handle.write(std::ptr::null_mut()) };
        let config = unsafe { *config };
        let pool = StateCheckpointPool::new(
            config.engine_epoch,
            config.pool_epoch,
            config.pool_id,
            config.byte_count,
            config.slot_count,
        )
        .map_err(state_error)?;
        unsafe {
            out_handle.write(Box::into_raw(Box::new(OrbitKvStatePoolHandle {
                pool: Mutex::new(pool),
                slot_count: config.slot_count,
            })));
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_identity(
    handle: *mut OrbitKvStatePoolHandle,
    output: *mut OrbitKvStatePoolIdentity,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if output.is_null() {
            return invalid("state pool identity output is required");
        }
        let handle = unsafe { handle_ref(handle) }?;
        unsafe { output.write(lock_pool(handle)?.identity().into()) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_stats(
    handle: *mut OrbitKvStatePoolHandle,
    output: *mut OrbitKvStatePoolStats,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if output.is_null() {
            return invalid("state pool stats output is required");
        }
        let handle = unsafe { handle_ref(handle) }?;
        unsafe { output.write(lock_pool(handle)?.stats().into()) };
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_prepare_batch(
    handle: *mut OrbitKvStatePoolHandle,
    items: *const OrbitKvStatePrepareItem,
    item_count: u32,
    outputs: *mut OrbitKvStateCopyIntent,
    output_capacity: u32,
    out_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { handle_ref(handle) }?;
        if item_count > handle.slot_count {
            return invalid("state prepare batch exceeds slot capacity");
        }
        let items = unsafe { input_slice(items, item_count, "state prepare") }?;
        if items
            .iter()
            .any(|item| item.reserved != 0 || item.expected_present > 1)
        {
            return invalid("state prepare fields are invalid");
        }
        let Some(output) = (unsafe {
            output_slice(
                outputs,
                output_capacity,
                out_count,
                item_count,
                "state prepare",
            )?
        }) else {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        };
        let values = items
            .iter()
            .map(|item| {
                (
                    item.owner_id,
                    (item.expected_present == 1).then(|| item.expected.into()),
                )
            })
            .collect::<Vec<_>>();
        let prepared = lock_pool(handle)?
            .prepare_batch(&values)
            .map_err(state_error)?;
        for (destination, value) in output.iter_mut().zip(prepared) {
            *destination = value.into();
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_submit_batch(
    handle: *mut OrbitKvStatePoolHandle,
    receipts: *const OrbitKvStateCopyReceipt,
    receipt_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { handle_ref(handle) }?;
        if receipt_count > handle.slot_count {
            return invalid("state receipt batch exceeds slot capacity");
        }
        let receipts = unsafe { input_slice(receipts, receipt_count, "state receipt") }?;
        let values = receipts
            .iter()
            .map(|receipt| StateCopyReceipt {
                transition: receipt.transition.into(),
                source: receipt.source.into(),
                destination: receipt.destination.into(),
                byte_count: receipt.byte_count,
                source_present: receipt.source_present,
                observed: receipt.observed,
                written: receipt.written,
                reserved8: receipt.reserved8,
                reserved32: receipt.reserved32,
            })
            .collect::<Vec<_>>();
        lock_pool(handle)?
            .submit_batch(&values)
            .map_err(state_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_complete_batch(
    handle: *mut OrbitKvStatePoolHandle,
    completion_receipt: *const OrbitKvStateCompletionReceipt,
    transitions: *const OrbitKvStateTransitionLease,
    transition_count: u32,
    outputs: *mut OrbitKvStatePublication,
    output_capacity: u32,
    out_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { handle_ref(handle) }?;
        if completion_receipt.is_null() || transition_count > handle.slot_count {
            return invalid("state completion input is invalid");
        }
        let completion_receipt = completion(unsafe { *completion_receipt })?;
        let transitions =
            unsafe { input_slice(transitions, transition_count, "state completion") }?;
        let Some(output) = (unsafe {
            output_slice(
                outputs,
                output_capacity,
                out_count,
                transition_count,
                "state completion",
            )?
        }) else {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        };
        let transitions = transitions
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<_>>();
        let publications = lock_pool(handle)?
            .complete_batch(&transitions, completion_receipt)
            .map_err(state_error)?;
        for (destination, value) in output.iter_mut().zip(publications) {
            *destination = value.into();
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_abort_batch(
    handle: *mut OrbitKvStatePoolHandle,
    items: *const OrbitKvStateAbortItem,
    item_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { handle_ref(handle) }?;
        if item_count > handle.slot_count {
            return invalid("state abort batch exceeds slot capacity");
        }
        let items = unsafe { input_slice(items, item_count, "state abort") }?;
        if items
            .iter()
            .any(|item| item.reserved != 0 || item.backend_unobserved > 1)
        {
            return invalid("state abort fields are invalid");
        }
        let values = items
            .iter()
            .map(|item| (item.transition.into(), item.backend_unobserved == 1))
            .collect::<Vec<_>>();
        lock_pool(handle)?
            .abort_batch(&values)
            .map_err(state_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_retire_owners_batch(
    handle: *mut OrbitKvStatePoolHandle,
    completion_receipt: *const OrbitKvStateCompletionReceipt,
    items: *const OrbitKvStateRetireOwnerItem,
    item_count: u32,
    outputs: *mut OrbitKvStateRetirementCertificate,
    output_capacity: u32,
    out_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { handle_ref(handle) }?;
        if completion_receipt.is_null() || item_count > handle.slot_count {
            return invalid("state retirement input is invalid");
        }
        let completion_receipt = completion(unsafe { *completion_receipt })?;
        let items = unsafe { input_slice(items, item_count, "state retirement") }?;
        let Some(output) = (unsafe {
            output_slice(
                outputs,
                output_capacity,
                out_count,
                item_count,
                "state retirement",
            )?
        }) else {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        };
        let values = items
            .iter()
            .map(|item| (item.owner_id, item.expected.into()))
            .collect::<Vec<_>>();
        let certificates = lock_pool(handle)?
            .retire_owners_batch(&values, completion_receipt)
            .map_err(state_error)?;
        for (destination, value) in output.iter_mut().zip(certificates) {
            *destination = value.into();
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_acknowledge_batch(
    handle: *mut OrbitKvStatePoolHandle,
    certificates: *const OrbitKvStateRetirementCertificate,
    certificate_count: u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { handle_ref(handle) }?;
        if certificate_count > handle.slot_count {
            return invalid("state acknowledgement batch exceeds slot capacity");
        }
        let certificates =
            unsafe { input_slice(certificates, certificate_count, "state acknowledgement") }?;
        let values = certificates
            .iter()
            .copied()
            .map(retirement)
            .collect::<Vec<_>>();
        lock_pool(handle)?
            .acknowledge_batch(&values)
            .map_err(state_error)?;
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_current_batch(
    handle: *mut OrbitKvStatePoolHandle,
    owner_ids: *const u64,
    owner_count: u32,
    outputs: *mut OrbitKvStateCurrent,
    output_capacity: u32,
    out_count: *mut u32,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { handle_ref(handle) }?;
        if owner_count > handle.slot_count {
            return invalid("state current batch exceeds slot capacity");
        }
        let owner_ids = unsafe { input_slice(owner_ids, owner_count, "state current") }?;
        let Some(output) = (unsafe {
            output_slice(
                outputs,
                output_capacity,
                out_count,
                owner_count,
                "state current",
            )?
        }) else {
            return Ok(ORBITKV_STATUS_BUFFER_TOO_SMALL);
        };
        let pool = lock_pool(handle)?;
        for (destination, owner_id) in output.iter_mut().zip(owner_ids) {
            let slot = pool.current(*owner_id);
            *destination = OrbitKvStateCurrent {
                owner_id: *owner_id,
                slot: slot.unwrap_or_default().into(),
                present: u32::from(slot.is_some()),
                reserved: 0,
            };
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_state_pool_destroy(
    handle: *mut OrbitKvStatePoolHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if !handle.is_null() {
            unsafe { drop(Box::from_raw(handle)) };
        }
        Ok(ORBITKV_STATUS_OK)
    })
}
