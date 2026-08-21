#![allow(clippy::missing_panics_doc)]

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_char;
use std::slice;
use std::sync::{Mutex, MutexGuard};

use orbitkv::kv_manager::{
    ArenaStats, BackendArenaRegistration, BackendBindReceipt, BackendCopyReceipt,
    BackendUnobservedReceipt, BatchCompletionReceipt, CanonicalKvManager, ClassLowering,
    CopyIntent, DetachedBinding, KvManagerError, ManagerConfig, ManagerStats, PageLease,
    PrefixAttachItem, PrefixLease, PrefixLookupHint, PrefixPublishItem, PrefixSemanticKey,
    PrepareBatchItem, ReclamationCertificate, ReclamationLease, ReclamationReceipt,
    ReleaseBatchItem, RequestForkItem, RequestLease, RequestView, SnapshotLease, SnapshotPage,
    StepLease, SubmissionLease, SubmitBatchItem, TailAction, WriteIntent,
};
use orbitkv::{KvPlanInput, compile_plan};

use crate::{
    ORBITKV_ABI_VERSION, ORBITKV_STATUS_BUFFER_TOO_SMALL, ORBITKV_STATUS_FAIL_STOPPED,
    ORBITKV_STATUS_INVALID_ARGUMENT, ORBITKV_STATUS_MANAGER_ERROR, ORBITKV_STATUS_OK,
    ORBITKV_STATUS_RETRYABLE_CONFLICT, ffi_boundary,
};

const MAX_PLAN_JSON_BYTES: usize = 1024 * 1024;
pub const ORBITKV_TAIL_NONE: u16 = 0;
pub const ORBITKV_TAIL_IN_PLACE: u16 = 1;
pub const ORBITKV_TAIL_COPY_ON_WRITE: u16 = 2;
pub const ORBITKV_TAIL_FRESH: u16 = 3;

pub struct OrbitKvManagerHandle {
    manager: Mutex<CanonicalKvManager>,
    total_page_capacity: u32,
    maximum_requests: u32,
    maximum_operations: u32,
    maximum_prefixes: u32,
    class_count: u32,
    maximum_write_intents_per_item: u32,
    maximum_completion_outputs_per_item: u32,
    arena_identities: Box<[OrbitKvArenaIdentity]>,
}

mod admin;
mod conversions;
mod layouts;
mod prefix;
mod reclamation;
mod request;
mod transaction;

pub use admin::*;
pub use layouts::*;
pub use prefix::*;
pub use reclamation::*;
pub use request::*;
pub use transaction::*;

fn invalid<T>(message: &str) -> Result<T, (i32, String)> {
    Err(invalid_pair(message))
}

fn invalid_pair(message: &str) -> (i32, String) {
    (ORBITKV_STATUS_INVALID_ARGUMENT, message.to_owned())
}

#[allow(clippy::needless_pass_by_value)]
fn core_error(error: KvManagerError) -> (i32, String) {
    let status = match &error {
        KvManagerError::StaleLease(_)
        | KvManagerError::StaleView
        | KvManagerError::PrefixMiss
        | KvManagerError::PrefixHintStale
        | KvManagerError::DuplicatePrefixKey => ORBITKV_STATUS_RETRYABLE_CONFLICT,
        KvManagerError::BatchQuarantined(_) => ORBITKV_STATUS_FAIL_STOPPED,
        _ => ORBITKV_STATUS_MANAGER_ERROR,
    };
    (status, error.to_string())
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
) -> Result<MutexGuard<'_, CanonicalKvManager>, (i32, String)> {
    handle.manager.lock().map_err(|_| {
        (
            ORBITKV_STATUS_MANAGER_ERROR,
            "manager lock is poisoned".to_owned(),
        )
    })
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
        return invalid(&format!("{label} input buffer is required"));
    }
    Ok(unsafe { slice::from_raw_parts(input, count as usize) })
}

unsafe fn preflight_output<T>(
    output: *mut T,
    capacity: u32,
    out_count: *mut u32,
    required: u32,
    label: &str,
) -> Result<bool, (i32, String)> {
    if out_count.is_null() {
        return invalid(&format!("{label} count output is required"));
    }
    unsafe { out_count.write(required) };
    if capacity < required {
        return Ok(true);
    }
    if required != 0 && output.is_null() {
        return invalid(&format!("{label} output buffer is required"));
    }
    Ok(false)
}

unsafe fn write_copy_slice<T: Copy>(values: &[T], output: *mut T) {
    for (index, value) in values.iter().copied().enumerate() {
        unsafe { output.add(index).write(value) };
    }
}

unsafe fn write_converted<T: Clone, U: From<T>>(values: &[T], output: *mut U) {
    for (index, value) in values.iter().cloned().enumerate() {
        unsafe { output.add(index).write(value.into()) };
    }
}

unsafe fn write_snapshot_pages(values: &[SnapshotPage], output: *mut OrbitKvSnapshotPage) {
    unsafe { write_converted(values, output) };
}

fn exact_len(length: usize) -> u32 {
    u32::try_from(length).expect("core output count fits preflighted ABI envelope")
}

fn u32_len(length: usize, label: &str) -> Result<u32, (i32, String)> {
    u32::try_from(length).map_err(|_| invalid_pair(&format!("{label} exceeds uint32_t")))
}

fn checked_mul(left: u32, right: u32, label: &str) -> Result<u32, (i32, String)> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_pair(&format!("{label} bound exceeds uint32_t")))
}

fn validate_count_limit(count: u32, maximum: u32, label: &str) -> Result<(), (i32, String)> {
    if count > maximum {
        return invalid(&format!(
            "{label} count {count} exceeds configured maximum {maximum}"
        ));
    }
    Ok(())
}

fn validate_nonzero_limit(count: u32, maximum: u32, label: &str) -> Result<(), (i32, String)> {
    if count == 0 {
        return invalid(&format!("{label} batch must not be empty"));
    }
    validate_count_limit(count, maximum, label)
}

fn maximum_batch(handle: &OrbitKvManagerHandle) -> u32 {
    handle.maximum_requests.min(handle.maximum_operations)
}

fn validate_submit_spans(
    items: &[OrbitKvSubmitBatchItem],
    receipt_count: u32,
    copy_receipt_count: u32,
) -> Result<(), (i32, String)> {
    let mut expected_receipt = 0_u32;
    let mut expected_copy = 0_u32;
    for item in items {
        if item.receipt_offset != expected_receipt || item.copy_receipt_offset != expected_copy {
            return invalid("submit spans must be canonical and gap-free");
        }
        expected_receipt = expected_receipt
            .checked_add(item.receipt_count)
            .ok_or_else(|| invalid_pair("submit receipt span overflows"))?;
        expected_copy = expected_copy
            .checked_add(item.copy_receipt_count)
            .ok_or_else(|| invalid_pair("submit copy span overflows"))?;
    }
    if expected_receipt != receipt_count || expected_copy != copy_receipt_count {
        return invalid("submit spans must cover their flat buffers exactly");
    }
    Ok(())
}

fn validate_hint(
    hint: &OrbitKvPrefixLookupHint,
    require_candidate: bool,
) -> Result<(), (i32, String)> {
    if hint.reserved != 0 || hint.reserved_padding != 0 {
        return invalid("prefix hint reserved fields must be zero");
    }
    if hint.candidate_present > 1 || (require_candidate && hint.candidate_present != 1) {
        return invalid("prefix hint candidate-present field is invalid");
    }
    if hint.candidate_present == 0
        && (hint.candidate != OrbitKvPrefixLease::default() || hint.resident_count != 0)
    {
        return invalid("prefix miss hint must use a zero candidate and resident count");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
