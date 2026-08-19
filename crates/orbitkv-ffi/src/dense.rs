use std::ffi::c_char;
use std::slice;
use std::sync::Mutex;

use orbitkv::{
    ClassId, DenseBackendHandle, DensePhysicalBindingReceipt, DensePhysicalHandle,
    DensePhysicalReclamationReceipt, DenseRetirementCertificate, DenseRuntimeCommand,
    DenseRuntimeService, RequestLease, RuntimeStatePlan,
};

use crate::{
    ORBITKV_STATUS_INVALID_ARGUMENT, ORBITKV_STATUS_OK, ORBITKV_STATUS_OWNER_ERROR, ffi_boundary,
};

pub const ORBITKV_DENSE_ABI_VERSION: u32 = 1;
pub const ORBITKV_DENSE_STATUS_BUFFER_TOO_SMALL: i32 = 2;
pub const ORBITKV_DENSE_RESPONSE_CAPACITY: usize = 1024 * 1024;

pub struct OrbitKvDenseHandle {
    service: Mutex<DenseRuntimeService>,
    certificate_capacity: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvDenseRequestLeaseV1 {
    pub slot: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvDenseCertificateV1 {
    pub abi_version: u32,
    pub class_id: u16,
    pub reserved: u16,
    pub certificate_id: u64,
    pub ordinal: u64,
    pub physical_slot: u64,
    pub physical_generation: u64,
    pub backend_index: u64,
    pub token_start: u64,
    pub token_end_exclusive: u64,
}

#[unsafe(no_mangle)]
pub extern "C" fn orbitkv_dense_abi_version() -> u32 {
    ORBITKV_DENSE_ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn orbitkv_dense_response_capacity() -> usize {
    ORBITKV_DENSE_RESPONSE_CAPACITY
}

/// Creates a Dense Runtime from one validated runtime `StatePlan`.
///
/// # Safety
///
/// `plan_json` must reference `plan_json_len` readable bytes. `out_dense` must
/// reference writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_create(
    plan_json: *const u8,
    plan_json_len: usize,
    out_dense: *mut *mut OrbitKvDenseHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if plan_json.is_null() || plan_json_len == 0 || out_dense.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "runtime StatePlan JSON and output Dense pointer are required".to_owned(),
            ));
        }
        unsafe {
            out_dense.write(std::ptr::null_mut());
        }
        let bytes = unsafe { slice::from_raw_parts(plan_json, plan_json_len) };
        let state_plan =
            serde_json::from_slice::<RuntimeStatePlan>(bytes).map_err(|error| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    format!("invalid runtime StatePlan JSON: {error}"),
                )
            })?;
        let service = DenseRuntimeService::from_state_plan(&state_plan)
            .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?;
        let certificate_capacity = state_plan
            .dense_runtime
            .as_ref()
            .ok_or((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "runtime StatePlan has no Dense artifact".to_owned(),
            ))?
            .classes
            .iter()
            .try_fold(0_u64, |total, class| total.checked_add(class.physical_slots))
            .and_then(|total| usize::try_from(total).ok())
            .ok_or((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "Dense certificate capacity does not fit the host".to_owned(),
            ))?;
        let handle = Box::new(OrbitKvDenseHandle {
            service: Mutex::new(service),
            certificate_capacity,
        });
        unsafe {
            out_dense.write(Box::into_raw(handle));
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Returns the maximum certificate batch emitted by this Dense Runtime.
///
/// # Safety
///
/// `dense` must be null or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_certificate_capacity(
    dense: *mut OrbitKvDenseHandle,
) -> usize {
    if dense.is_null() {
        return 0;
    }
    unsafe { (*dense).certificate_capacity }
}

/// Executes one Dense Runtime JSON command in-process.
///
/// The response length is always written to `out_response_len`. If
/// `response_buffer` is null or too small, the command is not executed and
/// `ORBITKV_DENSE_STATUS_BUFFER_TOO_SMALL` is returned.
///
/// # Safety
///
/// `dense` must be live, `command_json` must reference readable bytes, and
/// `out_response_len` must reference writable `usize` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_execute_json(
    dense: *mut OrbitKvDenseHandle,
    command_json: *const u8,
    command_json_len: usize,
    response_buffer: *mut u8,
    response_buffer_len: usize,
    out_response_len: *mut usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if dense.is_null()
            || command_json.is_null()
            || command_json_len == 0
            || out_response_len.is_null()
        {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "Dense handle, command JSON, and response length are required".to_owned(),
            ));
        }
        let command_bytes = unsafe { slice::from_raw_parts(command_json, command_json_len) };
        let command =
            serde_json::from_slice::<DenseRuntimeCommand>(command_bytes).map_err(|error| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    format!("invalid Dense Runtime command JSON: {error}"),
                )
            })?;
        let handle = unsafe { &*dense };
        let mut service = handle.service.lock().map_err(|_| {
            (
                ORBITKV_STATUS_OWNER_ERROR,
                "Dense Runtime mutex is poisoned".to_owned(),
            )
        })?;
        let preview = service.clone().execute(command.clone());
        let preview_bytes = serde_json::to_vec(&preview).map_err(|error| {
            (
                ORBITKV_STATUS_OWNER_ERROR,
                format!("cannot encode Dense Runtime response preview: {error}"),
            )
        })?;
        unsafe {
            out_response_len.write(preview_bytes.len());
        }
        if response_buffer.is_null() || response_buffer_len < preview_bytes.len() {
            return Ok(ORBITKV_DENSE_STATUS_BUFFER_TOO_SMALL);
        }
        let response = service.execute(command);
        let bytes = serde_json::to_vec(&response).map_err(|error| {
            (
                ORBITKV_STATUS_OWNER_ERROR,
                format!("cannot encode Dense Runtime response: {error}"),
            )
        })?;
        if bytes.len() > response_buffer_len {
            unsafe {
                out_response_len.write(bytes.len());
            }
            return Ok(ORBITKV_DENSE_STATUS_BUFFER_TOO_SMALL);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), response_buffer, bytes.len());
            out_response_len.write(bytes.len());
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Pins one immutable Dense view and returns its submission identity.
///
/// # Safety
///
/// All output pointers must be writable. The handle must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_submit_view(
    dense: *mut OrbitKvDenseHandle,
    request: OrbitKvDenseRequestLeaseV1,
    out_submission_id: *mut u64,
    out_live_blocks: *mut usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if dense.is_null() || out_submission_id.is_null() || out_live_blocks.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "Dense handle and submit outputs are required".to_owned(),
            ));
        }
        let handle = unsafe { &*dense };
        let mut service = handle.service.lock().map_err(|_| dense_mutex_error())?;
        let view = service
            .submit_view(request.into())
            .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?;
        unsafe {
            out_submission_id.write(view.submission_id);
            out_live_blocks.write(view.blocks.len());
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Completes a submission and advances semantic time.
///
/// A non-null binding JSON commits the prepared physical binding before the
/// frontier advances. The certificate buffer must have capacity for
/// `certificate_capacity` entries.
///
/// # Safety
///
/// Input buffers must be readable and output buffers writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_complete_step(
    dense: *mut OrbitKvDenseHandle,
    request: OrbitKvDenseRequestLeaseV1,
    submission_id: u64,
    boundary: u64,
    binding_json: *const u8,
    binding_json_len: usize,
    certificates: *mut OrbitKvDenseCertificateV1,
    certificate_capacity: usize,
    out_certificate_count: *mut usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if dense.is_null() || out_certificate_count.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "Dense handle and certificate count are required".to_owned(),
            ));
        }
        let binding = if binding_json_len == 0 {
            None
        } else {
            if binding_json.is_null() {
                return Err((
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    "binding JSON pointer is required".to_owned(),
                ));
            }
            let bytes = unsafe { slice::from_raw_parts(binding_json, binding_json_len) };
            Some(
                serde_json::from_slice::<DensePhysicalBindingReceipt>(bytes).map_err(|error| {
                    (
                        ORBITKV_STATUS_INVALID_ARGUMENT,
                        format!("invalid Dense binding receipt JSON: {error}"),
                    )
                })?,
            )
        };
        let handle = unsafe { &*dense };
        unsafe {
            out_certificate_count.write(handle.certificate_capacity);
        }
        if certificate_capacity < handle.certificate_capacity {
            return Ok(ORBITKV_DENSE_STATUS_BUFFER_TOO_SMALL);
        }
        if handle.certificate_capacity > 0 && certificates.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "certificate buffer is required".to_owned(),
            ));
        }
        let mut service = handle.service.lock().map_err(|_| dense_mutex_error())?;
        let actual = service
            .complete_step(request.into(), submission_id, boundary, binding.as_ref())
            .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?;
        unsafe {
            out_certificate_count.write(actual.len());
        }
        write_certificates(&actual, certificates);
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Releases one request and writes immediately executable certificates.
///
/// # Safety
///
/// The certificate buffer contract matches [`orbitkv_dense_complete_step`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_release_request(
    dense: *mut OrbitKvDenseHandle,
    request: OrbitKvDenseRequestLeaseV1,
    certificates: *mut OrbitKvDenseCertificateV1,
    certificate_capacity: usize,
    out_certificate_count: *mut usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if dense.is_null() || out_certificate_count.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "Dense handle and certificate count are required".to_owned(),
            ));
        }
        let handle = unsafe { &*dense };
        unsafe {
            out_certificate_count.write(handle.certificate_capacity);
        }
        if certificate_capacity < handle.certificate_capacity {
            return Ok(ORBITKV_DENSE_STATUS_BUFFER_TOO_SMALL);
        }
        if handle.certificate_capacity > 0 && certificates.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "certificate buffer is required".to_owned(),
            ));
        }
        let mut service = handle.service.lock().map_err(|_| dense_mutex_error())?;
        let actual = service
            .release_request(request.into())
            .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?;
        unsafe {
            out_certificate_count.write(actual.len());
        }
        write_certificates(&actual, certificates);
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Commits physical certificates and recycles the request lease.
///
/// # Safety
///
/// `certificates` must reference `certificate_count` readable entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_commit_reclamations_and_recycle(
    dense: *mut OrbitKvDenseHandle,
    request: OrbitKvDenseRequestLeaseV1,
    certificates: *const OrbitKvDenseCertificateV1,
    certificate_count: usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if dense.is_null() || (certificate_count > 0 && certificates.is_null()) {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "Dense handle and certificate buffer are required".to_owned(),
            ));
        }
        let handle = unsafe { &*dense };
        let mut service = handle.service.lock().map_err(|_| dense_mutex_error())?;
        let fingerprint = service.artifact_fingerprint().to_owned();
        let receipts = unsafe {
            certificate_receipts(certificates, certificate_count, &fingerprint)?
        };
        service
            .commit_reclamations_and_recycle(request.into(), &receipts)
            .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Commits physical certificates without recycling the request lease.
///
/// # Safety
///
/// `certificates` must reference `certificate_count` readable entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_commit_reclamations(
    dense: *mut OrbitKvDenseHandle,
    certificates: *const OrbitKvDenseCertificateV1,
    certificate_count: usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if dense.is_null() || (certificate_count > 0 && certificates.is_null()) {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "Dense handle and certificate buffer are required".to_owned(),
            ));
        }
        let handle = unsafe { &*dense };
        let mut service = handle.service.lock().map_err(|_| dense_mutex_error())?;
        let fingerprint = service.artifact_fingerprint().to_owned();
        let receipts = unsafe {
            certificate_receipts(certificates, certificate_count, &fingerprint)?
        };
        service
            .commit_reclamations(&receipts)
            .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?;
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Destroys a Dense Runtime handle.
///
/// # Safety
///
/// `dense` must be null or a live pointer returned by
/// [`orbitkv_dense_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_dense_destroy(dense: *mut OrbitKvDenseHandle) {
    if dense.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(dense));
    }));
}

impl From<OrbitKvDenseRequestLeaseV1> for RequestLease {
    fn from(value: OrbitKvDenseRequestLeaseV1) -> Self {
        Self {
            slot: value.slot,
            generation: value.generation,
        }
    }
}

impl OrbitKvDenseCertificateV1 {
    fn from_certificate(certificate: &DenseRetirementCertificate) -> Self {
        Self {
            abi_version: ORBITKV_DENSE_ABI_VERSION,
            class_id: certificate.logical.class_id.0,
            reserved: 0,
            certificate_id: certificate.certificate_id,
            ordinal: certificate.logical.ordinal,
            physical_slot: certificate.physical.slot,
            physical_generation: certificate.physical.generation,
            backend_index: certificate.backend.index,
            token_start: certificate.token_start,
            token_end_exclusive: certificate.token_end_exclusive,
        }
    }

    fn to_receipt(
        self,
        artifact_fingerprint: &str,
    ) -> Result<DensePhysicalReclamationReceipt, (i32, String)> {
        if self.abi_version != ORBITKV_DENSE_ABI_VERSION {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "Dense certificate ABI version is unsupported".to_owned(),
            ));
        }
        let class_id = ClassId(self.class_id);
        Ok(DensePhysicalReclamationReceipt {
            schema: "orbitkv.dense-physical-reclamation-receipt.v1".into(),
            artifact_fingerprint: artifact_fingerprint.to_owned(),
            certificate_id: self.certificate_id,
            physical: DensePhysicalHandle {
                class_id,
                slot: self.physical_slot,
                generation: self.physical_generation,
            },
            backend: DenseBackendHandle {
                domain: class_id,
                index: self.backend_index,
            },
        })
    }
}

fn write_certificates(
    certificates: &[DenseRetirementCertificate],
    output: *mut OrbitKvDenseCertificateV1,
) {
    for (index, certificate) in certificates.iter().enumerate() {
        unsafe {
            output
                .add(index)
                .write(OrbitKvDenseCertificateV1::from_certificate(certificate));
        }
    }
}

fn dense_mutex_error() -> (i32, String) {
    (
        ORBITKV_STATUS_OWNER_ERROR,
        "Dense Runtime mutex is poisoned".to_owned(),
    )
}

unsafe fn certificate_receipts(
    certificates: *const OrbitKvDenseCertificateV1,
    certificate_count: usize,
    artifact_fingerprint: &str,
) -> Result<Vec<DensePhysicalReclamationReceipt>, (i32, String)> {
    let inputs = if certificate_count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(certificates, certificate_count) }
    };
    inputs
        .iter()
        .map(|certificate| certificate.to_receipt(artifact_fingerprint))
        .collect()
}

#[cfg(test)]
mod tests {
    use orbitkv::{
        DenseRuntimeArtifact, KvClassSpec, KvPlanInput, KvPlanSource, RetentionKind,
        RuntimeCapsuleContract, RuntimeExecutionContract, RuntimeExecutionMode,
        RuntimeOwnerTransport, RuntimeStatePlanOptions, compile_runtime_state_plan,
    };

    use super::*;

    fn state_plan_bytes() -> Vec<u8> {
        let source = KvPlanSource::Legacy(KvPlanInput {
            page_tokens: 4,
            classes: vec![KvClassSpec {
                name: "swa".into(),
                layers: vec![0],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(8),
            }],
        });
        let plan = source.clone().compile().unwrap();
        let artifact = compile_runtime_state_plan(
            source,
            RuntimeStatePlanOptions {
                eviction_interval_tokens: 4,
                physical_plan: None,
                uniform_state_plan: None,
                dense_runtime: Some(DenseRuntimeArtifact::compile(&plan, 1, 4, 16).unwrap()),
                execution: RuntimeExecutionContract {
                    mode: RuntimeExecutionMode::Owner,
                    owner_transport: Some(RuntimeOwnerTransport::Sidecar),
                    uniform_state_plan_mode: None,
                    frontier: None,
                },
                capsule: RuntimeCapsuleContract {
                    enabled: false,
                    chunk_tokens: 4,
                    maximum_payload_bytes: 1 << 20,
                },
                prefix: None,
            },
        )
        .unwrap();
        serde_json::to_vec(&artifact).unwrap()
    }

    #[test]
    fn dense_json_abi_probes_without_executing() {
        let plan = state_plan_bytes();
        let mut handle = std::ptr::null_mut();
        let mut error = [0_i8; 256];
        assert_eq!(
            unsafe {
                orbitkv_dense_create(
                    plan.as_ptr(),
                    plan.len(),
                    &raw mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        let command = br#"{"op":"acquire_request"}"#;
        let mut required = 0_usize;
        assert_eq!(
            unsafe {
                orbitkv_dense_execute_json(
                    handle,
                    command.as_ptr(),
                    command.len(),
                    std::ptr::null_mut(),
                    0,
                    &raw mut required,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_DENSE_STATUS_BUFFER_TOO_SMALL
        );
        assert!(required > 0);
        assert!(required <= ORBITKV_DENSE_RESPONSE_CAPACITY);
        let mut response = vec![0_u8; required];
        assert_eq!(
            unsafe {
                orbitkv_dense_execute_json(
                    handle,
                    command.as_ptr(),
                    command.len(),
                    response.as_mut_ptr(),
                    response.len(),
                    &raw mut required,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        let value: serde_json::Value = serde_json::from_slice(&response[..required]).unwrap();
        assert_eq!(value["status"], "request_acquired");
        assert_eq!(value["request"]["generation"], 1);
        unsafe {
            orbitkv_dense_destroy(handle);
        }
    }

    #[test]
    fn dense_abi_layout_is_stable() {
        assert_eq!(std::mem::size_of::<OrbitKvDenseRequestLeaseV1>(), 8);
        assert_eq!(std::mem::align_of::<OrbitKvDenseRequestLeaseV1>(), 4);
        assert_eq!(std::mem::size_of::<OrbitKvDenseCertificateV1>(), 64);
        assert_eq!(std::mem::align_of::<OrbitKvDenseCertificateV1>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvDenseCertificateV1, certificate_id),
            8
        );
        assert_eq!(
            std::mem::offset_of!(OrbitKvDenseCertificateV1, backend_index),
            40
        );
    }
}
