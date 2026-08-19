use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::Mutex;

use orbitkv::{
    CacheKind, KvPlanSource, OwnerCommand, OwnerResponse, RuntimeExecutionMode, RuntimeStatePlan,
    SglangExecutionProof, SglangOwner, SglangSemanticProof,
};

mod dense;

pub use dense::*;

pub const ORBITKV_OWNER_ABI_VERSION: u32 = 1;
pub const ORBITKV_STATUS_OK: i32 = 0;
pub const ORBITKV_STATUS_NO_CERTIFICATE: i32 = 1;
pub const ORBITKV_STATUS_INVALID_ARGUMENT: i32 = -1;
pub const ORBITKV_STATUS_OWNER_ERROR: i32 = -2;
pub const ORBITKV_STATUS_PANIC: i32 = -3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvCertificateV1 {
    pub abi_version: u32,
    pub reserved: u32,
    pub certificate_id: u64,
    pub page_tokens: u64,
    pub token_start: u64,
    pub token_end_exclusive: u64,
    pub semantic_frontier: u64,
    pub window_tokens: u64,
    pub maximum_reclaimable_end: u64,
    pub execution_epoch: u64,
    pub plan_fingerprint: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvOwnerStatsV1 {
    pub abi_version: u32,
    pub reserved: u32,
    pub tracked_requests: u64,
    pub pending_certificates: u64,
    pub committed_reclamations: u64,
    pub committed_tokens: u64,
    pub plan_fingerprint: [u8; 32],
}

pub struct OrbitKvOwnerHandle {
    owner: Mutex<SglangOwner>,
}

#[unsafe(no_mangle)]
pub extern "C" fn orbitkv_owner_abi_version() -> u32 {
    ORBITKV_OWNER_ABI_VERSION
}

/// Creates an owner from a UTF-8 JSON runtime `StatePlan` or legacy retention plan.
///
/// # Safety
///
/// `plan_json` must reference `plan_json_len` readable bytes. `out_owner` must
/// reference writable storage for one pointer. The error buffer contract is
/// described by [`write_error`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_owner_create(
    plan_json: *const u8,
    plan_json_len: usize,
    out_owner: *mut *mut OrbitKvOwnerHandle,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        if plan_json.is_null() || plan_json_len == 0 || out_owner.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "plan JSON and output owner pointer are required".to_owned(),
            ));
        }
        unsafe {
            out_owner.write(std::ptr::null_mut());
        }
        let bytes = unsafe { slice::from_raw_parts(plan_json, plan_json_len) };
        let plan = if let Ok(artifact) = serde_json::from_slice::<RuntimeStatePlan>(bytes) {
            artifact
                .validate()
                .map_err(|error| (ORBITKV_STATUS_INVALID_ARGUMENT, error.to_string()))?;
            if artifact.execution.mode != RuntimeExecutionMode::Owner {
                return Err((
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    "runtime StatePlan execution mode is not owner".to_owned(),
                ));
            }
            artifact
                .semantic_source
                .compile()
                .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?
        } else {
            let source = serde_json::from_slice::<KvPlanSource>(bytes).map_err(|error| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    format!("invalid runtime StatePlan or retention plan JSON: {error}"),
                )
            })?;
            source
                .compile()
                .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?
        };
        let owner = SglangOwner::new(&plan)
            .map_err(|error| (ORBITKV_STATUS_OWNER_ERROR, error.to_string()))?;
        let handle = Box::new(OrbitKvOwnerHandle {
            owner: Mutex::new(owner),
        });
        unsafe {
            out_owner.write(Box::into_raw(handle));
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Plans one page-aligned SWA chunk-cache reclamation.
///
/// # Safety
///
/// `owner` must be a live pointer returned by [`orbitkv_owner_create`].
/// `request_id` must reference `request_id_len` readable UTF-8 bytes and
/// `out_certificate` must reference writable certificate storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_owner_plan_chunk_reclamation(
    owner: *mut OrbitKvOwnerHandle,
    request_id: *const u8,
    request_id_len: usize,
    observed_evicted_seqlen: u64,
    semantic_frontier: u64,
    execution_epoch: u64,
    out_certificate: *mut OrbitKvCertificateV1,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { owner_ref(owner)? };
        let request_id = unsafe { utf8_input(request_id, request_id_len, "request id")? };
        if out_certificate.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "output certificate pointer is required".to_owned(),
            ));
        }
        let mut owner = handle.owner.lock().map_err(|_| {
            (
                ORBITKV_STATUS_OWNER_ERROR,
                "owner mutex is poisoned".to_owned(),
            )
        })?;
        match owner.execute(OwnerCommand::PlanReclamation {
            request_id: request_id.to_owned(),
            observed_evicted_seqlen,
            semantic_frontier,
            execution_epoch,
            cache_kind: CacheKind::Chunk,
        }) {
            OwnerResponse::Reclamation { certificate: None } => Ok(ORBITKV_STATUS_NO_CERTIFICATE),
            OwnerResponse::Reclamation {
                certificate: Some(certificate),
            } => {
                let semantic = match certificate.semantic_proof {
                    SglangSemanticProof::SlidingWindow {
                        semantic_frontier,
                        window_tokens,
                        maximum_reclaimable_end,
                    } => (semantic_frontier, window_tokens, maximum_reclaimable_end),
                };
                let execution_epoch = match certificate.execution_proof {
                    SglangExecutionProof::NonOverlapSchedulerBarrier { execution_epoch } => {
                        execution_epoch
                    }
                    SglangExecutionProof::CompletionFrontiers { .. } => {
                        return Err((
                            ORBITKV_STATUS_OWNER_ERROR,
                            "owner ABI v1 cannot encode completion-frontier proof".to_owned(),
                        ));
                    }
                };
                let output = OrbitKvCertificateV1 {
                    abi_version: ORBITKV_OWNER_ABI_VERSION,
                    reserved: 0,
                    certificate_id: certificate.certificate_id,
                    page_tokens: certificate.page_tokens,
                    token_start: certificate.token_start,
                    token_end_exclusive: certificate.token_end_exclusive,
                    semantic_frontier: semantic.0,
                    window_tokens: semantic.1,
                    maximum_reclaimable_end: semantic.2,
                    execution_epoch,
                    plan_fingerprint: fingerprint_bytes(&certificate.plan_fingerprint)?,
                };
                unsafe {
                    out_certificate.write(output);
                }
                Ok(ORBITKV_STATUS_OK)
            }
            response => owner_response_error(response),
        }
    })
}

/// Atomically commits a batch of physically completed certificates.
///
/// # Safety
///
/// `owner` must be live and `certificate_ids` must reference
/// `certificate_count` readable `u64` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_owner_commit_reclamations(
    owner: *mut OrbitKvOwnerHandle,
    certificate_ids: *const u64,
    certificate_count: usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { owner_ref(owner)? };
        if certificate_count > 0 && certificate_ids.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "certificate id pointer is required".to_owned(),
            ));
        }
        let certificate_ids = if certificate_count == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(certificate_ids, certificate_count) }
        };
        let mut owner = handle.owner.lock().map_err(|_| {
            (
                ORBITKV_STATUS_OWNER_ERROR,
                "owner mutex is poisoned".to_owned(),
            )
        })?;
        match owner.execute(OwnerCommand::CommitReclamations {
            certificate_ids: certificate_ids.to_vec(),
        }) {
            OwnerResponse::Committed {
                certificate_ids: committed,
            } if committed == certificate_ids => Ok(ORBITKV_STATUS_OK),
            response => owner_response_error(response),
        }
    })
}

/// Releases request tracking after the engine has released physical state.
///
/// # Safety
///
/// `owner` must be live and `request_id` must reference readable UTF-8 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_owner_release_request(
    owner: *mut OrbitKvOwnerHandle,
    request_id: *const u8,
    request_id_len: usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { owner_ref(owner)? };
        let request_id = unsafe { utf8_input(request_id, request_id_len, "request id")? };
        let mut owner = handle.owner.lock().map_err(|_| {
            (
                ORBITKV_STATUS_OWNER_ERROR,
                "owner mutex is poisoned".to_owned(),
            )
        })?;
        match owner.execute(OwnerCommand::ReleaseRequest {
            request_id: request_id.to_owned(),
        }) {
            OwnerResponse::Released {
                request_id: released,
            } if released == request_id => Ok(ORBITKV_STATUS_OK),
            response => owner_response_error(response),
        }
    })
}

/// Returns owner counters without serialization.
///
/// # Safety
///
/// `owner` must be live and `out_stats` must reference writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_owner_stats(
    owner: *mut OrbitKvOwnerHandle,
    out_stats: *mut OrbitKvOwnerStatsV1,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    ffi_boundary(error_buffer, error_buffer_len, || {
        let handle = unsafe { owner_ref(owner)? };
        if out_stats.is_null() {
            return Err((
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "output stats pointer is required".to_owned(),
            ));
        }
        let owner = handle.owner.lock().map_err(|_| {
            (
                ORBITKV_STATUS_OWNER_ERROR,
                "owner mutex is poisoned".to_owned(),
            )
        })?;
        let stats = owner.stats();
        let output = OrbitKvOwnerStatsV1 {
            abi_version: ORBITKV_OWNER_ABI_VERSION,
            reserved: 0,
            tracked_requests: stats.tracked_requests,
            pending_certificates: stats.pending_certificates,
            committed_reclamations: stats.committed_reclamations,
            committed_tokens: stats.committed_tokens,
            plan_fingerprint: fingerprint_bytes(&stats.plan_fingerprint)?,
        };
        unsafe {
            out_stats.write(output);
        }
        Ok(ORBITKV_STATUS_OK)
    })
}

/// Destroys an owner handle.
///
/// # Safety
///
/// `owner` must either be null or a pointer returned by
/// [`orbitkv_owner_create`] that has not already been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orbitkv_owner_destroy(owner: *mut OrbitKvOwnerHandle) {
    if owner.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(owner));
    }));
}

type FfiResult = Result<i32, (i32, String)>;

fn ffi_boundary(
    error_buffer: *mut c_char,
    error_buffer_len: usize,
    operation: impl FnOnce() -> FfiResult,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(status)) => {
            unsafe {
                clear_error(error_buffer, error_buffer_len);
            }
            status
        }
        Ok(Err((status, message))) => {
            unsafe {
                write_error(error_buffer, error_buffer_len, &message);
            }
            status
        }
        Err(_) => {
            unsafe {
                write_error(
                    error_buffer,
                    error_buffer_len,
                    "panic crossed the ABI boundary",
                );
            }
            ORBITKV_STATUS_PANIC
        }
    }
}

unsafe fn owner_ref<'a>(
    owner: *mut OrbitKvOwnerHandle,
) -> Result<&'a OrbitKvOwnerHandle, (i32, String)> {
    if owner.is_null() {
        return Err((
            ORBITKV_STATUS_INVALID_ARGUMENT,
            "owner pointer is required".to_owned(),
        ));
    }
    Ok(unsafe { &*owner })
}

unsafe fn utf8_input<'a>(
    input: *const u8,
    input_len: usize,
    name: &str,
) -> Result<&'a str, (i32, String)> {
    if input.is_null() || input_len == 0 {
        return Err((
            ORBITKV_STATUS_INVALID_ARGUMENT,
            format!("{name} is required"),
        ));
    }
    let bytes = unsafe { slice::from_raw_parts(input, input_len) };
    std::str::from_utf8(bytes).map_err(|error| {
        (
            ORBITKV_STATUS_INVALID_ARGUMENT,
            format!("{name} is not UTF-8: {error}"),
        )
    })
}

fn owner_response_error(response: OwnerResponse) -> FfiResult {
    match response {
        OwnerResponse::Error { code, message } => {
            Err((ORBITKV_STATUS_OWNER_ERROR, format!("[{code}] {message}")))
        }
        other => Err((
            ORBITKV_STATUS_OWNER_ERROR,
            format!("unexpected owner response: {other:?}"),
        )),
    }
}

fn fingerprint_bytes(fingerprint: &str) -> Result<[u8; 32], (i32, String)> {
    let hex = fingerprint.strip_prefix("sha256:").ok_or_else(|| {
        (
            ORBITKV_STATUS_OWNER_ERROR,
            format!("unsupported plan fingerprint: {fingerprint}"),
        )
    })?;
    if hex.len() != 64 {
        return Err((
            ORBITKV_STATUS_OWNER_ERROR,
            format!("invalid SHA-256 fingerprint length: {}", hex.len()),
        ));
    }
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&hex[start..start + 2], 16).map_err(|error| {
            (
                ORBITKV_STATUS_OWNER_ERROR,
                format!("invalid SHA-256 fingerprint: {error}"),
            )
        })?;
    }
    Ok(output)
}

/// Writes a NUL-terminated UTF-8 error message, truncating to fit.
///
/// # Safety
///
/// If non-null, `error_buffer` must reference `error_buffer_len` writable
/// bytes.
unsafe fn write_error(error_buffer: *mut c_char, error_buffer_len: usize, message: &str) {
    if error_buffer.is_null() || error_buffer_len == 0 {
        return;
    }
    let sanitized = CString::new(message).unwrap_or_else(|_| {
        CString::new("error message contained an interior NUL").expect("static CString")
    });
    let bytes = sanitized.as_bytes();
    let copy_len = bytes.len().min(error_buffer_len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), error_buffer.cast::<u8>(), copy_len);
        error_buffer.add(copy_len).write(0);
    }
}

unsafe fn clear_error(error_buffer: *mut c_char, error_buffer_len: usize) {
    if !error_buffer.is_null() && error_buffer_len > 0 {
        unsafe {
            error_buffer.write(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use orbitkv::{
        KvClassSpec, KvPlanInput, RetentionKind, RuntimeCapsuleContract, RuntimeExecutionContract,
        RuntimeExecutionMode, RuntimeOwnerTransport, RuntimeStatePlanOptions,
        compile_runtime_state_plan,
    };

    use super::*;

    const PLAN: &[u8] = br#"{
      "page_tokens": 16,
      "classes": [
        {
          "name": "full",
          "layers": [0],
          "retention": "full",
          "bytes_per_token_per_layer": 128
        },
        {
          "name": "swa",
          "layers": [1],
          "retention": "sliding",
          "bytes_per_token_per_layer": 128,
          "window_tokens": 32
        }
      ]
    }"#;

    #[test]
    fn typed_abi_requires_physical_commit() {
        let mut handle = std::ptr::null_mut();
        let mut error = [0_i8; 256];
        assert_eq!(
            unsafe {
                orbitkv_owner_create(
                    PLAN.as_ptr(),
                    PLAN.len(),
                    &raw mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert!(!handle.is_null());

        let request = b"r0";
        let mut certificate = OrbitKvCertificateV1::default();
        assert_eq!(
            unsafe {
                orbitkv_owner_plan_chunk_reclamation(
                    handle,
                    request.as_ptr(),
                    request.len(),
                    0,
                    64,
                    1,
                    &raw mut certificate,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(certificate.token_end_exclusive, 32);
        assert_eq!(certificate.abi_version, ORBITKV_OWNER_ABI_VERSION);

        let mut second = OrbitKvCertificateV1::default();
        assert_eq!(
            unsafe {
                orbitkv_owner_plan_chunk_reclamation(
                    handle,
                    request.as_ptr(),
                    request.len(),
                    32,
                    80,
                    2,
                    &raw mut second,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OWNER_ERROR
        );

        assert_eq!(
            unsafe {
                orbitkv_owner_commit_reclamations(
                    handle,
                    &raw const certificate.certificate_id,
                    1,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(
            unsafe {
                orbitkv_owner_plan_chunk_reclamation(
                    handle,
                    request.as_ptr(),
                    request.len(),
                    32,
                    80,
                    2,
                    &raw mut second,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );

        unsafe {
            orbitkv_owner_destroy(handle);
        }
    }

    #[test]
    fn invalid_plan_fails_without_creating_owner() {
        let mut handle = std::ptr::null_mut();
        let mut error = [0_i8; 256];
        assert_eq!(
            unsafe {
                orbitkv_owner_create(
                    b"{}".as_ptr(),
                    2,
                    &raw mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_INVALID_ARGUMENT
        );
        assert!(handle.is_null());
        assert_ne!(error[0], 0);
    }

    #[test]
    fn owner_accepts_the_unified_runtime_state_plan() {
        let artifact = compile_runtime_state_plan(
            KvPlanSource::Legacy(KvPlanInput {
                page_tokens: 16,
                classes: vec![
                    KvClassSpec {
                        name: "full".into(),
                        layers: vec![0],
                        retention: RetentionKind::Full,
                        bytes_per_token_per_layer: 128,
                        window_tokens: None,
                    },
                    KvClassSpec {
                        name: "swa".into(),
                        layers: vec![1],
                        retention: RetentionKind::Sliding,
                        bytes_per_token_per_layer: 128,
                        window_tokens: Some(32),
                    },
                ],
            }),
            RuntimeStatePlanOptions {
                eviction_interval_tokens: 16,
                physical_plan: None,
                uniform_state_plan: None,
                dense_runtime: None,
                execution: RuntimeExecutionContract {
                    mode: RuntimeExecutionMode::Owner,
                    owner_transport: Some(RuntimeOwnerTransport::Ffi),
                    uniform_state_plan_mode: None,
                    frontier: None,
                },
                capsule: RuntimeCapsuleContract {
                    enabled: true,
                    chunk_tokens: 16,
                    maximum_payload_bytes: 1 << 20,
                },
                prefix: None,
            },
        )
        .unwrap();
        let bytes = serde_json::to_vec(&artifact).unwrap();
        let mut handle = std::ptr::null_mut();
        let mut error = [0_i8; 256];
        assert_eq!(
            unsafe {
                orbitkv_owner_create(
                    bytes.as_ptr(),
                    bytes.len(),
                    &raw mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert!(!handle.is_null());
        unsafe {
            orbitkv_owner_destroy(handle);
        }
    }

    #[test]
    fn abi_layout_is_stable() {
        assert_eq!(std::mem::size_of::<OrbitKvCertificateV1>(), 104);
        assert_eq!(std::mem::align_of::<OrbitKvCertificateV1>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvCertificateV1, certificate_id),
            8
        );
        assert_eq!(
            std::mem::offset_of!(OrbitKvCertificateV1, plan_fingerprint),
            72
        );
        assert_eq!(std::mem::size_of::<OrbitKvOwnerStatsV1>(), 72);
        assert_eq!(std::mem::align_of::<OrbitKvOwnerStatsV1>(), 8);
        assert_eq!(
            std::mem::offset_of!(OrbitKvOwnerStatsV1, plan_fingerprint),
            40
        );
    }
}
