//! cuBLASLt handle lifetime and initialization.

use cudarc::{cublaslt::CudaBlasLT, driver::CudaStream};
use std::sync::Arc;

pub(crate) fn try_create_cublaslt(
    stream: Arc<CudaStream>,
) -> std::result::Result<Arc<CudaBlasLT>, String> {
    // One process-wide handle per stream, held forever. Per-op handles were
    // created/destroyed thousands of times across search candidates;
    // `cublasLtDestroy` racing live work on other threads (or running after
    // its CUDA context is gone, via LLIR-graph drop order) corrupts
    // libcublasLt's internal state — observed as SIGSEGV in
    // pthread_mutex_unlock under cublasLtDestroy, flaky fuzz failures, and
    // nvrtc spinning forever on trivial kernels later in the process. The
    // cache keeps a permanent Arc so Drop (and thus cublasLtDestroy) never
    // runs; a handle is ~KBs and streams are process-stable.
    static HANDLES: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, Arc<CudaBlasLT>>>,
    > = std::sync::OnceLock::new();
    let key = stream.cu_stream() as usize;
    let handles = HANDLES.get_or_init(Default::default);
    let mut handles = handles.lock().unwrap();
    if let Some(handle) = handles.get(&key) {
        return Ok(handle.clone());
    }
    let created =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| CudaBlasLT::new(stream))) {
            Ok(Ok(handle)) => Arc::new(handle),
            Ok(Err(err)) => return Err(err.to_string()),
            Err(payload) => {
                let message = if let Some(message) = payload.downcast_ref::<String>() {
                    message.clone()
                } else if let Some(message) = payload.downcast_ref::<&str>() {
                    message.to_string()
                } else {
                    "cuBLASLt initialization panicked".to_string()
                };
                return Err(message);
            }
        };
    handles.insert(key, created.clone());
    Ok(created)
}
