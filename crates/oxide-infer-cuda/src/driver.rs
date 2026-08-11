//! Shared raw-driver helpers for resource cleanup paths.

use cuda_core::{CudaContext, DriverError, IntoResult, sys};
use std::mem::MaybeUninit;

/// Binds `context` without consulting cuda-oxide's sticky error state.
///
/// Cleanup must still release CUDA resources after an earlier asynchronous
/// error. Normal execution paths must use `CudaContext::bind_to_thread`.
pub(crate) fn bind_context_for_cleanup(context: &CudaContext) -> Result<(), DriverError> {
    let mut current = MaybeUninit::uninit();
    // SAFETY: CUDA writes one context handle to `current`. `context` remains
    // live for this call and is the exact owner of the resources being freed.
    unsafe {
        sys::cuCtxGetCurrent(current.as_mut_ptr()).result()?;
        let current = current.assume_init();
        if current.is_null() || current != context.cu_ctx() {
            sys::cuCtxSetCurrent(context.cu_ctx()).result()?;
        }
    }
    Ok(())
}
