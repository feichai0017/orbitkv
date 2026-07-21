//! Raw bindings to the dependency-light Loom Kernels CUDA C ABI.

use std::ffi::c_int;
#[cfg(feature = "cuda")]
use std::ffi::{c_char, c_void};

pub const LOOM_CUDA_SUCCESS: c_int = 0;
pub const LOOM_CUDA_INVALID_ARGUMENT: c_int = 1;
pub const LOOM_CUDA_UNSUPPORTED: c_int = 2;
pub const LOOM_CUDA_LAUNCH_ERROR: c_int = 3;
pub const LOOM_CUDA_UNAVAILABLE: c_int = 4;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_STREAM_NON_BLOCKING: u32 = 1;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn loom_cuda_status_string(status: c_int) -> *const c_char;

    pub fn loom_cuda_rms_norm_f32(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_f16(
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_bf16(
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_fp8_f32(
        input: *const f32,
        weight: *const f32,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_fp8_f16(
        input: *const u16,
        weight: *const u16,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_fp8_bf16(
        input: *const u16,
        weight: *const u16,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_add_rms_norm_f32(
        input: *mut f32,
        residual: *mut f32,
        weight: *const f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_add_rms_norm_f16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_add_rms_norm_bf16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_f16(
        input: *const u16,
        output: *mut u16,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_bf16(
        input: *const u16,
        output: *mut u16,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_dynamic_fp8_f16(
        input: *const u16,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        width: u32,
        group_size: u32,
        scale_ub: *const f32,
        scales_transposed: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_dynamic_fp8_bf16(
        input: *const u16,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        width: u32,
        group_size: u32,
        scale_ub: *const f32,
        scales_transposed: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    pub fn cudaFree(pointer: *mut c_void) -> c_int;
    pub fn cudaMemcpy(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        kind: c_int,
    ) -> c_int;
    pub fn cudaGetErrorString(error: c_int) -> *const c_char;
    pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> c_int;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    pub fn cudaEventCreate(event: *mut *mut c_void) -> c_int;
    pub fn cudaEventDestroy(event: *mut c_void) -> c_int;
    pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    pub fn cudaEventSynchronize(event: *mut c_void) -> c_int;
    pub fn cudaEventElapsedTime(
        milliseconds: *mut f32,
        start: *mut c_void,
        end: *mut c_void,
    ) -> c_int;
}

pub const fn compiled_with_cuda() -> bool {
    cfg!(feature = "cuda")
}
