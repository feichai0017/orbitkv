//! Owned CUDA resources used by operator implementations and benchmarks.

use crate::CudaExecutorError;
use loom_cuda_sys as sys;
use std::ffi::{c_void, CStr};
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr::NonNull;

/// An owned non-blocking CUDA stream.
#[derive(Debug)]
pub struct CudaStream(NonNull<c_void>);

impl CudaStream {
    pub fn new() -> Result<Self, CudaExecutorError> {
        let mut stream = std::ptr::null_mut();
        cuda_runtime_result(unsafe {
            sys::cudaStreamCreateWithFlags(&mut stream, sys::CUDA_STREAM_NON_BLOCKING)
        })?;
        let stream = NonNull::new(stream).ok_or_else(|| {
            CudaExecutorError::InvalidContract("CUDA returned a null stream".into())
        })?;
        Ok(Self(stream))
    }

    pub fn synchronize(&self) -> Result<(), CudaExecutorError> {
        cuda_runtime_result(unsafe { sys::cudaStreamSynchronize(self.raw()) })
    }

    pub const fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let _ = unsafe { sys::cudaStreamDestroy(self.raw()) };
    }
}

/// An owned CUDA allocation with an element count known to Rust.
#[derive(Debug)]
pub struct DeviceBuffer<T> {
    pointer: NonNull<T>,
    len: usize,
    marker: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    pub fn uninitialized(len: usize) -> Result<Self, CudaExecutorError> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            CudaExecutorError::InvalidContract("device allocation size overflow".into())
        })?;
        if bytes == 0 {
            return Err(CudaExecutorError::InvalidContract(
                "zero-sized device allocations are not supported".into(),
            ));
        }

        let mut pointer = std::ptr::null_mut();
        cuda_runtime_result(unsafe { sys::cudaMalloc(&mut pointer, bytes) })?;
        let pointer = NonNull::new(pointer.cast::<T>()).ok_or_else(|| {
            CudaExecutorError::InvalidContract("CUDA returned a null allocation".into())
        })?;
        Ok(Self {
            pointer,
            len,
            marker: PhantomData,
        })
    }

    pub fn from_slice(values: &[T]) -> Result<Self, CudaExecutorError> {
        let mut allocation = Self::uninitialized(values.len())?;
        allocation.copy_from_slice(values)?;
        Ok(allocation)
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn copy_from_slice(&mut self, values: &[T]) -> Result<(), CudaExecutorError> {
        self.require_len(values.len(), "host-to-device copy")?;
        cuda_runtime_result(unsafe {
            sys::cudaMemcpy(
                self.as_mut_ptr().cast::<c_void>(),
                values.as_ptr().cast::<c_void>(),
                self.len * size_of::<T>(),
                sys::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        })
    }

    pub fn copy_to_vec(&self) -> Result<Vec<T>, CudaExecutorError>
    where
        T: Default,
    {
        let mut values = vec![T::default(); self.len];
        cuda_runtime_result(unsafe {
            sys::cudaMemcpy(
                values.as_mut_ptr().cast::<c_void>(),
                self.as_ptr().cast::<c_void>(),
                self.len * size_of::<T>(),
                sys::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        })?;
        Ok(values)
    }

    pub(crate) const fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }

    pub(crate) const fn as_mut_ptr(&mut self) -> *mut T {
        self.pointer.as_ptr()
    }

    pub(crate) fn require_len(&self, actual: usize, name: &str) -> Result<(), CudaExecutorError> {
        if actual == self.len {
            Ok(())
        } else {
            Err(CudaExecutorError::InvalidContract(format!(
                "{name} has {actual} elements, expected {}",
                self.len
            )))
        }
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        let _ = unsafe { sys::cudaFree(self.pointer.as_ptr().cast::<c_void>()) };
    }
}

/// CUDA event used for device-side elapsed-time measurements.
#[derive(Debug)]
pub struct CudaEvent(NonNull<c_void>);

impl CudaEvent {
    pub fn new() -> Result<Self, CudaExecutorError> {
        let mut event = std::ptr::null_mut();
        cuda_runtime_result(unsafe { sys::cudaEventCreate(&mut event) })?;
        let event = NonNull::new(event).ok_or_else(|| {
            CudaExecutorError::InvalidContract("CUDA returned a null event".into())
        })?;
        Ok(Self(event))
    }

    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaExecutorError> {
        cuda_runtime_result(unsafe { sys::cudaEventRecord(self.raw(), stream.raw()) })
    }

    pub fn synchronize(&self) -> Result<(), CudaExecutorError> {
        cuda_runtime_result(unsafe { sys::cudaEventSynchronize(self.raw()) })
    }

    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, CudaExecutorError> {
        let mut milliseconds = 0.0;
        cuda_runtime_result(unsafe {
            sys::cudaEventElapsedTime(&mut milliseconds, self.raw(), end.raw())
        })?;
        Ok(milliseconds)
    }

    const fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let _ = unsafe { sys::cudaEventDestroy(self.raw()) };
    }
}

pub(crate) fn loom_status_result(status: i32) -> Result<(), CudaExecutorError> {
    if status == sys::LOOM_CUDA_SUCCESS {
        return Ok(());
    }
    let message = unsafe {
        let pointer = sys::loom_cuda_status_string(status);
        if pointer.is_null() {
            "unknown Loom CUDA status".to_owned()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(CudaExecutorError::KernelSubmission { status, message })
}

fn cuda_runtime_result(status: i32) -> Result<(), CudaExecutorError> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe {
        let pointer = sys::cudaGetErrorString(status);
        if pointer.is_null() {
            "unknown CUDA runtime status".to_owned()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(CudaExecutorError::KernelSubmission { status, message })
}
