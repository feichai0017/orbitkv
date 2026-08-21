use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

mod manager;

pub use manager::*;

pub const ORBITKV_ABI_VERSION: u32 = 5;
pub const ORBITKV_STATUS_OK: i32 = 0;
pub const ORBITKV_STATUS_BUFFER_TOO_SMALL: i32 = 1;
pub const ORBITKV_STATUS_INVALID_ARGUMENT: i32 = -1;
pub const ORBITKV_STATUS_MANAGER_ERROR: i32 = -2;
pub const ORBITKV_STATUS_PANIC: i32 = -3;

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
                    "panic crossed the OrbitKV ABI boundary",
                );
            }
            ORBITKV_STATUS_PANIC
        }
    }
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
