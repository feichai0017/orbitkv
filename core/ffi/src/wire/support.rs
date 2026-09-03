use crate::ORBITKV_STATUS_INVALID_ARGUMENT;

fn invalid<T>(message: String) -> Result<T, (i32, String)> {
    Err((ORBITKV_STATUS_INVALID_ARGUMENT, message))
}

/// Copies one caller-owned typed input range without retaining a reference to
/// caller memory.
///
/// # Safety
///
/// For nonzero `count`, `input` must be aligned for `T` and reference `count`
/// initialized `T` values in one readable allocation. The range's byte length
/// must not exceed `isize::MAX`. The range must not be mutated concurrently
/// while this function copies it.
pub(crate) unsafe fn copy_input<T: Copy>(
    input: *const T,
    count: u32,
    label: &str,
) -> Result<Vec<T>, (i32, String)> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if input.is_null() {
        return invalid(format!("{label} input buffer is required"));
    }
    let count = count as usize;
    let mut owned = Vec::with_capacity(count);
    unsafe {
        std::ptr::copy_nonoverlapping(input, owned.as_mut_ptr(), count);
        owned.set_len(count);
    }
    Ok(owned)
}

/// Preflights one caller-owned output range and publishes its required count.
///
/// # Safety
///
/// `out_count` must be writable when non-null. When `capacity >= required` and
/// `required != 0`, `output` must reference at least `capacity` writable `T`s.
pub(crate) unsafe fn preflight_output<T>(
    output: *mut T,
    capacity: u32,
    out_count: *mut u32,
    required: u32,
    label: &str,
) -> Result<bool, (i32, String)> {
    if out_count.is_null() {
        return invalid(format!("{label} count output is required"));
    }
    unsafe { out_count.write(required) };
    if capacity < required {
        return Ok(true);
    }
    if required != 0 && output.is_null() {
        return invalid(format!("{label} output buffer is required"));
    }
    Ok(false)
}

pub(crate) fn exact_len(length: usize) -> u32 {
    u32::try_from(length).expect("session output count fits preflighted wire envelope")
}

pub(crate) fn checked_mul(left: u32, right: u32, label: &str) -> Result<u32, (i32, String)> {
    left.checked_mul(right).ok_or_else(|| {
        (
            ORBITKV_STATUS_INVALID_ARGUMENT,
            format!("{label} bound overflows"),
        )
    })
}

pub(crate) fn validate_count_limit(
    count: u32,
    maximum: u32,
    label: &str,
) -> Result<(), (i32, String)> {
    if count > maximum {
        return invalid(format!(
            "{label} count {count} exceeds configured maximum {maximum}"
        ));
    }
    Ok(())
}

pub(crate) fn validate_nonzero_limit(
    count: u32,
    maximum: u32,
    label: &str,
) -> Result<(), (i32, String)> {
    if count == 0 {
        return invalid(format!("{label} batch must not be empty"));
    }
    validate_count_limit(count, maximum, label)
}

pub(crate) fn validate_bool(value: u32, label: &str) -> Result<bool, (i32, String)> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => invalid(format!("{label} must be zero or one")),
    }
}
