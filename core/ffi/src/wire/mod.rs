mod conversions;
mod layouts;
mod support;

pub use layouts::*;
pub(crate) use support::{
    checked_mul, copy_input, exact_len, preflight_output, validate_bool, validate_count_limit,
    validate_nonzero_limit,
};

#[unsafe(no_mangle)]
pub extern "C" fn orbitkv_wire_version() -> u32 {
    crate::ORBITKV_WIRE_VERSION
}
