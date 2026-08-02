//! Backend-independent contracts and CPU references for Loom Infer.

#![forbid(unsafe_code)]

mod dtype;
mod error;

pub mod rms_norm;

pub use dtype::DType;
pub use error::ContractError;
pub use rms_norm::{
    RmsNormSpec, rms_norm_bf16_reference, rms_norm_f16_reference, rms_norm_f32_reference,
};
