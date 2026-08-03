//! Backend-independent contracts and CPU references for Loom Infer.

#![forbid(unsafe_code)]

mod dtype;
mod error;

pub mod attention;
pub mod gemm;
pub mod rms_norm;

pub use attention::{Bf16SingleDecodeSpec, SINGLE_DECODE_HEAD_DIM, single_decode_bf16_reference};
pub use dtype::DType;
pub use error::ContractError;
pub use gemm::{Bf16GemmSpec, bf16_gemm_reference};
pub use rms_norm::{
    RmsNormSpec, rms_norm_bf16_reference, rms_norm_f16_reference, rms_norm_f32_reference,
};
