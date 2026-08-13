#![cfg_attr(feature = "cuda", allow(internal_features))]
#![cfg_attr(feature = "cuda", feature(core_intrinsics))]

//! Checked CUDA resources and offline artifact contracts for Oxide Infer.
//!
//! The artifact contract is pure Rust. Enable the transitional `cuda` feature
//! inside the pinned cuda-oxide toolchain for the current device providers.

pub mod artifact;

#[cfg(feature = "cuda")]
pub mod attention;
#[cfg(feature = "cuda")]
pub mod command;
#[cfg(feature = "cuda")]
mod device_status;
#[cfg(feature = "cuda")]
mod driver;
#[cfg(feature = "cuda")]
pub mod gemm;
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod interop;
#[cfg(feature = "cuda")]
pub mod memory;
#[cfg(feature = "cuda")]
pub mod rms_norm;
#[cfg(feature = "cuda")]
pub mod rope;
