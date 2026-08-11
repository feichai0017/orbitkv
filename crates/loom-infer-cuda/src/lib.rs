#![cfg_attr(feature = "cuda", allow(internal_features))]
#![cfg_attr(feature = "cuda", feature(core_intrinsics))]

//! Rust CUDA providers for Loom Infer.
//!
//! Enable the `cuda` feature inside the pinned cuda-oxide toolchain.

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
