//! Shared support for Loom Infer hardware validation programs.

#![forbid(unsafe_code)]

#[cfg(feature = "cuda")]
pub mod comparison;
pub mod reporting;
