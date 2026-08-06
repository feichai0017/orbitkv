//! Shared support for Loom Infer hardware validation programs.

#![forbid(unsafe_code)]

mod support;

#[cfg(feature = "cuda")]
pub mod benchmarks;
#[cfg(feature = "cuda")]
pub mod gates;

pub use support::reporting;
#[cfg(feature = "cuda")]
pub use support::{benchmark, comparison, fixture};
