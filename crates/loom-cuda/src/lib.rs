//! Safe Rust CUDA backend for Loom Kernels.
//!
//! CUDA is opt-in so the default workspace remains buildable on machines
//! without an NVIDIA toolkit. Enabling `cuda` compiles the handwritten kernels
//! and exposes owned or borrowed streams and device memory, events, and checked
//! operator entrypoints.

use thiserror::Error;

#[cfg(feature = "cuda")]
mod activation;
#[cfg(feature = "cuda")]
mod attention;
#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod layout;
#[cfg(feature = "cuda")]
pub use layout::{PagedDecodeLayout, RopePagedKvLayout, RowStridedLayout};
#[cfg(feature = "cuda")]
mod logits;
#[cfg(feature = "cuda")]
mod norm;
#[cfg(feature = "cuda")]
mod rope_kv;
#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
mod sampling;
#[cfg(feature = "cuda")]
mod speculative;
#[cfg(feature = "cuda")]
pub use activation::{Fp8ScaleLayout, SiluAndMulDynamicFp8Options};
#[cfg(feature = "cuda")]
pub use attention::paged_decode_attention_split_k_workspace_elements;
#[cfg(feature = "cuda")]
pub use backend::CudaBackend;

/// Validation, availability, or CUDA launch failure.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CudaExecutorError {
    #[error("invalid operator contract: {0}")]
    InvalidContract(String),
    #[error("Loom Kernels was built without the CUDA feature")]
    BackendUnavailable,
    #[error("CUDA kernel submission failed with status {status}: {message}")]
    KernelSubmission { status: i32, message: String },
}

/// Whether this build contains the native CUDA backend.
pub const fn compiled_with_cuda() -> bool {
    loom_cuda_sys::compiled_with_cuda()
}
