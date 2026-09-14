//! Provider adapters and their graph-visible execution contracts.

pub mod attention;
mod build;
pub mod cache;
pub(crate) mod cublaslt;
pub mod deepgemm;
pub mod flashattention;
pub mod flashinfer;
pub mod moe;
mod operation;
pub(crate) mod provider_source;
pub mod registry;

pub use crate::resource::{
    HostDeviceMemoryPlan, ResourceViolation, SharedDeviceMemoryAllocation, eval_resource_expression,
};
pub use operation::{CudaGraphCaptureResource, CudaGraphCaptureSharedState, DeviceBuffer, HostOp};

/// Provider operations registered with the CUDA search space.
///
/// Graph builders emit provider-neutral semantic custom ops. These providers
/// contribute legal implementations through egglog and are selected by the
/// ordinary device profiler.
pub type Ops = (
    orbitkv_ops::ops::attention::AttentionSemantics,
    cublaslt::CuBlasLt,
    cublaslt::CuBlasLtScaled,
    deepgemm::DeepGemm,
    deepgemm::BlockScaledQuantize,
    deepgemm::PrequantizedDeepGemm,
    flashattention::FlashAttention,
    moe::GLUMoE,
    flashinfer::FlashInferAttention,
    moe::fused::FusedMoE,
);

#[cfg(test)]
#[path = "../tests/unit/providers/cublaslt/helpers.rs"]
mod test_helpers;
#[cfg(test)]
pub(crate) use test_helpers::*;
