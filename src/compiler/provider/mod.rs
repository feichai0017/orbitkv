//! Typed contracts for external and generated execution providers.
//!
//! Provider selection is deliberately model-neutral. Model lowering supplies
//! tensor shapes and use counts; a capability admits concrete kernels from
//! hardware facts, numerical ABI, and measured qualification evidence.

mod deepgemm_sm90;

pub use deepgemm_sm90::{
    DEEPGEMM_REVISION, DeepGemmKernelContract, DeepGemmSm90Capability, DeepGemmTile, Fp8ProjectionFamily,
    Fp8ProjectionLayout, Fp8ProjectionShape, HistoricalDeepGemmSourceEvidence, LOW_LATENCY_ROW_BUCKETS,
    PACKED_ACTIVATION_ABI, PackedActivationScratch, ProviderContractError, QualificationState, render_aot_source,
};
