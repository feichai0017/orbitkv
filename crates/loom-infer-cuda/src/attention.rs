//! CUDA attention providers.
//!
//! Public paths remain stable while private modules follow operator domains.
//! The decode backend keeps one inline cuda-oxide artifact bundle because the
//! macro must discover every kernel in the same token tree.

mod decode;
mod prefill;

pub use decode::{
    AttentionProvider, Bf16PagedBatchDecodeAlgorithm, Bf16PagedBatchDecodeArgs,
    Bf16PagedBatchDecodePlan, Bf16SingleDecodeArgs, Bf16SingleDecodePlan,
    Bf16SingleDecodeSplitKArgs, Bf16SingleDecodeSplitKPlan, PagedBatchDecodeEnqueueError,
    PagedBatchDecodePlanError, SingleDecodeEnqueueError, SingleDecodePlanError,
};
pub use prefill::{
    Bf16RaggedPrefillAlgorithm, Bf16RaggedPrefillArgs, Bf16RaggedPrefillPlan, PrefillProvider,
    RaggedPrefillEnqueueError, RaggedPrefillPlanError,
};
