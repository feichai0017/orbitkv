//! FlashInfer kernel families available to compiler selection.
//!
//! Its tensor-core paged-prefill kernel also implements one-query-per-request
//! decode, as in FlashInfer's `use_tensor_cores` decode path. The algorithm is
//! selected by egglog/extraction and retained in every plan/capture identity.

use orbitkv_compiler::dtype::DType;
use orbitkv_ops::ops::attention::PagedKvLayout;

use crate::providers::attention::{
    AttentionKernelCapability, AttentionProviderCapabilities, PageSizes,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlashInferAlgorithm {
    /// Vectorized CUDA-core attention, exactly one query per request.
    #[default]
    CudaCoreDecode,
    /// Tensor-core paged attention for decode and causal packed prefill.
    TensorCore,
}

impl FlashInferAlgorithm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CudaCoreDecode => "cuda-core-decode",
            Self::TensorCore => "tensor-core",
        }
    }

    pub(super) fn from_name(name: &str) -> Option<Self> {
        match name {
            "cuda-core-decode" => Some(Self::CudaCoreDecode),
            "tensor-core" => Some(Self::TensorCore),
            _ => None,
        }
    }
}

// Instantiations admitted by wrapper.cu. F32 has no tensor-core or HD512 kernel.
const HALF_HEAD_DIMENSIONS: &[(usize, usize)] = &[(64, 64), (128, 128), (256, 256), (512, 512)];
const FLOAT_HEAD_DIMENSIONS: &[(usize, usize)] = &[(64, 64), (128, 128), (256, 256)];

const fn capability(algorithm: FlashInferAlgorithm, dtype: DType) -> AttentionKernelCapability {
    AttentionKernelCapability {
        algorithm: algorithm.as_str(),
        dtype,
        head_dimensions: match dtype {
            DType::F32 => FLOAT_HEAD_DIMENSIONS,
            _ => HALF_HEAD_DIMENSIONS,
        },
        layout: PagedKvLayout::TokenMajor,
        page_sizes: PageSizes::Any,
        supports_prefill: matches!(algorithm, FlashInferAlgorithm::TensorCore),
    }
}

pub const CAPABILITIES: AttentionProviderCapabilities = AttentionProviderCapabilities {
    name: "flashinfer",
    compute_majors: 8..=i32::MAX,
    kernels: &[
        capability(FlashInferAlgorithm::CudaCoreDecode, DType::F32),
        capability(FlashInferAlgorithm::CudaCoreDecode, DType::F16),
        capability(FlashInferAlgorithm::CudaCoreDecode, DType::Bf16),
        capability(FlashInferAlgorithm::TensorCore, DType::F16),
        capability(FlashInferAlgorithm::TensorCore, DType::Bf16),
    ],
};
