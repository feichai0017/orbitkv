use super::*;
use crate::providers::flashinfer;
use crate::providers::flashinfer::FlashInferAlgorithm;

mod algorithms;
pub(crate) mod reference;
mod search;

fn spec(dtype: DType, query_key_dim: usize, value_dim: usize) -> AttentionSpec {
    AttentionSpec {
        query_heads: 6,
        kv_heads: 2,
        query_key_dim,
        value_dim,
        dtype,
        scale: 0.125,
        mask: AttentionMask::Causal,
    }
}

#[test]
fn flashinfer_capabilities_distinguish_dtype_geometry_and_phase() {
    let accepts = |dtype, qk, value, prefill| {
        flashinfer::CAPABILITIES.supports_geometry(
            if prefill {
                FlashInferAlgorithm::TensorCore
            } else {
                FlashInferAlgorithm::CudaCoreDecode
            }
            .as_str(),
            spec(dtype, qk, value),
            PagedKvLayout::TokenMajor,
            16,
            prefill,
        )
    };
    assert!(accepts(DType::F16, 64, 64, true));
    assert!(accepts(DType::Bf16, 512, 512, true));
    assert!(accepts(DType::F32, 256, 256, false));
    assert!(!accepts(DType::F32, 256, 256, true));
    assert!(!accepts(DType::F32, 512, 512, false));
    assert!(!accepts(DType::Bf16, 256, 128, false));
    assert!(!accepts(DType::F64, 64, 64, false));
    assert!(!accepts(DType::Bf16, 96, 96, false));
    assert!(!flashinfer::CAPABILITIES.supports_geometry(
        FlashInferAlgorithm::CudaCoreDecode.as_str(),
        spec(DType::Bf16, 128, 128),
        PagedKvLayout::TokenMajor,
        16,
        true,
    ));
    assert!(!flashinfer::CAPABILITIES.supports_geometry(
        FlashInferAlgorithm::TensorCore.as_str(),
        spec(DType::F32, 128, 128),
        PagedKvLayout::TokenMajor,
        16,
        false,
    ));
}

#[test]
fn flashattention_capabilities_reject_unsupported_targets_and_abis() {
    let provider = crate::providers::flashattention::CAPABILITIES;
    assert!(provider.supports_target(9));
    for target in [8, 10, 12] {
        assert!(!provider.supports_target(target));
    }
    let accepts = |dtype, dim, layout| {
        provider.supports_geometry("fa3-paged", spec(dtype, dim, dim), layout, 16, true)
    };
    assert!(accepts(DType::Bf16, 256, PagedKvLayout::TokenMajor));
    assert!(!accepts(DType::F32, 128, PagedKvLayout::TokenMajor));
    assert!(!accepts(DType::F16, 512, PagedKvLayout::TokenMajor));
    assert!(!accepts(DType::F16, 128, PagedKvLayout::HeadMajor));
}

#[test]
fn views_visibility_and_page_abi_are_part_of_provider_admission() {
    let geometry = spec(DType::Bf16, 256, 256);
    let accepts = |geometry, layout, pages| {
        flashinfer::CAPABILITIES.supports_geometry(
            FlashInferAlgorithm::CudaCoreDecode.as_str(),
            geometry,
            layout,
            pages,
            false,
        )
    };
    assert!(!accepts(geometry, PagedKvLayout::HeadMajor, 16));
    assert!(!accepts(geometry, PagedKvLayout::TokenMajor, 0));
    assert!(!accepts(
        geometry,
        PagedKvLayout::TokenMajor,
        i32::MAX as usize + 1
    ));
    assert!(!accepts(
        AttentionSpec {
            mask: AttentionMask::Unmasked,
            ..geometry
        },
        PagedKvLayout::TokenMajor,
        16
    ));
    assert!(accepts(
        AttentionSpec {
            mask: AttentionMask::Sliding { window_left: 0 },
            ..geometry
        },
        PagedKvLayout::TokenMajor,
        16
    ));
    assert!(!accepts(
        AttentionSpec {
            mask: AttentionMask::Sliding {
                window_left: i32::MAX as usize + 1
            },
            ..geometry
        },
        PagedKvLayout::TokenMajor,
        16
    ));
}
