//! Numerical and storage contract shared by both activation quantization paths.
//!
//! These are ABI requirements, not model tuning knobs. Changing the block
//! geometry, scale layout, or numerical policy requires a new ABI identifier.
//! CUDA sources are rendered from these constants and enter provider identity.

use orbitkv_compiler::prelude::Expression;

use crate::resource::ResourceViolation;

pub(super) const PACKED_ACTIVATION_ABI: &str = "fp8-e4m3-row128-f32-kmajor-align4-packed-v1";
/// Both activation K tiles and checkpoint weight tiles use this granularity.
pub(super) use orbitkv_ops::ops::linear::FP8_SCALE_BLOCK as SCALE_BLOCK;
/// TMA requires a 16-byte stride for the F32 scale plane.
pub(super) const TMA_ALIGNMENT_BYTES: usize = 16;
pub(super) const BF16_BYTES: usize = std::mem::size_of::<half::bf16>();
pub(super) const SCALE_BYTES: usize = std::mem::size_of::<f32>();
pub(super) const SCALE_ROW_ALIGNMENT: usize = TMA_ALIGNMENT_BYTES / SCALE_BYTES;
/// One quantizer grid-y block is launched per row; CUDA grid-y is 16 bits.
pub(super) const MAX_QUANTIZER_ROWS: usize = u16::MAX as usize;
/// Finite E4M3 range and the provider's minimum per-block absolute maximum.
use orbitkv_ops::ops::linear::{FP8_MAX_FINITE, QUANTIZATION_AMAX_FLOOR};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedActivationLayout {
    pub scale_offset: usize,
    pub total_bytes: usize,
}

impl PackedActivationLayout {
    pub fn new(m: usize, k: usize) -> Result<Self, ResourceViolation> {
        if m > MAX_QUANTIZER_ROWS
            || k == 0
            || k > i32::MAX as usize
            || !k.is_multiple_of(SCALE_BLOCK)
        {
            return Err(ResourceViolation::HostResourcePlanning {
                name: "BlockScaledQuantize dimensions and launch bounds",
            });
        }
        let (quantized, scales) = scratch_bytes(m, k)?;
        let total_bytes =
            quantized
                .checked_add(scales)
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "BlockScaledQuantize packed activation",
                })?;
        Ok(Self {
            scale_offset: quantized,
            total_bytes,
        })
    }

    /// Symbolic counterpart of `new`, for the graph arena's byte accounting.
    pub fn output_bytes(rows: Expression, input_features: usize) -> Expression {
        rows * input_features
            + rows.ceil_div(SCALE_ROW_ALIGNMENT)
                * SCALE_ROW_ALIGNMENT
                * (input_features / SCALE_BLOCK)
                * SCALE_BYTES
    }

    pub fn scale_pointer(self, packed: u64) -> anyhow::Result<u64> {
        anyhow::ensure!(
            packed.is_multiple_of(TMA_ALIGNMENT_BYTES as u64),
            "packed activation must be TMA aligned"
        );
        packed
            .checked_add(self.scale_offset as u64)
            .ok_or_else(|| anyhow::anyhow!("packed activation pointer overflow"))
    }
}

/// Byte counts for the separate scratch planes used by the combined operator.
pub(super) fn scratch_bytes(m: usize, k: usize) -> Result<(usize, usize), ResourceViolation> {
    let quantized = m
        .checked_mul(k)
        .ok_or(ResourceViolation::ArithmeticOverflow {
            resource: "BlockScaledLinear activation scratch",
        })?;
    let scales = m
        .div_ceil(SCALE_ROW_ALIGNMENT)
        .checked_mul(SCALE_ROW_ALIGNMENT)
        .and_then(|rows| rows.checked_mul(k.div_ceil(SCALE_BLOCK)))
        .and_then(|elements| elements.checked_mul(SCALE_BYTES))
        .ok_or(ResourceViolation::ArithmeticOverflow {
            resource: "BlockScaledLinear scale scratch",
        })?;
    Ok((quantized, scales))
}

/// All consumers compile the same quantizer and the same layout definitions.
pub(super) fn quantizer_source() -> &'static str {
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SOURCE.get_or_init(|| {
        format!(
            include_str!("contract/quantizer.cu.in"),
            include_str!("quantize.cuh"),
            FP8_MAX_FINITE = FP8_MAX_FINITE,
            PACKED_ACTIVATION_ABI = PACKED_ACTIVATION_ABI,
            QUANTIZATION_AMAX_FLOOR = QUANTIZATION_AMAX_FLOOR,
            SCALE_BLOCK = SCALE_BLOCK,
            SCALE_ROW_ALIGNMENT = SCALE_ROW_ALIGNMENT,
        )
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/providers/deepgemm/contract/mod.rs"]
mod tests;
