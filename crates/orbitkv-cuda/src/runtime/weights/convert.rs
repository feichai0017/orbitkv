//! Explicit storage and floating-point conversion contracts. Byte views do not
//! require alignment, and typed allocations keep their original drop layout.

use anyhow::{Result, bail, ensure};
use half::{bf16, f16};
use orbitkv_compiler::dtype::DType;
use safetensors::Dtype;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Conversion {
    Identity,
    F16ToF32,
    Bf16ToF32,
    F32ToF16,
    Bf16ToF16,
    F32ToBf16,
    F16ToBf16,
}

pub(super) enum WeightData<'a> {
    Borrowed(&'a [u8]),
    F32(Vec<f32>),
    F16(Vec<f16>),
    Bf16(Vec<bf16>),
}

impl WeightData<'_> {
    pub(super) fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::F32(values) => bytemuck::cast_slice(values),
            Self::F16(values) => bytemuck::cast_slice(values),
            Self::Bf16(values) => bytemuck::cast_slice(values),
        }
    }
}

impl Conversion {
    pub(super) fn for_dtypes(source: Dtype, target: DType) -> Result<Self> {
        Ok(match (source, target) {
            (Dtype::F32, DType::F32) | (Dtype::BF16, DType::Bf16)
            | (Dtype::F16, DType::F16) | (Dtype::U8, DType::U8)
            // Checkpoints can store raw E8M0 scale encodings in U8 storage.
            | (Dtype::U8, DType::F8UE8M0)
            | (Dtype::F8_E4M3, DType::F8E4M3)
            | (Dtype::F8_E5M2, DType::F8E5M2)
            | (Dtype::F8_E8M0, DType::F8UE8M0) => Self::Identity,
            (Dtype::F16, DType::F32) => Self::F16ToF32,
            (Dtype::BF16, DType::F32) => Self::Bf16ToF32,
            (Dtype::F32, DType::F16) => Self::F32ToF16,
            (Dtype::BF16, DType::F16) => Self::Bf16ToF16,
            (Dtype::F32, DType::Bf16) => Self::F32ToBf16,
            (Dtype::F16, DType::Bf16) => Self::F16ToBf16,
            _ => bail!("unsupported weight conversion {source:?} -> {target:?}"),
        })
    }

    pub(super) fn is_conversion(self) -> bool {
        self != Self::Identity
    }

    pub(super) fn apply(self, bytes: &[u8]) -> Result<WeightData<'_>> {
        Ok(match self {
            Self::Identity => WeightData::Borrowed(bytes),
            Self::F16ToF32 => {
                WeightData::F32(convert(bytes, |bits| f16::from_le_bytes(bits).to_f32())?)
            }
            Self::Bf16ToF32 => {
                WeightData::F32(convert(bytes, |bits| bf16::from_le_bytes(bits).to_f32())?)
            }
            Self::F32ToF16 => WeightData::F16(convert(bytes, |bits| {
                f16::from_f32(f32::from_le_bytes(bits))
            })?),
            Self::Bf16ToF16 => WeightData::F16(convert(bytes, |bits| {
                f16::from_f32(bf16::from_le_bytes(bits).to_f32())
            })?),
            Self::F32ToBf16 => WeightData::Bf16(convert(bytes, |bits| {
                bf16::from_f32(f32::from_le_bytes(bits))
            })?),
            Self::F16ToBf16 => WeightData::Bf16(convert(bytes, |bits| {
                bf16::from_f32(f16::from_le_bytes(bits).to_f32())
            })?),
        })
    }
}

fn convert<const N: usize, T>(bytes: &[u8], convert: impl Fn([u8; N]) -> T) -> Result<Vec<T>> {
    let (chunks, remainder) = bytes.as_chunks::<N>();
    ensure!(
        remainder.is_empty(),
        "weight byte count is not divisible by source element size {N}"
    );
    Ok(chunks.iter().copied().map(convert).collect())
}

#[cfg(test)]
#[path = "../../../tests/unit/runtime/weights/convert/mod.rs"]
mod tests;
