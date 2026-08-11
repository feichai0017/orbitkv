//! RMSNorm contract and CPU reference implementations.

use crate::error::require_len;
use crate::{ContractError, DType};
use half::{bf16, f16};

/// Contract for contiguous two-dimensional RMSNorm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormSpec {
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    dtype: DType,
}

impl RmsNormSpec {
    pub fn new(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        dtype: DType,
    ) -> Result<Self, ContractError> {
        if rows == 0 || hidden_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ContractError::InvalidEpsilon(epsilon));
        }
        rows.checked_mul(hidden_size)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            rows,
            hidden_size,
            epsilon,
            dtype,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub const fn epsilon(self) -> f32 {
        self.epsilon
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn numel(self) -> usize {
        self.rows * self.hidden_size
    }
}

/// Computes F32 RMSNorm with F64 accumulation.
pub fn rms_norm_f32_reference(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    rms_norm_reference(input, weight, output, spec, DType::F32)
}

/// Computes FP16 RMSNorm with F64 accumulation over quantized inputs.
pub fn rms_norm_f16_reference(
    input: &[f16],
    weight: &[f16],
    output: &mut [f16],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    rms_norm_reference(input, weight, output, spec, DType::F16)
}

/// Computes BF16 RMSNorm with F64 accumulation over quantized inputs.
pub fn rms_norm_bf16_reference(
    input: &[bf16],
    weight: &[bf16],
    output: &mut [bf16],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    rms_norm_reference(input, weight, output, spec, DType::Bf16)
}

trait ReferenceElement: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl ReferenceElement for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl ReferenceElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl ReferenceElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

fn rms_norm_reference<T: ReferenceElement>(
    input: &[T],
    weight: &[T],
    output: &mut [T],
    spec: RmsNormSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;
    require_len("output", output.len(), spec.numel())?;

    for (input_row, output_row) in input
        .chunks_exact(spec.hidden_size())
        .zip(output.chunks_exact_mut(spec.hidden_size()))
    {
        let mean_square = input_row
            .iter()
            .map(|&value| {
                let value = f64::from(value.to_f32());
                value * value
            })
            .sum::<f64>()
            / spec.hidden_size() as f64;
        let inverse_rms = 1.0 / (mean_square + f64::from(spec.epsilon())).sqrt();

        for ((destination, &value), &scale) in output_row.iter_mut().zip(input_row).zip(weight) {
            let normalized = f64::from(value.to_f32()) * inverse_rms * f64::from(scale.to_f32());
            *destination = T::from_f32(normalized as f32);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
