//! Normalization contracts and CPU reference implementations.

use half::{bf16, f16};

use crate::contract::{require_len, ContractError, DType};
use crate::element::{DynamicFp8Input, LowPrecisionElement};
use crate::quantization::{fp8_e4m3fn_from_f32, DYNAMIC_FP8_MIN_SCALE, FP8_E4M3FN_MAX};

/// Contract for a two-dimensional RMSNorm operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormSpec {
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    dtype: DType,
}

/// Contract for RMSNorm followed by dynamic per-token FP8 quantization.
///
/// Inputs and weights are contiguous `[rows, hidden_size]` and
/// `[hidden_size]` tensors. The output contains FP8 E4M3FN storage bytes with
/// the same logical shape, and `rows` F32 scales satisfy approximately
/// `normalized = fp8(output) * scale`. With a mutable residual, the F32
/// `input + residual` sum drives RMSNorm and quantization while the mutable
/// residual output stores that same sum rounded to the input dtype.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormDynamicFp8Spec {
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    input_dtype: DType,
    output_dtype: DType,
}

/// Contract for RMSNorm followed by symmetric dynamic per-token INT8.
///
/// Inputs and weights are contiguous `[rows, hidden_size]` and
/// `[hidden_size]` tensors. The signed INT8 output has the same logical shape,
/// and `rows` F32 scales satisfy approximately
/// `normalized = int8(output) * scale`. An all-zero row has scale zero. With a
/// mutable residual, the F32 `input + residual` sum drives RMSNorm and
/// quantization while the residual output stores that sum rounded to the input
/// dtype.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormDynamicInt8Spec {
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    input_dtype: DType,
    output_dtype: DType,
}

impl RmsNormDynamicInt8Spec {
    /// Creates a validated shape and dtype contract.
    pub fn new(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        input_dtype: DType,
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
            input_dtype,
            output_dtype: DType::I8,
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

    pub const fn input_dtype(self) -> DType {
        self.input_dtype
    }

    pub const fn output_dtype(self) -> DType {
        self.output_dtype
    }

    pub const fn numel(self) -> usize {
        self.rows * self.hidden_size
    }

    pub const fn scale_count(self) -> usize {
        self.rows
    }
}

impl RmsNormDynamicFp8Spec {
    /// Creates a validated shape and dtype contract.
    pub fn new(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        input_dtype: DType,
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
            input_dtype,
            output_dtype: DType::Fp8E4M3Fn,
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

    pub const fn input_dtype(self) -> DType {
        self.input_dtype
    }

    pub const fn output_dtype(self) -> DType {
        self.output_dtype
    }

    pub const fn numel(self) -> usize {
        self.rows * self.hidden_size
    }

    pub const fn scale_count(self) -> usize {
        self.rows
    }
}

/// Contract for fused residual addition followed by RMSNorm.
///
/// Backends implementing this contract update both operands in place:
/// `residual = input + residual`, then
/// `input = RMSNorm(residual, weight, epsilon)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AddRmsNormSpec {
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    dtype: DType,
}

impl AddRmsNormSpec {
    /// Creates a validated fused Add+RMSNorm contract.
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

impl RmsNormSpec {
    /// Creates a validated RMSNorm contract.
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

/// Computes an F32 RMSNorm reference with F64 accumulation.
pub fn rms_norm_f32_reference(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F32 {
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
                let value = f64::from(value);
                value * value
            })
            .sum::<f64>()
            / spec.hidden_size() as f64;
        let inverse_rms = 1.0 / (mean_square + f64::from(spec.epsilon())).sqrt();

        for ((destination, &value), &scale) in output_row.iter_mut().zip(input_row).zip(weight) {
            *destination = (f64::from(value) * inverse_rms * f64::from(scale)) as f32;
        }
    }

    Ok(())
}

/// Computes an F16 RMSNorm reference with F64 accumulation over quantized inputs.
pub fn rms_norm_f16_reference(
    input: &[f16],
    weight: &[f16],
    output: &mut [f16],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    rms_norm_low_precision_reference(input, weight, output, spec, DType::F16)
}

/// Computes a BF16 RMSNorm reference with F64 accumulation over quantized inputs.
pub fn rms_norm_bf16_reference(
    input: &[bf16],
    weight: &[bf16],
    output: &mut [bf16],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    rms_norm_low_precision_reference(input, weight, output, spec, DType::Bf16)
}

/// Computes F32 RMSNorm followed by dynamic per-token FP8 E4M3FN quantization.
pub fn rms_norm_dynamic_fp8_f32_reference(
    input: &[f32],
    weight: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    residual: Option<&mut [f32]>,
    spec: RmsNormDynamicFp8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_fp8_reference(input, weight, output, scales, residual, spec, DType::F32)
}

/// Computes FP16 RMSNorm followed by dynamic per-token FP8 E4M3FN quantization.
pub fn rms_norm_dynamic_fp8_f16_reference(
    input: &[f16],
    weight: &[f16],
    output: &mut [u8],
    scales: &mut [f32],
    residual: Option<&mut [f16]>,
    spec: RmsNormDynamicFp8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_fp8_reference(input, weight, output, scales, residual, spec, DType::F16)
}

/// Computes BF16 RMSNorm followed by dynamic per-token FP8 E4M3FN quantization.
pub fn rms_norm_dynamic_fp8_bf16_reference(
    input: &[bf16],
    weight: &[bf16],
    output: &mut [u8],
    scales: &mut [f32],
    residual: Option<&mut [bf16]>,
    spec: RmsNormDynamicFp8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_fp8_reference(input, weight, output, scales, residual, spec, DType::Bf16)
}

/// Computes F32 RMSNorm followed by symmetric dynamic per-token INT8.
pub fn rms_norm_dynamic_int8_f32_reference(
    input: &[f32],
    weight: &[f32],
    output: &mut [i8],
    scales: &mut [f32],
    residual: Option<&mut [f32]>,
    spec: RmsNormDynamicInt8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_int8_reference(input, weight, output, scales, residual, spec, DType::F32)
}

/// Computes FP16 RMSNorm followed by symmetric dynamic per-token INT8.
pub fn rms_norm_dynamic_int8_f16_reference(
    input: &[f16],
    weight: &[f16],
    output: &mut [i8],
    scales: &mut [f32],
    residual: Option<&mut [f16]>,
    spec: RmsNormDynamicInt8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_int8_reference(input, weight, output, scales, residual, spec, DType::F16)
}

/// Computes BF16 RMSNorm followed by symmetric dynamic per-token INT8.
pub fn rms_norm_dynamic_int8_bf16_reference(
    input: &[bf16],
    weight: &[bf16],
    output: &mut [i8],
    scales: &mut [f32],
    residual: Option<&mut [bf16]>,
    spec: RmsNormDynamicInt8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_int8_reference(input, weight, output, scales, residual, spec, DType::Bf16)
}

/// Computes fused in-place F32 Add+RMSNorm with F64 accumulation.
///
/// On success `residual` contains the elementwise sum and `input` contains its
/// normalized, weighted value. The two mutable slices must not alias.
pub fn add_rms_norm_f32_reference(
    input: &mut [f32],
    residual: &mut [f32],
    weight: &[f32],
    spec: AddRmsNormSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F32 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("residual", residual.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;

    for (input_row, residual_row) in input
        .chunks_exact_mut(spec.hidden_size())
        .zip(residual.chunks_exact_mut(spec.hidden_size()))
    {
        let mut square_sum = 0.0_f64;
        for (input_value, residual_value) in input_row.iter().zip(residual_row.iter_mut()) {
            let sum = *input_value + *residual_value;
            *residual_value = sum;
            let sum = f64::from(sum);
            square_sum += sum * sum;
        }

        let mean_square = square_sum / spec.hidden_size() as f64;
        let inverse_rms = 1.0 / (mean_square + f64::from(spec.epsilon())).sqrt();
        for ((destination, &sum), &scale) in
            input_row.iter_mut().zip(residual_row.iter()).zip(weight)
        {
            *destination = (f64::from(sum) * inverse_rms * f64::from(scale)) as f32;
        }
    }

    Ok(())
}

/// Computes fused in-place FP16 Add+RMSNorm.
///
/// The elementwise sum is rounded to FP16 before the RMS statistic is
/// accumulated, matching a materialized FP16 residual tensor.
pub fn add_rms_norm_f16_reference(
    input: &mut [f16],
    residual: &mut [f16],
    weight: &[f16],
    spec: AddRmsNormSpec,
) -> Result<(), ContractError> {
    add_rms_norm_low_precision_reference(input, residual, weight, spec, DType::F16)
}

/// Computes fused in-place BF16 Add+RMSNorm.
///
/// The elementwise sum is rounded to BF16 before the RMS statistic is
/// accumulated, matching a materialized BF16 residual tensor.
pub fn add_rms_norm_bf16_reference(
    input: &mut [bf16],
    residual: &mut [bf16],
    weight: &[bf16],
    spec: AddRmsNormSpec,
) -> Result<(), ContractError> {
    add_rms_norm_low_precision_reference(input, residual, weight, spec, DType::Bf16)
}

fn rms_norm_low_precision_reference<T: LowPrecisionElement>(
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

fn rms_norm_dynamic_fp8_reference<T: DynamicFp8Input>(
    input: &[T],
    weight: &[T],
    output: &mut [u8],
    scales: &mut [f32],
    mut residual: Option<&mut [T]>,
    spec: RmsNormDynamicFp8Spec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.input_dtype() != expected_dtype || spec.output_dtype() != DType::Fp8E4M3Fn {
        return Err(ContractError::UnsupportedDType(spec.input_dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;
    require_len("output", output.len(), spec.numel())?;
    require_len("scales", scales.len(), spec.scale_count())?;
    if let Some(values) = residual.as_deref() {
        require_len("residual", values.len(), spec.numel())?;
    }

    let mut row_values = vec![0.0_f32; spec.hidden_size()];
    let mut normalized = vec![0.0_f32; spec.hidden_size()];
    for (row, scale) in scales.iter_mut().enumerate().take(spec.rows()) {
        let row_start = row * spec.hidden_size();
        let row_end = row_start + spec.hidden_size();
        let input_row = &input[row_start..row_end];
        let output_row = &mut output[row_start..row_end];

        let mut square_sum = 0.0_f64;
        if let Some(residual_values) = residual.as_deref_mut() {
            let residual_row = &mut residual_values[row_start..row_end];
            for (column, (&input_value, residual_value)) in
                input_row.iter().zip(residual_row.iter_mut()).enumerate()
            {
                let sum = input_value.to_f32() + residual_value.to_f32();
                let stored_sum = T::from_f32(sum);
                *residual_value = stored_sum;
                row_values[column] = sum;
                square_sum += f64::from(sum) * f64::from(sum);
            }
        } else {
            for (destination, &value) in row_values.iter_mut().zip(input_row) {
                *destination = value.to_f32();
                square_sum += f64::from(*destination) * f64::from(*destination);
            }
        }

        let mean_square = square_sum / spec.hidden_size() as f64;
        let inverse_rms = (1.0 / (mean_square + f64::from(spec.epsilon())).sqrt()) as f32;

        let mut absolute_maximum = 0.0_f32;
        for (column, (&value, &weight_value)) in row_values.iter().zip(weight).enumerate() {
            let rounded_normalized = T::round_to_storage(value * inverse_rms);
            let weighted = T::round_to_storage(rounded_normalized * weight_value.to_f32());
            normalized[column] = weighted;
            absolute_maximum = absolute_maximum.max(weighted.abs());
        }

        *scale = (absolute_maximum / FP8_E4M3FN_MAX).max(DYNAMIC_FP8_MIN_SCALE);
        for (destination, &value) in output_row.iter_mut().zip(&normalized) {
            *destination = fp8_e4m3fn_from_f32(value / *scale);
        }
    }

    Ok(())
}

fn rms_norm_dynamic_int8_reference<T: DynamicFp8Input>(
    input: &[T],
    weight: &[T],
    output: &mut [i8],
    scales: &mut [f32],
    mut residual: Option<&mut [T]>,
    spec: RmsNormDynamicInt8Spec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.input_dtype() != expected_dtype || spec.output_dtype() != DType::I8 {
        return Err(ContractError::UnsupportedDType(spec.input_dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;
    require_len("output", output.len(), spec.numel())?;
    require_len("scales", scales.len(), spec.scale_count())?;
    if let Some(values) = residual.as_deref() {
        require_len("residual", values.len(), spec.numel())?;
    }

    let mut row_values = vec![0.0_f32; spec.hidden_size()];
    let mut normalized = vec![0.0_f32; spec.hidden_size()];
    for (row, scale) in scales.iter_mut().enumerate().take(spec.rows()) {
        let row_start = row * spec.hidden_size();
        let row_end = row_start + spec.hidden_size();
        let input_row = &input[row_start..row_end];
        let output_row = &mut output[row_start..row_end];

        let mut square_sum = 0.0_f64;
        if let Some(residual_values) = residual.as_deref_mut() {
            let residual_row = &mut residual_values[row_start..row_end];
            for (column, (&input_value, residual_value)) in
                input_row.iter().zip(residual_row.iter_mut()).enumerate()
            {
                let sum = input_value.to_f32() + residual_value.to_f32();
                let stored_sum = T::from_f32(sum);
                *residual_value = stored_sum;
                row_values[column] = sum;
                square_sum += f64::from(row_values[column]) * f64::from(row_values[column]);
            }
        } else {
            for (destination, &value) in row_values.iter_mut().zip(input_row) {
                *destination = value.to_f32();
                square_sum += f64::from(*destination) * f64::from(*destination);
            }
        }

        let mean_square = square_sum / spec.hidden_size() as f64;
        let inverse_rms = (1.0 / (mean_square + f64::from(spec.epsilon())).sqrt()) as f32;

        let mut absolute_maximum = 0.0_f32;
        for (column, (&value, &weight_value)) in row_values.iter().zip(weight).enumerate() {
            // Match vLLM's native IR graph: normalization is materialized in
            // the weight dtype before the weight multiplication.
            let rounded_normalized = T::round_to_storage(value * inverse_rms);
            let weighted = T::round_to_storage(rounded_normalized * weight_value.to_f32());
            normalized[column] = weighted;
            absolute_maximum = absolute_maximum.max(weighted.abs());
        }

        *scale = crate::quantization::dynamic_int8_scale(absolute_maximum);
        if *scale == 0.0 {
            output_row.fill(0);
        } else {
            for (destination, &value) in output_row.iter_mut().zip(&normalized) {
                *destination = crate::quantization::quantize_dynamic_int8(value, absolute_maximum);
            }
        }
    }

    Ok(())
}

fn add_rms_norm_low_precision_reference<T: LowPrecisionElement>(
    input: &mut [T],
    residual: &mut [T],
    weight: &[T],
    spec: AddRmsNormSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("residual", residual.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;

    for (input_row, residual_row) in input
        .chunks_exact_mut(spec.hidden_size())
        .zip(residual.chunks_exact_mut(spec.hidden_size()))
    {
        let mut square_sum = 0.0_f64;
        for (input_value, residual_value) in input_row.iter().zip(residual_row.iter_mut()) {
            let quantized_sum = T::from_f32(input_value.to_f32() + residual_value.to_f32());
            *residual_value = quantized_sum;
            let sum = f64::from(quantized_sum.to_f32());
            square_sum += sum * sum;
        }

        let mean_square = square_sum / spec.hidden_size() as f64;
        let inverse_rms = 1.0 / (mean_square + f64::from(spec.epsilon())).sqrt();
        for ((destination, &sum), &scale) in
            input_row.iter_mut().zip(residual_row.iter()).zip(weight)
        {
            let normalized = f64::from(sum.to_f32()) * inverse_rms * f64::from(scale.to_f32());
            *destination = T::from_f32(normalized as f32);
        }
    }

    Ok(())
}
