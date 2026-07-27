//! Logits-processing contracts and CPU reference implementations.

use half::{bf16, f16};

use crate::contract::{require_len, ContractError, DType};

/// Temperatures below this value retain logits unchanged for greedy rows.
pub const LOGITS_PREPROCESS_GREEDY_TEMPERATURE_THRESHOLD: f32 = 1.0e-5;

/// Contract for one fused F32 logits-preprocessing pass.
///
/// The pass applies a dense blocked-token mask, sparse additive biases, sparse
/// suppression, and per-row temperature in that order. A blocked-mask value of
/// one and every sparse suppression target become negative infinity. Bias
/// targets must be unique. Temperatures below
/// [`LOGITS_PREPROCESS_GREEDY_TEMPERATURE_THRESHOLD`] use a divisor of one,
/// matching mixed greedy/random sampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogitsPreprocessSpec {
    rows: usize,
    vocab_size: usize,
    has_blocked_mask: bool,
    bias_count: usize,
    suppression_count: usize,
}

impl LogitsPreprocessSpec {
    pub fn new(
        rows: usize,
        vocab_size: usize,
        has_blocked_mask: bool,
        bias_count: usize,
        suppression_count: usize,
    ) -> Result<Self, ContractError> {
        if rows == 0 || vocab_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        rows.checked_mul(vocab_size)
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self {
            rows,
            vocab_size,
            has_blocked_mask,
            bias_count,
            suppression_count,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn vocab_size(self) -> usize {
        self.vocab_size
    }

    pub const fn has_blocked_mask(self) -> bool {
        self.has_blocked_mask
    }

    pub const fn bias_count(self) -> usize {
        self.bias_count
    }

    pub const fn suppression_count(self) -> usize {
        self.suppression_count
    }

    pub const fn logits_numel(self) -> usize {
        self.rows * self.vocab_size
    }

    pub const fn blocked_mask_numel(self) -> usize {
        if self.has_blocked_mask {
            self.logits_numel()
        } else {
            0
        }
    }
}

/// Applies fused logits preprocessing to contiguous F32 rows.
#[allow(clippy::too_many_arguments)]
pub fn logits_preprocess_f32_reference(
    logits: &mut [f32],
    temperatures: &[f32],
    blocked_mask: Option<&[u8]>,
    bias_row_ids: &[i32],
    bias_token_ids: &[i32],
    bias_values: &[f32],
    suppressed_row_ids: &[i32],
    suppressed_token_ids: &[i32],
    spec: LogitsPreprocessSpec,
) -> Result<(), ContractError> {
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("temperatures", temperatures.len(), spec.rows())?;
    let blocked_mask = blocked_mask.unwrap_or_default();
    require_len(
        "blocked_mask",
        blocked_mask.len(),
        spec.blocked_mask_numel(),
    )?;
    require_len("bias_row_ids", bias_row_ids.len(), spec.bias_count())?;
    require_len("bias_token_ids", bias_token_ids.len(), spec.bias_count())?;
    require_len("bias_values", bias_values.len(), spec.bias_count())?;
    require_len(
        "suppressed_row_ids",
        suppressed_row_ids.len(),
        spec.suppression_count(),
    )?;
    require_len(
        "suppressed_token_ids",
        suppressed_token_ids.len(),
        spec.suppression_count(),
    )?;

    for (row, &temperature) in temperatures.iter().enumerate() {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(ContractError::InvalidTemperature {
                row,
                value: temperature,
            });
        }
    }
    for (index, &value) in blocked_mask.iter().enumerate() {
        if value > 1 {
            return Err(ContractError::InvalidMaskValue { index, value });
        }
    }
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() && value != f32::NEG_INFINITY {
            return Err(ContractError::InvalidLogit {
                row: index / spec.vocab_size(),
                column: index % spec.vocab_size(),
                value,
            });
        }
    }
    validate_sparse_targets(
        "logit bias",
        bias_row_ids,
        bias_token_ids,
        spec.rows(),
        spec.vocab_size(),
    )?;
    for (entry, &value) in bias_values.iter().enumerate() {
        if !value.is_finite() {
            return Err(ContractError::InvalidLogitBias { entry, value });
        }
        for first_entry in 0..entry {
            if bias_row_ids[first_entry] == bias_row_ids[entry]
                && bias_token_ids[first_entry] == bias_token_ids[entry]
            {
                return Err(ContractError::DuplicateLogitBias {
                    first_entry,
                    second_entry: entry,
                    row_id: bias_row_ids[entry],
                    token_id: bias_token_ids[entry],
                });
            }
        }
    }
    validate_sparse_targets(
        "suppression",
        suppressed_row_ids,
        suppressed_token_ids,
        spec.rows(),
        spec.vocab_size(),
    )?;

    if spec.has_blocked_mask() {
        for (logit, &blocked) in logits.iter_mut().zip(blocked_mask) {
            if blocked != 0 {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    for ((&row_id, &token_id), &bias) in bias_row_ids.iter().zip(bias_token_ids).zip(bias_values) {
        logits[row_id as usize * spec.vocab_size() + token_id as usize] += bias;
    }
    for (&row_id, &token_id) in suppressed_row_ids.iter().zip(suppressed_token_ids) {
        logits[row_id as usize * spec.vocab_size() + token_id as usize] = f32::NEG_INFINITY;
    }
    for (row, &temperature) in logits.chunks_exact_mut(spec.vocab_size()).zip(temperatures) {
        let divisor = if temperature < LOGITS_PREPROCESS_GREEDY_TEMPERATURE_THRESHOLD {
            1.0
        } else {
            temperature
        };
        for value in row {
            *value /= divisor;
        }
    }
    Ok(())
}

fn validate_sparse_targets(
    parameter: &'static str,
    row_ids: &[i32],
    token_ids: &[i32],
    rows: usize,
    vocab_size: usize,
) -> Result<(), ContractError> {
    for (entry, (&row_id, &token_id)) in row_ids.iter().zip(token_ids).enumerate() {
        if row_id < 0 || row_id as usize >= rows {
            return Err(ContractError::SparseRowOutOfBounds {
                parameter,
                entry,
                row_id,
                rows,
            });
        }
        if token_id < 0 || token_id as usize >= vocab_size {
            return Err(ContractError::SparseTokenOutOfBounds {
                parameter,
                entry,
                token_id,
                vocab_size,
            });
        }
    }
    Ok(())
}

/// Contract for in-place min-p filtering over rank-2 logits.
///
/// Each row keeps tokens whose probability is at least `min_p[row]` times
/// the row's maximum probability. The softmax denominator cancels, so
/// backends can apply the equivalent threshold
/// `logit >= max(logits) + log(min_p)` without materializing probabilities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MinPFilterSpec {
    rows: usize,
    vocab_size: usize,
    dtype: DType,
}

impl MinPFilterSpec {
    /// Creates a validated contiguous logits contract.
    pub fn new(rows: usize, vocab_size: usize, dtype: DType) -> Result<Self, ContractError> {
        if rows == 0 || vocab_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        rows.checked_mul(vocab_size)
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self {
            rows,
            vocab_size,
            dtype,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn vocab_size(self) -> usize {
        self.vocab_size
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn logits_numel(self) -> usize {
        self.rows * self.vocab_size
    }
}

/// Applies min-p filtering to F32 logits in place.
pub fn min_p_filter_f32_reference(
    logits: &mut [f32],
    min_p: &[f32],
    spec: MinPFilterSpec,
) -> Result<(), ContractError> {
    min_p_filter_reference(logits, min_p, spec, DType::F32)
}

/// Applies min-p filtering to FP16 logits in place.
pub fn min_p_filter_f16_reference(
    logits: &mut [f16],
    min_p: &[f32],
    spec: MinPFilterSpec,
) -> Result<(), ContractError> {
    min_p_filter_reference(logits, min_p, spec, DType::F16)
}

/// Applies min-p filtering to BF16 logits in place.
pub fn min_p_filter_bf16_reference(
    logits: &mut [bf16],
    min_p: &[f32],
    spec: MinPFilterSpec,
) -> Result<(), ContractError> {
    min_p_filter_reference(logits, min_p, spec, DType::Bf16)
}

trait MinPElement: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl MinPElement for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl MinPElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl MinPElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

fn min_p_filter_reference<T: MinPElement>(
    logits: &mut [T],
    min_p: &[f32],
    spec: MinPFilterSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("min_p", min_p.len(), spec.rows())?;
    for (row, &probability) in min_p.iter().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ContractError::InvalidProbability {
                parameter: "min_p",
                row,
                value: probability,
            });
        }
    }

    for (row, &probability) in logits.chunks_exact_mut(spec.vocab_size()).zip(min_p) {
        if probability == 0.0 {
            continue;
        }
        let maximum = row
            .iter()
            .map(|&value| value.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = maximum + probability.ln();
        for value in row {
            if value.to_f32() < threshold {
                *value = T::from_f32(f32::NEG_INFINITY);
            }
        }
    }
    Ok(())
}
