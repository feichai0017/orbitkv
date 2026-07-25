//! Token-selection and logprob contracts with CPU references.

use half::{bf16, f16};

use crate::contract::{require_len, ContractError, DType};

/// Contract for fused greedy token selection and its normalized logprob.
///
/// Logits are contiguous `[rows, vocab_size]`. Each output row contains the
/// lowest token index attaining the maximum logit, its log-softmax value, and
/// an integration-defined sampled-token rank. The CUDA and Python adapters
/// match vLLM's tie-aware rank by counting all logits equal to the maximum.
/// This deterministic boundary is useful for greedy decode requests that ask
/// only for the sampled token's logprob.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GreedySampleLogprobsSpec {
    rows: usize,
    vocab_size: usize,
    dtype: DType,
}

/// Contract for normalizing and ranking one caller-selected token per row.
///
/// Logits are contiguous `[rows, vocab_size]`; token IDs are one int64 value
/// per row and must be in `[0, vocab_size)`. Outputs are F32 logprobs and
/// int64 ranks. Rank uses vLLM's tie-aware definition: the number of logits
/// greater than or equal to the selected logit. This boundary lets an engine
/// keep its own greedy, top-k/top-p, penalty, and random-sampling policy while
/// avoiding a materialized full-vocabulary F32 log-softmax tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectedTokenLogprobsSpec {
    rows: usize,
    vocab_size: usize,
    dtype: DType,
}

/// Contract for sparse repetition, frequency, and presence penalties.
///
/// Logits are F32 `[rows, vocab_size]`. Prompt and output token IDs are
/// row-major int64 matrices whose negative or out-of-vocabulary values are
/// padding. Repetition penalties apply once to the union of prompt and output
/// IDs; frequency and presence penalties use output IDs only. The CUDA backend
/// uses one caller-owned int64 hash workspace per row instead of materializing
/// full-vocabulary count and mask tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenPenaltiesSpec {
    rows: usize,
    vocab_size: usize,
    prompt_tokens: usize,
    output_tokens: usize,
    workspace_capacity: usize,
}

impl GreedySampleLogprobsSpec {
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

impl SelectedTokenLogprobsSpec {
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

impl TokenPenaltiesSpec {
    /// Creates a validated sparse-history penalty contract.
    pub fn new(
        rows: usize,
        vocab_size: usize,
        prompt_tokens: usize,
        output_tokens: usize,
        workspace_capacity: usize,
    ) -> Result<Self, ContractError> {
        if rows == 0
            || vocab_size == 0
            || prompt_tokens == 0
            || output_tokens == 0
            || workspace_capacity == 0
        {
            return Err(ContractError::ZeroDimension);
        }
        rows.checked_mul(vocab_size)
            .and_then(|_| rows.checked_mul(prompt_tokens))
            .and_then(|_| rows.checked_mul(output_tokens))
            .and_then(|_| rows.checked_mul(workspace_capacity))
            .ok_or(ContractError::ElementCountOverflow)?;
        let required = Self::required_workspace_capacity(prompt_tokens, output_tokens)?;
        if workspace_capacity < required || !workspace_capacity.is_power_of_two() {
            return Err(ContractError::TokenPenaltyWorkspaceTooSmall {
                required,
                actual: workspace_capacity,
            });
        }
        Ok(Self {
            rows,
            vocab_size,
            prompt_tokens,
            output_tokens,
            workspace_capacity,
        })
    }

    /// Returns the power-of-two slots needed for a maximum 0.5 hash load.
    pub fn required_workspace_capacity(
        prompt_tokens: usize,
        output_tokens: usize,
    ) -> Result<usize, ContractError> {
        let history_tokens = prompt_tokens
            .checked_add(output_tokens)
            .ok_or(ContractError::ElementCountOverflow)?;
        if history_tokens == 0 {
            return Err(ContractError::ZeroDimension);
        }
        history_tokens
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or(ContractError::ElementCountOverflow)
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn vocab_size(self) -> usize {
        self.vocab_size
    }

    pub const fn prompt_tokens(self) -> usize {
        self.prompt_tokens
    }

    pub const fn output_tokens(self) -> usize {
        self.output_tokens
    }

    pub const fn workspace_capacity(self) -> usize {
        self.workspace_capacity
    }

    pub const fn logits_numel(self) -> usize {
        self.rows * self.vocab_size
    }

    pub const fn prompt_token_numel(self) -> usize {
        self.rows * self.prompt_tokens
    }

    pub const fn output_token_numel(self) -> usize {
        self.rows * self.output_tokens
    }

    pub const fn workspace_numel(self) -> usize {
        self.rows * self.workspace_capacity
    }
}

/// Selects the first maximum F32 logit per row and returns its log-softmax.
pub fn greedy_sample_logprobs_f32_reference(
    logits: &[f32],
    token_ids: &mut [u32],
    logprobs: &mut [f32],
    spec: GreedySampleLogprobsSpec,
) -> Result<(), ContractError> {
    greedy_sample_logprobs_reference(logits, token_ids, logprobs, spec, DType::F32)
}

/// Selects the first maximum FP16 logit per row and returns its F32 log-softmax.
pub fn greedy_sample_logprobs_f16_reference(
    logits: &[f16],
    token_ids: &mut [u32],
    logprobs: &mut [f32],
    spec: GreedySampleLogprobsSpec,
) -> Result<(), ContractError> {
    greedy_sample_logprobs_reference(logits, token_ids, logprobs, spec, DType::F16)
}

/// Selects the first maximum BF16 logit per row and returns its F32 log-softmax.
pub fn greedy_sample_logprobs_bf16_reference(
    logits: &[bf16],
    token_ids: &mut [u32],
    logprobs: &mut [f32],
    spec: GreedySampleLogprobsSpec,
) -> Result<(), ContractError> {
    greedy_sample_logprobs_reference(logits, token_ids, logprobs, spec, DType::Bf16)
}

/// Returns F32 logprobs and tie-aware ranks for caller-selected F32 tokens.
pub fn selected_token_logprobs_f32_reference(
    logits: &[f32],
    token_ids: &[i64],
    logprobs: &mut [f32],
    ranks: &mut [i64],
    spec: SelectedTokenLogprobsSpec,
) -> Result<(), ContractError> {
    selected_token_logprobs_reference(logits, token_ids, logprobs, ranks, spec, DType::F32)
}

/// Returns F32 logprobs and tie-aware ranks for caller-selected FP16 tokens.
pub fn selected_token_logprobs_f16_reference(
    logits: &[f16],
    token_ids: &[i64],
    logprobs: &mut [f32],
    ranks: &mut [i64],
    spec: SelectedTokenLogprobsSpec,
) -> Result<(), ContractError> {
    selected_token_logprobs_reference(logits, token_ids, logprobs, ranks, spec, DType::F16)
}

/// Returns F32 logprobs and tie-aware ranks for caller-selected BF16 tokens.
pub fn selected_token_logprobs_bf16_reference(
    logits: &[bf16],
    token_ids: &[i64],
    logprobs: &mut [f32],
    ranks: &mut [i64],
    spec: SelectedTokenLogprobsSpec,
) -> Result<(), ContractError> {
    selected_token_logprobs_reference(logits, token_ids, logprobs, ranks, spec, DType::Bf16)
}

/// Applies vLLM-compatible repetition, frequency, and presence penalties.
pub fn apply_token_penalties_f32_reference(
    logits: &mut [f32],
    prompt_token_ids: &[i64],
    output_token_ids: &[i64],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    repetition_penalties: &[f32],
    spec: TokenPenaltiesSpec,
) -> Result<(), ContractError> {
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len(
        "prompt_token_ids",
        prompt_token_ids.len(),
        spec.prompt_token_numel(),
    )?;
    require_len(
        "output_token_ids",
        output_token_ids.len(),
        spec.output_token_numel(),
    )?;
    require_len("presence_penalties", presence_penalties.len(), spec.rows())?;
    require_len(
        "frequency_penalties",
        frequency_penalties.len(),
        spec.rows(),
    )?;
    require_len(
        "repetition_penalties",
        repetition_penalties.len(),
        spec.rows(),
    )?;

    let mut prompt_mask = vec![false; spec.vocab_size()];
    let mut output_counts = vec![0_u32; spec.vocab_size()];
    for row in 0..spec.rows() {
        prompt_mask.fill(false);
        output_counts.fill(0);
        let prompt_start = row * spec.prompt_tokens();
        for &token_id in &prompt_token_ids[prompt_start..prompt_start + spec.prompt_tokens()] {
            if let Ok(token) = usize::try_from(token_id) {
                if token < spec.vocab_size() {
                    prompt_mask[token] = true;
                }
            }
        }
        let output_start = row * spec.output_tokens();
        for &token_id in &output_token_ids[output_start..output_start + spec.output_tokens()] {
            if let Ok(token) = usize::try_from(token_id) {
                if token < spec.vocab_size() {
                    output_counts[token] = output_counts[token].saturating_add(1);
                }
            }
        }

        let logits_start = row * spec.vocab_size();
        for token in 0..spec.vocab_size() {
            let output_count = output_counts[token];
            if !prompt_mask[token] && output_count == 0 {
                continue;
            }
            let value = &mut logits[logits_start + token];
            let repetition = repetition_penalties[row];
            *value = if *value > 0.0 {
                *value / repetition
            } else {
                *value * repetition
            };
            if output_count != 0 {
                *value -= frequency_penalties[row] * output_count as f32;
                *value -= presence_penalties[row];
            }
        }
    }
    Ok(())
}

trait LogitElement: Copy {
    fn to_f32(self) -> f32;
}

impl LogitElement for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl LogitElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl LogitElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

fn greedy_sample_logprobs_reference<T: LogitElement>(
    logits: &[T],
    token_ids: &mut [u32],
    logprobs: &mut [f32],
    spec: GreedySampleLogprobsSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("token_ids", token_ids.len(), spec.rows())?;
    require_len("logprobs", logprobs.len(), spec.rows())?;

    for ((row, token_id), logprob) in logits
        .chunks_exact(spec.vocab_size())
        .zip(token_ids.iter_mut())
        .zip(logprobs.iter_mut())
    {
        let mut maximum = row[0].to_f32();
        let mut maximum_index = 0_usize;
        for (index, &value) in row.iter().enumerate().skip(1) {
            let value = value.to_f32();
            if value > maximum {
                maximum = value;
                maximum_index = index;
            }
        }

        let exponential_sum = row
            .iter()
            .map(|&value| f64::from(value.to_f32() - maximum).exp())
            .sum::<f64>();
        *token_id = maximum_index as u32;
        *logprob = -(exponential_sum.ln() as f32);
    }
    Ok(())
}

fn selected_token_logprobs_reference<T: LogitElement>(
    logits: &[T],
    token_ids: &[i64],
    logprobs: &mut [f32],
    ranks: &mut [i64],
    spec: SelectedTokenLogprobsSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("token_ids", token_ids.len(), spec.rows())?;
    require_len("logprobs", logprobs.len(), spec.rows())?;
    require_len("ranks", ranks.len(), spec.rows())?;

    for (row_index, (((row, &token_id), logprob), rank)) in logits
        .chunks_exact(spec.vocab_size())
        .zip(token_ids.iter())
        .zip(logprobs.iter_mut())
        .zip(ranks.iter_mut())
        .enumerate()
    {
        let selected_index =
            usize::try_from(token_id).map_err(|_| ContractError::TokenIdOutOfBounds {
                row: row_index,
                token_id,
                vocab_size: spec.vocab_size(),
            })?;
        if selected_index >= spec.vocab_size() {
            return Err(ContractError::TokenIdOutOfBounds {
                row: row_index,
                token_id,
                vocab_size: spec.vocab_size(),
            });
        }

        let selected = row[selected_index].to_f32();
        let maximum = row
            .iter()
            .map(|&value| value.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let exponential_sum = row
            .iter()
            .map(|&value| f64::from(value.to_f32() - maximum).exp())
            .sum::<f64>();
        *logprob = selected - maximum - exponential_sum.ln() as f32;
        *rank = row
            .iter()
            .filter(|&&value| value.to_f32() >= selected)
            .count() as i64;
    }
    Ok(())
}
