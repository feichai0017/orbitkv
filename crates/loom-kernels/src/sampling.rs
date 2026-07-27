//! Token-selection and logprob contracts with CPU references.

use half::{bf16, f16};
use std::mem::size_of;

use crate::contract::{require_len, ContractError, DType};

/// Largest top-k logprob list admitted by the fused reduction.
pub const MAX_TOPK_LOGPROBS: usize = 32;
/// Maximum absolute error admitted for a categorical probability-row sum.
pub const CATEGORICAL_PROBABILITY_SUM_TOLERANCE: f64 = 1.0e-5;
const CATEGORICAL_CDF_LOGICAL_WARPS: usize = 32;
const CATEGORICAL_CDF_WARP_SIZE: usize = 32;
const CATEGORICAL_CDF_LOGICAL_LANES: usize =
    CATEGORICAL_CDF_LOGICAL_WARPS * CATEGORICAL_CDF_WARP_SIZE;
const TOP_K_FILTER_ITEMS_PER_PARTITION: usize = 4096;
const TOP_P_RENORM_ITEMS_PER_PARTITION: usize = 4096;
const TOPK_TARGET_PARTITIONS: usize = 128;
const TOPK_SORT_CAPACITY_PER_PARTITION: usize = 4096;
const TOPK_PARTIAL_STATE_BYTES: usize = 12;
const TOPK_CANDIDATE_BYTES: usize = 8;

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

/// Contract for vLLM-compatible per-row top-k logit filtering.
///
/// Logits are `[rows, vocab_size]`; `top_ks` contains one int32 value per row
/// in `[1, vocab_size]`. Values strictly below the row's kth-largest logit are
/// replaced by negative infinity in place. Every value equal to the threshold
/// is retained, so boundary ties may leave more than `top_k` finite entries.
/// Logits may contain infinities but must not contain NaNs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopKFilterSpec {
    rows: usize,
    vocab_size: usize,
    dtype: DType,
}

/// Contract for fused per-row top-p filtering and probability renormalization.
///
/// Logits are `[rows, vocab_size]`; `top_ps` contains one F32 value per row in
/// `(0, 1]`. Tokens are ordered by descending logit and descending token ID for
/// deterministic ties. The shortest ordered prefix whose original softmax mass
/// reaches `top_p` is retained. Other logits become negative infinity in place,
/// and a separate contiguous F32 `[rows, vocab_size]` output contains
/// probabilities renormalized over the retained prefix. Logits may contain
/// negative infinity but not NaN or positive infinity, and every row must
/// contain at least one finite value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopPRenormSpec {
    rows: usize,
    vocab_size: usize,
    dtype: DType,
}

/// Contract for sampled-token plus top-k raw logprobs.
///
/// Logits are `[rows, vocab_size]`; sampled IDs contain one int64 value per
/// row. Each output row starts with the sampled token and then contains
/// `top_k` tokens ordered by descending logit and ascending token ID for ties.
/// Logprobs share one row normalization and the sampled-token rank counts all
/// logits greater than or equal to its value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopKSampledLogprobsSpec {
    rows: usize,
    vocab_size: usize,
    top_k: usize,
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

/// Contract for deterministic categorical sampling from normalized F32 rows.
///
/// Probabilities are contiguous `[rows, vocab_size]`. RNG state is contiguous
/// int64 `[rows, 2]`, with each row storing a non-negative `(seed, counter)`.
/// One successful call emits one int64 token per row and advances every
/// counter exactly once. The reference validates the complete call before
/// mutating either output so a rejected call consumes no RNG state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CategoricalSampleSpec {
    rows: usize,
    vocab_size: usize,
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

impl TopKFilterSpec {
    /// Creates a validated per-row top-k filtering contract.
    pub fn new(rows: usize, vocab_size: usize, dtype: DType) -> Result<Self, ContractError> {
        if rows == 0 || vocab_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        let workspace_partitions = vocab_size.div_ceil(TOP_K_FILTER_ITEMS_PER_PARTITION);
        rows.checked_mul(vocab_size)
            .and_then(|_| {
                workspace_partitions
                    .checked_mul(TOP_K_FILTER_ITEMS_PER_PARTITION)
                    .and_then(|elements| elements.checked_add(1))
                    .and_then(|elements| rows.checked_mul(elements))
            })
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

    /// Number of independently sorted vocabulary partitions per row.
    pub const fn workspace_partitions(self) -> usize {
        self.vocab_size.div_ceil(TOP_K_FILTER_ITEMS_PER_PARTITION)
    }

    /// Caller-owned sorted uint32 keys plus one threshold key per row.
    pub const fn workspace_elements(self) -> usize {
        self.rows * (self.workspace_partitions() * TOP_K_FILTER_ITEMS_PER_PARTITION + 1)
    }

    pub const fn workspace_bytes(self) -> usize {
        self.workspace_elements() * size_of::<u32>()
    }
}

impl TopPRenormSpec {
    /// Creates a validated top-p filtering and renormalization contract.
    pub fn new(rows: usize, vocab_size: usize, dtype: DType) -> Result<Self, ContractError> {
        if rows == 0 || vocab_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        rows.checked_mul(vocab_size)
            .and_then(|_| top_p_renorm_workspace_bytes_checked(rows, vocab_size))
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

    pub const fn probabilities_numel(self) -> usize {
        self.logits_numel()
    }

    /// Number of independently sorted vocabulary partitions per row.
    pub const fn workspace_partitions(self) -> usize {
        self.vocab_size.div_ceil(TOP_P_RENORM_ITEMS_PER_PARTITION)
    }

    /// Caller-owned byte workspace for sorted keys, prefix masses, and state.
    pub const fn workspace_bytes(self) -> usize {
        let sorted_elements =
            self.rows * self.workspace_partitions() * TOP_P_RENORM_ITEMS_PER_PARTITION;
        let after_maxima = sorted_elements * 12 + self.rows * 4;
        let threshold_offset = (after_maxima + 7) & !7;
        threshold_offset + self.rows * 8 + self.rows * 4
    }
}

impl TopKSampledLogprobsSpec {
    /// Creates a validated sampled-token plus top-k contract.
    pub fn new(
        rows: usize,
        vocab_size: usize,
        top_k: usize,
        dtype: DType,
    ) -> Result<Self, ContractError> {
        if rows == 0 || vocab_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        let maximum = vocab_size.min(MAX_TOPK_LOGPROBS);
        if top_k == 0 || top_k > maximum {
            return Err(ContractError::TopKLogprobsOutOfRange { top_k, maximum });
        }
        rows.checked_mul(vocab_size)
            .and_then(|_| rows.checked_mul(top_k + 1))
            .and_then(|_| rows.checked_mul(topk_workspace_partitions(rows, vocab_size, top_k)))
            .and_then(|partials| {
                partials.checked_mul(TOPK_PARTIAL_STATE_BYTES + top_k * TOPK_CANDIDATE_BYTES)
            })
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self {
            rows,
            vocab_size,
            top_k,
            dtype,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn vocab_size(self) -> usize {
        self.vocab_size
    }

    pub const fn top_k(self) -> usize {
        self.top_k
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn logits_numel(self) -> usize {
        self.rows * self.vocab_size
    }

    pub const fn output_width(self) -> usize {
        self.top_k + 1
    }

    pub const fn output_numel(self) -> usize {
        self.rows * self.output_width()
    }

    /// Number of parallel row partitions used by the CUDA reduction.
    pub const fn workspace_partitions(self) -> usize {
        topk_workspace_partitions(self.rows, self.vocab_size, self.top_k)
    }

    /// Caller-owned byte workspace for partial logsumexp and top-k states.
    pub const fn workspace_bytes(self) -> usize {
        self.rows
            * self.workspace_partitions()
            * (TOPK_PARTIAL_STATE_BYTES + self.top_k * TOPK_CANDIDATE_BYTES)
    }
}

const fn topk_workspace_partitions(rows: usize, vocab_size: usize, top_k: usize) -> usize {
    let target_blocks = TOPK_TARGET_PARTITIONS.div_ceil(rows);
    let capacity_partitions = vocab_size.div_ceil(TOPK_SORT_CAPACITY_PER_PARTITION);
    let desired = if target_blocks > capacity_partitions {
        target_blocks
    } else {
        capacity_partitions
    };
    let item_limit = vocab_size / top_k;
    let partitions = if desired < item_limit {
        desired
    } else {
        item_limit
    };
    if partitions == 0 {
        1
    } else {
        partitions
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

impl CategoricalSampleSpec {
    /// Creates a validated contiguous categorical-sampling contract.
    pub fn new(rows: usize, vocab_size: usize) -> Result<Self, ContractError> {
        if rows == 0 || vocab_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if vocab_size > i64::MAX as usize {
            return Err(ContractError::ElementCountOverflow);
        }
        rows.checked_mul(vocab_size)
            .and_then(|_| rows.checked_mul(2))
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self { rows, vocab_size })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn vocab_size(self) -> usize {
        self.vocab_size
    }

    pub const fn probabilities_numel(self) -> usize {
        self.rows * self.vocab_size
    }

    pub const fn rng_state_numel(self) -> usize {
        self.rows * 2
    }
}

/// Samples one token per normalized F32 row from explicit Philox state.
///
/// The stream mapping is stable: Philox4x32-10 uses the seed as its two-word
/// key and the counter as the low two words of its four-word counter. The first
/// output word becomes `(word + 0.5) / 2^32`, and inverse CDF visits tokens in
/// ascending token-ID order through a fixed 32-warp F32 reduction tree. This
/// mapping is Loom-owned and does not reproduce PyTorch's exponential-race
/// seed-to-token identity.
pub fn categorical_sample_f32_reference(
    probabilities: &[f32],
    rng_state: &mut [i64],
    token_ids: &mut [i64],
    spec: CategoricalSampleSpec,
) -> Result<(), ContractError> {
    require_len(
        "probabilities",
        probabilities.len(),
        spec.probabilities_numel(),
    )?;
    require_len("rng_state", rng_state.len(), spec.rng_state_numel())?;
    require_len("token_ids", token_ids.len(), spec.rows())?;

    for row in 0..spec.rows() {
        let seed = rng_state[row * 2];
        if seed < 0 {
            return Err(ContractError::InvalidRngState {
                row,
                component: "seed",
                value: seed,
            });
        }
        let counter = rng_state[row * 2 + 1];
        if counter < 0 {
            return Err(ContractError::InvalidRngState {
                row,
                component: "counter",
                value: counter,
            });
        }
        if counter == i64::MAX {
            return Err(ContractError::RngCounterExhausted { row });
        }
    }

    for (row, values) in probabilities.chunks_exact(spec.vocab_size()).enumerate() {
        let mut sum = 0.0_f64;
        let mut has_positive = false;
        for (column, &value) in values.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(ContractError::InvalidCategoricalProbability { row, column, value });
            }
            has_positive |= value > 0.0;
            sum += f64::from(value);
        }
        if !has_positive {
            return Err(ContractError::NoPositiveCategoricalProbability { row });
        }
        if (sum - 1.0).abs() > CATEGORICAL_PROBABILITY_SUM_TOLERANCE {
            return Err(ContractError::CategoricalProbabilitySumOutOfRange {
                row,
                sum,
                tolerance: CATEGORICAL_PROBABILITY_SUM_TOLERANCE,
            });
        }
    }

    for (row, token_id) in token_ids.iter_mut().enumerate() {
        let state_offset = row * 2;
        let seed = rng_state[state_offset] as u64;
        let counter = rng_state[state_offset + 1] as u64;
        let uniform = categorical_uniform_f64(seed, counter);
        let probability_offset = row * spec.vocab_size();
        let values = &probabilities[probability_offset..probability_offset + spec.vocab_size()];
        *token_id = categorical_inverse_cdf(values, uniform) as i64;
        rng_state[state_offset + 1] += 1;
    }
    Ok(())
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

/// Applies per-row top-k filtering to F32 logits in place.
pub fn top_k_filter_f32_reference(
    logits: &mut [f32],
    top_ks: &[i32],
    spec: TopKFilterSpec,
) -> Result<(), ContractError> {
    top_k_filter_reference(logits, top_ks, spec, DType::F32)
}

/// Applies per-row top-k filtering to FP16 logits in place.
pub fn top_k_filter_f16_reference(
    logits: &mut [f16],
    top_ks: &[i32],
    spec: TopKFilterSpec,
) -> Result<(), ContractError> {
    top_k_filter_reference(logits, top_ks, spec, DType::F16)
}

/// Applies per-row top-k filtering to BF16 logits in place.
pub fn top_k_filter_bf16_reference(
    logits: &mut [bf16],
    top_ks: &[i32],
    spec: TopKFilterSpec,
) -> Result<(), ContractError> {
    top_k_filter_reference(logits, top_ks, spec, DType::Bf16)
}

/// Filters F32 logits by top-p and returns renormalized F32 probabilities.
pub fn top_p_renorm_f32_reference(
    logits: &mut [f32],
    top_ps: &[f32],
    probabilities: &mut [f32],
    spec: TopPRenormSpec,
) -> Result<(), ContractError> {
    top_p_renorm_reference(logits, top_ps, probabilities, spec, DType::F32)
}

/// Filters FP16 logits by top-p and returns renormalized F32 probabilities.
pub fn top_p_renorm_f16_reference(
    logits: &mut [f16],
    top_ps: &[f32],
    probabilities: &mut [f32],
    spec: TopPRenormSpec,
) -> Result<(), ContractError> {
    top_p_renorm_reference(logits, top_ps, probabilities, spec, DType::F16)
}

/// Filters BF16 logits by top-p and returns renormalized F32 probabilities.
pub fn top_p_renorm_bf16_reference(
    logits: &mut [bf16],
    top_ps: &[f32],
    probabilities: &mut [f32],
    spec: TopPRenormSpec,
) -> Result<(), ContractError> {
    top_p_renorm_reference(logits, top_ps, probabilities, spec, DType::Bf16)
}

/// Returns sampled-token plus top-k F32 logprobs from F32 logits.
pub fn topk_sampled_logprobs_f32_reference(
    logits: &[f32],
    sampled_token_ids: &[i64],
    output_token_ids: &mut [i32],
    output_logprobs: &mut [f32],
    sampled_token_ranks: &mut [i64],
    spec: TopKSampledLogprobsSpec,
) -> Result<(), ContractError> {
    topk_sampled_logprobs_reference(
        logits,
        sampled_token_ids,
        output_token_ids,
        output_logprobs,
        sampled_token_ranks,
        spec,
        DType::F32,
    )
}

/// Returns sampled-token plus top-k F32 logprobs from FP16 logits.
pub fn topk_sampled_logprobs_f16_reference(
    logits: &[f16],
    sampled_token_ids: &[i64],
    output_token_ids: &mut [i32],
    output_logprobs: &mut [f32],
    sampled_token_ranks: &mut [i64],
    spec: TopKSampledLogprobsSpec,
) -> Result<(), ContractError> {
    topk_sampled_logprobs_reference(
        logits,
        sampled_token_ids,
        output_token_ids,
        output_logprobs,
        sampled_token_ranks,
        spec,
        DType::F16,
    )
}

/// Returns sampled-token plus top-k F32 logprobs from BF16 logits.
pub fn topk_sampled_logprobs_bf16_reference(
    logits: &[bf16],
    sampled_token_ids: &[i64],
    output_token_ids: &mut [i32],
    output_logprobs: &mut [f32],
    sampled_token_ranks: &mut [i64],
    spec: TopKSampledLogprobsSpec,
) -> Result<(), ContractError> {
    topk_sampled_logprobs_reference(
        logits,
        sampled_token_ids,
        output_token_ids,
        output_logprobs,
        sampled_token_ranks,
        spec,
        DType::Bf16,
    )
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
    fn negative_infinity() -> Self;
}

impl LogitElement for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn negative_infinity() -> Self {
        Self::NEG_INFINITY
    }
}

impl LogitElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn negative_infinity() -> Self {
        Self::NEG_INFINITY
    }
}

impl LogitElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn negative_infinity() -> Self {
        Self::NEG_INFINITY
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

fn top_k_filter_reference<T: LogitElement>(
    logits: &mut [T],
    top_ks: &[i32],
    spec: TopKFilterSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("top_ks", top_ks.len(), spec.rows())?;
    for (row, &top_k) in top_ks.iter().enumerate() {
        if top_k < 1 || usize::try_from(top_k).map_or(true, |value| value > spec.vocab_size()) {
            return Err(ContractError::TopKFilterOutOfRange {
                row,
                top_k,
                vocab_size: spec.vocab_size(),
            });
        }
    }

    let mut ordered = Vec::with_capacity(spec.vocab_size());
    for (row, &top_k) in logits
        .chunks_exact_mut(spec.vocab_size())
        .zip(top_ks.iter())
    {
        if top_k as usize == spec.vocab_size() {
            continue;
        }
        ordered.clear();
        ordered.extend(row.iter().map(|&value| value.to_f32()));
        ordered.select_nth_unstable_by(top_k as usize - 1, |left, right| right.total_cmp(left));
        let threshold = ordered[top_k as usize - 1];
        for value in row {
            if value.to_f32() < threshold {
                *value = T::negative_infinity();
            }
        }
    }
    Ok(())
}

fn top_p_renorm_reference<T: LogitElement>(
    logits: &mut [T],
    top_ps: &[f32],
    probabilities: &mut [f32],
    spec: TopPRenormSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("top_ps", top_ps.len(), spec.rows())?;
    require_len(
        "probabilities",
        probabilities.len(),
        spec.probabilities_numel(),
    )?;
    for (row, &top_p) in top_ps.iter().enumerate() {
        if !top_p.is_finite() || !(top_p > 0.0 && top_p <= 1.0) {
            return Err(ContractError::InvalidProbability {
                parameter: "top_p",
                row,
                value: top_p,
            });
        }
    }
    for (row_index, row) in logits.chunks_exact(spec.vocab_size()).enumerate() {
        let mut has_finite = false;
        for (column, &value) in row.iter().enumerate() {
            let value = value.to_f32();
            if value.is_finite() {
                has_finite = true;
            } else if value != f32::NEG_INFINITY {
                return Err(ContractError::InvalidLogit {
                    row: row_index,
                    column,
                    value,
                });
            }
        }
        if !has_finite {
            return Err(ContractError::NoFiniteLogit { row: row_index });
        }
    }

    let mut ordered = Vec::with_capacity(spec.vocab_size());
    let mut weights = vec![0.0_f64; spec.vocab_size()];
    for (row_index, &top_p) in top_ps.iter().enumerate() {
        let start = row_index * spec.vocab_size();
        let row = &mut logits[start..start + spec.vocab_size()];
        let output = &mut probabilities[start..start + spec.vocab_size()];
        output.fill(0.0);
        let maximum = row
            .iter()
            .map(|&value| value.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0_f64;
        for (column, &value) in row.iter().enumerate() {
            let weight = f64::from(value.to_f32() - maximum).exp();
            weights[column] = weight;
            total += weight;
        }
        ordered.clear();
        ordered.extend(0..spec.vocab_size());
        ordered.sort_unstable_by(|&left, &right| {
            row[right]
                .to_f32()
                .total_cmp(&row[left].to_f32())
                .then_with(|| right.cmp(&left))
        });

        let mut retained_sum = 0.0_f64;
        let retained = if top_p == 1.0 {
            retained_sum = total;
            spec.vocab_size()
        } else {
            let target = f64::from(top_p) * total;
            let mut retained = 0_usize;
            for &column in &ordered {
                retained_sum += weights[column];
                retained += 1;
                if retained_sum >= target {
                    break;
                }
            }
            retained
        };
        for &column in &ordered[..retained] {
            output[column] = (weights[column] / retained_sum) as f32;
        }
        for &column in &ordered[retained..] {
            row[column] = T::negative_infinity();
        }
    }
    Ok(())
}

fn top_p_renorm_workspace_bytes_checked(rows: usize, vocab_size: usize) -> Option<usize> {
    let partitions = vocab_size.div_ceil(TOP_P_RENORM_ITEMS_PER_PARTITION);
    let sorted_elements = rows
        .checked_mul(partitions)?
        .checked_mul(TOP_P_RENORM_ITEMS_PER_PARTITION)?;
    let after_prefix = sorted_elements.checked_mul(12)?;
    let after_maxima = after_prefix.checked_add(rows.checked_mul(4)?)?;
    let threshold_offset = after_maxima.checked_add(7)? & !7;
    threshold_offset
        .checked_add(rows.checked_mul(8)?)?
        .checked_add(rows.checked_mul(4)?)
}

pub(crate) fn categorical_inverse_cdf(probabilities: &[f32], uniform: f64) -> usize {
    let chunks_per_warp = probabilities.len().div_ceil(CATEGORICAL_CDF_LOGICAL_LANES);
    let mut warp_sums = [0.0_f32; CATEGORICAL_CDF_LOGICAL_WARPS];
    let mut last_positive = 0;

    for (warp, warp_sum) in warp_sums.iter_mut().enumerate() {
        let warp_start = warp * chunks_per_warp * CATEGORICAL_CDF_WARP_SIZE;
        let warp_end = probabilities
            .len()
            .min(warp_start + chunks_per_warp * CATEGORICAL_CDF_WARP_SIZE);
        let mut lane_sums = [0.0_f32; CATEGORICAL_CDF_WARP_SIZE];
        for (lane, lane_sum) in lane_sums.iter_mut().enumerate() {
            let mut token = warp_start + lane;
            while token < warp_end {
                let probability = probabilities[token];
                *lane_sum += probability;
                if probability > 0.0 {
                    last_positive = last_positive.max(token);
                }
                token += CATEGORICAL_CDF_WARP_SIZE;
            }
        }
        *warp_sum = categorical_warp_sum(lane_sums);
    }

    let mut warp_prefixes = [0.0_f32; CATEGORICAL_CDF_LOGICAL_WARPS];
    let mut prefix = 0.0_f32;
    let mut target_warp = None;
    for (warp, &warp_sum) in warp_sums.iter().enumerate() {
        warp_prefixes[warp] = prefix;
        let next = prefix + warp_sum;
        if target_warp.is_none() && f64::from(next) > uniform {
            target_warp = Some(warp);
        }
        prefix = next;
    }

    if let Some(warp) = target_warp {
        let warp_start = warp * chunks_per_warp * CATEGORICAL_CDF_WARP_SIZE;
        let warp_end = probabilities
            .len()
            .min(warp_start + chunks_per_warp * CATEGORICAL_CDF_WARP_SIZE);
        let mut prefix = warp_prefixes[warp];
        for start in (warp_start..warp_end).step_by(CATEGORICAL_CDF_WARP_SIZE) {
            let mut values = [0.0_f32; CATEGORICAL_CDF_WARP_SIZE];
            for (lane, value) in values.iter_mut().enumerate() {
                let token = start + lane;
                if token < warp_end {
                    *value = probabilities[token];
                }
            }
            let inclusive = categorical_warp_inclusive_sum(values);
            for (lane, &cumulative) in inclusive.iter().enumerate() {
                let token = start + lane;
                if token < warp_end && f64::from(prefix + cumulative) > uniform {
                    return token;
                }
            }
            prefix += inclusive[CATEGORICAL_CDF_WARP_SIZE - 1];
        }
    }

    last_positive
}

fn categorical_warp_sum(mut values: [f32; CATEGORICAL_CDF_WARP_SIZE]) -> f32 {
    for offset in [16, 8, 4, 2, 1] {
        let previous = values;
        for lane in 0..CATEGORICAL_CDF_WARP_SIZE - offset {
            values[lane] = previous[lane] + previous[lane + offset];
        }
    }
    values[0]
}

fn categorical_warp_inclusive_sum(
    mut values: [f32; CATEGORICAL_CDF_WARP_SIZE],
) -> [f32; CATEGORICAL_CDF_WARP_SIZE] {
    for offset in [1, 2, 4, 8, 16] {
        let previous = values;
        for lane in offset..CATEGORICAL_CDF_WARP_SIZE {
            values[lane] = previous[lane] + previous[lane - offset];
        }
    }
    values
}

pub(crate) fn categorical_philox_word(seed: u64, counter: u64) -> u32 {
    let counter = [counter as u32, (counter >> 32) as u32, 0_u32, 0_u32];
    let key = [seed as u32, (seed >> 32) as u32];
    philox4x32_10(counter, key)[0]
}

fn categorical_uniform_f64(seed: u64, counter: u64) -> f64 {
    (f64::from(categorical_philox_word(seed, counter)) + 0.5) * (1.0 / 4_294_967_296.0)
}

fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    const MULTIPLIER_0: u32 = 0xD251_1F53;
    const MULTIPLIER_1: u32 = 0xCD9E_8D57;
    const WEYL_0: u32 = 0x9E37_79B9;
    const WEYL_1: u32 = 0xBB67_AE85;

    for round in 0..10 {
        let product_0 = u64::from(MULTIPLIER_0) * u64::from(counter[0]);
        let product_1 = u64::from(MULTIPLIER_1) * u64::from(counter[2]);
        counter = [
            (product_1 >> 32) as u32 ^ counter[1] ^ key[0],
            product_1 as u32,
            (product_0 >> 32) as u32 ^ counter[3] ^ key[1],
            product_0 as u32,
        ];
        if round != 9 {
            key[0] = key[0].wrapping_add(WEYL_0);
            key[1] = key[1].wrapping_add(WEYL_1);
        }
    }
    counter
}

#[allow(clippy::too_many_arguments)]
fn topk_sampled_logprobs_reference<T: LogitElement>(
    logits: &[T],
    sampled_token_ids: &[i64],
    output_token_ids: &mut [i32],
    output_logprobs: &mut [f32],
    sampled_token_ranks: &mut [i64],
    spec: TopKSampledLogprobsSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("sampled_token_ids", sampled_token_ids.len(), spec.rows())?;
    require_len(
        "output_token_ids",
        output_token_ids.len(),
        spec.output_numel(),
    )?;
    require_len(
        "output_logprobs",
        output_logprobs.len(),
        spec.output_numel(),
    )?;
    require_len(
        "sampled_token_ranks",
        sampled_token_ranks.len(),
        spec.rows(),
    )?;

    for row_index in 0..spec.rows() {
        let row_start = row_index * spec.vocab_size();
        let row = &logits[row_start..row_start + spec.vocab_size()];
        let token_id = sampled_token_ids[row_index];
        let sampled_index =
            usize::try_from(token_id).map_err(|_| ContractError::TokenIdOutOfBounds {
                row: row_index,
                token_id,
                vocab_size: spec.vocab_size(),
            })?;
        if sampled_index >= spec.vocab_size() {
            return Err(ContractError::TokenIdOutOfBounds {
                row: row_index,
                token_id,
                vocab_size: spec.vocab_size(),
            });
        }

        let maximum = row
            .iter()
            .map(|&value| value.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let exponential_sum = row
            .iter()
            .map(|&value| f64::from(value.to_f32() - maximum).exp())
            .sum::<f64>();
        let log_normalizer = maximum + exponential_sum.ln() as f32;
        let sampled = row[sampled_index].to_f32();
        let output_start = row_index * spec.output_width();
        output_token_ids[output_start] = sampled_index as i32;
        output_logprobs[output_start] = sampled - log_normalizer;
        sampled_token_ranks[row_index] = row
            .iter()
            .filter(|&&value| value.to_f32() >= sampled)
            .count() as i64;

        let mut candidates = row
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, value.to_f32()))
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|(left_index, left), (right_index, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_index.cmp(right_index))
        });
        for (slot, &(index, value)) in candidates[..spec.top_k()].iter().enumerate() {
            output_token_ids[output_start + slot + 1] = index as i32;
            output_logprobs[output_start + slot + 1] = value - log_normalizer;
        }
    }
    Ok(())
}
