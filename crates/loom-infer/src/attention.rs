//! Single-request decode-attention contracts and CPU references.

use crate::ContractError;
use crate::error::require_len;
use half::bf16;

/// Head dimension admitted by the first single-decode contract.
pub const SINGLE_DECODE_HEAD_DIM: usize = 128;

/// BF16 single-request decode over contiguous NHD key and value caches.
///
/// Query and output use `[query_heads, 128]`. Key and value use
/// `[kv_len, kv_heads, 128]`. Scores and online-softmax state use F32. LSE is
/// `log2(sum(exp(scaled_logit)))` to match FlashInfer's returned state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16SingleDecodeSpec {
    kv_len: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    softmax_scale: f32,
}

impl Bf16SingleDecodeSpec {
    /// Creates the first fixed BF16, NHD, full-window decode contract.
    pub fn new(
        kv_len: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ContractError> {
        if kv_len == 0 || num_query_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if head_dim != SINGLE_DECODE_HEAD_DIM {
            return Err(ContractError::UnsupportedHeadDimension {
                expected: SINGLE_DECODE_HEAD_DIM,
                actual: head_dim,
            });
        }
        if !num_query_heads.is_multiple_of(num_kv_heads) {
            return Err(ContractError::InvalidHeadMapping {
                query_heads: num_query_heads,
                kv_heads: num_kv_heads,
            });
        }

        num_query_heads
            .checked_mul(head_dim)
            .ok_or(ContractError::ElementCountOverflow)?;
        let elements_per_token = num_kv_heads
            .checked_mul(head_dim)
            .ok_or(ContractError::ElementCountOverflow)?;
        kv_len
            .checked_mul(elements_per_token)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            kv_len,
            num_query_heads,
            num_kv_heads,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub const fn kv_len(self) -> usize {
        self.kv_len
    }

    pub const fn num_query_heads(self) -> usize {
        self.num_query_heads
    }

    pub const fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    pub const fn head_dim(self) -> usize {
        SINGLE_DECODE_HEAD_DIM
    }

    pub const fn softmax_scale(self) -> f32 {
        self.softmax_scale
    }

    pub const fn gqa_group_size(self) -> usize {
        self.num_query_heads / self.num_kv_heads
    }

    pub const fn query_numel(self) -> usize {
        self.num_query_heads * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn kv_numel(self) -> usize {
        self.kv_len * self.num_kv_heads * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn output_numel(self) -> usize {
        self.query_numel()
    }

    pub const fn lse_numel(self) -> usize {
        self.num_query_heads
    }

    pub const fn kv_head_for_query_head(self, query_head: usize) -> Option<usize> {
        if query_head < self.num_query_heads {
            Some(query_head / self.gqa_group_size())
        } else {
            None
        }
    }
}

/// Computes BF16 single-request decode with F32 online softmax.
///
/// Dot products visit head components in ascending order. Softmax visits KV
/// tokens in ascending order. Output rounds once after normalization.
pub fn single_decode_bf16_reference(
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    output: &mut [bf16],
    lse: &mut [f32],
    spec: Bf16SingleDecodeSpec,
) -> Result<(), ContractError> {
    require_len("Q", query.len(), spec.query_numel())?;
    require_len("K", key.len(), spec.kv_numel())?;
    require_len("V", value.len(), spec.kv_numel())?;
    require_len("O", output.len(), spec.output_numel())?;
    require_len("LSE", lse.len(), spec.lse_numel())?;

    for (query_head, lse_slot) in lse.iter_mut().enumerate() {
        let query_offset = query_head * spec.head_dim();
        let kv_head = spec
            .kv_head_for_query_head(query_head)
            .expect("enumerated query head is inside the validated specification");
        let mut output_state = [0.0_f32; SINGLE_DECODE_HEAD_DIM];
        let mut max_score = 0.0_f32;
        let mut normalizer = 0.0_f32;

        for token in 0..spec.kv_len() {
            let kv_offset = (token * spec.num_kv_heads() + kv_head) * SINGLE_DECODE_HEAD_DIM;
            let dot = (0..SINGLE_DECODE_HEAD_DIM).fold(0.0_f32, |sum, component| {
                query[query_offset + component]
                    .to_f32()
                    .mul_add(key[kv_offset + component].to_f32(), sum)
            });
            let score = dot * spec.softmax_scale();

            if token == 0 {
                max_score = score;
                normalizer = 1.0;
                for component in 0..SINGLE_DECODE_HEAD_DIM {
                    output_state[component] = value[kv_offset + component].to_f32();
                }
                continue;
            }

            let next_max = max_score.max(score);
            let previous_weight = (max_score - next_max).exp();
            let current_weight = (score - next_max).exp();
            normalizer = normalizer * previous_weight + current_weight;
            for component in 0..SINGLE_DECODE_HEAD_DIM {
                output_state[component] = output_state[component] * previous_weight
                    + value[kv_offset + component].to_f32() * current_weight;
            }
            max_score = next_max;
        }

        let output_offset = query_head * SINGLE_DECODE_HEAD_DIM;
        for component in 0..SINGLE_DECODE_HEAD_DIM {
            output[output_offset + component] =
                bf16::from_f32(output_state[component] / normalizer);
        }
        *lse_slot = (max_score + normalizer.ln()) * core::f32::consts::LOG2_E;
    }

    Ok(())
}

#[cfg(test)]
#[path = "attention/tests.rs"]
mod tests;
