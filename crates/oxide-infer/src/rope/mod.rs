//! Standard rotary position embedding contract and CPU reference.

use crate::error::require_len;
use crate::{ContractError, DType};
use half::bf16;

/// Standard NeoX split-half RoPE over contiguous NHD query and key tensors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16RopePosIdsSpec {
    tokens: usize,
    query_heads: usize,
    key_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_scale: f32,
    rope_theta: f32,
}

impl Bf16RopePosIdsSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokens: usize,
        query_heads: usize,
        key_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_scale: f32,
        rope_theta: f32,
    ) -> Result<Self, ContractError> {
        if tokens == 0 || query_heads == 0 || key_heads == 0 || head_dim == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > head_dim {
            return Err(ContractError::InvalidRotaryDimension {
                rotary_dim,
                head_dim,
            });
        }
        if !rope_scale.is_finite() || rope_scale <= 0.0 {
            return Err(ContractError::InvalidRopeScale(rope_scale));
        }
        if !rope_theta.is_finite() || rope_theta <= 1.0 {
            return Err(ContractError::InvalidRopeTheta(rope_theta));
        }
        tokens
            .checked_mul(query_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;
        tokens
            .checked_mul(key_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            tokens,
            query_heads,
            key_heads,
            head_dim,
            rotary_dim,
            rope_scale,
            rope_theta,
        })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn query_heads(self) -> usize {
        self.query_heads
    }

    pub const fn key_heads(self) -> usize {
        self.key_heads
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn rotary_dim(self) -> usize {
        self.rotary_dim
    }

    pub const fn rope_scale(self) -> f32 {
        self.rope_scale
    }

    pub const fn rope_theta(self) -> f32 {
        self.rope_theta
    }

    pub const fn query_numel(self) -> usize {
        self.tokens * self.query_heads * self.head_dim
    }

    pub const fn key_numel(self) -> usize {
        self.tokens * self.key_heads * self.head_dim
    }

    pub const fn position_numel(self) -> usize {
        self.tokens
    }

    pub const fn dtype(self) -> DType {
        DType::Bf16
    }
}

/// Applies standard NeoX split-half RoPE with F32 arithmetic and BF16 outputs.
pub fn rope_pos_ids_bf16_reference(
    query: &[bf16],
    key: &[bf16],
    position_ids: &[i32],
    query_output: &mut [bf16],
    key_output: &mut [bf16],
    spec: Bf16RopePosIdsSpec,
) -> Result<(), ContractError> {
    require_len("query", query.len(), spec.query_numel())?;
    require_len("key", key.len(), spec.key_numel())?;
    require_len("position_ids", position_ids.len(), spec.position_numel())?;
    require_len("query_output", query_output.len(), spec.query_numel())?;
    require_len("key_output", key_output.len(), spec.key_numel())?;

    for (token, &position) in position_ids.iter().enumerate() {
        if position < 0 {
            return Err(ContractError::NegativePositionId { token, position });
        }
    }

    rotate_tensor(query, position_ids, query_output, spec.query_heads(), spec);
    rotate_tensor(key, position_ids, key_output, spec.key_heads(), spec);
    Ok(())
}

fn rotate_tensor(
    input: &[bf16],
    position_ids: &[i32],
    output: &mut [bf16],
    heads: usize,
    spec: Bf16RopePosIdsSpec,
) {
    let half = spec.rotary_dim() / 2;
    for (token, &position) in position_ids.iter().enumerate() {
        for head in 0..heads {
            let base = (token * heads + head) * spec.head_dim();
            for pair in 0..half {
                let exponent = 2.0 * pair as f32 / spec.rotary_dim() as f32;
                let inverse_frequency = spec.rope_theta().powf(-exponent) / spec.rope_scale();
                let angle = position as f32 * inverse_frequency;
                let (sin, cos) = angle.sin_cos();
                let first = input[base + pair].to_f32();
                let second = input[base + half + pair].to_f32();
                output[base + pair] = bf16::from_f32(first * cos - second * sin);
                output[base + half + pair] = bf16::from_f32(second * cos + first * sin);
            }
            output[base + spec.rotary_dim()..base + spec.head_dim()]
                .copy_from_slice(&input[base + spec.rotary_dim()..base + spec.head_dim()]);
        }
    }
}

#[cfg(test)]
mod tests;
