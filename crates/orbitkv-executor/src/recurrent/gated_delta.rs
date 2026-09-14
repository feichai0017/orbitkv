//! Gated-delta semantics after input projection and causal convolution.

use luminal::{
    dtype::DType,
    prelude::{Expression, GraphTensor},
};

use super::{GatedDeltaGeometry, GatedDeltaStepInputs, RecurrentError, gated_delta_step};

/// Projected inputs for one decode token.
///
/// `convolved_qkv` is the output of the causal depthwise convolution. The
/// remaining tensors come directly from checkpoint-backed projections.
#[derive(Clone, Copy)]
pub struct GatedDeltaProjectedInputs {
    pub convolved_qkv: GraphTensor,
    pub output_gate: GraphTensor,
    pub update_gate_logits: GraphTensor,
    pub decay_gate_logits: GraphTensor,
    pub decay_log_rates: GraphTensor,
    pub decay_bias: GraphTensor,
    pub output_norm: GraphTensor,
    pub previous_state: GraphTensor,
    pub batch_size: Expression,
}

/// Gated token values and the recurrent state produced by one decode token.
#[derive(Clone, Copy)]
pub struct GatedDeltaProjectedOutputs {
    pub values: GraphTensor,
    pub next_state: GraphTensor,
}

/// Builds the checkpoint-defined gated-delta recurrence and output gate.
///
/// This function intentionally starts after causal convolution: convolution
/// history is a separate persistent state with its own arena and commit proof.
/// Keeping the two state machines separate prevents an incomplete convolution
/// implementation from being presented as an executable decoder layer.
///
/// # Errors
///
/// Rejects incompatible graph ownership, dtype, shape, or head geometry.
pub fn gated_delta_projected_step(
    inputs: GatedDeltaProjectedInputs,
    geometry: GatedDeltaGeometry,
    output_norm_epsilon: f32,
) -> Result<GatedDeltaProjectedOutputs, RecurrentError> {
    geometry.validate()?;
    let GatedDeltaProjectedInputs {
        convolved_qkv,
        output_gate,
        update_gate_logits,
        decay_gate_logits,
        decay_log_rates,
        decay_bias,
        output_norm,
        previous_state,
        batch_size,
    } = inputs;
    let key_elements = geometry
        .key_heads
        .checked_mul(geometry.key_width)
        .ok_or(RecurrentError::InvalidGeometry)?;
    let value_elements = geometry
        .value_heads
        .checked_mul(geometry.value_width)
        .ok_or(RecurrentError::InvalidGeometry)?;
    let projected_elements = key_elements
        .checked_mul(2)
        .and_then(|elements| elements.checked_add(value_elements))
        .ok_or(RecurrentError::InvalidGeometry)?;
    if !output_norm_epsilon.is_finite()
        || output_norm_epsilon <= 0.0
        || [
            convolved_qkv,
            output_gate,
            update_gate_logits,
            decay_gate_logits,
            decay_log_rates,
            decay_bias,
            output_norm,
            previous_state,
        ]
        .iter()
        .any(|tensor| tensor.dtype != DType::F32 || tensor.graph_ref != convolved_qkv.graph_ref)
        || convolved_qkv.dims() != [batch_size, projected_elements.into()]
        || output_gate.dims() != [batch_size, value_elements.into()]
        || update_gate_logits.dims() != [batch_size, geometry.value_heads.into()]
        || decay_gate_logits.dims() != update_gate_logits.dims()
        || decay_log_rates.dims() != [Expression::from(geometry.value_heads)]
        || decay_bias.dims() != decay_log_rates.dims()
        || output_norm.dims() != [Expression::from(geometry.value_width)]
    {
        return Err(RecurrentError::InvalidGeometry);
    }

    let query = convolved_qkv
        .slice((.., ..key_elements))
        .split_dims(1, geometry.key_width);
    let key = convolved_qkv
        .slice((.., key_elements..key_elements * 2))
        .split_dims(1, geometry.key_width);
    let value = convolved_qkv
        .slice((.., key_elements * 2..))
        .split_dims(1, geometry.value_width);
    let decay_argument = decay_gate_logits + decay_bias.expand_dim(0, batch_size);
    let log_decay = -decay_log_rates.exp().expand_dim(0, batch_size) * softplus(&decay_argument);
    let recurrent = gated_delta_step(
        GatedDeltaStepInputs {
            query,
            key,
            value,
            log_decay,
            update_gate: update_gate_logits.sigmoid(),
            previous_state,
            batch_size,
        },
        geometry,
    )?;
    let values = recurrent.values.std_norm(2, output_norm_epsilon)
        * output_norm.expand_lhs([batch_size, Expression::from(geometry.value_heads)])
        * output_gate.split_dims(1, geometry.value_width).swish();
    Ok(GatedDeltaProjectedOutputs {
        values,
        next_state: recurrent.next_state,
    })
}

pub(crate) fn softplus(input: &GraphTensor) -> GraphTensor {
    input.maximum_f32(0.0) + ((-input.abs()).exp() + 1.0).log()
}

#[cfg(test)]
#[path = "../../tests/unit/recurrent/gated_delta/mod.rs"]
mod tests;
