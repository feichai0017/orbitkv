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

fn softplus(input: &GraphTensor) -> GraphTensor {
    input.maximum_f32(0.0) + ((-input.abs()).exp() + 1.0).log()
}

#[cfg(test)]
mod tests {
    use luminal::prelude::{CompileOptions, Graph, ReferenceRuntime, Runtime};

    use super::*;
    use crate::{GatedDeltaReferenceInput, gated_delta_reference};

    #[test]
    fn projected_gate_and_readout_match_independent_reference() {
        let geometry = GatedDeltaGeometry {
            key_heads: 1,
            value_heads: 2,
            key_width: 2,
            value_width: 2,
            normalization_epsilon: 1e-6,
        };
        let convolved = [1.0, 2.0, 2.0, 1.0, 4.0, 2.0, 1.0, 3.0];
        let output_gate = [0.5, -0.25, 0.75, -1.0];
        let update_logits = [0.0_f32, 1.0];
        let decay_logits = [-0.5, 0.25];
        let decay_log_rates = [0.0, 0.5_f32.ln()];
        let decay_bias = [0.25, -0.25];
        let norm = [1.5, 0.5];
        let state = [0.1, 0.2, 0.3, 0.4, -0.1, 0.2, 0.4, -0.2];
        let log_decay = decay_logits
            .iter()
            .zip(decay_bias)
            .zip(decay_log_rates)
            .map(|((&gate, bias), rate)| -rate.exp() * softplus_f32(gate + bias))
            .collect::<Vec<_>>();
        let beta = update_logits
            .iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let reference = gated_delta_reference(
            geometry,
            GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: 1,
                query: &convolved[..2],
                key: &convolved[2..4],
                value: &convolved[4..],
                log_decay: &log_decay,
                update_gate: &beta,
                initial_state: &state,
            },
        )
        .unwrap();
        let expected = reference
            .values
            .chunks_exact(2)
            .zip(output_gate.chunks_exact(2))
            .flat_map(|(values, gate)| {
                let inverse = (values.iter().map(|value| value * value).sum::<f32>() / 2.0 + 1e-6)
                    .sqrt()
                    .recip();
                values
                    .iter()
                    .zip(gate)
                    .zip(norm)
                    .map(move |((&value, &gate), weight)| {
                        value * inverse * weight * gate / (1.0 + (-gate).exp())
                    })
            })
            .collect::<Vec<_>>();

        let mut graph = Graph::new();
        let qkv = graph.named_tensor("convolved_qkv", (1, 8));
        let z = graph.named_tensor("output_gate", (1, 4));
        let b = graph.named_tensor("update_gate", (1, 2));
        let a = graph.named_tensor("decay_gate", (1, 2));
        let a_log = graph.named_tensor("decay_log_rates", 2);
        let dt_bias = graph.named_tensor("decay_bias", 2);
        let norm_weight = graph.named_tensor("output_norm", 2);
        let previous_state = graph.named_tensor("previous_state", (1, 2, 2, 2));
        let outputs = gated_delta_projected_step(
            GatedDeltaProjectedInputs {
                convolved_qkv: qkv,
                output_gate: z,
                update_gate_logits: b,
                decay_gate_logits: a,
                decay_log_rates: a_log,
                decay_bias: dt_bias,
                output_norm: norm_weight,
                previous_state,
                batch_size: 1.into(),
            },
            geometry,
            1e-6,
        )
        .unwrap();
        let values = outputs.values.output();
        let next_state = outputs.next_state.output();
        let mut runtime = graph.compile(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        runtime.set_data(qkv, convolved.to_vec());
        runtime.set_data(z, output_gate.to_vec());
        runtime.set_data(b, update_logits.to_vec());
        runtime.set_data(a, decay_logits.to_vec());
        runtime.set_data(a_log, decay_log_rates.to_vec());
        runtime.set_data(dt_bias, decay_bias.to_vec());
        runtime.set_data(norm_weight, norm.to_vec());
        runtime.set_data(previous_state, state.to_vec());
        runtime.execute(&graph.dyn_map);

        assert_close(runtime.get_f32(values), &expected);
        assert_close(runtime.get_f32(next_state), &reference.state);
    }

    fn softplus_f32(value: f32) -> f32 {
        value.max(0.0) + (-value.abs()).exp().ln_1p()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }
}
