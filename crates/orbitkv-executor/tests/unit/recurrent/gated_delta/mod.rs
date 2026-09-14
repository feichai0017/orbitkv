use orbitkv_compiler::prelude::{CompileOptions, Graph, ReferenceRuntime, Runtime};

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

#[test]
fn softplus_stays_stable_across_large_signed_gate_logits() {
    let values = [-100.0_f32, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0];
    let mut graph = Graph::new();
    let input = graph.named_tensor("gate", values.len());
    let output = softplus(&input).output();
    let mut runtime = graph.compile(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    runtime.set_data(input, values.to_vec());
    runtime.execute(&graph.dyn_map);
    let expected = values.map(softplus_f32);
    assert_close(runtime.get_f32(output), &expected);
    assert!(
        runtime
            .get_f32(output)
            .iter()
            .all(|value| value.is_finite())
    );
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
