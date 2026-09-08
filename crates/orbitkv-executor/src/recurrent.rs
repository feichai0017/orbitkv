//! Backend-neutral recurrent delta-rule semantics.

use thiserror::Error;

/// Static head geometry for a gated delta-rule state transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaGeometry {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub normalization_epsilon: f32,
}

/// Contiguous `[batch, sequence, heads, width]` inputs and an initial
/// `[batch, heads, key_width, value_width]` state.
#[derive(Clone, Copy, Debug)]
pub struct GatedDeltaReferenceInput<'a> {
    pub batch_size: usize,
    pub sequence_tokens: usize,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub log_decay: &'a [f32],
    pub update_gate: &'a [f32],
    pub initial_state: &'a [f32],
}

/// Reference token values and the state after the final token.
#[derive(Clone, Debug, PartialEq)]
pub struct GatedDeltaReferenceOutput {
    pub values: Box<[f32]>,
    pub state: Box<[f32]>,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RecurrentError {
    #[error("recurrent delta-rule geometry is invalid")]
    InvalidGeometry,
    #[error("recurrent delta-rule {field} length is {actual}, expected {expected}")]
    InputLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl GatedDeltaGeometry {
    fn validate(self) -> Result<(), RecurrentError> {
        if self.key_heads == 0
            || self.value_heads == 0
            || !self.value_heads.is_multiple_of(self.key_heads)
            || self.key_width == 0
            || u16::try_from(self.key_width).is_err()
            || self.value_width == 0
            || !self.normalization_epsilon.is_finite()
            || self.normalization_epsilon < 0.0
        {
            return Err(RecurrentError::InvalidGeometry);
        }
        Ok(())
    }

    fn query_scale(self) -> f32 {
        let width =
            u16::try_from(self.key_width).expect("validated recurrent key width must fit u16");
        f32::from(width).sqrt().recip()
    }
}

/// Executes the normalized gated delta recurrence with f32 accumulation.
///
/// For each token and head, this computes:
///
/// `S' = exp(g) S + k^T (beta (v - k exp(g) S))`
///
/// `y = q S' / sqrt(key_width)`
///
/// where `q` and `k` are independently L2-normalized with the declared
/// epsilon. Processing a sequence in multiple calls is equivalent to one call
/// when the prior call's returned state is supplied to the next call.
///
/// # Errors
///
/// Rejects zero, overflowing, non-finite geometry and any non-canonical input
/// length. Input values may contain IEEE NaNs or infinities and are propagated.
pub fn gated_delta_reference(
    geometry: GatedDeltaGeometry,
    input: GatedDeltaReferenceInput<'_>,
) -> Result<GatedDeltaReferenceOutput, RecurrentError> {
    geometry.validate()?;
    if input.batch_size == 0 || input.sequence_tokens == 0 {
        return Err(RecurrentError::InvalidGeometry);
    }
    let query_len = checked_product(&[
        input.batch_size,
        input.sequence_tokens,
        geometry.key_heads,
        geometry.key_width,
    ])?;
    let value_len = checked_product(&[
        input.batch_size,
        input.sequence_tokens,
        geometry.value_heads,
        geometry.value_width,
    ])?;
    let gate_len = checked_product(&[
        input.batch_size,
        input.sequence_tokens,
        geometry.value_heads,
    ])?;
    let state_len = checked_product(&[
        input.batch_size,
        geometry.value_heads,
        geometry.key_width,
        geometry.value_width,
    ])?;
    require_len("query", input.query, query_len)?;
    require_len("key", input.key, query_len)?;
    require_len("value", input.value, value_len)?;
    require_len("log decay", input.log_decay, gate_len)?;
    require_len("update gate", input.update_gate, gate_len)?;
    require_len("initial state", input.initial_state, state_len)?;

    let mut state = input.initial_state.to_vec();
    let mut values = vec![0.0_f32; value_len];
    let query_scale = geometry.query_scale();
    let state_head_len = geometry.key_width * geometry.value_width;
    for batch in 0..input.batch_size {
        for token in 0..input.sequence_tokens {
            for head in 0..geometry.value_heads {
                let gate_index =
                    (batch * input.sequence_tokens + token) * geometry.value_heads + head;
                let key_head = head / (geometry.value_heads / geometry.key_heads);
                let query_base = ((batch * input.sequence_tokens + token) * geometry.key_heads
                    + key_head)
                    * geometry.key_width;
                let value_base = gate_index * geometry.value_width;
                let state_base = (batch * geometry.value_heads + head) * state_head_len;
                transition_head(
                    geometry,
                    &input.query[query_base..query_base + geometry.key_width],
                    &input.key[query_base..query_base + geometry.key_width],
                    &input.value[value_base..value_base + geometry.value_width],
                    input.log_decay[gate_index],
                    input.update_gate[gate_index],
                    &mut state[state_base..state_base + state_head_len],
                    &mut values[value_base..value_base + geometry.value_width],
                    query_scale,
                );
            }
        }
    }
    Ok(GatedDeltaReferenceOutput {
        values: values.into_boxed_slice(),
        state: state.into_boxed_slice(),
    })
}

#[allow(clippy::too_many_arguments)]
fn transition_head(
    geometry: GatedDeltaGeometry,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    log_decay: f32,
    update_gate: f32,
    state: &mut [f32],
    output: &mut [f32],
    query_scale: f32,
) {
    let query_norm = inverse_norm(query, geometry.normalization_epsilon) * query_scale;
    let key_norm = inverse_norm(key, geometry.normalization_epsilon);
    let decay = log_decay.exp();
    let mut memory = vec![0.0_f32; geometry.value_width];
    for key_index in 0..geometry.key_width {
        let normalized_key = key[key_index] * key_norm;
        let row =
            &mut state[key_index * geometry.value_width..(key_index + 1) * geometry.value_width];
        for (state_value, memory_value) in row.iter_mut().zip(&mut memory) {
            *state_value *= decay;
            *memory_value += *state_value * normalized_key;
        }
    }
    for value_index in 0..geometry.value_width {
        let delta = (value[value_index] - memory[value_index]) * update_gate;
        let mut result = 0.0_f32;
        for key_index in 0..geometry.key_width {
            let state_index = key_index * geometry.value_width + value_index;
            state[state_index] += key[key_index] * key_norm * delta;
            result += state[state_index] * query[key_index] * query_norm;
        }
        output[value_index] = result;
    }
}

fn inverse_norm(values: &[f32], epsilon: f32) -> f32 {
    (values.iter().map(|value| value * value).sum::<f32>() + epsilon)
        .sqrt()
        .recip()
}

fn checked_product(values: &[usize]) -> Result<usize, RecurrentError> {
    values
        .iter()
        .copied()
        .try_fold(1_usize, usize::checked_mul)
        .ok_or(RecurrentError::InvalidGeometry)
}

fn require_len(field: &'static str, values: &[f32], expected: usize) -> Result<(), RecurrentError> {
    if values.len() != expected {
        return Err(RecurrentError::InputLength {
            field,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod graph {
    use luminal::{
        dtype::DType,
        prelude::{Expression, GraphTensor},
    };

    use super::{GatedDeltaGeometry, RecurrentError};

    /// One-token semantic inputs. Every tensor uses f32 accumulation layout.
    #[derive(Clone, Copy)]
    pub struct GatedDeltaStepInputs {
        pub query: GraphTensor,
        pub key: GraphTensor,
        pub value: GraphTensor,
        pub log_decay: GraphTensor,
        pub update_gate: GraphTensor,
        pub previous_state: GraphTensor,
        pub batch_size: Expression,
    }

    /// Observable token values and the next recurrent state.
    #[derive(Clone, Copy)]
    pub struct GatedDeltaStepOutputs {
        pub values: GraphTensor,
        pub next_state: GraphTensor,
    }

    /// Expands one normalized gated-delta transition into pure Luminal HLIR.
    ///
    /// This is the semantic graph that future CUDA candidates must match in
    /// egglog. It intentionally materializes `next_state`; required-alias
    /// selection is a backend concern, not part of the mathematical contract.
    ///
    /// # Errors
    ///
    /// Rejects invalid geometry, shape, dtype, or graph ownership.
    pub fn gated_delta_step(
        inputs: GatedDeltaStepInputs,
        geometry: GatedDeltaGeometry,
    ) -> Result<GatedDeltaStepOutputs, RecurrentError> {
        geometry.validate()?;
        let GatedDeltaStepInputs {
            query,
            key,
            value,
            log_decay,
            update_gate,
            previous_state,
            batch_size,
        } = inputs;
        if [query, key, value, log_decay, update_gate, previous_state]
            .iter()
            .any(|tensor| tensor.dtype != DType::F32 || tensor.graph_ref != query.graph_ref)
            || query.dims()
                != [
                    batch_size,
                    geometry.key_heads.into(),
                    geometry.key_width.into(),
                ]
            || key.dims() != query.dims()
            || value.dims()
                != [
                    batch_size,
                    geometry.value_heads.into(),
                    geometry.value_width.into(),
                ]
            || log_decay.dims() != [batch_size, geometry.value_heads.into()]
            || update_gate.dims() != log_decay.dims()
            || previous_state.dims()
                != [
                    batch_size,
                    geometry.value_heads.into(),
                    geometry.key_width.into(),
                    geometry.value_width.into(),
                ]
        {
            return Err(RecurrentError::InvalidGeometry);
        }

        let group_size = geometry.value_heads / geometry.key_heads;
        let query = query.expand_dim(2, group_size).merge_dims(1, 2) * 1.0;
        let key = key.expand_dim(2, group_size).merge_dims(1, 2) * 1.0;
        let query_norm = normalize(&query, geometry.key_width, geometry.normalization_epsilon)
            * geometry.query_scale();
        let key_norm = normalize(&key, geometry.key_width, geometry.normalization_epsilon);
        let decay = log_decay
            .exp()
            .expand_dim(2, geometry.key_width)
            .expand_dim(3, geometry.value_width);
        let decayed_state = previous_state * decay;
        let key_matrix = key_norm.expand_dim(3, geometry.value_width);
        let memory = (decayed_state * key_matrix).sum(2);
        let delta = (value - memory) * update_gate.expand_dim(2, geometry.value_width);
        let next_state = decayed_state + key_matrix * delta.expand_dim(2, geometry.key_width);
        let values = (next_state * query_norm.expand_dim(3, geometry.value_width)).sum(2);
        Ok(GatedDeltaStepOutputs { values, next_state })
    }

    fn normalize(input: &GraphTensor, width: usize, epsilon: f32) -> GraphTensor {
        let inverse_norm = (input.square().sum(2) + epsilon)
            .sqrt()
            .reciprocal()
            .expand_dim(2, width);
        *input * inverse_norm
    }
}
#[cfg(feature = "cuda")]
mod gated_delta;
#[cfg(feature = "cuda")]
mod state_graph;

#[cfg(feature = "cuda")]
pub use gated_delta::{
    GatedDeltaProjectedInputs, GatedDeltaProjectedOutputs, gated_delta_projected_step,
};
#[cfg(feature = "cuda")]
pub use graph::{GatedDeltaStepInputs, GatedDeltaStepOutputs, gated_delta_step};
#[cfg(feature = "cuda")]
pub use state_graph::RecurrentStateGraphArena;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_transition_has_exact_delta_rule_result() {
        let output = gated_delta_reference(
            GatedDeltaGeometry {
                key_heads: 1,
                value_heads: 1,
                key_width: 1,
                value_width: 1,
                normalization_epsilon: 0.0,
            },
            GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: 1,
                query: &[1.0],
                key: &[1.0],
                value: &[4.0],
                log_decay: &[0.5_f32.ln()],
                update_gate: &[0.25],
                initial_state: &[2.0],
            },
        )
        .unwrap();
        assert_eq!(output.state.as_ref(), &[1.75]);
        assert_eq!(output.values.as_ref(), &[1.75]);
    }

    #[test]
    fn chunked_reference_continues_from_returned_state() {
        let geometry = GatedDeltaGeometry {
            key_heads: 1,
            value_heads: 1,
            key_width: 2,
            value_width: 2,
            normalization_epsilon: 1e-6,
        };
        let query = [1.0, 2.0, 2.0, 1.0, -1.0, 3.0];
        let key = [2.0, 1.0, 1.0, -2.0, 0.5, 1.0];
        let value = [0.5, 1.0, 2.0, -1.0, 1.5, 0.25];
        let decay = [0.9_f32.ln(), 0.8_f32.ln(), 0.7_f32.ln()];
        let gate = [0.25, 0.5, 0.75];
        let initial = [0.1, 0.2, 0.3, 0.4];
        let whole = gated_delta_reference(
            geometry,
            GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: 3,
                query: &query,
                key: &key,
                value: &value,
                log_decay: &decay,
                update_gate: &gate,
                initial_state: &initial,
            },
        )
        .unwrap();
        let first = gated_delta_reference(
            geometry,
            GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: 1,
                query: &query[..2],
                key: &key[..2],
                value: &value[..2],
                log_decay: &decay[..1],
                update_gate: &gate[..1],
                initial_state: &initial,
            },
        )
        .unwrap();
        let second = gated_delta_reference(
            geometry,
            GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: 2,
                query: &query[2..],
                key: &key[2..],
                value: &value[2..],
                log_decay: &decay[1..],
                update_gate: &gate[1..],
                initial_state: &first.state,
            },
        )
        .unwrap();
        let joined = first
            .values
            .iter()
            .chain(second.values.iter())
            .copied()
            .collect::<Vec<_>>();
        assert_close(&joined, &whole.values);
        assert_close(&second.state, &whole.state);
    }

    #[test]
    fn grouped_query_heads_feed_multiple_value_state_heads() {
        let geometry = GatedDeltaGeometry {
            key_heads: 1,
            value_heads: 2,
            key_width: 1,
            value_width: 1,
            normalization_epsilon: 0.0,
        };
        let output = gated_delta_reference(
            geometry,
            GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: 1,
                query: &[1.0],
                key: &[1.0],
                value: &[2.0, 4.0],
                log_decay: &[0.0, 0.0],
                update_gate: &[0.5, 0.25],
                initial_state: &[0.0, 0.0],
            },
        )
        .unwrap();
        assert_eq!(output.state.as_ref(), &[1.0, 1.0]);
        assert_eq!(output.values.as_ref(), &[1.0, 1.0]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn luminal_semantic_step_matches_independent_reference() {
        use luminal::prelude::{CompileOptions, Graph, ReferenceRuntime, Runtime};

        let geometry = GatedDeltaGeometry {
            key_heads: 2,
            value_heads: 2,
            key_width: 2,
            value_width: 2,
            normalization_epsilon: 1e-6,
        };
        let query = [1.0, 2.0, -1.0, 0.5, 2.0, 1.0, 0.25, -0.5];
        let key = [2.0, 1.0, 0.5, 1.5, -1.0, 2.0, 1.0, 0.25];
        let value = [0.5, 1.0, 2.0, -1.0, 1.5, 0.25, -0.5, 2.0];
        let log_decay = [0.9_f32.ln(), 0.8_f32.ln(), 0.7_f32.ln(), 0.6_f32.ln()];
        let update_gate = [0.25, 0.5, 0.75, 0.4];
        let initial_state = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.2, 0.4, 0.6, 0.8,
        ];
        let expected = gated_delta_reference(
            geometry,
            GatedDeltaReferenceInput {
                batch_size: 2,
                sequence_tokens: 1,
                query: &query,
                key: &key,
                value: &value,
                log_decay: &log_decay,
                update_gate: &update_gate,
                initial_state: &initial_state,
            },
        )
        .unwrap();

        let mut graph = Graph::new();
        let batch = luminal::prelude::Expression::from('b');
        let q = graph.named_tensor("query", (batch, 2, 2));
        let k = graph.named_tensor("key", (batch, 2, 2));
        let v = graph.named_tensor("value", (batch, 2, 2));
        let g = graph.named_tensor("log_decay", (batch, 2));
        let beta = graph.named_tensor("update_gate", (batch, 2));
        let state = graph.named_tensor("previous_state", (batch, 2, 2, 2));
        let outputs = gated_delta_step(
            GatedDeltaStepInputs {
                query: q,
                key: k,
                value: v,
                log_decay: g,
                update_gate: beta,
                previous_state: state,
                batch_size: batch,
            },
            geometry,
        )
        .unwrap();
        let values = outputs.values.output();
        let next_state = outputs.next_state.output();
        graph.set_dim('b', 2);
        let mut runtime = graph.compile(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        runtime.set_data(q, &query);
        runtime.set_data(k, &key);
        runtime.set_data(v, &value);
        runtime.set_data(g, &log_decay);
        runtime.set_data(beta, &update_gate);
        runtime.set_data(state, &initial_state);
        runtime.execute(&graph.dyn_map);

        assert_close(runtime.get_f32(values), &expected.values);
        assert_close(runtime.get_f32(next_state), &expected.state);

        graph.build_search_space::<luminal_cuda_lite::runtime::CudaRuntime>(
            CompileOptions::default(),
        );
        assert!(
            egraph_has_kernel(&graph, "KernelDeltaStateUpdate"),
            "the complete recurrence graph must expose the in-place CUDA state candidate",
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn luminal_grouped_heads_match_independent_reference() {
        use luminal::prelude::{CompileOptions, Graph, ReferenceRuntime, Runtime};

        let geometry = GatedDeltaGeometry {
            key_heads: 1,
            value_heads: 2,
            key_width: 2,
            value_width: 1,
            normalization_epsilon: 1e-6,
        };
        let query = [1.0, 2.0];
        let key = [2.0, 1.0];
        let value = [0.5, 1.5];
        let log_decay = [0.9_f32.ln(), 0.8_f32.ln()];
        let update_gate = [0.25, 0.75];
        let initial_state = [0.1, 0.2, 0.3, 0.4];
        let expected = gated_delta_reference(
            geometry,
            GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: 1,
                query: &query,
                key: &key,
                value: &value,
                log_decay: &log_decay,
                update_gate: &update_gate,
                initial_state: &initial_state,
            },
        )
        .unwrap();

        let mut graph = Graph::new();
        let q = graph.named_tensor("query", (1, 1, 2));
        let k = graph.named_tensor("key", (1, 1, 2));
        let v = graph.named_tensor("value", (1, 2, 1));
        let g = graph.named_tensor("log_decay", (1, 2));
        let beta = graph.named_tensor("update_gate", (1, 2));
        let state = graph.named_tensor("previous_state", (1, 2, 2, 1));
        let outputs = gated_delta_step(
            GatedDeltaStepInputs {
                query: q,
                key: k,
                value: v,
                log_decay: g,
                update_gate: beta,
                previous_state: state,
                batch_size: 1.into(),
            },
            geometry,
        )
        .unwrap();
        let values = outputs.values.output();
        let next_state = outputs.next_state.output();
        let mut runtime = graph.compile(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        runtime.set_data(q, &query);
        runtime.set_data(k, &key);
        runtime.set_data(v, &value);
        runtime.set_data(g, &log_decay);
        runtime.set_data(beta, &update_gate);
        runtime.set_data(state, &initial_state);
        runtime.execute(&graph.dyn_map);

        assert_close(runtime.get_f32(values), &expected.values);
        assert_close(runtime.get_f32(next_state), &expected.state);

        graph.build_search_space::<luminal_cuda_lite::runtime::CudaRuntime>(
            CompileOptions::default(),
        );
        assert!(egraph_has_kernel(&graph, "KernelDeltaStateUpdate"));
    }

    #[cfg(feature = "cuda")]
    fn egraph_has_kernel(graph: &luminal::prelude::Graph, kind: &str) -> bool {
        let egraph = graph.egraph().expect("CUDA search space");
        egraph.eclasses.values().any(|(sort, nodes)| {
            sort == "IR"
                && nodes.iter().any(|node| {
                    let Some(("Op", children)) = egraph
                        .enodes
                        .get(node)
                        .map(|(label, children)| (label.as_str(), children))
                    else {
                        return false;
                    };
                    children.first().is_some_and(|kind_class| {
                        egraph.eclasses[kind_class]
                            .1
                            .iter()
                            .any(|kind_node| egraph.enodes[kind_node].0 == kind)
                    })
                })
        })
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }
}
