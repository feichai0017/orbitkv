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
fn compiler_semantic_step_matches_independent_reference() {
    use orbitkv_compiler::prelude::{CompileOptions, Graph, ReferenceRuntime, Runtime};

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
    let batch = orbitkv_compiler::prelude::Expression::from('b');
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

    graph.build_search_space::<orbitkv_cuda::runtime::CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_kernel(&graph, "KernelDeltaStateUpdate"),
        "the complete recurrence graph must expose the in-place CUDA state candidate",
    );
}

#[cfg(feature = "cuda")]
#[test]
fn compiler_grouped_heads_match_independent_reference() {
    use orbitkv_compiler::prelude::{CompileOptions, Graph, ReferenceRuntime, Runtime};

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

    graph.build_search_space::<orbitkv_cuda::runtime::CudaRuntime>(CompileOptions::default());
    assert!(egraph_has_kernel(&graph, "KernelDeltaStateUpdate"));
}

#[cfg(feature = "cuda")]
fn egraph_has_kernel(graph: &orbitkv_compiler::prelude::Graph, kind: &str) -> bool {
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
