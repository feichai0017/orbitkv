use luminal::prelude::{CompileOptions, ReferenceRuntime, Runtime};

use super::*;
use crate::model::{
    DecoderActivation, DecoderBlockLayout, DecoderLayerKind, DecoderNormWeights,
    DecoderWeightFormat,
};
use crate::{
    AttentionClass, AttentionVisibility, CausalConvolutionGeometry, FixedStateClass,
    FixedStateStorage, GatedDeltaReferenceInput, causal_convolution_reference,
    gated_delta_reference,
};

#[test]
fn decode_core_updates_convolution_and_recurrent_state() {
    let mut graph = Graph::new();
    let config = config();
    let core = GatedDeltaCore::new(&mut graph, &config, 0, DType::F32).unwrap();
    let hidden = graph.named_tensor("hidden", (1, 4));
    let recurrent_state = graph.named_tensor("recurrent_state", (1, 2, 2, 1));
    let convolution_history = graph.named_tensor("convolution_history", (1, 6, 2));
    let output = core
        .forward_decode(&hidden, &recurrent_state, &convolution_history, 1.into())
        .unwrap();
    let hidden_out = output.hidden.output();
    let recurrent_out = output.next_recurrent_state.output();
    let convolution_out = output.next_convolution_history.output();
    let mut runtime = graph.compile(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    runtime.set_data(hidden, vec![1.0_f32, 2.0, 3.0, 4.0]);
    runtime.set_data(core.input_qkv.weight, projection_weights());
    runtime.set_data(
        core.input_z.weight,
        vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    );
    runtime.set_data(core.input_b, vec![0.0_f32; 8]);
    runtime.set_data(core.input_a, vec![0.0_f32; 8]);
    runtime.set_data(core.convolution_weight, vec![1.0_f32; 18]);
    runtime.set_data(core.decay_log_rates, vec![0.0_f32; 2]);
    runtime.set_data(core.decay_bias, vec![0.0_f32; 2]);
    runtime.set_data(core.output_norm, vec![1.0_f32]);
    runtime.set_data(
        core.output.weight,
        vec![1.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    );
    runtime.set_data(recurrent_state, vec![0.0_f32; 4]);
    runtime.set_data(convolution_history, vec![0.0_f32; 12]);
    runtime.execute(&graph.dyn_map);

    let convolution = causal_convolution_reference(
        CausalConvolutionGeometry {
            channels: 6,
            kernel_width: 3,
        },
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
        &[1.0; 18],
        &[0.0; 12],
    )
    .unwrap();
    let expected_recurrent = gated_delta_reference(
        GatedDeltaGeometry {
            key_heads: 1,
            value_heads: 2,
            key_width: 2,
            value_width: 1,
            normalization_epsilon: 1e-6,
        },
        GatedDeltaReferenceInput {
            batch_size: 1,
            sequence_tokens: 1,
            query: &convolution.values[..2],
            key: &convolution.values[2..4],
            value: &convolution.values[4..],
            log_decay: &[-std::f32::consts::LN_2; 2],
            update_gate: &[0.5; 2],
            initial_state: &[0.0; 4],
        },
    )
    .unwrap();
    let output_gate = 1.0_f32 / (1.0 + (-1.0_f32).exp());
    let normalized = expected_recurrent
        .values
        .iter()
        .map(|value| value / (value * value + 1e-6).sqrt() * output_gate)
        .collect::<Vec<_>>();
    let combined = normalized[0] + normalized[1];
    let expected_hidden = [combined, 0.0, 0.0, combined];
    assert_close(runtime.get_f32(hidden_out), &expected_hidden);
    assert_close(runtime.get_f32(recurrent_out), &expected_recurrent.state);
    assert_eq!(
        runtime.get_f32(convolution_out).as_slice(),
        convolution.history.as_ref()
    );
}

#[test]
fn state_graph_composes_both_manager_owned_state_classes() {
    use luminal_cuda_lite::runtime::CudaRuntime;
    use orbitkv::RecurrentFamily;

    let config = config();
    let plan = crate::tests::support::executor_plan_with_fixed_states(
        "stateful",
        16,
        vec![AttentionClass {
            class_id: 0,
            name: "full".into(),
            layers: vec![1].into_boxed_slice(),
            page_tokens: 16,
            key_bytes_per_token_per_layer: 128,
            value_bytes_per_token_per_layer: 128,
            visibility: AttentionVisibility::Full,
        }],
        vec![
            FixedStateClass {
                state_id: 1,
                name: "recurrent".into(),
                layers: vec![0].into_boxed_slice(),
                storage: FixedStateStorage::Recurrent {
                    family: RecurrentFamily::Gdn,
                    bytes_per_layer: 16,
                    slots_per_request: 2,
                    bytes_per_request: 32,
                },
            },
            FixedStateClass {
                state_id: 2,
                name: "convolution".into(),
                layers: vec![0].into_boxed_slice(),
                storage: FixedStateStorage::Convolution {
                    bytes_per_layer: 24,
                    kernel_width: 3,
                    slots_per_request: 2,
                    bytes_per_request: 48,
                },
            },
        ],
    );
    let registrations = [
        FixedStateArenaRegistration {
            state_id: 1,
            engine_epoch: 1,
            pool_epoch: 2,
            pool_id: 3,
            slot_count: 4,
            slot_bytes: 16,
        },
        FixedStateArenaRegistration {
            state_id: 2,
            engine_epoch: 1,
            pool_epoch: 3,
            pool_id: 4,
            slot_count: 4,
            slot_bytes: 24,
        },
    ];
    let mut graph = Graph::new();
    let mut states =
        GatedDeltaStateGraph::new(&mut graph, &config, &plan, &registrations, 'b'.into()).unwrap();
    let hidden = graph.named_tensor("hidden", ('b', 4)).as_dtype(DType::Bf16);
    let query_indptr = graph
        .named_tensor("query_indptr", Expression::from('b') + 1)
        .as_dtype(DType::Int);
    let output = states
        .apply_packed_core(&mut graph, &config, 0, &hidden, query_indptr)
        .unwrap()
        .output();
    let bindings = states.finish();
    assert_eq!(bindings.recurrent.state_id, 1);
    assert_eq!(bindings.convolution.state_id, 2);
    assert_eq!(
        GatedDeltaStateBindings::write_policies(),
        [
            crate::FixedStateWritePolicy::RequiredInPlace,
            crate::FixedStateWritePolicy::RequiredInPlace,
        ]
    );
    assert_eq!(
        bindings.graph_bindings().map(|binding| binding.state_id),
        [1, 2]
    );
    assert_eq!(output.dtype, DType::Bf16);
    graph.set_dim('b', 1);
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(egraph_has_kernel(&graph, "CustomOpKind"));
    assert_eq!(egraph_kernel_count(&graph, "KernelScatterNoCopy"), 2);
}

fn projection_weights() -> Vec<f32> {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
}

fn egraph_has_kernel(graph: &Graph, kind: &str) -> bool {
    egraph_kernel_count(graph, kind) != 0
}

fn egraph_kernel_count(graph: &Graph, kind: &str) -> usize {
    let egraph = graph.egraph().expect("CUDA search space");
    egraph
        .eclasses
        .values()
        .filter(|(sort, nodes)| {
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
        .count()
}

fn config() -> DecoderConfig {
    DecoderConfig {
        layers: 2,
        hidden_size: 4,
        intermediate_size: 8,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 64,
        vocabulary_size: 16,
        tensor_prefix: "model".into(),
        rope_theta: 10_000.0,
        rotary_dimensions: 64,
        rms_epsilon: 1e-6,
        tied_embeddings: true,
        embedding_scale: 1.0,
        activation: DecoderActivation::Silu,
        block_layout: DecoderBlockLayout::PreNorm,
        norm_weights: DecoderNormWeights::Direct,
        local_rope_theta: None,
        attention_softmax_scale: 64_f64.sqrt().recip(),
        attention_output_gate: false,
        layer_kinds: Some(
            vec![DecoderLayerKind::Linear, DecoderLayerKind::Full].into_boxed_slice(),
        ),
        gated_delta: Some(GatedDeltaConfig {
            key_heads: 1,
            value_heads: 2,
            key_width: 2,
            value_width: 1,
            convolution_kernel_width: 3,
        }),
        weight_format: DecoderWeightFormat::Float,
    }
}
