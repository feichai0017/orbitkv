use super::*;
#[cfg(feature = "cuda")]
use luminal::prelude::{DType, Graph, bf16};

#[test]
fn reference_keeps_only_future_visible_history() {
    let geometry = CausalConvolutionGeometry {
        channels: 2,
        kernel_width: 3,
    };
    let output = causal_convolution_reference(
        geometry,
        &[5.0, 6.0],
        &[1.0, 2.0, 3.0, -1.0, 0.5, 2.0],
        &[1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    assert_eq!(output.history.as_ref(), &[2.0, 5.0, 4.0, 6.0]);
    let expected = [20.0_f32, 11.0].map(silu);
    for (&actual, expected) in output.values.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-6);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn luminal_step_matches_minimal_history_reference() {
    use luminal::prelude::{CompileOptions, Graph, ReferenceRuntime, Runtime};

    let geometry = CausalConvolutionGeometry {
        channels: 2,
        kernel_width: 3,
    };
    let input_data = [5.0_f32, 6.0];
    let weight_data = [1.0_f32, 2.0, 3.0, -1.0, 0.5, 2.0];
    let history_data = [1.0_f32, 2.0, 3.0, 4.0];
    let expected =
        causal_convolution_reference(geometry, &input_data, &weight_data, &history_data).unwrap();
    let mut graph = Graph::new();
    let input = graph.named_tensor("input", (1, 2));
    let weights = graph.named_tensor("weights", (2, 3));
    let history = graph.named_tensor("history", (1, 2, 2));
    let output = causal_convolution_step(
        CausalConvolutionStepInputs {
            input,
            weights,
            previous_history: history,
            batch_size: 1.into(),
        },
        geometry,
    )
    .unwrap();
    let values = output.values.output();
    let next = output.next_history.output();
    let mut runtime = graph.compile(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    runtime.set_data(input, input_data.to_vec());
    runtime.set_data(weights, weight_data.to_vec());
    runtime.set_data(history, history_data.to_vec());
    runtime.execute(&graph.dyn_map);
    for (&actual, &expected) in runtime.get_f32(values).iter().zip(expected.values.iter()) {
        assert!((actual - expected).abs() <= 1e-6);
    }
    assert_eq!(runtime.get_f32(next).as_slice(), expected.history.as_ref());
}

#[cfg(feature = "cuda")]
#[test]
fn convolution_arena_uses_bf16_minimal_history_and_manifest_order() {
    use luminal::prelude::{CompileOptions, ReferenceRuntime, Runtime};

    let class = crate::FixedStateClass {
        state_id: 5,
        name: "convolution".into(),
        layers: vec![3, 7].into_boxed_slice(),
        storage: crate::FixedStateStorage::Convolution {
            bytes_per_layer: 8,
            kernel_width: 3,
            slots_per_request: 2,
            bytes_per_request: 32,
        },
    };
    let registration = crate::FixedStateArenaRegistration {
        state_id: 5,
        engine_epoch: 1,
        pool_epoch: 2,
        pool_id: 3,
        slot_count: 4,
        slot_bytes: 16,
    };
    let geometry = CausalConvolutionGeometry {
        channels: 2,
        kernel_width: 3,
    };
    let mut graph = Graph::new();
    let mut arena =
        ConvolutionStateGraphArena::new(&mut graph, &class, registration, 2.into()).unwrap();
    let selected = arena.layer_state(7, geometry).unwrap();
    let observed = selected.cast(DType::F32).output();
    arena.commit_layer(7, geometry, selected).unwrap();
    let binding = arena.finish();
    let mut runtime = graph.compile(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    runtime.set_data(
        binding.arena_input,
        (0..32)
            .map(|value| bf16::from_f32(f32::from(u16::try_from(value).unwrap())))
            .collect::<Vec<_>>(),
    );
    runtime.set_data(binding.destination_slots, vec![2_i32, 0]);
    runtime.execute(&graph.dyn_map);
    assert_eq!(
        runtime.get_f32(observed),
        &vec![20.0, 21.0, 22.0, 23.0, 4.0, 5.0, 6.0, 7.0]
    );
}

#[cfg(feature = "cuda")]
#[test]
fn convolution_arena_retains_a_materialized_commit_candidate() {
    use luminal::prelude::CompileOptions;
    use luminal_cuda_lite::runtime::CudaRuntime;

    let class = crate::FixedStateClass {
        state_id: 5,
        name: "convolution".into(),
        layers: vec![3].into_boxed_slice(),
        storage: crate::FixedStateStorage::Convolution {
            bytes_per_layer: 8,
            kernel_width: 3,
            slots_per_request: 2,
            bytes_per_request: 16,
        },
    };
    let registration = crate::FixedStateArenaRegistration {
        state_id: 5,
        engine_epoch: 1,
        pool_epoch: 2,
        pool_id: 3,
        slot_count: 4,
        slot_bytes: 8,
    };
    let geometry = CausalConvolutionGeometry {
        channels: 2,
        kernel_width: 3,
    };
    let mut graph = Graph::new();
    let mut arena =
        ConvolutionStateGraphArena::new(&mut graph, &class, registration, 1.into()).unwrap();
    let selected = arena.layer_state(3, geometry).unwrap();
    arena
        .commit_layer(3, geometry, selected + selected)
        .unwrap();
    let _binding = arena.finish();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(egraph_has_kernel(&graph, "KernelScatter"));
    assert!(!egraph_has_kernel(&graph, "KernelScatterNoCopy"));
}

#[cfg(feature = "cuda")]
fn egraph_has_kernel(graph: &Graph, kind: &str) -> bool {
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
