use orbitkv_compiler::{op::CustomOp, prelude::*};

use super::{convolution::PackedConvolutionKernel, delta_scan::PackedDeltaScanKernel, *};
use crate::{
    kernel::KernelOp,
    resource::plan_static_llir_resources,
    runtime::CudaRuntime,
    tests::utilities::{
        ForcedExtractionConfig, llir_kernel_names, try_extract_forced_op_llir_where,
    },
};

fn convolution_spec() -> PackedConvolutionSpec {
    PackedConvolutionSpec {
        channels: 6,
        kernel_width: 3,
    }
}

fn delta_spec() -> PackedDeltaScanSpec {
    PackedDeltaScanSpec {
        key_heads: 1,
        value_heads: 2,
        key_width: 2,
        value_width: 1,
        normalization_epsilon: 1e-6,
    }
}

fn scan_plan(graph: &mut Graph, state: GraphTensor, indptr: GraphTensor) -> PackedDeltaScanPlan {
    PackedDeltaScanPlan {
        query: graph.named_tensor("query", ('s', 1, 2)),
        key: graph.named_tensor("key", ('s', 1, 2)),
        value: graph.named_tensor("value", ('s', 2, 1)),
        log_decay: graph.named_tensor("decay", ('s', 2)),
        update_gate: graph.named_tensor("gate", ('s', 2)),
        state,
        query_indptr: indptr,
    }
}

#[test]
fn packed_shapes_and_input_order_are_explicit() {
    let mut graph = Graph::new();
    let indptr = graph
        .named_tensor("indptr", Expression::from('b') + 1)
        .as_dtype(DType::Int);
    let history = graph
        .named_tensor("history", ('b', 6, 2))
        .as_dtype(DType::Bf16);
    let convolution = packed_causal_convolution(
        PackedConvolutionPlan {
            input: graph.named_tensor("input", ('s', 6)).as_dtype(DType::Bf16),
            weights: graph.named_tensor("weights", (6, 3)).as_dtype(DType::Bf16),
            history,
            query_indptr: indptr,
        },
        convolution_spec(),
    );
    assert_eq!(convolution.values.dims(), [Expression::from('s'), 6.into()]);
    assert_eq!(
        convolution.history.dims(),
        [Expression::from('b'), 6.into(), 2.into()]
    );
    let state = graph.named_tensor("state", ('b', 2, 2, 1));
    let plan = scan_plan(&mut graph, state, indptr);
    let output = packed_delta_scan(plan, delta_spec());
    assert_eq!(
        output.values.dims(),
        [Expression::from('s'), 2.into(), 1.into()]
    );
    assert_eq!(
        output.state.dims(),
        [Expression::from('b'), 2.into(), 2.into(), 1.into()]
    );
    let packed_node = graph.get_sources(output.values.id)[1];
    assert_eq!(
        graph.get_sources(packed_node),
        [
            plan.query.id,
            plan.key.id,
            plan.value.id,
            plan.log_decay.id,
            plan.update_gate.id,
            state.id,
            indptr.id,
        ]
    );
}

#[test]
fn strided_query_is_materialized_at_the_custom_op_boundary() {
    let mut graph = Graph::new();
    let indptr = graph
        .named_tensor("indptr", Expression::from('b') + 1)
        .as_dtype(DType::Int);
    let state = graph.named_tensor("state", ('b', 2, 2, 1));
    let mut plan = scan_plan(&mut graph, state, indptr);
    plan.query = graph
        .named_tensor("strided_query", (2, 's', 1))
        .permute((1, 2, 0));
    let source = plan.query;
    let output = packed_delta_scan(plan, delta_spec());
    let packed_node = graph.get_sources(output.values.id)[1];
    let query_node = graph.get_sources(packed_node)[0];
    assert_ne!(query_node, source.id);
    assert!(
        graph
            .try_get_op::<orbitkv_compiler::hlir::Gather>(query_node)
            .is_some()
    );
}

#[test]
fn production_sources_compile_with_static_and_dynamic_geometry() {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    for (tokens, requests) in [(5.into(), 2.into()), ('s'.into(), 'b'.into())] {
        let convolution = PackedConvolutionKernel {
            tokens,
            requests,
            spec: convolution_spec(),
        };
        let scan = PackedDeltaScanKernel {
            tokens,
            requests,
            spec: delta_spec(),
        };
        assert_eq!(convolution.compiler_facts(4), "(packed-convolution-op 4)");
        assert_eq!(scan.compiler_facts(7), "(packed-delta-scan-op 7)");
        for (source, entry, input_count, op) in [
            (
                convolution.source(),
                "packed_causal_convolution",
                4,
                &convolution as &dyn KernelOp,
            ),
            (
                scan.source(),
                "packed_delta_scan",
                7,
                &scan as &dyn KernelOp,
            ),
        ] {
            assert!(!source.contains('@'));
            let dynamic = !op.all_dyn_vars().is_empty();
            assert_eq!(
                op.kernel_parameter_count(input_count, dynamic),
                input_count + 1 + usize::from(dynamic)
            );
            let ptx = compile_ptx_with_opts(
                source,
                CompileOptions {
                    arch: Some("compute_90"),
                    include_paths: vec!["/usr/local/cuda/include".into()],
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(ptx.to_src().contains(entry));
        }
    }
}

fn slot_indices(graph: &mut Graph, slots: GraphTensor, shape: &[usize]) -> GraphTensor {
    let size: usize = shape.iter().product();
    let mut base = slots * size;
    for (axis, &width) in shape.iter().enumerate() {
        base = base.expand_dim(axis + 1, width);
    }
    graph.iota('z', shape).expand_dim(0, 'b') + base
}

fn slotted_graph(observe_previous: bool) -> Graph {
    let mut graph = Graph::new();
    let slots = graph.named_tensor("slots", 'b').as_dtype(DType::Int);
    let indptr = graph
        .named_tensor("indptr", Expression::from('b') + 1)
        .as_dtype(DType::Int);
    let history = graph
        .named_tensor("history_arena", 48)
        .persist()
        .as_dtype(DType::Bf16);
    let state = graph.named_tensor("state_arena", 16).persist();
    let history_indices = slot_indices(&mut graph, slots, &[6, 2]);
    let state_indices = slot_indices(&mut graph, slots, &[2, 2, 1]);
    let convolution = packed_causal_convolution(
        PackedConvolutionPlan {
            input: graph.named_tensor("input", ('s', 6)).as_dtype(DType::Bf16),
            weights: graph.named_tensor("weights", (6, 3)).as_dtype(DType::Bf16),
            history: history.gather(history_indices),
            query_indptr: indptr,
        },
        convolution_spec(),
    );
    let plan = scan_plan(&mut graph, state.gather(state_indices), indptr);
    let scan = packed_delta_scan(plan, delta_spec());
    convolution.values.output();
    scan.values.output();
    convolution
        .history
        .scatter(history_indices, history)
        .output();
    scan.state.scatter(state_indices, state).output();
    if observe_previous {
        state.output();
        history.output();
    }
    graph.set_dim('s', 5);
    graph.set_dim('b', 2);
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    graph
}

fn joint_no_copy(graph: &Graph) -> LLIRGraph {
    try_extract_forced_op_llir_where(
        graph,
        &["KernelScatterNoCopy"],
        ForcedExtractionConfig::new(0xCA51_50A7).attempts_per_node(512),
        |llir| {
            llir_kernel_names(llir)
                .iter()
                .filter(|&&name| name == "ScatterNoCopy")
                .count()
                == 2
        },
    )
    .expect("both state commits must be reachable in one extracted program")
}

#[test]
fn both_state_commits_pass_candidate_local_alias_validation() {
    let graph = slotted_graph(false);
    let llir = joint_no_copy(&graph);
    plan_static_llir_resources(&llir, &graph.dyn_map).unwrap();
}

#[test]
fn observing_old_state_rejects_mutating_commit_candidate() {
    let graph = slotted_graph(true);
    let llir = joint_no_copy(&graph);
    assert!(plan_static_llir_resources(&llir, &graph.dyn_map).is_err());
}
