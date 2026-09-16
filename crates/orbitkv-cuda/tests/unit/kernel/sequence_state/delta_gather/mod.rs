use super::*;
use as_any::Downcast;
use orbitkv_compiler::prelude::petgraph::{Direction, visit::EdgeRef};

use crate::{
    kernel::sequence_state::{PackedDeltaScanPlan, PackedDeltaScanSpec, packed_delta_scan},
    resource::plan_static_llir_resources,
    runtime::CudaRuntime,
    tests::utilities::{
        ForcedExtractionConfig, llir_kernel_names, op_ir_nodes, try_extract_forced_op_llir_where,
    },
};

mod commit;
mod device;

const EXTRACTION: ForcedExtractionConfig = ForcedExtractionConfig::new(0xD317_A45E)
    .attempts_per_node(256)
    .node_seed_stride(256);

fn spec(width: usize) -> PackedDeltaScanSpec {
    PackedDeltaScanSpec {
        key_heads: 1,
        value_heads: 2,
        key_width: width,
        value_width: 3,
        normalization_epsilon: 1e-6,
        round_normalized_qk_to_bf16: true,
        round_final_state_to_bf16: true,
    }
}

fn scan(
    graph: &mut Graph,
    state: GraphTensor,
    spec: PackedDeltaScanSpec,
) -> super::super::PackedDeltaScanOutput {
    let requests = state.dims()[0];
    packed_delta_scan(
        PackedDeltaScanPlan {
            query: graph.tensor((5, spec.key_heads, spec.key_width)),
            key: graph.tensor((5, spec.key_heads, spec.key_width)),
            value: graph.tensor((5, spec.value_heads, spec.value_width)),
            log_decay: graph.tensor((5, spec.value_heads)),
            update_gate: graph.tensor((5, spec.value_heads)),
            state,
            query_indptr: graph.tensor(requests + 1).as_dtype(DType::Int),
        },
        spec,
    )
}

fn has_candidate(graph: &mut Graph) -> bool {
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    !op_ir_nodes(graph.egraph().unwrap(), "KernelDeltaGather").is_empty()
}

fn extract(graph: &Graph, with_commit: bool) -> LLIRGraph {
    try_extract_forced_op_llir_where(graph, &["KernelDeltaGather"], EXTRACTION, |llir| {
        let names = llir_kernel_names(llir);
        names.contains(&"PackedDeltaGather") && (!with_commit || names.contains(&"ScatterNoCopy"))
    })
    .expect("the requested gather scan program must remain reachable")
}

#[test]
fn gathered_state_is_reachable_without_a_materialization_input() {
    for width in [1, 128, 129] {
        let mut graph = Graph::new();
        let spec = spec(width);
        let arena = graph.tensor(8 * spec.value_heads * width * spec.value_width);
        let indices = graph
            .tensor((2, spec.value_heads, width, spec.value_width))
            .as_dtype(DType::Int);
        let outputs = scan(&mut graph, arena.gather(indices), spec);
        outputs.values.output();
        outputs.state.output();
        assert_eq!(has_candidate(&mut graph), width <= 128);
        if width > 128 {
            continue;
        }
        let llir = extract(&graph, false);
        let node = llir
            .node_indices()
            .find(|&node| {
                llir[node]
                    .to_dialect::<dyn KernelOp>()
                    .is_some_and(|kernel| kernel.kernel_name() == "PackedDeltaGather")
            })
            .unwrap();
        let mut edges = llir
            .edges_directed(node, Direction::Incoming)
            .collect::<Vec<_>>();
        edges.sort_by_key(|edge| edge.id());
        assert_eq!(edges.len(), 8);
        let arena_input = edges[5].source();
        assert!(
            llir[arena_input].to_dialect::<dyn KernelOp>().is_none(),
            "arena must be read directly from its input"
        );
        let kernel = llir[node].to_dialect::<dyn KernelOp>().unwrap();
        let kernel = (***kernel).downcast_ref::<KernelDeltaGather>().unwrap();
        assert_eq!(kernel.output_aliases_input(), None);
        assert_eq!(
            kernel.output_size().to_usize(),
            Some((5 + 2 * width) * 2 * 3)
        );
        plan_static_llir_resources(&llir, &graph.dyn_map).unwrap();
    }
}

#[test]
fn noncontiguous_arena_and_indices_keep_their_addressing() {
    let mut graph = Graph::new();
    let spec = spec(2);
    let arena = graph.tensor((32, 4)).slice((.., ..2)).permute((1, 0));
    let indices = graph
        .tensor((2, 2, 2, 3, 2))
        .as_dtype(DType::Int)
        .slice((.., .., .., .., ..1))
        .squeeze(4)
        .permute((1, 0, 2, 3));
    let output = scan(&mut graph, arena.gather(indices), spec);
    output.values.output();
    output.state.output();
    assert!(has_candidate(&mut graph));
    let llir = extract(&graph, false);
    let candidate = llir
        .node_weights()
        .filter_map(|op| op.to_dialect::<dyn KernelOp>())
        .find_map(|kernel| (***kernel).downcast_ref::<KernelDeltaGather>())
        .unwrap();
    assert_eq!(candidate.index_shape, indices.dims());
    assert_eq!(
        candidate.index_strides,
        indices
            .shape
            .strides
            .iter()
            .map(|stride| stride.simplify())
            .collect::<Vec<_>>()
    );
    assert_eq!(candidate.data_shape, arena.dims());
    assert_eq!(
        candidate.data_strides,
        arena
            .shape
            .strides
            .iter()
            .map(|stride| stride.simplify())
            .collect::<Vec<_>>()
    );
}

#[test]
fn different_gather_geometry_and_converted_state_are_not_direct_candidates() {
    for converted in [false, true] {
        let mut graph = Graph::new();
        let spec = spec(2);
        if converted {
            let arena = graph.tensor(96).as_dtype(DType::Bf16);
            let indices = graph.tensor((2, 2, 2, 3)).as_dtype(DType::Int);
            scan(&mut graph, arena.gather(indices).cast(DType::F32), spec)
                .values
                .output();
        } else {
            let arena = graph.tensor(96);
            let indices = graph.tensor((2, 2, 3, 2)).as_dtype(DType::Int);
            let state = arena.gather(indices);
            // Bypass the frontend shape assertion to exercise the backend's
            // exact geometry proof, including equal-size permuted dimensions.
            let kernel = PackedDeltaScanKernel {
                tokens: 5.into(),
                requests: 2.into(),
                spec,
            };
            let inputs = vec![
                graph.tensor((5, 1, 2)),
                graph.tensor((5, 1, 2)),
                graph.tensor((5, 2, 3)),
                graph.tensor((5, 2)),
                graph.tensor((5, 2)),
                state,
                graph.tensor(3).as_dtype(DType::Int),
            ];
            let size = kernel.output_size();
            graph.custom_op(kernel, inputs, size, DType::F32).output();
        }
        assert!(!has_candidate(&mut graph), "converted={converted}");
    }
}

#[test]
fn index_dtype_must_be_proven_by_the_graph() {
    let mut graph = Graph::new();
    let arena = graph.tensor(96);
    let indices = graph.tensor((2, 2, 2, 3)).as_dtype(DType::Int);
    scan(&mut graph, arena.gather(indices), spec(2))
        .values
        .output();
    // Deliberately invalidate an already built frontend graph to test backend
    // admission independently of gather()'s earlier integer dtype assertion.
    indices.as_dtype(DType::F32);
    assert!(!has_candidate(&mut graph));
}

fn committed_graph(observe_previous: bool) -> Graph {
    let mut graph = Graph::new();
    let arena = graph.tensor(96).persist();
    let indices = graph.tensor((2, 2, 2, 3)).as_dtype(DType::Int);
    let output = scan(&mut graph, arena.gather(indices), spec(2));
    output.values.output();
    output.state.scatter(indices, arena).output();
    if observe_previous {
        arena.output();
    }
    assert!(has_candidate(&mut graph));
    graph
}

#[test]
fn direct_reads_preserve_candidate_local_commit_hazards() {
    for observe_previous in [false, true] {
        let graph = committed_graph(observe_previous);
        let llir = extract(&graph, true);
        assert_eq!(
            plan_static_llir_resources(&llir, &graph.dyn_map).is_ok(),
            !observe_previous
        );
    }
}

#[test]
fn all_gather_metadata_dimensions_participate_in_the_kernel_abi() {
    let index = Expression::from('z');
    let kernel = KernelDeltaGather {
        scan: PackedDeltaScanKernel {
            tokens: 't'.into(),
            requests: 'b'.into(),
            spec: spec(2),
        },
        index_shape: vec!['b'.into(), 2.into(), 2.into(), 3.into()],
        index_strides: vec![index * Expression::from('p'), index * 6, index * 3, index],
        data_shape: vec!['n'.into(), 5.into()],
        data_strides: vec![index * Expression::from('d'), index],
    };
    assert_eq!(
        kernel.all_dyn_vars(),
        ['t', 'b', 'p', 'n', 'd']
            .map(Symbol::from)
            .into_iter()
            .collect()
    );
    let source = kernel.source();
    assert!(!source.contains('@'));
    assert!(source.contains("const long long gather_index = state_indices["));
    assert!(source.contains("state[width] = state_arena["));
    assert_eq!(kernel.kernel_parameter_count(8, true), 10);
}

#[test]
fn memory_estimates_count_one_state_read_and_one_packed_store() {
    let scan = PackedDeltaScanKernel {
        tokens: 5.into(),
        requests: 2.into(),
        spec: spec(2),
    };
    let kernel = KernelDeltaGather {
        scan: scan.clone(),
        ..Default::default()
    };
    let state_elements = 2 * 2 * 2 * 3;
    let token_reads = 5 * 2 * (2 * 2 + 2 + 3);
    let indptr_reads = 2 * 2 * 2;
    assert_eq!(
        kernel.bytes_loaded().to_usize(),
        Some((state_elements * 2 + token_reads + indptr_reads) * 4)
    );
    assert_eq!(kernel.bytes_stored(), scan.output_bytes());
}
