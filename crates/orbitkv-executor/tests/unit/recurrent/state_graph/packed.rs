use super::*;
use orbitkv_compiler::prelude::DType;
use orbitkv_cuda::kernel::sequence_state::{
    PackedDeltaScanPlan, PackedDeltaScanSpec, packed_delta_scan,
};

#[test]
fn packed_delta_reads_use_the_manifest_arena_and_preserve_commit_candidates() {
    let mut graph = Graph::new();
    graph.set_dim_interval('b', 1, 3);
    graph.set_dim_interval('s', 1, 8);
    graph.set_dim('b', 2);
    graph.set_dim('s', 5);
    let geometry = GatedDeltaGeometry {
        key_heads: 1,
        value_heads: 1,
        key_width: 2,
        value_width: 2,
        normalization_epsilon: 1e-6,
    };
    let mut arena =
        RecurrentStateGraphArena::new(&mut graph, &class(), registration(), 'b'.into()).unwrap();
    // The second read uses the first commit's logical arena version. Layer 7
    // also has a nonzero offset within each manifest-defined state slot.
    for layer in [3, 7] {
        let previous = arena.layer_state(layer, geometry).unwrap();
        let output = packed_delta_scan(
            PackedDeltaScanPlan {
                query: graph.tensor(('s', 1, 2)),
                key: graph.tensor(('s', 1, 2)),
                value: graph.tensor(('s', 1, 2)),
                log_decay: graph.tensor(('s', 1)),
                update_gate: graph.tensor(('s', 1)),
                state: previous,
                query_indptr: graph.tensor(Expression::from('b') + 1).as_dtype(DType::Int),
            },
            PackedDeltaScanSpec {
                key_heads: geometry.key_heads,
                value_heads: geometry.value_heads,
                key_width: geometry.key_width,
                value_width: geometry.value_width,
                normalization_epsilon: geometry.normalization_epsilon,
                round_normalized_qk_to_bf16: true,
                round_final_state_to_bf16: false,
            },
        );
        output.values.output();
        arena.commit_layer(layer, geometry, output.state).unwrap();
    }
    let binding = arena.finish();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(egraph_has_kernel(&graph, "KernelDeltaGather"));
    assert!(egraph_has_kernel(&graph, "KernelScatterNoCopy"));
    assert_ne!(binding.arena_input.id, binding.arena_output.id);
}
