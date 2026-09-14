use orbitkv_compiler::prelude::*;

use crate::{
    runtime::CudaRuntime,
    tests::utilities::{
        ForcedExtractionConfig, assert_close, extract_forced_kernel_llir, get_cuda_stream,
        llir_kernel_names, op_ir_nodes, try_extract_forced_op_llir_where,
    },
};

struct StateGraph {
    graph: Graph,
    state: GraphTensor,
    decay: GraphTensor,
    key: GraphTensor,
    delta: GraphTensor,
    next_state: GraphTensor,
}

fn state_graph() -> StateGraph {
    let mut graph = Graph::new();
    let batch = Expression::from('b');
    let state = graph.named_tensor("state", (batch, 2, 2, 2)).persist();
    let decay = graph.named_tensor("decay", (batch, 2));
    let key = graph.named_tensor("key", (batch, 2, 2));
    let delta = graph.named_tensor("delta", (batch, 2, 2));
    let decayed = state * decay.expand_dim(2, 2).expand_dim(3, 2);
    let update = key.expand_dim(3, 2) * delta.expand_dim(2, 2);
    let next_state = decayed + update;
    let next_state = next_state.output();
    graph.set_dim('b', 2);
    StateGraph {
        graph,
        state,
        decay,
        key,
        delta,
        next_state,
    }
}

fn dependent_state_graph() -> StateGraph {
    let mut graph = Graph::new();
    let batch = Expression::from('b');
    let state = graph.named_tensor("state", (batch, 2, 2, 2)).persist();
    let decay = graph.named_tensor("decay", (batch, 2));
    let key = graph.named_tensor("key", (batch, 2, 2));
    let value = graph.named_tensor("value", (batch, 2, 2));
    let gate = graph.named_tensor("gate", (batch, 2));
    let decayed = state * decay.expand_dim(2, 2).expand_dim(3, 2);
    let key_matrix = key.expand_dim(3, 2);
    let memory = (decayed * key_matrix).sum(2);
    let delta = (value - memory) * gate.expand_dim(2, 2);
    let next_state = (decayed + key_matrix * delta.expand_dim(2, 2)).output();
    graph.set_dim('b', 2);
    StateGraph {
        graph,
        state,
        decay,
        key,
        delta,
        next_state,
    }
}

fn slotted_state_graph() -> StateGraph {
    let mut graph = Graph::new();
    let batch = Expression::from('b');
    let arena = graph.named_tensor("arena", 32).persist();
    let slots = graph.named_tensor("slots", batch).as_dtype(DType::Int);
    let decay = graph.named_tensor("decay", (batch, 2));
    let key = graph.named_tensor("key", (batch, 2, 2));
    let delta = graph.named_tensor("delta", (batch, 2, 2));
    let local = graph.iota('z', (2, 2, 2)).expand_dim(0, batch);
    let base = (slots * 8)
        .expand_dim(1, 2)
        .expand_dim(2, 2)
        .expand_dim(3, 2);
    let indices = local + base;
    let state = arena.gather(indices);
    let decayed = state * decay.expand_dim(2, 2).expand_dim(3, 2);
    let update = key.expand_dim(3, 2) * delta.expand_dim(2, 2);
    let next_state = decayed + update;
    let committed = next_state.scatter(indices, arena).output();
    graph.set_dim('b', 2);
    StateGraph {
        graph,
        state: arena,
        decay,
        key,
        delta,
        next_state: committed,
    }
}

#[test]
fn delta_state_update_is_an_egglog_candidate() {
    let mut model = state_graph();
    model
        .graph
        .build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        !op_ir_nodes(
            model.graph.egraph().expect("built search space"),
            "KernelDeltaStateUpdate",
        )
        .is_empty(),
        "the full decayed rank-one state equation should admit the CUDA candidate",
    );
    let llir = extract_forced_kernel_llir(
        &model.graph,
        "KernelDeltaStateUpdate",
        "DeltaStateUpdate",
        ForcedExtractionConfig::new(0xDE17_A57E).attempts_per_node(64),
        true,
    );
    assert!(llir.node_weights().any(|op| {
        op.to_dialect::<dyn crate::kernel::KernelOp>()
            .is_some_and(|kernel| kernel.kernel_name() == "DeltaStateUpdate")
    }));
}

#[test]
fn incomplete_state_equation_does_not_admit_candidate() {
    let mut graph = Graph::new();
    let state = graph.tensor((2, 2, 2, 2));
    let update = graph.tensor((2, 2, 2, 2));
    (state + update).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        op_ir_nodes(
            graph.egraph().expect("built search space"),
            "KernelDeltaStateUpdate"
        )
        .is_empty(),
        "a bare state addition is not the complete update equation",
    );
}

#[test]
fn dependent_delta_can_read_old_state_before_in_place_update() {
    let mut model = dependent_state_graph();
    model
        .graph
        .build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(
        &model.graph,
        "KernelDeltaStateUpdate",
        "DeltaStateUpdate",
        ForcedExtractionConfig::new(0xD3E3_DA7A)
            .attempts_per_node(256)
            .node_seed_stride(32),
        false,
    );
    crate::resource::plan_static_llir_resources(&llir, &model.graph.dyn_map)
        .expect("an ordered old-state read must remain legal before mutation");
    assert!(llir.node_weights().any(|op| {
        op.to_dialect::<dyn crate::kernel::KernelOp>()
            .is_some_and(|kernel| kernel.kernel_name() == "DeltaStateUpdate")
    }));
}

#[test]
fn slotted_arena_can_select_in_place_update_and_commit() {
    let mut model = slotted_state_graph();
    model
        .graph
        .build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = try_extract_forced_op_llir_where(
        &model.graph,
        &["KernelScatterNoCopy"],
        ForcedExtractionConfig::new(0x5107_A4E0)
            .attempts_per_node(512)
            .node_seed_stride(64),
        |llir| {
            let names = llir_kernel_names(llir);
            names.contains(&"ScatterNoCopy") && names.contains(&"DeltaStateUpdate")
        },
    )
    .expect("slot arena should expose a joint update-and-commit candidate");
    crate::resource::plan_static_llir_resources(&llir, &model.graph.dyn_map)
        .expect("joint candidate must satisfy alias and resource contracts");
}

#[test]
#[ignore = "requires a CUDA device"]
fn delta_state_update_cuda_matches_reference_expression() {
    let stream = get_cuda_stream().expect("CUDA device");
    let mut model = state_graph();
    model
        .graph
        .build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(
        &model.graph,
        "KernelDeltaStateUpdate",
        "DeltaStateUpdate",
        ForcedExtractionConfig::new(0xDE17_A57E).attempts_per_node(64),
        true,
    );
    let state = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.2, 0.4, 0.6, 0.8,
    ];
    let decay = vec![0.9, 0.8, 0.7, 0.6];
    let key = vec![1.0, 2.0, -1.0, 0.5, 2.0, 1.0, 0.25, -0.5];
    let delta = vec![0.5, 1.0, 2.0, -1.0, 1.5, 0.25, -0.5, 2.0];
    let expected_state = reference_update(&state, &decay, &key, &delta);

    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_llir(&llir);
    runtime.set_data(model.state, state);
    runtime.set_data(model.decay, decay);
    runtime.set_data(model.key, key);
    runtime.set_data(model.delta, delta);
    runtime.execute(&model.graph.dyn_map);

    assert_close(
        &runtime.get_f32(model.next_state),
        &expected_state,
        1e-6,
        1e-6,
    );
}

fn reference_update(state: &[f32], decay: &[f32], key: &[f32], delta: &[f32]) -> Vec<f32> {
    let mut output = state.to_vec();
    for batch in 0..2 {
        for head in 0..2 {
            let bh = batch * 2 + head;
            for key_index in 0..2 {
                for value_index in 0..2 {
                    let index = ((bh * 2 + key_index) * 2) + value_index;
                    output[index] = state[index] * decay[bh]
                        + key[bh * 2 + key_index] * delta[bh * 2 + value_index];
                }
            }
        }
    }
    output
}
