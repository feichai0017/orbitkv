use luminal::prelude::{CompileOptions, Graph, ReferenceRuntime, Runtime};

#[test]
fn gated_query_projection_deinterleaves_each_head() {
    let mut graph = Graph::new();
    let projected = graph.named_tensor("q_gate", (1, 8));
    let per_head = projected.split_dims(1, 4);
    let query = per_head.slice((.., .., ..2)).merge_dims(1, 2).output();
    let gate = per_head.slice((.., .., 2..)).merge_dims(1, 2).output();
    let mut runtime = graph.compile(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    runtime.set_data(
        projected,
        vec![1.0_f32, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
    );
    runtime.execute(&graph.dyn_map);
    assert_eq!(runtime.get_f32(query), &vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(runtime.get_f32(gate), &vec![10.0, 20.0, 30.0, 40.0]);
}
