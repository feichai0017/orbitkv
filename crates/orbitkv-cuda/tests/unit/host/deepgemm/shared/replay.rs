use super::*;
use crate::runtime::CudaRuntime;

fn build() -> (
    Graph,
    GraphTensor,
    Vec<(GraphTensor, GraphTensor, GraphTensor)>,
) {
    let mut graph = Graph::default();
    let input = graph.tensor((3usize, 256usize)).as_dtype(DType::Bf16);
    let mut bindings = Vec::new();
    for n in [128usize, 256] {
        let weight = graph.tensor((n, 256usize)).as_dtype(DType::F8E4M3);
        let scale = graph.tensor((n / 128, 2usize));
        let output = block_scaled_linear(
            input,
            weight,
            scale,
            BlockScaledLinearSpec {
                rows: 3.into(),
                output_features: n,
                input_features: 256,
                weight_block_rows: BLOCK,
                weight_block_columns: BLOCK,
            },
        )
        .output();
        bindings.push((weight, scale, output));
    }
    (graph, input, bindings)
}

fn set_inputs(
    runtime: &mut CudaRuntime,
    input: GraphTensor,
    bindings: &[(GraphTensor, GraphTensor, GraphTensor)],
) {
    runtime.set_data(input, vec![bf16::from_f32(1.0); 3 * 256]);
    for (index, &(weight, scale, _)) in bindings.iter().enumerate() {
        let n = (index + 1) * 128;
        runtime.set_data(weight, vec![0x38_u8; n * 256]);
        runtime.set_data(scale, vec![1.0_f32; (n / 128) * 2]);
    }
}

#[test]
#[ignore = "requires SM90 GPU; exercises egglog shared-provider extraction and schedule replay"]
fn shared_fp8_semantic_search_and_schedule_replay_on_sm90() {
    let context = crate::cudarc::driver::CudaContext::new(0).expect("SM90 GPU required");
    assert_eq!(context.compute_capability().unwrap(), (9, 0));
    let stream = context.default_stream();
    let (mut graph, input, bindings) = build();
    let options = CompileOptions::default()
        .compiler_facts(SHARED_QUANTIZATION_COMPILER_FACT)
        .search_graph_limit(8);
    let mut runtime = CudaRuntime::initialize(stream.clone());
    set_inputs(&mut runtime, input, &bindings);
    // Explicit ablation excludes only the combined implementation, ensuring
    // this regression exercises the new semantic extraction path. Ordinary
    // production search retains both sets of candidates.
    graph.build_search_space_exclude_ops::<CudaRuntime, DeepGemm>(options.clone());
    assert_eq!(
        count_operations(graph.egraph().unwrap(), "BlockScaledQuantize"),
        1
    );
    runtime = graph.search(runtime, options);
    runtime.execute(&graph.dyn_map);
    for &(_, _, output) in &bindings {
        assert!(
            runtime
                .get_bf16(output)
                .iter()
                .all(|value| value.to_f32() == 256.0)
        );
    }
    let bytes = serde_json::to_vec(graph.selected_schedule().unwrap()).unwrap();
    let (mut loaded, input, bindings) = build();
    loaded.install_selected_schedule(serde_json::from_slice(&bytes).unwrap());
    let mut replay = CudaRuntime::initialize(stream);
    set_inputs(&mut replay, input, &bindings);
    loaded.load_selected_schedule(&mut replay).unwrap();
    replay.execute(&loaded.dyn_map);
    for &(_, _, output) in &bindings {
        assert!(
            replay
                .get_bf16(output)
                .iter()
                .all(|value| value.to_f32() == 256.0)
        );
    }
}
