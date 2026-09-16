use super::*;
use cudarc::driver::CudaContext;
use half::bf16;

// Independent scalar evaluation of the frontend's BF16 primitives. In
// particular, this is not a widened mathematical SiLU reference.
fn decomposed_reference(gate: bf16, up: bf16) -> bf16 {
    let round = bf16::from_f32;
    let negative = round(-gate.to_f32());
    let log2e = round(std::f32::consts::LOG2_E);
    let scaled = round(negative.to_f32() * log2e.to_f32());
    let exponential = round(scaled.to_f32().exp2());
    let denominator = round(exponential.to_f32() + 1.0);
    let sigmoid = round(denominator.to_f32().recip());
    let activated = round(gate.to_f32() * sigmoid.to_f32());
    round(activated.to_f32() * up.to_f32())
}

fn store_once_reference(gate: bf16, up: bf16) -> bf16 {
    let gate = gate.to_f32();
    bf16::from_f32(gate / (1.0 + (-gate).exp()) * up.to_f32())
}

#[test]
fn reference_distinguishes_the_two_arithmetic_contracts() {
    let gate = bf16::from_f32(2.0);
    let up = bf16::from_f32(1.0);
    assert_eq!(decomposed_reference(gate, up).to_f32(), 1.765625);
    assert_eq!(store_once_reference(gate, up).to_f32(), 1.7578125);
}

#[test]
#[ignore = "requires CUDA; forces the matched BF16 candidate"]
fn matched_kernel_preserves_each_bf16_operation() {
    let mut graph = Graph::new();
    let (input, output) = decomposed(&mut graph, 3.into(), 2 * WIDTH);
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(
        &graph,
        "KernelSwiglu",
        "SwigluBf16Decomposed",
        EXTRACTION,
        true,
    );
    let mut values = Vec::new();
    let mut expected = Vec::new();
    for (gate, up) in [(2.0, 1.0), (-2.0, 1.0), (1.5, 0.75)] {
        let gate = bf16::from_f32(gate);
        let up = bf16::from_f32(up);
        values.extend([gate; WIDTH]);
        values.extend([up; WIDTH]);
        expected.extend([decomposed_reference(gate, up); WIDTH]);
    }
    let context = CudaContext::new(0).unwrap();
    let mut runtime = CudaRuntime::initialize(context.new_stream().unwrap());
    runtime.set_data(input, values);
    runtime.load_llir(&llir);
    runtime.execute(&graph.dyn_map);
    assert_eq!(runtime.get_bf16(output.id), expected);
}

#[test]
#[ignore = "requires CUDA; exercises custom operation input materialization"]
fn custom_operation_preserves_store_once_arithmetic_for_strided_input() {
    let mut graph = Graph::new();
    let input = graph.tensor((3, 3 * WIDTH)).as_dtype(DType::Bf16);
    let output = fused_swiglu(input.slice((.., ..2 * WIDTH)), WIDTH).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut values = Vec::new();
    let mut expected = Vec::new();
    for (gate, up) in [(2.0, 1.0), (-2.0, 1.0), (1.5, 0.75)] {
        let gate = bf16::from_f32(gate);
        let up = bf16::from_f32(up);
        values.extend([gate; WIDTH]);
        values.extend([up; WIDTH]);
        values.extend([bf16::from_f32(19.0); WIDTH]);
        expected.extend([store_once_reference(gate, up); WIDTH]);
    }
    let context = CudaContext::new(0).unwrap();
    let mut runtime = CudaRuntime::initialize(context.new_stream().unwrap());
    runtime.set_data(input, values.clone());
    runtime = graph.search(runtime, CompileOptions::default().search_graph_limit(1));
    runtime.set_data(input, values);
    runtime.execute(&graph.dyn_map);
    assert_eq!(runtime.get_bf16(output.id), expected);
}
