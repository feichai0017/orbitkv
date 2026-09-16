use super::*;
use as_any::Downcast;
use orbitkv_compiler::{hlir::Gather, prelude::*};

use crate::{
    runtime::CudaRuntime,
    tests::utilities::{ForcedExtractionConfig, extract_forced_kernel_llir, op_ir_nodes},
};

mod device;

const WIDTH: usize = 8;
const EXTRACTION: ForcedExtractionConfig = ForcedExtractionConfig::new(0x0057_191A)
    .attempts_per_node(64)
    .node_seed_stride(64);

fn contains_swiglu(graph: &Graph) -> bool {
    !op_ir_nodes(graph.egraph().unwrap(), "KernelSwiglu").is_empty()
}

fn decomposed(graph: &mut Graph, rows: Expression, pitch: usize) -> (GraphTensor, GraphTensor) {
    let input = graph.tensor((rows, pitch)).as_dtype(DType::Bf16);
    let gate = input.slice((.., ..WIDTH));
    let up = input.slice((.., WIDTH..2 * WIDTH));
    (input, (gate.swish() * up).output())
}

#[test]
fn bf16_dense_rows_admit_the_exact_decomposition() {
    for rows in [3.into(), 's'.into()] {
        let mut graph = Graph::new();
        decomposed(&mut graph, rows, 2 * WIDTH);
        graph.build_search_space::<CudaRuntime>(CompileOptions::default());
        assert!(contains_swiglu(&graph), "dense rows {rows} did not match");
        let llir = extract_forced_kernel_llir(
            &graph,
            "KernelSwiglu",
            "SwigluBf16Decomposed",
            EXTRACTION,
            false,
        );
        assert!(llir.node_weights().any(|operation| {
            operation
                .to_dialect::<dyn KernelOp>()
                .and_then(|kernel| (***kernel).downcast_ref::<SwigluKernel>())
                .is_some_and(|kernel| kernel.arithmetic == SwigluArithmetic::Bf16Decomposed)
        }));
    }
}

#[test]
fn wider_storage_rows_do_not_admit_dense_swiglu() {
    let mut graph = Graph::new();
    decomposed(&mut graph, 3.into(), 3 * WIDTH);
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(!contains_swiglu(&graph));
}

#[test]
fn transposed_intermediate_does_not_admit_dense_swiglu() {
    let mut graph = Graph::new();
    let input = graph.tensor((WIDTH, 2 * WIDTH)).as_dtype(DType::Bf16);
    let gate = input.slice((.., ..WIDTH));
    let up = input.slice((.., WIDTH..));
    (gate.swish().t() * up).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(!contains_swiglu(&graph));
}

#[test]
fn widened_activation_does_not_admit_bf16_decomposition() {
    let mut graph = Graph::new();
    let input = graph.tensor((3, 2 * WIDTH)).as_dtype(DType::Bf16);
    let gate = input.slice((.., ..WIDTH)).cast(DType::F32);
    let up = input.slice((.., WIDTH..)).cast(DType::F32);
    ((gate.swish().cast(DType::Bf16).cast(DType::F32) * up).cast(DType::Bf16)).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(!contains_swiglu(&graph));
}

#[test]
fn custom_operation_materializes_strided_views_and_retains_its_arithmetic() {
    let mut graph = Graph::new();
    let input = graph.tensor((3, 3 * WIDTH)).as_dtype(DType::Bf16);
    let view = input.slice((.., ..2 * WIDTH));
    let output = fused_swiglu(view, WIDTH);
    let materialized = graph.get_sources(output.id)[0];
    assert_ne!(materialized, input.id);
    let gather = graph.get_op::<Gather>(materialized);
    assert_eq!(gather.input_shapes[1], view.shape);
    let llir = graph.custom_ops.last().unwrap().to_llir_op();
    let kernel = llir.to_dialect::<dyn KernelOp>().unwrap();
    assert_eq!(
        (***kernel)
            .downcast_ref::<SwigluKernel>()
            .unwrap()
            .arithmetic,
        SwigluArithmetic::StoreOnce,
    );
}

#[test]
fn custom_operation_keeps_contiguous_inputs_without_a_copy() {
    let mut graph = Graph::new();
    let input = graph.tensor((3, 2 * WIDTH)).as_dtype(DType::Bf16);
    let output = fused_swiglu(input, WIDTH);
    assert_eq!(graph.get_sources(output.id), [input.id]);
}
