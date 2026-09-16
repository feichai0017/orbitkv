use super::*;
use half::bf16;
use orbitkv_compiler::prelude::*;

use crate::{
    runtime::CudaRuntime,
    tests::utilities::{ForcedExtractionConfig, extract_forced_kernel_llir, op_ir_nodes},
};

mod device;

const EXTRACTION: ForcedExtractionConfig = ForcedExtractionConfig::new(0x0051_1A17)
    .attempts_per_node(64)
    .node_seed_stride(64);

fn activation_product(gate: GraphTensor, up: GraphTensor) -> GraphTensor {
    let activated = gate
        .cast(DType::F32)
        .swish()
        .cast(DType::Bf16)
        .cast(DType::F32);
    (activated * up.cast(DType::F32)).cast(DType::Bf16)
}

fn contains_candidate(graph: &mut Graph) -> bool {
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    !op_ir_nodes(graph.egraph().unwrap(), "KernelSiluMul").is_empty()
}

fn inputs(graph: &mut Graph, rows: Expression, width: usize) -> (GraphTensor, GraphTensor) {
    (
        graph.tensor((rows, width)).as_dtype(DType::Bf16),
        graph.tensor((rows, width)).as_dtype(DType::Bf16),
    )
}

#[test]
fn contiguous_bf16_inputs_admit_a_reachable_vector_candidate() {
    for rows in [1.into(), 3.into(), 's'.into()] {
        let mut graph = Graph::new();
        graph.set_dim_interval('s', 1, 1_000_000);
        let (gate, up) = inputs(&mut graph, rows, VECTOR_ELEMENTS * 2);
        activation_product(gate, up).output();
        let matched = contains_candidate(&mut graph);
        assert!(matched, "rows={rows}");
        extract_forced_kernel_llir(&graph, "KernelSiluMul", "SiluMul", EXTRACTION, false);
    }
}

#[test]
fn widths_without_full_vectors_and_different_dtypes_are_rejected() {
    for width in [1, VECTOR_ELEMENTS - 1, VECTOR_ELEMENTS + 1] {
        let mut graph = Graph::new();
        let (gate, up) = inputs(&mut graph, 3.into(), width);
        activation_product(gate, up).output();
        assert!(!contains_candidate(&mut graph), "width={width}");
    }
    for dtype in [DType::F16, DType::F32] {
        let mut graph = Graph::new();
        let gate = graph.tensor((3, VECTOR_ELEMENTS)).as_dtype(dtype);
        let up = graph.tensor((3, VECTOR_ELEMENTS)).as_dtype(dtype);
        activation_product(gate, up).output();
        assert!(!contains_candidate(&mut graph), "dtype={dtype:?}");
    }
}

#[test]
fn dynamic_rows_without_a_positive_span_proof_are_rejected() {
    for lower_bound in [None, Some(0)] {
        let mut graph = Graph::new();
        if let Some(lower_bound) = lower_bound {
            graph.set_dim_interval('s', lower_bound, 32);
        }
        let (gate, up) = inputs(&mut graph, 's'.into(), VECTOR_ELEMENTS);
        activation_product(gate, up).output();
        assert!(
            !contains_candidate(&mut graph),
            "lower bound={lower_bound:?}"
        );
    }
}

#[test]
fn transposed_sliced_and_broadcast_inputs_are_rejected() {
    for variant in 0..4 {
        let mut graph = Graph::new();
        let (gate, up) = inputs(&mut graph, VECTOR_ELEMENTS.into(), VECTOR_ELEMENTS * 2);
        let gate = gate.slice((.., ..VECTOR_ELEMENTS));
        let up = up.slice((.., ..VECTOR_ELEMENTS));
        let (gate, up) = match variant {
            0 => (gate, up), // Row pitch is twice the logical width.
            1 => (gate.t(), up),
            2 => (gate, up.t()),
            3 => (
                gate.slice((..1, ..))
                    .squeeze(0)
                    .expand_dim(0, VECTOR_ELEMENTS),
                up,
            ),
            _ => unreachable!(),
        };
        activation_product(gate, up).output();
        assert!(!contains_candidate(&mut graph), "view variant={variant}");
    }
}

#[test]
fn activation_rounding_is_part_of_the_match() {
    let mut graph = Graph::new();
    let (gate, up) = inputs(&mut graph, 3.into(), VECTOR_ELEMENTS);
    (gate.cast(DType::F32).swish() * up.cast(DType::F32))
        .cast(DType::Bf16)
        .output();
    assert!(!contains_candidate(&mut graph));
}

#[test]
fn modified_activation_constants_are_rejected() {
    let mut graph = Graph::new();
    let (gate, up) = inputs(&mut graph, 3.into(), VECTOR_ELEMENTS);
    let gate = gate.cast(DType::F32);
    let activated = (gate * (1.0 + (-gate * 1.5).exp2()).reciprocal())
        .cast(DType::Bf16)
        .cast(DType::F32);
    (activated * up.cast(DType::F32)).cast(DType::Bf16).output();
    assert!(!contains_candidate(&mut graph));
}

// Independent evaluation of the frontend's F32 primitive semantics. Device
// comparisons use the unfused GPU graph to avoid treating libm and libdevice
// transcendental approximations as bitwise equivalent.
fn reference(gate: bf16, up: bf16) -> bf16 {
    let gate = gate.to_f32();
    let sigmoid = (1.0 + (-gate * std::f32::consts::LOG2_E).exp2()).recip();
    let activated = bf16::from_f32(gate * sigmoid);
    bf16::from_f32(activated.to_f32() * up.to_f32())
}

#[test]
fn independent_reference_exposes_the_intermediate_rounding_boundary() {
    let gate = bf16::from_f32(2.0);
    let up = bf16::from_bits(0x3e83);
    let rounded = reference(gate, up);
    let store_once = bf16::from_f32(gate.to_f32() / (1.0 + (-gate.to_f32()).exp()) * up.to_f32());
    assert_eq!(rounded.to_bits(), 0x3ee6);
    assert_eq!(store_once.to_bits(), 0x3ee7);
}

#[test]
fn dynamic_vector_count_widens_before_multiplying() {
    let kernel = KernelSiluMul {
        size: Expression::from('s') * 65_536,
    };
    assert!(kernel.source().contains("static_cast<long long>(const_s)"));
    let dimensions = [(Symbol::from('s'), 1_000_000)].into_iter().collect();
    assert_eq!(
        (kernel.size / VECTOR_ELEMENTS).exec(&dimensions),
        Some(8_192_000_000)
    );
}
