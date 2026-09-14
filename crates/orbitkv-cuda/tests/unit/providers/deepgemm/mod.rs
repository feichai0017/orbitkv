use super::*;
use half::bf16;
use orbitkv_compiler::{
    op::{CustomOp, LLIROp, Runtime},
    prelude::{CompileOptions, Graph},
};

mod capture;
mod reference;
use reference::BlockScaledLinearReference;

#[derive(Debug)]
struct DirectReference(BlockScaledLinearReference);

impl CustomOp for DirectReference {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn HostOp>(Box::new(DirectReferenceHost(self.0.clone())) as Box<dyn HostOp>)
    }
}

#[derive(Debug)]
struct DirectReferenceHost(BlockScaledLinearReference);

impl EgglogOp for DirectReferenceHost {
    fn sort(&self) -> SortDef {
        self.0.sort()
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for DirectReferenceHost {
    fn deployment_eligible(&self) -> bool {
        true
    }

    fn prepare_compilation(
        &self,
        stream: &Arc<CudaStream>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        self.0.prepare_compilation(stream, dyn_map)
    }

    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        self.0.execute(stream, self_node, inputs, buffers, dyn_map)
    }

    fn output_size(&self) -> Expression {
        self.0.output_size()
    }

    fn output_bytes(&self) -> Expression {
        self.0.output_bytes()
    }

    fn output_dtype(&self) -> DType {
        self.0.output_dtype()
    }

    fn cuda_graph_capture_arity(&self) -> Option<usize> {
        self.0.cuda_graph_capture_arity()
    }

    fn cuda_graph_capture_dyn_dims(&self) -> Vec<orbitkv_compiler::prelude::Symbol> {
        self.0.cuda_graph_capture_dyn_dims()
    }

    fn prepare_cuda_graph_capture(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        self.0
            .prepare_cuda_graph_capture(stream, self_node, inputs, buffers, dyn_map)
    }

    fn cuda_graph_capture_resources(&self) -> Vec<super::super::CudaGraphCaptureResource> {
        self.0.cuda_graph_capture_resources()
    }

    fn device_memory_plan(
        &self,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        self.0
            .device_memory_plan(self_node, inputs, buffer_lengths, dyn_map)
    }

    fn resource_buffer_nodes(&self, inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        self.0.resource_buffer_nodes(inputs)
    }
}

fn graph(
    rows: usize,
    output_features: usize,
    input_features: usize,
) -> (Graph, GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let mut graph = Graph::default();
    let input = graph.tensor((rows, input_features)).as_dtype(DType::Bf16);
    let weight = graph
        .tensor((output_features, input_features))
        .as_dtype(DType::F8E4M3);
    let weight_scale = graph.tensor((
        output_features.div_ceil(BLOCK),
        input_features.div_ceil(BLOCK),
    ));
    let output = block_scaled_linear(
        input,
        weight,
        weight_scale,
        BlockScaledLinearSpec {
            rows: rows.into(),
            output_features,
            input_features,
            weight_block_rows: BLOCK,
            weight_block_columns: BLOCK,
        },
    )
    .output();
    (graph, input, weight, weight_scale, output)
}

#[test]
fn deepgemm_candidates_share_reference_eclass() {
    let (mut graph, _, _, _, _) = graph(1, 128, 128);
    for (target, expected) in [
        (None, 0),
        (Some(crate::target::CudaTarget { major: 8, minor: 0 }), 0),
        (
            Some(crate::target::CudaTarget { major: 9, minor: 0 }),
            jit::SEARCH_VARIANTS,
        ),
        (
            Some(crate::target::CudaTarget {
                major: 10,
                minor: 0,
            }),
            0,
        ),
    ] {
        graph.build_search_space::<crate::runtime::CudaRuntime>(
            CompileOptions::default().compiler_facts(
                target.map_or_else(String::new, crate::target::CudaTarget::compiler_facts),
            ),
        );
        let count = graph
            .egraph()
            .unwrap()
            .enodes
            .values()
            .filter(|(label, _)| label == "DeepGemm")
            .count();
        assert_eq!(count, expected, "execution target {target:?}");
    }
}

#[test]
fn semantic_reference_is_not_deployment_eligible() {
    let reference = BlockScaledLinearReference::new(BlockScaledLinearSpec {
        rows: 1.into(),
        output_features: 128,
        input_features: 128,
        weight_block_rows: BLOCK,
        weight_block_columns: BLOCK,
    });
    assert!(!reference.deployment_eligible());
    assert!(DirectReferenceHost(reference).deployment_eligible());
}

#[test]
#[ignore = "requires an SM90 GPU and a prefetched DeepGEMM provider"]
fn production_search_rejects_the_portable_reference() {
    let Ok(context) = crate::cudarc::driver::CudaContext::new(0) else {
        return;
    };
    if context.compute_capability().ok() != Some((9, 0)) {
        return;
    }
    let stream = context.default_stream();
    let (mut graph, input, weight, scale, _) = graph(1, 128, 128);
    let mut runtime = crate::runtime::CudaRuntime::initialize(stream);
    runtime.set_data(input, vec![bf16::from_f32(1.0); 128]);
    runtime.set_data(weight, vec![0x38_u8; 128 * 128]);
    runtime.set_data(scale, vec![1.0_f32]);
    graph.build_search_space::<crate::runtime::CudaRuntime>(
        CompileOptions::default().compiler_facts(runtime.compilation_facts()),
    );
    let _runtime = graph.search(runtime, CompileOptions::default().search_graph_limit(8));
    let schedule = serde_json::to_string(graph.selected_schedule().unwrap()).unwrap();
    assert!(schedule.contains("deepgemm@"));
    assert!(!schedule.contains("BlockScaledLinearReference"));
}

#[test]
#[ignore = "requires an SM90 GPU and nvcc; compiles the pinned DeepGEMM provider"]
fn deepgemm_candidate_executes_on_sm90() {
    let Ok(context) = crate::cudarc::driver::CudaContext::new(0) else {
        return;
    };
    if context.compute_capability().ok() != Some((9, 0)) {
        return;
    }
    let stream = context.default_stream();
    let mut graph = Graph::default();
    let input = graph.tensor((1usize, 128usize)).as_dtype(DType::Bf16);
    let weight = graph.tensor((128usize, 128usize)).as_dtype(DType::F8E4M3);
    let scale = graph.tensor((1usize, 1usize));
    let output = graph
        .custom_op(
            DeepGemm {
                rows: 1.into(),
                output_features: 128,
                input_features: 128,
                variant: 0,
                provider: jit::provider_identity().unwrap(),
                scratch: Arc::new(Mutex::new(None)),
            },
            vec![input, weight, scale],
            (1usize, 128usize),
            DType::Bf16,
        )
        .output();
    let mut runtime = crate::runtime::CudaRuntime::initialize(stream);
    runtime.set_data(input, vec![bf16::from_f32(1.0); 128]);
    runtime.set_data(weight, vec![0x38_u8; 128 * 128]);
    runtime.set_data(scale, vec![1.0_f32]);
    runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
    runtime.execute(&graph.dyn_map);
    assert_eq!(runtime.get_bf16(output).first().unwrap().to_f32(), 128.0);
}

#[test]
#[ignore = "requires an SM90 GPU and nvcc; compares independent and DeepGEMM providers"]
fn deepgemm_matches_independent_reference_on_sm90() {
    let Ok(context) = crate::cudarc::driver::CudaContext::new(0) else {
        return;
    };
    if context.compute_capability().ok() != Some((9, 0)) {
        return;
    }
    let stream = context.default_stream();
    let (m, n, k) = (16usize, 256usize, 256usize);
    let spec = BlockScaledLinearSpec {
        rows: m.into(),
        output_features: n,
        input_features: k,
        weight_block_rows: BLOCK,
        weight_block_columns: BLOCK,
    };
    let mut graph = Graph::default();
    let input = graph.tensor((m, k)).as_dtype(DType::Bf16);
    let weight = graph.tensor((n, k)).as_dtype(DType::F8E4M3);
    let scale = graph.tensor((2usize, 2usize));
    let reference = graph
        .custom_op(
            DirectReference(BlockScaledLinearReference::new(spec)),
            vec![input, weight, scale],
            (m, n),
            DType::Bf16,
        )
        .output();
    let deepgemm = graph
        .custom_op(
            DeepGemm {
                rows: m.into(),
                output_features: n,
                input_features: k,
                variant: 0,
                provider: jit::provider_identity().unwrap(),
                scratch: Arc::new(Mutex::new(None)),
            },
            vec![input, weight, scale],
            (m, n),
            DType::Bf16,
        )
        .output();
    let input_values = (0..m * k)
        .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 16.0))
        .collect::<Vec<_>>();
    let weight_values = (0..n * k)
        .map(|index| if index % 3 == 0 { 0xb0_u8 } else { 0x30_u8 })
        .collect::<Vec<_>>();
    let mut runtime = crate::runtime::CudaRuntime::initialize(stream);
    runtime.set_data(input, input_values);
    runtime.set_data(weight, weight_values);
    runtime.set_data(scale, vec![0.25_f32, 0.5, 0.75, 1.0]);
    runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
    runtime.execute(&graph.dyn_map);
    let reference = runtime.get_bf16(reference);
    let actual = runtime.get_bf16(deepgemm);
    let maximum_error = reference
        .iter()
        .zip(actual)
        .map(|(reference, actual)| (reference.to_f32() - actual.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum_error <= 0.25, "maximum error was {maximum_error}");
}

#[test]
#[ignore = "requires an SM90 GPU and nvcc; profiles the provider search space"]
fn search_selects_a_valid_provider_and_preserves_numerics_on_sm90() {
    let Ok(context) = crate::cudarc::driver::CudaContext::new(0) else {
        return;
    };
    if context.compute_capability().ok() != Some((9, 0)) {
        return;
    }
    let stream = context.default_stream();
    let (mut graph, input, weight, scale, output) = graph(16, 5120, 5120);
    let mut runtime = crate::runtime::CudaRuntime::initialize(stream);
    runtime.set_data(input, vec![bf16::from_f32(1.0); 16 * 5120]);
    runtime.set_data(weight, vec![0x38_u8; 5120 * 5120]);
    runtime.set_data(scale, vec![1.0_f32; 40 * 40]);
    graph.build_search_space::<crate::runtime::CudaRuntime>(
        CompileOptions::default().compiler_facts(runtime.compilation_facts()),
    );
    runtime = graph.search(runtime, CompileOptions::default().search_graph_limit(8));
    runtime.execute(&graph.dyn_map);
    let values = runtime.get_bf16(output);
    assert_eq!(values.len(), 16 * 5120);
    assert!(values.iter().all(|value| value.to_f32() == 5120.0));
    let schedule = serde_json::to_string(graph.selected_schedule().unwrap()).unwrap();
    assert!(schedule.contains(&jit::provider_identity().unwrap()));
}

#[test]
#[ignore = "requires an SM90 GPU and nvcc; validates large production DeepGEMM shapes"]
fn deepgemm_large_shapes_match_across_all_variants_on_sm90() {
    let Ok(context) = crate::cudarc::driver::CudaContext::new(0) else {
        return;
    };
    if context.compute_capability().ok() != Some((9, 0)) {
        return;
    }
    for &(m, n, k) in &[
        (1usize, 5_120usize, 5_120usize),
        (1, 17_408, 5_120),
        (1, 5_120, 17_408),
        (4, 12_288, 5_120),
        (4, 5_120, 6_144),
    ] {
        let mut reference = None;
        for variant in 0..jit::SEARCH_VARIANTS {
            let stream = context.default_stream();
            let mut graph = Graph::default();
            let input = graph.tensor((m, k)).as_dtype(DType::Bf16);
            let weight = graph.tensor((n, k)).as_dtype(DType::F8E4M3);
            let scale = graph.tensor((n.div_ceil(BLOCK), k.div_ceil(BLOCK)));
            let output = graph
                .custom_op(
                    DeepGemm {
                        rows: m.into(),
                        output_features: n,
                        input_features: k,
                        variant,
                        provider: jit::provider_identity().unwrap(),
                        scratch: Arc::new(Mutex::new(None)),
                    },
                    vec![input, weight, scale],
                    (m, n),
                    DType::Bf16,
                )
                .output();
            let input_values = (0..m * k)
                .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 16.0))
                .collect::<Vec<_>>();
            let weight_values = (0..n * k)
                .map(|index| if index % 3 == 0 { 0xb0_u8 } else { 0x30_u8 })
                .collect::<Vec<_>>();
            let scale_values = (0..n.div_ceil(BLOCK) * k.div_ceil(BLOCK))
                .map(|index| 0.25 + (index % 4) as f32 * 0.25)
                .collect::<Vec<_>>();
            let mut runtime = crate::runtime::CudaRuntime::initialize(stream);
            runtime.set_data(input, input_values);
            runtime.set_data(weight, weight_values);
            runtime.set_data(scale, scale_values);
            runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
            runtime.execute(&graph.dyn_map);
            let values = runtime.get_bf16(output);
            assert!(values.iter().any(|value| value.to_f32() != 0.0));
            if variant == 0 {
                reference = Some(values);
            } else {
                let maximum_error = reference
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(&values)
                    .map(|(reference, actual)| (reference.to_f32() - actual.to_f32()).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    maximum_error <= 0.5,
                    "DeepGEMM variant {variant} differs from variant 0 for M={m}, N={n}, K={k}: max_abs={maximum_error}"
                );
            }
        }
    }
}

impl<const PREQUANTIZED: bool> CustomOp for DeepGemmImpl<PREQUANTIZED> {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn HostOp>(Box::new(self.clone()) as Box<dyn HostOp>)
    }
}
