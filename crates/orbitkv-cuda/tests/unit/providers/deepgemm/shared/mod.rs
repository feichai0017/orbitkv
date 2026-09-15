use super::super::tests::device_selection;
use super::*;
use half::bf16;
use orbitkv_compiler::op::CustomOp;
use orbitkv_compiler::{
    op::Runtime,
    prelude::{CompileOptions, Graph},
};

mod benchmark;
mod replay;
mod selection_rules;

#[derive(Default, Debug)]
struct TestDeepGemm<const PREQUANTIZED: bool>;

impl<const PREQUANTIZED: bool> EgglogOp for TestDeepGemm<PREQUANTIZED> {
    fn sort(&self) -> SortDef {
        DeepGemmImpl::<PREQUANTIZED>::default().sort()
    }
    fn n_inputs(&self) -> usize {
        3
    }
    fn egglog_declarations(&self) -> Vec<String> {
        vec![
            BLOCK_SCALED_LINEAR_DECLARATIONS.to_owned(),
            crate::target::DECLARATIONS.to_owned(),
        ]
    }
    fn egglog_primitives(&self) -> Vec<EgglogPrimitive> {
        DeepGemmImpl::<PREQUANTIZED>::default().egglog_primitives()
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn rewrites(&self) -> Vec<Rule> {
        // Exercise the actual backend rules without requiring a device or a
        // local provider checkout just to construct the e-graph.
        DeepGemmImpl::<PREQUANTIZED>::provider_rewrites("test-provider")
    }
}

struct RewriteRuntime;

impl Runtime for RewriteRuntime {
    type Ops = (TestDeepGemm<false>, TestDeepGemm<true>, BlockScaledQuantize);
    fn extra_egglog() -> String {
        format!(
            "{}\n(set (cuda-target-sm-count) 78)",
            crate::target::CudaTarget { major: 9, minor: 0 }.compiler_facts()
        )
    }
    type CompileArg = ();
    type ExecReturn = ();
    const CLEANUP_HLIR: bool = false;
    fn initialize(_: ()) -> Self {
        Self
    }
    fn compile(
        &mut self,
        _: &orbitkv_compiler::search::SearchSpace,
        _: &orbitkv_compiler::prelude::DynMap,
        _: &CompileOptions,
        _: &mut dyn rand::RngCore,
    ) {
        unreachable!("host-only rewrite test")
    }
    fn load_llir(&mut self, _: &orbitkv_compiler::graph::LLIRGraph) {
        unreachable!()
    }
    fn execute(&mut self, _: &orbitkv_compiler::prelude::DynMap) {
        unreachable!()
    }
}

fn fanout_graph(distinct_input: bool) -> Graph {
    let mut graph = Graph::default();
    let x = graph.tensor((3usize, 256usize)).as_dtype(DType::Bf16);
    for n in [128usize, 256] {
        let input = if distinct_input && n == 256 {
            graph.tensor((3usize, 256usize)).as_dtype(DType::Bf16)
        } else {
            x
        };
        let w = graph.tensor((n, 256usize)).as_dtype(DType::F8E4M3);
        let ws = graph.tensor((n / 128, 2usize));
        block_scaled_linear(
            input,
            w,
            ws,
            BlockScaledLinearSpec {
                rows: 3.into(),
                output_features: n,
                input_features: 256,
                weight_block_rows: BLOCK,
                weight_block_columns: BLOCK,
            },
        )
        .output();
    }
    graph
}

fn op_kinds(
    egraph: &SerializedEGraph,
    children: &[orbitkv_compiler::egglog_utils::ClassId],
) -> Vec<String> {
    egraph.eclasses[&children[0]]
        .1
        .iter()
        .map(|node| egraph.enodes[node].0.clone())
        .collect()
}

fn count_operations(egraph: &SerializedEGraph, name: &str) -> usize {
    egraph
        .enodes
        .values()
        .filter(|(label, children)| {
            label == "Op" && op_kinds(egraph, children).iter().any(|kind| kind == name)
        })
        .count()
}

#[test]
fn shared_quantization_is_opt_in_and_keeps_combined_candidates() {
    for enabled in [false, true] {
        let mut graph = fanout_graph(false);
        graph.build_search_space::<RewriteRuntime>(CompileOptions::default().compiler_facts(
            if enabled {
                SHARED_QUANTIZATION_COMPILER_FACT
            } else {
                ""
            },
        ));
        let egraph = graph.egraph().unwrap();
        assert_eq!(
            count_operations(egraph, "DeepGemm"),
            2 * selection::SEARCH_VARIANTS
        );
        assert_eq!(
            count_operations(egraph, "DeepGemmPrequantized"),
            if enabled {
                2 * selection::SEARCH_VARIANTS
            } else {
                0
            }
        );
        assert_eq!(
            count_operations(egraph, "BlockScaledQuantize"),
            usize::from(enabled)
        );
        if enabled {
            // Each semantic output retains both representations in its own
            // e-class. The shared producer is not unioned with its BF16 input.
            for (label, nodes) in egraph.eclasses.values().filter(|(label, _)| label == "IR") {
                let _ = label;
                let kinds = nodes
                    .iter()
                    .flat_map(|node| {
                        let (label, children) = &egraph.enodes[node];
                        if label == "Op" {
                            op_kinds(egraph, children)
                        } else {
                            vec![]
                        }
                    })
                    .collect::<Vec<_>>();
                if kinds.iter().any(|kind| kind == "DeepGemmPrequantized") {
                    assert!(kinds.iter().any(|kind| kind == "DeepGemm"));
                    assert!(!kinds.iter().any(|kind| kind == "BlockScaledQuantize"));
                }
            }
        }
    }
}

#[test]
fn quantization_shares_by_input_and_contract_not_output_width() {
    for (distinct_input, expected) in [(false, 1), (true, 2)] {
        let mut graph = fanout_graph(distinct_input);
        graph.build_search_space::<RewriteRuntime>(
            CompileOptions::default().compiler_facts(SHARED_QUANTIZATION_COMPILER_FACT),
        );
        assert_eq!(
            count_operations(graph.egraph().unwrap(), "BlockScaledQuantize"),
            expected
        );
    }
}

#[test]
fn packed_layout_accounts_for_alignment_dynamic_rows_and_empty_batches() {
    let quantize = BlockScaledQuantize {
        rows: 'm'.into(),
        input_features: 256,
        provider: "test".to_owned(),
    };
    for m in [0usize, 1, 3, 4, 5, 16, 127, 128] {
        let layout = PackedActivationLayout::new(m, 256).unwrap();
        assert_eq!(layout.scale_offset, m * 256);
        assert!(layout.scale_offset.is_multiple_of(128));
        assert_eq!(layout.total_bytes, m * 256 + m.div_ceil(4) * 4 * 2 * 4);
        let dyn_map = [(orbitkv_compiler::prelude::Symbol::from('m'), m)]
            .into_iter()
            .collect();
        assert_eq!(
            quantize.output_bytes().exec(&dyn_map),
            Some(layout.total_bytes)
        );
    }
    assert!(PackedActivationLayout::new(1, 127).is_err());
    assert!(PackedActivationLayout::new(1, 0).is_err());
    assert!(PackedActivationLayout::new(65_536, 128).is_err());
    assert!(scratch_bytes(usize::MAX, 128).is_err());
}

#[test]
fn shared_buffers_are_fully_accounted_and_have_no_hidden_scratch() {
    let (x, q, w, ws, y) = (
        NodeIndex::new(0),
        NodeIndex::new(1),
        NodeIndex::new(2),
        NodeIndex::new(3),
        NodeIndex::new(4),
    );
    let quantize = BlockScaledQuantize {
        rows: 3.into(),
        input_features: 256,
        provider: "test".to_owned(),
    };
    let gemm = PrequantizedDeepGemm {
        rows: 3.into(),
        selection: Selection {
            row_limit: 3,
            config: tiling::candidates(3, 128, 256, 78)[0],
        },
        prepared: Arc::new(OnceLock::new()),
        ..Default::default()
    };
    let bytes = PackedActivationLayout::new(3, 256).unwrap().total_bytes;
    let mut lengths: FxHashMap<_, _> = [
        (x, 3 * 256 * 2),
        (q, bytes),
        (w, 128 * 256),
        (ws, 8),
        (y, 3 * 128 * 2),
    ]
    .into_iter()
    .collect();
    let dyn_map = Default::default();
    assert_eq!(
        quantize
            .device_memory_plan(q, &[x], &lengths, &dyn_map)
            .unwrap()
            .persistent_bytes,
        0
    );
    assert_eq!(
        gemm.device_memory_plan(y, &[q, w, ws], &lengths, &dyn_map)
            .unwrap()
            .persistent_bytes,
        0
    );
    lengths.insert(q, bytes - 1);
    assert!(
        quantize
            .device_memory_plan(q, &[x], &lengths, &dyn_map)
            .is_err()
    );
    assert!(
        gemm.device_memory_plan(y, &[q, w, ws], &lengths, &dyn_map)
            .is_err()
    );
    assert!(
        quantize
            .device_memory_plan(q, &[], &lengths, &dyn_map)
            .is_err()
    );
}

fn decode_e4m3(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * 2.0_f32.powi(-9)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn encode_e4m3(value: f32) -> u8 {
    // Independent exhaustive nearest-even oracle for finite values. The
    // quantization clamp keeps the magnitude within E4M3's finite range.
    let magnitude = value.abs();
    let mut best = 0_u8;
    let mut error = f32::INFINITY;
    for code in 0_u8..=126 {
        let distance = (decode_e4m3(code) - magnitude).abs();
        if distance < error || (distance == error && code & 1 == 0) {
            best = code;
            error = distance;
        }
    }
    best | if value.is_sign_negative() { 128 } else { 0 }
}

fn independent_packed_quantization(values: &[bf16], m: usize, k: usize) -> Vec<u8> {
    let layout = PackedActivationLayout::new(m, k).unwrap();
    let mut packed = vec![0_u8; layout.total_bytes];
    for row in 0..m {
        for block in 0..k / BLOCK {
            let start = row * k + block * BLOCK;
            let values = &values[start..start + BLOCK];
            let maximum = values
                .iter()
                .map(|value| value.to_f32().abs())
                .fold(0.0_f32, f32::max);
            let scale = maximum.max(1.0e-4) / 448.0;
            let offset = layout.scale_offset + (block * m.div_ceil(4) * 4 + row) * 4;
            packed[offset..offset + 4].copy_from_slice(&scale.to_ne_bytes());
            for (column, value) in values.iter().enumerate() {
                packed[start + column] = encode_e4m3(value.to_f32() / scale);
            }
        }
    }
    packed
}

#[test]
fn independent_fp8_oracle_covers_subnormals_saturation_and_ties() {
    for (value, code) in [
        (0.0, 0x00),
        (-0.0, 0x80),
        (1.0, 0x38),
        (-0.5, 0xb0),
        (448.0, 0x7e),
        (1.0625, 0x38),
        (1.1875, 0x3a),
        (2.0_f32.powi(-9), 0x01),
    ] {
        assert_eq!(encode_e4m3(value), code);
    }
}

#[test]
#[ignore = "requires an SM90 GPU and nvcc; validates shared FP8 fanout and capture replay"]
fn shared_fp8_fanout_matches_combined_on_sm90() {
    let context = crate::cudarc::driver::CudaContext::new(0).expect("SM90 GPU required");
    assert_eq!(context.compute_capability().unwrap(), (9, 0));
    let stream = context.default_stream();
    for m in [1usize, 3, 4, 5, 8, 16] {
        let k = 256usize;
        let provider = jit::provider_identity().unwrap();
        let mut graph = Graph::default();
        let input = graph.tensor((m, k)).as_dtype(DType::Bf16);
        let packed_bytes = PackedActivationLayout::new(m, k).unwrap().total_bytes;
        let packed = graph
            .custom_op(
                BlockScaledQuantize {
                    rows: m.into(),
                    input_features: k,
                    provider: provider.clone(),
                },
                vec![input],
                packed_bytes,
                DType::U8,
            )
            .output();
        let mut pairs = Vec::new();
        let mut weights = Vec::new();
        for n in [128usize, 256] {
            let weight = graph.tensor((n, k)).as_dtype(DType::F8E4M3).persist();
            let scales = graph.tensor((n.div_ceil(BLOCK), k / BLOCK)).persist();
            weights.push((weight, scales, n));
            for variant in 0..selection::SEARCH_VARIANTS {
                let combined = graph
                    .custom_op(
                        DeepGemm {
                            rows: m.into(),
                            selection: device_selection(&stream, m, n, k, variant),
                            prepared: Arc::new(OnceLock::new()),
                            provider: provider.clone(),
                            scratch: Arc::new(Mutex::new(None)),
                        },
                        vec![input, weight, scales],
                        (m, n),
                        DType::Bf16,
                    )
                    .output();
                let shared = graph
                    .custom_op(
                        PrequantizedDeepGemm {
                            rows: m.into(),
                            selection: device_selection(&stream, m, n, k, variant),
                            prepared: Arc::new(OnceLock::new()),
                            provider: provider.clone(),
                            scratch: Arc::new(Mutex::new(None)),
                        },
                        vec![packed, weight, scales],
                        (m, n),
                        DType::Bf16,
                    )
                    .output();
                pairs.push((combined, shared));
            }
        }
        let mut runtime = crate::runtime::CudaRuntime::initialize(stream.clone());
        for (weight, scales, n) in weights {
            runtime.set_data(
                weight,
                (0..n * k)
                    .map(|i| if i % 3 == 0 { 0xb0_u8 } else { 0x30_u8 })
                    .collect::<Vec<_>>(),
            );
            runtime.set_data(
                scales,
                (0..n.div_ceil(BLOCK) * (k / BLOCK))
                    .map(|i| 0.25 + i as f32 * 0.25)
                    .collect::<Vec<_>>(),
            );
        }
        runtime.set_data(input, vec![bf16::from_f32(0.0); m * k]);
        runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
        // Updating the shared input between executions must refresh every
        // consumer, including replay of the captured parent CUDA graph.
        for phase in 0..3 {
            let values = (0..m * k)
                .map(|i| {
                    bf16::from_f32(match phase {
                        0 => 0.0,
                        1 => ((i % 31) as f32 - 15.0) / 16.0,
                        _ => ((i % 19) as f32 - 9.0) * 1.0e-6,
                    })
                })
                .collect::<Vec<_>>();
            let expected_packed = independent_packed_quantization(&values, m, k);
            runtime.set_data(input, values);
            runtime.execute(&graph.dyn_map);
            for &(combined, shared) in &pairs {
                assert_eq!(
                    runtime.get_bf16(combined),
                    runtime.get_bf16(shared),
                    "combined/shared mismatch at M={m}, phase={phase}"
                );
            }
            let raw = runtime.get_u8(packed);
            assert_eq!(
                raw, expected_packed,
                "independent FP8 quantization mismatch at M={m}, phase={phase}"
            );
            let offset = m * k;
            for block in 0..k / BLOCK {
                for row in m..m.div_ceil(4) * 4 {
                    let start = offset + (block * m.div_ceil(4) * 4 + row) * 4;
                    assert_eq!(
                        &raw[start..start + 4],
                        &[0; 4],
                        "uninitialized scale padding"
                    );
                }
            }
        }
    }
}

impl CustomOp for BlockScaledQuantize {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn HostOp>(Box::new(self.clone()) as Box<dyn HostOp>)
    }
}
