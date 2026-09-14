use super::*;
use crate::runtime::{CapturedCudaExecution, CudaRuntime};

#[derive(Clone, Copy, Debug)]
enum Mode {
    Combined,
    Shared,
    QuantizeOnly,
    GemmOnly,
}

fn build_fixture(
    stream: Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    mode: Mode,
) -> (Graph, CudaRuntime, Vec<GraphTensor>) {
    let mut graph = Graph::default();
    let provider = jit::provider_identity().unwrap();
    let values = (0..m * k)
        .map(|i| bf16::from_f32(((i % 31) as f32 - 15.0) / 16.0))
        .collect::<Vec<_>>();
    let packed_values = independent_packed_quantization(&values, m, k);
    let input = graph.tensor((m, k)).as_dtype(DType::Bf16).persist();
    let packed_input = matches!(mode, Mode::GemmOnly).then(|| {
        graph
            .tensor(packed_values.len())
            .as_dtype(DType::U8)
            .persist()
    });
    let packed = if matches!(mode, Mode::Shared | Mode::QuantizeOnly) {
        Some(graph.custom_op(
            BlockScaledQuantize {
                rows: m.into(),
                input_features: k,
                provider: provider.clone(),
            },
            vec![input],
            packed_values.len(),
            DType::U8,
        ))
    } else {
        packed_input
    };
    let mut outputs = Vec::new();
    let mut parameters = Vec::new();
    if matches!(mode, Mode::QuantizeOnly) {
        packed.unwrap().output();
    } else {
        for _ in 0..2 {
            let weight = graph.tensor((n, k)).as_dtype(DType::F8E4M3).persist();
            let scale = graph.tensor((n.div_ceil(BLOCK), k / BLOCK)).persist();
            parameters.push((weight, scale));
            let output = if matches!(mode, Mode::Combined) {
                graph.custom_op(
                    DeepGemm {
                        rows: m.into(),
                        output_features: n,
                        input_features: k,
                        variant: 0,
                        provider: provider.clone(),
                        scratch: Arc::new(Mutex::new(None)),
                    },
                    vec![input, weight, scale],
                    (m, n),
                    DType::Bf16,
                )
            } else {
                graph.custom_op(
                    PrequantizedDeepGemm {
                        rows: m.into(),
                        output_features: n,
                        input_features: k,
                        variant: 0,
                        provider: provider.clone(),
                        scratch: Arc::new(Mutex::new(None)),
                    },
                    vec![packed.unwrap(), weight, scale],
                    (m, n),
                    DType::Bf16,
                )
            };
            outputs.push(output.output());
        }
    }
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.set_data(input, values);
    if let Some(packed_input) = packed_input {
        runtime.set_data(packed_input, packed_values);
    }
    for (weight, scale) in parameters {
        runtime.set_data(
            weight,
            (0..n * k)
                .map(|i| if i % 3 == 0 { 0xb0_u8 } else { 0x30_u8 })
                .collect::<Vec<_>>(),
        );
        runtime.set_data(
            scale,
            (0..n.div_ceil(BLOCK) * (k / BLOCK))
                .map(|i| 0.25 + (i % 4) as f32 * 0.25)
                .collect::<Vec<_>>(),
        );
    }
    runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
    for _ in 0..5 {
        runtime.execute(&graph.dyn_map);
    }
    (graph, runtime, outputs)
}

fn median_device_us(capture: &CapturedCudaExecution, stream: &Arc<CudaStream>) -> f32 {
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.context().new_event(flags).unwrap();
    let end = stream.context().new_event(flags).unwrap();
    for _ in 0..10 {
        capture.launch().unwrap();
    }
    stream.synchronize().unwrap();
    let mut samples = Vec::new();
    for _ in 0..31 {
        start.record(stream).unwrap();
        for _ in 0..20 {
            capture.launch().unwrap();
        }
        end.record(stream).unwrap();
        end.synchronize().unwrap();
        samples.push(start.elapsed_ms(&end).unwrap() * 1_000.0 / 20.0);
    }
    samples.sort_by(f32::total_cmp);
    samples[samples.len() / 2]
}

#[test]
#[ignore = "requires exclusive SM90 GPU; emits warmed dual-consumer quantize/GEMM/whole-graph timings"]
fn shared_fp8_dual_consumer_benchmark_on_sm90() {
    let context = crate::cudarc::driver::CudaContext::new(0).expect("SM90 GPU required");
    assert_eq!(context.compute_capability().unwrap(), (9, 0));
    let stream = context.default_stream();
    let (n, k) = (17_408usize, 5_120usize);
    for m in [1usize, 4, 8, 64] {
        let mut reference = None;
        let mut records = Vec::new();
        for mode in [
            Mode::Combined,
            Mode::Shared,
            Mode::QuantizeOnly,
            Mode::GemmOnly,
        ] {
            let (graph, mut runtime, outputs) = build_fixture(stream.clone(), m, n, k, mode);
            if !outputs.is_empty() {
                let actual = outputs
                    .iter()
                    .map(|output| runtime.get_bf16(*output))
                    .collect::<Vec<_>>();
                if let Some(reference) = &reference {
                    assert_eq!(
                        &actual, reference,
                        "mode {mode:?} differs from combined at M={m}"
                    );
                } else {
                    reference = Some(actual);
                }
            }
            let capture = runtime.capture_execution(&graph.dyn_map).unwrap();
            let us = median_device_us(&capture, &stream);
            records.push(serde_json::json!({
                "mode": format!("{mode:?}"), "median_device_us": us,
                "runtime_intermediate_bytes": runtime.intermediate_buffer_bytes(),
            }));
            drop(capture);
            stream.synchronize().unwrap();
        }
        eprintln!(
            "SHARED_FP8_BENCH {}",
            serde_json::json!({
                "m": m, "n": n, "k": k, "consumers": 2, "variant": 0,
                "trials": 31, "launches_per_trial": 20, "records": records,
                "scope": "isolated same-input two-projection fixture, not full-model serving",
            })
        );
    }
}
