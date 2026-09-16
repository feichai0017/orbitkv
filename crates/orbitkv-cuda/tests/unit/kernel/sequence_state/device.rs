use super::*;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, sys,
};
use half::bf16;
use std::sync::Arc;

#[derive(serde::Deserialize)]
struct ConvolutionFixture {
    channels: usize,
    kernel_width: usize,
    indptr: Vec<i32>,
    input: Vec<u16>,
    weights: Vec<u16>,
    history: Vec<u16>,
    output: Vec<u16>,
    next_history: Vec<u16>,
}

#[derive(serde::Deserialize)]
struct DeltaScanFixture {
    key_width: usize,
    value_width: usize,
    indptr: Vec<i32>,
    query: Vec<u16>,
    key: Vec<u16>,
    value: Vec<u16>,
    log_decay_f32_bits: Vec<i32>,
    beta: Vec<u16>,
    initial_state: Vec<u16>,
    output: Vec<u16>,
    final_state: Vec<u16>,
}

#[test]
#[ignore = "requires CUDA and independent frozen Torch convolution output"]
fn convolution_matches_frozen_prefill_decode_and_nonzero_history() {
    let fixture: ConvolutionFixture = serde_json::from_str(include_str!(
        "../../../fixtures/sequence_state/convolution.json"
    ))
    .unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let kernel = PackedConvolutionKernel {
        tokens: usize::try_from(*fixture.indptr.last().unwrap())
            .unwrap()
            .into(),
        requests: (fixture.indptr.len() - 1).into(),
        spec: PackedConvolutionSpec {
            channels: fixture.channels,
            kernel_width: fixture.kernel_width,
        },
    };
    let mut cache = FxHashMap::default();
    let (function, _module, _, _, _, _, _) = kernel.compile(&stream, &mut cache);
    let input = stream.clone_htod(&fixture.input).unwrap();
    let weights = stream.clone_htod(&fixture.weights).unwrap();
    let history = stream.clone_htod(&fixture.history).unwrap();
    let indptr = stream.clone_htod(&fixture.indptr).unwrap();
    let expected: Vec<_> = fixture
        .output
        .iter()
        .chain(&fixture.next_history)
        .copied()
        .collect();
    let output = stream.alloc_zeros::<u16>(expected.len()).unwrap();
    unsafe {
        stream
            .launch_builder(&function)
            .arg(&output.device_ptr(&stream).0)
            .arg(&input.device_ptr(&stream).0)
            .arg(&weights.device_ptr(&stream).0)
            .arg(&history.device_ptr(&stream).0)
            .arg(&indptr.device_ptr(&stream).0)
            .launch(LaunchConfig {
                grid_dim: ((fixture.indptr.len() - 1) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    let actual = stream.clone_dtoh(&output).unwrap();
    let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    let maximum = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (bf16::from_bits(a).to_f32() - bf16::from_bits(b).to_f32()).abs())
        .fold(0.0_f32, f32::max);
    println!("convolution mismatches={mismatches}, max_error={maximum}");
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "requires CUDA and a frozen BF16-boundary delta-rule fixture"]
fn delta_scans_match_frozen_bf16_boundary_output_and_cache_bits() {
    let fixture: DeltaScanFixture = serde_json::from_str(include_str!(
        "../../../fixtures/sequence_state/delta_scan.json"
    ))
    .unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let spec = PackedDeltaScanSpec {
        key_heads: 1,
        value_heads: 1,
        key_width: fixture.key_width,
        value_width: fixture.value_width,
        normalization_epsilon: 1e-6,
        round_normalized_qk_to_bf16: true,
        round_final_state_to_bf16: true,
    };
    let scan = PackedDeltaScanKernel {
        tokens: 2.into(),
        requests: 1.into(),
        spec,
    };
    let registers = super::super::delta_registers::KernelDeltaRegisters::from_scan(scan.clone());
    let from_bf16 = |values: &[u16]| {
        values
            .iter()
            .copied()
            .map(bf16::from_bits)
            .map(bf16::to_f32)
            .collect::<Vec<_>>()
    };
    let query = stream.clone_htod(&from_bf16(&fixture.query)).unwrap();
    let key = stream.clone_htod(&from_bf16(&fixture.key)).unwrap();
    let value = stream.clone_htod(&from_bf16(&fixture.value)).unwrap();
    let decay = stream
        .clone_htod(
            &fixture
                .log_decay_f32_bits
                .iter()
                .map(|bits| f32::from_bits(*bits as u32))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let beta = stream.clone_htod(&from_bf16(&fixture.beta)).unwrap();
    let state = stream
        .clone_htod(&from_bf16(&fixture.initial_state))
        .unwrap();
    let indptr = stream.clone_htod(&fixture.indptr).unwrap();
    let pointers = [
        query.device_ptr(&stream).0,
        key.device_ptr(&stream).0,
        value.device_ptr(&stream).0,
        decay.device_ptr(&stream).0,
        beta.device_ptr(&stream).0,
        state.device_ptr(&stream).0,
    ];
    let output_elements = fixture.output.len();
    let expected_elements = output_elements + fixture.final_state.len();
    let mut cache = FxHashMap::default();
    for op in [&scan as &dyn KernelOp, &registers as &dyn KernelOp] {
        let (function, _module, _, grid, block, shared, _) = op.compile(&stream, &mut cache);
        let output = stream.alloc_zeros::<u32>(expected_elements).unwrap();
        let output_ptr = output.device_ptr(&stream).0;
        let indptr_ptr = indptr.device_ptr(&stream).0;
        let mut launch = stream.launch_builder(&function);
        launch.arg(&output_ptr);
        for pointer in &pointers {
            launch.arg(pointer);
        }
        launch.arg(&indptr_ptr);
        unsafe {
            launch
                .launch(LaunchConfig {
                    grid_dim: (
                        grid.0.to_usize().unwrap() as u32,
                        grid.1.to_usize().unwrap() as u32,
                        grid.2.to_usize().unwrap() as u32,
                    ),
                    block_dim: (
                        block.0.to_usize().unwrap() as u32,
                        block.1.to_usize().unwrap() as u32,
                        block.2.to_usize().unwrap() as u32,
                    ),
                    shared_mem_bytes: shared.to_usize().unwrap() as u32,
                })
                .unwrap();
        }
        let actual = stream.clone_dtoh(&output).unwrap();
        let actual_output = actual[..output_elements]
            .iter()
            .map(|bits| bf16::from_f32(f32::from_bits(*bits)).to_bits())
            .collect::<Vec<_>>();
        let expected_state = fixture
            .final_state
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32().to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            actual_output,
            fixture.output,
            "{} token output",
            op.kernel_name()
        );
        assert_eq!(
            actual[output_elements..],
            expected_state,
            "{} final state",
            op.kernel_name()
        );
    }
}

#[test]
#[ignore = "requires CUDA; validates F32 recurrent state across chunk boundaries"]
fn f32_recurrent_state_is_chunk_invariant_and_bf16_state_is_not() {
    const TOKENS: usize = 4;
    const SPLIT: usize = 2;
    const KEY_WIDTH: usize = 128;
    const VALUE_WIDTH: usize = 8;
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut cache = FxHashMap::<String, (Arc<CudaModule>, CudaFunction)>::default();
    let query = scan_values(TOKENS * KEY_WIDTH, 3);
    let key = scan_values(TOKENS * KEY_WIDTH, 7);
    let value = scan_values(TOKENS * VALUE_WIDTH, 11);
    let decay = vec![-0.125_f32; TOKENS];
    let beta = vec![0.625_f32; TOKENS];
    let state = scan_values(KEY_WIDTH * VALUE_WIDTH, 17)
        .into_iter()
        .map(|value| value + 1.0 / 8192.0)
        .collect::<Vec<_>>();

    for round_state in [false, true] {
        let spec = PackedDeltaScanSpec {
            key_heads: 1,
            value_heads: 1,
            key_width: KEY_WIDTH,
            value_width: VALUE_WIDTH,
            normalization_epsilon: 1e-6,
            round_normalized_qk_to_bf16: true,
            round_final_state_to_bf16: round_state,
        };
        let full = launch_register_scan(
            &stream, &mut cache, spec, &query, &key, &value, &decay, &beta, &state,
        );
        let first = launch_register_scan(
            &stream,
            &mut cache,
            spec,
            &query[..SPLIT * KEY_WIDTH],
            &key[..SPLIT * KEY_WIDTH],
            &value[..SPLIT * VALUE_WIDTH],
            &decay[..SPLIT],
            &beta[..SPLIT],
            &state,
        );
        let first_token_elements = SPLIT * VALUE_WIDTH;
        let second = launch_register_scan(
            &stream,
            &mut cache,
            spec,
            &query[SPLIT * KEY_WIDTH..],
            &key[SPLIT * KEY_WIDTH..],
            &value[SPLIT * VALUE_WIDTH..],
            &decay[SPLIT..],
            &beta[SPLIT..],
            &first[first_token_elements..],
        );
        let full_token_elements = TOKENS * VALUE_WIDTH;
        let mut chunked_values = first[..first_token_elements].to_vec();
        chunked_values.extend_from_slice(&second[..first_token_elements]);
        let output_equal = chunked_values
            .iter()
            .zip(&full[..full_token_elements])
            .all(|(left, right)| left.to_bits() == right.to_bits());
        let state_equal = second[first_token_elements..]
            .iter()
            .zip(&full[full_token_elements..])
            .all(|(left, right)| left.to_bits() == right.to_bits());
        if round_state {
            assert!(
                !output_equal || !state_equal,
                "BF16 submission boundary must exercise the known chunking drift"
            );
        } else {
            assert!(
                output_equal,
                "F32 recurrent token values must be chunk invariant"
            );
            assert!(
                state_equal,
                "F32 recurrent final state must be chunk invariant"
            );
        }
    }
}

fn scan_values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let mixed = index
                .wrapping_mul(1_664_525)
                .wrapping_add(seed.wrapping_mul(1_013_904_223)) as u32;
            let sign = if mixed & 1 == 0 { 1.0 } else { -1.0 };
            let exponent = i32::try_from((mixed >> 1) % 13).unwrap() - 8;
            let mantissa = 1.0 + ((mixed >> 8) & 0xff) as f32 / 256.0;
            bf16::from_f32(sign * mantissa * 2.0_f32.powi(exponent)).to_f32()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn launch_register_scan(
    stream: &Arc<CudaStream>,
    cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    spec: PackedDeltaScanSpec,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    decay: &[f32],
    beta: &[f32],
    state: &[f32],
) -> Vec<f32> {
    let tokens = decay.len();
    assert_eq!(beta.len(), tokens);
    let scan = PackedDeltaScanKernel {
        tokens: tokens.into(),
        requests: 1.into(),
        spec,
    };
    let registers = super::super::delta_registers::KernelDeltaRegisters::from_scan(scan);
    let (function, _module, _, grid, block, shared, _) = registers.compile(stream, cache);
    let inputs =
        [query, key, value, decay, beta, state].map(|values| stream.clone_htod(values).unwrap());
    let indptr = stream
        .clone_htod(&[0_i32, i32::try_from(tokens).unwrap()])
        .unwrap();
    let output = stream
        .alloc_zeros::<f32>(tokens * spec.value_width + spec.key_width * spec.value_width)
        .unwrap();
    let output_ptr = output.device_ptr(stream).0;
    let pointers = inputs
        .iter()
        .map(|input| input.device_ptr(stream).0)
        .collect::<Vec<_>>();
    let indptr_ptr = indptr.device_ptr(stream).0;
    let mut launch = stream.launch_builder(&function);
    launch.arg(&output_ptr);
    for pointer in &pointers {
        launch.arg(pointer);
    }
    launch.arg(&indptr_ptr);
    unsafe {
        launch
            .launch(LaunchConfig {
                grid_dim: (
                    grid.0.to_usize().unwrap() as u32,
                    grid.1.to_usize().unwrap() as u32,
                    grid.2.to_usize().unwrap() as u32,
                ),
                block_dim: (
                    block.0.to_usize().unwrap() as u32,
                    block.1.to_usize().unwrap() as u32,
                    block.2.to_usize().unwrap() as u32,
                ),
                shared_mem_bytes: shared.to_usize().unwrap() as u32,
            })
            .unwrap();
    }
    stream.clone_dtoh(&output).unwrap()
}

#[test]
fn register_candidate_is_extracted_and_key_width_is_bounded() {
    for width in [2, 128, 129] {
        let mut graph = Graph::new();
        let state = graph.tensor((2, 2, width, 3));
        let indptr = graph.tensor(3).as_dtype(DType::Int);
        let output = packed_delta_scan(
            PackedDeltaScanPlan {
                query: graph.tensor((5, 1, width)),
                key: graph.tensor((5, 1, width)),
                value: graph.tensor((5, 2, 3)),
                log_decay: graph.tensor((5, 2)),
                update_gate: graph.tensor((5, 2)),
                state,
                query_indptr: indptr,
            },
            PackedDeltaScanSpec {
                key_heads: 1,
                value_heads: 2,
                key_width: width,
                value_width: 3,
                normalization_epsilon: 1e-6,
                round_normalized_qk_to_bf16: false,
                round_final_state_to_bf16: false,
            },
        );
        output.values.output();
        output.state.output();
        graph.build_search_space::<CudaRuntime>(CompileOptions::default());
        let candidate = try_extract_forced_op_llir_where(
            &graph,
            &["KernelDeltaRegisters"],
            ForcedExtractionConfig::new(319).attempts_per_node(128),
            |_| true,
        );
        assert_eq!(candidate.is_ok(), width <= 128);
    }
}

#[test]
#[ignore = "requires exclusive CUDA; validates register scan and measures captured executions"]
fn register_scan_preserves_bits_and_measures_device_time() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    for (width, columns, lengths) in [
        (2, 3, vec![2, 0, 3]),
        (65, 35, vec![1, 3, 0]),
        (128, 128, vec![1; 8]),
        (128, 128, vec![4; 8]),
    ] {
        let requests = lengths.len();
        let tokens: usize = lengths.iter().sum();
        let spec = PackedDeltaScanSpec {
            key_heads: 1,
            value_heads: 3,
            key_width: width,
            value_width: columns,
            normalization_epsilon: 1e-6,
            round_normalized_qk_to_bf16: width == 128,
            round_final_state_to_bf16: width == 128,
        };
        let scan = PackedDeltaScanKernel {
            tokens: tokens.into(),
            requests: requests.into(),
            spec,
        };
        let registers =
            super::super::delta_registers::KernelDeltaRegisters::from_scan(scan.clone());
        let values = |count: usize, offset: usize| {
            (0..count)
                .map(|i| {
                    let mixed = (i
                        .wrapping_mul(1_664_525)
                        .wrapping_add(offset.wrapping_mul(1_013_904_223)))
                        as u32;
                    let sign = if mixed & 1 == 0 { 1.0 } else { -1.0 };
                    let exponent = i32::try_from((mixed >> 1) % 13).unwrap() - 8;
                    let mantissa = 1.0 + ((mixed >> 8) & 0xff) as f32 / 256.0;
                    bf16::from_f32(sign * mantissa * 2.0_f32.powi(exponent)).to_f32()
                })
                .collect::<Vec<_>>()
        };
        let host_inputs = [
            values(tokens * width, 1),
            values(tokens * width, 7),
            values(tokens * 3 * columns, 13),
            vec![-0.125; tokens * 3],
            vec![0.625; tokens * 3],
            values(requests * 3 * width * columns, 19),
        ];
        let inputs = host_inputs
            .iter()
            .map(|data| stream.clone_htod(data).unwrap())
            .collect::<Vec<_>>();
        let mut offsets = vec![0_i32];
        for count in &lengths {
            offsets.push(offsets.last().unwrap() + *count as i32);
        }
        let indptr = stream.clone_htod(&offsets).unwrap();
        let mut cache = FxHashMap::default();
        let mut outputs = Vec::new();
        let mut timings = Vec::new();
        for op in [&scan as &dyn KernelOp, &registers as &dyn KernelOp] {
            let (function, _module, _, grid, block, shared, _) = op.compile(&stream, &mut cache);
            let output = stream
                .alloc_zeros::<f32>(op.output_size().to_usize().unwrap())
                .unwrap();
            let output_ptr = output.device_ptr(&stream).0;
            let pointers = inputs
                .iter()
                .map(|input| input.device_ptr(&stream).0)
                .collect::<Vec<_>>();
            let indptr_ptr = indptr.device_ptr(&stream).0;
            stream.synchronize().unwrap();
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .unwrap();
            for _ in 0..20 {
                let mut launch = stream.launch_builder(&function);
                launch.arg(&output_ptr);
                for pointer in &pointers {
                    launch.arg(pointer);
                }
                launch.arg(&indptr_ptr);
                unsafe {
                    launch
                        .launch(LaunchConfig {
                            grid_dim: (
                                grid.0.to_usize().unwrap() as u32,
                                grid.1.to_usize().unwrap() as u32,
                                grid.2.to_usize().unwrap() as u32,
                            ),
                            block_dim: (
                                block.0.to_usize().unwrap() as u32,
                                block.1.to_usize().unwrap() as u32,
                                block.2.to_usize().unwrap() as u32,
                            ),
                            shared_mem_bytes: shared.to_usize().unwrap() as u32,
                        })
                        .unwrap();
                }
            }
            let capture = stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .unwrap();
            capture.launch().unwrap();
            stream.synchronize().unwrap();
            let start = context
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            let end = context
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            let mut times = Vec::new();
            for _ in 0..15 {
                start.record(&stream).unwrap();
                capture.launch().unwrap();
                end.record(&stream).unwrap();
                end.synchronize().unwrap();
                times.push(start.elapsed_ms(&end).unwrap() * 1000.0 / 20.0);
            }
            times.sort_by(f32::total_cmp);
            timings.push(times[times.len() / 2]);
            outputs.push(stream.clone_dtoh(&output).unwrap());
        }
        assert!(outputs.iter().flatten().all(|value| value.is_finite()));
        assert_eq!(
            outputs[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            outputs[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        println!(
            "scan K={width} V={columns} lengths={lengths:?} original_us={} registers_us={}",
            timings[0], timings[1]
        );
    }
}
