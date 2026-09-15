//! The previous implementation is a frozen test fixture, never a search candidate.

use super::*;
use cudarc::driver::{CudaGraph, DevicePtr, sys};

const CAPTURE_REPETITIONS: usize = 100;

pub(super) fn capture_quantizer(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    pointers: (u64, u64, u64),
    shape: (usize, usize),
    threads: usize,
) -> CudaGraph {
    let (output, scales, input) = pointers;
    let (m, k) = (shape.0 as i32, shape.1 as i32);
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    for _ in 0..CAPTURE_REPETITIONS {
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&output)
                .arg(&scales)
                .arg(&input)
                .arg(&m)
                .arg(&k)
                .launch(LaunchConfig {
                    grid_dim: ((shape.1 / BLOCK) as u32, shape.0 as u32, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
    }
    stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .unwrap()
}

fn measure_pair(stream: &Arc<CudaStream>, captures: &[CudaGraph; 2]) -> [f32; 2] {
    let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.context().new_event(flags).unwrap();
    let end = stream.context().new_event(flags).unwrap();
    for _ in 0..5 {
        for capture in captures {
            capture.launch().unwrap();
        }
    }
    stream.synchronize().unwrap();
    let mut samples = [Vec::new(), Vec::new()];
    for trial in 0..31 {
        for offset in 0..2 {
            let index = (trial + offset) % 2;
            start.record(stream).unwrap();
            captures[index].launch().unwrap();
            end.record(stream).unwrap();
            end.synchronize().unwrap();
            samples[index]
                .push(start.elapsed_ms(&end).unwrap() * 1000.0 / CAPTURE_REPETITIONS as f32);
        }
    }
    samples.map(|mut values| {
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    })
}

#[test]
#[ignore = "requires exclusive CUDA GPU; checks packed bits and reports quantizer device time"]
fn warp_quantizer_matches_reference_and_measures_device_time() {
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let source = format!(
        "{}\n{}",
        contract::quantizer_source(),
        include_str!("quantize_previous.cuh")
    );
    let image = compile_module_image_for_current_device(&context, &source).unwrap();
    let module = context.load_module(image).unwrap();
    let kernels = [
        module
            .load_function("previous_block_scaled_quantize")
            .unwrap(),
        module.load_function("block_scaled_quantize").unwrap(),
    ];
    // Cover every finite BF16 encoding, including both zero signs, subnormals,
    // extreme exponents, and mixed magnitudes within a scale group.
    let finite = (0..=u16::MAX)
        .map(bf16::from_bits)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    for (m, k) in [
        (1, 128),
        (3, 5120),
        (8, 5120),
        (32, 5120),
        (256, 17408),
        (finite.len() / BLOCK, BLOCK),
    ] {
        let exhaustive = m * k == finite.len();
        let values = if exhaustive {
            finite.clone()
        } else {
            (0..m * k)
                .map(|index| {
                    let signed = (index % 127) as f32 - 63.0;
                    bf16::from_f32(
                        signed
                            * if (index / BLOCK).is_multiple_of(3) {
                                1.0e-7
                            } else {
                                0.125
                            },
                    )
                })
                .collect::<Vec<_>>()
        };
        let layout = PackedActivationLayout::new(m, k).unwrap();
        let input = stream
            .clone_htod(
                &values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let input_ptr = input.device_ptr(&stream).0;
        let outputs = [
            stream
                .clone_htod(&vec![0xcd_u8; layout.total_bytes])
                .unwrap(),
            stream
                .clone_htod(&vec![0xcd_u8; layout.total_bytes])
                .unwrap(),
        ];
        stream.synchronize().unwrap();
        let captures = std::array::from_fn(|index| {
            let pointer = outputs[index].device_ptr(&stream).0;
            capture_quantizer(
                &stream,
                &kernels[index],
                (pointer, layout.scale_pointer(pointer).unwrap(), input_ptr),
                (m, k),
                if index == 0 {
                    BLOCK
                } else {
                    contract::QUANTIZER_THREADS
                },
            )
        });
        let [previous_us, warp_us] = measure_pair(&stream, &captures);
        let previous = stream.clone_dtoh(&outputs[0]).unwrap();
        let actual = stream.clone_dtoh(&outputs[1]).unwrap();
        if exhaustive || m < 32 {
            assert_eq!(actual, independent_packed_quantization(&values, m, k));
        }
        println!(
            "{}",
            serde_json::json!({"m":m,"k":k,"previous_us":previous_us,
                "warp_us":warp_us,"previous_packed_bit_exact":actual == previous,"exhaustive_finite_bf16":exhaustive})
        );
    }
    stream.synchronize().unwrap();
}
