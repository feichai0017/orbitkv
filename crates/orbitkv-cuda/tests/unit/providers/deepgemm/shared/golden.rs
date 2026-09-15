//! Frozen outputs from Torch/DeepGEMM, independent of the CUDA implementation.

use super::*;
use cudarc::driver::DevicePtr;

#[derive(serde::Deserialize)]
struct Fixture {
    m: usize,
    k: usize,
    input_bf16_bits: Vec<u16>,
    fp8_bytes: Vec<u8>,
    scale_f32_bits: Vec<u32>,
}

fn fixture() -> (Fixture, Vec<u8>) {
    let data: Fixture =
        serde_json::from_str(include_str!("../../../../fixtures/fp8/row128-rne.json")).unwrap();
    assert_eq!(data.input_bf16_bits.len(), data.m * data.k);
    let mut expected = data.fp8_bytes.clone();
    expected.extend(
        data.scale_f32_bits
            .iter()
            .flat_map(|bits| bits.to_ne_bytes()),
    );
    assert_eq!(
        expected.len(),
        PackedActivationLayout::new(data.m, data.k)
            .unwrap()
            .total_bytes
    );
    (data, expected)
}

#[test]
fn independent_cpu_quantization_matches_frozen_torch_bits() {
    let (data, expected) = fixture();
    let values = data
        .input_bf16_bits
        .iter()
        .copied()
        .map(bf16::from_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        independent_packed_quantization(&values, data.m, data.k),
        expected
    );
}

#[test]
#[ignore = "requires CUDA; verifies the FP8 rounding contract against frozen independent output"]
fn cuda_quantization_matches_frozen_torch_bits_and_rejects_previous_rounding() {
    let (data, expected) = fixture();
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let source = format!(
        "{}\n{}",
        contract::quantizer_source(),
        include_str!("quantize_previous.cuh")
    );
    let module = context
        .load_module(compile_module_image_for_current_device(&context, &source).unwrap())
        .unwrap();
    let input = stream.clone_htod(&data.input_bf16_bits).unwrap();
    let layout = PackedActivationLayout::new(data.m, data.k).unwrap();
    let output = stream
        .clone_htod(&vec![0xcd_u8; layout.total_bytes])
        .unwrap();
    let pointer = output.device_ptr(&stream).0;
    stream.synchronize().unwrap();
    for (name, threads, matches_reference) in [
        ("previous_block_scaled_quantize", BLOCK, false),
        ("block_scaled_quantize", contract::QUANTIZER_THREADS, true),
    ] {
        let kernel = module.load_function(name).unwrap();
        let capture = super::quantizer::capture_quantizer(
            &stream,
            &kernel,
            (
                pointer,
                layout.scale_pointer(pointer).unwrap(),
                input.device_ptr(&stream).0,
            ),
            (data.m, data.k),
            threads,
        );
        capture.launch().unwrap();
        stream.synchronize().unwrap();
        let actual = stream.clone_dtoh(&output).unwrap();
        assert_eq!(
            actual == expected,
            matches_reference,
            "{name} rounding contract"
        );
    }
}
