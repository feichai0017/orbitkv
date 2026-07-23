use crate::*;
use half::{bf16, f16};

#[test]
fn rms_norm_matches_hand_computed_result() {
    let spec = RmsNormSpec::new(1, 2, 1.0e-6, DType::F32).unwrap();
    let input = [3.0, 4.0];
    let weight = [1.0, 0.5];
    let mut output = [0.0; 2];

    rms_norm_f32_reference(&input, &weight, &mut output, spec).unwrap();

    let inverse_rms = 1.0_f32 / (12.5_f32 + 1.0e-6).sqrt();
    assert!((output[0] - 3.0 * inverse_rms).abs() < 1.0e-6);
    assert!((output[1] - 2.0 * inverse_rms).abs() < 1.0e-6);
}

#[test]
fn rms_norm_validates_every_buffer() {
    let spec = RmsNormSpec::new(2, 4, 1.0e-5, DType::F32).unwrap();
    let error = rms_norm_f32_reference(&[0.0; 7], &[1.0; 4], &mut [0.0; 8], spec).unwrap_err();
    assert_eq!(
        error,
        ContractError::LengthMismatch {
            buffer: "input",
            expected: 8,
            actual: 7,
        }
    );
}

#[test]
fn low_precision_references_quantize_the_f32_result() {
    let input_f32 = [3.0_f32, 4.0];
    let weight_f32 = [1.0_f32, 0.5];

    let f16_spec = RmsNormSpec::new(1, 2, 1.0e-6, DType::F16).unwrap();
    let input_f16 = input_f32.map(f16::from_f32);
    let weight_f16 = weight_f32.map(f16::from_f32);
    let mut output_f16 = [f16::ZERO; 2];
    rms_norm_f16_reference(&input_f16, &weight_f16, &mut output_f16, f16_spec).unwrap();

    let bf16_spec = RmsNormSpec::new(1, 2, 1.0e-6, DType::Bf16).unwrap();
    let input_bf16 = input_f32.map(bf16::from_f32);
    let weight_bf16 = weight_f32.map(bf16::from_f32);
    let mut output_bf16 = [bf16::ZERO; 2];
    rms_norm_bf16_reference(&input_bf16, &weight_bf16, &mut output_bf16, bf16_spec).unwrap();

    let inverse_rms = 1.0_f32 / (12.5_f32 + 1.0e-6).sqrt();
    let expected = [3.0 * inverse_rms, 2.0 * inverse_rms];
    for (actual, expected) in output_f16.iter().map(|value| value.to_f32()).zip(expected) {
        assert!((actual - expected).abs() < 1.0e-3);
    }
    for (actual, expected) in output_bf16.iter().map(|value| value.to_f32()).zip(expected) {
        assert!((actual - expected).abs() < 1.0e-2);
    }
}

#[test]
fn add_rms_norm_updates_both_f32_buffers() {
    let spec = AddRmsNormSpec::new(1, 2, 1.0e-6, DType::F32).unwrap();
    let mut input = [1.0_f32, 2.0];
    let mut residual = [2.0_f32, 2.0];
    let weight = [1.0_f32, 0.5];

    add_rms_norm_f32_reference(&mut input, &mut residual, &weight, spec).unwrap();

    assert_eq!(residual, [3.0, 4.0]);
    let inverse_rms = 1.0_f32 / (12.5_f32 + 1.0e-6).sqrt();
    assert!((input[0] - 3.0 * inverse_rms).abs() < 1.0e-6);
    assert!((input[1] - 2.0 * inverse_rms).abs() < 1.0e-6);
}

#[test]
fn add_rms_norm_low_precision_materializes_quantized_residual() {
    let mut input = [f16::from_f32(0.3333), f16::from_f32(-0.7777)];
    let mut residual = [f16::from_f32(0.1111), f16::from_f32(0.2222)];
    let original_input = input;
    let original_residual = residual;
    let weight = [f16::ONE; 2];
    let spec = AddRmsNormSpec::new(1, 2, 1.0e-5, DType::F16).unwrap();

    add_rms_norm_f16_reference(&mut input, &mut residual, &weight, spec).unwrap();

    for index in 0..2 {
        assert_eq!(
            residual[index],
            f16::from_f32(original_input[index].to_f32() + original_residual[index].to_f32())
        );
    }
    assert!(input.iter().all(|value| value.is_finite()));
}

#[test]
fn add_rms_norm_validates_residual_length_and_dtype() {
    let spec = AddRmsNormSpec::new(2, 4, 1.0e-5, DType::F32).unwrap();
    let error =
        add_rms_norm_f32_reference(&mut [0.0; 8], &mut [0.0; 7], &[1.0; 4], spec).unwrap_err();
    assert_eq!(
        error,
        ContractError::LengthMismatch {
            buffer: "residual",
            expected: 8,
            actual: 7,
        }
    );

    let wrong_dtype = AddRmsNormSpec::new(1, 2, 1.0e-5, DType::Bf16).unwrap();
    assert_eq!(
        add_rms_norm_f16_reference(
            &mut [f16::ZERO; 2],
            &mut [f16::ZERO; 2],
            &[f16::ONE; 2],
            wrong_dtype,
        ),
        Err(ContractError::UnsupportedDType(DType::Bf16))
    );
}

#[test]
fn fp8_e4m3fn_encoding_matches_known_values_and_ties() {
    let fixtures = [
        (0.0, 0x00),
        (-0.0, 0x80),
        (1.0, 0x38),
        (-1.0, 0xb8),
        (448.0, 0x7e),
        (500.0, 0x7e),
        (2.0_f32.powi(-9), 0x01),
        (1.0625, 0x38),
        (1.1875, 0x3a),
    ];
    for (value, expected) in fixtures {
        assert_eq!(fp8_e4m3fn_from_f32(value), expected, "value={value}");
    }
    assert_eq!(fp8_e4m3fn_from_f32(f32::NAN), 0x7f);
    assert_eq!(fp8_e4m3fn_to_f32(0x38), 1.0);
    assert_eq!(fp8_e4m3fn_to_f32(0x7e), 448.0);
}

#[test]
fn dynamic_fp8_reference_emits_per_row_scale_and_zero_floor() {
    let spec = RmsNormDynamicFp8Spec::new(2, 2, 1.0e-6, DType::Bf16).unwrap();
    let input = [
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
        bf16::ZERO,
        bf16::ZERO,
    ];
    let weight = [bf16::ONE, bf16::ONE];
    let mut output = [0_u8; 4];
    let mut scales = [0.0_f32; 2];

    rms_norm_dynamic_fp8_bf16_reference(&input, &weight, &mut output, &mut scales, spec).unwrap();

    assert_eq!(output[1], 0x7e);
    assert_eq!(&output[2..], &[0x00, 0x00]);
    assert!(scales[0] > DYNAMIC_FP8_MIN_SCALE);
    assert_eq!(scales[1], DYNAMIC_FP8_MIN_SCALE);
}

#[test]
fn dynamic_fp8_reference_validates_output_and_scale_lengths() {
    let spec = RmsNormDynamicFp8Spec::new(2, 4, 1.0e-5, DType::F16).unwrap();
    let error = rms_norm_dynamic_fp8_f16_reference(
        &[f16::ZERO; 8],
        &[f16::ONE; 4],
        &mut [0_u8; 7],
        &mut [0.0; 2],
        spec,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ContractError::LengthMismatch {
            buffer: "output",
            expected: 8,
            actual: 7,
        }
    );
}
