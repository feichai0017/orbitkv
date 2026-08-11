use super::*;

#[test]
fn f32_reference_matches_hand_computed_result() {
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
fn reference_validates_dtype_and_all_buffer_lengths() {
    let spec = RmsNormSpec::new(2, 4, 1.0e-5, DType::F32).unwrap();
    for (actual, expected) in [
        (
            rms_norm_f32_reference(&[0.0; 7], &[1.0; 4], &mut [0.0; 8], spec),
            ContractError::LengthMismatch {
                buffer: "input",
                expected: 8,
                actual: 7,
            },
        ),
        (
            rms_norm_f32_reference(&[0.0; 8], &[1.0; 3], &mut [0.0; 8], spec),
            ContractError::LengthMismatch {
                buffer: "weight",
                expected: 4,
                actual: 3,
            },
        ),
        (
            rms_norm_f32_reference(&[0.0; 8], &[1.0; 4], &mut [0.0; 7], spec),
            ContractError::LengthMismatch {
                buffer: "output",
                expected: 8,
                actual: 7,
            },
        ),
    ] {
        assert_eq!(actual, Err(expected));
    }

    let wrong_dtype = RmsNormSpec::new(2, 4, 1.0e-5, DType::Bf16).unwrap();
    assert_eq!(
        rms_norm_f32_reference(&[0.0; 8], &[1.0; 4], &mut [0.0; 8], wrong_dtype),
        Err(ContractError::UnsupportedDType(DType::Bf16))
    );
}

#[test]
fn spec_rejects_invalid_dimensions_and_epsilon() {
    assert_eq!(
        RmsNormSpec::new(0, 4, 1.0e-5, DType::F32),
        Err(ContractError::ZeroDimension)
    );
    assert_eq!(
        RmsNormSpec::new(usize::MAX, 2, 1.0e-5, DType::F32),
        Err(ContractError::ElementCountOverflow)
    );

    for epsilon in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        assert!(matches!(
            RmsNormSpec::new(1, 4, epsilon, DType::F32),
            Err(ContractError::InvalidEpsilon(value)) if value.to_bits() == epsilon.to_bits()
        ));
    }
}

#[test]
fn low_precision_references_round_the_output_dtype() {
    let input_f16 = [f16::from_f32(3.0), f16::from_f32(4.0)];
    let weight_f16 = [f16::ONE, f16::from_f32(0.5)];
    let mut output_f16 = [f16::ZERO; 2];
    let f16_spec = RmsNormSpec::new(1, 2, 1.0e-6, DType::F16).unwrap();
    rms_norm_f16_reference(&input_f16, &weight_f16, &mut output_f16, f16_spec).unwrap();

    let input_bf16 = [bf16::from_f32(3.0), bf16::from_f32(4.0)];
    let weight_bf16 = [bf16::ONE, bf16::from_f32(0.5)];
    let mut output_bf16 = [bf16::ZERO; 2];
    let bf16_spec = RmsNormSpec::new(1, 2, 1.0e-6, DType::Bf16).unwrap();
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
