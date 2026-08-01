use crate::*;
use half::{bf16, f16};

#[test]
fn silu_and_mul_matches_split_half_definition() {
    let spec = SiluAndMulSpec::new(1, 3, DType::F32).unwrap();
    let input = [0.0_f32, 1.0, -1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 3];

    silu_and_mul_f32_reference(&input, &mut output, spec).unwrap();

    assert_eq!(output[0], 0.0);
    assert!((output[1] - 3.0 / (1.0 + (-1.0_f32).exp())).abs() < 1.0e-6);
    assert!((output[2] - (-4.0 / (1.0 + 1.0_f32.exp()))).abs() < 1.0e-6);
}

#[test]
fn silu_and_mul_low_precision_rounds_activation_before_multiply() {
    let spec = SiluAndMulSpec::new(1, 1, DType::F16).unwrap();
    let gate = f16::from_f32(0.3333);
    let up = f16::from_f32(1.7777);
    let mut output = [f16::ZERO];

    silu_and_mul_f16_reference(&[gate, up], &mut output, spec).unwrap();

    let gate_f32 = gate.to_f32();
    let activated = f16::from_f32(gate_f32 / (1.0 + (-gate_f32).exp()));
    let expected = f16::from_f32(activated.to_f32() * up.to_f32());
    assert_eq!(output[0], expected);
}

#[test]
fn silu_and_mul_validates_buffer_lengths_and_dtype() {
    let spec = SiluAndMulSpec::new(2, 4, DType::Bf16).unwrap();
    let error =
        silu_and_mul_bf16_reference(&[bf16::ZERO; 15], &mut [bf16::ZERO; 8], spec).unwrap_err();
    assert_eq!(
        error,
        ContractError::LengthMismatch {
            buffer: "input",
            expected: 16,
            actual: 15,
        }
    );

    let wrong_dtype = SiluAndMulSpec::new(1, 2, DType::F32).unwrap();
    assert_eq!(
        silu_and_mul_f16_reference(&[f16::ZERO; 4], &mut [f16::ZERO; 2], wrong_dtype,),
        Err(ContractError::UnsupportedDType(DType::F32))
    );
}
#[test]
fn silu_and_mul_dynamic_fp8_validates_group_contract() {
    assert_eq!(
        SiluAndMulDynamicFp8Spec::new(1, 128, 32, DType::F16),
        Err(ContractError::InvalidGroupSize(32))
    );
    assert_eq!(
        SiluAndMulDynamicFp8Spec::new(1, 192, 128, DType::Bf16),
        Err(ContractError::WidthNotDivisible {
            width: 192,
            group_size: 128,
        })
    );
}

#[test]
fn silu_and_mul_dynamic_fp8_uses_f32_activation_and_per_group_scales() {
    let spec = SiluAndMulDynamicFp8Spec::new(1, 128, 64, DType::F16).unwrap();
    let gate_value = f16::from_f32(0.3333);
    let up_value = f16::from_f32(1.7777);
    let mut input = [f16::ZERO; 256];
    input[0] = gate_value;
    input[128] = up_value;
    let mut output = [0_u8; 128];
    let mut scales = [0.0_f32; 2];

    silu_and_mul_dynamic_fp8_f16_reference(&input, &mut output, &mut scales, spec).unwrap();

    let gate_f32 = gate_value.to_f32();
    let sigmoid_gate = 1.0 / (1.0 + (-gate_f32).exp());
    let full_precision = gate_f32 * sigmoid_gate * up_value.to_f32();
    assert_eq!(scales[0], full_precision.abs() / FP8_E4M3FN_MAX);
    assert_eq!(scales[1], DYNAMIC_FP8_MIN_SCALE);
    assert_eq!(output[0], 0x7e);
    assert!(output[1..].iter().all(|&value| value == 0));
}

#[test]
fn silu_and_mul_dynamic_fp8_validates_buffers_and_dtype() {
    let spec = SiluAndMulDynamicFp8Spec::new(2, 64, 64, DType::Bf16).unwrap();
    let error = silu_and_mul_dynamic_fp8_bf16_reference(
        &[bf16::ZERO; 256],
        &mut [0_u8; 127],
        &mut [0.0_f32; 2],
        spec,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ContractError::LengthMismatch {
            buffer: "output",
            expected: 128,
            actual: 127,
        }
    );

    let wrong_dtype = SiluAndMulDynamicFp8Spec::new(1, 64, 64, DType::F32).unwrap();
    assert_eq!(
        silu_and_mul_dynamic_fp8_f16_reference(
            &[f16::ZERO; 128],
            &mut [0_u8; 64],
            &mut [0.0_f32; 1],
            wrong_dtype,
        ),
        Err(ContractError::UnsupportedDType(DType::F32))
    );
}

#[test]
fn silu_and_mul_dynamic_int8_matches_compiled_bf16_path() {
    let spec = SiluAndMulDynamicInt8Spec::new(2, 4, DType::Bf16).unwrap();
    let input = [
        bf16::from_f32(0.3333),
        bf16::from_f32(-0.75),
        bf16::ZERO,
        bf16::from_f32(2.0),
        bf16::from_f32(1.7777),
        bf16::from_f32(0.5),
        bf16::from_f32(3.0),
        bf16::from_f32(-1.0),
        bf16::ZERO,
        bf16::ZERO,
        bf16::ZERO,
        bf16::ZERO,
        bf16::from_f32(1.0),
        bf16::from_f32(-1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(-2.0),
    ];
    let mut output = [0_i8; 8];
    let mut scales = [f32::NAN; 2];

    silu_and_mul_dynamic_int8_bf16_reference(&input, &mut output, &mut scales, spec).unwrap();

    let mut materialized = [bf16::ZERO; 8];
    for row in 0..2 {
        let input_offset = row * 8;
        for column in 0..4 {
            let gate = input[input_offset + column].to_f32();
            let up = input[input_offset + 4 + column].to_f32();
            materialized[row * 4 + column] = bf16::from_f32(gate / (1.0 + (-gate).exp()) * up);
        }
    }
    for row in 0..2 {
        let values = &materialized[row * 4..row * 4 + 4];
        let absolute_maximum = values
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0_f32, f32::max);
        assert_eq!(scales[row], dynamic_int8_scale(absolute_maximum));
        for (column, value) in values.iter().enumerate() {
            assert_eq!(
                output[row * 4 + column],
                quantize_dynamic_int8(value.to_f32(), absolute_maximum)
            );
        }
    }
    assert_eq!(scales[1], 0.0);
    assert!(output[4..].iter().all(|&value| value == 0));
}

#[test]
fn silu_and_mul_dynamic_int8_validates_buffers_and_dtype() {
    let spec = SiluAndMulDynamicInt8Spec::new(2, 4, DType::F16).unwrap();
    let error = silu_and_mul_dynamic_int8_f16_reference(
        &[f16::ZERO; 16],
        &mut [0_i8; 7],
        &mut [0.0_f32; 2],
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

    let wrong_dtype = SiluAndMulDynamicInt8Spec::new(1, 4, DType::F32).unwrap();
    assert_eq!(
        silu_and_mul_dynamic_int8_f16_reference(
            &[f16::ZERO; 8],
            &mut [0_i8; 4],
            &mut [0.0_f32; 1],
            wrong_dtype,
        ),
        Err(ContractError::UnsupportedDType(DType::F32))
    );
}
