use super::*;

#[test]
fn storage_encodings_are_borrowed_and_never_reinterpreted_by_width_alone() {
    let bytes = [0_u8, 127, 128, 255];
    for (source, target) in [
        (Dtype::F32, DType::F32),
        (Dtype::F16, DType::F16),
        (Dtype::BF16, DType::Bf16),
        (Dtype::U8, DType::U8),
        (Dtype::F8_E4M3, DType::F8E4M3),
        (Dtype::F8_E5M2, DType::F8E5M2),
        (Dtype::F8_E8M0, DType::F8UE8M0),
        (Dtype::U8, DType::F8UE8M0),
    ] {
        let data = Conversion::for_dtypes(source, target)
            .unwrap()
            .apply(&bytes)
            .unwrap();
        assert_eq!(data.as_bytes().as_ptr(), bytes.as_ptr());
        assert_eq!(data.as_bytes(), bytes);
    }
    for (source, target) in [
        (Dtype::F8_E4M3, DType::F8E5M2),
        (Dtype::U8, DType::F8E4M3),
        (Dtype::I32, DType::F32),
    ] {
        assert!(Conversion::for_dtypes(source, target).is_err());
    }
}

#[test]
fn unaligned_half_inputs_preserve_all_finite_values_and_nan_classification() {
    let mut bytes = vec![0_u8];
    for bits in 0..=u16::MAX {
        bytes.extend(bits.to_le_bytes());
    }
    for (conversion, mantissa_bits, bias) in [
        (Conversion::F16ToF32, 10, 15),
        (Conversion::Bf16ToF32, 7, 127),
    ] {
        let values = conversion.apply(&bytes[1..]).unwrap();
        let values: &[f32] = bytemuck::cast_slice(values.as_bytes());
        for (bits, &value) in values.iter().enumerate() {
            let sign = if bits & 0x8000 != 0 { -1.0_f64 } else { 1.0 };
            let exponent = (bits & 0x7fff) >> mantissa_bits;
            let fraction = bits & ((1 << mantissa_bits) - 1);
            let maximum_exponent = (1 << (15 - mantissa_bits)) - 1;
            let expected = if exponent == maximum_exponent {
                if fraction == 0 {
                    sign * f64::INFINITY
                } else {
                    f64::NAN
                }
            } else if exponent == 0 {
                sign * (fraction as f64) * 2.0_f64.powi(1 - bias - mantissa_bits)
            } else {
                sign * (1.0 + fraction as f64 / (1 << mantissa_bits) as f64)
                    * 2.0_f64.powi(exponent as i32 - bias)
            } as f32;
            if expected.is_nan() {
                assert!(value.is_nan());
            } else {
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "{conversion:?} bits={bits:x}"
                );
            }
        }
    }
}

#[test]
fn float_narrowing_uses_round_to_nearest_even_and_cross_half_conversion() {
    // Midpoints around one, signed zero, overflow and a representable subnormal.
    let floats = [
        0.0_f32,
        -0.0,
        1.0 + 2.0_f32.powi(-11),
        1.0 + 3.0 * 2.0_f32.powi(-11),
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0_f32.powi(-24),
    ];
    let half = Conversion::F32ToF16
        .apply(bytemuck::cast_slice(&floats))
        .unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, u16>(half.as_bytes()),
        &[0, 0x8000, 0x3c00, 0x3c02, 0x7c00, 0xfc00, 1]
    );
    let floats = [1.0_f32 + 2.0_f32.powi(-8), 1.0 + 3.0 * 2.0_f32.powi(-8)];
    let bfloat = Conversion::F32ToBf16
        .apply(bytemuck::cast_slice(&floats))
        .unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, u16>(bfloat.as_bytes()),
        &[0x3f80, 0x3f82]
    );
    let half = [0x3e00_u16, 0xc080]; // 1.5 and -2.25
    let converted = Conversion::F16ToBf16
        .apply(bytemuck::cast_slice(&half))
        .unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, u16>(converted.as_bytes()),
        &[0x3fc0, 0xc010]
    );
    let restored = Conversion::Bf16ToF16.apply(converted.as_bytes()).unwrap();
    assert_eq!(restored.as_bytes(), bytemuck::cast_slice::<_, u8>(&half));
}

#[test]
fn incomplete_source_elements_are_errors_and_empty_inputs_are_valid() {
    for conversion in [
        Conversion::F16ToF32,
        Conversion::Bf16ToF32,
        Conversion::F32ToF16,
        Conversion::Bf16ToF16,
        Conversion::F32ToBf16,
        Conversion::F16ToBf16,
    ] {
        assert!(conversion.apply(&[0]).is_err());
        assert!(conversion.apply(&[]).unwrap().as_bytes().is_empty());
    }
}
