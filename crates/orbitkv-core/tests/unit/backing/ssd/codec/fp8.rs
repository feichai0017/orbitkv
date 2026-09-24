use super::*;

#[test]
fn rounding_subnormals_signed_zero_and_reencoding() {
    for format in [StorageFormat::Fp8FromBf16, StorageFormat::Fp8FromFp16] {
        let to_bits = |x| {
            if format == StorageFormat::Fp8FromBf16 {
                bf16::from_f32(x).to_bits()
            } else {
                f16::from_f32(x).to_bits()
            }
        };
        for (x, expected) in [
            (0.0, 0),
            (-0.0, 128),
            (1.0625, 0x38),
            (1.1875, 0x3a),
            (448.0, 0x7e),
            (1.0 / 512.0, 1),
            (1.0 / 1024.0, 0),
        ] {
            let mut output = [0];
            assert!(encode(format, &to_bits(x).to_le_bytes(), &mut output));
            assert_eq!(output, [expected]);
        }
        for bits in 0..=255u8 {
            if bits & 127 == 127 {
                continue;
            }
            let mut reconstructed = [0; 2];
            assert!(decode(format, &[bits], &mut reconstructed));
            let mut encoded = [0];
            assert!(encode(format, &reconstructed, &mut encoded));
            assert_eq!(encoded, [bits]);
        }
        for x in [f32::INFINITY, f32::NAN, 1024.0] {
            assert!(!encode(format, &to_bits(x).to_le_bytes(), &mut [0]));
        }
    }
}
