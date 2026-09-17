//! The element codec: every float dtype round-trips within its precision,
//! and ulp distance treats the two zeros as one value.

use kern_manifest::types::DType;
use kern_manifest::values::{from_f64, to_f64, ulp_distance};

#[test]
fn floats_roundtrip_within_their_precision() {
    for dt in [DType::Bf16, DType::F16, DType::F32] {
        let vals = [0.0, -1.5, 3.25, 1e-3, -1e2];
        let dec = to_f64(dt, &from_f64(dt, &vals));
        for (a, b) in vals.iter().zip(&dec) {
            assert!((a - b).abs() <= a.abs() * 1e-2, "{dt}: {a} vs {b}");
        }
    }
    // fp8 e4m3: three mantissa bits, so within a sixteenth; 0x7f is its NaN.
    let vals = [0.0, 1.0, -1.0, 0.5, 448.0, 0.001953125, 2.5, -13.0];
    let enc = from_f64(DType::Fp8E4m3, &vals);
    for (v, back) in vals.iter().zip(to_f64(DType::Fp8E4m3, &enc)) {
        assert!((back - v).abs() <= v.abs() * 0.0625 + 1e-9, "{v} -> {back}");
    }
    assert!(to_f64(DType::Fp8E4m3, &[0x7f])[0].is_nan());
}

#[test]
fn ulps() {
    let a = from_f64(DType::Bf16, &[1.0]);
    let b = from_f64(DType::Bf16, &[1.0078125]); // next bf16 above 1.0
    assert_eq!(ulp_distance(DType::Bf16, &a, &b), Some(1));
    let z = from_f64(DType::F32, &[0.0]);
    let nz = from_f64(DType::F32, &[-0.0]);
    assert_eq!(ulp_distance(DType::F32, &z, &nz), Some(0));
    let p = from_f64(DType::F32, &[1e-45]);
    let n = from_f64(DType::F32, &[-1e-45]);
    assert_eq!(ulp_distance(DType::F32, &p, &n), Some(2));
}

#[test]
fn microscale_and_signed_byte_preserve_checkpoint_bits() {
    let codes: Vec<u8> = (0..=254).collect();
    let scales = to_f64(DType::Fp8E8m0, &codes);
    assert_eq!(scales[0], 2f64.powi(-127));
    assert_eq!(scales[127], 1.0);
    assert_eq!(scales[254], 2f64.powi(127));
    assert_eq!(from_f64(DType::Fp8E8m0, &scales), codes);
    assert!(to_f64(DType::Fp8E8m0, &[255])[0].is_nan());
    assert_eq!(ulp_distance(DType::Fp8E8m0, &[126], &[128]), Some(2));
    assert_eq!(ulp_distance(DType::Fp8E8m0, &[255], &[128]), None);
    let bytes: Vec<u8> = (0..=255).collect();
    let signed = to_f64(DType::I8, &bytes);
    assert_eq!(signed[128], -128.0);
    assert_eq!(signed[255], -1.0);
    assert_eq!(from_f64(DType::I8, &signed), bytes);
}
