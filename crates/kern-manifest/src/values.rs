//! Element codec: raw buffer bytes <-> f64 per manifest dtype. Used for
//! domain checks on input writes and by `kern test` (synthesis, comparison).
//! Lossless for every integer dtype below 2^53 and for every float dtype.

use crate::types::DType;
use half::{bf16, f16};

/// Decode `bytes` (a whole number of elements) into f64.
pub fn to_f64(dtype: DType, bytes: &[u8]) -> Vec<f64> {
    let n = dtype.bytes() as usize;
    bytes
        .chunks_exact(n)
        .map(|c| match dtype {
            DType::Bf16 => bf16::from_le_bytes([c[0], c[1]]).to_f64(),
            DType::F16 => f16::from_le_bytes([c[0], c[1]]).to_f64(),
            DType::F32 => f32::from_le_bytes(c.try_into().unwrap()) as f64,
            DType::Fp8E4m3 => fp8_e4m3_to_f64(c[0]),
            DType::Fp8E8m0 => {
                if c[0] == 255 {
                    f64::NAN
                } else {
                    2f64.powi(c[0] as i32 - 127)
                }
            }
            DType::I8 => (c[0] as i8) as f64,
            DType::I32 => i32::from_le_bytes(c.try_into().unwrap()) as f64,
            DType::U32 => u32::from_le_bytes(c.try_into().unwrap()) as f64,
            DType::I64 => i64::from_le_bytes(c.try_into().unwrap()) as f64,
            DType::U64 => u64::from_le_bytes(c.try_into().unwrap()) as f64,
            DType::U8 => c[0] as f64,
        })
        .collect()
}

/// Encode f64 values as `dtype` (round-to-nearest for floats, truncation
/// for integers — callers synthesizing integers pass integral values).
pub fn from_f64(dtype: DType, vals: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * dtype.bytes() as usize);
    for &v in vals {
        match dtype {
            DType::Bf16 => out.extend_from_slice(&bf16::from_f64(v).to_le_bytes()),
            DType::F16 => out.extend_from_slice(&f16::from_f64(v).to_le_bytes()),
            DType::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
            DType::Fp8E4m3 => out.push(f64_to_fp8_e4m3(v)),
            DType::Fp8E8m0 => out.push(f64_to_fp8_e8m0(v)),
            DType::I8 => out.push(v as i8 as u8),
            DType::I32 => out.extend_from_slice(&(v as i32).to_le_bytes()),
            DType::U32 => out.extend_from_slice(&(v as u32).to_le_bytes()),
            DType::I64 => out.extend_from_slice(&(v as i64).to_le_bytes()),
            DType::U64 => out.extend_from_slice(&(v as u64).to_le_bytes()),
            DType::U8 => out.push(v as u8),
        }
    }
    out
}

/// Distance in representable steps between two same-dtype float values
/// (0 for bit-equal; `None` for integer dtypes or NaN involvement).
pub fn ulp_distance(dtype: DType, a: &[u8], b: &[u8]) -> Option<u64> {
    let key = |c: &[u8]| -> Option<i64> {
        // Sign-magnitude -> monotone integer key so adjacent floats differ by 1.
        let (bits, sign_bit): (i64, i64) = match dtype {
            DType::Bf16 | DType::F16 => (u16::from_le_bytes([c[0], c[1]]) as i64, 1 << 15),
            DType::F32 => (u32::from_le_bytes(c.try_into().unwrap()) as i64, 1 << 31),
            DType::Fp8E4m3 => (c[0] as i64, 1 << 7),
            DType::Fp8E8m0 => return (c[0] != 255).then_some(c[0] as i64),
            _ => return None,
        };
        Some(if bits & sign_bit != 0 { sign_bit - (bits & (sign_bit - 1)) - sign_bit } else { bits })
    };
    let (fa, fb) = (to_f64(dtype, a)[0], to_f64(dtype, b)[0]);
    if fa.is_nan() || fb.is_nan() {
        return None;
    }
    if fa == fb {
        return Some(0); // +0 == -0
    }
    Some((key(a)? - key(b)?).unsigned_abs())
}

fn fp8_e4m3_to_f64(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xf) as i32;
    let man = (b & 7) as f64;
    if exp == 0xf && man == 7.0 {
        return f64::NAN;
    }
    sign * if exp == 0 { man / 8.0 * 2f64.powi(-6) } else { (1.0 + man / 8.0) * 2f64.powi(exp - 7) }
}

fn f64_to_fp8_e4m3(v: f64) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let sign = if v.is_sign_negative() { 0x80 } else { 0 };
    let a = v.abs().min(448.0);
    if a < 2f64.powi(-9) {
        return sign; // rounds to zero
    }
    let exp = a.log2().floor() as i32;
    if exp < -6 {
        let man = (a / 2f64.powi(-6) * 8.0).round() as u8;
        return sign | man.min(7);
    }
    let man = ((a / 2f64.powi(exp) - 1.0) * 8.0).round();
    let (exp, man) = if man >= 8.0 { (exp + 1, 0.0) } else { (exp, man) };
    let e = (exp + 7).clamp(0, 15) as u8;
    let m = man as u8;
    if e == 15 && m == 7 {
        return sign | 0x7e; // max finite, never NaN
    }
    sign | (e << 3) | m
}

// E8M0 has only positive powers of two and NaN; zero saturates to the
// smallest value. Nearest-value ties choose the even encoded exponent.
fn f64_to_fp8_e8m0(v: f64) -> u8 {
    if !v.is_finite() || v < 0.0 {
        return 255;
    }
    if v <= 2f64.powi(-127) {
        return 0;
    }
    if v >= 2f64.powi(127) {
        return 254;
    }
    let exponent = v.log2().floor() as i32;
    let code = (exponent + 127) as u8;
    let midpoint = 1.5 * 2f64.powi(exponent);
    if v > midpoint || (v == midpoint && !code.is_multiple_of(2)) {
        code + 1
    } else {
        code
    }
}
