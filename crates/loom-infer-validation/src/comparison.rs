//! Finite numerical comparisons shared by hardware validation programs.

use half::bf16;
use std::error::Error;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub max_abs: f32,
    pub bit_mismatches: usize,
    pub digest: u64,
}

pub fn digest_bf16(values: &[bf16]) -> u64 {
    values.iter().fold(FNV_OFFSET_BASIS, |digest, value| {
        (digest ^ u64::from(value.to_bits())).wrapping_mul(FNV_PRIME)
    })
}

pub fn compare_bf16(
    actual: &[bf16],
    expected: &[bf16],
    label: &str,
) -> Result<Comparison, Box<dyn Error>> {
    compare_bits(actual, expected, label, bf16::to_f32, |value| {
        u64::from(value.to_bits())
    })
}

pub fn compare_f32(
    actual: &[f32],
    expected: &[f32],
    label: &str,
) -> Result<Comparison, Box<dyn Error>> {
    compare_bits(
        actual,
        expected,
        label,
        |value| value,
        |value| u64::from(value.to_bits()),
    )
}

fn compare_bits<T: Copy>(
    actual: &[T],
    expected: &[T],
    label: &str,
    to_f32: impl Fn(T) -> f32,
    to_bits: impl Fn(T) -> u64,
) -> Result<Comparison, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} comparison length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        )
        .into());
    }

    let mut comparison = Comparison {
        max_abs: 0.0,
        bit_mismatches: 0,
        digest: FNV_OFFSET_BASIS,
    };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual_f32 = to_f32(actual);
        if !actual_f32.is_finite() {
            return Err(format!("non-finite {label} output at index {index}").into());
        }
        comparison.max_abs = comparison
            .max_abs
            .max((actual_f32 - to_f32(expected)).abs());
        let actual_bits = to_bits(actual);
        comparison.bit_mismatches += usize::from(actual_bits != to_bits(expected));
        comparison.digest ^= actual_bits;
        comparison.digest = comparison.digest.wrapping_mul(FNV_PRIME);
    }
    Ok(comparison)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_comparison_tracks_error_mismatches_and_digest() {
        let actual = [bf16::from_f32(1.0), bf16::from_f32(2.0)];
        let expected = [bf16::from_f32(1.0), bf16::from_f32(3.0)];

        let comparison = compare_bf16(&actual, &expected, "BF16").unwrap();

        assert_eq!(comparison.max_abs, 1.0);
        assert_eq!(comparison.bit_mismatches, 1);
        assert_eq!(comparison.digest, 0xf25b_8807_a293_a56d);
        assert_eq!(comparison.digest, digest_bf16(&actual));
    }

    #[test]
    fn comparison_rejects_non_finite_values() {
        let error = compare_f32(&[f32::NAN], &[0.0], "F32").unwrap_err();

        assert!(error.to_string().contains("non-finite F32 output"));
    }

    #[test]
    fn comparison_rejects_different_lengths() {
        let error = compare_f32(&[0.0], &[], "F32").unwrap_err();

        assert!(error.to_string().contains("comparison length mismatch"));
    }
}
