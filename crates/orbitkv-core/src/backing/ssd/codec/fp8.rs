use std::sync::OnceLock;

use half::{bf16, f16};
use orbitkv_state::StorageFormat;

struct Tables {
    encode: Box<[u16]>,
    decode: [u16; 256],
}

// E4M3FN values, including subnormals. The sole positive NaN is 0x7f.
fn value(bits: u8) -> f32 {
    let exponent = (bits >> 3) & 15;
    let mantissa = bits & 7;
    let magnitude = if exponent == 0 {
        mantissa as f32 / 512.0
    } else if exponent == 15 && mantissa == 7 {
        f32::NAN
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    };
    if bits & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn tables(format: StorageFormat) -> &'static Tables {
    static BF16: OnceLock<Tables> = OnceLock::new();
    static FP16: OnceLock<Tables> = OnceLock::new();
    let table = match format {
        StorageFormat::Fp8FromBf16 => &BF16,
        StorageFormat::Fp8FromFp16 => &FP16,
        StorageFormat::Exact => unreachable!("exact segments bypass quantization"),
    };
    table.get_or_init(|| {
        let bf = format == StorageFormat::Fp8FromBf16;
        let levels: Vec<_> = (0..127).map(value).collect();
        let encode = (0..=u16::MAX)
            .map(|bits| {
                let input = if bf {
                    bf16::from_bits(bits).to_f32()
                } else {
                    f16::from_bits(bits).to_f32()
                };
                let magnitude = input.abs();
                // Preserve nonfinite values and outliers exactly by declining the object.
                if !input.is_finite() || magnitude > 448.0 {
                    return 256;
                }
                let upper = levels.partition_point(|v| *v < magnitude);
                let lower = upper.saturating_sub(1);
                let down = magnitude - levels[lower];
                let up = levels[upper] - magnitude;
                let index = if down < up || (down == up && lower.is_multiple_of(2)) {
                    lower
                } else {
                    upper
                };
                index as u16 | ((bits >> 8) & 128)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let decode = std::array::from_fn(|bits| {
            if bf {
                bf16::from_f32(value(bits as u8)).to_bits()
            } else {
                f16::from_f32(value(bits as u8)).to_bits()
            }
        });
        Tables { encode, decode }
    })
}

pub(super) fn encode(format: StorageFormat, input: &[u8], output: &mut [u8]) -> bool {
    if input.len() != output.len() * 2 {
        return false;
    }
    let table = tables(format);
    for (value, out) in input.chunks_exact(2).zip(output) {
        let encoded = table.encode[u16::from_le_bytes([value[0], value[1]]) as usize];
        if encoded > 255 {
            return false;
        }
        *out = encoded as u8;
    }
    true
}

pub(super) fn decode(format: StorageFormat, input: &[u8], output: &mut [u8]) -> bool {
    if output.len() != input.len() * 2 {
        return false;
    }
    let table = tables(format);
    for (&value, out) in input.iter().zip(output.chunks_exact_mut(2)) {
        out.copy_from_slice(&table.decode[value as usize].to_le_bytes());
    }
    true
}

#[cfg(test)]
#[path = "../../../../tests/unit/backing/ssd/codec/fp8.rs"]
mod tests;
