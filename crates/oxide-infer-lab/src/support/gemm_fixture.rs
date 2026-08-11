pub(crate) const CENSUS_SHAPES: [(usize, usize, usize); 5] = [
    (1, 1_536, 1_536),
    (1, 256, 1_536),
    (1, 17_920, 1_536),
    (1, 1_536, 8_960),
    (1, 151_936, 1_536),
];
pub(crate) const MINIMUM_SHAPE: (usize, usize, usize) = (1, 16, 64);

pub(crate) fn exact_activation_value(column: usize) -> f32 {
    ((column % 8) + 1) as f32 / 64.0
}

pub(crate) fn exact_weight_value(row: usize, column: usize) -> f32 {
    (((row * 3 + column * 5) % 16) + 1) as f32 / 256.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transposed_dot(row: usize, k: usize) -> f32 {
        (0..k).fold(0.0_f32, |sum, reduction| {
            exact_activation_value(reduction).mul_add(exact_weight_value(row, reduction), sum)
        })
    }

    fn untransposed_dot(output_column: usize, n: usize, k: usize) -> f32 {
        (0..k).fold(0.0_f32, |sum, reduction| {
            let flat_index = reduction * n + output_column;
            let stored_row = flat_index / k;
            let stored_column = flat_index % k;
            exact_activation_value(reduction)
                .mul_add(exact_weight_value(stored_row, stored_column), sum)
        })
    }

    #[test]
    fn exact_fixture_is_nonzero_and_transpose_sensitive_for_every_gate_shape() {
        for (_, n, k) in std::iter::once(MINIMUM_SHAPE).chain(CENSUS_SHAPES) {
            let sample_columns = usize::min(n, 16);
            let expected = (0..sample_columns)
                .map(|row| transposed_dot(row, k))
                .collect::<Vec<_>>();
            assert!(
                expected.iter().all(|&value| value != 0.0),
                "exact fixture produced a zero sample for N={n}, K={k}: {expected:?}"
            );
            assert!(
                expected
                    .iter()
                    .enumerate()
                    .any(|(column, &value)| { value != untransposed_dot(column, n, k) }),
                "exact fixture does not distinguish W[N,K]^T for N={n}, K={k}"
            );
        }
    }

    #[test]
    fn exact_fixture_products_fit_the_covered_f32_integer_bound() {
        for row in 0..16 {
            for column in 0..16 {
                let scaled =
                    exact_activation_value(column) * exact_weight_value(row, column) * 16_384.0;
                assert_eq!(scaled, scaled.round());
                assert!(scaled <= 128.0);
            }
        }
        let max_covered_k = CENSUS_SHAPES.iter().map(|shape| shape.2).max().unwrap();
        assert!(max_covered_k * 128 < 1 << 24);
    }
}
