use super::*;

fn bf16_slice(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

#[test]
fn reference_multiplies_a_by_transposed_row_major_weight() {
    let spec = Bf16DenseGemmSpec::new(2, 3, 2).unwrap();
    let a = bf16_slice(&[1.0, 2.0, 3.0, 4.0]);
    let weight = bf16_slice(&[5.0, 6.0, 7.0, 8.0, -1.0, 0.5]);
    let mut output = vec![bf16::ZERO; spec.output_numel()];

    bf16_dense_gemm_reference(&a, &weight, &mut output, spec).unwrap();

    assert_eq!(output, bf16_slice(&[17.0, 23.0, 0.0, 39.0, 53.0, -1.0]));
}

#[test]
fn reference_accumulates_in_f32_and_rounds_output_once() {
    let spec = Bf16DenseGemmSpec::new(1, 1, 3).unwrap();
    let a = bf16_slice(&[1.0, 1.0, 1.0]);
    let weight = bf16_slice(&[1.0, 0.003_906_25, 0.003_906_25]);
    let mut output = [bf16::ZERO];

    bf16_dense_gemm_reference(&a, &weight, &mut output, spec).unwrap();

    // Rounding each partial sum to BF16 would lose both 1/256 terms. The
    // contract retains them in F32 and rounds only the completed dot product.
    assert_eq!(output[0], bf16::from_f32(1.007_812_5));
}

#[test]
fn reference_requires_exact_buffer_lengths() {
    let spec = Bf16DenseGemmSpec::new(2, 3, 4).unwrap();

    for (actual, expected) in [
        (
            bf16_dense_gemm_reference(
                &[bf16::ZERO; 7],
                &[bf16::ZERO; 12],
                &mut [bf16::ZERO; 6],
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "A",
                expected: 8,
                actual: 7,
            },
        ),
        (
            bf16_dense_gemm_reference(
                &[bf16::ZERO; 8],
                &[bf16::ZERO; 11],
                &mut [bf16::ZERO; 6],
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "W",
                expected: 12,
                actual: 11,
            },
        ),
        (
            bf16_dense_gemm_reference(
                &[bf16::ZERO; 8],
                &[bf16::ZERO; 12],
                &mut [bf16::ZERO; 5],
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "D",
                expected: 6,
                actual: 5,
            },
        ),
    ] {
        assert_eq!(actual, Err(expected));
    }
}

#[test]
fn spec_rejects_zero_dimensions_and_each_element_count_overflow() {
    for dimensions in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        assert_eq!(
            Bf16DenseGemmSpec::new(dimensions.0, dimensions.1, dimensions.2),
            Err(ContractError::ZeroDimension)
        );
    }

    for dimensions in [(usize::MAX, 1, 2), (1, usize::MAX, 2), (usize::MAX, 2, 1)] {
        assert_eq!(
            Bf16DenseGemmSpec::new(dimensions.0, dimensions.1, dimensions.2),
            Err(ContractError::ElementCountOverflow)
        );
    }
}

#[test]
fn spec_reports_fixed_tensor_shapes() {
    let spec = Bf16DenseGemmSpec::new(2, 3, 4).unwrap();

    assert_eq!(spec.m(), 2);
    assert_eq!(spec.n(), 3);
    assert_eq!(spec.k(), 4);
    assert_eq!(spec.a_numel(), 8);
    assert_eq!(spec.weight_numel(), 12);
    assert_eq!(spec.output_numel(), 6);
}
