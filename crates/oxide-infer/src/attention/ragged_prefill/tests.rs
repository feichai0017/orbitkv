use super::*;

fn bf16_values(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

#[test]
fn metadata_maps_bottom_right_causal_bounds() {
    let spec = Bf16RaggedPrefillSpec::new(2, 3, 6, 4, 2, 128).unwrap();
    let metadata = spec.validate_metadata(&[0, 2, 3], &[0, 4, 6]).unwrap();

    assert_eq!(metadata.request_row_ranges(0), Some(((0, 2), (0, 4))));
    assert_eq!(metadata.request_row_ranges(1), Some(((2, 3), (4, 6))));
    assert_eq!(metadata.request_row_ranges(2), None);
    assert_eq!(metadata.causal_kv_end(0, 0), Some(3));
    assert_eq!(metadata.causal_kv_end(0, 1), Some(4));
    assert_eq!(metadata.causal_kv_end(0, 2), None);
    assert_eq!(metadata.causal_kv_end(1, 0), Some(6));
}

#[test]
fn reference_applies_bottom_right_causal_mask() {
    let spec = Bf16RaggedPrefillSpec::new(1, 2, 4, 1, 1, 128).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key = vec![bf16::ZERO; spec.kv_numel()];
    let mut value = Vec::with_capacity(spec.kv_numel());
    for token in [1.0, 2.0, 3.0, 100.0] {
        value.extend(std::iter::repeat_n(
            bf16::from_f32(token),
            SINGLE_DECODE_HEAD_DIM,
        ));
    }
    let mut output = vec![bf16::NAN; spec.output_numel()];
    let mut lse = vec![f32::NAN; spec.lse_numel()];

    ragged_prefill_bf16_reference(
        &query,
        &key,
        &value,
        &[0, 2],
        &[0, 4],
        &mut output,
        &mut lse,
        spec,
    )
    .unwrap();

    let first = bf16::from_f32(2.0);
    let second = bf16::from_f32(26.5);
    assert!(
        output[..SINGLE_DECODE_HEAD_DIM]
            .iter()
            .all(|&value| value == first)
    );
    assert!(
        output[SINGLE_DECODE_HEAD_DIM..]
            .iter()
            .all(|&value| value == second)
    );
    assert_eq!(lse[0], 3.0_f32.log2());
    assert_eq!(lse[1], 4.0_f32.log2());
}

#[test]
fn reference_maps_gqa_heads_per_request() {
    let spec = Bf16RaggedPrefillSpec::new(2, 2, 2, 4, 2, 128).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key = vec![bf16::ZERO; spec.kv_numel()];
    let mut value = vec![bf16::ZERO; spec.kv_numel()];
    for token in 0..2 {
        let base = token * 2 * SINGLE_DECODE_HEAD_DIM;
        value[base..base + SINGLE_DECODE_HEAD_DIM].fill(bf16::from_f32(token as f32 + 1.0));
        value[base + SINGLE_DECODE_HEAD_DIM..base + 2 * SINGLE_DECODE_HEAD_DIM]
            .fill(bf16::from_f32(-(token as f32 + 1.0)));
    }
    let mut output = vec![bf16::NAN; spec.output_numel()];
    let mut lse = vec![f32::NAN; spec.lse_numel()];

    ragged_prefill_bf16_reference(
        &query,
        &key,
        &value,
        &[0, 1, 2],
        &[0, 1, 2],
        &mut output,
        &mut lse,
        spec,
    )
    .unwrap();

    let expected = bf16_values(
        &std::iter::repeat_n(1.0, 2 * SINGLE_DECODE_HEAD_DIM)
            .chain(std::iter::repeat_n(-1.0, 2 * SINGLE_DECODE_HEAD_DIM))
            .chain(std::iter::repeat_n(2.0, 2 * SINGLE_DECODE_HEAD_DIM))
            .chain(std::iter::repeat_n(-2.0, 2 * SINGLE_DECODE_HEAD_DIM))
            .collect::<Vec<_>>(),
    );
    assert_eq!(output, expected);
    assert_eq!(lse, [0.0; 8]);
}

#[test]
fn metadata_rejects_invalid_indptr_and_lengths() {
    let spec = Bf16RaggedPrefillSpec::new(2, 3, 5, 4, 2, 128).unwrap();
    let cases = [
        (
            spec.validate_metadata(&[0, 3], &[0, 2, 5]),
            ContractError::LengthMismatch {
                buffer: "qo_indptr",
                expected: 3,
                actual: 2,
            },
        ),
        (
            spec.validate_metadata(&[1, 2, 4], &[0, 2, 5]),
            ContractError::InvalidIndptrStart {
                buffer: "qo_indptr",
                actual: 1,
            },
        ),
        (
            spec.validate_metadata(&[0, 2, 1], &[0, 2, 5]),
            ContractError::NonMonotonicIndptr {
                buffer: "qo_indptr",
                request: 1,
                start: 2,
                end: 1,
            },
        ),
        (
            spec.validate_metadata(&[0, 0, 3], &[0, 2, 5]),
            ContractError::EmptyRaggedRequest {
                buffer: "qo_indptr",
                request: 0,
            },
        ),
        (
            spec.validate_metadata(&[0, 1, 3], &[0, 1, 4]),
            ContractError::LengthMismatch {
                buffer: "kv_indptr",
                expected: 5,
                actual: 4,
            },
        ),
        (
            spec.validate_metadata(&[0, 2, 3], &[0, 1, 5]),
            ContractError::RaggedQueryLongerThanKv {
                request: 0,
                query_len: 2,
                kv_len: 1,
            },
        ),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, Err(expected));
    }
}

#[test]
fn spec_rejects_shapes_and_overflow() {
    for shape in [
        (0, 1, 1, 1, 1, 128),
        (1, 0, 1, 1, 1, 128),
        (1, 1, 0, 1, 1, 128),
        (1, 1, 1, 0, 1, 128),
        (1, 1, 1, 1, 0, 128),
        (1, 1, 1, 1, 1, 0),
    ] {
        assert_eq!(
            Bf16RaggedPrefillSpec::new(shape.0, shape.1, shape.2, shape.3, shape.4, shape.5),
            Err(ContractError::ZeroDimension)
        );
    }
    assert_eq!(
        Bf16RaggedPrefillSpec::new(1, 1, 1, 1, 1, 64),
        Err(ContractError::UnsupportedHeadDimension {
            expected: 128,
            actual: 64,
        })
    );
    assert_eq!(
        Bf16RaggedPrefillSpec::new(1, 1, 1, 6, 4, 128),
        Err(ContractError::InvalidHeadMapping {
            query_heads: 6,
            kv_heads: 4,
        })
    );
    assert_eq!(
        Bf16RaggedPrefillSpec::new(1, i32::MAX as usize + 1, 1, 1, 1, 128),
        Err(ContractError::ElementCountOverflow)
    );
}

#[test]
fn reference_requires_exact_tensor_lengths() {
    let spec = Bf16RaggedPrefillSpec::new(1, 1, 1, 1, 1, 128).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key = vec![bf16::ZERO; spec.kv_numel()];
    let value = vec![bf16::ZERO; spec.kv_numel()];
    let output = vec![bf16::ZERO; spec.output_numel()];
    let lse = vec![0.0_f32; spec.lse_numel()];

    let cases = [
        (
            ragged_prefill_bf16_reference(
                &query[..query.len() - 1],
                &key,
                &value,
                &[0, 1],
                &[0, 1],
                &mut output.clone(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "Q",
                expected: query.len(),
                actual: query.len() - 1,
            },
        ),
        (
            ragged_prefill_bf16_reference(
                &query,
                &key[..key.len() - 1],
                &value,
                &[0, 1],
                &[0, 1],
                &mut output.clone(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "K",
                expected: key.len(),
                actual: key.len() - 1,
            },
        ),
        (
            ragged_prefill_bf16_reference(
                &query,
                &key,
                &value[..value.len() - 1],
                &[0, 1],
                &[0, 1],
                &mut output.clone(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "V",
                expected: value.len(),
                actual: value.len() - 1,
            },
        ),
        (
            ragged_prefill_bf16_reference(
                &query,
                &key,
                &value,
                &[0, 1],
                &[0, 1],
                &mut output[..output.len() - 1].to_vec(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "O",
                expected: output.len(),
                actual: output.len() - 1,
            },
        ),
        (
            ragged_prefill_bf16_reference(
                &query,
                &key,
                &value,
                &[0, 1],
                &[0, 1],
                &mut output.clone(),
                &mut [],
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "LSE",
                expected: lse.len(),
                actual: 0,
            },
        ),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, Err(expected));
    }
}
