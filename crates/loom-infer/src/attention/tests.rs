use super::*;

fn bf16_slice(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

#[test]
fn reference_matches_two_token_softmax() {
    let spec = Bf16SingleDecodeSpec::new(2, 1, 1, SINGLE_DECODE_HEAD_DIM).unwrap();
    let mut query = vec![bf16::ZERO; spec.query_numel()];
    let mut key = vec![bf16::ZERO; spec.kv_numel()];
    let mut value = vec![bf16::ZERO; spec.kv_numel()];
    query[0] = bf16::from_f32((SINGLE_DECODE_HEAD_DIM as f32).sqrt());
    key[SINGLE_DECODE_HEAD_DIM] = bf16::ONE;
    value[SINGLE_DECODE_HEAD_DIM..].fill(bf16::from_f32(2.0));
    let mut output = vec![bf16::ZERO; spec.output_numel()];
    let mut lse = vec![0.0_f32; spec.lse_numel()];

    single_decode_bf16_reference(&query, &key, &value, &mut output, &mut lse, spec).unwrap();

    let second_score = query[0].to_f32() * spec.softmax_scale();
    let second_probability = second_score.exp() / (1.0 + second_score.exp());
    let expected_output = bf16::from_f32(2.0 * second_probability);
    assert!(output.iter().all(|&value| value == expected_output));
    let expected_lse_log2 = (1.0 + second_score.exp()).ln() * core::f32::consts::LOG2_E;
    assert!((lse[0] - expected_lse_log2).abs() < 1.0e-6);
}

#[test]
fn reference_maps_gqa_heads_to_their_kv_head() {
    let spec = Bf16SingleDecodeSpec::new(1, 4, 2, SINGLE_DECODE_HEAD_DIM).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key = vec![bf16::ZERO; spec.kv_numel()];
    let mut value = vec![bf16::ZERO; spec.kv_numel()];
    value[..SINGLE_DECODE_HEAD_DIM].fill(bf16::from_f32(1.5));
    value[SINGLE_DECODE_HEAD_DIM..].fill(bf16::from_f32(-2.0));
    let mut output = vec![bf16::ZERO; spec.output_numel()];
    let mut lse = vec![f32::NAN; spec.lse_numel()];

    single_decode_bf16_reference(&query, &key, &value, &mut output, &mut lse, spec).unwrap();

    let expected = bf16_slice(
        &std::iter::repeat_n(1.5, 2 * SINGLE_DECODE_HEAD_DIM)
            .chain(std::iter::repeat_n(-2.0, 2 * SINGLE_DECODE_HEAD_DIM))
            .collect::<Vec<_>>(),
    );
    assert_eq!(output, expected);
    assert_eq!(lse, [0.0; 4]);
}

#[test]
fn reference_keeps_large_logits_finite() {
    let spec = Bf16SingleDecodeSpec::new(3, 1, 1, SINGLE_DECODE_HEAD_DIM).unwrap();
    let query = vec![bf16::from_f32(64.0); spec.query_numel()];
    let key = bf16_slice(
        &std::iter::repeat_n(-64.0, SINGLE_DECODE_HEAD_DIM)
            .chain(std::iter::repeat_n(0.0, SINGLE_DECODE_HEAD_DIM))
            .chain(std::iter::repeat_n(64.0, SINGLE_DECODE_HEAD_DIM))
            .collect::<Vec<_>>(),
    );
    let value = vec![bf16::ONE; spec.kv_numel()];
    let mut output = vec![bf16::ZERO; spec.output_numel()];
    let mut lse = vec![0.0; spec.lse_numel()];

    single_decode_bf16_reference(&query, &key, &value, &mut output, &mut lse, spec).unwrap();

    assert!(output.iter().all(|value| value.to_f32().is_finite()));
    assert!(lse[0].is_finite());
}

#[test]
fn reference_requires_all_exact_buffer_lengths() {
    let spec = Bf16SingleDecodeSpec::new(2, 2, 1, SINGLE_DECODE_HEAD_DIM).unwrap();
    let q = vec![bf16::ZERO; spec.query_numel()];
    let k = vec![bf16::ZERO; spec.kv_numel()];
    let v = vec![bf16::ZERO; spec.kv_numel()];
    let o = vec![bf16::ZERO; spec.output_numel()];
    let lse = vec![0.0; spec.lse_numel()];

    let cases = [
        (
            single_decode_bf16_reference(
                &q[..q.len() - 1],
                &k,
                &v,
                &mut o.clone(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "Q",
                expected: q.len(),
                actual: q.len() - 1,
            },
        ),
        (
            single_decode_bf16_reference(
                &q,
                &k[..k.len() - 1],
                &v,
                &mut o.clone(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "K",
                expected: k.len(),
                actual: k.len() - 1,
            },
        ),
        (
            single_decode_bf16_reference(
                &q,
                &k,
                &v[..v.len() - 1],
                &mut o.clone(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "V",
                expected: v.len(),
                actual: v.len() - 1,
            },
        ),
        (
            single_decode_bf16_reference(
                &q,
                &k,
                &v,
                &mut o[..o.len() - 1].to_vec(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "O",
                expected: o.len(),
                actual: o.len() - 1,
            },
        ),
        (
            single_decode_bf16_reference(
                &q,
                &k,
                &v,
                &mut o.clone(),
                &mut lse[..lse.len() - 1].to_vec(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "LSE",
                expected: lse.len(),
                actual: lse.len() - 1,
            },
        ),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, Err(expected));
    }
}

#[test]
fn spec_rejects_unsupported_shapes_and_overflow() {
    for dimensions in [(0, 1, 1, 128), (1, 0, 1, 128), (1, 1, 0, 128), (1, 1, 1, 0)] {
        assert_eq!(
            Bf16SingleDecodeSpec::new(dimensions.0, dimensions.1, dimensions.2, dimensions.3),
            Err(ContractError::ZeroDimension)
        );
    }
    assert_eq!(
        Bf16SingleDecodeSpec::new(1, 1, 1, 64),
        Err(ContractError::UnsupportedHeadDimension {
            expected: SINGLE_DECODE_HEAD_DIM,
            actual: 64,
        })
    );
    assert_eq!(
        Bf16SingleDecodeSpec::new(1, 6, 4, SINGLE_DECODE_HEAD_DIM),
        Err(ContractError::InvalidHeadMapping {
            query_heads: 6,
            kv_heads: 4,
        })
    );
    assert_eq!(
        Bf16SingleDecodeSpec::new(usize::MAX, 1, 1, SINGLE_DECODE_HEAD_DIM),
        Err(ContractError::ElementCountOverflow)
    );
    assert_eq!(
        Bf16SingleDecodeSpec::new(1, usize::MAX, 1, SINGLE_DECODE_HEAD_DIM),
        Err(ContractError::ElementCountOverflow)
    );
}

#[test]
fn spec_reports_fixed_shapes_and_mapping() {
    let spec = Bf16SingleDecodeSpec::new(7, 8, 2, SINGLE_DECODE_HEAD_DIM).unwrap();

    assert_eq!(spec.kv_len(), 7);
    assert_eq!(spec.num_query_heads(), 8);
    assert_eq!(spec.num_kv_heads(), 2);
    assert_eq!(spec.head_dim(), 128);
    assert_eq!(spec.gqa_group_size(), 4);
    assert_eq!(spec.kv_head_for_query_head(0), Some(0));
    assert_eq!(spec.kv_head_for_query_head(3), Some(0));
    assert_eq!(spec.kv_head_for_query_head(4), Some(1));
    assert_eq!(spec.kv_head_for_query_head(7), Some(1));
    assert_eq!(spec.kv_head_for_query_head(8), None);
    assert_eq!(spec.query_numel(), 8 * 128);
    assert_eq!(spec.kv_numel(), 7 * 2 * 128);
    assert_eq!(spec.output_numel(), 8 * 128);
    assert_eq!(spec.lse_numel(), 8);
}
