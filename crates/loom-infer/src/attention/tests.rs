use super::*;

fn bf16_slice(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn deterministic_bf16(len: usize, salt: u64) -> Vec<bf16> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let signed = (state % 2001) as i32 - 1000;
        values.push(bf16::from_f32(signed as f32 / 2048.0));
    }
    values
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

#[test]
fn split_k_spec_balances_contiguous_nonempty_partitions() {
    let decode = Bf16SingleDecodeSpec::new(10, 4, 2, SINGLE_DECODE_HEAD_DIM).unwrap();
    let spec = Bf16SingleDecodeSplitKSpec::new(decode, 4).unwrap();

    assert_eq!(spec.decode(), decode);
    assert_eq!(spec.partitions(), 4);
    assert_eq!(
        (0..4)
            .map(|partition| spec.partition_token_range(partition).unwrap())
            .collect::<Vec<_>>(),
        [(0, 3), (3, 6), (6, 8), (8, 10)]
    );
    assert_eq!(spec.partition_token_range(4), None);
    assert_eq!(spec.partial_state_width(), 130);
    assert_eq!(spec.partial_state_count(), 16);
    assert_eq!(spec.workspace_numel(), 16 * 130);
    assert_eq!(spec.workspace_bytes(), 16 * 130 * size_of::<f32>());
    assert_eq!(spec.partial_state_offset(0, 0), Some(0));
    assert_eq!(spec.partial_state_offset(0, 3), Some(3 * 130));
    assert_eq!(spec.partial_state_offset(1, 0), Some(4 * 130));
    assert_eq!(spec.partial_state_offset(4, 0), None);
    assert_eq!(spec.partial_state_offset(0, 4), None);
}

#[test]
fn split_k_spec_rejects_invalid_counts_and_workspace_overflow() {
    let decode = Bf16SingleDecodeSpec::new(7, 8, 2, SINGLE_DECODE_HEAD_DIM).unwrap();
    for partitions in [0, 8] {
        assert_eq!(
            Bf16SingleDecodeSplitKSpec::new(decode, partitions),
            Err(ContractError::InvalidPartitionCount {
                partitions,
                kv_len: 7,
            })
        );
    }

    let enormous_heads = usize::MAX / SINGLE_DECODE_HEAD_DIM;
    let decode = Bf16SingleDecodeSpec::new(1, enormous_heads, 1, SINGLE_DECODE_HEAD_DIM).unwrap();
    assert_eq!(
        Bf16SingleDecodeSplitKSpec::new(decode, 1),
        Err(ContractError::ElementCountOverflow)
    );
}

#[test]
fn split_k_partials_expose_log2_softmax_state_layout() {
    let decode = Bf16SingleDecodeSpec::new(2, 1, 1, SINGLE_DECODE_HEAD_DIM).unwrap();
    let spec = Bf16SingleDecodeSplitKSpec::new(decode, 2).unwrap();
    let mut query = vec![bf16::ZERO; decode.query_numel()];
    let mut key = vec![bf16::ZERO; decode.kv_numel()];
    let mut value = vec![bf16::ZERO; decode.kv_numel()];
    query[0] = bf16::from_f32((SINGLE_DECODE_HEAD_DIM as f32).sqrt());
    key[SINGLE_DECODE_HEAD_DIM] = bf16::ONE;
    value[..SINGLE_DECODE_HEAD_DIM].fill(bf16::from_f32(1.5));
    value[SINGLE_DECODE_HEAD_DIM..].fill(bf16::from_f32(-2.0));
    let mut workspace = vec![f32::NAN; spec.workspace_numel()];

    single_decode_bf16_split_k_partials_reference(&query, &key, &value, &mut workspace, spec)
        .unwrap();

    let first = &workspace[..SINGLE_DECODE_PARTIAL_STATE_WIDTH];
    let second = &workspace[SINGLE_DECODE_PARTIAL_STATE_WIDTH..];
    assert_eq!(first[0], 0.0);
    assert_eq!(first[1], 1.0);
    assert!(first[2..].iter().all(|&value| value == 1.5));
    let second_score_log2 = query[0].to_f32() * decode.softmax_scale() * core::f32::consts::LOG2_E;
    assert_eq!(second[0], second_score_log2);
    assert_eq!(second[1], 1.0);
    assert!(second[2..].iter().all(|&value| value == -2.0));
}

#[test]
fn split_k_reference_matches_direct_decode_across_shapes() {
    for (kv_len, query_heads, kv_heads, partitions, salt) in [
        (1, 8, 8, 1, 0x1001),
        (7, 8, 1, 3, 0x2001),
        (33, 8, 1, 8, 0x3001),
        (127, 16, 4, 8, 0x4001),
    ] {
        let decode =
            Bf16SingleDecodeSpec::new(kv_len, query_heads, kv_heads, SINGLE_DECODE_HEAD_DIM)
                .unwrap();
        let split = Bf16SingleDecodeSplitKSpec::new(decode, partitions).unwrap();
        let query = deterministic_bf16(decode.query_numel(), salt);
        let key = deterministic_bf16(decode.kv_numel(), salt ^ 0x4b45_5900);
        let value = deterministic_bf16(decode.kv_numel(), salt ^ 0x5641_4c55_4500);
        let mut expected_output = vec![bf16::ZERO; decode.output_numel()];
        let mut expected_lse = vec![0.0; decode.lse_numel()];
        single_decode_bf16_reference(
            &query,
            &key,
            &value,
            &mut expected_output,
            &mut expected_lse,
            decode,
        )
        .unwrap();
        let mut workspace = vec![f32::NAN; split.workspace_numel()];
        let mut actual_output = vec![bf16::NAN; decode.output_numel()];
        let mut actual_lse = vec![f32::NAN; decode.lse_numel()];

        single_decode_bf16_split_k_reference(
            &query,
            &key,
            &value,
            &mut workspace,
            &mut actual_output,
            &mut actual_lse,
            split,
        )
        .unwrap();

        assert_eq!(
            actual_output, expected_output,
            "output mismatch for KV={kv_len}, QH={query_heads}, KVH={kv_heads}, split={partitions}"
        );
        for (actual, expected) in actual_lse.into_iter().zip(expected_lse) {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "LSE mismatch for KV={kv_len}, QH={query_heads}, KVH={kv_heads}, split={partitions}: {actual} versus {expected}"
            );
        }
    }
}

#[test]
fn split_k_reference_keeps_large_logits_finite() {
    let decode = Bf16SingleDecodeSpec::new(3, 1, 1, SINGLE_DECODE_HEAD_DIM).unwrap();
    let split = Bf16SingleDecodeSplitKSpec::new(decode, 3).unwrap();
    let query = vec![bf16::from_f32(64.0); decode.query_numel()];
    let key = bf16_slice(
        &std::iter::repeat_n(-64.0, SINGLE_DECODE_HEAD_DIM)
            .chain(std::iter::repeat_n(0.0, SINGLE_DECODE_HEAD_DIM))
            .chain(std::iter::repeat_n(64.0, SINGLE_DECODE_HEAD_DIM))
            .collect::<Vec<_>>(),
    );
    let value = vec![bf16::ONE; decode.kv_numel()];
    let mut workspace = vec![f32::NAN; split.workspace_numel()];
    let mut output = vec![bf16::NAN; decode.output_numel()];
    let mut lse = vec![f32::NAN; decode.lse_numel()];

    single_decode_bf16_split_k_reference(
        &query,
        &key,
        &value,
        &mut workspace,
        &mut output,
        &mut lse,
        split,
    )
    .unwrap();

    assert!(workspace.iter().all(|value| value.is_finite()));
    assert!(output.iter().all(|value| value.to_f32().is_finite()));
    assert!(lse.iter().all(|value| value.is_finite()));
}

#[test]
fn split_k_references_require_exact_buffer_lengths() {
    let decode = Bf16SingleDecodeSpec::new(2, 2, 1, SINGLE_DECODE_HEAD_DIM).unwrap();
    let split = Bf16SingleDecodeSplitKSpec::new(decode, 2).unwrap();
    let query = vec![bf16::ZERO; decode.query_numel()];
    let key = vec![bf16::ZERO; decode.kv_numel()];
    let value = vec![bf16::ZERO; decode.kv_numel()];
    let workspace = vec![0.0_f32; split.workspace_numel()];
    let output = vec![bf16::ZERO; decode.output_numel()];
    let lse = vec![0.0_f32; decode.lse_numel()];

    assert_eq!(
        single_decode_bf16_split_k_partials_reference(
            &query[..query.len() - 1],
            &key,
            &value,
            &mut workspace.clone(),
            split,
        ),
        Err(ContractError::LengthMismatch {
            buffer: "Q",
            expected: query.len(),
            actual: query.len() - 1,
        })
    );
    assert_eq!(
        single_decode_bf16_split_k_partials_reference(
            &query,
            &key,
            &value,
            &mut workspace[..workspace.len() - 1].to_vec(),
            split,
        ),
        Err(ContractError::LengthMismatch {
            buffer: "workspace",
            expected: workspace.len(),
            actual: workspace.len() - 1,
        })
    );
    assert_eq!(
        single_decode_bf16_split_k_merge_reference(
            &workspace,
            &mut output[..output.len() - 1].to_vec(),
            &mut lse.clone(),
            split,
        ),
        Err(ContractError::LengthMismatch {
            buffer: "O",
            expected: output.len(),
            actual: output.len() - 1,
        })
    );
}
