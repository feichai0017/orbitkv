use super::*;
use crate::ContractError;
use half::bf16;

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

#[test]
fn paged_spec_reports_fixed_shapes_and_mapping() {
    let spec = Bf16PagedBatchDecodeSpec::new(
        3,
        7,
        8,
        2,
        SINGLE_DECODE_HEAD_DIM,
        PAGED_BATCH_DECODE_PAGE_SIZE,
    )
    .unwrap();

    assert_eq!(spec.batch_size(), 3);
    assert_eq!(spec.max_num_pages(), 7);
    assert_eq!(spec.num_query_heads(), 8);
    assert_eq!(spec.num_kv_heads(), 2);
    assert_eq!(spec.head_dim(), 128);
    assert_eq!(spec.page_size(), 16);
    assert_eq!(spec.gqa_group_size(), 4);
    assert_eq!(spec.kv_head_for_query_head(0), Some(0));
    assert_eq!(spec.kv_head_for_query_head(3), Some(0));
    assert_eq!(spec.kv_head_for_query_head(4), Some(1));
    assert_eq!(spec.kv_head_for_query_head(7), Some(1));
    assert_eq!(spec.kv_head_for_query_head(8), None);
    assert_eq!(spec.query_numel(), 3 * 8 * 128);
    assert_eq!(spec.kv_pages_numel(), 7 * 16 * 2 * 128);
    assert_eq!(spec.output_numel(), 3 * 8 * 128);
    assert_eq!(spec.lse_numel(), 3 * 8);
    assert_eq!(spec.page_indptr_numel(), 4);
    assert_eq!(spec.last_page_len_numel(), 3);
}

#[test]
fn paged_spec_rejects_unsupported_shapes_and_overflow() {
    for dimensions in [
        (0, 1, 1, 1, 128, 16),
        (1, 0, 1, 1, 128, 16),
        (1, 1, 0, 1, 128, 16),
        (1, 1, 1, 0, 128, 16),
        (1, 1, 1, 1, 0, 16),
        (1, 1, 1, 1, 128, 0),
    ] {
        assert_eq!(
            Bf16PagedBatchDecodeSpec::new(
                dimensions.0,
                dimensions.1,
                dimensions.2,
                dimensions.3,
                dimensions.4,
                dimensions.5,
            ),
            Err(ContractError::ZeroDimension)
        );
    }
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, 1, 1, 1, 64, 16),
        Err(ContractError::UnsupportedHeadDimension {
            expected: SINGLE_DECODE_HEAD_DIM,
            actual: 64,
        })
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, 1, 1, 1, 128, 8),
        Err(ContractError::UnsupportedPageSize {
            expected: PAGED_BATCH_DECODE_PAGE_SIZE,
            actual: 8,
        })
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, 1, 6, 4, 128, 16),
        Err(ContractError::InvalidHeadMapping {
            query_heads: 6,
            kv_heads: 4,
        })
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(usize::MAX, 1, 1, 1, 128, 16),
        Err(ContractError::ElementCountOverflow)
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, usize::MAX, 1, 1, 128, 16),
        Err(ContractError::ElementCountOverflow)
    );
}

#[test]
fn paged_table_reports_lengths_and_maps_logical_tokens() {
    let spec = Bf16PagedBatchDecodeSpec::new(3, 8, 4, 2, 128, 16).unwrap();
    let table = spec
        .validate_page_table(&[0, 2, 3, 6], &[5, 1, 7, 2, 0, 6], &[3, 16, 1])
        .unwrap();

    assert_eq!(table.spec(), spec);
    assert_eq!(table.page_indptr(), [0, 2, 3, 6]);
    assert_eq!(table.page_indices(), [5, 1, 7, 2, 0, 6]);
    assert_eq!(table.last_page_len(), [3, 16, 1]);
    assert_eq!(table.request_page_range(0), Some((0, 2)));
    assert_eq!(table.request_page_range(1), Some((2, 3)));
    assert_eq!(table.request_page_range(2), Some((3, 6)));
    assert_eq!(table.request_page_range(3), None);
    assert_eq!(table.request_kv_len(0), Some(19));
    assert_eq!(table.request_kv_len(1), Some(16));
    assert_eq!(table.request_kv_len(2), Some(33));
    assert_eq!(table.request_kv_len(3), None);
    assert_eq!(table.physical_page_for_token(0, 0), Some((5, 0)));
    assert_eq!(table.physical_page_for_token(0, 15), Some((5, 15)));
    assert_eq!(table.physical_page_for_token(0, 16), Some((1, 0)));
    assert_eq!(table.physical_page_for_token(0, 18), Some((1, 2)));
    assert_eq!(table.physical_page_for_token(0, 19), None);
    assert_eq!(table.physical_page_for_token(1, 15), Some((7, 15)));
    assert_eq!(table.physical_page_for_token(2, 32), Some((6, 0)));
    assert_eq!(table.physical_page_for_token(3, 0), None);
}

#[test]
fn paged_table_rejects_malformed_metadata() {
    let spec = Bf16PagedBatchDecodeSpec::new(2, 4, 4, 2, 128, 16).unwrap();

    let cases = [
        (
            spec.validate_page_table(&[0, 1], &[0, 1], &[16, 16]),
            ContractError::LengthMismatch {
                buffer: "page_indptr",
                expected: 3,
                actual: 2,
            },
        ),
        (
            spec.validate_page_table(&[0, 1, 2], &[0, 1], &[16]),
            ContractError::LengthMismatch {
                buffer: "last_page_len",
                expected: 2,
                actual: 1,
            },
        ),
        (
            spec.validate_page_table(&[1, 2, 3], &[0, 1, 2], &[16, 16]),
            ContractError::InvalidPageIndptrStart { actual: 1 },
        ),
        (
            spec.validate_page_table(&[0, 2, 1], &[0], &[16, 16]),
            ContractError::NonMonotonicPageIndptr {
                request: 1,
                start: 2,
                end: 1,
            },
        ),
        (
            spec.validate_page_table(&[0, 0, 1], &[0], &[16, 16]),
            ContractError::EmptyPagedRequest { request: 0 },
        ),
        (
            spec.validate_page_table(&[0, 1, 2], &[0, 1], &[0, 16]),
            ContractError::InvalidLastPageLength {
                request: 0,
                length: 0,
                page_size: 16,
            },
        ),
        (
            spec.validate_page_table(&[0, 1, 2], &[0, 1], &[16, 17]),
            ContractError::InvalidLastPageLength {
                request: 1,
                length: 17,
                page_size: 16,
            },
        ),
        (
            spec.validate_page_table(&[0, 1, 2], &[0], &[16, 16]),
            ContractError::LengthMismatch {
                buffer: "page_indices",
                expected: 2,
                actual: 1,
            },
        ),
        (
            spec.validate_page_table(&[0, 1, 2], &[-1, 1], &[16, 16]),
            ContractError::PageIndexOutOfRange {
                position: 0,
                index: -1,
                max_num_pages: 4,
            },
        ),
        (
            spec.validate_page_table(&[0, 1, 2], &[0, 4], &[16, 16]),
            ContractError::PageIndexOutOfRange {
                position: 1,
                index: 4,
                max_num_pages: 4,
            },
        ),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, Err(expected));
    }
}

#[test]
fn paged_reference_matches_contiguous_decode_per_request() {
    let spec = Bf16PagedBatchDecodeSpec::new(3, 7, 8, 2, 128, 16).unwrap();
    let page_indptr = [0, 1, 3, 6];
    let page_indices = [4, 6, 1, 5, 0, 3];
    let last_page_len = [1, 7, 16];
    let table = spec
        .validate_page_table(&page_indptr, &page_indices, &last_page_len)
        .unwrap();
    let query = deterministic_bf16(spec.query_numel(), 0x5041_4745_4451);
    let key_pages = deterministic_bf16(spec.kv_pages_numel(), 0x5041_4745_444b);
    let value_pages = deterministic_bf16(spec.kv_pages_numel(), 0x5041_4745_4456);
    let mut actual_output = vec![bf16::NAN; spec.output_numel()];
    let mut actual_lse = vec![f32::NAN; spec.lse_numel()];

    paged_batch_decode_bf16_reference(
        &query,
        &key_pages,
        &value_pages,
        &page_indptr,
        &page_indices,
        &last_page_len,
        &mut actual_output,
        &mut actual_lse,
        spec,
    )
    .unwrap();

    for request in 0..spec.batch_size() {
        let kv_len = table.request_kv_len(request).unwrap();
        let direct =
            Bf16SingleDecodeSpec::new(kv_len, spec.num_query_heads(), spec.num_kv_heads(), 128)
                .unwrap();
        let mut contiguous_key = Vec::with_capacity(direct.kv_numel());
        let mut contiguous_value = Vec::with_capacity(direct.kv_numel());
        for token in 0..kv_len {
            let (physical_page, page_offset) =
                table.physical_page_for_token(request, token).unwrap();
            let start = (physical_page * spec.page_size() + page_offset)
                * spec.num_kv_heads()
                * spec.head_dim();
            let end = start + spec.num_kv_heads() * spec.head_dim();
            contiguous_key.extend_from_slice(&key_pages[start..end]);
            contiguous_value.extend_from_slice(&value_pages[start..end]);
        }
        let query_start = request * direct.query_numel();
        let output_start = request * direct.output_numel();
        let lse_start = request * direct.lse_numel();
        let mut expected_output = vec![bf16::NAN; direct.output_numel()];
        let mut expected_lse = vec![f32::NAN; direct.lse_numel()];

        single_decode_bf16_reference(
            &query[query_start..query_start + direct.query_numel()],
            &contiguous_key,
            &contiguous_value,
            &mut expected_output,
            &mut expected_lse,
            direct,
        )
        .unwrap();

        assert_eq!(
            &actual_output[output_start..output_start + direct.output_numel()],
            expected_output,
            "paged output mismatch for request {request}"
        );
        assert_eq!(
            &actual_lse[lse_start..lse_start + direct.lse_numel()],
            expected_lse,
            "paged LSE mismatch for request {request}"
        );
    }
}

#[test]
fn paged_reference_requires_exact_tensor_lengths() {
    let spec = Bf16PagedBatchDecodeSpec::new(2, 2, 2, 1, 128, 16).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    let value_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    let output = vec![bf16::ZERO; spec.output_numel()];
    let lse = vec![0.0_f32; spec.lse_numel()];
    let page_indptr = [0, 1, 2];
    let page_indices = [0, 1];
    let last_page_len = [16, 16];

    let cases = [
        (
            paged_batch_decode_bf16_reference(
                &query[..query.len() - 1],
                &key_pages,
                &value_pages,
                &page_indptr,
                &page_indices,
                &last_page_len,
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
            paged_batch_decode_bf16_reference(
                &query,
                &key_pages[..key_pages.len() - 1],
                &value_pages,
                &page_indptr,
                &page_indices,
                &last_page_len,
                &mut output.clone(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "K_pages",
                expected: key_pages.len(),
                actual: key_pages.len() - 1,
            },
        ),
        (
            paged_batch_decode_bf16_reference(
                &query,
                &key_pages,
                &value_pages[..value_pages.len() - 1],
                &page_indptr,
                &page_indices,
                &last_page_len,
                &mut output.clone(),
                &mut lse.clone(),
                spec,
            ),
            ContractError::LengthMismatch {
                buffer: "V_pages",
                expected: value_pages.len(),
                actual: value_pages.len() - 1,
            },
        ),
        (
            paged_batch_decode_bf16_reference(
                &query,
                &key_pages,
                &value_pages,
                &page_indptr,
                &page_indices,
                &last_page_len,
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
            paged_batch_decode_bf16_reference(
                &query,
                &key_pages,
                &value_pages,
                &page_indptr,
                &page_indices,
                &last_page_len,
                &mut output.clone(),
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
