use super::*;

#[test]
fn metadata_maps_bottom_right_causal_pages() {
    let spec = Bf16PagedPrefillSpec::new(2, 3, 6, 4, 2, 128, 16).unwrap();
    let metadata = spec
        .validate_metadata(&[0, 2, 3], &[0, 2, 4], &[4, 1, 5, 2], &[3, 16])
        .unwrap();

    assert_eq!(metadata.spec(), spec);
    assert_eq!(metadata.request_query_range(0), Some((0, 2)));
    assert_eq!(metadata.request_query_range(1), Some((2, 3)));
    assert_eq!(metadata.request_query_range(2), None);
    assert_eq!(metadata.request_page_range(0), Some((0, 2)));
    assert_eq!(metadata.request_page_range(1), Some((2, 4)));
    assert_eq!(metadata.request_kv_len(0), Some(19));
    assert_eq!(metadata.request_kv_len(1), Some(32));
    assert_eq!(metadata.causal_kv_end(0, 0), Some(18));
    assert_eq!(metadata.causal_kv_end(0, 1), Some(19));
    assert_eq!(metadata.causal_kv_end(0, 2), None);
    assert_eq!(metadata.physical_page_for_token(0, 0), Some((4, 0)));
    assert_eq!(metadata.physical_page_for_token(0, 16), Some((1, 0)));
    assert_eq!(metadata.physical_page_for_token(0, 18), Some((1, 2)));
    assert_eq!(metadata.physical_page_for_token(0, 19), None);
}

#[test]
fn reference_applies_bottom_right_mask_over_physical_pages() {
    let spec = Bf16PagedPrefillSpec::new(1, 2, 2, 1, 1, 128, 16).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    let mut value_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    for (logical_token, token_value) in [1.0, 2.0, 3.0, 100.0].into_iter().enumerate() {
        let start = (spec.page_size() + logical_token) * spec.head_dim();
        value_pages[start..start + spec.head_dim()].fill(bf16::from_f32(token_value));
    }
    let mut output = vec![bf16::NAN; spec.output_numel()];
    let mut lse = vec![f32::NAN; spec.lse_numel()];

    paged_prefill_bf16_reference(
        &query,
        &key_pages,
        &value_pages,
        &[0, 2],
        &[0, 1],
        &[1],
        &[4],
        &mut output,
        &mut lse,
        spec,
    )
    .unwrap();

    assert!(
        output[..spec.head_dim()]
            .iter()
            .all(|&value| value == bf16::from_f32(2.0))
    );
    assert!(
        output[spec.head_dim()..]
            .iter()
            .all(|&value| value == bf16::from_f32(26.5))
    );
    assert_eq!(lse, [3.0_f32.log2(), 4.0_f32.log2()]);
}

#[test]
fn reference_maps_gqa_heads_and_reused_pages() {
    let spec = Bf16PagedPrefillSpec::new(2, 2, 2, 4, 2, 128, 16).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    let mut value_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    for physical_page in 0..2 {
        for kv_head in 0..2 {
            let value = if kv_head == 0 {
                physical_page as f32 + 1.0
            } else {
                -(physical_page as f32 + 1.0)
            };
            let start = (physical_page * spec.page_size() * spec.num_kv_heads() + kv_head)
                * spec.head_dim();
            value_pages[start..start + spec.head_dim()].fill(bf16::from_f32(value));
        }
    }
    let mut output = vec![bf16::NAN; spec.output_numel()];
    let mut lse = vec![f32::NAN; spec.lse_numel()];

    paged_prefill_bf16_reference(
        &query,
        &key_pages,
        &value_pages,
        &[0, 1, 2],
        &[0, 1, 2],
        &[1, 0],
        &[1, 1],
        &mut output,
        &mut lse,
        spec,
    )
    .unwrap();

    for request in 0..2 {
        let positive = bf16::from_f32(if request == 0 { 2.0 } else { 1.0 });
        let negative = bf16::from_f32(-positive.to_f32());
        let request_offset = request * spec.num_query_heads() * spec.head_dim();
        assert!(
            output[request_offset..request_offset + 2 * spec.head_dim()]
                .iter()
                .all(|&value| value == positive)
        );
        assert!(
            output[request_offset + 2 * spec.head_dim()..request_offset + 4 * spec.head_dim()]
                .iter()
                .all(|&value| value == negative)
        );
    }
    assert_eq!(lse, [0.0; 8]);
}

#[test]
fn metadata_rejects_malformed_query_and_page_tables() {
    let spec = Bf16PagedPrefillSpec::new(2, 3, 4, 4, 2, 128, 16).unwrap();
    let cases = [
        (
            spec.validate_metadata(&[0, 3], &[0, 1, 2], &[0, 1], &[16, 16]),
            ContractError::LengthMismatch {
                buffer: "qo_indptr",
                expected: 3,
                actual: 2,
            },
        ),
        (
            spec.validate_metadata(&[1, 2, 4], &[0, 1, 2], &[0, 1], &[16, 16]),
            ContractError::InvalidIndptrStart {
                buffer: "qo_indptr",
                actual: 1,
            },
        ),
        (
            spec.validate_metadata(&[0, 2, 1], &[0, 1, 2], &[0, 1], &[16, 16]),
            ContractError::NonMonotonicIndptr {
                buffer: "qo_indptr",
                request: 1,
                start: 2,
                end: 1,
            },
        ),
        (
            spec.validate_metadata(&[0, 2, 3], &[1, 2, 3], &[0, 1, 2], &[16, 16]),
            ContractError::InvalidPageIndptrStart { actual: 1 },
        ),
        (
            spec.validate_metadata(&[0, 2, 3], &[0, 2, 1], &[0], &[16, 16]),
            ContractError::NonMonotonicPageIndptr {
                request: 1,
                start: 2,
                end: 1,
            },
        ),
        (
            spec.validate_metadata(&[0, 2, 3], &[0, 0, 1], &[0], &[16, 16]),
            ContractError::EmptyPagedRequest { request: 0 },
        ),
        (
            spec.validate_metadata(&[0, 2, 3], &[0, 1, 2], &[0, 1], &[16, 0]),
            ContractError::InvalidLastPageLength {
                request: 1,
                length: 0,
                page_size: 16,
            },
        ),
        (
            spec.validate_metadata(&[0, 2, 3], &[0, 1, 2], &[0, 4], &[16, 16]),
            ContractError::PageIndexOutOfRange {
                position: 1,
                index: 4,
                max_num_pages: 4,
            },
        ),
        (
            spec.validate_metadata(&[0, 2, 3], &[0, 1, 2], &[0, 1], &[1, 16]),
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
fn spec_and_reference_require_exact_contracts() {
    for shape in [
        (0, 1, 1, 1, 1, 128, 16),
        (1, 0, 1, 1, 1, 128, 16),
        (1, 1, 0, 1, 1, 128, 16),
        (1, 1, 1, 0, 1, 128, 16),
        (1, 1, 1, 1, 0, 128, 16),
        (1, 1, 1, 1, 1, 0, 16),
        (1, 1, 1, 1, 1, 128, 0),
    ] {
        assert_eq!(
            Bf16PagedPrefillSpec::new(
                shape.0, shape.1, shape.2, shape.3, shape.4, shape.5, shape.6,
            ),
            Err(ContractError::ZeroDimension)
        );
    }
    assert_eq!(
        Bf16PagedPrefillSpec::new(1, 1, 1, 1, 1, 64, 16),
        Err(ContractError::UnsupportedHeadDimension {
            expected: 128,
            actual: 64,
        })
    );
    assert_eq!(
        Bf16PagedPrefillSpec::new(1, 1, 1, 1, 1, 128, 8),
        Err(ContractError::UnsupportedPageSize {
            expected: 16,
            actual: 8,
        })
    );
    assert_eq!(
        Bf16PagedPrefillSpec::new(1, 1, 1, 6, 4, 128, 16),
        Err(ContractError::InvalidHeadMapping {
            query_heads: 6,
            kv_heads: 4,
        })
    );
    assert_eq!(
        Bf16PagedPrefillSpec::new(1, i32::MAX as usize + 1, 1, 1, 1, 128, 16),
        Err(ContractError::ElementCountOverflow)
    );
    assert_eq!(
        Bf16PagedPrefillSpec::new(1, 1, usize::MAX, 1, 1, 128, 16),
        Err(ContractError::ElementCountOverflow)
    );

    let spec = Bf16PagedPrefillSpec::new(1, 1, 1, 1, 1, 128, 16).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    let value_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    let mut output = vec![bf16::ZERO; spec.output_numel()];
    let mut lse = vec![0.0_f32; spec.lse_numel()];
    assert_eq!(
        paged_prefill_bf16_reference(
            &query[..query.len() - 1],
            &key_pages,
            &value_pages,
            &[0, 1],
            &[0, 1],
            &[0],
            &[1],
            &mut output,
            &mut lse,
            spec,
        ),
        Err(ContractError::LengthMismatch {
            buffer: "Q",
            expected: query.len(),
            actual: query.len() - 1,
        })
    );
}
