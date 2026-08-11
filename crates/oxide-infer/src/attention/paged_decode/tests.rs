use super::*;
use crate::ContractError;
use crate::attention::{
    Bf16SingleDecodeSpec, SINGLE_DECODE_HEAD_DIM, single_decode_bf16_reference,
};
use half::bf16;

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

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn pack_paged_kv(
    logical_nhd: &[bf16],
    max_num_pages: usize,
    num_kv_heads: usize,
    layout: PagedKvLayout,
) -> Vec<bf16> {
    match layout {
        PagedKvLayout::Nhd => logical_nhd.to_vec(),
        PagedKvLayout::Hnd => {
            let mut packed = vec![bf16::NAN; logical_nhd.len()];
            for page in 0..max_num_pages {
                for token in 0..PAGED_BATCH_DECODE_PAGE_SIZE {
                    for head in 0..num_kv_heads {
                        for component in 0..SINGLE_DECODE_HEAD_DIM {
                            let source = (((page * PAGED_BATCH_DECODE_PAGE_SIZE + token)
                                * num_kv_heads
                                + head)
                                * SINGLE_DECODE_HEAD_DIM)
                                + component;
                            let target = (((page * num_kv_heads + head)
                                * PAGED_BATCH_DECODE_PAGE_SIZE
                                + token)
                                * SINGLE_DECODE_HEAD_DIM)
                                + component;
                            packed[target] = logical_nhd[source];
                        }
                    }
                }
            }
            packed
        }
    }
}

fn assert_paged_matches_contiguous(
    spec: Bf16PagedBatchDecodeSpec,
    page_indptr: &[i32],
    page_indices: &[i32],
    last_page_len: &[i32],
    salt: u64,
) {
    let table = spec
        .validate_page_table(page_indptr, page_indices, last_page_len)
        .unwrap();
    let query = deterministic_bf16(spec.query_numel(), salt ^ 0x51);
    let key_pages = deterministic_bf16(spec.kv_pages_numel(), salt ^ 0x4b);
    let value_pages = deterministic_bf16(spec.kv_pages_numel(), salt ^ 0x56);
    let mut actual_output = vec![bf16::NAN; spec.output_numel()];
    let mut actual_lse = vec![f32::NAN; spec.lse_numel()];

    paged_batch_decode_bf16_reference(
        &query,
        &key_pages,
        &value_pages,
        page_indptr,
        page_indices,
        last_page_len,
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
            for kv_head in 0..spec.num_kv_heads() {
                let start = match spec.kv_layout() {
                    PagedKvLayout::Nhd => {
                        ((physical_page * spec.page_size() + page_offset) * spec.num_kv_heads()
                            + kv_head)
                            * spec.head_dim()
                    }
                    PagedKvLayout::Hnd => {
                        ((physical_page * spec.num_kv_heads() + kv_head) * spec.page_size()
                            + page_offset)
                            * spec.head_dim()
                    }
                };
                let end = start + spec.head_dim();
                contiguous_key.extend_from_slice(&key_pages[start..end]);
                contiguous_value.extend_from_slice(&value_pages[start..end]);
            }
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
fn paged_spec_reports_fixed_shapes_and_mapping() {
    let spec = Bf16PagedBatchDecodeSpec::new(
        3,
        7,
        8,
        2,
        SINGLE_DECODE_HEAD_DIM,
        PAGED_BATCH_DECODE_PAGE_SIZE,
        PagedKvLayout::Hnd,
    )
    .unwrap();

    assert_eq!(spec.batch_size(), 3);
    assert_eq!(spec.max_num_pages(), 7);
    assert_eq!(spec.num_query_heads(), 8);
    assert_eq!(spec.num_kv_heads(), 2);
    assert_eq!(spec.head_dim(), 128);
    assert_eq!(spec.page_size(), 16);
    assert_eq!(spec.kv_layout(), PagedKvLayout::Hnd);
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
                PagedKvLayout::Nhd,
            ),
            Err(ContractError::ZeroDimension)
        );
    }
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, 1, 1, 1, 64, 16, PagedKvLayout::Nhd),
        Err(ContractError::UnsupportedHeadDimension {
            expected: SINGLE_DECODE_HEAD_DIM,
            actual: 64,
        })
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, 1, 1, 1, 128, 8, PagedKvLayout::Nhd),
        Err(ContractError::UnsupportedPageSize {
            expected: PAGED_BATCH_DECODE_PAGE_SIZE,
            actual: 8,
        })
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, 1, 6, 4, 128, 16, PagedKvLayout::Nhd),
        Err(ContractError::InvalidHeadMapping {
            query_heads: 6,
            kv_heads: 4,
        })
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(usize::MAX, 1, 1, 1, 128, 16, PagedKvLayout::Nhd),
        Err(ContractError::ElementCountOverflow)
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, usize::MAX, 1, 1, 128, 16, PagedKvLayout::Nhd),
        Err(ContractError::ElementCountOverflow)
    );
    let beyond_i32 = usize::try_from(i32::MAX).unwrap() + 1;
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(beyond_i32, 1, 1, 1, 128, 16, PagedKvLayout::Nhd),
        Err(ContractError::ElementCountOverflow)
    );
    assert_eq!(
        Bf16PagedBatchDecodeSpec::new(1, beyond_i32, 1, 1, 128, 16, PagedKvLayout::Nhd),
        Err(ContractError::ElementCountOverflow)
    );
}

#[test]
fn paged_table_reports_lengths_and_maps_logical_tokens() {
    let spec = Bf16PagedBatchDecodeSpec::new(3, 8, 4, 2, 128, 16, PagedKvLayout::Nhd).unwrap();
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
    let spec = Bf16PagedBatchDecodeSpec::new(2, 4, 4, 2, 128, 16, PagedKvLayout::Nhd).unwrap();

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
    let page_indptr = [0, 1, 3, 6];
    let page_indices = [4, 6, 1, 5, 0, 3];
    let last_page_len = [1, 7, 16];
    for layout in [PagedKvLayout::Nhd, PagedKvLayout::Hnd] {
        let spec = Bf16PagedBatchDecodeSpec::new(3, 7, 8, 2, 128, 16, layout).unwrap();
        assert_paged_matches_contiguous(
            spec,
            &page_indptr,
            &page_indices,
            &last_page_len,
            0x5041_4745_4400,
        );
    }
}

#[test]
fn generated_paged_cases_match_contiguous_decode() {
    let mut state = 0xd1ff_3a11_5eed_0001_u64;
    for case in 0..64_u64 {
        let batch_size = (next_u64(&mut state) % 4 + 1) as usize;
        let max_num_pages = 8;
        let num_kv_heads = if next_u64(&mut state).is_multiple_of(2) {
            1
        } else {
            2
        };
        let group_size = [1, 2, 4][(next_u64(&mut state) % 3) as usize];
        let num_query_heads = num_kv_heads * group_size;
        let mut page_indptr = vec![0_i32];
        let mut page_indices = Vec::new();
        let mut last_page_len = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let request_pages = (next_u64(&mut state) % 3 + 1) as usize;
            for _ in 0..request_pages {
                page_indices.push((next_u64(&mut state) % max_num_pages as u64) as i32);
            }
            page_indptr.push(i32::try_from(page_indices.len()).unwrap());
            last_page_len.push((next_u64(&mut state) % 16 + 1) as i32);
        }
        for layout in [PagedKvLayout::Nhd, PagedKvLayout::Hnd] {
            let spec = Bf16PagedBatchDecodeSpec::new(
                batch_size,
                max_num_pages,
                num_query_heads,
                num_kv_heads,
                128,
                16,
                layout,
            )
            .unwrap();
            assert_paged_matches_contiguous(
                spec,
                &page_indptr,
                &page_indices,
                &last_page_len,
                case ^ state,
            );
        }
    }
}

#[test]
fn nhd_and_hnd_encode_the_same_logical_cache() {
    let query = deterministic_bf16(2 * 8 * 128, 0x0051_5545_5259);
    let logical_key = deterministic_bf16(5 * 16 * 2 * 128, 0x004b_4559);
    let logical_value = deterministic_bf16(5 * 16 * 2 * 128, 0x0056_414c_5545);
    let page_indptr = [0, 2, 5];
    let page_indices = [4, 1, 3, 0, 2];
    let last_page_len = [7, 13];
    let mut outputs = Vec::new();
    let mut lses = Vec::new();

    for layout in [PagedKvLayout::Nhd, PagedKvLayout::Hnd] {
        let spec = Bf16PagedBatchDecodeSpec::new(2, 5, 8, 2, 128, 16, layout).unwrap();
        let key = pack_paged_kv(&logical_key, 5, 2, layout);
        let value = pack_paged_kv(&logical_value, 5, 2, layout);
        let mut output = vec![bf16::NAN; spec.output_numel()];
        let mut lse = vec![f32::NAN; spec.lse_numel()];
        paged_batch_decode_bf16_reference(
            &query,
            &key,
            &value,
            &page_indptr,
            &page_indices,
            &last_page_len,
            &mut output,
            &mut lse,
            spec,
        )
        .unwrap();
        outputs.push(output);
        lses.push(lse);
    }

    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(lses[0], lses[1]);

    let wrong_spec =
        Bf16PagedBatchDecodeSpec::new(2, 5, 8, 2, 128, 16, PagedKvLayout::Nhd).unwrap();
    let hnd_key = pack_paged_kv(&logical_key, 5, 2, PagedKvLayout::Hnd);
    let hnd_value = pack_paged_kv(&logical_value, 5, 2, PagedKvLayout::Hnd);
    let mut wrong_output = vec![bf16::NAN; wrong_spec.output_numel()];
    let mut wrong_lse = vec![f32::NAN; wrong_spec.lse_numel()];
    paged_batch_decode_bf16_reference(
        &query,
        &hnd_key,
        &hnd_value,
        &page_indptr,
        &page_indices,
        &last_page_len,
        &mut wrong_output,
        &mut wrong_lse,
        wrong_spec,
    )
    .unwrap();
    assert_ne!(wrong_output, outputs[0]);
    assert_ne!(wrong_lse, lses[0]);
}

#[test]
fn paged_reference_requires_exact_tensor_lengths() {
    let spec = Bf16PagedBatchDecodeSpec::new(2, 2, 2, 1, 128, 16, PagedKvLayout::Hnd).unwrap();
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
