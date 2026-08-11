use super::*;

fn deterministic_bf16(len: usize, salt: u64) -> Vec<bf16> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bf16::from_f32(((state % 2001) as i32 - 1000) as f32 / 2048.0)
        })
        .collect()
}

#[test]
fn reference_rotates_and_appends_to_arbitrary_final_pages() {
    let spec = Bf16RopePagedKvAppendSpec::new(3, 8, 4, 2, 128, 16).unwrap();
    let page_indptr = [0, 1, 3, 5];
    let page_indices = [7, 2, 6, 5, 1];
    let last_page_len = [3, 1, 16];
    let page_refcounts = [0, 1, 1, 0, 0, 1, 1, 1];
    let query = deterministic_bf16(spec.query_numel(), 1);
    let key = deterministic_bf16(spec.key_numel(), 2);
    let value = deterministic_bf16(spec.value_numel(), 3);
    let mut query_output = vec![bf16::NAN; spec.query_output_numel()];
    let key_sentinel = bf16::from_f32(-7.0);
    let value_sentinel = bf16::from_f32(9.0);
    let mut key_pages = vec![key_sentinel; spec.kv_pages_numel()];
    let mut value_pages = vec![value_sentinel; spec.kv_pages_numel()];

    rope_paged_kv_append_bf16_reference(
        &query,
        &key,
        &value,
        &page_indptr,
        &page_indices,
        &last_page_len,
        &page_refcounts,
        &mut query_output,
        &mut key_pages,
        &mut value_pages,
        spec,
    )
    .unwrap();

    let table = spec
        .validate_metadata(&page_indptr, &page_indices, &last_page_len, &page_refcounts)
        .unwrap()
        .page_table();
    let expected_slots = [(7, 2), (6, 0), (1, 15)];
    for (request, &expected_slot) in expected_slots.iter().enumerate() {
        let position = table.request_kv_len(request).unwrap() - 1;
        assert_eq!(
            table.physical_page_for_token(request, position),
            Some(expected_slot)
        );
        for kv_head in 0..spec.num_kv_heads() {
            let destination =
                ((expected_slot.0 * 16 + expected_slot.1) * spec.num_kv_heads() + kv_head) * 128;
            let source = (request * spec.num_kv_heads() + kv_head) * 128;
            assert_eq!(
                &value_pages[destination..destination + 128],
                &value[source..source + 128]
            );
            assert!(
                key_pages[destination..destination + 128]
                    .iter()
                    .all(|value| *value != key_sentinel)
            );
        }
    }
    assert!(query_output.iter().all(|value| value.to_f32().is_finite()));
}

#[test]
fn metadata_rejects_duplicate_final_slots() {
    let spec = Bf16RopePagedKvAppendSpec::new(2, 2, 2, 1, 128, 16).unwrap();
    assert_eq!(
        spec.validate_metadata(&[0, 1, 2], &[1, 1], &[4, 4], &[0, 1]),
        Err(ContractError::DuplicatePageAppendSlot {
            first_request: 0,
            second_request: 1,
            physical_page: 1,
            offset: 3,
        })
    );
}

#[test]
fn metadata_requires_one_refcount_per_physical_page() {
    let spec = Bf16RopePagedKvAppendSpec::new(1, 2, 2, 1, 128, 16).unwrap();
    assert_eq!(
        spec.validate_metadata(&[0, 1], &[1], &[4], &[0]),
        Err(ContractError::LengthMismatch {
            buffer: "page_refcounts",
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        spec.validate_metadata(&[0, 1], &[1], &[4], &[-1, 1]),
        Err(ContractError::PageReferenceCountTooSmall {
            physical_page: 0,
            minimum: 0,
            actual: -1,
        })
    );
}

#[test]
fn reference_requires_exact_input_and_cache_lengths() {
    let spec = Bf16RopePagedKvAppendSpec::new(1, 1, 1, 1, 128, 16).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key = vec![bf16::ZERO; spec.key_numel()];
    let value = vec![bf16::ZERO; spec.value_numel()];
    let mut query_output = vec![bf16::ZERO; spec.query_output_numel()];
    let mut key_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    let mut value_pages = vec![bf16::ZERO; spec.kv_pages_numel()];
    assert_eq!(
        rope_paged_kv_append_bf16_reference(
            &query[..query.len() - 1],
            &key,
            &value,
            &[0, 1],
            &[0],
            &[1],
            &[1],
            &mut query_output,
            &mut key_pages,
            &mut value_pages,
            spec,
        ),
        Err(ContractError::LengthMismatch {
            buffer: "query",
            expected: query.len(),
            actual: query.len() - 1,
        })
    );
}

#[test]
fn explicit_reference_supports_shuffled_multi_token_requests() {
    let spec = Bf16RopePagedKvAppendTokensSpec::new(6, 3, 8, 4, 2, 128, 16).unwrap();
    let page_indptr = [0, 2, 4, 6];
    let page_indices = [7, 3, 2, 6, 5, 1];
    let last_page_len = [5, 8, 4];
    let page_refcounts = [0, 1, 1, 1, 0, 1, 1, 1];
    let batch_indices = [2, 0, 1, 0, 2, 1];
    let positions = [17, 3, 20, 16, 16, 19];
    let expected_slots = [(1, 1), (7, 3), (6, 4), (3, 0), (1, 0), (6, 3)];
    let query = deterministic_bf16(spec.query_numel(), 11);
    let key = deterministic_bf16(spec.key_numel(), 12);
    let value = deterministic_bf16(spec.value_numel(), 13);
    let mut query_output = vec![bf16::NAN; spec.query_output_numel()];
    let key_sentinel = bf16::from_f32(-7.0);
    let value_sentinel = bf16::from_f32(9.0);
    let mut key_pages = vec![key_sentinel; spec.kv_pages_numel()];
    let mut value_pages = vec![value_sentinel; spec.kv_pages_numel()];

    let metadata = spec
        .validate_metadata(
            &batch_indices,
            &positions,
            &page_indptr,
            &page_indices,
            &last_page_len,
            &page_refcounts,
        )
        .unwrap();
    for (token, &slot) in expected_slots.iter().enumerate() {
        assert_eq!(
            metadata.request_for_token(token),
            Some(batch_indices[token] as usize)
        );
        assert_eq!(metadata.physical_slot_for_token(token), Some(slot));
    }
    rope_paged_kv_append_tokens_bf16_reference(
        &query,
        &key,
        &value,
        &batch_indices,
        &positions,
        &page_indptr,
        &page_indices,
        &last_page_len,
        &page_refcounts,
        &mut query_output,
        &mut key_pages,
        &mut value_pages,
        spec,
    )
    .unwrap();

    for (token, &(physical_page, page_offset)) in expected_slots.iter().enumerate() {
        for kv_head in 0..spec.num_kv_heads() {
            let source = (token * spec.num_kv_heads() + kv_head) * 128;
            let destination =
                ((physical_page * 16 + page_offset) * spec.num_kv_heads() + kv_head) * 128;
            assert_eq!(
                &value_pages[destination..destination + 128],
                &value[source..source + 128]
            );
            assert!(
                key_pages[destination..destination + 128]
                    .iter()
                    .all(|value| *value != key_sentinel)
            );
        }
    }
    assert!(query_output.iter().all(|value| value.to_f32().is_finite()));
}

#[test]
fn explicit_metadata_rejects_invalid_mappings_and_duplicate_physical_slots() {
    let spec = Bf16RopePagedKvAppendTokensSpec::new(2, 2, 4, 2, 1, 128, 16).unwrap();
    let page_indptr = [0, 2, 4];
    let page_indices = [3, 1, 3, 2];
    let last_page_len = [4, 4];
    let page_refcounts = [0, 1, 1, 2];

    assert_eq!(
        spec.validate_metadata(
            &[0, 2],
            &[1, 17],
            &page_indptr,
            &page_indices,
            &last_page_len,
            &page_refcounts,
        ),
        Err(ContractError::AppendBatchIndexOutOfRange {
            token: 1,
            index: 2,
            batch_size: 2,
        })
    );
    assert_eq!(
        spec.validate_metadata(
            &[0, 1],
            &[1, 20],
            &page_indptr,
            &page_indices,
            &last_page_len,
            &page_refcounts,
        ),
        Err(ContractError::AppendPositionOutOfRange {
            token: 1,
            request: 1,
            position: 20,
            kv_len: 20,
        })
    );
    assert_eq!(
        spec.validate_metadata(
            &[0, 1],
            &[2, 2],
            &page_indptr,
            &page_indices,
            &last_page_len,
            &page_refcounts,
        ),
        Err(ContractError::DuplicatePageAppendTokenSlot {
            first_token: 0,
            second_token: 1,
            physical_page: 3,
            offset: 2,
        })
    );
}

#[test]
fn one_token_reference_rejects_shared_target_without_writes() {
    let spec = Bf16RopePagedKvAppendSpec::new(2, 2, 2, 1, 128, 16).unwrap();
    let query = deterministic_bf16(spec.query_numel(), 21);
    let key = deterministic_bf16(spec.key_numel(), 22);
    let value = deterministic_bf16(spec.value_numel(), 23);
    let mut query_output = vec![bf16::NAN; spec.query_output_numel()];
    let mut key_pages = deterministic_bf16(spec.kv_pages_numel(), 24);
    let mut value_pages = deterministic_bf16(spec.kv_pages_numel(), 25);
    let key_before = key_pages.clone();
    let value_before = value_pages.clone();

    assert_eq!(
        rope_paged_kv_append_bf16_reference(
            &query,
            &key,
            &value,
            &[0, 1, 2],
            &[1, 1],
            &[4, 5],
            &[0, 2],
            &mut query_output,
            &mut key_pages,
            &mut value_pages,
            spec,
        ),
        Err(ContractError::NonExclusivePageAppendTarget {
            physical_page: 1,
            reference_count: 2,
        })
    );
    assert_eq!(key_pages, key_before);
    assert_eq!(value_pages, value_before);
    assert!(query_output.iter().all(|value| value.to_f32().is_nan()));
}

#[test]
fn one_token_reference_rejects_underreported_shared_target_without_writes() {
    let spec = Bf16RopePagedKvAppendSpec::new(2, 2, 2, 1, 128, 16).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key = vec![bf16::ZERO; spec.key_numel()];
    let value = vec![bf16::ZERO; spec.value_numel()];
    let mut query_output = vec![bf16::NAN; spec.query_output_numel()];
    let mut key_pages = deterministic_bf16(spec.kv_pages_numel(), 26);
    let mut value_pages = deterministic_bf16(spec.kv_pages_numel(), 27);
    let key_before = key_pages.clone();
    let value_before = value_pages.clone();

    assert_eq!(
        rope_paged_kv_append_bf16_reference(
            &query,
            &key,
            &value,
            &[0, 1, 2],
            &[1, 1],
            &[4, 5],
            &[0, 1],
            &mut query_output,
            &mut key_pages,
            &mut value_pages,
            spec,
        ),
        Err(ContractError::PageReferenceCountTooSmall {
            physical_page: 1,
            minimum: 2,
            actual: 1,
        })
    );
    assert_eq!(key_pages, key_before);
    assert_eq!(value_pages, value_before);
    assert!(query_output.iter().all(|value| value.to_f32().is_nan()));
}

#[test]
fn explicit_reference_rejects_shared_target_without_writes() {
    let spec = Bf16RopePagedKvAppendTokensSpec::new(2, 2, 3, 2, 1, 128, 16).unwrap();
    let query = deterministic_bf16(spec.query_numel(), 31);
    let key = deterministic_bf16(spec.key_numel(), 32);
    let value = deterministic_bf16(spec.value_numel(), 33);
    let mut query_output = vec![bf16::NAN; spec.query_output_numel()];
    let mut key_pages = deterministic_bf16(spec.kv_pages_numel(), 34);
    let mut value_pages = deterministic_bf16(spec.kv_pages_numel(), 35);
    let key_before = key_pages.clone();
    let value_before = value_pages.clone();

    assert_eq!(
        rope_paged_kv_append_tokens_bf16_reference(
            &query,
            &key,
            &value,
            &[0, 1],
            &[2, 3],
            &[0, 1, 2],
            &[2, 2],
            &[4, 5],
            &[0, 0, 2],
            &mut query_output,
            &mut key_pages,
            &mut value_pages,
            spec,
        ),
        Err(ContractError::NonExclusivePageAppendTarget {
            physical_page: 2,
            reference_count: 2,
        })
    );
    assert_eq!(key_pages, key_before);
    assert_eq!(value_pages, value_before);
    assert!(query_output.iter().all(|value| value.to_f32().is_nan()));
}

#[test]
fn explicit_reference_rejects_underreported_shared_target_without_writes() {
    let spec = Bf16RopePagedKvAppendTokensSpec::new(2, 2, 3, 2, 1, 128, 16).unwrap();
    let query = vec![bf16::ZERO; spec.query_numel()];
    let key = vec![bf16::ZERO; spec.key_numel()];
    let value = vec![bf16::ZERO; spec.value_numel()];
    let mut query_output = vec![bf16::NAN; spec.query_output_numel()];
    let mut key_pages = deterministic_bf16(spec.kv_pages_numel(), 36);
    let mut value_pages = deterministic_bf16(spec.kv_pages_numel(), 37);
    let key_before = key_pages.clone();
    let value_before = value_pages.clone();

    assert_eq!(
        rope_paged_kv_append_tokens_bf16_reference(
            &query,
            &key,
            &value,
            &[0, 1],
            &[2, 3],
            &[0, 1, 2],
            &[2, 2],
            &[4, 5],
            &[0, 0, 1],
            &mut query_output,
            &mut key_pages,
            &mut value_pages,
            spec,
        ),
        Err(ContractError::PageReferenceCountTooSmall {
            physical_page: 2,
            minimum: 2,
            actual: 1,
        })
    );
    assert_eq!(key_pages, key_before);
    assert_eq!(value_pages, value_before);
    assert!(query_output.iter().all(|value| value.to_f32().is_nan()));
}

#[test]
fn append_preserves_each_requests_logical_kv_except_its_private_target() {
    let spec = Bf16RopePagedKvAppendSpec::new(2, 4, 2, 1, 128, 16).unwrap();
    let page_indptr = [0, 2, 4];
    let page_indices = [0, 1, 0, 2];
    let last_page_len = [3, 4];
    let page_refcounts = [2, 1, 1, 0];
    let query = deterministic_bf16(spec.query_numel(), 41);
    let key = deterministic_bf16(spec.key_numel(), 42);
    let value = deterministic_bf16(spec.value_numel(), 43);
    let mut query_output = vec![bf16::NAN; spec.query_output_numel()];
    let mut key_pages = deterministic_bf16(spec.kv_pages_numel(), 44);
    let mut value_pages = deterministic_bf16(spec.kv_pages_numel(), 45);
    let key_before = key_pages.clone();
    let value_before = value_pages.clone();
    let metadata = spec
        .validate_metadata(&page_indptr, &page_indices, &last_page_len, &page_refcounts)
        .unwrap();

    rope_paged_kv_append_bf16_reference(
        &query,
        &key,
        &value,
        &page_indptr,
        &page_indices,
        &last_page_len,
        &page_refcounts,
        &mut query_output,
        &mut key_pages,
        &mut value_pages,
        spec,
    )
    .unwrap();

    let page_stride = spec.page_size() * spec.num_kv_heads() * spec.head_dim();
    assert_eq!(
        &key_pages[..page_stride],
        &key_before[..page_stride],
        "shared prefix key page changed"
    );
    assert_eq!(
        &value_pages[..page_stride],
        &value_before[..page_stride],
        "shared prefix value page changed"
    );
    for request in 0..spec.batch_size() {
        let table = metadata.page_table();
        let kv_len = table.request_kv_len(request).unwrap();
        for position in 0..kv_len {
            let (page, offset) = table.physical_page_for_token(request, position).unwrap();
            let destination = (page * spec.page_size() + offset) * spec.head_dim();
            if position + 1 == kv_len {
                let source = request * spec.head_dim();
                assert_eq!(
                    &value_pages[destination..destination + spec.head_dim()],
                    &value[source..source + spec.head_dim()]
                );
            } else {
                assert_eq!(
                    &key_pages[destination..destination + spec.head_dim()],
                    &key_before[destination..destination + spec.head_dim()]
                );
                assert_eq!(
                    &value_pages[destination..destination + spec.head_dim()],
                    &value_before[destination..destination + spec.head_dim()]
                );
            }
        }
    }
}

#[test]
fn explicit_spec_enforces_token_limit() {
    assert_eq!(
        Bf16RopePagedKvAppendTokensSpec::new(65, 1, 1, 1, 1, 128, 16),
        Err(ContractError::UnsupportedAppendTokenCount {
            maximum: ROPE_PAGED_KV_APPEND_MAX_TOKENS,
            actual: 65,
        })
    );
}
