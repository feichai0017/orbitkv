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
        &mut query_output,
        &mut key_pages,
        &mut value_pages,
        spec,
    )
    .unwrap();

    let table = spec
        .validate_page_table(&page_indptr, &page_indices, &last_page_len)
        .unwrap();
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
fn page_table_rejects_duplicate_final_slots() {
    let spec = Bf16RopePagedKvAppendSpec::new(2, 2, 2, 1, 128, 16).unwrap();
    assert_eq!(
        spec.validate_page_table(&[0, 1, 2], &[1, 1], &[4, 4]),
        Err(ContractError::DuplicatePageAppendSlot {
            first_request: 0,
            second_request: 1,
            physical_page: 1,
            offset: 3,
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
    let page_indices = [7, 3, 2, 6, 3, 1];
    let last_page_len = [5, 8, 4];
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

    assert_eq!(
        spec.validate_metadata(
            &[0, 2],
            &[1, 17],
            &page_indptr,
            &page_indices,
            &last_page_len,
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
fn explicit_spec_enforces_token_limit() {
    assert_eq!(
        Bf16RopePagedKvAppendTokensSpec::new(65, 1, 1, 1, 1, 128, 16),
        Err(ContractError::UnsupportedAppendTokenCount {
            maximum: ROPE_PAGED_KV_APPEND_MAX_TOKENS,
            actual: 65,
        })
    );
}
