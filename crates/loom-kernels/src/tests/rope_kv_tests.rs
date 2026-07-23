use crate::*;

#[test]
fn rotary_contract_rejects_invalid_partial_dimensions() {
    assert_eq!(
        RotaryEmbeddingSpec::new(1, 2, 1, 8, 3, 16, DType::F16, RotaryStyle::NeoX,),
        Err(ContractError::InvalidRotaryDimension {
            rotary_dim: 3,
            head_size: 8,
        })
    );
    assert_eq!(
        RotaryEmbeddingSpec::new(1, 2, 1, 8, 10, 16, DType::F16, RotaryStyle::NeoX,),
        Err(ContractError::InvalidRotaryDimension {
            rotary_dim: 10,
            head_size: 8,
        })
    );
}

#[test]
fn rotary_reference_supports_both_pairing_styles_and_partial_rope() {
    let cos_sin_cache = [1.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let positions = [1_i64];

    let neox = RotaryEmbeddingSpec::new(1, 1, 1, 6, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
    let mut neox_query = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut neox_key = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    rotary_embedding_f32_reference(
        &mut neox_query,
        &mut neox_key,
        &positions,
        &cos_sin_cache,
        neox,
    )
    .unwrap();
    assert_eq!(neox_query, [-3.0, -4.0, 1.0, 2.0, 5.0, 6.0]);
    assert_eq!(neox_key, [-9.0, -10.0, 7.0, 8.0, 11.0, 12.0]);

    let interleaved =
        RotaryEmbeddingSpec::new(1, 1, 1, 6, 4, 2, DType::F32, RotaryStyle::Interleaved).unwrap();
    let mut interleaved_query = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut interleaved_key = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    rotary_embedding_f32_reference(
        &mut interleaved_query,
        &mut interleaved_key,
        &positions,
        &cos_sin_cache,
        interleaved,
    )
    .unwrap();
    assert_eq!(interleaved_query, [-2.0, 1.0, -4.0, 3.0, 5.0, 6.0]);
    assert_eq!(interleaved_key, [-8.0, 7.0, -10.0, 9.0, 11.0, 12.0]);
}

#[test]
fn fused_rope_paged_write_rotates_padding_but_skips_its_cache_slot() {
    let rotary = RotaryEmbeddingSpec::new(2, 1, 1, 4, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
    let spec = RopePagedKvWriteSpec::new(rotary, 2, 2, 2).unwrap();
    let mut query = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut key = [9.0_f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let value = [17.0_f32, 18.0, 19.0, 20.0];
    let positions = [0_i64, 1];
    let cos_sin_cache = [1.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let slots = [3_i64, -1];
    let mut key_cache = [-99.0_f32; 16];
    let mut value_cache = [-99.0_f32; 8];

    rope_paged_kv_write_f32_reference(
        &mut query,
        &mut key,
        &value,
        &positions,
        &cos_sin_cache,
        &mut key_cache,
        &mut value_cache,
        &slots,
        spec,
    )
    .unwrap();

    assert_eq!(&query[..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&query[4..], &[-7.0, -8.0, 5.0, 6.0]);
    assert_eq!(&key[4..], &[-15.0, -16.0, 13.0, 14.0]);
    assert!(key_cache[..12].iter().all(|&value| value == -99.0));
    assert_eq!(&key_cache[12..], &[9.0, 10.0, 11.0, 12.0]);
    assert!(value_cache[..6].iter().all(|&value| value == -99.0));
    assert_eq!(&value_cache[6..], &[17.0, 18.0]);
}

#[test]
fn fused_rope_paged_write_rejects_bad_metadata_before_mutation() {
    let rotary = RotaryEmbeddingSpec::new(2, 1, 1, 4, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
    let spec = RopePagedKvWriteSpec::new(rotary, 4, 1, 2).unwrap();
    let original = [1.0_f32; 8];
    let mut query = original;
    let mut key = original;
    let value = [2.0_f32; 8];
    let cache = [1.0_f32, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let mut key_cache = [0.0_f32; 8];
    let mut value_cache = [0.0_f32; 8];

    let error = rope_paged_kv_write_f32_reference(
        &mut query,
        &mut key,
        &value,
        &[0, 1],
        &cache,
        &mut key_cache,
        &mut value_cache,
        &[1, 1],
        spec,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ContractError::DuplicateSlot {
            first_token: 0,
            second_token: 1,
            slot: 1,
        }
    );
    assert_eq!(query, original);
    assert_eq!(key, original);

    let error =
        rotary_embedding_f32_reference(&mut query, &mut key, &[0, 2], &cache, rotary).unwrap_err();
    assert_eq!(
        error,
        ContractError::PositionOutOfBounds {
            token: 1,
            position: 2,
            max_position: 2,
        }
    );
    assert_eq!(query, original);
    assert_eq!(key, original);
}
