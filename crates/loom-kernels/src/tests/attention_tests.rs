use crate::*;
use half::bf16;

#[test]
fn paged_decode_attention_follows_block_indirection_and_gqa_mapping() {
    let spec = PagedDecodeAttentionSpec::new(1, 2, 1, 1, 1, 2, 2, 2, 4, 1.0, DType::F32).unwrap();
    let query = [1.0_f32, 1.0];
    // Physical block 0 is the second logical block. Its second slot is
    // outside the active sequence and must not affect the result.
    let key_cache = [2.0_f32, 99.0, 0.0, 1.0];
    let value_cache = [20.0_f32, 99.0, 1.0, 10.0];
    let block_tables = [1_i64, 0];
    let mut output = [-1.0_f32; 2];

    paged_decode_attention_f32_reference(
        &query,
        &key_cache,
        &value_cache,
        &block_tables,
        &[3],
        &mut output,
        spec,
    )
    .unwrap();

    let expected =
        (1.0 + 10.0 * 1.0_f32.exp() + 20.0 * 2.0_f32.exp()) / (1.0 + 1.0_f32.exp() + 2.0_f32.exp());
    assert!((output[0] - expected).abs() < 1.0e-6);
    assert!((output[1] - expected).abs() < 1.0e-6);
}

#[test]
fn paged_decode_attention_supports_low_precision_and_distinct_value_width() {
    let spec = PagedDecodeAttentionSpec::new(1, 1, 1, 2, 3, 1, 1, 1, 1, 0.5, DType::Bf16).unwrap();
    let query = [bf16::from_f32(2.0), bf16::from_f32(-1.0)];
    let key_cache = [bf16::from_f32(4.0), bf16::from_f32(3.0)];
    let value_cache = [
        bf16::from_f32(1.25),
        bf16::from_f32(-2.5),
        bf16::from_f32(7.0),
    ];
    let mut output = [bf16::ZERO; 3];

    paged_decode_attention_bf16_reference(
        &query,
        &key_cache,
        &value_cache,
        &[0],
        &[1],
        &mut output,
        spec,
    )
    .unwrap();

    assert_eq!(output, value_cache);
}

#[test]
fn paged_decode_attention_validates_metadata_before_mutating_output() {
    assert_eq!(
        PagedDecodeAttentionSpec::new(1, 3, 2, 4, 4, 1, 16, 1, 16, 0.5, DType::F16),
        Err(ContractError::HeadCountNotDivisible {
            query_heads: 3,
            kv_heads: 2,
        })
    );
    assert_eq!(
        PagedDecodeAttentionSpec::new(1, 2, 1, 4, 4, 1, 16, 1, 16, 0.0, DType::F16),
        Err(ContractError::InvalidScale(0.0))
    );

    let spec = PagedDecodeAttentionSpec::new(2, 1, 1, 1, 1, 2, 2, 2, 4, 1.0, DType::F32).unwrap();
    let query = [1.0_f32; 2];
    let key_cache = [1.0_f32; 4];
    let value_cache = [2.0_f32; 4];
    let mut output = [-7.0_f32; 2];

    let error = paged_decode_attention_f32_reference(
        &query,
        &key_cache,
        &value_cache,
        &[0, 1, 1, -1],
        &[3, 5],
        &mut output,
        spec,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ContractError::SequenceLengthOutOfBounds {
            sequence: 1,
            length: 5,
            capacity: 4,
        }
    );
    assert_eq!(output, [-7.0; 2]);

    let error = paged_decode_attention_f32_reference(
        &query,
        &key_cache,
        &value_cache,
        &[0, 2, 1, -1],
        &[3, 1],
        &mut output,
        spec,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ContractError::BlockIdOutOfBounds {
            sequence: 0,
            logical_block: 1,
            block_id: 2,
            num_blocks: 2,
        }
    );
    assert_eq!(output, [-7.0; 2]);
}
