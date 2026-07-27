use crate::*;
use half::{bf16, f16};

#[test]
fn greedy_sample_logprobs_selects_first_tie_and_normalizes() {
    let spec = GreedySampleLogprobsSpec::new(2, 4, DType::F32).unwrap();
    let logits = [1.0_f32, 3.0, 3.0, -1.0, -2.0, -1.0, 2.0, 0.0];
    let mut token_ids = [u32::MAX; 2];
    let mut logprobs = [0.0_f32; 2];

    greedy_sample_logprobs_f32_reference(&logits, &mut token_ids, &mut logprobs, spec).unwrap();

    assert_eq!(token_ids, [1, 2]);
    let first_sum = (-2.0_f64).exp() + 1.0 + 1.0 + (-4.0_f64).exp();
    let second_sum = (-4.0_f64).exp() + (-3.0_f64).exp() + 1.0 + (-2.0_f64).exp();
    assert!((logprobs[0] + first_sum.ln() as f32).abs() < 1.0e-6);
    assert!((logprobs[1] + second_sum.ln() as f32).abs() < 1.0e-6);
}

#[test]
fn greedy_sample_logprobs_supports_low_precision_and_validates_buffers() {
    let spec = GreedySampleLogprobsSpec::new(1, 3, DType::Bf16).unwrap();
    let logits = [
        bf16::from_f32(-1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(0.5),
    ];
    let mut token_ids = [u32::MAX];
    let mut logprobs = [0.0_f32];
    greedy_sample_logprobs_bf16_reference(&logits, &mut token_ids, &mut logprobs, spec).unwrap();
    assert_eq!(token_ids, [1]);
    assert!(logprobs[0].is_finite() && logprobs[0] < 0.0);

    assert_eq!(
        greedy_sample_logprobs_bf16_reference(&logits, &mut [u32::MAX; 2], &mut logprobs, spec,),
        Err(ContractError::LengthMismatch {
            buffer: "token_ids",
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn selected_token_logprobs_normalizes_and_counts_tie_aware_ranks() {
    let spec = SelectedTokenLogprobsSpec::new(2, 4, DType::F32).unwrap();
    let logits = [1.0_f32, 3.0, 3.0, -1.0, -2.0, -1.0, 2.0, 0.0];
    let token_ids = [0_i64, 1_i64];
    let mut logprobs = [0.0_f32; 2];
    let mut ranks = [0_i64; 2];

    selected_token_logprobs_f32_reference(&logits, &token_ids, &mut logprobs, &mut ranks, spec)
        .unwrap();

    let first_sum = (-2.0_f64).exp() + 1.0 + 1.0 + (-4.0_f64).exp();
    let second_sum = (-4.0_f64).exp() + (-3.0_f64).exp() + 1.0 + (-2.0_f64).exp();
    assert!((logprobs[0] - (-2.0 - first_sum.ln() as f32)).abs() < 1.0e-6);
    assert!((logprobs[1] - (-3.0 - second_sum.ln() as f32)).abs() < 1.0e-6);
    assert_eq!(ranks, [3, 3]);
}

#[test]
fn selected_token_logprobs_validates_ids_and_low_precision_buffers() {
    let spec = SelectedTokenLogprobsSpec::new(1, 3, DType::Bf16).unwrap();
    let logits = [
        bf16::from_f32(-1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(0.5),
    ];
    let mut logprobs = [0.0_f32];
    let mut ranks = [0_i64];
    selected_token_logprobs_bf16_reference(&logits, &[2_i64], &mut logprobs, &mut ranks, spec)
        .unwrap();
    assert!(logprobs[0].is_finite() && logprobs[0] < 0.0);
    assert_eq!(ranks, [2]);

    assert_eq!(
        selected_token_logprobs_bf16_reference(&logits, &[-1_i64], &mut logprobs, &mut ranks, spec,),
        Err(ContractError::TokenIdOutOfBounds {
            row: 0,
            token_id: -1,
            vocab_size: 3,
        })
    );
    assert_eq!(
        selected_token_logprobs_bf16_reference(&logits, &[3_i64], &mut logprobs, &mut ranks, spec,),
        Err(ContractError::TokenIdOutOfBounds {
            row: 0,
            token_id: 3,
            vocab_size: 3,
        })
    );
}

#[test]
fn top_k_filter_applies_per_row_thresholds_and_preserves_boundary_ties() {
    let spec = TopKFilterSpec::new(3, 5, DType::F32).unwrap();
    assert_eq!(spec.workspace_partitions(), 1);
    assert_eq!(spec.workspace_elements(), 12_291);
    assert_eq!(spec.workspace_bytes(), 49_164);
    let mut logits = [
        5.0_f32, 4.0, 4.0, 1.0, -1.0, //
        -2.0, 3.0, 0.0, 1.0, 2.0, //
        7.0, 7.0, 6.0, 5.0, 4.0,
    ];

    top_k_filter_f32_reference(&mut logits, &[2, 5, 1], spec).unwrap();

    assert_eq!(
        logits,
        [
            5.0_f32,
            4.0,
            4.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            -2.0,
            3.0,
            0.0,
            1.0,
            2.0,
            7.0,
            7.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ]
    );

    let large_spec = TopKFilterSpec::new(7, 151_936, DType::F16).unwrap();
    assert_eq!(large_spec.workspace_partitions(), 38);
    assert_eq!(large_spec.workspace_elements(), 1_089_543);
    assert_eq!(large_spec.workspace_bytes(), 4_358_172);
}

#[test]
fn top_k_filter_supports_low_precision_and_validates_before_mutation() {
    let f16_spec = TopKFilterSpec::new(1, 4, DType::F16).unwrap();
    let mut f16_logits = [
        f16::from_f32(-2.0),
        f16::from_f32(0.0),
        f16::from_f32(3.0),
        f16::from_f32(1.0),
    ];
    top_k_filter_f16_reference(&mut f16_logits, &[2], f16_spec).unwrap();
    assert_eq!(
        f16_logits,
        [
            f16::NEG_INFINITY,
            f16::NEG_INFINITY,
            f16::from_f32(3.0),
            f16::from_f32(1.0),
        ]
    );

    let bf16_spec = TopKFilterSpec::new(2, 3, DType::Bf16).unwrap();
    let original = [
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(0.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
    ];
    let mut bf16_logits = original;
    assert_eq!(
        top_k_filter_bf16_reference(&mut bf16_logits, &[2, 0], bf16_spec),
        Err(ContractError::TopKFilterOutOfRange {
            row: 1,
            top_k: 0,
            vocab_size: 3,
        })
    );
    assert_eq!(bf16_logits, original);
    assert_eq!(
        top_k_filter_bf16_reference(&mut bf16_logits, &[2, 4], bf16_spec),
        Err(ContractError::TopKFilterOutOfRange {
            row: 1,
            top_k: 4,
            vocab_size: 3,
        })
    );
    assert_eq!(
        top_k_filter_bf16_reference(&mut bf16_logits, &[2], bf16_spec),
        Err(ContractError::LengthMismatch {
            buffer: "top_ks",
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        top_k_filter_f32_reference(
            &mut [0.0_f32; 6],
            &[1, 1],
            TopKFilterSpec::new(2, 3, DType::Bf16).unwrap(),
        ),
        Err(ContractError::UnsupportedDType(DType::Bf16))
    );
}

#[test]
fn topk_sampled_logprobs_normalize_rank_and_order_ties_by_token() {
    let spec = TopKSampledLogprobsSpec::new(2, 5, 3, DType::F32).unwrap();
    let logits = [3.0_f32, 1.0, 3.0, 2.0, -1.0, -2.0, 4.0, 0.0, 1.0, 3.0];
    let sampled = [3_i64, 2_i64];
    let mut token_ids = [-1_i32; 8];
    let mut logprobs = [0.0_f32; 8];
    let mut ranks = [0_i64; 2];

    topk_sampled_logprobs_f32_reference(
        &logits,
        &sampled,
        &mut token_ids,
        &mut logprobs,
        &mut ranks,
        spec,
    )
    .unwrap();

    assert_eq!(token_ids, [3, 0, 2, 3, 2, 1, 4, 3]);
    assert_eq!(ranks, [3, 4]);
    let first_normalizer =
        3.0 + (1.0_f64 + (-2.0_f64).exp() + 1.0 + (-1.0_f64).exp() + (-4.0_f64).exp()).ln() as f32;
    assert!((logprobs[0] - (2.0 - first_normalizer)).abs() < 1.0e-6);
    assert!((logprobs[1] - (3.0 - first_normalizer)).abs() < 1.0e-6);
    assert!((logprobs[2] - (3.0 - first_normalizer)).abs() < 1.0e-6);
}

#[test]
fn topk_sampled_logprobs_validate_k_ids_and_low_precision() {
    assert_eq!(
        TopKSampledLogprobsSpec::new(1, 64, 33, DType::Bf16),
        Err(ContractError::TopKLogprobsOutOfRange {
            top_k: 33,
            maximum: MAX_TOPK_LOGPROBS,
        })
    );
    let spec = TopKSampledLogprobsSpec::new(1, 3, 2, DType::Bf16).unwrap();
    let logits = [
        bf16::from_f32(-1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(0.5),
    ];
    let mut token_ids = [-1_i32; 3];
    let mut logprobs = [0.0_f32; 3];
    let mut ranks = [0_i64; 1];
    topk_sampled_logprobs_bf16_reference(
        &logits,
        &[2],
        &mut token_ids,
        &mut logprobs,
        &mut ranks,
        spec,
    )
    .unwrap();
    assert_eq!(token_ids, [2, 1, 2]);
    assert_eq!(ranks, [2]);
    assert_eq!(
        topk_sampled_logprobs_bf16_reference(
            &logits,
            &[3],
            &mut token_ids,
            &mut logprobs,
            &mut ranks,
            spec,
        ),
        Err(ContractError::TokenIdOutOfBounds {
            row: 0,
            token_id: 3,
            vocab_size: 3,
        })
    );
}

#[test]
fn topk_sampled_logprobs_workspace_bounds_every_radix_partition() {
    let cases = [
        (1, 151_936, 20, 128, 22_016),
        (32, 151_936, 20, 38, 209_152),
        (1, 5, 3, 1, 36),
        (1, 524_289, 1, 129, 2_580),
    ];
    for (rows, vocab_size, top_k, partitions, workspace_bytes) in cases {
        let spec = TopKSampledLogprobsSpec::new(rows, vocab_size, top_k, DType::F32).unwrap();
        assert_eq!(spec.workspace_partitions(), partitions);
        assert_eq!(spec.workspace_bytes(), workspace_bytes);
        assert!(vocab_size.div_ceil(partitions) <= 4_096);
    }
}

#[test]
fn token_penalties_match_sparse_history_semantics() {
    let spec = TokenPenaltiesSpec::new(2, 6, 5, 4, 32).unwrap();
    let mut logits = [
        2.0_f32, -2.0, 0.0, 4.0, -4.0, 1.0, -1.0, 3.0, -3.0, 2.0, -2.0, 0.5,
    ];
    let prompt_token_ids = [0_i64, 1, 1, -1, 6, 1, 2, 5, 8, -1];
    let output_token_ids = [1_i64, 1, 2, 6, 2, 2, 3, -1];
    let presence = [0.4_f32, 0.25];
    let frequency = [0.2_f32, -0.5];
    let repetition = [2.0_f32, 1.25];

    apply_token_penalties_f32_reference(
        &mut logits,
        &prompt_token_ids,
        &output_token_ids,
        &presence,
        &frequency,
        &repetition,
        spec,
    )
    .unwrap();

    let expected = [
        1.0_f32, -4.8, -0.6, 4.0, -4.0, 1.0, -1.0, 2.4, -3.0, 1.85, -2.0, 0.4,
    ];
    for (actual, expected) in logits.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
}

#[test]
fn token_penalties_require_a_power_of_two_workspace_at_half_load() {
    assert_eq!(
        TokenPenaltiesSpec::required_workspace_capacity(5, 4),
        Ok(32)
    );
    assert_eq!(
        TokenPenaltiesSpec::new(2, 6, 5, 4, 16),
        Err(ContractError::TokenPenaltyWorkspaceTooSmall {
            required: 32,
            actual: 16,
        })
    );
    assert_eq!(
        TokenPenaltiesSpec::new(2, 6, 5, 4, 48),
        Err(ContractError::TokenPenaltyWorkspaceTooSmall {
            required: 32,
            actual: 48,
        })
    );
}
