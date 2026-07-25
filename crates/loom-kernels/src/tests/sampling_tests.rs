use crate::*;
use half::bf16;

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
