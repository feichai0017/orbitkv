use crate::sampling::{categorical_inverse_cdf, categorical_philox_word};
use crate::*;
use half::{bf16, f16};

#[test]
fn categorical_sample_uses_canonical_philox_and_ascending_inverse_cdf() {
    assert_eq!(categorical_philox_word(0, 0), 0x6627_e8d5);
    assert_eq!(categorical_inverse_cdf(&[0.25, 0.25, 0.5], 0.0), 0);
    assert_eq!(categorical_inverse_cdf(&[0.25, 0.25, 0.5], 0.25), 1);
    assert_eq!(categorical_inverse_cdf(&[0.25, 0.25, 0.5], 0.5), 2);
    assert_eq!(categorical_inverse_cdf(&[0.0, 0.5, 0.0, 0.5], 0.0), 1);
    assert_eq!(
        categorical_inverse_cdf(&[0.2, 0.3, 0.499_995], 0.999_999),
        2
    );
}

#[test]
fn categorical_sample_replays_and_advances_each_counter_once() {
    let spec = CategoricalSampleSpec::new(3, 4).unwrap();
    let probabilities = [
        0.1_f32, 0.2, 0.3, 0.4, //
        1.0, 0.0, 0.0, 0.0, //
        0.0, 0.25, 0.75, 0.0,
    ];
    let initial_state = [0_i64, 0, 17, 41, i64::MAX, i64::MAX - 1];
    let mut first_state = initial_state;
    let mut replay_state = initial_state;
    let mut first_tokens = [-1_i64; 3];
    let mut replay_tokens = [-1_i64; 3];

    categorical_sample_f32_reference(&probabilities, &mut first_state, &mut first_tokens, spec)
        .unwrap();
    categorical_sample_f32_reference(&probabilities, &mut replay_state, &mut replay_tokens, spec)
        .unwrap();

    assert_eq!(first_tokens, replay_tokens);
    assert_eq!(first_tokens[0], 2);
    assert_eq!(first_tokens[1], 0);
    assert!(matches!(first_tokens[2], 1 | 2));
    assert_eq!(first_state, replay_state);
    assert_eq!(first_state, [0, 1, 17, 42, i64::MAX, i64::MAX]);
}

#[test]
fn categorical_sample_rejects_invalid_inputs_atomically() {
    let spec = CategoricalSampleSpec::new(2, 3).unwrap();
    let valid_probabilities = [0.2_f32, 0.3, 0.5, 0.0, 0.25, 0.75];

    let cases = [
        (
            [0.2_f32, 0.3, 0.5, 0.0, -0.25, 1.25],
            [7_i64, 3, 9, 4],
            ContractError::InvalidCategoricalProbability {
                row: 1,
                column: 1,
                value: -0.25,
            },
        ),
        (
            [0.2_f32, 0.3, 0.5, 0.0, 0.0, 0.0],
            [7_i64, 3, 9, 4],
            ContractError::NoPositiveCategoricalProbability { row: 1 },
        ),
        (
            [0.2_f32, 0.3, 0.5, 0.1, 0.2, 0.3],
            [7_i64, 3, 9, 4],
            ContractError::CategoricalProbabilitySumOutOfRange {
                row: 1,
                sum: 0.6000000163912773,
                tolerance: CATEGORICAL_PROBABILITY_SUM_TOLERANCE,
            },
        ),
    ];
    for (probabilities, initial_state, expected_error) in cases {
        let mut state = initial_state;
        let mut tokens = [91_i64, 92];
        assert_eq!(
            categorical_sample_f32_reference(&probabilities, &mut state, &mut tokens, spec,),
            Err(expected_error)
        );
        assert_eq!(state, initial_state);
        assert_eq!(tokens, [91, 92]);
    }

    for (initial_state, expected_error) in [
        (
            [7_i64, 3, -1, 4],
            ContractError::InvalidRngState {
                row: 1,
                component: "seed",
                value: -1,
            },
        ),
        (
            [7_i64, 3, 9, -1],
            ContractError::InvalidRngState {
                row: 1,
                component: "counter",
                value: -1,
            },
        ),
        (
            [7_i64, 3, 9, i64::MAX],
            ContractError::RngCounterExhausted { row: 1 },
        ),
    ] {
        let mut state = initial_state;
        let mut tokens = [91_i64, 92];
        assert_eq!(
            categorical_sample_f32_reference(&valid_probabilities, &mut state, &mut tokens, spec,),
            Err(expected_error)
        );
        assert_eq!(state, initial_state);
        assert_eq!(tokens, [91, 92]);
    }

    let mut state = [7_i64, 3, 9, 4];
    let initial_state = state;
    let mut tokens = [91_i64, 92];
    let mut non_finite = valid_probabilities;
    non_finite[4] = f32::INFINITY;
    assert_eq!(
        categorical_sample_f32_reference(&non_finite, &mut state, &mut tokens, spec),
        Err(ContractError::InvalidCategoricalProbability {
            row: 1,
            column: 1,
            value: f32::INFINITY,
        })
    );
    assert_eq!(state, initial_state);
    assert_eq!(tokens, [91, 92]);

    non_finite[4] = f32::NAN;
    assert!(matches!(
        categorical_sample_f32_reference(&non_finite, &mut state, &mut tokens, spec),
        Err(ContractError::InvalidCategoricalProbability {
            row: 1,
            column: 1,
            value,
        }) if value.is_nan()
    ));
    assert_eq!(state, initial_state);
    assert_eq!(tokens, [91, 92]);
}

#[test]
fn categorical_sample_validates_exact_buffer_lengths_before_mutation() {
    let spec = CategoricalSampleSpec::new(2, 3).unwrap();
    let probabilities = [0.2_f32, 0.3, 0.5, 0.0, 0.25, 0.75];
    let mut state = [7_i64, 3, 9, 4];
    let initial_state = state;
    let mut tokens = [91_i64, 92];

    assert_eq!(
        categorical_sample_f32_reference(&probabilities[..5], &mut state, &mut tokens, spec),
        Err(ContractError::LengthMismatch {
            buffer: "probabilities",
            expected: 6,
            actual: 5,
        })
    );
    assert_eq!(state, initial_state);
    assert_eq!(tokens, [91, 92]);
    assert_eq!(
        categorical_sample_f32_reference(&probabilities, &mut state[..3], &mut tokens, spec),
        Err(ContractError::LengthMismatch {
            buffer: "rng_state",
            expected: 4,
            actual: 3,
        })
    );
    assert_eq!(state, initial_state);
    assert_eq!(tokens, [91, 92]);
    assert_eq!(
        categorical_sample_f32_reference(&probabilities, &mut state, &mut tokens[..1], spec),
        Err(ContractError::LengthMismatch {
            buffer: "token_ids",
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(state, initial_state);
    assert_eq!(tokens, [91, 92]);
}

#[test]
fn categorical_sample_matches_declared_distribution_and_never_selects_zero_mass() {
    const SAMPLES: usize = 65_536;
    let spec = CategoricalSampleSpec::new(SAMPLES, 4).unwrap();
    let mut probabilities = Vec::with_capacity(spec.probabilities_numel());
    let mut state = Vec::with_capacity(spec.rng_state_numel());
    for counter in 0..SAMPLES {
        probabilities.extend_from_slice(&[0.0_f32, 0.125, 0.375, 0.5]);
        state.extend_from_slice(&[31_i64, counter as i64]);
    }
    let mut tokens = vec![-1_i64; SAMPLES];

    categorical_sample_f32_reference(&probabilities, &mut state, &mut tokens, spec).unwrap();

    let mut counts = [0_usize; 4];
    for token in tokens {
        counts[token as usize] += 1;
    }
    assert_eq!(counts[0], 0);
    for (count, expected) in counts.into_iter().zip([0.0_f64, 0.125, 0.375, 0.5]) {
        let observed = count as f64 / SAMPLES as f64;
        assert!(
            (observed - expected).abs() < 0.008,
            "observed={observed}, expected={expected}, counts={counts:?}"
        );
    }
    assert_eq!(&state[..4], &[31, 1, 31, 2]);
    assert_eq!(
        &state[state.len() - 4..],
        &[31, SAMPLES as i64 - 1, 31, SAMPLES as i64]
    );
}

#[test]
fn categorical_sample_spec_rejects_zero_and_overflowing_shapes() {
    assert_eq!(
        CategoricalSampleSpec::new(0, 4),
        Err(ContractError::ZeroDimension)
    );
    assert_eq!(
        CategoricalSampleSpec::new(4, 0),
        Err(ContractError::ZeroDimension)
    );
    assert_eq!(
        CategoricalSampleSpec::new(usize::MAX, 2),
        Err(ContractError::ElementCountOverflow)
    );
}

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
fn top_p_renorm_filters_and_normalizes_with_deterministic_ties() {
    let spec = TopPRenormSpec::new(2, 4, DType::F32).unwrap();
    assert_eq!(spec.workspace_partitions(), 1);
    assert_eq!(spec.workspace_bytes(), 98_336);
    let mut logits = vec![
        0.0,
        0.0,
        0.0,
        0.0,
        0.6_f32.ln(),
        0.25_f32.ln(),
        0.15_f32.ln(),
        f32::NEG_INFINITY,
    ];
    let mut probabilities = vec![-1.0; 8];
    top_p_renorm_f32_reference(&mut logits, &[0.5, 0.7], &mut probabilities, spec).unwrap();

    assert_eq!(
        logits,
        vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.6_f32.ln(),
            0.25_f32.ln(),
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ]
    );
    assert_eq!(&probabilities[..4], &[0.0, 0.0, 0.5, 0.5]);
    assert!((probabilities[4] - 0.6 / 0.85).abs() < 1.0e-6);
    assert!((probabilities[5] - 0.25 / 0.85).abs() < 1.0e-6);
    assert_eq!(&probabilities[6..], &[0.0, 0.0]);
}

#[test]
fn top_p_renorm_supports_low_precision_and_validates_before_mutation() {
    let spec = TopPRenormSpec::new(2, 3, DType::Bf16).unwrap();
    let original = [
        bf16::from_f32(3.0),
        bf16::from_f32(2.0),
        bf16::NEG_INFINITY,
        bf16::from_f32(1.0),
        bf16::from_f32(0.0),
        bf16::from_f32(-1.0),
    ];
    let mut logits = original;
    let mut probabilities = [7.0; 6];
    assert_eq!(
        top_p_renorm_bf16_reference(&mut logits, &[0.9, 0.0], &mut probabilities, spec),
        Err(ContractError::InvalidProbability {
            parameter: "top_p",
            row: 1,
            value: 0.0,
        })
    );
    assert_eq!(logits, original);
    assert_eq!(probabilities, [7.0; 6]);

    let mut invalid = original;
    invalid[4] = bf16::from_f32(f32::INFINITY);
    assert_eq!(
        top_p_renorm_bf16_reference(&mut invalid, &[0.9, 0.9], &mut probabilities, spec),
        Err(ContractError::InvalidLogit {
            row: 1,
            column: 1,
            value: f32::INFINITY,
        })
    );

    let all_masked_spec = TopPRenormSpec::new(1, 2, DType::Bf16).unwrap();
    assert_eq!(
        top_p_renorm_bf16_reference(
            &mut [bf16::NEG_INFINITY; 2],
            &[0.9],
            &mut [0.0; 2],
            all_masked_spec,
        ),
        Err(ContractError::NoFiniteLogit { row: 0 })
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
