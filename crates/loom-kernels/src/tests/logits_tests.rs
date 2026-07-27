use crate::*;
use half::f16;

#[test]
fn logits_preprocess_fuses_mask_bias_suppression_and_temperature() {
    let spec = LogitsPreprocessSpec::new(2, 5, true, 3, 3).unwrap();
    let mut logits = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, //
        -1.0, -2.0, -3.0, -4.0, -5.0,
    ];
    let blocked_mask = [0_u8, 1, 0, 0, 0, 0, 0, 0, 1, 0];

    logits_preprocess_f32_reference(
        &mut logits,
        &[0.0, 0.5],
        Some(&blocked_mask),
        &[0, 0, 1],
        &[0, 2, 4],
        &[0.25, -0.5, 1.0],
        &[0, 1, 1],
        &[3, 0, 0],
        spec,
    )
    .unwrap();

    assert_eq!(
        logits,
        [
            1.25,
            f32::NEG_INFINITY,
            2.5,
            f32::NEG_INFINITY,
            5.0,
            f32::NEG_INFINITY,
            -4.0,
            -6.0,
            f32::NEG_INFINITY,
            -8.0,
        ]
    );
}

#[test]
fn logits_preprocess_validates_all_metadata_before_mutation() {
    let spec = LogitsPreprocessSpec::new(1, 3, false, 2, 0).unwrap();
    let original = [1.0_f32, 2.0, 3.0];
    let mut logits = original;

    let error = logits_preprocess_f32_reference(
        &mut logits,
        &[1.0],
        None,
        &[0, 0],
        &[1, 1],
        &[0.5, 0.25],
        &[],
        &[],
        spec,
    )
    .unwrap_err();

    assert_eq!(
        error,
        ContractError::DuplicateLogitBias {
            first_entry: 0,
            second_entry: 1,
            row_id: 0,
            token_id: 1,
        }
    );
    assert_eq!(logits, original);

    let invalid_temperature = LogitsPreprocessSpec::new(1, 3, false, 0, 0).unwrap();
    assert_eq!(
        logits_preprocess_f32_reference(
            &mut logits,
            &[-1.0],
            None,
            &[],
            &[],
            &[],
            &[],
            &[],
            invalid_temperature,
        ),
        Err(ContractError::InvalidTemperature {
            row: 0,
            value: -1.0,
        })
    );
    assert_eq!(logits, original);
}

#[test]
fn min_p_filter_matches_the_softmax_ratio_definition() {
    let spec = MinPFilterSpec::new(3, 4, DType::F32).unwrap();
    let original = [
        1.0_f32, 3.0, 2.0, -1.0, //
        -2.0, -1.0, 2.0, 0.0, //
        4.0, 4.0, 3.0, -8.0,
    ];
    let mut logits = original;

    min_p_filter_f32_reference(&mut logits, &[0.0, 0.2, 1.0], spec).unwrap();

    assert_eq!(&logits[..4], &original[..4]);
    let threshold = 2.0 + 0.2_f32.ln();
    for (actual, &input) in logits[4..8].iter().zip(&original[4..8]) {
        if input < threshold {
            assert_eq!(*actual, f32::NEG_INFINITY);
        } else {
            assert_eq!(*actual, input);
        }
    }
    assert_eq!(
        &logits[8..],
        &[4.0, 4.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
    );
}

#[test]
fn min_p_filter_validates_metadata_before_mutating_logits() {
    let spec = MinPFilterSpec::new(2, 2, DType::F16).unwrap();
    let original = [
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    let mut logits = original;

    let error = min_p_filter_f16_reference(&mut logits, &[0.5, 1.1], spec).unwrap_err();

    assert_eq!(
        error,
        ContractError::InvalidProbability {
            parameter: "min_p",
            row: 1,
            value: 1.1,
        }
    );
    assert_eq!(logits, original);
}
