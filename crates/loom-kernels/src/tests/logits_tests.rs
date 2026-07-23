use crate::*;
use half::f16;

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
