use super::*;

#[test]
fn reference_rotates_split_halves_and_preserves_tail() {
    let spec = Bf16RopePosIdsSpec::new(1, 1, 1, 6, 4, 1.0, 10_000.0).unwrap();
    let query = [
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
        bf16::from_f32(6.0),
    ];
    let key = query;
    let mut query_output = [bf16::NAN; 6];
    let mut key_output = [bf16::NAN; 6];

    rope_pos_ids_bf16_reference(&query, &key, &[1], &mut query_output, &mut key_output, spec)
        .unwrap();

    let expected_pair_0 = (
        1.0_f32 * 1.0_f32.cos() - 3.0_f32 * 1.0_f32.sin(),
        3.0_f32 * 1.0_f32.cos() + 1.0_f32 * 1.0_f32.sin(),
    );
    let angle_1 = 10_000.0_f32.powf(-0.5);
    let expected_pair_1 = (
        2.0 * angle_1.cos() - 4.0 * angle_1.sin(),
        4.0 * angle_1.cos() + 2.0 * angle_1.sin(),
    );
    assert_eq!(
        query_output[0],
        bf16::from_f32(expected_pair_0.0),
        "first split-half component"
    );
    assert_eq!(query_output[2], bf16::from_f32(expected_pair_0.1));
    assert_eq!(query_output[1], bf16::from_f32(expected_pair_1.0));
    assert_eq!(query_output[3], bf16::from_f32(expected_pair_1.1));
    assert_eq!(&query_output[4..], &query[4..]);
    assert_eq!(query_output, key_output);
}

#[test]
fn zero_position_is_bit_exact_for_long_context_contract() {
    let spec = Bf16RopePosIdsSpec::new(1, 2, 1, 128, 128, 1.0, 10_000.0).unwrap();
    let query = (0..spec.query_numel())
        .map(|index| bf16::from_f32(index as f32 / 256.0 - 0.5))
        .collect::<Vec<_>>();
    let key = (0..spec.key_numel())
        .map(|index| bf16::from_f32(index as f32 / 128.0 - 0.25))
        .collect::<Vec<_>>();
    let mut query_output = vec![bf16::NAN; spec.query_numel()];
    let mut key_output = vec![bf16::NAN; spec.key_numel()];

    rope_pos_ids_bf16_reference(&query, &key, &[0], &mut query_output, &mut key_output, spec)
        .unwrap();

    assert_eq!(query, query_output);
    assert_eq!(key, key_output);
}

#[test]
fn reference_accepts_large_position_ids_and_stays_finite() {
    let spec = Bf16RopePosIdsSpec::new(1, 1, 1, 128, 128, 1.0, 10_000.0).unwrap();
    let query = vec![bf16::from_f32(0.25); spec.query_numel()];
    let key = vec![bf16::from_f32(-0.5); spec.key_numel()];
    let mut query_output = vec![bf16::NAN; spec.query_numel()];
    let mut key_output = vec![bf16::NAN; spec.key_numel()];

    rope_pos_ids_bf16_reference(
        &query,
        &key,
        &[32_767],
        &mut query_output,
        &mut key_output,
        spec,
    )
    .unwrap();

    assert!(query_output.iter().all(|value| value.to_f32().is_finite()));
    assert!(key_output.iter().all(|value| value.to_f32().is_finite()));
}

#[test]
fn spec_and_reference_reject_invalid_contracts() {
    assert_eq!(
        Bf16RopePosIdsSpec::new(1, 1, 1, 128, 127, 1.0, 10_000.0),
        Err(ContractError::InvalidRotaryDimension {
            rotary_dim: 127,
            head_dim: 128,
        })
    );
    assert_eq!(
        Bf16RopePosIdsSpec::new(1, 1, 1, 128, 128, 0.0, 10_000.0),
        Err(ContractError::InvalidRopeScale(0.0))
    );
    assert_eq!(
        Bf16RopePosIdsSpec::new(1, 1, 1, 128, 128, 1.0, 1.0),
        Err(ContractError::InvalidRopeTheta(1.0))
    );

    let spec = Bf16RopePosIdsSpec::new(1, 1, 1, 2, 2, 1.0, 10_000.0).unwrap();
    assert_eq!(
        rope_pos_ids_bf16_reference(
            &[bf16::ZERO; 2],
            &[bf16::ZERO; 2],
            &[-1],
            &mut [bf16::ZERO; 2],
            &mut [bf16::ZERO; 2],
            spec,
        ),
        Err(ContractError::NegativePositionId {
            token: 0,
            position: -1,
        })
    );
    assert_eq!(
        rope_pos_ids_bf16_reference(
            &[bf16::ZERO; 1],
            &[bf16::ZERO; 2],
            &[0],
            &mut [bf16::ZERO; 2],
            &mut [bf16::ZERO; 2],
            spec,
        ),
        Err(ContractError::LengthMismatch {
            buffer: "query",
            expected: 2,
            actual: 1,
        })
    );
}
