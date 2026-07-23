use crate::*;

#[test]
fn greedy_speculative_verify_handles_mismatch_full_accept_and_ragged_rows() {
    let spec = GreedySpeculativeVerifySpec::new(3, 7, 4).unwrap();
    let draft = [10_i32, 11, 12, 20, 21, 22, 23];
    let target = [10_i64, 99, 12, 20, 21, 22, 23];
    let bonus = [100_i32, 200, 300];
    let cumulative = [3_i32, 3, 7];
    let mut output = [0_i32; 15];
    let mut accepted = [-1_i32; 3];
    let mut emitted = [-1_i32; 3];

    greedy_speculative_verify_reference(
        &draft,
        &target,
        &bonus,
        &cumulative,
        &mut output,
        &mut accepted,
        &mut emitted,
        spec,
    )
    .unwrap();

    assert_eq!(
        output,
        [10, 99, -1, -1, -1, 200, -1, -1, -1, -1, 20, 21, 22, 23, 300,]
    );
    assert_eq!(accepted, [1, 0, 4]);
    assert_eq!(emitted, [2, 1, 5]);
}

#[test]
fn greedy_speculative_verify_validates_metadata_before_mutating_outputs() {
    let spec = GreedySpeculativeVerifySpec::new(2, 3, 2).unwrap();
    let mut output = [17_i32; 6];
    let mut accepted = [17_i32; 2];
    let mut emitted = [17_i32; 2];
    let error = greedy_speculative_verify_reference(
        &[1, 2, 3],
        &[1, 2, 3],
        &[4, 5],
        &[2, 1],
        &mut output,
        &mut accepted,
        &mut emitted,
        spec,
    )
    .unwrap_err();

    assert_eq!(
        error,
        ContractError::InvalidCumulativeDraftLength {
            request: 1,
            previous: 2,
            current: 1,
            draft_tokens: 3,
            max_draft_tokens: 2,
        }
    );
    assert_eq!(output, [17; 6]);
    assert_eq!(accepted, [17; 2]);
    assert_eq!(emitted, [17; 2]);
}
