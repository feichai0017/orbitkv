use half::bf16;

use crate::*;

#[test]
fn moe_permute_is_stable_and_emits_grouped_gemm_offsets() {
    let spec = MoePermuteSpec::new(4, 4, 2, 3, 3, false, DType::F32).unwrap();
    let hidden: Vec<f32> = (0..16).map(|value| value as f32).collect();
    let topk_ids = [2, 0, 1, 2, 0, 1, 2, 1];
    let mut permuted = vec![-1.0_f32; spec.permuted_hidden_numel()];
    let mut offsets = vec![-1_i64; spec.expert_offset_count()];
    let mut inverse = vec![-1_i32; spec.assignment_count()];
    let mut assignments = vec![-1_i32; spec.assignment_count()];

    moe_permute_f32_reference(
        &hidden,
        &topk_ids,
        None,
        &mut permuted,
        &mut offsets,
        &mut inverse,
        &mut assignments,
        spec,
    )
    .unwrap();

    assert_eq!(offsets, [0, 2, 5, 8]);
    assert_eq!(assignments, [1, 4, 2, 5, 7, 0, 3, 6]);
    assert_eq!(inverse, [5, 0, 2, 6, 1, 3, 7, 4]);
    let expected_tokens = [0_usize, 2, 1, 2, 3, 0, 1, 3];
    for (row, &token) in expected_tokens.iter().enumerate() {
        assert_eq!(
            &permuted[row * 4..row * 4 + 4],
            &hidden[token * 4..token * 4 + 4]
        );
    }
}

#[test]
fn moe_permute_maps_expert_parallel_routes_and_zeroes_remote_rows() {
    let spec = MoePermuteSpec::new(4, 2, 2, 3, 2, true, DType::F32).unwrap();
    let hidden = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let topk_ids = [2, 0, 1, 2, 0, 1, 2, 1];
    let expert_map = [1_i32, -1, 0];
    let mut permuted = vec![17.0_f32; spec.permuted_hidden_numel()];
    let mut offsets = vec![-1_i64; spec.expert_offset_count()];
    let mut inverse = vec![-1_i32; spec.assignment_count()];
    let mut assignments = vec![-1_i32; spec.assignment_count()];

    moe_permute_f32_reference(
        &hidden,
        &topk_ids,
        Some(&expert_map),
        &mut permuted,
        &mut offsets,
        &mut inverse,
        &mut assignments,
        spec,
    )
    .unwrap();

    assert_eq!(offsets, [0, 3, 5]);
    assert_eq!(assignments, [0, 3, 6, 1, 4, 8, 8, 8]);
    assert_eq!(inverse, [0, 3, 5, 1, 4, 6, 2, 7]);
    assert!(permuted[5 * 2..].iter().all(|&value| value == 0.0));
}

#[test]
fn moe_permute_orders_remote_tail_by_global_expert() {
    let spec = MoePermuteSpec::new(6, 1, 2, 6, 3, true, DType::F32).unwrap();
    let hidden = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let topk_ids = [4, 0, 1, 5, 3, 2, 5, 1, 0, 4, 2, 3];
    let expert_map = [0_i32, 1, 2, -1, -1, -1];
    let mut permuted = vec![17.0_f32; spec.permuted_hidden_numel()];
    let mut offsets = vec![-1_i64; spec.expert_offset_count()];
    let mut inverse = vec![-1_i32; spec.assignment_count()];
    let mut assignments = vec![-1_i32; spec.assignment_count()];

    moe_permute_f32_reference(
        &hidden,
        &topk_ids,
        Some(&expert_map),
        &mut permuted,
        &mut offsets,
        &mut inverse,
        &mut assignments,
        spec,
    )
    .unwrap();

    assert_eq!(offsets, [0, 2, 4, 6]);
    assert_eq!(assignments, [1, 8, 2, 7, 5, 10, 12, 12, 12, 12, 12, 12]);
    assert_eq!(inverse, [8, 0, 2, 10, 6, 4, 11, 3, 1, 9, 5, 7]);
    assert_eq!(&permuted[..6], &[0.0, 4.0, 1.0, 3.0, 2.0, 5.0]);
    assert!(permuted[6..].iter().all(|&value| value == 0.0));
}

#[test]
fn moe_permute_preserves_fp8_storage_bytes() {
    let spec = MoePermuteSpec::new(3, 4, 2, 3, 3, false, DType::Fp8E4M3Fn).unwrap();
    let hidden = [
        0x38_u8, 0x40, 0x44, 0x48, 0xB8, 0xC0, 0xC4, 0xC8, 0x01, 0x02, 0x03, 0x04,
    ];
    let topk_ids = [2, 0, 1, 2, 0, 1];
    let mut permuted = vec![0xFF_u8; spec.permuted_hidden_numel()];
    let mut offsets = vec![-1_i64; spec.expert_offset_count()];
    let mut inverse = vec![-1_i32; spec.assignment_count()];
    let mut assignments = vec![-1_i32; spec.assignment_count()];

    moe_permute_fp8_e4m3fn_reference(
        &hidden,
        &topk_ids,
        None,
        &mut permuted,
        &mut offsets,
        &mut inverse,
        &mut assignments,
        spec,
    )
    .unwrap();

    assert_eq!(offsets, [0, 2, 4, 6]);
    assert_eq!(assignments, [1, 4, 2, 5, 0, 3]);
    let expected_tokens = [0_usize, 2, 1, 2, 0, 1];
    for (row, &token) in expected_tokens.iter().enumerate() {
        assert_eq!(
            &permuted[row * 4..row * 4 + 4],
            &hidden[token * 4..token * 4 + 4]
        );
    }
}

#[test]
fn moe_combine_skips_remote_assignments_and_rounds_once() {
    let spec = MoeCombineSpec::new(2, 2, 2, 2, DType::Bf16).unwrap();
    let expert_outputs = [
        bf16::from_f32(2.0),
        bf16::from_f32(4.0),
        bf16::from_f32(6.0),
        bf16::from_f32(8.0),
        bf16::ZERO,
        bf16::ZERO,
        bf16::ZERO,
        bf16::ZERO,
    ];
    let routing_weights = [0.25_f32, 0.75, 0.4, 0.6];
    let inverse = [0_i32, 2, 3, 1];
    let offsets = [0_i64, 1, 2];
    let mut output = [bf16::from_f32(17.0); 4];

    moe_combine_bf16_reference(
        &expert_outputs,
        &routing_weights,
        &inverse,
        &offsets,
        &mut output,
        spec,
    )
    .unwrap();

    assert_eq!(output[0], bf16::from_f32(0.5));
    assert_eq!(output[1], bf16::from_f32(1.0));
    assert_eq!(output[2], bf16::from_f32(3.6));
    assert_eq!(output[3], bf16::from_f32(4.8));
}

#[test]
fn moe_references_validate_metadata_before_mutating_outputs() {
    assert_eq!(
        MoePermuteSpec::new(1, 1, 1, 1, 1, false, DType::I8),
        Err(ContractError::UnsupportedDType(DType::I8))
    );
    assert_eq!(
        MoeCombineSpec::new(1, 1, 1, 1, DType::Fp8E4M3Fn),
        Err(ContractError::UnsupportedDType(DType::Fp8E4M3Fn))
    );
    let oversized_expert_count = i32::MAX as usize + 1;
    assert_eq!(
        MoePermuteSpec::new(1, 1, 1, oversized_expert_count, 1, true, DType::F32,),
        Err(ContractError::MoeExpertCountOutOfRange {
            num_experts: oversized_expert_count,
        })
    );

    let spec = MoePermuteSpec::new(1, 2, 1, 2, 2, false, DType::F32).unwrap();
    let mut permuted = [17.0_f32; 2];
    let mut offsets = [17_i64; 3];
    let mut inverse = [17_i32; 1];
    let mut assignments = [17_i32; 1];
    let error = moe_permute_f32_reference(
        &[1.0, 2.0],
        &[2],
        None,
        &mut permuted,
        &mut offsets,
        &mut inverse,
        &mut assignments,
        spec,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ContractError::MoeExpertIdOutOfRange {
            assignment: 0,
            expert_id: 2,
            num_experts: 2,
        }
    );
    assert_eq!(permuted, [17.0; 2]);
    assert_eq!(offsets, [17; 3]);
    assert_eq!(inverse, [17]);
    assert_eq!(assignments, [17]);

    let combine_spec = MoeCombineSpec::new(1, 2, 2, 1, DType::F32).unwrap();
    let mut output = [17.0_f32; 2];
    let error = moe_combine_f32_reference(
        &[1.0, 2.0, 3.0, 4.0],
        &[0.5, f32::NAN],
        &[0, 1],
        &[0, 2],
        &mut output,
        combine_spec,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ContractError::MoeRoutingWeightNotFinite { assignment: 1, .. }
    ));
    assert_eq!(output, [17.0; 2]);
}
