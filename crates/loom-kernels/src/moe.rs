//! MoE token-movement contracts and CPU reference implementations.
//!
//! Loom owns only the routing metadata, activation movement, and weighted
//! combine boundaries around an engine-selected grouped GEMM. Matrix
//! multiplication is deliberately outside this module.

use half::{bf16, f16};

use crate::contract::{require_len, ContractError, DType};

/// Contract for stable expert-major token permutation.
///
/// `top_k` expert assignments are stored contiguously for every token. With an
/// expert map, global expert IDs map to local IDs and `-1` marks assignments
/// owned by another expert-parallel rank. Valid assignments are sorted by
/// local expert. Remote assignments follow every local expert, are ordered by
/// global expert ID, and retain their original flattened order within each
/// expert. The remote order matches vLLM's grouped-GEMM movement metadata even
/// though remote activation rows themselves are zero-filled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoePermuteSpec {
    tokens: usize,
    hidden_size: usize,
    top_k: usize,
    num_experts: usize,
    num_local_experts: usize,
    has_expert_map: bool,
    dtype: DType,
}

impl MoePermuteSpec {
    /// Creates a contiguous MoE dispatch-movement contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokens: usize,
        hidden_size: usize,
        top_k: usize,
        num_experts: usize,
        num_local_experts: usize,
        has_expert_map: bool,
        dtype: DType,
    ) -> Result<Self, ContractError> {
        if tokens == 0
            || hidden_size == 0
            || top_k == 0
            || num_experts == 0
            || num_local_experts == 0
        {
            return Err(ContractError::ZeroDimension);
        }
        if !matches!(
            dtype,
            DType::F32 | DType::F16 | DType::Bf16 | DType::Fp8E4M3Fn
        ) {
            return Err(ContractError::UnsupportedDType(dtype));
        }
        if num_experts > i32::MAX as usize {
            return Err(ContractError::MoeExpertCountOutOfRange { num_experts });
        }
        if top_k > num_experts {
            return Err(ContractError::MoeTopKOutOfRange { top_k, num_experts });
        }
        if num_local_experts > num_experts {
            return Err(ContractError::MoeLocalExpertCountOutOfRange {
                num_local_experts,
                num_experts,
            });
        }
        if !has_expert_map && num_local_experts != num_experts {
            return Err(ContractError::MoeExpertMapRequired {
                num_local_experts,
                num_experts,
            });
        }
        let assignments = tokens
            .checked_mul(top_k)
            .ok_or(ContractError::ElementCountOverflow)?;
        if assignments > i32::MAX as usize {
            return Err(ContractError::MoeAssignmentCountOutOfRange { assignments });
        }
        assignments
            .checked_mul(hidden_size)
            .ok_or(ContractError::ElementCountOverflow)?;
        num_experts
            .checked_mul(2)
            .ok_or(ContractError::ElementCountOverflow)?;
        num_local_experts
            .checked_add(1)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            tokens,
            hidden_size,
            top_k,
            num_experts,
            num_local_experts,
            has_expert_map,
            dtype,
        })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub const fn top_k(self) -> usize {
        self.top_k
    }

    pub const fn num_experts(self) -> usize {
        self.num_experts
    }

    pub const fn num_local_experts(self) -> usize {
        self.num_local_experts
    }

    pub const fn has_expert_map(self) -> bool {
        self.has_expert_map
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn assignment_count(self) -> usize {
        self.tokens * self.top_k
    }

    pub const fn hidden_numel(self) -> usize {
        self.tokens * self.hidden_size
    }

    pub const fn permuted_hidden_numel(self) -> usize {
        self.assignment_count() * self.hidden_size
    }

    pub const fn expert_offset_count(self) -> usize {
        self.num_local_experts + 1
    }
}

/// Contract for inverse permutation and weighted expert-output reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeCombineSpec {
    tokens: usize,
    hidden_size: usize,
    top_k: usize,
    num_local_experts: usize,
    dtype: DType,
}

impl MoeCombineSpec {
    /// Creates a contiguous weighted combine contract.
    pub fn new(
        tokens: usize,
        hidden_size: usize,
        top_k: usize,
        num_local_experts: usize,
        dtype: DType,
    ) -> Result<Self, ContractError> {
        if tokens == 0 || hidden_size == 0 || top_k == 0 || num_local_experts == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
            return Err(ContractError::UnsupportedDType(dtype));
        }
        let assignments = tokens
            .checked_mul(top_k)
            .ok_or(ContractError::ElementCountOverflow)?;
        if assignments > i32::MAX as usize {
            return Err(ContractError::MoeAssignmentCountOutOfRange { assignments });
        }
        assignments
            .checked_mul(hidden_size)
            .ok_or(ContractError::ElementCountOverflow)?;
        tokens
            .checked_mul(hidden_size)
            .ok_or(ContractError::ElementCountOverflow)?;
        num_local_experts
            .checked_add(1)
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self {
            tokens,
            hidden_size,
            top_k,
            num_local_experts,
            dtype,
        })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub const fn top_k(self) -> usize {
        self.top_k
    }

    pub const fn num_local_experts(self) -> usize {
        self.num_local_experts
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn assignment_count(self) -> usize {
        self.tokens * self.top_k
    }

    pub const fn expert_output_numel(self) -> usize {
        self.assignment_count() * self.hidden_size
    }

    pub const fn output_numel(self) -> usize {
        self.tokens * self.hidden_size
    }

    pub const fn expert_offset_count(self) -> usize {
        self.num_local_experts + 1
    }
}

trait MoeStorage: Copy {
    fn zero() -> Self;
}

trait MoeArithmetic: MoeStorage {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl MoeStorage for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl MoeArithmetic for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl MoeStorage for f16 {
    fn zero() -> Self {
        Self::ZERO
    }
}

impl MoeArithmetic for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl MoeStorage for bf16 {
    fn zero() -> Self {
        Self::ZERO
    }
}

impl MoeArithmetic for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl MoeStorage for u8 {
    fn zero() -> Self {
        0
    }
}

fn validate_expert_map(
    expert_map: Option<&[i32]>,
    spec: MoePermuteSpec,
) -> Result<(), ContractError> {
    match (spec.has_expert_map(), expert_map) {
        (false, None) => Ok(()),
        (true, Some(mapping)) => {
            require_len("expert_map", mapping.len(), spec.num_experts())?;
            for (global_expert, &local_expert) in mapping.iter().enumerate() {
                if local_expert < -1 || local_expert >= spec.num_local_experts() as i32 {
                    return Err(ContractError::MoeExpertMapOutOfRange {
                        global_expert,
                        local_expert,
                        num_local_experts: spec.num_local_experts(),
                    });
                }
            }
            Ok(())
        }
        _ => Err(ContractError::MoeExpertMapPresenceMismatch {
            expected: spec.has_expert_map(),
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn moe_permute_reference<T: MoeStorage>(
    hidden_states: &[T],
    topk_ids: &[i32],
    expert_map: Option<&[i32]>,
    permuted_hidden_states: &mut [T],
    expert_offsets: &mut [i64],
    inverse_permutation: &mut [i32],
    permuted_assignment_ids: &mut [i32],
    spec: MoePermuteSpec,
) -> Result<(), ContractError> {
    require_len("hidden_states", hidden_states.len(), spec.hidden_numel())?;
    require_len("topk_ids", topk_ids.len(), spec.assignment_count())?;
    require_len(
        "permuted_hidden_states",
        permuted_hidden_states.len(),
        spec.permuted_hidden_numel(),
    )?;
    require_len(
        "expert_offsets",
        expert_offsets.len(),
        spec.expert_offset_count(),
    )?;
    require_len(
        "inverse_permutation",
        inverse_permutation.len(),
        spec.assignment_count(),
    )?;
    require_len(
        "permuted_assignment_ids",
        permuted_assignment_ids.len(),
        spec.assignment_count(),
    )?;
    validate_expert_map(expert_map, spec)?;

    let mut assignments = Vec::with_capacity(spec.assignment_count());
    for (assignment, &global_expert) in topk_ids.iter().enumerate() {
        let Ok(global_expert_index) = usize::try_from(global_expert) else {
            return Err(ContractError::MoeExpertIdOutOfRange {
                assignment,
                expert_id: global_expert,
                num_experts: spec.num_experts(),
            });
        };
        if global_expert_index >= spec.num_experts() {
            return Err(ContractError::MoeExpertIdOutOfRange {
                assignment,
                expert_id: global_expert,
                num_experts: spec.num_experts(),
            });
        }
        let local_expert = expert_map.map_or(global_expert, |mapping| mapping[global_expert_index]);
        let sort_key = if local_expert < 0 {
            spec.num_experts() + global_expert_index
        } else {
            local_expert as usize
        };
        assignments.push((sort_key, assignment));
    }
    assignments.sort_by_key(|&(expert, _)| expert);

    let sentinel = spec.assignment_count() as i32;
    permuted_hidden_states.fill(T::zero());
    permuted_assignment_ids.fill(sentinel);
    expert_offsets.fill(0);

    for &(local_expert, _) in &assignments {
        if local_expert < spec.num_local_experts() {
            expert_offsets[local_expert + 1] += 1;
        }
    }
    for expert in 0..spec.num_local_experts() {
        expert_offsets[expert + 1] += expert_offsets[expert];
    }

    for (permuted_row, &(local_expert, assignment)) in assignments.iter().enumerate() {
        inverse_permutation[assignment] = permuted_row as i32;
        if local_expert >= spec.num_local_experts() {
            continue;
        }
        permuted_assignment_ids[permuted_row] = assignment as i32;
        let token = assignment / spec.top_k();
        let source = token * spec.hidden_size();
        let destination = permuted_row * spec.hidden_size();
        permuted_hidden_states[destination..destination + spec.hidden_size()]
            .copy_from_slice(&hidden_states[source..source + spec.hidden_size()]);
    }
    Ok(())
}

/// Stable F32 expert-major activation permutation reference.
#[allow(clippy::too_many_arguments)]
pub fn moe_permute_f32_reference(
    hidden_states: &[f32],
    topk_ids: &[i32],
    expert_map: Option<&[i32]>,
    permuted_hidden_states: &mut [f32],
    expert_offsets: &mut [i64],
    inverse_permutation: &mut [i32],
    permuted_assignment_ids: &mut [i32],
    spec: MoePermuteSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F32 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    moe_permute_reference(
        hidden_states,
        topk_ids,
        expert_map,
        permuted_hidden_states,
        expert_offsets,
        inverse_permutation,
        permuted_assignment_ids,
        spec,
    )
}

/// Stable FP16 expert-major activation permutation reference.
#[allow(clippy::too_many_arguments)]
pub fn moe_permute_f16_reference(
    hidden_states: &[f16],
    topk_ids: &[i32],
    expert_map: Option<&[i32]>,
    permuted_hidden_states: &mut [f16],
    expert_offsets: &mut [i64],
    inverse_permutation: &mut [i32],
    permuted_assignment_ids: &mut [i32],
    spec: MoePermuteSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F16 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    moe_permute_reference(
        hidden_states,
        topk_ids,
        expert_map,
        permuted_hidden_states,
        expert_offsets,
        inverse_permutation,
        permuted_assignment_ids,
        spec,
    )
}

/// Stable BF16 expert-major activation permutation reference.
#[allow(clippy::too_many_arguments)]
pub fn moe_permute_bf16_reference(
    hidden_states: &[bf16],
    topk_ids: &[i32],
    expert_map: Option<&[i32]>,
    permuted_hidden_states: &mut [bf16],
    expert_offsets: &mut [i64],
    inverse_permutation: &mut [i32],
    permuted_assignment_ids: &mut [i32],
    spec: MoePermuteSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::Bf16 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    moe_permute_reference(
        hidden_states,
        topk_ids,
        expert_map,
        permuted_hidden_states,
        expert_offsets,
        inverse_permutation,
        permuted_assignment_ids,
        spec,
    )
}

/// Stable FP8 E4M3FN expert-major byte permutation reference.
///
/// Permutation does not interpret FP8 numerically; it preserves every storage
/// byte exactly and zero-fills remote expert-parallel rows.
#[allow(clippy::too_many_arguments)]
pub fn moe_permute_fp8_e4m3fn_reference(
    hidden_states: &[u8],
    topk_ids: &[i32],
    expert_map: Option<&[i32]>,
    permuted_hidden_states: &mut [u8],
    expert_offsets: &mut [i64],
    inverse_permutation: &mut [i32],
    permuted_assignment_ids: &mut [i32],
    spec: MoePermuteSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::Fp8E4M3Fn {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    moe_permute_reference(
        hidden_states,
        topk_ids,
        expert_map,
        permuted_hidden_states,
        expert_offsets,
        inverse_permutation,
        permuted_assignment_ids,
        spec,
    )
}

fn validate_combine_metadata(
    routing_weights: &[f32],
    inverse_permutation: &[i32],
    expert_offsets: &[i64],
    spec: MoeCombineSpec,
) -> Result<usize, ContractError> {
    for (assignment, &weight) in routing_weights.iter().enumerate() {
        if !weight.is_finite() {
            return Err(ContractError::MoeRoutingWeightNotFinite { assignment, weight });
        }
    }

    let mut previous = 0_i64;
    for (expert, &current) in expert_offsets.iter().enumerate() {
        if (expert == 0 && current != 0)
            || current < previous
            || usize::try_from(current).map_or(true, |value| value > spec.assignment_count())
        {
            return Err(ContractError::MoeExpertOffsetOutOfRange {
                expert,
                previous,
                current,
                assignments: spec.assignment_count(),
            });
        }
        previous = current;
    }

    let mut seen = vec![None; spec.assignment_count()];
    for (assignment, &permuted_row) in inverse_permutation.iter().enumerate() {
        let Ok(permuted_row) = usize::try_from(permuted_row) else {
            return Err(ContractError::MoePermutationIndexOutOfRange {
                assignment,
                permuted_row: inverse_permutation[assignment],
                assignments: spec.assignment_count(),
            });
        };
        if permuted_row >= spec.assignment_count() {
            return Err(ContractError::MoePermutationIndexOutOfRange {
                assignment,
                permuted_row: inverse_permutation[assignment],
                assignments: spec.assignment_count(),
            });
        }
        if let Some(first_assignment) = seen[permuted_row].replace(assignment) {
            return Err(ContractError::MoeDuplicatePermutationIndex {
                first_assignment,
                second_assignment: assignment,
                permuted_row,
            });
        }
    }
    Ok(previous as usize)
}

fn moe_combine_reference<T: MoeArithmetic>(
    expert_outputs: &[T],
    routing_weights: &[f32],
    inverse_permutation: &[i32],
    expert_offsets: &[i64],
    output: &mut [T],
    spec: MoeCombineSpec,
) -> Result<(), ContractError> {
    require_len(
        "expert_outputs",
        expert_outputs.len(),
        spec.expert_output_numel(),
    )?;
    require_len(
        "routing_weights",
        routing_weights.len(),
        spec.assignment_count(),
    )?;
    require_len(
        "inverse_permutation",
        inverse_permutation.len(),
        spec.assignment_count(),
    )?;
    require_len(
        "expert_offsets",
        expert_offsets.len(),
        spec.expert_offset_count(),
    )?;
    require_len("output", output.len(), spec.output_numel())?;
    let valid_assignments =
        validate_combine_metadata(routing_weights, inverse_permutation, expert_offsets, spec)?;

    let mut combined = vec![T::zero(); spec.output_numel()];
    for token in 0..spec.tokens() {
        for column in 0..spec.hidden_size() {
            let mut accumulator = 0.0_f32;
            for route in 0..spec.top_k() {
                let assignment = token * spec.top_k() + route;
                let permuted_row = inverse_permutation[assignment] as usize;
                if permuted_row < valid_assignments {
                    accumulator += routing_weights[assignment]
                        * expert_outputs[permuted_row * spec.hidden_size() + column].to_f32();
                }
            }
            combined[token * spec.hidden_size() + column] = T::from_f32(accumulator);
        }
    }
    output.copy_from_slice(&combined);
    Ok(())
}

/// F32 inverse permutation and weighted expert-output reduction reference.
pub fn moe_combine_f32_reference(
    expert_outputs: &[f32],
    routing_weights: &[f32],
    inverse_permutation: &[i32],
    expert_offsets: &[i64],
    output: &mut [f32],
    spec: MoeCombineSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F32 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    moe_combine_reference(
        expert_outputs,
        routing_weights,
        inverse_permutation,
        expert_offsets,
        output,
        spec,
    )
}

/// FP16 inverse permutation and weighted expert-output reduction reference.
pub fn moe_combine_f16_reference(
    expert_outputs: &[f16],
    routing_weights: &[f32],
    inverse_permutation: &[i32],
    expert_offsets: &[i64],
    output: &mut [f16],
    spec: MoeCombineSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F16 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    moe_combine_reference(
        expert_outputs,
        routing_weights,
        inverse_permutation,
        expert_offsets,
        output,
        spec,
    )
}

/// BF16 inverse permutation and weighted expert-output reduction reference.
pub fn moe_combine_bf16_reference(
    expert_outputs: &[bf16],
    routing_weights: &[f32],
    inverse_permutation: &[i32],
    expert_offsets: &[i64],
    output: &mut [bf16],
    spec: MoeCombineSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::Bf16 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    moe_combine_reference(
        expert_outputs,
        routing_weights,
        inverse_permutation,
        expert_offsets,
        output,
        spec,
    )
}
