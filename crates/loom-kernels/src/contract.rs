//! Shared tensor metadata and contract errors.

use std::fmt;

/// Element type stored by a tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    /// IEEE-754 single precision.
    F32,
    /// IEEE-754 half precision.
    F16,
    /// Brain floating point with an eight-bit exponent.
    Bf16,
    /// Signed eight-bit integer.
    I8,
    /// FP8 E4M3 finite-numbers encoding.
    Fp8E4M3Fn,
}

impl DType {
    /// Returns the number of bytes occupied by one element.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I8 | Self::Fp8E4M3Fn => 1,
        }
    }
}

/// A shape and stride contract without a data pointer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorSpec {
    /// Builds a row-major contiguous tensor specification.
    pub fn contiguous(dtype: DType, shape: impl Into<Vec<usize>>) -> Result<Self, ContractError> {
        let shape = shape.into();
        validate_shape(&shape)?;

        let mut strides = vec![1_usize; shape.len()];
        for index in (0..shape.len().saturating_sub(1)).rev() {
            strides[index] = strides[index + 1]
                .checked_mul(shape[index + 1])
                .ok_or(ContractError::ElementCountOverflow)?;
        }

        Ok(Self {
            dtype,
            shape,
            strides,
        })
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }
}

/// Operator contract or host-buffer validation failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContractError {
    EmptyShape,
    ZeroDimension,
    ElementCountOverflow,
    InvalidEpsilon(f32),
    InvalidGroupSize(usize),
    WidthNotDivisible {
        width: usize,
        group_size: usize,
    },
    InvalidRotaryDimension {
        rotary_dim: usize,
        head_size: usize,
    },
    PositionOutOfBounds {
        token: usize,
        position: i64,
        max_position: usize,
    },
    SlotOutOfBounds {
        token: usize,
        slot: i64,
        slot_capacity: usize,
    },
    DuplicateSlot {
        first_token: usize,
        second_token: usize,
        slot: usize,
    },
    TokenIdOutOfBounds {
        row: usize,
        token_id: i64,
        vocab_size: usize,
    },
    TokenPenaltyWorkspaceTooSmall {
        required: usize,
        actual: usize,
    },
    TopKLogprobsOutOfRange {
        top_k: usize,
        maximum: usize,
    },
    TopKFilterOutOfRange {
        row: usize,
        top_k: i32,
        vocab_size: usize,
    },
    InvalidLogit {
        row: usize,
        column: usize,
        value: f32,
    },
    NoFiniteLogit {
        row: usize,
    },
    TargetTokenIdOutOfI32Range {
        token: usize,
        token_id: i64,
    },
    DraftTokenCapacityExceeded {
        draft_tokens: usize,
        capacity: usize,
    },
    InvalidCumulativeDraftLength {
        request: usize,
        previous: i32,
        current: i32,
        draft_tokens: usize,
        max_draft_tokens: usize,
    },
    FinalCumulativeDraftLengthMismatch {
        expected: usize,
        actual: i32,
    },
    InvalidProbability {
        parameter: &'static str,
        row: usize,
        value: f32,
    },
    InvalidCategoricalProbability {
        row: usize,
        column: usize,
        value: f32,
    },
    NoPositiveCategoricalProbability {
        row: usize,
    },
    CategoricalProbabilitySumOutOfRange {
        row: usize,
        sum: f64,
        tolerance: f64,
    },
    InvalidRngState {
        row: usize,
        component: &'static str,
        value: i64,
    },
    RngCounterExhausted {
        row: usize,
    },
    InvalidTemperature {
        row: usize,
        value: f32,
    },
    InvalidMaskValue {
        index: usize,
        value: u8,
    },
    SparseRowOutOfBounds {
        parameter: &'static str,
        entry: usize,
        row_id: i32,
        rows: usize,
    },
    SparseTokenOutOfBounds {
        parameter: &'static str,
        entry: usize,
        token_id: i32,
        vocab_size: usize,
    },
    InvalidLogitBias {
        entry: usize,
        value: f32,
    },
    DuplicateLogitBias {
        first_entry: usize,
        second_entry: usize,
        row_id: i32,
        token_id: i32,
    },
    MoeTopKOutOfRange {
        top_k: usize,
        num_experts: usize,
    },
    MoeExpertCountOutOfRange {
        num_experts: usize,
    },
    MoeLocalExpertCountOutOfRange {
        num_local_experts: usize,
        num_experts: usize,
    },
    MoeExpertMapRequired {
        num_local_experts: usize,
        num_experts: usize,
    },
    MoeExpertMapPresenceMismatch {
        expected: bool,
    },
    MoeAssignmentCountOutOfRange {
        assignments: usize,
    },
    MoeExpertIdOutOfRange {
        assignment: usize,
        expert_id: i32,
        num_experts: usize,
    },
    MoeExpertMapOutOfRange {
        global_expert: usize,
        local_expert: i32,
        num_local_experts: usize,
    },
    MoeRoutingWeightNotFinite {
        assignment: usize,
        weight: f32,
    },
    MoeExpertOffsetOutOfRange {
        expert: usize,
        previous: i64,
        current: i64,
        assignments: usize,
    },
    MoePermutationIndexOutOfRange {
        assignment: usize,
        permuted_row: i32,
        assignments: usize,
    },
    MoeDuplicatePermutationIndex {
        first_assignment: usize,
        second_assignment: usize,
        permuted_row: usize,
    },
    InvalidScale(f32),
    HeadCountNotDivisible {
        query_heads: usize,
        kv_heads: usize,
    },
    SequenceLengthOutOfBounds {
        sequence: usize,
        length: i64,
        capacity: usize,
    },
    MaxSequenceLengthOutOfBounds {
        length: usize,
        capacity: usize,
    },
    BlockIdOutOfBounds {
        sequence: usize,
        logical_block: usize,
        block_id: i64,
        num_blocks: usize,
    },
    LengthMismatch {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    UnsupportedDType(DType),
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => write!(formatter, "tensor shape must not be empty"),
            Self::ZeroDimension => write!(formatter, "tensor dimensions must be non-zero"),
            Self::ElementCountOverflow => write!(formatter, "tensor element count overflowed"),
            Self::InvalidEpsilon(value) => write!(
                formatter,
                "RMSNorm epsilon must be finite and positive, got {value}"
            ),
            Self::InvalidGroupSize(value) => write!(
                formatter,
                "FP8 block group size must be 64 or 128, got {value}"
            ),
            Self::WidthNotDivisible { width, group_size } => write!(
                formatter,
                "output width {width} is not divisible by FP8 group size {group_size}"
            ),
            Self::InvalidRotaryDimension {
                rotary_dim,
                head_size,
            } => write!(
                formatter,
                "rotary dimension must be non-zero, even, and no larger than head size; got rotary_dim={rotary_dim}, head_size={head_size}"
            ),
            Self::PositionOutOfBounds {
                token,
                position,
                max_position,
            } => write!(
                formatter,
                "position {position} for token {token} is outside [0, {max_position})"
            ),
            Self::SlotOutOfBounds {
                token,
                slot,
                slot_capacity,
            } => write!(
                formatter,
                "cache slot {slot} for token {token} is outside [0, {slot_capacity})"
            ),
            Self::DuplicateSlot {
                first_token,
                second_token,
                slot,
            } => write!(
                formatter,
                "cache slot {slot} is assigned to both token {first_token} and token {second_token}"
            ),
            Self::TokenIdOutOfBounds {
                row,
                token_id,
                vocab_size,
            } => write!(
                formatter,
                "selected token ID {token_id} for row {row} is outside [0, {vocab_size})"
            ),
            Self::TokenPenaltyWorkspaceTooSmall { required, actual } => write!(
                formatter,
                "token-penalty workspace requires at least {required} hash slots per row, got {actual}"
            ),
            Self::TopKLogprobsOutOfRange { top_k, maximum } => write!(
                formatter,
                "top-k logprobs must request between 1 and {maximum} entries, got {top_k}"
            ),
            Self::TopKFilterOutOfRange {
                row,
                top_k,
                vocab_size,
            } => write!(
                formatter,
                "top-k filter value {top_k} for row {row} is outside [1, {vocab_size}]"
            ),
            Self::InvalidLogit { row, column, value } => write!(
                formatter,
                "logit at row {row}, column {column} must be finite or negative infinity, got {value}"
            ),
            Self::NoFiniteLogit { row } => {
                write!(formatter, "logits row {row} must contain a finite value")
            }
            Self::TargetTokenIdOutOfI32Range { token, token_id } => write!(
                formatter,
                "target token ID {token_id} at flattened position {token} does not fit int32"
            ),
            Self::DraftTokenCapacityExceeded {
                draft_tokens,
                capacity,
            } => write!(
                formatter,
                "flattened draft token count {draft_tokens} exceeds ragged capacity {capacity}"
            ),
            Self::InvalidCumulativeDraftLength {
                request,
                previous,
                current,
                draft_tokens,
                max_draft_tokens,
            } => write!(
                formatter,
                "cumulative draft boundary {current} for request {request} is invalid after {previous}; total={draft_tokens}, per-request maximum={max_draft_tokens}"
            ),
            Self::FinalCumulativeDraftLengthMismatch { expected, actual } => write!(
                formatter,
                "final cumulative draft boundary must equal {expected}, got {actual}"
            ),
            Self::InvalidProbability {
                parameter,
                row,
                value,
            } => write!(
                formatter,
                "{parameter} for row {row} must be finite and in [0, 1], got {value}"
            ),
            Self::InvalidCategoricalProbability { row, column, value } => write!(
                formatter,
                "categorical probability at row {row}, column {column} must be finite and non-negative, got {value}"
            ),
            Self::NoPositiveCategoricalProbability { row } => write!(
                formatter,
                "categorical probability row {row} must contain a positive value"
            ),
            Self::CategoricalProbabilitySumOutOfRange {
                row,
                sum,
                tolerance,
            } => write!(
                formatter,
                "categorical probability row {row} must sum to 1 within {tolerance}, got {sum}"
            ),
            Self::InvalidRngState {
                row,
                component,
                value,
            } => write!(
                formatter,
                "categorical RNG {component} for row {row} must be non-negative, got {value}"
            ),
            Self::RngCounterExhausted { row } => write!(
                formatter,
                "categorical RNG counter for row {row} cannot advance beyond int64::MAX"
            ),
            Self::InvalidTemperature { row, value } => write!(
                formatter,
                "temperature for row {row} must be finite and non-negative, got {value}"
            ),
            Self::InvalidMaskValue { index, value } => write!(
                formatter,
                "blocked-mask value at flattened index {index} must be 0 or 1, got {value}"
            ),
            Self::SparseRowOutOfBounds {
                parameter,
                entry,
                row_id,
                rows,
            } => write!(
                formatter,
                "{parameter} row ID {row_id} at entry {entry} is outside [0, {rows})"
            ),
            Self::SparseTokenOutOfBounds {
                parameter,
                entry,
                token_id,
                vocab_size,
            } => write!(
                formatter,
                "{parameter} token ID {token_id} at entry {entry} is outside [0, {vocab_size})"
            ),
            Self::InvalidLogitBias { entry, value } => write!(
                formatter,
                "logit bias at entry {entry} must be finite, got {value}"
            ),
            Self::DuplicateLogitBias {
                first_entry,
                second_entry,
                row_id,
                token_id,
            } => write!(
                formatter,
                "logit bias entries {first_entry} and {second_entry} both target row {row_id}, token {token_id}"
            ),
            Self::MoeTopKOutOfRange { top_k, num_experts } => write!(
                formatter,
                "MoE top-k must be in [1, {num_experts}], got {top_k}"
            ),
            Self::MoeExpertCountOutOfRange { num_experts } => write!(
                formatter,
                "MoE global expert count {num_experts} does not fit int32 expert IDs"
            ),
            Self::MoeLocalExpertCountOutOfRange {
                num_local_experts,
                num_experts,
            } => write!(
                formatter,
                "MoE local expert count {num_local_experts} exceeds global expert count {num_experts}"
            ),
            Self::MoeExpertMapRequired {
                num_local_experts,
                num_experts,
            } => write!(
                formatter,
                "MoE expert map is required when local expert count {num_local_experts} differs from global expert count {num_experts}"
            ),
            Self::MoeExpertMapPresenceMismatch { expected } => write!(
                formatter,
                "MoE expert-map presence does not match the contract; expected={expected}"
            ),
            Self::MoeAssignmentCountOutOfRange { assignments } => write!(
                formatter,
                "MoE flattened assignment count {assignments} does not fit int32"
            ),
            Self::MoeExpertIdOutOfRange {
                assignment,
                expert_id,
                num_experts,
            } => write!(
                formatter,
                "MoE expert ID {expert_id} at assignment {assignment} is outside [0, {num_experts})"
            ),
            Self::MoeExpertMapOutOfRange {
                global_expert,
                local_expert,
                num_local_experts,
            } => write!(
                formatter,
                "MoE expert-map value {local_expert} for global expert {global_expert} is outside [-1, {num_local_experts})"
            ),
            Self::MoeRoutingWeightNotFinite { assignment, weight } => write!(
                formatter,
                "MoE routing weight {weight} at assignment {assignment} must be finite"
            ),
            Self::MoeExpertOffsetOutOfRange {
                expert,
                previous,
                current,
                assignments,
            } => write!(
                formatter,
                "MoE expert offset {current} at boundary {expert} is invalid after {previous}; assignment capacity is {assignments}"
            ),
            Self::MoePermutationIndexOutOfRange {
                assignment,
                permuted_row,
                assignments,
            } => write!(
                formatter,
                "MoE inverse permutation row {permuted_row} at assignment {assignment} is outside [0, {assignments})"
            ),
            Self::MoeDuplicatePermutationIndex {
                first_assignment,
                second_assignment,
                permuted_row,
            } => write!(
                formatter,
                "MoE assignments {first_assignment} and {second_assignment} both map to permuted row {permuted_row}"
            ),
            Self::InvalidScale(value) => write!(
                formatter,
                "attention scale must be finite and positive, got {value}"
            ),
            Self::HeadCountNotDivisible {
                query_heads,
                kv_heads,
            } => write!(
                formatter,
                "query head count {query_heads} must be divisible by KV head count {kv_heads}"
            ),
            Self::SequenceLengthOutOfBounds {
                sequence,
                length,
                capacity,
            } => write!(
                formatter,
                "sequence length {length} for sequence {sequence} is outside [1, {capacity}]"
            ),
            Self::MaxSequenceLengthOutOfBounds { length, capacity } => write!(
                formatter,
                "maximum sequence length {length} exceeds block-table capacity {capacity}"
            ),
            Self::BlockIdOutOfBounds {
                sequence,
                logical_block,
                block_id,
                num_blocks,
            } => write!(
                formatter,
                "physical block ID {block_id} for sequence {sequence}, logical block {logical_block} is outside [0, {num_blocks})"
            ),
            Self::LengthMismatch {
                buffer,
                expected,
                actual,
            } => write!(
                formatter,
                "{buffer} length mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedDType(dtype) => {
                write!(formatter, "CPU reference does not support dtype {dtype:?}")
            }
        }
    }
}

impl std::error::Error for ContractError {}

fn validate_shape(shape: &[usize]) -> Result<(), ContractError> {
    if shape.is_empty() {
        return Err(ContractError::EmptyShape);
    }
    if shape.contains(&0) {
        return Err(ContractError::ZeroDimension);
    }
    shape
        .iter()
        .try_fold(1_usize, |elements, &dimension| {
            elements.checked_mul(dimension)
        })
        .ok_or(ContractError::ElementCountOverflow)?;
    Ok(())
}

pub(crate) fn require_len(
    buffer: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ContractError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ContractError::LengthMismatch {
            buffer,
            expected,
            actual,
        })
    }
}
