use crate::DType;
use std::fmt;

/// Operator contract or host-buffer validation failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContractError {
    ZeroDimension,
    ElementCountOverflow,
    InvalidEpsilon(f32),
    UnsupportedHeadDimension {
        expected: usize,
        actual: usize,
    },
    UnsupportedPageSize {
        expected: usize,
        actual: usize,
    },
    InvalidHeadMapping {
        query_heads: usize,
        kv_heads: usize,
    },
    InvalidPartitionCount {
        partitions: usize,
        kv_len: usize,
    },
    InvalidIndptrStart {
        buffer: &'static str,
        actual: i32,
    },
    NonMonotonicIndptr {
        buffer: &'static str,
        request: usize,
        start: i32,
        end: i32,
    },
    EmptyRaggedRequest {
        buffer: &'static str,
        request: usize,
    },
    RaggedQueryLongerThanKv {
        request: usize,
        query_len: usize,
        kv_len: usize,
    },
    InvalidPageIndptrStart {
        actual: i32,
    },
    NonMonotonicPageIndptr {
        request: usize,
        start: i32,
        end: i32,
    },
    EmptyPagedRequest {
        request: usize,
    },
    PageIndexOutOfRange {
        position: usize,
        index: i32,
        max_num_pages: usize,
    },
    InvalidLastPageLength {
        request: usize,
        length: i32,
        page_size: usize,
    },
    InvalidRotaryDimension {
        rotary_dim: usize,
        head_dim: usize,
    },
    InvalidRopeScale(f32),
    InvalidRopeTheta(f32),
    NegativePositionId {
        token: usize,
        position: i32,
    },
    UnsupportedDType(DType),
    LengthMismatch {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "tensor dimensions must be nonzero"),
            Self::ElementCountOverflow => write!(formatter, "tensor element count overflowed"),
            Self::InvalidEpsilon(value) => {
                write!(
                    formatter,
                    "RMSNorm epsilon must be finite and positive, got {value}"
                )
            }
            Self::UnsupportedHeadDimension { expected, actual } => write!(
                formatter,
                "unsupported attention head dimension: expected {expected}, got {actual}"
            ),
            Self::UnsupportedPageSize { expected, actual } => write!(
                formatter,
                "unsupported KV page size: expected {expected}, got {actual}"
            ),
            Self::InvalidHeadMapping {
                query_heads,
                kv_heads,
            } => write!(
                formatter,
                "query heads must be divisible by KV heads: query={query_heads}, KV={kv_heads}"
            ),
            Self::InvalidPartitionCount { partitions, kv_len } => write!(
                formatter,
                "attention partition count must be in 1..={kv_len}, got {partitions}"
            ),
            Self::InvalidIndptrStart { buffer, actual } => {
                write!(formatter, "{buffer} must start at zero, got {actual}")
            }
            Self::NonMonotonicIndptr {
                buffer,
                request,
                start,
                end,
            } => write!(
                formatter,
                "{buffer} must be nondecreasing at request {request}, got {start} then {end}"
            ),
            Self::EmptyRaggedRequest { buffer, request } => {
                write!(formatter, "{buffer} request {request} has zero length")
            }
            Self::RaggedQueryLongerThanKv {
                request,
                query_len,
                kv_len,
            } => write!(
                formatter,
                "ragged prefill request {request} has query length {query_len} greater than KV length {kv_len}"
            ),
            Self::InvalidPageIndptrStart { actual } => {
                write!(formatter, "page indptr must start at zero, got {actual}")
            }
            Self::NonMonotonicPageIndptr {
                request,
                start,
                end,
            } => write!(
                formatter,
                "page indptr must be nondecreasing at request {request}, got {start} then {end}"
            ),
            Self::EmptyPagedRequest { request } => {
                write!(formatter, "paged decode request {request} has no KV pages")
            }
            Self::PageIndexOutOfRange {
                position,
                index,
                max_num_pages,
            } => write!(
                formatter,
                "page index at position {position} must be in 0..{max_num_pages}, got {index}"
            ),
            Self::InvalidLastPageLength {
                request,
                length,
                page_size,
            } => write!(
                formatter,
                "last page length for request {request} must be in 1..={page_size}, got {length}"
            ),
            Self::InvalidRotaryDimension {
                rotary_dim,
                head_dim,
            } => write!(
                formatter,
                "RoPE rotary dimension must be positive, even, and no greater than head dimension \
                 {head_dim}, got {rotary_dim}"
            ),
            Self::InvalidRopeScale(value) => {
                write!(
                    formatter,
                    "RoPE scale must be finite and positive, got {value}"
                )
            }
            Self::InvalidRopeTheta(value) => {
                write!(
                    formatter,
                    "RoPE theta must be finite and greater than one, got {value}"
                )
            }
            Self::NegativePositionId { token, position } => write!(
                formatter,
                "RoPE position ID at token {token} must be nonnegative, got {position}"
            ),
            Self::UnsupportedDType(dtype) => write!(formatter, "unsupported dtype {dtype:?}"),
            Self::LengthMismatch {
                buffer,
                expected,
                actual,
            } => write!(
                formatter,
                "{buffer} length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for ContractError {}

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
