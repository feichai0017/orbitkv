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
    InvalidHeadMapping {
        query_heads: usize,
        kv_heads: usize,
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
            Self::InvalidHeadMapping {
                query_heads,
                kv_heads,
            } => write!(
                formatter,
                "query heads must be divisible by KV heads: query={query_heads}, KV={kv_heads}"
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
