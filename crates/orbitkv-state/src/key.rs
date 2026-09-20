use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{StateComponent, StateFormat};

pub type Digest = [u8; 32];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ContractError {
    #[error("token range end {end} precedes start {start}")]
    InvalidTokenRange { start: u64, end: u64 },
}

/// Half-open logical token interval represented by one state object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenRange {
    pub start: u64,
    pub end: u64,
}

impl TokenRange {
    pub fn new(start: u64, end: u64) -> Result<Self, ContractError> {
        if end < start {
            return Err(ContractError::InvalidTokenRange { start, end });
        }
        Ok(Self { start, end })
    }

    pub fn len(self) -> u64 {
        self.end - self.start
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Stable logical identity of one restorable state component.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StateKey {
    pub content: Digest,
    pub span: TokenRange,
    pub component: StateComponent,
    pub format: StateFormat,
}
