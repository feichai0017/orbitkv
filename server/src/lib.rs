#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use thiserror::Error;

/// Scheduler output. It contains intent only and cannot name physical KV pages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchIntent {
    pub requests: Box<[RequestIntent]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestIntent {
    pub request_id: u64,
    pub input_tokens: Box<[u32]>,
    pub target_boundary: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchOutput {
    pub requests: Box<[RequestOutput]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestOutput {
    pub request_id: u64,
    pub token_id: u32,
    pub finished: bool,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BatchIntentError {
    #[error("batch must contain at least one request")]
    Empty,
    #[error("request ids in one batch must be unique")]
    DuplicateRequest,
    #[error("request input must be nonempty and end at target_boundary")]
    InvalidBoundary,
}

impl BatchIntent {
    /// Validates and constructs one scheduler batch.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty batch, duplicate request ids, or a request
    /// whose target boundary cannot contain its submitted input tokens.
    pub fn new(requests: impl Into<Box<[RequestIntent]>>) -> Result<Self, BatchIntentError> {
        let requests = requests.into();
        if requests.is_empty() {
            return Err(BatchIntentError::Empty);
        }
        let mut seen = BTreeSet::new();
        for request in &requests {
            if !seen.insert(request.request_id) {
                return Err(BatchIntentError::DuplicateRequest);
            }
            if request.input_tokens.is_empty()
                || request.target_boundary
                    < u64::try_from(request.input_tokens.len())
                        .map_err(|_| BatchIntentError::InvalidBoundary)?
            {
                return Err(BatchIntentError::InvalidBoundary);
            }
        }
        Ok(Self { requests })
    }
}

/// The only server-to-runtime boundary. Implementations own scheduling of
/// `OrbitKV` state transactions and Luminal execution; the server never sees a
/// page identifier or mutates a KV table.
pub trait Engine {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Executes one validated scheduling intent.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined engine error without transferring KV
    /// ownership to the server.
    fn execute(&mut self, batch: BatchIntent) -> Result<BatchOutput, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_request_ids() {
        let request = RequestIntent {
            request_id: 7,
            input_tokens: vec![1].into_boxed_slice(),
            target_boundary: 1,
        };
        assert_eq!(
            BatchIntent::new(vec![request.clone(), request].into_boxed_slice()),
            Err(BatchIntentError::DuplicateRequest)
        );
    }

    #[test]
    fn accepts_prefill_and_decode_intents_without_physical_state() {
        let batch = BatchIntent::new(
            vec![
                RequestIntent {
                    request_id: 1,
                    input_tokens: vec![10, 11, 12].into_boxed_slice(),
                    target_boundary: 3,
                },
                RequestIntent {
                    request_id: 2,
                    input_tokens: vec![20].into_boxed_slice(),
                    target_boundary: 41,
                },
            ]
            .into_boxed_slice(),
        )
        .unwrap();
        assert_eq!(batch.requests.len(), 2);
    }
}
