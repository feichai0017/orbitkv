use std::collections::BTreeSet;

use thiserror::Error;

/// Stable server identity for one generation request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RequestId(pub u64);

/// One scheduler microbatch. It contains semantic intent only and cannot name
/// physical KV pages, generations, slots, or executor buffers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchIntent {
    pub requests: Box<[RequestIntent]>,
}

/// Tokens newly submitted for one request and the logical boundary after they
/// are appended. Prefill submits several tokens; decode normally submits one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestIntent {
    pub request_id: RequestId,
    pub input_tokens: Box<[u32]>,
    pub target_boundary: u64,
    pub sampling: SamplingIntent,
}

/// Generation semantics supported by the local engine boundary.
///
/// Sampling is deliberately explicit here: a frontend adapter must reject
/// parameters that the engine cannot honor instead of silently changing them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingIntent {
    pub max_output_tokens: u32,
    pub stop_token_ids: Box<[u32]>,
}

impl SamplingIntent {
    /// Creates the currently supported deterministic sampling contract.
    #[must_use]
    pub fn greedy(max_output_tokens: u32, stop_token_ids: impl Into<Box<[u32]>>) -> Self {
        Self {
            max_output_tokens,
            stop_token_ids: stop_token_ids.into(),
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BatchIntentError {
    #[error("batch must contain at least one request")]
    Empty,
    #[error("request ids in one batch must be unique")]
    DuplicateRequest,
    #[error("request input must be nonempty and fit its target boundary")]
    InvalidBoundary,
    #[error("request must permit at least one output token")]
    InvalidSampling,
}

impl BatchIntent {
    /// Validates and constructs one scheduler microbatch.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty batch, duplicate request ids, or a request
    /// whose logical boundary cannot contain its submitted input tokens.
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
            if request.sampling.max_output_tokens == 0 {
                return Err(BatchIntentError::InvalidSampling);
            }
        }
        Ok(Self { requests })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_request_ids() {
        let request = RequestIntent {
            request_id: RequestId(7),
            input_tokens: vec![1].into_boxed_slice(),
            target_boundary: 1,
            sampling: SamplingIntent::greedy(1, []),
        };
        assert_eq!(
            BatchIntent::new(vec![request.clone(), request].into_boxed_slice()),
            Err(BatchIntentError::DuplicateRequest)
        );
    }

    #[test]
    fn accepts_prefill_and_decode_without_physical_state() {
        let batch = BatchIntent::new(
            vec![
                RequestIntent {
                    request_id: RequestId(1),
                    input_tokens: vec![10, 11, 12].into_boxed_slice(),
                    target_boundary: 3,
                    sampling: SamplingIntent::greedy(8, [2]),
                },
                RequestIntent {
                    request_id: RequestId(2),
                    input_tokens: vec![20].into_boxed_slice(),
                    target_boundary: 41,
                    sampling: SamplingIntent::greedy(1, []),
                },
            ]
            .into_boxed_slice(),
        )
        .unwrap();
        assert_eq!(batch.requests.len(), 2);
    }

    #[test]
    fn rejects_zero_output_budget() {
        assert_eq!(
            BatchIntent::new(
                vec![RequestIntent {
                    request_id: RequestId(1),
                    input_tokens: vec![10].into_boxed_slice(),
                    target_boundary: 1,
                    sampling: SamplingIntent::greedy(0, []),
                }]
                .into_boxed_slice(),
            ),
            Err(BatchIntentError::InvalidSampling)
        );
    }
}
