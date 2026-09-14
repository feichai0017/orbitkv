//! Artifact-bound decoder tuning policy.
//!
//! `buckets` maps workload preferences to feasible shape representatives;
//! Representative device metadata is shared with startup in `model::representative`.
//! The public JSON policy stays independent of CUDA allocation details.

use serde::{Deserialize, Serialize};

use super::DecoderError;

mod buckets;

pub(super) use buckets::decoder_compile_options;

// The decoder keeps the one-query-token shape separate because it selects
// single-token kernels. Multi-request decode remains a packed query shape.
const SINGLE_QUERY_TOKEN: usize = 1;
const FIRST_MULTI_TOKEN_QUERY: usize = SINGLE_QUERY_TOKEN + 1;

// Stable tuning defaults, not limits imposed by a model or GPU. Callers may
// choose any nonzero budget that fits their available compilation resources.
const DEFAULT_RETAINED_CANDIDATES: usize = 1;
const DEFAULT_PROFILING_TRIALS: usize = 3;
const DEFAULT_MAXIMUM_BUCKETS: usize = 32;

// The current attention and fixed-state metadata ABI uses i32 indices.
const PROFILE_INDEX_BYTES: usize = std::mem::size_of::<i32>();

/// Dynamic-shape and search policy for one compiled decoder executable.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecoderCompileConfig {
    pub maximum_query_tokens: usize,
    pub representative_prefill_tokens: usize,
    pub maximum_batch_size: usize,
    pub maximum_context_pages: usize,
    pub representative_context_pages: usize,
    pub search_graphs: usize,
    pub search_seed: u64,
}

impl DecoderCompileConfig {
    pub(super) fn validate(self) -> Result<(), DecoderError> {
        // Attention metadata uses signed 32-bit device indices, including CSR offsets.
        let int_bytes = PROFILE_INDEX_BYTES;
        if self.maximum_query_tokens < FIRST_MULTI_TOKEN_QUERY
            || !(FIRST_MULTI_TOKEN_QUERY..=self.maximum_query_tokens)
                .contains(&self.representative_prefill_tokens)
            || self.maximum_batch_size == 0
            || self.maximum_batch_size > self.maximum_query_tokens
            || self.maximum_context_pages == 0
            || !(1..=self.maximum_context_pages).contains(&self.representative_context_pages)
            || self.search_graphs == 0
            || self.maximum_query_tokens > i32::MAX as usize
            || self.maximum_batch_size > i32::MAX as usize
            || self.maximum_context_pages > i32::MAX as usize
            || self.maximum_query_tokens.checked_mul(int_bytes).is_none()
            || self.maximum_context_pages.checked_mul(int_bytes).is_none()
            || self
                .maximum_batch_size
                .checked_add(1)
                .and_then(|rows| rows.checked_mul(int_bytes))
                .is_none()
        {
            return Err(DecoderError::InvalidGeometry("compile buckets"));
        }
        Ok(())
    }
}

/// Optional workload-specific tuning; independent of executable capacity.
/// Empty representative lists preserve the legacy single-batch/two-token-bucket
/// policy. This profile, including experimental candidate flags, is artifact-bound.
/// Representatives are preferences: correlated values can move within their
/// intervals to satisfy packed-request geometry. Synthetic KV pages are private
/// to each request; shared-prefix physical layouts require a separate fixture.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecoderTuningProfile {
    pub batch_sizes: Vec<usize>,
    pub prefill_tokens: Vec<usize>,
    pub context_pages: Vec<usize>,
    /// Candidate graphs retained for deployment-mode measurement in each bucket.
    pub keep_best: usize,
    /// Repeated measurements per candidate; must be nonzero.
    pub trials: usize,
    /// Cooperative search limit; synchronous compiler calls cannot be preempted.
    pub search_time_limit_ms: Option<u64>,
    /// Caller-selected bound on the Cartesian bucket space before feasibility
    /// filtering. This bounds planning work as well as the number of profiles.
    pub maximum_buckets: usize,
    pub enable_shared_fp8_quantization: bool,
}

impl Default for DecoderTuningProfile {
    fn default() -> Self {
        Self {
            batch_sizes: Vec::new(),
            prefill_tokens: Vec::new(),
            context_pages: Vec::new(),
            keep_best: DEFAULT_RETAINED_CANDIDATES,
            trials: DEFAULT_PROFILING_TRIALS,
            search_time_limit_ms: None,
            maximum_buckets: DEFAULT_MAXIMUM_BUCKETS,
            enable_shared_fp8_quantization: false,
        }
    }
}

impl DecoderTuningProfile {
    fn validate(&self, compile: DecoderCompileConfig) -> Result<(), DecoderError> {
        if self.keep_best == 0
            || self.keep_best > compile.search_graphs
            || self.trials == 0
            || self.maximum_buckets == 0
            || self.search_time_limit_ms == Some(0)
        {
            return Err(DecoderError::InvalidGeometry("tuning budget"));
        }
        Ok(())
    }

    /// Parses a workload profile. Capacity-dependent checks run before compilation.
    ///
    /// # Errors
    /// Rejects malformed JSON and unknown profile fields.
    pub fn from_json(bytes: &[u8]) -> Result<Self, DecoderError> {
        serde_json::from_slice(bytes).map_err(DecoderError::from)
    }
}
