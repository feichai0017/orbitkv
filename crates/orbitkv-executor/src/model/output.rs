//! Artifact-bound selection of token rows before final normalization and projection.

use orbitkv_compiler::prelude::GraphTensor;
use serde::{Deserialize, Serialize};

use super::{CompiledDecoder, DecoderError, DecoderStep};

/// Rows projected to vocabulary logits and sampled by a compiled decoder.
/// Selection preserves packed request order and does not affect layer/state updates.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderOutputRows {
    /// Every query token, for prompt log probabilities and numerical diagnostics.
    AllTokens,
    /// The final query token of each nonempty request, for generation.
    LastTokenPerRequest,
}

impl DecoderOutputRows {
    pub(super) fn select(self, hidden: &GraphTensor, query_indptr: &GraphTensor) -> GraphTensor {
        match self {
            Self::AllTokens => *hidden,
            Self::LastTokenPerRequest => {
                // Input validation requires strictly increasing CSR offsets. The
                // terminal offsets therefore identify valid final query tokens.
                let rows = query_indptr.slice(1..) - 1;
                let width = hidden.dims()[1].to_usize().expect("static hidden width");
                orbitkv_ops::gather_rows(*hidden, rows, width)
            }
        }
    }

    fn count(self, step: DecoderStep<'_>) -> usize {
        match self {
            Self::AllTokens => step.tokens.len(),
            Self::LastTokenPerRequest => step.classes[0].attention.query_indptr.len() - 1,
        }
    }
}

impl CompiledDecoder {
    pub(super) fn read_sampled_tokens(
        &self,
        step: DecoderStep<'_>,
    ) -> Result<Box<[u32]>, DecoderError> {
        let rows = self.compile.output_rows.count(step);
        let raw = self.runtime.get_i32(self.decoder.outputs.sampled_tokens);
        if raw.len() != rows {
            return Err(DecoderError::InvalidGeometry("sampled token output"));
        }
        raw.iter()
            .map(|&token| {
                u32::try_from(token)
                    .ok()
                    .filter(|&token| {
                        token < u32::try_from(self.vocabulary_size).unwrap_or(u32::MAX)
                    })
                    .ok_or(DecoderError::InvalidGeometry("sampled token output"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }

    pub(super) fn read_logits(&self, step: DecoderStep<'_>) -> Result<Box<[f32]>, DecoderError> {
        let logits = self.runtime.get_f32(self.decoder.outputs.logits);
        let expected = self
            .compile
            .output_rows
            .count(step)
            .checked_mul(self.vocabulary_size)
            .ok_or(DecoderError::InputCapacity)?;
        if logits.len() != expected || logits.iter().any(|value| !value.is_finite()) {
            return Err(DecoderError::InvalidGeometry("logits output"));
        }
        Ok(logits.into_boxed_slice())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model/output/mod.rs"]
mod tests;
