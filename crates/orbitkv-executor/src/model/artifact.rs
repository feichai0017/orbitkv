use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{DecoderCompileConfig, DecoderConfig, DecoderError, DecoderWeightFeatures};
use crate::{ExecutorArena, ExecutorPlan, FixedStateArenaRegistration};
use luminal::graph::SelectedSchedule;

const DECODER_ARTIFACT_SCHEMA: u32 = 1;

/// Portable graph-selection artifact for one native decoder configuration.
///
/// It contains no weights, device pointers, or KV contents. Its identity binds
/// the selected schedule to model semantics and persistent-state geometry.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderArtifact {
    pub(super) schema: u32,
    pub(super) identity: String,
    pub(super) schedule: SelectedSchedule,
}

#[derive(Serialize)]
struct DecoderArtifactIdentity<'a> {
    manifest_fingerprint: &'a str,
    compiler_facts_digest: &'a str,
    page_tokens: u32,
    decoder: &'a DecoderConfig,
    weights: DecoderWeightFeatures,
    arenas: Vec<DecoderArtifactArena>,
    fixed_states: Vec<DecoderArtifactFixedState>,
    compile: DecoderCompileConfig,
}

#[derive(Serialize)]
struct DecoderArtifactArena {
    class_id: u16,
    backend_base_index: u64,
    page_count: u32,
}

#[derive(Serialize)]
struct DecoderArtifactFixedState {
    state_id: u16,
    slot_count: u32,
    slot_bytes: u64,
}

impl DecoderArtifact {
    /// Serializes the artifact as compact JSON bytes.
    ///
    /// # Errors
    ///
    /// Returns serialization failures without emitting partial data.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DecoderError> {
        serde_json::to_vec(self).map_err(DecoderError::from)
    }

    /// Parses a decoder artifact and rejects unknown schemas.
    ///
    /// # Errors
    ///
    /// Rejects malformed JSON or an unknown artifact schema.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecoderError> {
        let artifact = serde_json::from_slice::<Self>(bytes)?;
        if artifact.schema != DECODER_ARTIFACT_SCHEMA {
            return Err(DecoderError::Artifact(format!(
                "schema {} != {DECODER_ARTIFACT_SCHEMA}",
                artifact.schema
            )));
        }
        Ok(artifact)
    }
}

pub(super) fn new_artifact(identity: String, schedule: SelectedSchedule) -> DecoderArtifact {
    DecoderArtifact {
        schema: DECODER_ARTIFACT_SCHEMA,
        identity,
        schedule,
    }
}

pub(super) fn decoder_artifact_identity(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    fixed_states: &[FixedStateArenaRegistration],
    weights: DecoderWeightFeatures,
    compile: DecoderCompileConfig,
    compiler_facts_digest: &str,
) -> Result<String, DecoderError> {
    let identity = DecoderArtifactIdentity {
        manifest_fingerprint: &plan.manifest_fingerprint,
        compiler_facts_digest,
        page_tokens: plan.page_tokens,
        decoder: config,
        weights,
        arenas: arenas
            .iter()
            .map(|arena| DecoderArtifactArena {
                class_id: arena.class_id,
                backend_base_index: arena.backend_base_index,
                page_count: arena.page_count,
            })
            .collect(),
        fixed_states: fixed_states
            .iter()
            .map(|state| DecoderArtifactFixedState {
                state_id: state.state_id,
                slot_count: state.slot_count,
                slot_bytes: state.slot_bytes,
            })
            .collect(),
        compile,
    };
    let bytes = serde_json::to_vec(&identity)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
