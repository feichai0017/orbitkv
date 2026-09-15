use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

use super::{
    DecoderCompileConfig, DecoderConfig, DecoderError, DecoderTuningProfile, DecoderWeightFeatures,
};
use crate::{ExecutorArena, ExecutorPlan, FixedStateArenaRegistration};
use orbitkv_compiler::graph::SelectedSchedule;
use orbitkv_cuda::CudaModuleArtifact;
use orbitkv_cuda::environment::CudaExecutionEnvironment;

// A decoder artifact is a complete, target-validated execution program.
const DECODER_ARTIFACT_SCHEMA: u32 = 11;

/// Selected schedule and CUDA module images for one native decoder configuration.
///
/// It contains no weights, device pointers, or KV contents. Its identity binds
/// the selected schedule to model semantics and persistent-state geometry.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderArtifact {
    pub(super) schema: u32,
    pub(super) identity: String,
    pub(super) schedule: SelectedSchedule,
    pub(super) cuda_modules: CudaModuleArtifact,
    pub(super) environment: CudaExecutionEnvironment,
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
    tuning: &'a DecoderTuningProfile,
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
    /// Current serialization schema. Consumers can inspect compatibility
    /// without duplicating a version literal outside the artifact owner.
    pub const SCHEMA_VERSION: u32 = DECODER_ARTIFACT_SCHEMA;

    /// Serializes the artifact as compact JSON bytes.
    ///
    /// # Errors
    ///
    /// Returns serialization failures without emitting partial data.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DecoderError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(DecoderError::from)
    }

    /// Parses a decoder artifact and rejects unknown schemas.
    ///
    /// # Errors
    ///
    /// Rejects malformed JSON or an unknown artifact schema.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecoderError> {
        let artifact = serde_json::from_slice::<Self>(bytes)?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Reads a complete artifact without retaining an additional JSON buffer.
    /// Callers should buffer file input. Schema and image checks are identical
    /// to [`Self::from_bytes`]; device validation still occurs before loading.
    ///
    /// # Errors
    /// Returns I/O, malformed JSON, schema or image integrity errors.
    pub fn read_from(reader: impl Read) -> Result<Self, DecoderError> {
        let artifact = serde_json::from_reader::<_, Self>(reader)?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Writes compact JSON without allocating a second artifact-sized buffer.
    /// The caller owns buffering, flushing and atomic file publication.
    ///
    /// # Errors
    /// Returns schema, serialization or writer errors. Output may be partial
    /// on failure and must not be published as a complete artifact.
    pub fn write_to(&self, writer: impl Write) -> Result<(), DecoderError> {
        self.validate()?;
        serde_json::to_writer(writer, self).map_err(DecoderError::from)
    }

    /// Number of distinct generated CUDA modules in this execution program.
    #[must_use]
    pub fn module_image_count(&self) -> usize {
        self.cuda_modules.image_count()
    }

    pub(super) fn validate(&self) -> Result<(), DecoderError> {
        if self.schema != DECODER_ARTIFACT_SCHEMA {
            return Err(DecoderError::Artifact(format!(
                "schema {} != {DECODER_ARTIFACT_SCHEMA}",
                self.schema
            )));
        }
        Ok(())
    }

    /// Reject environment and image drift before weight loading or preparation.
    pub(super) fn validate_for_device(
        &self,
        context: &std::sync::Arc<orbitkv_cuda::cudarc::driver::CudaContext>,
    ) -> Result<(), DecoderError> {
        self.validate()?;
        self.environment
            .validate_for_device(context)
            .map_err(DecoderError::Artifact)?;
        self.cuda_modules
            .validate_for_device(context)
            .map_err(DecoderError::Artifact)
    }
}

pub(super) fn new_artifact(
    identity: String,
    schedule: SelectedSchedule,
    cuda_modules: CudaModuleArtifact,
    environment: CudaExecutionEnvironment,
) -> DecoderArtifact {
    DecoderArtifact {
        schema: DECODER_ARTIFACT_SCHEMA,
        identity,
        schedule,
        cuda_modules,
        environment,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model/artifact/mod.rs"]
mod tests;

#[cfg(test)]
pub(super) fn decoder_artifact_identity(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    fixed_states: &[FixedStateArenaRegistration],
    weights: DecoderWeightFeatures,
    compile: DecoderCompileConfig,
    compiler_facts_digest: &str,
) -> Result<String, DecoderError> {
    decoder_artifact_identity_with_tuning(
        config,
        plan,
        arenas,
        fixed_states,
        weights,
        compile,
        compiler_facts_digest,
        &DecoderTuningProfile::default(),
    )
}

#[allow(clippy::too_many_arguments)] // Mirrors the public compilation boundary.
pub(super) fn decoder_artifact_identity_with_tuning(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    fixed_states: &[FixedStateArenaRegistration],
    weights: DecoderWeightFeatures,
    compile: DecoderCompileConfig,
    compiler_facts_digest: &str,
    tuning: &DecoderTuningProfile,
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
        tuning,
    };
    let bytes = serde_json::to_vec(&identity)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
