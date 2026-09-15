//! Decoder loading, artifact persistence and preparation before readiness.

use std::{
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use orbitkv::StatePoolIdentity;
use orbitkv_executor::{
    ExecutorArena, ExecutorPlan,
    model::{
        CompiledDecoder, DecoderArtifact, DecoderCompileConfig, DecoderConfig,
        DecoderPreparationReport, DecoderStorage, DecoderTuningProfile,
    },
};
use serde::Serialize;

use super::{ModelEngine, ModelEngineConfig, ModelEngineError};

/// Evidence published only after all configured startup work succeeds.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct EngineStartupReport {
    pub elapsed: Duration,
    pub preparation: Option<DecoderPreparationReport>,
}

impl ModelEngine {
    /// Startup evidence shared by every handle to this engine.
    #[must_use]
    pub fn startup_report(&self) -> &EngineStartupReport {
        &self.shared.startup_report
    }
}

pub(super) fn initialize_decoder(
    config: &ModelEngineConfig,
    decoder_config: &DecoderConfig,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    fixed_states: &[(u16, StatePoolIdentity)],
    tuning: &DecoderTuningProfile,
) -> Result<(CompiledDecoder, Option<DecoderPreparationReport>), ModelEngineError> {
    let weights = checkpoint_weights(&config.model_directory)?;
    let artifact = config
        .decoder_artifact
        .as_deref()
        .filter(|path| path.exists())
        .map(read_decoder_artifact)
        .transpose()?;
    let page_tokens =
        usize::try_from(config.page_tokens).map_err(|_| ModelEngineError::InvalidConfig)?;
    let maximum_context_pages = config
        .page_counts
        .iter()
        .copied()
        .max()
        .and_then(|pages| usize::try_from(pages).ok())
        .ok_or(ModelEngineError::InvalidConfig)?;
    let (mut decoder, selected_artifact) = CompiledDecoder::compile_or_load_on_device_with_tuning(
        decoder_config,
        plan,
        DecoderStorage::new(arenas, fixed_states),
        config.device_index,
        &weights,
        DecoderCompileConfig {
            output_rows: orbitkv_executor::model::DecoderOutputRows::AllTokens,
            maximum_query_tokens: config.maximum_batch_tokens,
            representative_prefill_tokens: config.representative_prefill_tokens,
            maximum_batch_size: config.maximum_active_requests,
            maximum_context_pages,
            representative_context_pages: config
                .representative_prefill_tokens
                .div_ceil(page_tokens)
                .max(1),
            search_graphs: config.search_graphs,
            search_seed: config.search_seed,
        },
        tuning,
        artifact.as_ref(),
    )
    .map_err(initialization_error)?;
    decoder.set_graph_cache_capacity(config.graph_cache_capacity);
    if let Some(path) = &config.decoder_artifact {
        if artifact.is_some() {
            eprintln!("decoder: loaded schedule artifact {}", path.display());
        } else {
            persist_decoder_artifact(path, &selected_artifact)?;
            eprintln!("decoder: stored schedule artifact {}", path.display());
        }
    }
    for bucket in decoder.cache_update_buckets() {
        eprintln!(
            "decoder: bucket {} persistent KV in-place={}/{} copy-back tensors={} bytes={}",
            bucket.bucket_index,
            bucket.in_place_tensors,
            bucket.tensor_count,
            bucket.copy_back_tensors,
            bucket.copy_back_bytes,
        );
    }
    if !decoder.cache_updates_in_place() {
        return Err(ModelEngineError::Initialization(
            "selected decoder schedule materializes persistent token-KV updates".into(),
        ));
    }
    let preparation = config
        .prepare_execution
        .then(|| decoder.prepare_execution())
        .transpose()
        .map_err(initialization_error)?;
    Ok((decoder, preparation))
}

fn initialization_error(error: impl std::fmt::Display) -> ModelEngineError {
    ModelEngineError::Initialization(error.to_string())
}

fn read_decoder_artifact(path: &Path) -> Result<DecoderArtifact, ModelEngineError> {
    let file = std::fs::File::open(path).map_err(initialization_error)?;
    DecoderArtifact::read_from(BufReader::new(file)).map_err(initialization_error)
}

fn persist_decoder_artifact(
    path: &Path,
    artifact: &DecoderArtifact,
) -> Result<(), ModelEngineError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        artifact
            .write_to(&mut writer)
            .map_err(initialization_error)?;
        writer.flush().map_err(initialization_error)?;
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
    temporary
        .persist_noclobber(path)
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
    Ok(())
}

fn checkpoint_weights(directory: &Path) -> Result<Vec<PathBuf>, ModelEngineError> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(directory)
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?
    {
        let path = entry
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?
            .path();
        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(ModelEngineError::Initialization(
            "checkpoint has no safetensors weights".into(),
        ));
    }
    Ok(files)
}

#[cfg(test)]
#[path = "../../tests/unit/model_engine/startup/mod.rs"]
mod tests;
