#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use orbitkv_engine::{FrontendDtype, HttpFrontendConfig, serve_openai};
use orbitkv_engine::{ModelEngine, ModelEngineConfig};
use orbitkv_executor::model::DecoderTuningProfile;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Parser)]
#[command(name = "orbitkv-serve")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Strict compiled decoder schedule. Loads when present; otherwise creates
    /// it atomically after the one-time search.
    #[arg(long)]
    decoder_artifact: Option<PathBuf>,
    /// Maximum simultaneously materialized decoder buckets (positive).
    #[arg(long, default_value_t = orbitkv_executor::model::DEFAULT_GRAPH_CACHE_CAPACITY)]
    graph_cache_capacity: std::num::NonZeroUsize,
    /// Prepare retained artifact representatives before accepting requests.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    prepare_execution: bool,
    /// Artifact-bound workload representatives and bounded search settings, as JSON.
    #[arg(long)]
    tuning_profile: Option<PathBuf>,
    #[arg(long)]
    frontend_model: Option<PathBuf>,
    #[arg(long)]
    served_model: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value_t = 16)]
    page_tokens: u64,
    #[arg(long, value_delimiter = ',', required = true)]
    page_counts: Vec<u32>,
    #[arg(long, default_value_t = 4_096)]
    max_model_tokens: u64,
    #[arg(long, default_value_t = 512)]
    max_prefill_tokens: usize,
    #[arg(long, default_value_t = 1_024)]
    max_batch_tokens: usize,
    #[arg(long, default_value_t = 2)]
    max_active_requests: usize,
    #[arg(long, default_value_t = 64)]
    max_queued_requests: usize,
    #[arg(long, default_value_t = 64)]
    event_buffer_size: usize,
    #[arg(long, default_value_t = 1_000)]
    batch_wait_micros: u64,
    #[arg(long, default_value_t = 2)]
    search_graphs: usize,
    #[arg(long, default_value_t = 7)]
    search_seed: u64,
}

struct ServeConfig {
    engine: ModelEngineConfig,
    frontend: HttpFrontendConfig,
    tuning: DecoderTuningProfile,
}

impl ServeConfig {
    #[cfg(test)]
    fn from_args(args: impl IntoIterator<Item = String>) -> Result<Self> {
        Self::from_parsed(Args::try_parse_from(args)?)
    }

    fn from_parsed(args: Args) -> Result<Self> {
        let public_model = args.served_model.unwrap_or_else(|| {
            args.model
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("orbitkv-model")
                .to_string()
        });
        let frontend_model = args
            .frontend_model
            .as_ref()
            .unwrap_or(&args.model)
            .to_string_lossy()
            .into_owned();
        let logical_kv_page_count = args
            .max_model_tokens
            .checked_add(args.page_tokens.saturating_sub(1))
            .context("logical KV capacity overflow")?
            .checked_div(args.page_tokens)
            .context("--page-tokens must be nonzero")?
            .checked_mul(u64::try_from(args.max_active_requests)?)
            .context("logical KV capacity overflow")?;
        let tuning = args
            .tuning_profile
            .as_deref()
            .map(|path| {
                let bytes = std::fs::read(path)
                    .with_context(|| format!("reading tuning profile {}", path.display()))?;
                DecoderTuningProfile::from_json(&bytes)
                    .with_context(|| format!("parsing tuning profile {}", path.display()))
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            tuning,
            engine: ModelEngineConfig {
                model_directory: args.model,
                decoder_artifact: args.decoder_artifact,
                graph_cache_capacity: args.graph_cache_capacity,
                prepare_execution: args.prepare_execution,
                device_index: args.device,
                page_tokens: args.page_tokens,
                page_counts: args.page_counts,
                maximum_model_tokens: args.max_model_tokens,
                maximum_prefill_tokens: args.max_prefill_tokens,
                maximum_batch_tokens: args.max_batch_tokens,
                representative_prefill_tokens: args.max_prefill_tokens,
                maximum_active_requests: args.max_active_requests,
                maximum_queued_requests: args.max_queued_requests,
                event_buffer_size: args.event_buffer_size,
                batch_wait_timeout: Duration::from_micros(args.batch_wait_micros),
                search_graphs: args.search_graphs,
                search_seed: args.search_seed,
            },
            frontend: HttpFrontendConfig {
                model: frontend_model,
                served_model_names: vec![public_model],
                host: args.host,
                port: args.port,
                maximum_model_tokens: args.max_model_tokens,
                kv_page_tokens: args.page_tokens,
                logical_kv_page_count,
                maximum_sequences: u64::try_from(args.max_active_requests)?,
                maximum_batch_tokens: u64::try_from(args.max_batch_tokens)?,
                dtype: FrontendDtype::Bfloat16,
            },
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let stage_trace = orbitkv_executor::diagnostics::install_stage_trace_from_env()?;
    let config = ServeConfig::from_parsed(Args::parse())?;
    let engine = Arc::new(ModelEngine::start_with_tuning(
        config.engine,
        config.tuning,
    )?);
    eprintln!(
        "ORBITKV_ENGINE_STARTUP {}",
        serde_json::to_string(engine.startup_report())?
    );
    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        signal.cancel();
    });
    let result = serve_openai(Arc::clone(&engine), config.frontend, shutdown).await;
    // Joining the worker can wait for CUDA and request retirement. Keep that
    // blocking work off the async frontend's runtime threads.
    let report = tokio::task::spawn_blocking(move || engine.shutdown())
        .await
        .context("engine shutdown task panicked")??;
    eprintln!(
        "ORBITKV_ENGINE_SHUTDOWN {}",
        serde_json::to_string(&report)?
    );
    if let Some(trace) = stage_trace {
        trace.finish()?;
    }
    result
}

#[cfg(unix)]
async fn wait_for_shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to install SIGTERM handler");
    tokio::select! {
        result = tokio::signal::ctrl_c() => result.expect("failed to install Ctrl-C handler"),
        _ = terminate.recv() => {}
    }
}

#[cfg(not(unix))]
async fn wait_for_shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl-C handler");
}

#[cfg(test)]
#[path = "../../../tests/unit/bin/serve/mod.rs"]
mod tests;
