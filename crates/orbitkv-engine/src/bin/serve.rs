#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use orbitkv_engine::{ModelEngine, ModelEngineConfig};
use orbitkv_server::{FrontendDtype, HttpFrontendConfig, serve_openai};
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
        Ok(Self {
            engine: ModelEngineConfig {
                model_directory: args.model,
                decoder_artifact: args.decoder_artifact,
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
    let config = ServeConfig::from_parsed(Args::parse())?;
    let engine = Arc::new(ModelEngine::start(config.engine)?);
    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        signal.cancel();
    });
    serve_openai(engine, config.frontend, shutdown).await
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
mod tests {
    use super::*;

    #[test]
    fn parses_complete_server_configuration() {
        let config = ServeConfig::from_args(
            [
                "orbitkv-serve",
                "--model",
                "/models/checkpoint",
                "--page-counts",
                "128,66",
                "--max-model-tokens",
                "1024",
                "--max-active-requests",
                "2",
            ]
            .map(str::to_string),
        )
        .unwrap();
        assert_eq!(config.engine.page_counts, [128, 66]);
        assert_eq!(config.engine.decoder_artifact, None);
        assert_eq!(config.frontend.logical_kv_page_count, 128);
        assert_eq!(config.frontend.served_model_names, ["checkpoint"]);
        assert_eq!(config.frontend.model, "/models/checkpoint");
    }

    #[test]
    fn rejects_missing_or_malformed_required_arguments() {
        assert!(ServeConfig::from_args(["orbitkv-serve"].map(str::to_string)).is_err());
        assert!(
            ServeConfig::from_args(
                [
                    "orbitkv-serve",
                    "--model",
                    "/model",
                    "--page-counts",
                    "1,bad"
                ]
                .map(str::to_string)
            )
            .is_err()
        );
    }

    #[test]
    fn frontend_assets_can_be_separate_from_model_weights() {
        let config = ServeConfig::from_args(
            [
                "orbitkv-serve",
                "--model",
                "/models/weights",
                "--frontend-model",
                "/models/tokenizer",
                "--served-model",
                "public-model",
                "--page-counts",
                "128,66",
            ]
            .map(str::to_string),
        )
        .unwrap();
        assert_eq!(
            config.engine.model_directory,
            PathBuf::from("/models/weights")
        );
        assert_eq!(config.frontend.model, "/models/tokenizer");
        assert_eq!(config.frontend.served_model_names, ["public-model"]);
    }

    #[test]
    fn decoder_artifact_path_is_forwarded_to_the_engine() {
        let config = ServeConfig::from_args(
            [
                "orbitkv-serve",
                "--model",
                "/models/weights",
                "--decoder-artifact",
                "/artifacts/decoder.json",
                "--page-counts",
                "128,66",
            ]
            .map(str::to_string),
        )
        .unwrap();
        assert_eq!(
            config.engine.decoder_artifact,
            Some(PathBuf::from("/artifacts/decoder.json"))
        );
    }
}
