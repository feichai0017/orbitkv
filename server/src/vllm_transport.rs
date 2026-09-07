//! Local transport that lets the vLLM Rust HTTP frontend drive one `OrbitKV`
//! engine without importing another scheduler or KV allocator.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::encode_msgpack;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
    UtilityCallOutput,
};
use vllm_engine_core_client::protocol::request::{EngineCoreRequest, EngineCoreRequestType};
use vllm_engine_core_client::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
use vllm_engine_core_client::{TransportMode, protocol};
use vllm_server::{
    ApiServerOptions, ChatTemplateContentFormatOption, Config, CoordinatorMode, CorsConfig,
    GenerationConfigMode, HttpListenerMode, ParserSelection, RendererSelection,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

use crate::{Engine, VllmBridge};

/// Model dtype reported to the reusable HTTP frontend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontendDtype {
    Float16,
    Bfloat16,
    Float32,
}

impl From<FrontendDtype> for ModelDtype {
    fn from(value: FrontendDtype) -> Self {
        match value {
            FrontendDtype::Float16 => Self::Float16,
            FrontendDtype::Bfloat16 => Self::BFloat16,
            FrontendDtype::Float32 => Self::Float32,
        }
    }
}

/// Configuration for the optional OpenAI-compatible HTTP surface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpFrontendConfig {
    pub model: String,
    pub served_model_names: Vec<String>,
    pub host: String,
    pub port: u16,
    pub maximum_model_tokens: u64,
    pub kv_page_tokens: u64,
    /// Logical scheduler blocks guaranteed by the engine's per-class arenas.
    /// This is not the sum of heterogeneous physical class pages.
    pub logical_kv_page_count: u64,
    pub maximum_sequences: u64,
    pub maximum_batch_tokens: u64,
    pub dtype: FrontendDtype,
}

impl HttpFrontendConfig {
    fn validate(&self) -> Result<()> {
        if self.model.is_empty()
            || self.maximum_model_tokens == 0
            || self.kv_page_tokens == 0
            || self.logical_kv_page_count == 0
            || self.maximum_sequences == 0
            || self.maximum_batch_tokens == 0
            || self.served_model_names.iter().any(String::is_empty)
        {
            bail!("invalid HTTP frontend configuration");
        }
        self.logical_kv_page_count
            .checked_mul(self.kv_page_tokens)
            .context("KV capacity overflow")?;
        Ok(())
    }
}

/// Runs vLLM's Rust `OpenAI` HTTP/tokenizer/chat frontend against one local
/// `OrbitKV` engine.
///
/// # Errors
///
/// Returns startup, transport, local-engine, or frontend failures.
pub async fn serve_openai<E>(
    engine: Arc<E>,
    config: HttpFrontendConfig,
    shutdown: CancellationToken,
) -> Result<()>
where
    E: Engine + 'static,
{
    config.validate()?;
    let namespace = tempfile::Builder::new().prefix("orbitkv-http-").tempdir()?;
    let input_address = ipc_endpoint(&namespace, "input.sock");
    let output_address = ipc_endpoint(&namespace, "output.sock");
    let server_shutdown = shutdown.child_token();
    let bridge_shutdown = shutdown.child_token();
    let mut bridge = tokio::spawn(run_bridge(
        engine,
        input_address.clone(),
        output_address.clone(),
        config.clone(),
        bridge_shutdown.clone(),
    ));

    let server_config = server_config(&config, input_address, output_address);
    let server = vllm_server::serve(server_config, server_shutdown.clone());
    tokio::pin!(server);
    let (server_result, bridge_result) = tokio::select! {
        server_result = &mut server => {
            bridge_shutdown.cancel();
            let bridge_result = bridge.await.context("frontend bridge task panicked")?;
            (server_result, bridge_result)
        }
        joined = &mut bridge => {
            server_shutdown.cancel();
            let bridge_result = joined.context("frontend bridge task panicked")?;
            let server_result = server.await;
            (server_result, bridge_result)
        }
    };
    match (server_result, bridge_result) {
        (Err(server), _) => Err(server.context("OpenAI HTTP frontend failed")),
        (Ok(()), Err(bridge)) => Err(bridge.context("local frontend bridge failed")),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn server_config(
    config: &HttpFrontendConfig,
    input_address: String,
    output_address: String,
) -> Config {
    Config {
        transport_mode: TransportMode::Bootstrapped {
            input_address,
            output_address,
            engine_start_index: 0,
            engine_count: 1,
            data_parallel_size: 1,
            ready_timeout: Duration::from_secs(30),
        },
        coordinator_mode: CoordinatorMode::None,
        model: config.model.clone(),
        generation_config: GenerationConfigMode::Vllm,
        served_model_name: config.served_model_names.clone(),
        listener_mode: HttpListenerMode::BindTcp {
            host: config.host.clone(),
            port: config.port,
        },
        tool_call_parser: ParserSelection::None,
        reasoning_parser: ParserSelection::None,
        renderer: RendererSelection::Hf,
        language_model_only: true,
        chat_template: None,
        default_chat_template_kwargs: None,
        limit_mm_per_prompt: HashMap::new(),
        chat_template_content_format: ChatTemplateContentFormatOption::String,
        max_logprobs: Some(0),
        api_server_options: ApiServerOptions::default(),
        cors: CorsConfig::default(),
        tls: None,
        api_keys: Vec::new(),
        disable_log_stats: true,
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(10),
        keep_alive_timeout: Duration::from_secs(5),
        profiler: None,
    }
}

async fn run_bridge<E>(
    engine: Arc<E>,
    input_address: String,
    output_address: String,
    config: HttpFrontendConfig,
    shutdown: CancellationToken,
) -> Result<()>
where
    E: Engine + 'static,
{
    wait_for_ipc_endpoint(&input_address, &shutdown).await?;
    wait_for_ipc_endpoint(&output_address, &shutdown).await?;
    let mut options = SocketOptions::default();
    options.peer_identity(PeerIdentity::try_from(EngineId::from_engine_index(0))?);
    let mut input = DealerSocket::with_options(options);
    input.connect(&input_address).await?;
    input
        .send(ZmqMessage::from(encode_msgpack(&ready_response(&config))?))
        .await?;
    let mut output = PushSocket::new();
    output.connect(&output_address).await?;

    let adapter = Arc::new(VllmBridge::new(engine));
    let (output_tx, mut output_rx) = mpsc::unbounded_channel();
    let mut requests = tokio::task::JoinSet::new();
    let run_result: Result<()> = async {
        loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled() => return Ok(()),
                Some(outputs) = output_rx.recv() => {
                    output.send(ZmqMessage::from(encode_msgpack(&outputs)?)).await?;
                }
                Some(joined) = requests.join_next(), if !requests.is_empty() => {
                    joined.context("frontend request task panicked")??;
                }
                received = input.recv() => {
                    handle_message(
                        received?,
                        Arc::clone(&adapter),
                        &output_tx,
                        &mut requests,
                    )
                    .await?;
                }
            }
        }
    }
    .await;
    let abort_result = adapter
        .abort_all()
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()));
    requests.abort_all();
    while requests.join_next().await.is_some() {}
    match (run_result, abort_result) {
        (Err(error), _) | (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

async fn handle_message<E>(
    message: ZmqMessage,
    bridge: Arc<VllmBridge<E>>,
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    requests: &mut tokio::task::JoinSet<Result<()>>,
) -> Result<()>
where
    E: Engine + 'static,
{
    let frames = message.into_vec();
    if frames.len() != 2 {
        bail!("expected two frontend request frames, got {}", frames.len());
    }
    match EngineCoreRequestType::from_frame(&frames[0]) {
        Some(EngineCoreRequestType::Add) => {
            let request: EngineCoreRequest = protocol::decode_msgpack(&frames[1])?;
            let request_id = request.request_id.clone();
            match bridge.add(request).await {
                Ok(mut submission) => {
                    let output_tx = output_tx.clone();
                    let request_bridge = Arc::clone(&bridge);
                    requests.spawn(async move {
                        let result = async {
                            loop {
                                match submission.next_output().await {
                                    Ok(Some(output)) => send_request_output(&output_tx, output)?,
                                    Ok(None) => break,
                                    Err(error) => {
                                        send_request_error(
                                            &output_tx,
                                            request_id.clone(),
                                            error.to_string(),
                                        )?;
                                        request_bridge
                                            .abort(&request_id)
                                            .await
                                            .map_err(|abort| anyhow::anyhow!(abort.to_string()))?;
                                        break;
                                    }
                                }
                            }
                            Ok(())
                        }
                        .await;
                        if result.is_err() {
                            let _ = request_bridge.abort(&request_id).await;
                        }
                        result
                    });
                }
                Err(error) => send_request_error(output_tx, request_id, error.to_string())?,
            }
        }
        Some(EngineCoreRequestType::Abort) => {
            let ids: Vec<String> = protocol::decode_msgpack(&frames[1])?;
            for id in ids {
                bridge
                    .abort(&id)
                    .await
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            }
        }
        Some(EngineCoreRequestType::Utility) => {
            let request: protocol::utility::EngineCoreUtilityRequest =
                protocol::decode_msgpack(&frames[1])?;
            send_utility_error(output_tx, request.call_id, &request.method_name)?;
        }
        Some(EngineCoreRequestType::StartDpWave) | None => {
            bail!("unsupported frontend request type");
        }
    }
    Ok(())
}

fn send_request_output(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    output: EngineCoreOutput,
) -> Result<()> {
    let finished_requests = output
        .finished()
        .then(|| BTreeSet::from([output.request_id.clone()]));
    send_outputs(
        output_tx,
        RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![output],
            timestamp: now_secs(),
            finished_requests,
            ..Default::default()
        }
        .into(),
    )
}

fn send_request_error(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    request_id: String,
    message: String,
) -> Result<()> {
    send_request_output(
        output_tx,
        EngineCoreOutput {
            request_id,
            finish_reason: Some(EngineCoreFinishReason::Error),
            stop_reason: Some(vllm_engine_core_client::protocol::output::StopReason::Text(
                message,
            )),
            ..EngineCoreOutput::default()
        },
    )
}

fn send_utility_error(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    call_id: protocol::utility::UtilityCallId,
    method_name: &str,
) -> Result<()> {
    send_outputs(
        output_tx,
        UtilityCallOutput {
            engine_index: 0,
            timestamp: now_secs(),
            output: UtilityOutput {
                call_id,
                failure_message: Some(format!("utility method {method_name} is not supported")),
                result: Some(UtilityResultEnvelope::without_type_info(
                    protocol::OpaqueValue::Nil,
                )),
            },
        }
        .into(),
    )
}

fn send_outputs(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    outputs: EngineCoreOutputs,
) -> Result<()> {
    output_tx
        .send(outputs)
        .map_err(|_| anyhow::anyhow!("frontend output channel closed"))
}

fn ready_response(config: &HttpFrontendConfig) -> EngineCoreReadyResponse {
    let kv_tokens = config
        .logical_kv_page_count
        .checked_mul(config.kv_page_tokens);
    let blocks_per_request = config.maximum_model_tokens.div_ceil(config.kv_page_tokens);
    EngineCoreReadyResponse {
        max_model_len: config.maximum_model_tokens,
        num_gpu_blocks: config.logical_kv_page_count,
        block_size: config.kv_page_tokens,
        dp_stats_address: None,
        dtype: config.dtype.into(),
        vllm_version: "orbitkv-frontend".to_string(),
        world_size: 1,
        effective_data_parallel_size: 1,
        tensor_parallel_size: 1,
        pipeline_parallel_size: 1,
        decode_context_parallel_size: 1,
        data_parallel_rank: 0,
        max_num_seqs: config.maximum_sequences,
        max_num_batched_tokens: config.maximum_batch_tokens,
        instance_id: "orbitkv-local".to_string(),
        supports_lora: false,
        max_loras: 0,
        kv_cache_size_tokens: kv_tokens,
        kv_cache_max_concurrency: u32::try_from(config.logical_kv_page_count)
            .ok()
            .zip(u32::try_from(blocks_per_request).ok())
            .map(|(pages, request_pages)| f64::from(pages) / f64::from(request_pages)),
        kv_events_config: None,
        weight_transfer_backend: None,
        enable_sleep_mode: false,
        supports_draft_weight_updates: false,
    }
}

fn ipc_endpoint(namespace: &TempDir, name: &str) -> String {
    format!("ipc://{}", namespace.path().join(name).display())
}

async fn wait_for_ipc_endpoint(address: &str, shutdown: &CancellationToken) -> Result<()> {
    let path = PathBuf::from(
        address
            .strip_prefix("ipc://")
            .context("frontend endpoint must use ipc transport")?,
    );
    loop {
        if path.exists() {
            return Ok(());
        }
        tokio::select! {
            () = shutdown.cancelled() => bail!("shutdown before frontend transport was ready"),
            () = tokio::time::sleep(Duration::from_millis(20)) => {}
        }
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |duration| duration.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Mutex;

    use futures_util::stream;

    use super::*;
    use crate::{
        BatchIntent, EngineAbortFuture, EngineEvent, EngineEventStream, EngineFuture, FinishReason,
        RequestId, TokenOutput,
    };

    #[test]
    fn ready_metadata_reports_only_declared_local_capacity() {
        let config = HttpFrontendConfig {
            model: "/model".to_string(),
            served_model_names: vec!["model".to_string()],
            host: "127.0.0.1".to_string(),
            port: 8000,
            maximum_model_tokens: 4096,
            kv_page_tokens: 16,
            logical_kv_page_count: 128,
            maximum_sequences: 8,
            maximum_batch_tokens: 512,
            dtype: FrontendDtype::Bfloat16,
        };
        let ready = ready_response(&config);
        assert_eq!(ready.kv_cache_size_tokens, Some(2048));
        assert_eq!(ready.kv_cache_max_concurrency, Some(0.5));
        assert!(!ready.supports_lora);
    }

    #[test]
    fn server_config_disables_nonlocal_control_planes() {
        let config = HttpFrontendConfig {
            model: "/model".to_string(),
            served_model_names: Vec::new(),
            host: "127.0.0.1".to_string(),
            port: 8000,
            maximum_model_tokens: 4096,
            kv_page_tokens: 16,
            logical_kv_page_count: 128,
            maximum_sequences: 8,
            maximum_batch_tokens: 512,
            dtype: FrontendDtype::Bfloat16,
        };
        let resolved = server_config(&config, "ipc://input".into(), "ipc://output".into());
        assert_eq!(resolved.coordinator_mode, CoordinatorMode::None);
        assert!(resolved.language_model_only);
        assert_eq!(resolved.tool_call_parser, ParserSelection::None);
        assert_eq!(resolved.reasoning_parser, ParserSelection::None);
        assert_eq!(resolved.max_logprobs, Some(0));
        assert!(resolved.disable_log_stats);
        assert_eq!(resolved.grpc_port, None);
    }

    #[derive(Default)]
    struct HttpTestEngine {
        batches: Mutex<Vec<BatchIntent>>,
    }

    impl Engine for HttpTestEngine {
        type Error = Infallible;

        fn execute(&self, batch: BatchIntent) -> EngineFuture<'_, Self::Error> {
            let request_id = batch.requests[0].request_id;
            let output_token = *batch.requests[0]
                .input_tokens
                .last()
                .expect("validated prompt");
            self.batches.lock().unwrap().push(batch);
            Box::pin(async move {
                Ok(Box::pin(stream::iter([
                    Ok(EngineEvent::BatchStarted {
                        request_ids: vec![request_id].into_boxed_slice(),
                    }),
                    Ok(EngineEvent::Token(TokenOutput {
                        request_id,
                        token_id: output_token,
                    })),
                    Ok(EngineEvent::Finished {
                        request_id,
                        reason: FinishReason::Length,
                    }),
                ])) as EngineEventStream<Self::Error>)
            })
        }

        fn abort(&self, _request_id: RequestId) -> EngineAbortFuture<'_, Self::Error> {
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires ORBITKV_MODEL_DIR containing tokenizer assets"]
    async fn openai_completion_reaches_the_local_engine() {
        let model = std::env::var("ORBITKV_MODEL_DIR")
            .expect("ORBITKV_MODEL_DIR must point to a released checkpoint");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let engine = Arc::new(HttpTestEngine::default());
        let shutdown = CancellationToken::new();
        let server = tokio::spawn(serve_openai(
            Arc::clone(&engine),
            HttpFrontendConfig {
                model,
                served_model_names: vec!["orbitkv-test".to_string()],
                host: "127.0.0.1".to_string(),
                port,
                maximum_model_tokens: 4096,
                kv_page_tokens: 16,
                logical_kv_page_count: 128,
                maximum_sequences: 8,
                maximum_batch_tokens: 512,
                dtype: FrontendDtype::Bfloat16,
            },
            shutdown.clone(),
        ));
        let client = reqwest::Client::new();
        let base = format!("http://127.0.0.1:{port}");
        let mut ready = false;
        for _ in 0..200 {
            if client
                .get(format!("{base}/v1/models"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                ready = true;
                break;
            }
            assert!(!server.is_finished(), "server exited during startup");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(ready, "OpenAI server did not become ready");

        let response = client
            .post(format!("{base}/v1/completions"))
            .json(&serde_json::json!({
                "model": "orbitkv-test",
                "prompt": "hello",
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: serde_json::Value = response.json().await.unwrap();
        assert!(status.is_success(), "{body}");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert!(
            body["choices"][0]["text"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        );
        {
            let batches = engine.batches.lock().unwrap();
            assert_eq!(batches.len(), 1);
            assert_eq!(batches[0].requests[0].sampling.max_output_tokens, 1);
        }

        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(15), server)
            .await
            .expect("server shutdown timed out")
            .expect("server task panicked")
            .expect("server failed");
    }
}
