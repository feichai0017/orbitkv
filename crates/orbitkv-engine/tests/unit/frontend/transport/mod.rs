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
