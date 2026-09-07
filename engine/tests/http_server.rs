#![cfg(feature = "server")]
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use orbitkv_engine::{EngineStats, ModelEngine, ModelEngineConfig};
use orbitkv_server::{FrontendDtype, HttpFrontendConfig, serve_openai};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

const EXPECTED_TOKENS: [u64; 4] = [106, 107, 106, 106];

fn engine_config(model_directory: PathBuf) -> ModelEngineConfig {
    ModelEngineConfig {
        model_directory,
        device_index: 0,
        page_tokens: 16,
        page_counts: vec![128, 66],
        maximum_model_tokens: 1_024,
        maximum_prefill_tokens: 512,
        maximum_batch_tokens: 1_024,
        representative_prefill_tokens: 512,
        maximum_active_requests: 2,
        maximum_queued_requests: 8,
        event_buffer_size: 16,
        batch_wait_timeout: Duration::from_millis(10),
        search_graphs: 2,
        search_seed: 7,
    }
}

fn frontend_config(model: &std::path::Path, port: u16) -> HttpFrontendConfig {
    HttpFrontendConfig {
        model: model.to_string_lossy().into_owned(),
        served_model_names: vec!["orbitkv-test".to_string()],
        host: "127.0.0.1".to_string(),
        port,
        maximum_model_tokens: 1_024,
        kv_page_tokens: 16,
        logical_kv_page_count: 128,
        maximum_sequences: 2,
        maximum_batch_tokens: 1_024,
        dtype: FrontendDtype::Bfloat16,
    }
}

fn completion(request_id: &str, output_tokens: u32, stream: bool) -> Value {
    json!({
        "model": "orbitkv-test",
        "request_id": request_id,
        "prompt": (0_u32..16).collect::<Vec<_>>(),
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": true,
        "return_token_ids": true,
        "stream": stream
    })
}

async fn wait_until_ready(client: &reqwest::Client, base: &str) {
    for _ in 0..300 {
        if client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("model-backed HTTP server did not become ready");
}

async fn wait_until_drained(engine: &ModelEngine) -> EngineStats {
    for _ in 0..500 {
        if let Ok(stats) = engine.stats().await
            && stats.queued_requests == 0
            && stats.active_requests == 0
            && stats.manager.active_requests == 0
            && stats.manager.reserved_pages == 0
            && stats.manager.writing_pages == 0
            && stats.manager.active_pages == 0
            && stats.manager.retiring_pages == 0
            && stats.manager.quarantined_pages == 0
            && stats.manager.pending_reclamations == 0
        {
            return stats;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("model engine did not drain");
}

fn response_tokens(body: &Value) -> Vec<u64> {
    body["choices"][0]["token_ids"]
        .as_array()
        .expect("completion token IDs")
        .iter()
        .map(|token| token.as_u64().expect("numeric token ID"))
        .collect()
}

fn stream_tokens(body: &str) -> Vec<u64> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| serde_json::from_str::<Value>(payload).expect("valid SSE JSON"))
        .flat_map(|chunk| {
            chunk["choices"][0]["token_ids"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .map(|token| token.as_u64().expect("numeric token ID"))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires model weights, matching frontend assets, CUDA, and FlashInfer headers"]
async fn openai_http_executes_streams_batches_cancels_and_drains() {
    let model = PathBuf::from(
        std::env::var_os("ORBITKV_MODEL_DIR")
            .expect("ORBITKV_MODEL_DIR must point to a released checkpoint"),
    );
    let frontend_model = PathBuf::from(
        std::env::var_os("ORBITKV_FRONTEND_MODEL_DIR")
            .expect("ORBITKV_FRONTEND_MODEL_DIR must contain matching tokenizer assets"),
    );
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let engine = Arc::new(ModelEngine::start(engine_config(model)).unwrap());
    let shutdown = CancellationToken::new();
    let server = tokio::spawn(serve_openai(
        Arc::clone(&engine),
        frontend_config(&frontend_model, port),
        shutdown.clone(),
    ));
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    wait_until_ready(&client, &base).await;

    let response = client
        .post(format!("{base}/v1/completions"))
        .json(&completion("non-stream", 4, false))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let body: Value = response.json().await.unwrap();
    assert_eq!(response_tokens(&body), EXPECTED_TOKENS);
    assert_eq!(body["choices"][0]["finish_reason"], "length");

    let response = client
        .post(format!("{base}/v1/completions"))
        .json(&completion("stream", 4, true))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let body = response.text().await.unwrap();
    assert!(body.contains("data: [DONE]"));
    assert_eq!(stream_tokens(&body), EXPECTED_TOKENS);

    let first = client
        .post(format!("{base}/v1/completions"))
        .json(&completion("batch-a", 8, false))
        .send();
    let second = client
        .post(format!("{base}/v1/completions"))
        .json(&completion("batch-b", 8, false))
        .send();
    let (first, second) = tokio::join!(first, second);
    let first = first.unwrap();
    let second = second.unwrap();
    assert!(first.status().is_success());
    assert!(second.status().is_success());
    let (first, second) = tokio::join!(first.json::<Value>(), second.json::<Value>());
    let first_tokens = response_tokens(&first.unwrap());
    let second_tokens = response_tokens(&second.unwrap());
    assert_eq!(first_tokens, second_tokens);
    assert_eq!(first_tokens.len(), 8);

    let response = client
        .post(format!("{base}/v1/completions"))
        .json(&completion("cancel", 1_000, true))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let dispatches_before_drop = loop {
        let stats = engine.stats().await.unwrap();
        if stats.active_requests == 1 {
            break stats.model_dispatches;
        }
        tokio::task::yield_now().await;
    };
    drop(response);
    let stats = wait_until_drained(&engine).await;
    assert!(stats.model_dispatches - dispatches_before_drop < 1_000);
    assert_eq!(stats.cancelled_requests, 1);
    assert!(stats.multi_request_dispatches >= 1);

    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(15), server)
        .await
        .expect("server shutdown timed out")
        .expect("server task panicked")
        .expect("server failed");
    let stats = wait_until_drained(&engine).await;
    assert_eq!(stats.manager.total_request_page_refs, 0);
    assert_eq!(stats.manager.total_reader_pins, 0);
}
