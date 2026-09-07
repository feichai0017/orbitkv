#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use futures_util::StreamExt;
use orbitkv_engine::{ModelEngine, ModelEngineConfig};
use orbitkv_server::{
    BatchIntent, Engine, EngineEvent, FinishReason, RequestId, RequestIntent, SamplingIntent,
};

const REFERENCE_TOKENS: [u32; 8] = [236_743, 199, 236_820, 34_280, 236_813, 208, 236_820, 34_280];

fn engine_config() -> ModelEngineConfig {
    ModelEngineConfig {
        model_directory: std::env::var_os("ORBITKV_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("ORBITKV_MODEL_DIR must point to a released checkpoint"),
        device_index: 0,
        page_tokens: 16,
        page_counts: vec![64, 33],
        maximum_model_tokens: 1_024,
        maximum_prefill_tokens: 512,
        representative_prefill_tokens: 512,
        search_graphs: std::env::var("ORBITKV_SEARCH_GRAPHS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2),
        search_seed: 7,
    }
}

fn request(
    request_id: u64,
    prompt_tokens: usize,
    output_tokens: u32,
    stop_tokens: &[u32],
) -> BatchIntent {
    BatchIntent::new(
        vec![RequestIntent {
            request_id: RequestId(request_id),
            input_tokens: (0..u32::try_from(prompt_tokens).unwrap())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            target_boundary: u64::try_from(prompt_tokens).unwrap(),
            sampling: SamplingIntent::greedy(output_tokens, stop_tokens.to_vec()),
        }]
        .into_boxed_slice(),
    )
    .unwrap()
}

fn assert_drained(stats: orbitkv::kv_manager::ManagerStats) {
    assert_eq!(stats.active_requests, 0);
    assert_eq!(stats.active_snapshots, 0);
    assert_eq!(stats.reserved_pages, 0);
    assert_eq!(stats.writing_pages, 0);
    assert_eq!(stats.active_pages, 0);
    assert_eq!(stats.retiring_pages, 0);
    assert_eq!(stats.quarantined_pages, 0);
    assert_eq!(stats.pending_reclamations, 0);
    assert_eq!(stats.total_request_page_refs, 0);
    assert_eq!(stats.total_reader_pins, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
async fn released_hybrid_engine_streams_stops_cancels_and_drains() {
    let engine = ModelEngine::start(engine_config()).unwrap();

    let events = engine
        .execute(request(1, 512, 8, &[]))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(matches!(
        events.first(),
        Some(EngineEvent::BatchStarted { .. })
    ));
    let tokens = events
        .iter()
        .filter_map(|event| match event {
            EngineEvent::Token(output) => Some(output.token_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tokens, REFERENCE_TOKENS);
    assert_eq!(
        events.last(),
        Some(&EngineEvent::Finished {
            request_id: RequestId(1),
            reason: FinishReason::Length,
        })
    );
    assert_drained(engine.stats().await.unwrap());

    let stop_events = engine
        .execute(request(2, 16, 8, &[106]))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(stop_events.len(), 2);
    assert_eq!(
        stop_events.last(),
        Some(&EngineEvent::Finished {
            request_id: RequestId(2),
            reason: FinishReason::Stop { token_id: 106 },
        })
    );
    assert_drained(engine.stats().await.unwrap());

    let cancelled_id = RequestId(3);
    let cancelled_stream = engine
        .execute(request(cancelled_id.0, 512, 256, &[]))
        .await
        .unwrap();
    engine.abort(cancelled_id).await.unwrap();
    let cancelled_events = cancelled_stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        cancelled_events.last(),
        Some(&EngineEvent::Finished {
            request_id: cancelled_id,
            reason: FinishReason::Cancelled,
        })
    );
    assert_drained(engine.stats().await.unwrap());
}
