#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use futures_util::StreamExt;
use orbitkv_engine::{ModelEngine, ModelEngineConfig};
use orbitkv_server::{
    BatchIntent, Engine, EngineEvent, FinishReason, RequestId, RequestIntent, SamplingIntent,
};

const REFERENCE_TOKENS: [u32; 8] = [236_743, 199, 236_820, 34_280, 236_813, 208, 236_820, 34_280];

fn engine_config(maximum_active_requests: usize) -> ModelEngineConfig {
    let concurrent_requests = u32::try_from(maximum_active_requests).unwrap();
    ModelEngineConfig {
        model_directory: std::env::var_os("ORBITKV_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("ORBITKV_MODEL_DIR must point to a released checkpoint"),
        device_index: 0,
        page_tokens: 16,
        page_counts: vec![64 * concurrent_requests, 33 * concurrent_requests],
        maximum_model_tokens: 1_024,
        maximum_prefill_tokens: 512,
        maximum_batch_tokens: 512 * maximum_active_requests,
        representative_prefill_tokens: 512,
        maximum_active_requests,
        maximum_queued_requests: 8,
        event_buffer_size: 16,
        batch_wait_timeout: std::time::Duration::from_millis(1),
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
    let engine = ModelEngine::start(engine_config(1)).unwrap();

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
    assert_drained(engine.stats().await.unwrap().manager);

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
    assert_drained(engine.stats().await.unwrap().manager);

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
    assert_drained(engine.stats().await.unwrap().manager);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
async fn released_hybrid_engine_batches_concurrent_prefill_and_decode() {
    let mut config = engine_config(2);
    config.batch_wait_timeout = std::time::Duration::from_millis(10);
    let engine = ModelEngine::start(config).unwrap();
    let (first, second) = tokio::join!(
        engine.execute(request(11, 512, 8, &[])),
        engine.execute(request(12, 512, 8, &[])),
    );
    let first = first.unwrap();
    let second = second.unwrap();
    let (first_events, second_events) =
        tokio::join!(first.collect::<Vec<_>>(), second.collect::<Vec<_>>(),);
    let first_events = first_events
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let second_events = second_events
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(visible_tokens(&first_events), REFERENCE_TOKENS);
    assert_eq!(visible_tokens(&second_events), REFERENCE_TOKENS);
    assert!(matches!(
        first_events.last(),
        Some(EngineEvent::Finished {
            reason: FinishReason::Length,
            ..
        })
    ));
    assert!(matches!(
        second_events.last(),
        Some(EngineEvent::Finished {
            reason: FinishReason::Length,
            ..
        })
    ));

    let stats = engine.stats().await.unwrap();
    assert_eq!(stats.maximum_observed_batch_size, 2);
    assert!(stats.multi_request_dispatches >= 2);
    assert_eq!(stats.admitted_requests, 2);
    assert_eq!(stats.completed_requests, 2);
    assert_eq!(stats.queued_requests, 0);
    assert_eq!(stats.active_requests, 0);
    assert_drained(stats.manager);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
async fn released_hybrid_engine_batches_decode_with_late_prefill() {
    let mut config = engine_config(2);
    config.batch_wait_timeout = std::time::Duration::from_millis(1);
    let engine = ModelEngine::start(config).unwrap();
    let first = engine.execute(request(21, 512, 32, &[])).await.unwrap();
    let first_task = tokio::spawn(async move { first.collect::<Vec<_>>().await });

    loop {
        let stats = engine.stats().await.unwrap();
        if stats.model_dispatches >= 2 && stats.active_requests == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }

    let second = engine.execute(request(22, 16, 4, &[])).await.unwrap();
    let second_events = second
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let first_events = first_task
        .await
        .unwrap()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let first_tokens = visible_tokens(&first_events);
    let second_tokens = visible_tokens(&second_events);
    assert_eq!(first_tokens.len(), 32);
    assert_eq!(&first_tokens[..REFERENCE_TOKENS.len()], &REFERENCE_TOKENS);
    assert_eq!(second_tokens.len(), 4);
    assert_eq!(second_tokens[0], 106);
    let stats = engine.stats().await.unwrap();
    assert!(stats.mixed_phase_dispatches >= 1);
    assert!(stats.multi_request_dispatches >= 1);
    assert_eq!(stats.maximum_observed_batch_size, 2);
    assert_eq!(stats.queued_requests, 0);
    assert_eq!(stats.active_requests, 0);
    assert_eq!(stats.completed_requests, 2);
    assert_drained(stats.manager);
}

fn visible_tokens(events: &[EngineEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|event| match event {
            EngineEvent::Token(output) => Some(output.token_id),
            _ => None,
        })
        .collect()
}
