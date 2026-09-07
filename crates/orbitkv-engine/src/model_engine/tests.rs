use super::*;
use orbitkv::{KvClassSpec, KvPlanInput, TokenStorageKind, plan::RetentionKind};
use orbitkv_server::SamplingIntent;

fn config(page_counts: Vec<u32>) -> ModelEngineConfig {
    ModelEngineConfig {
        model_directory: PathBuf::from("/unused"),
        decoder_artifact: None,
        device_index: 0,
        page_tokens: 16,
        page_counts,
        maximum_model_tokens: 1_024,
        maximum_prefill_tokens: 512,
        maximum_batch_tokens: 512,
        representative_prefill_tokens: 512,
        maximum_active_requests: 1,
        maximum_queued_requests: 2,
        event_buffer_size: 16,
        batch_wait_timeout: Duration::from_millis(1),
        search_graphs: 2,
        search_seed: 7,
    }
}

fn request(
    request_id: u64,
    input_tokens: &[u32],
    target_boundary: u64,
    output_tokens: u32,
) -> BatchIntent {
    BatchIntent {
        requests: vec![RequestIntent {
            request_id: RequestId(request_id),
            input_tokens: input_tokens.into(),
            target_boundary,
            sampling: SamplingIntent::greedy(output_tokens, []),
        }]
        .into_boxed_slice(),
    }
}

fn hybrid_plan() -> orbitkv::CompiledKvPlan {
    orbitkv::compile_plan(KvPlanInput {
        page_tokens: 16,
        classes: vec![
            KvClassSpec {
                name: "full".into(),
                layers: vec![1],
                retention: RetentionKind::Full,
                bytes_per_token_per_layer: 128,
                window_tokens: None,
                storage: TokenStorageKind::TokenKv,
                components: Vec::new(),
            },
            KvClassSpec {
                name: "sliding".into(),
                layers: vec![0],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(512),
                storage: TokenStorageKind::TokenKv,
                components: Vec::new(),
            },
        ],
    })
    .unwrap()
}

fn engine_with_channel(
    commands: SyncSender<WorkerCommand>,
) -> (ModelEngine, Arc<Mutex<RequestRegistry>>) {
    let registry = Arc::new(Mutex::new(RequestRegistry {
        accepting: true,
        cancellations: BTreeMap::new(),
    }));
    let engine = ModelEngine {
        shared: Arc::new(EngineShared {
            commands,
            registry: Arc::clone(&registry),
            shutdown: Arc::new(AtomicBool::new(false)),
            worker: Mutex::new(None),
            maximum_model_tokens: 32,
            maximum_prefill_tokens: 16,
            maximum_total_requests: 2,
            event_buffer_size: 4,
        }),
    };
    (engine, registry)
}

#[test]
fn rejects_invalid_capacity_configuration_without_starting_a_worker() {
    let mut invalid = config(vec![64, 33]);
    invalid.page_tokens = 0;
    assert_eq!(invalid.validate(), Err(ModelEngineError::InvalidConfig));
}

#[test]
fn persists_decoder_artifact_without_overwriting_an_existing_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("decoder.json");
    let artifact = DecoderArtifact::from_bytes(
        br#"{"schema":1,"identity":"test","schedule":{"dim_buckets":{},"buckets":[]}}"#,
    )
    .unwrap();

    persist_decoder_artifact(&path, &artifact).unwrap();
    let first = std::fs::read(&path).unwrap();
    assert_eq!(first, artifact.to_bytes().unwrap());
    assert!(persist_decoder_artifact(&path, &artifact).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), first);
}

#[test]
fn decoder_artifact_rejects_unknown_schema() {
    let error = DecoderArtifact::from_bytes(
        br#"{"schema":2,"identity":"test","schedule":{"dim_buckets":{},"buckets":[]}}"#,
    )
    .unwrap_err();
    assert!(error.to_string().contains("schema 2 != 1"));
}

#[test]
fn decoder_artifact_read_rejects_oversized_input() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("oversized.json");
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(MAX_DECODER_ARTIFACT_BYTES + 1).unwrap();

    let error = read_decoder_artifact(&path).unwrap_err();
    assert!(error.to_string().contains("decoder artifact exceeds"));
}

#[test]
fn validates_each_compiled_class_page_budget_independently() {
    let plan = hybrid_plan();
    assert_eq!(validate_page_budgets(&plan, &config(vec![64, 33])), Ok(()));
    assert_eq!(
        validate_page_budgets(&plan, &config(vec![64, 32])),
        Err(ModelEngineError::InvalidConfig)
    );

    let mut concurrent = config(vec![128, 66]);
    concurrent.maximum_active_requests = 2;
    concurrent.maximum_batch_tokens = 1_024;
    assert_eq!(validate_page_budgets(&plan, &concurrent), Ok(()));
    concurrent.page_counts[1] = 65;
    assert_eq!(
        validate_page_budgets(&plan, &concurrent),
        Err(ModelEngineError::InvalidConfig)
    );
    assert_eq!(
        validate_page_budgets(&plan, &config(vec![64])),
        Err(ModelEngineError::InvalidConfig)
    );
}

#[test]
fn validates_fresh_prompt_and_model_length_contract() {
    let accepted = validate_batch(request(1, &[10, 11], 2, 3), 4, 2).unwrap();
    assert_eq!(accepted.request_id, RequestId(1));
    assert_eq!(
        validate_batch(request(2, &[10], 2, 1), 4, 2),
        Err(ModelEngineError::ContinuationUnsupported)
    );
    assert_eq!(
        validate_batch(request(3, &[10, 11], 2, 4), 4, 2),
        Err(ModelEngineError::ModelLength)
    );
    assert_eq!(
        validate_batch(request(6, &[10, 11, 12], 3, 1), 4, 2),
        Err(ModelEngineError::ModelLength)
    );
    assert_eq!(
        validate_batch(
            BatchIntent {
                requests: vec![
                    request(4, &[10], 1, 1).requests[0].clone(),
                    request(5, &[11], 1, 1).requests[0].clone(),
                ]
                .into_boxed_slice(),
            },
            4,
            2,
        ),
        Err(ModelEngineError::BatchSize)
    );
}

#[tokio::test]
async fn duplicate_submission_preserves_the_active_cancellation_handle() {
    let (commands, _receiver) = mpsc::sync_channel(2);
    let (engine, registry) = engine_with_channel(commands);
    let active = Arc::new(AtomicBool::new(false));
    registry
        .lock()
        .unwrap()
        .cancellations
        .insert(RequestId(7), Arc::clone(&active));

    let result = engine.execute(request(7, &[10], 1, 1)).await;
    assert!(matches!(result, Err(ModelEngineError::DuplicateRequest)));
    let registry = registry.lock().unwrap();
    assert!(Arc::ptr_eq(
        registry.cancellations.get(&RequestId(7)).unwrap(),
        &active
    ));
}

#[tokio::test]
async fn failed_worker_send_removes_the_request_registry_entry() {
    let (commands, receiver) = mpsc::sync_channel(2);
    drop(receiver);
    let (engine, registry) = engine_with_channel(commands);

    let result = engine.execute(request(8, &[10], 1, 1)).await;
    assert!(matches!(result, Err(ModelEngineError::WorkerUnavailable)));
    let registry = registry.lock().unwrap();
    assert!(!registry.accepting);
    assert!(registry.cancellations.is_empty());
}

#[tokio::test]
async fn bounded_admission_rejects_excess_requests_without_registry_leaks() {
    let (commands, _receiver) = mpsc::sync_channel(2);
    let (engine, registry) = engine_with_channel(commands);
    let _first = engine.execute(request(10, &[1], 1, 1)).await.unwrap();
    let _second = engine.execute(request(11, &[2], 1, 1)).await.unwrap();
    assert!(matches!(
        engine.execute(request(12, &[3], 1, 1)).await,
        Err(ModelEngineError::QueueFull)
    ));
    assert_eq!(registry.lock().unwrap().cancellations.len(), 2);
}

#[test]
fn maps_each_csr_row_to_its_last_sampled_token() {
    assert_eq!(
        sampled_rows(&[10, 11, 20, 30, 31], &[0, 2, 3, 5]).unwrap(),
        vec![11, 20, 31]
    );
    assert!(sampled_rows(&[10], &[0, 2]).is_err());
    assert!(sampled_rows(&[10, 11], &[0, 2, 1]).is_err());
}

#[test]
fn dispatch_batches_decode_before_prefill_with_a_token_budget() {
    let (first, _first_rx) = active_request(1, &[1, 2], None);
    let (decode, _decode_rx) = active_request(2, &[3], Some(30));
    let (second, _second_rx) = active_request(3, &[4, 5, 6], None);
    let active = vec![first, decode, second];
    let dispatch = build_dispatch(&active, 4).unwrap();
    assert_eq!(dispatch.active_indices, vec![1, 0]);
    assert_eq!(
        dispatch.request_ids,
        vec![EngineRequestId(2), EngineRequestId(1)]
    );
    assert_eq!(dispatch.target_boundaries, vec![2, 2]);
    assert_eq!(dispatch.tokens, vec![30, 1, 2]);
    assert_eq!(dispatch.positions, vec![1, 0, 1]);
}

#[test]
fn dispatch_skips_backpressured_requests_without_losing_ready_work() {
    let (blocked, _blocked_rx) = active_request(1, &[1], Some(10));
    for token_id in [8, 9, 10] {
        blocked
            .output
            .try_send(Ok(EngineEvent::Token(TokenOutput {
                request_id: RequestId(1),
                token_id,
            })))
            .unwrap();
    }
    let (ready, _ready_rx) = active_request(2, &[2], Some(20));
    let dispatch = build_dispatch(&[blocked, ready], 2).unwrap();
    assert_eq!(dispatch.active_indices, vec![1]);
    assert_eq!(dispatch.tokens, vec![20]);
}

fn active_request(
    request_id: u64,
    input: &[u32],
    next_token: Option<u32>,
) -> (
    ActiveRequest,
    async_mpsc::Receiver<Result<EngineEvent, ModelEngineError>>,
) {
    let (output, receiver) = async_mpsc::channel(4);
    (
        ActiveRequest {
            request: RequestIntent {
                request_id: RequestId(request_id),
                input_tokens: input.into(),
                target_boundary: input.len() as u64,
                sampling: SamplingIntent::greedy(4, []),
            },
            output,
            cancelled: Arc::new(AtomicBool::new(false)),
            boundary: input.len() as u64,
            next_token,
            generated_tokens: 0,
        },
        receiver,
    )
}
