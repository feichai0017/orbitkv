#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::Instant;

use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaStream, runtime::CudaRuntime};
use orbitkv::{
    CacheSharingPolicy, EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence,
    EngineRequestId, HfRetentionOptions, RuntimeSession, compile_hf_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
};
use orbitkv_executor::{
    AttentionBatch, ExecutorArena, ExecutorPlan, PreparedBatch,
    model::{DecoderConfig, DecoderGraph, DecoderWeightLayout},
};

const PAGE_TOKENS: u64 = 16;
const PAGE_COUNT: u32 = 64;

fn model_directory() -> PathBuf {
    std::env::var_os("ORBITKV_MODEL_DIR")
        .map(PathBuf::from)
        .expect("ORBITKV_MODEL_DIR must point to a local released checkpoint")
}

struct PreparedModelRun {
    session: RuntimeSession,
    executor_plan: ExecutorPlan,
    arena: ExecutorArena,
    prepared: PreparedBatch,
    attention: AttentionBatch,
}

struct RuntimeBatch<'a> {
    tokens: &'a [u32],
    attention: &'a AttentionBatch,
    writes: &'a [u64],
    cache_bytes: usize,
}

fn prepare_model_run(config_bytes: &[u8], prompt_len: usize) -> PreparedModelRun {
    let manifest = compile_hf_runtime_manifest(
        config_bytes,
        HfRetentionOptions {
            page_tokens: PAGE_TOKENS,
            kv_dtype_bytes: 2,
        },
    )
    .unwrap();
    let manager_plan = orbitkv::compile_plan(
        manifest
            .attention_state_plan
            .as_ref()
            .unwrap()
            .token_manager_plan()
            .unwrap(),
    )
    .unwrap();
    let registration = BackendArenaRegistration {
        pool_id: 1,
        class_id: 0,
        backend_domain: 1,
        page_count: PAGE_COUNT,
        reserved: 0,
        backend_base_index: 0,
    };
    let manager = CanonicalKvManager::new(
        &manager_plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 2,
            maximum_prefixes: 1,
            maximum_reclamations: PAGE_COUNT,
            maximum_step_tokens: 64,
        },
        &[registration],
    )
    .unwrap();
    let mut session = RuntimeSession::new(manager, CacheSharingPolicy::SharedPrefix);
    let request_id = EngineRequestId(1);
    session.acquire_requests(&[request_id]).unwrap();
    let source = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: u64::try_from(prompt_len).unwrap(),
        }])
        .unwrap();
    let view = session.prepared_execution_view(source.batch_id).unwrap();
    let arena = ExecutorArena::bind(session.arena_stats()[0], registration).unwrap();
    let executor_plan = ExecutorPlan::compile(&manifest).unwrap();
    let attention = executor_plan.attention_batch(0, &view).unwrap();
    let prepared = executor_plan.lower_prepared(source, &[arena]).unwrap();
    PreparedModelRun {
        session,
        executor_plan,
        arena,
        prepared,
        attention,
    }
}

fn upload_runtime_inputs(
    runtime: &mut CudaRuntime,
    decoder: &DecoderGraph,
    attention: &AttentionBatch,
    prompt: &[u32],
    writes: &[u64],
) {
    runtime.set_data(
        decoder.inputs.token_ids,
        prompt
            .iter()
            .map(|token| i32::try_from(*token).unwrap())
            .collect::<Vec<_>>(),
    );
    runtime.set_data(
        decoder.inputs.positions,
        (0..i32::try_from(prompt.len()).unwrap()).collect::<Vec<_>>(),
    );
    runtime.set_data(
        decoder.inputs.write_slots,
        writes
            .iter()
            .map(|slot| i32::try_from(*slot).unwrap())
            .collect::<Vec<_>>(),
    );
    decoder.inputs.attention.upload(runtime, attention).unwrap();
}

fn initialize_cache(runtime: &mut CudaRuntime, decoder: &DecoderGraph, cache_bytes: usize) {
    for &(key, value) in &decoder.outputs.cache_inputs {
        runtime.set_zeros(key, cache_bytes);
        runtime.set_zeros(value, cache_bytes);
    }
}

fn complete(
    session: &mut RuntimeSession,
    prepared: &PreparedBatch,
    arena: ExecutorArena,
    completion_value: u64,
) {
    let evidence = prepared.execution_evidence_after_success(&[arena]).unwrap();
    let ticket = session.submit_execution(&evidence).unwrap();
    let publication = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 1,
                completion_value,
                confirmed: true,
            },
        )
        .unwrap();
    session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::default(),
        })
        .unwrap();
}

fn prepare_runtime(
    phase: &str,
    graph: &mut Graph,
    decoder: &DecoderGraph,
    stream: std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    batch: &RuntimeBatch<'_>,
) -> CudaRuntime {
    let started = Instant::now();
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(graph, model_dir.join("model.safetensors").to_str().unwrap());
    eprintln!(
        "{phase}: weights loaded after {:.1}s",
        started.elapsed().as_secs_f64()
    );
    initialize_cache(&mut runtime, decoder, batch.cache_bytes);
    upload_runtime_inputs(
        &mut runtime,
        decoder,
        batch.attention,
        batch.tokens,
        batch.writes,
    );
    eprintln!("{phase}: graph compile started");
    let mut runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
    eprintln!(
        "{phase}: graph compiled after {:.1}s",
        started.elapsed().as_secs_f64()
    );
    initialize_cache(&mut runtime, decoder, batch.cache_bytes);
    upload_runtime_inputs(
        &mut runtime,
        decoder,
        batch.attention,
        batch.tokens,
        batch.writes,
    );
    runtime
}

fn greedy_token(logits: &[f32], vocabulary_size: usize) -> u32 {
    u32::try_from(
        logits[logits.len() - vocabulary_size..]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .expect("nonempty vocabulary")
            .0,
    )
    .unwrap()
}

fn execute_and_read_logits(
    phase: &str,
    runtime: &mut CudaRuntime,
    graph: &Graph,
    logits: &GraphTensor,
    expected_values: usize,
) -> Vec<f32> {
    let started = Instant::now();
    runtime.execute(&graph.dyn_map);
    eprintln!(
        "{phase}: executed after {:.3}s",
        started.elapsed().as_secs_f64()
    );
    let values = runtime.get_f32(*logits);
    assert_eq!(values.len(), expected_values);
    assert!(values.iter().all(|value| value.is_finite()));
    values
}

fn prepare_decode_step(
    prepared_run: &mut PreparedModelRun,
    target_boundary: u64,
) -> (AttentionBatch, PreparedBatch) {
    let source = prepared_run
        .session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: EngineRequestId(1),
            target_boundary,
        }])
        .unwrap();
    let view = prepared_run
        .session
        .prepared_execution_view(source.batch_id)
        .unwrap();
    let attention = prepared_run
        .executor_plan
        .attention_batch(0, &view)
        .unwrap();
    let prepared = prepared_run
        .executor_plan
        .lower_prepared(source, &[prepared_run.arena])
        .unwrap();
    (attention, prepared)
}

fn build_graph(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    query_tokens: usize,
    context_pages: usize,
) -> (Graph, DecoderGraph) {
    let mut graph = Graph::default();
    let decoder = DecoderGraph::build(
        &mut graph,
        config,
        DecoderWeightLayout {
            qkv_bias: true,
            qk_norm: false,
        },
        plan,
        usize::try_from(PAGE_COUNT).unwrap(),
    )
    .unwrap();
    graph.set_dim('s', query_tokens);
    graph.set_dim('b', 1);
    graph.set_dim('c', context_pages);
    (graph, decoder)
}

fn transfer_cache(
    source: &mut CudaRuntime,
    source_graph: &DecoderGraph,
    target: &mut CudaRuntime,
    target_graph: &DecoderGraph,
) {
    for (&(source_key, source_value), &(target_key, target_value)) in source_graph
        .outputs
        .cache_updates
        .iter()
        .zip(&target_graph.outputs.cache_inputs)
    {
        let key = source.remove_buffer(source_key);
        let value = source.remove_buffer(source_value);
        target.set_buffer(target_key, key);
        target.set_buffer(target_value, value);
    }
}

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
fn released_decoder_runs_with_orbitkv_owned_pages() {
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let prompt = [1_u32, 2, 3, 4];
    let mut prepared_run = prepare_model_run(&config_bytes, prompt.len());
    let writes = &prepared_run.prepared.steps()[0].classes[0].write_slots;

    let context = luminal_cuda_lite::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let (mut graph, decoder) = build_graph(
        &config,
        &prepared_run.executor_plan,
        prompt.len(),
        prepared_run.attention.page_indices.len(),
    );

    let cache_bytes = usize::try_from(PAGE_COUNT).unwrap()
        * usize::try_from(PAGE_TOKENS).unwrap()
        * config.kv_heads
        * config.head_dim
        * 2;
    let mut runtime = prepare_runtime(
        "prefill",
        &mut graph,
        &decoder,
        stream.clone(),
        &model_dir,
        &RuntimeBatch {
            tokens: &prompt,
            attention: &prepared_run.attention,
            writes,
            cache_bytes,
        },
    );
    let logits = execute_and_read_logits(
        "prefill",
        &mut runtime,
        &graph,
        &decoder.outputs.logits,
        prompt.len() * config.vocabulary_size,
    );
    let next_token = greedy_token(&logits, config.vocabulary_size);
    complete(
        &mut prepared_run.session,
        &prepared_run.prepared,
        prepared_run.arena,
        1,
    );

    let (decode_attention, decode) =
        prepare_decode_step(&mut prepared_run, u64::try_from(prompt.len() + 1).unwrap());
    let (mut decode_graph, decode_decoder) = build_graph(
        &config,
        &prepared_run.executor_plan,
        1,
        decode_attention.page_indices.len(),
    );
    let mut decode_runtime = prepare_runtime(
        "decode",
        &mut decode_graph,
        &decode_decoder,
        stream,
        &model_dir,
        &RuntimeBatch {
            tokens: &[next_token],
            attention: &decode_attention,
            writes: &decode.steps()[0].classes[0].write_slots,
            cache_bytes,
        },
    );
    transfer_cache(&mut runtime, &decoder, &mut decode_runtime, &decode_decoder);
    upload_runtime_inputs(
        &mut decode_runtime,
        &decode_decoder,
        &decode_attention,
        &[next_token],
        &decode.steps()[0].classes[0].write_slots,
    );
    decode_runtime.set_data(
        decode_decoder.inputs.positions,
        vec![i32::try_from(prompt.len()).unwrap()],
    );
    execute_and_read_logits(
        "decode",
        &mut decode_runtime,
        &decode_graph,
        &decode_decoder.outputs.logits,
        config.vocabulary_size,
    );
    complete(&mut prepared_run.session, &decode, prepared_run.arena, 2);
}
