#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use orbitkv::{
    CacheSharingPolicy, EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence,
    EngineRequestId, HfRetentionOptions, RuntimeSession, compile_hf_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
};
use orbitkv_executor::{
    AttentionBatch, ExecutorArena, ExecutorPlan, PreparedBatch,
    model::{
        CompiledDecoder, DecoderCompileConfig, DecoderConfig, DecoderStep, DecoderWeightLayout,
    },
};

const PAGE_TOKENS: u64 = 16;
const PAGE_COUNT: u32 = 64;

fn model_directory() -> PathBuf {
    std::env::var_os("ORBITKV_MODEL_DIR")
        .map(PathBuf::from)
        .expect("ORBITKV_MODEL_DIR must point to a local released checkpoint")
}

fn search_graphs() -> usize {
    std::env::var("ORBITKV_SEARCH_GRAPHS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2)
}

struct PreparedModelRun {
    session: RuntimeSession,
    executor_plan: ExecutorPlan,
    arena: ExecutorArena,
    prepared: PreparedBatch,
    attention: AttentionBatch,
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
            maximum_operations: 4,
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

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn benchmark_decode_dispatch(
    decoder: &mut CompiledDecoder,
    step: DecoderStep<'_>,
    expected_token: u32,
) {
    let iterations = std::env::var("ORBITKV_DECODE_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20usize);
    let mut eager = Vec::with_capacity(iterations);
    let mut replay = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        if iteration.is_multiple_of(2) {
            benchmark_eager_decode(decoder, step, expected_token, &mut eager);
            benchmark_replay_decode(decoder, step, expected_token, &mut replay);
        } else {
            benchmark_replay_decode(decoder, step, expected_token, &mut replay);
            benchmark_eager_decode(decoder, step, expected_token, &mut eager);
        }
    }
    let eager_median = median_duration(&mut eager);
    let replay_median = median_duration(&mut replay);
    eprintln!(
        "matched decode dispatch: iterations={iterations} eager_median_us={:.1} outer_graph_median_us={:.1} ratio={:.3}",
        eager_median.as_secs_f64() * 1e6,
        replay_median.as_secs_f64() * 1e6,
        replay_median.as_secs_f64() / eager_median.as_secs_f64(),
    );
}

fn benchmark_eager_decode(
    decoder: &mut CompiledDecoder,
    step: DecoderStep<'_>,
    expected_token: u32,
    samples: &mut Vec<Duration>,
) {
    let started = Instant::now();
    let output = decoder.execute(step).unwrap();
    samples.push(started.elapsed());
    assert_eq!(output.token_ids.as_ref(), &[expected_token]);
    assert!(decoder.has_captured_decode());
}

fn benchmark_replay_decode(
    decoder: &mut CompiledDecoder,
    step: DecoderStep<'_>,
    expected_token: u32,
    samples: &mut Vec<Duration>,
) {
    let started = Instant::now();
    let output = decoder.replay_decode(step).unwrap();
    samples.push(started.elapsed());
    assert_eq!(output.token_ids.as_ref(), &[expected_token]);
}

fn execute_step(
    phase: &str,
    decoder: &mut CompiledDecoder,
    tokens: &[u32],
    positions: &[u32],
    attention: &AttentionBatch,
    prepared: &PreparedBatch,
) -> orbitkv_executor::model::DecoderDiagnosticOutput {
    let started = Instant::now();
    let logits = decoder
        .execute_with_logits(DecoderStep {
            tokens,
            positions,
            write_slots: &prepared.steps()[0].classes[0].write_slots,
            attention,
        })
        .unwrap();
    eprintln!(
        "{phase}: dispatched precompiled bucket after {:.3}s",
        started.elapsed().as_secs_f64()
    );
    logits
}

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
#[allow(clippy::too_many_lines)]
fn released_decoder_reuses_one_compiled_runtime_and_kv_arena() {
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let prompt = [1_u32, 2, 3, 4];
    let mut prepared_run = prepare_model_run(&config_bytes, prompt.len());
    let context = luminal_cuda_lite::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let compile = DecoderCompileConfig {
        maximum_query_tokens: 8,
        representative_prefill_tokens: prompt.len(),
        maximum_batch_size: 1,
        maximum_context_pages: usize::try_from(PAGE_COUNT).unwrap(),
        representative_context_pages: prepared_run.attention.page_indices.len(),
        search_graphs: search_graphs(),
        search_seed: 7,
    };
    let compile_started = Instant::now();
    let mut decoder = CompiledDecoder::compile(
        &config,
        DecoderWeightLayout {
            qkv_bias: true,
            qk_norm: false,
        },
        &prepared_run.executor_plan,
        usize::try_from(PAGE_COUNT).unwrap(),
        &stream,
        &[model_dir.join("model.safetensors")],
        compile,
    )
    .unwrap();
    eprintln!(
        "decoder: one-time bucket search/compile completed after {:.1}s",
        compile_started.elapsed().as_secs_f64()
    );
    assert_eq!(decoder.compile_config(), compile);
    assert_eq!(decoder.compiled_bucket_count(), 2);
    assert_eq!(decoder.persistent_cache_count(), config.layers * 2);
    let cache_updates_in_place = decoder.cache_updates_in_place();
    eprintln!("decoder: selected KV updates in-place={cache_updates_in_place}");
    assert_eq!(
        decoder
            .cache_bindings(&prepared_run.executor_plan)
            .unwrap()
            .len(),
        config.layers
    );

    let positions = (0..u32::try_from(prompt.len()).unwrap()).collect::<Vec<_>>();
    let output = execute_step(
        "prefill",
        &mut decoder,
        &prompt,
        &positions,
        &prepared_run.attention,
        &prepared_run.prepared,
    );
    assert_eq!(output.logits.len(), prompt.len() * config.vocabulary_size);
    assert_eq!(output.token_ids.len(), prompt.len());
    assert_eq!(decoder.active_bucket_index(), 1);
    let next_token = greedy_token(&output.logits, config.vocabulary_size);
    assert_eq!(output.token_ids.last().copied(), Some(next_token));
    complete(
        &mut prepared_run.session,
        &prepared_run.prepared,
        prepared_run.arena,
        1,
    );

    let (decode_attention, decode) =
        prepare_decode_step(&mut prepared_run, u64::try_from(prompt.len() + 1).unwrap());
    let decode_started = Instant::now();
    let decode_output = decoder
        .capture_decode_with_logits(DecoderStep {
            tokens: &[next_token],
            positions: &[u32::try_from(prompt.len()).unwrap()],
            write_slots: &decode.steps()[0].classes[0].write_slots,
            attention: &decode_attention,
        })
        .unwrap();
    eprintln!(
        "decode capture: warmup and outer graph construction completed after {:.3}s",
        decode_started.elapsed().as_secs_f64()
    );
    assert_eq!(decode_output.logits.len(), config.vocabulary_size);
    assert_eq!(decode_output.token_ids.len(), 1);
    assert_eq!(
        decode_output.token_ids[0],
        greedy_token(&decode_output.logits, config.vocabulary_size)
    );
    assert!(decoder.has_captured_decode());
    assert_eq!(decoder.active_bucket_index(), 0);
    assert_eq!(decoder.cache_updates_in_place(), cache_updates_in_place);
    complete(&mut prepared_run.session, &decode, prepared_run.arena, 2);

    let second_token = decode_output.token_ids[0];
    let (second_attention, second_decode) =
        prepare_decode_step(&mut prepared_run, u64::try_from(prompt.len() + 2).unwrap());
    let replay_started = Instant::now();
    let second_output = decoder
        .replay_decode_with_logits(DecoderStep {
            tokens: &[second_token],
            positions: &[u32::try_from(prompt.len() + 1).unwrap()],
            write_slots: &second_decode.steps()[0].classes[0].write_slots,
            attention: &second_attention,
        })
        .unwrap();
    eprintln!(
        "decode replay with diagnostic logits completed after {:.3}s",
        replay_started.elapsed().as_secs_f64()
    );
    assert_eq!(second_output.logits.len(), config.vocabulary_size);
    assert_eq!(second_output.token_ids.len(), 1);
    assert_eq!(
        second_output.token_ids[0],
        greedy_token(&second_output.logits, config.vocabulary_size)
    );
    assert_eq!(decoder.active_bucket_index(), 0);
    complete(
        &mut prepared_run.session,
        &second_decode,
        prepared_run.arena,
        3,
    );

    let third_token = second_output.token_ids[0];
    let (third_attention, third_decode) =
        prepare_decode_step(&mut prepared_run, u64::try_from(prompt.len() + 3).unwrap());
    let started = Instant::now();
    let third_output = decoder
        .replay_decode(DecoderStep {
            tokens: &[third_token],
            positions: &[u32::try_from(prompt.len() + 2).unwrap()],
            write_slots: &third_decode.steps()[0].classes[0].write_slots,
            attention: &third_attention,
        })
        .unwrap();
    eprintln!(
        "device-only CUDA graph replay: dispatched after {:.3}s",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(third_output.token_ids.len(), 1);
    assert_eq!(decoder.active_bucket_index(), 0);
    benchmark_decode_dispatch(
        &mut decoder,
        DecoderStep {
            tokens: &[third_token],
            positions: &[u32::try_from(prompt.len() + 2).unwrap()],
            write_slots: &third_decode.steps()[0].classes[0].write_slots,
            attention: &third_attention,
        },
        third_output.token_ids[0],
    );
    complete(
        &mut prepared_run.session,
        &third_decode,
        prepared_run.arena,
        4,
    );
}
