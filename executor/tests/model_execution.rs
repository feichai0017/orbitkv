#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence, EngineReleaseEvidence,
    EngineReleaseOutcome, EngineRequestId, EngineRetirementEvidence, HfRetentionOptions,
    RuntimeSession, compile_attention_state_plan, compile_hf_runtime_manifest, compile_plan,
    compile_runtime_manifest,
    kv_manager::{
        BackendArenaRegistration, CanonicalKvManager, ManagerConfig, PhysicalResidencePolicy,
    },
    plan::RetentionKind,
};
use orbitkv_executor::{
    AttentionBatch, ExecutorArena, ExecutorPlan, PreparedBatch,
    model::{CompiledDecoder, DecoderClassStep, DecoderCompileConfig, DecoderConfig, DecoderStep},
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
    arenas: Box<[ExecutorArena]>,
    prepared: PreparedBatch,
    attention: Box<[AttentionBatch]>,
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
    let registrations = manager_plan
        .classes
        .iter()
        .enumerate()
        .map(|(index, _)| BackendArenaRegistration {
            pool_id: u32::try_from(index + 1).unwrap(),
            class_id: u16::try_from(index).unwrap(),
            backend_domain: u16::try_from(index + 1).unwrap(),
            page_count: PAGE_COUNT,
            reserved: 0,
            backend_base_index: 0,
        })
        .collect::<Vec<_>>();
    let manager = CanonicalKvManager::new(
        &manager_plan,
        ManagerConfig {
            maximum_requests: 2,
            maximum_operations: 4,
            maximum_prefixes: 1,
            maximum_reclamations: PAGE_COUNT * u32::try_from(registrations.len()).unwrap(),
            maximum_step_tokens: u32::try_from(prompt_len.max(64)).unwrap(),
        },
        &registrations,
    )
    .unwrap();
    let mut session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
    let request_id = EngineRequestId(1);
    session.acquire_requests(&[request_id]).unwrap();
    let source = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: u64::try_from(prompt_len).unwrap(),
        }])
        .unwrap();
    let view = session.prepared_execution_view(source.batch_id).unwrap();
    let arenas = session
        .arena_stats()
        .iter()
        .copied()
        .zip(registrations)
        .map(|(stats, registration)| ExecutorArena::bind(stats, registration).unwrap())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let executor_plan = ExecutorPlan::compile(&manifest).unwrap();
    let attention = executor_plan.attention_batches(&view).unwrap();
    let prepared = executor_plan.lower_prepared(source, &arenas).unwrap();
    PreparedModelRun {
        session,
        executor_plan,
        arenas,
        prepared,
        attention,
    }
}

fn prepare_hybrid_policy_run(
    config: &DecoderConfig,
    prompt_len: usize,
    physical_residence: PhysicalResidencePolicy,
) -> PreparedModelRun {
    let input = hybrid_policy_input(config);
    let manifest = compile_runtime_manifest(input.clone()).unwrap();
    let manager_plan = compile_plan(
        compile_attention_state_plan(input)
            .unwrap()
            .token_manager_plan()
            .unwrap(),
    )
    .unwrap();
    let registrations = hybrid_registrations();
    let manager = CanonicalKvManager::new_with_residence(
        &manager_plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 4,
            maximum_prefixes: 1,
            maximum_reclamations: PAGE_COUNT * 2,
            maximum_step_tokens: u32::try_from(prompt_len.max(64)).unwrap(),
        },
        &registrations,
        physical_residence,
    )
    .unwrap();
    let mut session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
    let request_id = EngineRequestId(1);
    session.acquire_requests(&[request_id]).unwrap();
    let source = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: u64::try_from(prompt_len).unwrap(),
        }])
        .unwrap();
    let view = session.prepared_execution_view(source.batch_id).unwrap();
    let arenas = session
        .arena_stats()
        .iter()
        .copied()
        .zip(registrations)
        .map(|(stats, registration)| ExecutorArena::bind(stats, registration).unwrap())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let executor_plan = ExecutorPlan::compile(&manifest).unwrap();
    let attention = executor_plan.attention_batches(&view).unwrap();
    let prepared = executor_plan.lower_prepared(source, &arenas).unwrap();
    PreparedModelRun {
        session,
        executor_plan,
        arenas,
        prepared,
        attention,
    }
}

fn hybrid_policy_input(config: &DecoderConfig) -> AttentionStatePlanInput {
    let key_bytes_per_token_per_layer =
        u64::try_from(config.kv_heads * config.head_dim * 2).unwrap();
    let mut full_layers = Vec::new();
    let mut sliding_layers = Vec::new();
    for layer in 0..u32::try_from(config.layers).unwrap() {
        if layer.is_multiple_of(2) {
            full_layers.push(layer);
        } else {
            sliding_layers.push(layer);
        }
    }
    AttentionStatePlanInput {
        page_tokens: PAGE_TOKENS,
        states: vec![
            AttentionStateSpec {
                name: "global".into(),
                layers: full_layers,
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer,
                    value_bytes_per_token_per_layer: key_bytes_per_token_per_layer,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "local".into(),
                layers: sliding_layers,
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer,
                    value_bytes_per_token_per_layer: key_bytes_per_token_per_layer,
                    retention: RetentionKind::Sliding,
                    window_tokens: Some(64),
                },
            },
        ],
    }
}

fn hybrid_registrations() -> [BackendArenaRegistration; 2] {
    [
        BackendArenaRegistration {
            pool_id: 1,
            class_id: 0,
            backend_domain: 1,
            page_count: PAGE_COUNT,
            reserved: 0,
            backend_base_index: 0,
        },
        BackendArenaRegistration {
            pool_id: 2,
            class_id: 1,
            backend_domain: 2,
            page_count: PAGE_COUNT,
            reserved: 0,
            backend_base_index: 0,
        },
    ]
}

fn complete(
    session: &mut RuntimeSession,
    prepared: &PreparedBatch,
    arenas: &[ExecutorArena],
    completion_value: u64,
) {
    let evidence = prepared.execution_evidence_after_success(arenas).unwrap();
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
            reclamation_receipts: publication
                .retirements
                .iter()
                .map(|retirement| EngineRetirementEvidence {
                    page: retirement.page,
                    backend_domain: retirement.backend_domain,
                    acknowledged: true,
                    backend_index: retirement.backend_index,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
        .unwrap();
}

fn release_and_drain(session: &mut RuntimeSession, request_id: EngineRequestId) {
    let release = session
        .prepare_release_batch(&[request_id])
        .expect("prepare release");
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: release
                .retirements
                .iter()
                .map(|retirement| EngineRetirementEvidence {
                    page: retirement.page,
                    backend_domain: retirement.backend_domain,
                    acknowledged: true,
                    backend_index: retirement.backend_index,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
    let stats = session.stats();
    assert_eq!(stats.active_requests, 0);
    assert_eq!(stats.active_snapshots, 0);
    assert_eq!(stats.active_pages, 0);
    assert_eq!(stats.retiring_pages, 0);
    assert_eq!(stats.quarantined_pages, 0);
    assert_eq!(stats.pending_reclamations, 0);
    assert_eq!(stats.total_request_page_refs, 0);
    assert_eq!(stats.total_reader_pins, 0);
    assert!(
        session
            .arena_stats()
            .iter()
            .all(|arena| arena.free_pages == u64::from(arena.page_count))
    );
}

fn prepare_decode_step(
    prepared_run: &mut PreparedModelRun,
    target_boundary: u64,
) -> (Box<[AttentionBatch]>, PreparedBatch, bool) {
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
    let attention = prepared_run.executor_plan.attention_batches(&view).unwrap();
    let reused_generation = source
        .steps
        .iter()
        .flat_map(|step| step.write_intents.iter())
        .any(|write| write.page_generation > 1);
    let prepared = prepared_run
        .executor_plan
        .lower_prepared(source, &prepared_run.arenas)
        .unwrap();
    (attention, prepared, reused_generation)
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

fn compile_hybrid_decoder(
    config: &DecoderConfig,
    run: &PreparedModelRun,
    model_dir: &std::path::Path,
    prompt_tokens: usize,
) -> (CompiledDecoder, Duration) {
    let context = luminal_cuda_lite::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let compile = DecoderCompileConfig {
        maximum_query_tokens: prompt_tokens,
        representative_prefill_tokens: prompt_tokens,
        maximum_batch_size: 1,
        maximum_context_pages: usize::try_from(PAGE_COUNT).unwrap(),
        representative_context_pages: run
            .attention
            .iter()
            .map(|attention| attention.page_indices.len())
            .max()
            .unwrap(),
        search_graphs: search_graphs(),
        search_seed: 7,
    };
    let started = Instant::now();
    let decoder = CompiledDecoder::compile(
        config,
        &run.executor_plan,
        &run.arenas,
        &stream,
        &[model_dir.join("model.safetensors")],
        compile,
    )
    .unwrap();
    (decoder, started.elapsed())
}

struct ResidenceArmResult {
    prefill_token_ids: Box<[u32]>,
    prefill_logits: Box<[f32]>,
    decode_token_ids: Box<[u32]>,
    decode_logits: Box<[f32]>,
    sliding_resident_pages: u64,
    sliding_resident_bytes: u64,
}

fn execute_residence_arm(
    policy: PhysicalResidencePolicy,
    run: &mut PreparedModelRun,
    decoder: &mut CompiledDecoder,
    prompt: &[u32],
    positions: &[u32],
    compile_elapsed: Duration,
) -> ResidenceArmResult {
    let before = run.session.arena_stats();
    let execute_started = Instant::now();
    let output = execute_step(
        "residence ablation prefill",
        decoder,
        prompt,
        positions,
        &run.attention,
        &run.prepared,
    );
    let prefill_elapsed = execute_started.elapsed();
    let next_token = *output.token_ids.last().expect("prefill token");
    complete(&mut run.session, &run.prepared, &run.arenas, 1);
    let target_boundary = u64::try_from(prompt.len() + 1).unwrap();
    let (decode_attention, decode, _) = prepare_decode_step(run, target_boundary);
    let decode_classes = decoder_class_steps(&decode, &decode_attention);
    let decode_started = Instant::now();
    let decode_output = decoder
        .execute_with_logits(DecoderStep {
            tokens: &[next_token],
            positions: &[u32::try_from(prompt.len()).unwrap()],
            classes: &decode_classes,
        })
        .unwrap();
    let decode_elapsed = decode_started.elapsed();
    complete(&mut run.session, &decode, &run.arenas, 2);
    let after = run.session.arena_stats();
    let sliding = after
        .iter()
        .find(|arena| arena.class_id == 1)
        .expect("Sliding arena");
    eprintln!(
        "residence ablation: policy={policy:?} compile_seconds={:.3} prefill_seconds={:.6} decode_seconds={:.6} sliding_resident_pages={} sliding_resident_bytes={} total_resident_pages={} total_resident_bytes={} free_before={} free_after={}",
        compile_elapsed.as_secs_f64(),
        prefill_elapsed.as_secs_f64(),
        decode_elapsed.as_secs_f64(),
        sliding.resident_pages,
        sliding.resident_bytes,
        after.iter().map(|arena| arena.resident_pages).sum::<u64>(),
        after.iter().map(|arena| arena.resident_bytes).sum::<u64>(),
        before.iter().map(|arena| arena.free_pages).sum::<u64>(),
        after.iter().map(|arena| arena.free_pages).sum::<u64>(),
    );
    ResidenceArmResult {
        prefill_token_ids: output.token_ids,
        prefill_logits: output.logits,
        decode_token_ids: decode_output.token_ids,
        decode_logits: decode_output.logits,
        sliding_resident_pages: sliding.resident_pages,
        sliding_resident_bytes: sliding.resident_bytes,
    }
}

fn assert_greedy_output(
    output: &orbitkv_executor::model::DecoderDiagnosticOutput,
    rows: usize,
    vocabulary_size: usize,
) {
    assert_eq!(output.logits.len(), rows * vocabulary_size);
    assert_eq!(output.token_ids.len(), rows);
    assert_eq!(
        output.token_ids.last().copied(),
        Some(greedy_token(&output.logits, vocabulary_size))
    );
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
    attention: &[AttentionBatch],
    prepared: &PreparedBatch,
) -> orbitkv_executor::model::DecoderDiagnosticOutput {
    let started = Instant::now();
    let classes = decoder_class_steps(prepared, attention);
    let logits = decoder
        .execute_with_logits(DecoderStep {
            tokens,
            positions,
            classes: &classes,
        })
        .unwrap();
    eprintln!(
        "{phase}: dispatched precompiled bucket after {:.3}s",
        started.elapsed().as_secs_f64()
    );
    logits
}

fn decoder_class_steps<'a>(
    prepared: &'a PreparedBatch,
    attention: &'a [AttentionBatch],
) -> Vec<DecoderClassStep<'a>> {
    prepared.steps()[0]
        .classes
        .iter()
        .zip(attention)
        .map(|(class, attention)| DecoderClassStep {
            class_id: class.class_id,
            write_slots: &class.write_slots,
            attention,
        })
        .collect()
}

fn qualify_hybrid_reference_probes(config_bytes: &[u8], decoder: &mut CompiledDecoder) {
    for (prompt_len, expected) in [(1_usize, 9_450_u32), (2, 3_302), (4, 236_764), (16, 106)] {
        let mut probe = prepare_model_run(config_bytes, prompt_len);
        let tokens = (0..u32::try_from(prompt_len).unwrap()).collect::<Vec<_>>();
        let output = execute_step(
            "released hybrid parity probe",
            decoder,
            &tokens,
            &tokens,
            &probe.attention,
            &probe.prepared,
        );
        assert_eq!(output.token_ids.last().copied(), Some(expected));
        complete(&mut probe.session, &probe.prepared, &probe.arenas, 1);
        release_and_drain(&mut probe.session, EngineRequestId(1));
    }
}

struct HybridGenerationMetrics {
    prefill_elapsed: Duration,
    decode_durations: Vec<Duration>,
    full_pages: u64,
    sliding_pages: u64,
    final_token: u32,
}

fn qualify_hybrid_generation(
    run: &mut PreparedModelRun,
    decoder: &mut CompiledDecoder,
    config: &DecoderConfig,
    prompt: &[u32],
    positions: &[u32],
    reference_tokens: &[u32],
) -> HybridGenerationMetrics {
    let prefill_started = Instant::now();
    let output = execute_step(
        "released hybrid prefill",
        decoder,
        prompt,
        positions,
        &run.attention,
        &run.prepared,
    );
    let prefill_elapsed = prefill_started.elapsed();
    assert_greedy_output(&output, prompt.len(), config.vocabulary_size);
    assert_eq!(output.token_ids.last().copied(), Some(reference_tokens[0]));
    complete(&mut run.session, &run.prepared, &run.arenas, 1);

    let mut token = *output.token_ids.last().unwrap();
    let mut generated = vec![token];
    let mut decode_durations = Vec::new();
    let mut reused_generation = false;
    for offset in 1..u64::try_from(reference_tokens.len()).unwrap() {
        let boundary = u64::try_from(prompt.len()).unwrap() + offset;
        let (attention, prepared, reused) = prepare_decode_step(run, boundary);
        reused_generation |= reused;
        let classes = decoder_class_steps(&prepared, &attention);
        let started = Instant::now();
        let output = decoder
            .execute(DecoderStep {
                tokens: &[token],
                positions: &[u32::try_from(boundary - 1).unwrap()],
                classes: &classes,
            })
            .unwrap();
        decode_durations.push(started.elapsed());
        token = output.token_ids[0];
        generated.push(token);
        complete(&mut run.session, &prepared, &run.arenas, offset + 1);
    }

    let arenas = run.session.arena_stats();
    let full_pages = arenas
        .iter()
        .find(|arena| arena.class_id == 0)
        .unwrap()
        .active_pages;
    let sliding = arenas.iter().find(|arena| arena.class_id == 1).unwrap();
    let sliding_pages = sliding.active_pages;
    assert_eq!(full_pages, 35);
    assert_eq!(sliding_pages, 33);
    assert!(sliding.free_pages > 0);
    assert!(reused_generation);
    assert_eq!(generated, reference_tokens);
    release_and_drain(&mut run.session, EngineRequestId(1));

    HybridGenerationMetrics {
        prefill_elapsed,
        decode_durations,
        full_pages,
        sliding_pages,
        final_token: token,
    }
}

fn qualify_reused_cancelled_request(
    run: &mut PreparedModelRun,
    decoder: &mut CompiledDecoder,
    prompt: &[u32],
    positions: &[u32],
) {
    let request_id = EngineRequestId(2);
    run.session.acquire_requests(&[request_id]).unwrap();
    let source = run
        .session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 16,
        }])
        .unwrap();
    assert!(
        source
            .steps
            .iter()
            .flat_map(|step| step.write_intents.iter())
            .any(|write| write.page_generation > 1)
    );
    let view = run
        .session
        .prepared_execution_view(source.batch_id)
        .unwrap();
    let attention = run.executor_plan.attention_batches(&view).unwrap();
    let prepared = run
        .executor_plan
        .lower_prepared(source, &run.arenas)
        .unwrap();
    let output = execute_step(
        "released hybrid cancellation prefill",
        decoder,
        &prompt[..16],
        &positions[..16],
        &attention,
        &prepared,
    );
    assert_eq!(output.token_ids.last().copied(), Some(106));
    complete(&mut run.session, &prepared, &run.arenas, 35);
    release_and_drain(&mut run.session, request_id);
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
        representative_context_pages: prepared_run.attention[0].page_indices.len(),
        search_graphs: search_graphs(),
        search_seed: 7,
    };
    let compile_started = Instant::now();
    let mut decoder = CompiledDecoder::compile(
        &config,
        &prepared_run.executor_plan,
        &prepared_run.arenas,
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
        &prepared_run.arenas,
        1,
    );

    let (decode_attention, decode, _) =
        prepare_decode_step(&mut prepared_run, u64::try_from(prompt.len() + 1).unwrap());
    let decode_started = Instant::now();
    let decode_classes = decoder_class_steps(&decode, &decode_attention);
    let decode_output = decoder
        .capture_decode_with_logits(DecoderStep {
            tokens: &[next_token],
            positions: &[u32::try_from(prompt.len()).unwrap()],
            classes: &decode_classes,
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
    complete(&mut prepared_run.session, &decode, &prepared_run.arenas, 2);

    let second_token = decode_output.token_ids[0];
    let (second_attention, second_decode, _) =
        prepare_decode_step(&mut prepared_run, u64::try_from(prompt.len() + 2).unwrap());
    let replay_started = Instant::now();
    let second_classes = decoder_class_steps(&second_decode, &second_attention);
    let second_output = decoder
        .replay_decode_with_logits(DecoderStep {
            tokens: &[second_token],
            positions: &[u32::try_from(prompt.len() + 1).unwrap()],
            classes: &second_classes,
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
        &prepared_run.arenas,
        3,
    );

    let third_token = second_output.token_ids[0];
    let (third_attention, third_decode, _) =
        prepare_decode_step(&mut prepared_run, u64::try_from(prompt.len() + 3).unwrap());
    let started = Instant::now();
    let third_classes = decoder_class_steps(&third_decode, &third_attention);
    let third_output = decoder
        .replay_decode(DecoderStep {
            tokens: &[third_token],
            positions: &[u32::try_from(prompt.len() + 2).unwrap()],
            classes: &third_classes,
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
            classes: &third_classes,
        },
        third_output.token_ids[0],
    );
    complete(
        &mut prepared_run.session,
        &third_decode,
        &prepared_run.arenas,
        4,
    );
}

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
fn released_dense_checkpoint_executes_multi_class_policy_plumbing() {
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let prompt = [1_u32, 2, 3, 4];
    let mut prepared_run =
        prepare_hybrid_policy_run(&config, prompt.len(), PhysicalResidencePolicy::Compiled);
    assert_eq!(prepared_run.executor_plan.classes.len(), 2);
    assert_eq!(prepared_run.attention.len(), 2);
    assert_eq!(prepared_run.prepared.steps()[0].classes.len(), 2);

    let context = luminal_cuda_lite::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let compile = DecoderCompileConfig {
        maximum_query_tokens: 8,
        representative_prefill_tokens: prompt.len(),
        maximum_batch_size: 1,
        maximum_context_pages: usize::try_from(PAGE_COUNT).unwrap(),
        representative_context_pages: prepared_run
            .attention
            .iter()
            .map(|attention| attention.page_indices.len())
            .max()
            .unwrap(),
        search_graphs: search_graphs(),
        search_seed: 7,
    };
    let compile_started = Instant::now();
    let mut decoder = CompiledDecoder::compile(
        &config,
        &prepared_run.executor_plan,
        &prepared_run.arenas,
        &stream,
        &[model_dir.join("model.safetensors")],
        compile,
    )
    .unwrap();
    eprintln!(
        "multi-class compile: elapsed_seconds={:.1} classes={} layer_counts={:?} buckets={} persistent_cache_tensors={}",
        compile_started.elapsed().as_secs_f64(),
        prepared_run.executor_plan.classes.len(),
        prepared_run
            .executor_plan
            .classes
            .iter()
            .map(|class| class.layers.len())
            .collect::<Vec<_>>(),
        decoder.compiled_bucket_count(),
        decoder.persistent_cache_count(),
    );
    assert_eq!(decoder.compiled_bucket_count(), 2);
    assert_eq!(decoder.persistent_cache_count(), config.layers * 2);
    let positions = (0..u32::try_from(prompt.len()).unwrap()).collect::<Vec<_>>();
    let output = execute_step(
        "multi-class prefill",
        &mut decoder,
        &prompt,
        &positions,
        &prepared_run.attention,
        &prepared_run.prepared,
    );
    assert_greedy_output(&output, prompt.len(), config.vocabulary_size);
    complete(
        &mut prepared_run.session,
        &prepared_run.prepared,
        &prepared_run.arenas,
        1,
    );

    let next_token = *output.token_ids.last().unwrap();
    let (attention, prepared, _) = prepare_decode_step(&mut prepared_run, 5);
    let classes = decoder_class_steps(&prepared, &attention);
    let decode_started = Instant::now();
    let decode_output = decoder
        .capture_decode_with_logits(DecoderStep {
            tokens: &[next_token],
            positions: &[4],
            classes: &classes,
        })
        .unwrap();
    eprintln!(
        "multi-class decode capture: elapsed_seconds={:.3}",
        decode_started.elapsed().as_secs_f64()
    );
    assert_greedy_output(&decode_output, 1, config.vocabulary_size);
    assert!(decoder.has_captured_decode());
    complete(
        &mut prepared_run.session,
        &prepared,
        &prepared_run.arenas,
        2,
    );
}

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
fn released_checkpoint_compares_compiled_and_request_lifetime_residence() {
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let prompt_tokens = 80_usize;
    let prompt = (1..=u32::try_from(prompt_tokens).unwrap()).collect::<Vec<_>>();
    let positions = (0..u32::try_from(prompt_tokens).unwrap()).collect::<Vec<_>>();
    let mut runs = [
        (
            PhysicalResidencePolicy::Compiled,
            prepare_hybrid_policy_run(&config, prompt_tokens, PhysicalResidencePolicy::Compiled),
        ),
        (
            PhysicalResidencePolicy::RequestLifetime,
            prepare_hybrid_policy_run(
                &config,
                prompt_tokens,
                PhysicalResidencePolicy::RequestLifetime,
            ),
        ),
    ];
    assert_eq!(runs[0].1.executor_plan, runs[1].1.executor_plan);
    assert_eq!(runs[0].1.attention, runs[1].1.attention);
    assert!(
        runs[0]
            .1
            .arenas
            .iter()
            .zip(&runs[1].1.arenas)
            .all(|(left, right)| {
                left.class_id == right.class_id
                    && left.backend_domain == right.backend_domain
                    && left.page_count == right.page_count
                    && left.backend_base_index == right.backend_base_index
            })
    );
    let (mut decoder, compile_elapsed) =
        compile_hybrid_decoder(&config, &runs[0].1, &model_dir, prompt_tokens);
    eprintln!(
        "residence ablation: one-time compile_seconds={:.3}",
        compile_elapsed.as_secs_f64(),
    );
    let mut outputs = Vec::new();

    for (policy, run) in &mut runs {
        outputs.push(execute_residence_arm(
            *policy,
            run,
            &mut decoder,
            &prompt,
            &positions,
            compile_elapsed,
        ));
    }

    assert_eq!(outputs[0].prefill_token_ids, outputs[1].prefill_token_ids);
    assert_eq!(outputs[0].prefill_logits, outputs[1].prefill_logits);
    assert_eq!(outputs[0].decode_token_ids, outputs[1].decode_token_ids);
    assert_eq!(outputs[0].decode_logits, outputs[1].decode_logits);
    assert_eq!(outputs[0].sliding_resident_pages, 5);
    assert_eq!(outputs[1].sliding_resident_pages, 6);
    assert_eq!(outputs[0].sliding_resident_bytes, 491_520);
    assert_eq!(outputs[1].sliding_resident_bytes, 589_824);
}

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
fn released_hybrid_checkpoint_crosses_window_and_drains() {
    const REFERENCE_TOKENS: [u32; 34] = [
        236_743, 199, 236_820, 34_280, 236_813, 208, 236_820, 34_280, 236_813, 208, 236_820,
        34_280, 236_813, 208, 236_820, 34_280, 236_813, 208, 236_820, 34_280, 236_813, 208,
        236_820, 34_280, 236_813, 208, 236_820, 34_280, 236_813, 208, 236_820, 34_280, 236_813,
        208,
    ];
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let prompt_tokens = 512_usize;
    let prompt = (0..prompt_tokens)
        .map(|index| u32::try_from(index % config.vocabulary_size).unwrap())
        .collect::<Vec<_>>();
    let positions = (0..u32::try_from(prompt_tokens).unwrap()).collect::<Vec<_>>();
    let mut run = prepare_model_run(&config_bytes, prompt_tokens);
    assert_eq!(run.executor_plan.classes.len(), 2);
    assert_eq!(run.executor_plan.classes[0].layers.len(), 3);
    assert_eq!(run.executor_plan.classes[1].layers.len(), 15);
    let (mut decoder, compile_elapsed) =
        compile_hybrid_decoder(&config, &run, &model_dir, prompt_tokens);
    qualify_hybrid_reference_probes(&config_bytes, &mut decoder);
    let mut metrics = qualify_hybrid_generation(
        &mut run,
        &mut decoder,
        &config,
        &prompt,
        &positions,
        &REFERENCE_TOKENS,
    );
    qualify_reused_cancelled_request(&mut run, &mut decoder, &prompt, &positions);
    let median_decode = median_duration(&mut metrics.decode_durations);
    eprintln!(
        "released hybrid lifecycle: compile_seconds={:.3} prefill_seconds={:.6} decode_iterations={} decode_median_seconds={:.6} full_pages={} sliding_pages={} final_token={}",
        compile_elapsed.as_secs_f64(),
        metrics.prefill_elapsed.as_secs_f64(),
        metrics.decode_durations.len(),
        median_decode.as_secs_f64(),
        metrics.full_pages,
        metrics.sliding_pages,
        metrics.final_token,
    );
}
