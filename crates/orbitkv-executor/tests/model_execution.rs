#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use orbitkv::{
    CacheSharingPolicy, EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence,
    EngineReleaseEvidence, EngineReleaseOutcome, EngineRequestId, EngineRetirementEvidence,
    HfRetentionOptions, RuntimeSession, RuntimeSessionError, compile_hf_runtime_manifest,
    kv_manager::{
        BackendArenaRegistration, CanonicalKvManager, KvManagerError, ManagerConfig,
        PhysicalResidencePolicy,
    },
};
use orbitkv_executor::{
    AttentionBatch, ExecutorArena, ExecutorPlan, PreparedBatch,
    model::{CompiledDecoder, DecoderClassStep, DecoderCompileConfig, DecoderConfig, DecoderStep},
};

const PAGE_TOKENS: u64 = 16;
const PAGE_COUNT: u32 = 64;
const HYBRID_REFERENCE_TOKENS: [u32; 34] = [
    236_743, 199, 236_820, 34_280, 236_813, 208, 236_820, 34_280, 236_813, 208, 236_820, 34_280,
    236_813, 208, 236_820, 34_280, 236_813, 208, 236_820, 34_280, 236_813, 208, 236_820, 34_280,
    236_813, 208, 236_820, 34_280, 236_813, 208, 236_820, 34_280, 236_813, 208,
];
const HYBRID_STABLE_REFERENCE_PREFIX: usize = 13;

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
    prepare_model_run_with_residence(config_bytes, prompt_len, PhysicalResidencePolicy::Compiled)
}

fn prepare_model_run_with_residence(
    config_bytes: &[u8],
    prompt_len: usize,
    physical_residence: PhysicalResidencePolicy,
) -> PreparedModelRun {
    prepare_model_run_with_page_counts(config_bytes, prompt_len, physical_residence, None)
}

fn prepare_model_run_with_page_counts(
    config_bytes: &[u8],
    prompt_len: usize,
    physical_residence: PhysicalResidencePolicy,
    page_counts: Option<&[u32]>,
) -> PreparedModelRun {
    let mut harness = model_harness(
        config_bytes,
        2,
        4,
        prompt_len.max(64),
        physical_residence,
        page_counts,
    );
    let request_id = EngineRequestId(1);
    harness.session.acquire_requests(&[request_id]).unwrap();
    let (attention, prepared) = prepare_model_batch(
        &mut harness,
        &[request_id],
        u64::try_from(prompt_len).unwrap(),
    );
    PreparedModelRun {
        session: harness.session,
        executor_plan: harness.executor_plan,
        arenas: harness.arenas,
        prepared,
        attention,
    }
}

struct ModelBatchHarness {
    session: RuntimeSession,
    executor_plan: ExecutorPlan,
    arenas: Box<[ExecutorArena]>,
}

fn model_harness(
    config_bytes: &[u8],
    maximum_requests: usize,
    maximum_operations: usize,
    maximum_step_tokens: usize,
    physical_residence: PhysicalResidencePolicy,
    page_counts: Option<&[u32]>,
) -> ModelBatchHarness {
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
    if let Some(page_counts) = page_counts {
        assert_eq!(page_counts.len(), manager_plan.classes.len());
    }
    let registrations = manager_plan
        .classes
        .iter()
        .enumerate()
        .map(|(index, _)| BackendArenaRegistration {
            pool_id: u32::try_from(index + 1).unwrap(),
            class_id: u16::try_from(index).unwrap(),
            backend_domain: u16::try_from(index + 1).unwrap(),
            page_count: page_counts.map_or(PAGE_COUNT, |counts| counts[index]),
            reserved: 0,
            backend_base_index: 0,
        })
        .collect::<Vec<_>>();
    let manager = CanonicalKvManager::new_with_residence(
        &manager_plan,
        ManagerConfig {
            maximum_requests: u32::try_from(maximum_requests).unwrap(),
            maximum_operations: u32::try_from(maximum_operations).unwrap(),
            maximum_prefixes: 1,
            maximum_reclamations: registrations.iter().map(|arena| arena.page_count).sum(),
            maximum_step_tokens: u32::try_from(maximum_step_tokens).unwrap(),
        },
        &registrations,
        physical_residence,
    )
    .unwrap();
    let session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
    let arenas = session
        .arena_stats()
        .iter()
        .copied()
        .zip(registrations)
        .map(|(stats, registration)| ExecutorArena::bind(stats, registration).unwrap())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let executor_plan = ExecutorPlan::compile(&manifest).unwrap();
    ModelBatchHarness {
        session,
        executor_plan,
        arenas,
    }
}

fn model_batch_harness(
    config_bytes: &[u8],
    maximum_requests: usize,
    maximum_step_tokens: usize,
) -> ModelBatchHarness {
    let mut harness = model_harness(
        config_bytes,
        maximum_requests,
        maximum_requests,
        maximum_step_tokens,
        PhysicalResidencePolicy::Compiled,
        None,
    );
    let request_ids = (0..maximum_requests)
        .map(|index| EngineRequestId(u64::try_from(index + 1).unwrap()))
        .collect::<Vec<_>>();
    harness.session.acquire_requests(&request_ids).unwrap();
    harness
}

fn prepare_model_batch(
    harness: &mut ModelBatchHarness,
    request_ids: &[EngineRequestId],
    target_boundary: u64,
) -> (Box<[AttentionBatch]>, PreparedBatch) {
    let intents = request_ids
        .iter()
        .copied()
        .map(|request_id| EngineAppendIntent {
            request_id,
            target_boundary,
        })
        .collect::<Vec<_>>();
    let source = harness.session.prepare_append_batch(&intents).unwrap();
    let view = harness
        .session
        .prepared_execution_view(source.batch_id)
        .unwrap();
    let attention = harness.executor_plan.attention_batches(&view).unwrap();
    let prepared = harness
        .executor_plan
        .lower_prepared(source, &harness.arenas)
        .unwrap();
    (attention, prepared)
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
    release_requests_and_drain(session, &[request_id]);
}

fn release_requests_and_drain(session: &mut RuntimeSession, request_ids: &[EngineRequestId]) {
    let release = session
        .prepare_release_batch(request_ids)
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
    prepare_decode_step_for_request(prepared_run, EngineRequestId(1), target_boundary)
}

fn prepare_decode_step_for_request(
    prepared_run: &mut PreparedModelRun,
    request_id: EngineRequestId,
    target_boundary: u64,
) -> (Box<[AttentionBatch]>, PreparedBatch, bool) {
    let source = prepared_run
        .session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
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
    generated_tokens: Box<[u32]>,
    total_elapsed: Duration,
    prefill_elapsed: Duration,
    decode_elapsed: Duration,
    manager_elapsed: Duration,
    full_resident_pages: u64,
    sliding_resident_pages: u64,
    total_resident_bytes: u64,
    sliding_resident_bytes: u64,
}

fn execute_released_residence_arm(
    policy: PhysicalResidencePolicy,
    config_bytes: &[u8],
    decoder: &mut CompiledDecoder,
    prompt: &[u32],
    positions: &[u32],
    generated_token_count: usize,
) -> ResidenceArmResult {
    let mut run = prepare_model_run_with_residence(config_bytes, prompt.len(), policy);
    let total_started = Instant::now();
    let mut manager_elapsed = Duration::ZERO;
    let classes = decoder_class_steps(&run.prepared, &run.attention);
    let prefill_started = Instant::now();
    let output = decoder
        .execute(DecoderStep {
            tokens: prompt,
            positions,
            classes: &classes,
        })
        .unwrap();
    let prefill_elapsed = prefill_started.elapsed();
    let mut token = *output.token_ids.last().expect("prefill token");
    let mut generated_tokens = vec![token];
    let publish_started = Instant::now();
    complete(&mut run.session, &run.prepared, &run.arenas, 1);
    manager_elapsed += publish_started.elapsed();

    let mut decode_elapsed = Duration::ZERO;
    for offset in 1..u64::try_from(generated_token_count).unwrap() {
        let manager_started = Instant::now();
        let boundary = u64::try_from(prompt.len()).unwrap() + offset;
        let (attention, prepared, _) = prepare_decode_step(&mut run, boundary);
        manager_elapsed += manager_started.elapsed();
        let classes = decoder_class_steps(&prepared, &attention);
        let execute_started = Instant::now();
        let output = decoder
            .execute(DecoderStep {
                tokens: &[token],
                positions: &[u32::try_from(boundary - 1).unwrap()],
                classes: &classes,
            })
            .unwrap();
        decode_elapsed += execute_started.elapsed();
        token = output.token_ids[0];
        generated_tokens.push(token);
        let publish_started = Instant::now();
        complete(&mut run.session, &prepared, &run.arenas, offset + 1);
        manager_elapsed += publish_started.elapsed();
    }
    assert_eq!(
        &generated_tokens[..HYBRID_STABLE_REFERENCE_PREFIX],
        &HYBRID_REFERENCE_TOKENS[..HYBRID_STABLE_REFERENCE_PREFIX]
    );

    let after = run.session.arena_stats();
    let full = after
        .iter()
        .find(|arena| arena.class_id == 0)
        .expect("Full arena");
    let sliding = after
        .iter()
        .find(|arena| arena.class_id == 1)
        .expect("Sliding arena");
    let total_resident_bytes = after.iter().map(|arena| arena.resident_bytes).sum();
    let release_started = Instant::now();
    release_and_drain(&mut run.session, EngineRequestId(1));
    manager_elapsed += release_started.elapsed();
    let total_elapsed = total_started.elapsed();
    eprintln!(
        "released residence arm: policy={policy:?} total_seconds={:.6} prefill_seconds={:.6} decode_seconds={:.6} manager_seconds={:.6} full_pages={} sliding_pages={} sliding_bytes={} total_bytes={}",
        total_elapsed.as_secs_f64(),
        prefill_elapsed.as_secs_f64(),
        decode_elapsed.as_secs_f64(),
        manager_elapsed.as_secs_f64(),
        full.resident_pages,
        sliding.resident_pages,
        sliding.resident_bytes,
        total_resident_bytes,
    );
    ResidenceArmResult {
        generated_tokens: generated_tokens.into_boxed_slice(),
        total_elapsed,
        prefill_elapsed,
        decode_elapsed,
        manager_elapsed,
        full_resident_pages: full.resident_pages,
        sliding_resident_pages: sliding.resident_pages,
        total_resident_bytes,
        sliding_resident_bytes: sliding.resident_bytes,
    }
}

fn benchmark_epochs() -> usize {
    let epochs: usize = std::env::var("ORBITKV_RESIDENCE_BENCH_EPOCHS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    assert!((4..=30).contains(&epochs) && epochs.is_multiple_of(2));
    epochs
}

fn residence_decode_tokens() -> usize {
    let tokens = std::env::var("ORBITKV_RESIDENCE_DECODE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);
    assert!((HYBRID_REFERENCE_TOKENS.len()..=512).contains(&tokens));
    tokens
}

fn print_released_residence_summary(
    compile_elapsed: Duration,
    generated_tokens: usize,
    compiled: &mut [ResidenceArmResult],
    conservative: &mut [ResidenceArmResult],
) {
    assert_eq!(compiled.len(), conservative.len());
    let reference = &compiled[0].generated_tokens;
    assert!(
        compiled
            .iter()
            .chain(conservative.iter())
            .all(|result| result.generated_tokens == *reference)
    );
    assert_eq!(
        &reference[..HYBRID_STABLE_REFERENCE_PREFIX],
        &HYBRID_REFERENCE_TOKENS[..HYBRID_STABLE_REFERENCE_PREFIX]
    );
    let mut compiled_total = compiled
        .iter()
        .map(|result| result.total_elapsed)
        .collect::<Vec<_>>();
    let mut conservative_total = conservative
        .iter()
        .map(|result| result.total_elapsed)
        .collect::<Vec<_>>();
    let mut compiled_prefill = compiled
        .iter()
        .map(|result| result.prefill_elapsed)
        .collect::<Vec<_>>();
    let mut conservative_prefill = conservative
        .iter()
        .map(|result| result.prefill_elapsed)
        .collect::<Vec<_>>();
    let mut compiled_decode = compiled
        .iter()
        .map(|result| result.decode_elapsed)
        .collect::<Vec<_>>();
    let mut conservative_decode = conservative
        .iter()
        .map(|result| result.decode_elapsed)
        .collect::<Vec<_>>();
    let mut compiled_manager = compiled
        .iter()
        .map(|result| result.manager_elapsed)
        .collect::<Vec<_>>();
    let mut conservative_manager = conservative
        .iter()
        .map(|result| result.manager_elapsed)
        .collect::<Vec<_>>();
    let compiled_median = median_duration(&mut compiled_total);
    let conservative_median = median_duration(&mut conservative_total);
    let improvements = compiled
        .iter()
        .zip(conservative.iter())
        .map(|(compiled, conservative)| {
            conservative.total_elapsed.as_secs_f64() - compiled.total_elapsed.as_secs_f64()
        })
        .collect::<Vec<_>>();
    let sample_count = f64::from(u32::try_from(improvements.len()).unwrap());
    let improvement_mean = improvements.iter().sum::<f64>() / sample_count;
    let improvement_variance = improvements
        .iter()
        .map(|sample| (sample - improvement_mean).powi(2))
        .sum::<f64>()
        / f64::from(u32::try_from(improvements.len() - 1).unwrap());
    let margin = student_t_975(improvements.len()) * (improvement_variance / sample_count).sqrt();
    let compiled_sample = &compiled[0];
    let conservative_sample = &conservative[0];
    assert_eq!(
        compiled_sample.full_resident_pages,
        conservative_sample.full_resident_pages
    );
    assert!(compiled_sample.sliding_resident_pages < conservative_sample.sliding_resident_pages);
    assert!(compiled_sample.total_resident_bytes < conservative_sample.total_resident_bytes);
    eprintln!(
        "released residence summary: compile_seconds={:.3} paired_epochs={} generated_tokens={} compiled_total_median_seconds={:.6} request_lifetime_total_median_seconds={:.6} time_ratio={:.4} paired_mean_improvement_seconds={:.6} paired_95ci_low_seconds={:.6} paired_95ci_high_seconds={:.6} compiled_prefill_median_seconds={:.6} request_lifetime_prefill_median_seconds={:.6} compiled_decode_median_seconds={:.6} request_lifetime_decode_median_seconds={:.6} compiled_manager_median_seconds={:.6} request_lifetime_manager_median_seconds={:.6} compiled_full_pages={} compiled_sliding_pages={} request_lifetime_sliding_pages={} compiled_sliding_bytes={} request_lifetime_sliding_bytes={} compiled_total_bytes={} request_lifetime_total_bytes={}",
        compile_elapsed.as_secs_f64(),
        compiled.len(),
        generated_tokens,
        compiled_median.as_secs_f64(),
        conservative_median.as_secs_f64(),
        compiled_median.as_secs_f64() / conservative_median.as_secs_f64(),
        improvement_mean,
        improvement_mean - margin,
        improvement_mean + margin,
        median_duration(&mut compiled_prefill).as_secs_f64(),
        median_duration(&mut conservative_prefill).as_secs_f64(),
        median_duration(&mut compiled_decode).as_secs_f64(),
        median_duration(&mut conservative_decode).as_secs_f64(),
        median_duration(&mut compiled_manager).as_secs_f64(),
        median_duration(&mut conservative_manager).as_secs_f64(),
        compiled_sample.full_resident_pages,
        compiled_sample.sliding_resident_pages,
        conservative_sample.sliding_resident_pages,
        compiled_sample.sliding_resident_bytes,
        conservative_sample.sliding_resident_bytes,
        compiled_sample.total_resident_bytes,
        conservative_sample.total_resident_bytes,
    );
}

fn student_t_975(samples: usize) -> f64 {
    const CRITICAL: [f64; 30] = [
        f64::INFINITY,
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
    ];
    CRITICAL[samples - 1]
}

fn qualify_released_capacity(config_bytes: &[u8]) {
    let page_counts = [35_u32, 33_u32];
    let mut compiled = prepare_model_run_with_page_counts(
        config_bytes,
        512,
        PhysicalResidencePolicy::Compiled,
        Some(&page_counts),
    );
    let mut conservative = prepare_model_run_with_page_counts(
        config_bytes,
        512,
        PhysicalResidencePolicy::RequestLifetime,
        Some(&page_counts),
    );
    complete(
        &mut compiled.session,
        &compiled.prepared,
        &compiled.arenas,
        1,
    );
    complete(
        &mut conservative.session,
        &conservative.prepared,
        &conservative.arenas,
        1,
    );
    for offset in 1..=48_u64 {
        let (_, prepared, _) = prepare_decode_step(&mut compiled, 512 + offset);
        complete(
            &mut compiled.session,
            &prepared,
            &compiled.arenas,
            offset + 1,
        );
    }
    for offset in 1..=16_u64 {
        let (_, prepared, _) = prepare_decode_step(&mut conservative, 512 + offset);
        complete(
            &mut conservative.session,
            &prepared,
            &conservative.arenas,
            offset + 1,
        );
    }
    assert_eq!(
        compiled.session.prepare_append_batch(&[EngineAppendIntent {
            request_id: EngineRequestId(1),
            target_boundary: 561,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    assert_eq!(
        conservative
            .session
            .prepare_append_batch(&[EngineAppendIntent {
                request_id: EngineRequestId(1),
                target_boundary: 529,
            }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    assert_eq!(compiled.session.arena_stats()[1].resident_pages, 32);
    assert_eq!(conservative.session.arena_stats()[1].resident_pages, 33);
    release_and_drain(&mut compiled.session, EngineRequestId(1));
    release_and_drain(&mut conservative.session, EngineRequestId(1));
    eprintln!(
        "released residence capacity: full_pages=35 sliding_pages=33 compiled_max_boundary=560 request_lifetime_max_boundary=528 final_drains=2"
    );
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
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2
    } else {
        samples[middle]
    }
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

fn batch_decoder_class_steps<'a>(
    write_slots: &'a [Vec<u64>],
    attention: &'a [AttentionBatch],
) -> Vec<DecoderClassStep<'a>> {
    write_slots
        .iter()
        .zip(attention)
        .map(|(write_slots, attention)| DecoderClassStep {
            class_id: attention.class_id,
            write_slots,
            attention,
        })
        .collect()
}

fn batch_write_slots(prepared: &PreparedBatch, class_count: usize) -> Vec<Vec<u64>> {
    let mut write_slots = vec![Vec::new(); class_count];
    for step in prepared.steps() {
        assert_eq!(step.classes.len(), class_count);
        for (class_index, class) in step.classes.iter().enumerate() {
            assert_eq!(usize::from(class.class_id), class_index);
            write_slots[class_index].extend(&class.write_slots);
        }
    }
    write_slots
}

fn row_logits(logits: &[f32], row: usize, vocabulary_size: usize) -> &[f32] {
    &logits[row * vocabulary_size..(row + 1) * vocabulary_size]
}

fn top_two(logits: &[f32]) -> [(u32, f32); 2] {
    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token, value)| (u32::try_from(token).unwrap(), value))
        .collect::<Vec<_>>();
    ranked.select_nth_unstable_by(1, |left, right| right.1.total_cmp(&left.1));
    let mut top = [ranked[0], ranked[1]];
    top.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    top
}

fn maximum_absolute_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .max_by(f32::total_cmp)
        .unwrap()
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
fn released_checkpoint_bounds_single_and_multi_request_logits() {
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let prompt = std::iter::once(2_u32)
        .chain((0..31).map(|index| 100 + ((2 * 97 + index * 13) % 5_000)))
        .collect::<Vec<_>>();
    let prompt_positions = (0..u32::try_from(prompt.len()).unwrap()).collect::<Vec<_>>();
    let batch_size = 8_usize;
    let mut run = model_batch_harness(&config_bytes, batch_size + 1, prompt.len() * batch_size);
    let context = luminal_cuda_lite::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let compile = DecoderCompileConfig {
        maximum_query_tokens: 1_024,
        representative_prefill_tokens: 512,
        maximum_batch_size: batch_size,
        maximum_context_pages: 512,
        representative_context_pages: 32,
        search_graphs: search_graphs(),
        search_seed: 7,
    };
    let mut decoder = CompiledDecoder::compile(
        &config,
        &run.executor_plan,
        &run.arenas,
        &stream,
        &[model_dir.join("model.safetensors")],
        compile,
    )
    .unwrap();

    let (single_attention, single_prepared) =
        prepare_model_batch(&mut run, &[EngineRequestId(1)], prompt.len() as u64);
    let single_slots = batch_write_slots(&single_prepared, single_attention.len());
    let single_classes = batch_decoder_class_steps(&single_slots, &single_attention);
    let single_prefill = decoder
        .execute_with_logits(DecoderStep {
            tokens: &prompt,
            positions: &prompt_positions,
            classes: &single_classes,
        })
        .unwrap();
    complete(&mut run.session, &single_prepared, &run.arenas, 1);

    let batch_ids = (2..=u64::try_from(batch_size + 1).unwrap())
        .map(EngineRequestId)
        .collect::<Vec<_>>();
    let (batch_attention, batch_prepared) =
        prepare_model_batch(&mut run, &batch_ids, prompt.len() as u64);
    let batch_slots = batch_write_slots(&batch_prepared, batch_attention.len());
    let batch_classes = batch_decoder_class_steps(&batch_slots, &batch_attention);
    let batch_tokens = prompt.repeat(batch_size);
    let batch_positions = prompt_positions.repeat(batch_size);
    let batch_prefill = decoder
        .execute_with_logits(DecoderStep {
            tokens: &batch_tokens,
            positions: &batch_positions,
            classes: &batch_classes,
        })
        .unwrap();
    complete(&mut run.session, &batch_prepared, &run.arenas, 2);

    let single_row = row_logits(
        &single_prefill.logits,
        prompt.len() - 1,
        config.vocabulary_size,
    );
    let batch_first_row = row_logits(
        &batch_prefill.logits,
        prompt.len() - 1,
        config.vocabulary_size,
    );
    for row in 1..batch_size {
        assert_eq!(
            batch_first_row,
            row_logits(
                &batch_prefill.logits,
                (row + 1) * prompt.len() - 1,
                config.vocabulary_size,
            )
        );
    }
    let single_token = single_prefill.token_ids[prompt.len() - 1];
    let batch_token = batch_prefill.token_ids[prompt.len() - 1];
    for row in 1..batch_size {
        assert_eq!(
            batch_token,
            batch_prefill.token_ids[(row + 1) * prompt.len() - 1]
        );
    }
    let single_top = top_two(single_row);
    let batch_top = top_two(batch_first_row);
    let mut maximum_difference = maximum_absolute_difference(single_row, batch_first_row);
    let mut argmax_mismatches = usize::from(single_token != batch_token);
    eprintln!(
        "batch parity token=0 single_top={single_top:?} batch_top={batch_top:?} single_margin={} batch_margin={} max_abs={}",
        single_top[0].1 - single_top[1].1,
        batch_top[0].1 - batch_top[1].1,
        maximum_difference,
    );
    let teacher_tokens = [
        506_u32, 236_743, 236_778, 236_771, 236_778, 236_800, 236_772, 236_778, 236_771, 236_778,
        236_812, 13_434, 1_051, 236_761, 108, 818,
    ];

    for generated in 1..16 {
        let target = u64::try_from(prompt.len() + generated).unwrap();
        let position = u32::try_from(prompt.len() + generated - 1).unwrap();
        let input_token = teacher_tokens[generated - 1];
        let (single_attention, single_prepared) =
            prepare_model_batch(&mut run, &[EngineRequestId(1)], target);
        let single_slots = batch_write_slots(&single_prepared, single_attention.len());
        let single_classes = batch_decoder_class_steps(&single_slots, &single_attention);
        let single_output = decoder
            .execute_with_logits(DecoderStep {
                tokens: &[input_token],
                positions: &[position],
                classes: &single_classes,
            })
            .unwrap();
        complete(
            &mut run.session,
            &single_prepared,
            &run.arenas,
            u64::try_from(generated * 2 + 1).unwrap(),
        );

        let (batch_attention, batch_prepared) = prepare_model_batch(&mut run, &batch_ids, target);
        let batch_slots = batch_write_slots(&batch_prepared, batch_attention.len());
        let batch_classes = batch_decoder_class_steps(&batch_slots, &batch_attention);
        let batch_input = vec![input_token; batch_size];
        let batch_position = vec![position; batch_size];
        let batch_output = decoder
            .execute_with_logits(DecoderStep {
                tokens: &batch_input,
                positions: &batch_position,
                classes: &batch_classes,
            })
            .unwrap();
        complete(
            &mut run.session,
            &batch_prepared,
            &run.arenas,
            u64::try_from(generated * 2 + 2).unwrap(),
        );

        let single_logits = row_logits(&single_output.logits, 0, config.vocabulary_size);
        let batch_first = row_logits(&batch_output.logits, 0, config.vocabulary_size);
        for row in 1..batch_size {
            assert_eq!(
                batch_first,
                row_logits(&batch_output.logits, row, config.vocabulary_size)
            );
        }
        let single_token = single_output.token_ids[0];
        let batch_token = batch_output.token_ids[0];
        assert!(
            batch_output
                .token_ids
                .iter()
                .all(|&token| token == batch_token)
        );
        let single_top = top_two(single_logits);
        let batch_top = top_two(batch_first);
        let difference = maximum_absolute_difference(single_logits, batch_first);
        maximum_difference = maximum_difference.max(difference);
        argmax_mismatches += usize::from(single_token != batch_token);
        if single_token != batch_token {
            let single_gap = single_logits[usize::try_from(single_token).unwrap()]
                - single_logits[usize::try_from(batch_token).unwrap()];
            let batch_gap = batch_first[usize::try_from(batch_token).unwrap()]
                - batch_first[usize::try_from(single_token).unwrap()];
            assert!(single_gap <= 2.0 * difference && batch_gap <= 2.0 * difference);
        }
        eprintln!(
            "batch parity token={generated} single_top={single_top:?} batch_top={batch_top:?} single_margin={} batch_margin={} max_abs={}",
            single_top[0].1 - single_top[1].1,
            batch_top[0].1 - batch_top[1].1,
            difference,
        );
    }

    let mut request_ids = vec![EngineRequestId(1)];
    request_ids.extend(&batch_ids);
    release_requests_and_drain(&mut run.session, &request_ids);
    eprintln!(
        "batch parity summary: maximum_absolute_logit_difference={maximum_difference} argmax_mismatches={argmax_mismatches}"
    );
    assert!(maximum_difference <= 1.0);
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
fn released_checkpoint_compares_compiled_and_request_lifetime_residence() {
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let prompt_tokens = 512_usize;
    let prompt = (0..u32::try_from(prompt_tokens).unwrap()).collect::<Vec<_>>();
    let positions = (0..u32::try_from(prompt_tokens).unwrap()).collect::<Vec<_>>();
    let compiled = prepare_model_run_with_residence(
        &config_bytes,
        prompt_tokens,
        PhysicalResidencePolicy::Compiled,
    );
    let conservative = prepare_model_run_with_residence(
        &config_bytes,
        prompt_tokens,
        PhysicalResidencePolicy::RequestLifetime,
    );
    assert_eq!(compiled.executor_plan, conservative.executor_plan);
    assert_eq!(compiled.attention, conservative.attention);
    let (mut decoder, compile_elapsed) =
        compile_hybrid_decoder(&config, &compiled, &model_dir, prompt_tokens);
    eprintln!(
        "released residence benchmark: one-time compile_seconds={:.3}",
        compile_elapsed.as_secs_f64(),
    );
    drop((compiled, conservative));
    let generated_token_count = residence_decode_tokens();
    let warmup_compiled = execute_released_residence_arm(
        PhysicalResidencePolicy::Compiled,
        &config_bytes,
        &mut decoder,
        &prompt,
        &positions,
        generated_token_count,
    );
    let warmup_conservative = execute_released_residence_arm(
        PhysicalResidencePolicy::RequestLifetime,
        &config_bytes,
        &mut decoder,
        &prompt,
        &positions,
        generated_token_count,
    );
    assert_eq!(
        warmup_compiled.generated_tokens,
        warmup_conservative.generated_tokens
    );

    let epochs = benchmark_epochs();
    let mut compiled_outputs = Vec::with_capacity(epochs / 2);
    let mut conservative_outputs = Vec::with_capacity(epochs / 2);
    for epoch in 0..epochs {
        let policies = if epoch.is_multiple_of(2) {
            [
                PhysicalResidencePolicy::Compiled,
                PhysicalResidencePolicy::RequestLifetime,
            ]
        } else {
            [
                PhysicalResidencePolicy::RequestLifetime,
                PhysicalResidencePolicy::Compiled,
            ]
        };
        for policy in policies {
            let result = execute_released_residence_arm(
                policy,
                &config_bytes,
                &mut decoder,
                &prompt,
                &positions,
                generated_token_count,
            );
            match policy {
                PhysicalResidencePolicy::Compiled => compiled_outputs.push(result),
                PhysicalResidencePolicy::RequestLifetime => conservative_outputs.push(result),
            }
        }
    }
    print_released_residence_summary(
        compile_elapsed,
        generated_token_count,
        &mut compiled_outputs,
        &mut conservative_outputs,
    );
    qualify_released_capacity(&config_bytes);
}

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR, a CUDA device, and FlashInfer headers"]
fn released_hybrid_checkpoint_crosses_window_and_drains() {
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
        &HYBRID_REFERENCE_TOKENS,
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
