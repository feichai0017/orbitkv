//! Test-only full-decoder layer boundary diagnostics across bucket geometries.

use super::*;
use crate::model::{AttentionProviderPolicy, DecoderStorage, DecoderTuningProfile};
use orbitkv::{
    CacheSharingPolicy, EngineCompletionEvidence, EnginePublicationEvidence, EngineReleaseEvidence,
    EngineReleaseOutcome, StateCheckpointPool, compile_hf_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
};
use orbitkv_cuda::cudarc::driver::CudaContext;
use std::{io::Write, path::PathBuf};

const PAGE_TOKENS: u64 = 16;
const PAGE_COUNT: u32 = 16;

#[derive(Clone, Copy)]
enum LayerTraceMode {
    RepeatedBatch,
    ChunkedPrefill,
}

#[test]
#[ignore = "requires CUDA and the Qwen3.8 checkpoint"]
#[allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]
fn repeated_prefill_reports_first_cross_bucket_layer_divergence() {
    run_layer_trace(LayerTraceMode::RepeatedBatch);
}

#[test]
#[ignore = "requires CUDA and the Qwen3.8 checkpoint"]
fn chunked_prefill_reports_first_layer_boundary_divergence() {
    run_layer_trace(LayerTraceMode::ChunkedPrefill);
}

#[allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]
fn run_layer_trace(mode: LayerTraceMode) {
    let model = PathBuf::from(std::env::var_os("ORBITKV_MODEL_DIR").unwrap());
    let config_bytes = std::fs::read(model.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let manifest = compile_hf_runtime_manifest(
        &config_bytes,
        orbitkv::HfRetentionOptions {
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
            maximum_requests: 8,
            maximum_operations: 8,
            maximum_prefixes: 1,
            maximum_reclamations: registrations.iter().map(|arena| arena.page_count).sum(),
            maximum_step_tokens: 32,
        },
        &registrations,
    )
    .unwrap();
    let plan = ExecutorPlan::compile(&manifest).unwrap();
    let fixed_pools = plan
        .fixed_states
        .iter()
        .enumerate()
        .map(|(index, class)| {
            let (slots_per_request, bytes_per_request) = match class.storage {
                crate::FixedStateStorage::Recurrent {
                    slots_per_request,
                    bytes_per_request,
                    ..
                }
                | crate::FixedStateStorage::Convolution {
                    slots_per_request,
                    bytes_per_request,
                    ..
                } => (slots_per_request, bytes_per_request),
            };
            let offset = u32::try_from(index + 1).unwrap();
            (
                class.state_id,
                StateCheckpointPool::new(
                    manager.arena_stats()[0].engine_epoch,
                    manager.arena_stats()[0].pool_epoch + u64::from(offset),
                    manager.arena_stats()[0].pool_id + offset,
                    bytes_per_request / u64::from(slots_per_request),
                    8 * slots_per_request,
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let mut session =
        RuntimeSession::with_fixed_states(manager, CacheSharingPolicy::RequestPrivate, fixed_pools)
            .unwrap();
    let arenas = session
        .arena_stats()
        .iter()
        .copied()
        .zip(registrations)
        .map(|(stats, registration)| ExecutorArena::bind(stats, registration).unwrap())
        .collect::<Vec<_>>();
    let fixed_identities = session
        .fixed_state_stats()
        .iter()
        .map(|(state_id, stats)| (*state_id, stats.identity))
        .collect::<Vec<_>>();
    let mut weights = std::fs::read_dir(&model)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|value| value == "safetensors"))
        .collect::<Vec<_>>();
    weights.sort();
    let layer = std::env::var("ORBITKV_LAYER_TRACE_LAYER")
        .map_or(config.layers - 1, |value| value.parse::<usize>().unwrap());
    let boundary = match std::env::var("ORBITKV_LAYER_TRACE_BOUNDARY")
        .as_deref()
        .unwrap_or("output")
    {
        "input" => DecoderLayerDiagnosticBoundary::Input,
        "normalized" => DecoderLayerDiagnosticBoundary::Normalized,
        "state" => DecoderLayerDiagnosticBoundary::State,
        "residual" => DecoderLayerDiagnosticBoundary::Residual,
        "feed_forward_normalized" => DecoderLayerDiagnosticBoundary::FeedForwardNormalized,
        "gate" => DecoderLayerDiagnosticBoundary::Gate,
        "up" => DecoderLayerDiagnosticBoundary::Up,
        "activated" => DecoderLayerDiagnosticBoundary::Activated,
        "product" => DecoderLayerDiagnosticBoundary::Product,
        "down" => DecoderLayerDiagnosticBoundary::Down,
        "add_operands" => DecoderLayerDiagnosticBoundary::AddOperands,
        "add_operands_and_output" => DecoderLayerDiagnosticBoundary::AddOperandsAndOutput,
        "input_and_output" => DecoderLayerDiagnosticBoundary::InputAndOutput,
        "attention_normalized" => DecoderLayerDiagnosticBoundary::AttentionNormalized,
        "attention_q" => DecoderLayerDiagnosticBoundary::AttentionQ,
        "attention_k" => DecoderLayerDiagnosticBoundary::AttentionK,
        "attention_v" => DecoderLayerDiagnosticBoundary::AttentionV,
        "attention_readout" => DecoderLayerDiagnosticBoundary::AttentionReadout,
        "attention_projected" => DecoderLayerDiagnosticBoundary::AttentionProjected,
        "attention_readout_and_projected" => {
            DecoderLayerDiagnosticBoundary::AttentionReadoutAndProjected
        }
        "output" => DecoderLayerDiagnosticBoundary::Output,
        value => panic!("unknown diagnostic boundary {value:?}"),
    };
    assert!(layer < config.layers);
    let compile = DecoderCompileConfig {
        output_rows: DecoderOutputRows::AllTokens,
        maximum_query_tokens: 32,
        representative_prefill_tokens: 4,
        maximum_batch_size: 8,
        maximum_context_pages: 8,
        representative_context_pages: 8,
        search_graphs: 1,
        search_seed: 7,
    };
    let attention_provider = match std::env::var("ORBITKV_LAYER_TRACE_PROVIDER")
        .as_deref()
        .unwrap_or("all")
    {
        "all" => AttentionProviderPolicy::All,
        "flashattention" => AttentionProviderPolicy::FlashAttention,
        "flashinfer" => AttentionProviderPolicy::FlashInfer,
        value => panic!("unknown attention provider {value:?}"),
    };
    let (batch_sizes, prefill_tokens, context_pages, maximum_buckets) = match mode {
        LayerTraceMode::RepeatedBatch => (vec![1, 8], vec![4], vec![], 8),
        // Reproduce the ragged trace's b/s/c combinations rather than relying
        // on independent intervals whose one representative may not satisfy
        // the provider's correlated metadata guard.
        LayerTraceMode::ChunkedPrefill => (vec![1, 4, 8], vec![8, 24, 32], vec![4, 8], 32),
    };
    let tuning = DecoderTuningProfile {
        batch_sizes,
        prefill_tokens,
        context_pages,
        keep_best: 1,
        initial_candidates: 1,
        hotspot_candidates: 0,
        trials: 1,
        maximum_buckets,
        attention_provider,
        ..DecoderTuningProfile::default()
    };
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut decoder = CompiledDecoder::compile_with_layer_outputs_for_debug(
        &config,
        &plan,
        DecoderStorage::new(&arenas, &fixed_identities),
        &stream,
        &weights,
        compile,
        &tuning,
        layer,
        boundary,
    )
    .unwrap();

    // Same fixed token IDs as case 0 of the frozen full-model oracle.
    let tokens = [238_547_u32, 128_132, 99_790, 161_321];
    let (single, batch) = match mode {
        LayerTraceMode::RepeatedBatch => (
            run_prefill(&mut decoder, &mut session, &plan, &arenas, &[1], &tokens, 1),
            run_prefill(
                &mut decoder,
                &mut session,
                &plan,
                &arenas,
                &(2..=9).collect::<Vec<_>>(),
                &tokens.repeat(8),
                2,
            ),
        ),
        LayerTraceMode::ChunkedPrefill => {
            let full = run_prefill(
                &mut decoder,
                &mut session,
                &plan,
                &arenas,
                &(1..=8).collect::<Vec<_>>(),
                &tokens.repeat(8),
                1,
            );
            let chunked =
                run_chunked_prefill(&mut decoder, &mut session, &plan, &arenas, &tokens, 2);
            (full, chunked)
        }
    };
    if matches!(mode, LayerTraceMode::ChunkedPrefill) {
        let report = compare_chunked_prefill(layer, boundary, &config, &single, &batch);
        write_report(&report);
        println!("{report}");
        return;
    }
    assert_eq!(single.len() % tokens.len(), 0);
    let segment_elements = match boundary {
        DecoderLayerDiagnosticBoundary::AddOperands => vec![
            tokens.len() * config.hidden_size,
            tokens.len() * config.hidden_size,
        ],
        DecoderLayerDiagnosticBoundary::AddOperandsAndOutput => {
            vec![tokens.len() * config.hidden_size; 3]
        }
        DecoderLayerDiagnosticBoundary::InputAndOutput => {
            vec![tokens.len() * config.hidden_size; 2]
        }
        DecoderLayerDiagnosticBoundary::AttentionReadoutAndProjected => vec![
            tokens.len() * config.query_heads * config.head_dim,
            tokens.len() * config.hidden_size,
        ],
        _ => vec![single.len()],
    };
    assert_eq!(segment_elements.iter().sum::<usize>(), single.len());
    assert_eq!(batch.len(), single.len() * 8);
    let mut differing = 0_usize;
    let mut maximum = 0.0_f32;
    let mut replicas_differing = Vec::new();
    let mut segment_differences = Vec::new();
    for replica in 0..8 {
        let mut replica_differing = 0;
        let mut single_offset = 0;
        let mut batch_offset = 0;
        for &elements in &segment_elements {
            let expected = &single[single_offset..single_offset + elements];
            let actual =
                &batch[batch_offset + replica * elements..batch_offset + (replica + 1) * elements];
            let part_differing = actual
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| actual != expected)
                .count();
            replica_differing += part_differing;
            segment_differences.push(part_differing);
            for (&actual, &expected) in actual.iter().zip(expected) {
                maximum = maximum.max((actual - expected).abs());
            }
            single_offset += elements;
            batch_offset += elements * 8;
        }
        differing += replica_differing;
        replicas_differing.push(replica_differing);
    }
    let report = serde_json::json!({
        "schema": 1,
        "layer": layer,
        "kind": format!("{:?}", config.layer_kind(layer)),
        "boundary": format!("{boundary:?}"),
        "attention_provider": format!("{attention_provider:?}"),
        "tokens_per_request": tokens.len(),
        "elements_per_request": single.len(),
        "batch_size": 8,
        "differing_elements": differing,
        "maximum_absolute_error": maximum,
        "replica_differing_elements": replicas_differing,
        "replica_operand_differing_elements": segment_differences,
    });
    write_report(&report);
    println!("{report}");
}

#[allow(clippy::too_many_lines)]
fn run_prefill(
    decoder: &mut CompiledDecoder,
    session: &mut RuntimeSession,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    request_ids: &[u64],
    tokens: &[u32],
    completion_value: u64,
) -> Box<[f32]> {
    let request_ids = request_ids
        .iter()
        .copied()
        .map(EngineRequestId)
        .collect::<Vec<_>>();
    session.acquire_requests(&request_ids).unwrap();
    let tokens_per_request = tokens.len() / request_ids.len();
    assert_eq!(tokens_per_request * request_ids.len(), tokens.len());
    let queries = request_ids
        .iter()
        .enumerate()
        .map(|(index, &request_id)| LayerQuery {
            request_id,
            start: 0,
            tokens: &tokens[index * tokens_per_request..(index + 1) * tokens_per_request],
        })
        .collect::<Vec<_>>();
    let output =
        execute_layer_submission(decoder, session, plan, arenas, &queries, completion_value);
    release_layer_requests(session, &request_ids);
    output
}

#[derive(Clone, Copy)]
struct LayerQuery<'a> {
    request_id: EngineRequestId,
    start: usize,
    tokens: &'a [u32],
}

fn run_chunked_prefill(
    decoder: &mut CompiledDecoder,
    session: &mut RuntimeSession,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    tokens: &[u32; 4],
    completion_value: u64,
) -> Box<[f32]> {
    let request_ids = (101..=108).map(EngineRequestId).collect::<Vec<_>>();
    session.acquire_requests(&request_ids[..4]).unwrap();
    let first_queries = request_ids[..4]
        .iter()
        .map(|&request_id| LayerQuery {
            request_id,
            start: 0,
            tokens: &tokens[..2],
        })
        .collect::<Vec<_>>();
    let first = execute_layer_submission(
        decoder,
        session,
        plan,
        arenas,
        &first_queries,
        completion_value,
    );

    session.acquire_requests(&request_ids[4..]).unwrap();
    let second_queries = request_ids
        .iter()
        .enumerate()
        .map(|(index, &request_id)| LayerQuery {
            request_id,
            start: if index < 4 { 2 } else { 0 },
            tokens: if index < 4 { &tokens[2..] } else { &tokens[..] },
        })
        .collect::<Vec<_>>();
    let second = execute_layer_submission(
        decoder,
        session,
        plan,
        arenas,
        &second_queries,
        completion_value + 1,
    );
    release_layer_requests(session, &request_ids);
    reconstruct_chunked_rows(&first, &second)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn execute_layer_submission(
    decoder: &mut CompiledDecoder,
    session: &mut RuntimeSession,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    queries: &[LayerQuery<'_>],
    completion_value: u64,
) -> Box<[f32]> {
    let tokens = queries
        .iter()
        .flat_map(|query| query.tokens.iter().copied())
        .collect::<Vec<_>>();
    let source = session
        .prepare_append_batch(
            &queries
                .iter()
                .map(|query| EngineAppendIntent {
                    request_id: query.request_id,
                    target_boundary: u64::try_from(query.start + query.tokens.len()).unwrap(),
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let view = session.prepared_execution_view(source.batch_id).unwrap();
    let attention = plan.attention_batches(&view).unwrap();
    let prepared = plan.lower_prepared(source, arenas).unwrap();
    let write_slots = batch_write_slots(&prepared, attention.len());
    let classes = write_slots
        .iter()
        .zip(&attention)
        .map(|(write_slots, attention)| DecoderClassStep {
            class_id: attention.class_id,
            write_slots,
            attention,
        })
        .collect::<Vec<_>>();
    let states = prepared
        .fixed_state_requests()
        .map(|(request_id, states)| DecoderFixedStateStep { request_id, states })
        .collect::<Vec<_>>();
    let positions = queries
        .iter()
        .flat_map(|query| query.start..query.start + query.tokens.len())
        .map(|position| u32::try_from(position).unwrap())
        .collect::<Vec<_>>();
    let output = decoder
        .execute_with_fixed_states_and_layer_outputs(
            DecoderStep {
                tokens: &tokens,
                positions: &positions,
                classes: &classes,
            },
            &states,
        )
        .unwrap();
    let evidence = prepared
        .execution_evidence_after_state_success(arenas, &output.fixed_states)
        .unwrap();
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
                .map(|retirement| orbitkv::EngineRetirementEvidence {
                    page: retirement.page,
                    backend_domain: retirement.backend_domain,
                    acknowledged: true,
                    backend_index: retirement.backend_index,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
        .unwrap();
    output.hidden
}

fn release_layer_requests(session: &mut RuntimeSession, request_ids: &[EngineRequestId]) {
    let release = session.prepare_release_batch(request_ids).unwrap();
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: release
                .retirements
                .iter()
                .map(|retirement| orbitkv::EngineRetirementEvidence {
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
}

fn reconstruct_chunked_rows(first: &[f32], second: &[f32]) -> Box<[f32]> {
    assert_eq!(first.len() % 8, 0);
    let width = first.len() / 8;
    assert_eq!(second.len(), 24 * width);
    let mut output = Vec::with_capacity(32 * width);
    for request in 0..4 {
        output.extend_from_slice(&first[request * 2 * width..(request + 1) * 2 * width]);
        output.extend_from_slice(&second[request * 2 * width..(request + 1) * 2 * width]);
    }
    output.extend_from_slice(&second[8 * width..]);
    output.into_boxed_slice()
}

#[allow(clippy::float_cmp)]
fn compare_chunked_prefill(
    layer: usize,
    boundary: DecoderLayerDiagnosticBoundary,
    config: &DecoderConfig,
    full: &[f32],
    chunked: &[f32],
) -> serde_json::Value {
    assert_eq!(full.len(), chunked.len());
    let differing = full
        .iter()
        .zip(chunked)
        .filter(|(expected, actual)| expected != actual)
        .count();
    let maximum = full
        .iter()
        .zip(chunked)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    serde_json::json!({
        "schema": 1, "mode": "chunked_prefill", "layer": layer,
        "kind": format!("{:?}", config.layer_kind(layer)),
        "boundary": format!("{boundary:?}"), "full_requests": 8,
        "prompt_tokens": 4, "chunks": [2, 2],
        "differing_elements": differing, "maximum_absolute_error": maximum,
    })
}

fn write_report(report: &serde_json::Value) {
    if let Some(output) = std::env::var_os("ORBITKV_LAYER_TRACE_OUTPUT")
        && !std::path::Path::new(&output).exists()
    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap();
        file.write_all(&serde_json::to_vec_pretty(report).unwrap())
            .unwrap();
    }
}

fn batch_write_slots(prepared: &crate::PreparedBatch, class_count: usize) -> Vec<Vec<u64>> {
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
