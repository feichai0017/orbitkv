//! Device qualification for packed multi-request token and fixed-state execution.

use super::*;
use orbitkv_executor::model::{DecoderFixedStateStep, DecoderStorage, DecoderTuningProfile};

fn batch_capacity(batch: usize, requested: Option<usize>) -> usize {
    assert!((1..=16).contains(&batch));
    let capacity = requested.unwrap_or(batch);
    assert!((batch..=16).contains(&capacity));
    capacity
}

#[allow(clippy::too_many_lines)]
pub(super) fn run() {
    let batch = std::env::var("ORBITKV_QUALIFICATION_BATCH_SIZE")
        .map_or(4, |value| value.parse::<usize>().unwrap());
    let capacity = batch_capacity(
        batch,
        std::env::var("ORBITKV_QUALIFICATION_BATCH_CAPACITY")
            .ok()
            .map(|value| value.parse::<usize>().unwrap()),
    );
    let ragged = std::env::var("ORBITKV_QUALIFICATION_RAGGED").is_ok_and(|value| value == "1");
    assert!(!ragged || batch > 1);
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let mut harness = model_harness(
        &config_bytes,
        capacity,
        capacity,
        capacity * 4,
        PhysicalResidencePolicy::Compiled,
        None,
    );
    let requests = (1..=batch)
        .map(|id| EngineRequestId(u64::try_from(id).unwrap()))
        .collect::<Vec<_>>();
    harness.session.acquire_requests(&requests).unwrap();
    let context = luminal_cuda_lite::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let compile = DecoderCompileConfig {
        maximum_query_tokens: capacity * 4,
        representative_prefill_tokens: capacity * 4,
        maximum_batch_size: capacity,
        maximum_context_pages: usize::try_from(PAGE_COUNT).unwrap(),
        representative_context_pages: capacity,
        search_graphs: search_graphs(),
        search_seed: 7,
    };
    let tuning = std::env::var_os("ORBITKV_TUNING_PROFILE")
        .map(|path| DecoderTuningProfile::from_json(&std::fs::read(path).unwrap()).unwrap())
        .unwrap_or_default();
    let fixed_states = fixed_state_identities(&harness.session);
    let artifact_path = std::env::var_os("ORBITKV_DECODER_ARTIFACT").map(PathBuf::from);
    let artifact = artifact_path
        .as_deref()
        .filter(|path| path.exists())
        .map(|path| DecoderArtifact::from_bytes(&std::fs::read(path).unwrap()).unwrap());
    let started = Instant::now();
    let (mut decoder, selected) = CompiledDecoder::compile_or_load_with_tuning(
        &config,
        &harness.executor_plan,
        DecoderStorage::new(&harness.arenas, &fixed_states),
        &stream,
        &weight_files(&model_dir),
        compile,
        &tuning,
        artifact.as_ref(),
    )
    .unwrap();
    let compile_seconds = started.elapsed().as_secs_f64();
    if artifact.is_none()
        && let Some(path) = &artifact_path
    {
        std::fs::write(path, selected.to_bytes().unwrap()).unwrap();
    }
    assert!(decoder.cache_updates_in_place());
    let reference = std::env::var_os("ORBITKV_REFERENCE_DIR").map(PathBuf::from);
    let mut completion = 1;
    if ragged {
        execute_batch(
            &mut decoder,
            &mut harness,
            &requests[..batch / 2],
            2,
            &[1_u32, 2].repeat(batch / 2),
            &[0_u32, 1].repeat(batch / 2),
            None,
            "ragged-warmup",
            completion,
            config.vocabulary_size,
        );
        completion += 1;
    }
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    for row in 0..batch {
        if ragged && row < batch / 2 {
            tokens.extend([3, 4]);
            positions.extend([2, 3]);
        } else {
            tokens.extend([1, 2, 3, 4]);
            positions.extend([0, 1, 2, 3]);
        }
    }
    let prefill_reference = reference.as_ref().map(|path| path.join("prefill-last.f32"));
    let mut next_tokens = execute_batch(
        &mut decoder,
        &mut harness,
        &requests,
        4,
        &tokens,
        &positions,
        prefill_reference.as_deref(),
        "prefill",
        completion,
        config.vocabulary_size,
    );
    completion += 1;
    for step in 1..8_u64 {
        let phase = if step == 1 {
            "decode".to_owned()
        } else {
            format!("decode-{step}")
        };
        let logits_reference = reference
            .as_ref()
            .map(|path| path.join(format!("{phase}.f32")));
        next_tokens = execute_batch(
            &mut decoder,
            &mut harness,
            &requests,
            4 + step,
            &next_tokens,
            &vec![u32::try_from(3 + step).unwrap(); batch],
            logits_reference.as_deref(),
            &phase,
            completion,
            config.vocabulary_size,
        );
        completion += 1;
    }
    release_requests_and_drain(&mut harness.session, &requests);
    assert!(
        harness
            .session
            .fixed_state_stats()
            .iter()
            .all(|(_, state)| {
                state.active_owners == 0
                    && state.pending_transitions == 0
                    && state.pending_retirements == 0
                    && state.free_slots == u64::from(state.identity.slot_count)
            })
    );
    eprintln!(
        "ORBITKV_DECODER_QUALIFICATION {}",
        serde_json::json!({
            "schema": 1, "batch_size": batch, "batch_capacity": capacity,
            "ragged_prefill": ragged,
            "reference_enabled": reference.is_some(),
            "parity_steps": if reference.is_some() { 8 } else { 0 },
            "parity_request_steps": if reference.is_some() { 8 * batch } else { 0 },
            "drain_passed": true, "compile_seconds": compile_seconds,
            "artifact_mode": if artifact.is_some() { "replay" } else { "search" },
        })
    );
}

#[test]
fn qualification_capacity_can_stay_fixed_across_workloads() {
    assert_eq!(batch_capacity(4, None), 4);
    assert_eq!(batch_capacity(4, Some(8)), batch_capacity(8, Some(8)));
    assert_eq!(batch_capacity(8, Some(8)), 8);
}

#[test]
fn qualification_capacity_rejects_an_unrepresentable_workload() {
    for (batch, capacity) in [(0, None), (17, None), (8, Some(4)), (4, Some(17))] {
        assert!(std::panic::catch_unwind(|| batch_capacity(batch, capacity)).is_err());
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_batch(
    decoder: &mut CompiledDecoder,
    harness: &mut ModelBatchHarness,
    requests: &[EngineRequestId],
    target_boundary: u64,
    tokens: &[u32],
    positions: &[u32],
    reference: Option<&std::path::Path>,
    phase: &str,
    completion_value: u64,
    vocabulary_size: usize,
) -> Vec<u32> {
    let (attention, prepared) = prepare_model_batch(harness, requests, target_boundary);
    let query_indptr = &attention[0].query_indptr;
    assert_eq!(query_indptr.len(), requests.len() + 1);
    assert_eq!(
        usize::try_from(*query_indptr.last().unwrap()).unwrap(),
        tokens.len()
    );
    let write_slots = batch_write_slots(&prepared, attention.len());
    let classes = batch_decoder_class_steps(&write_slots, &attention);
    let states = prepared
        .fixed_state_requests()
        .map(|(request_id, states)| DecoderFixedStateStep { request_id, states })
        .collect::<Vec<_>>();
    let started = Instant::now();
    let output = decoder
        .execute_with_fixed_states_and_logits(
            DecoderStep {
                tokens,
                positions,
                classes: &classes,
            },
            &states,
        )
        .unwrap();
    eprintln!(
        "ORBITKV_DECODER_STEP {}",
        serde_json::json!({
            "phase": phase, "batch_size": requests.len(), "query_tokens": tokens.len(),
            "query_indptr": query_indptr, "diagnostic_logits": true,
            "seconds": started.elapsed().as_secs_f64(),
        })
    );
    assert_eq!(output.token_ids.len(), tokens.len());
    assert_eq!(output.logits.len(), tokens.len() * vocabulary_size);
    let next_tokens = query_indptr[1..]
        .iter()
        .enumerate()
        .map(|(request, &end)| {
            let row = usize::try_from(end).unwrap() - 1;
            reference.map_or(output.token_ids[row], |path| {
                assert_reference_logits(
                    &format!("{phase}-request-{request}"),
                    row_logits(&output.logits, row, vocabulary_size),
                    path,
                    None,
                )
            })
        })
        .collect();
    complete_with_fixed_states(
        &mut harness.session,
        &prepared,
        &harness.arenas,
        &output.fixed_states,
        completion_value,
    );
    next_tokens
}
