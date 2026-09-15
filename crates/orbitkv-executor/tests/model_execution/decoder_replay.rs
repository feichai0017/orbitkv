//! Fixed-artifact model replay, repeated request lifecycles and bucket eviction.

use super::*;

#[allow(clippy::too_many_lines)]
pub(super) fn run(sequences: usize) {
    let stage_trace = orbitkv_executor::diagnostics::install_stage_trace_from_env().unwrap();
    let model_dir = model_directory();
    let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    let prompt = [1_u32, 2, 3, 4];
    let positions = [0_u32, 1, 2, 3];
    let mut run = prepare_model_run(&config_bytes, prompt.len());
    let context = orbitkv_cuda::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let compile = DecoderCompileConfig {
        output_rows: orbitkv_executor::model::DecoderOutputRows::AllTokens,
        maximum_query_tokens: 8,
        representative_prefill_tokens: prompt.len(),
        maximum_batch_size: 1,
        maximum_context_pages: usize::try_from(PAGE_COUNT).unwrap(),
        representative_context_pages: run.attention[0].page_indices.len(),
        search_graphs: search_graphs(),
        search_seed: 7,
    };
    let fixed_states = fixed_state_identities(&run.session);
    let artifact_path = std::env::var_os("ORBITKV_DECODER_ARTIFACT").map(PathBuf::from);
    let artifact = artifact_path
        .as_deref()
        .filter(|path| path.exists())
        .map(|path| DecoderArtifact::from_bytes(&std::fs::read(path).unwrap()).unwrap());
    let tuning = std::env::var_os("ORBITKV_TUNING_PROFILE")
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
        .unwrap_or_default();
    eprintln!(
        "ORBITKV_MODULE_ARTIFACT {}",
        serde_json::json!({
            "loaded_image_count": artifact.as_ref().map(DecoderArtifact::module_image_count),
        })
    );
    let started = Instant::now();
    let (mut decoder, selected_artifact) = CompiledDecoder::compile_or_load_with_tuning(
        &config,
        &run.executor_plan,
        orbitkv_executor::model::DecoderStorage::new(&run.arenas, &fixed_states),
        &stream,
        &weight_files(&model_dir),
        compile,
        &tuning,
        artifact.as_ref(),
    )
    .unwrap();
    if artifact.is_none()
        && let Some(path) = artifact_path
    {
        std::fs::write(path, selected_artifact.to_bytes().unwrap()).unwrap();
    }
    eprintln!(
        "Qwen3.8 27B FP8: decoder schedule ready after {:.3}s",
        started.elapsed().as_secs_f64()
    );
    let cache_updates = decoder.cache_update_buckets();
    eprintln!("Qwen3.8 27B FP8: cache updates {cache_updates:?}");
    assert!(decoder.cache_updates_in_place());
    assert!(cache_updates.iter().all(|bucket| {
        bucket.in_place_tensors == bucket.tensor_count
            && bucket.copy_back_tensors == 0
            && bucket.copy_back_bytes == 0
    }));

    let requested_capacity = std::env::var("ORBITKV_GRAPH_CACHE_CAPACITY").map_or(
        orbitkv_executor::model::DEFAULT_GRAPH_CACHE_CAPACITY,
        |value| value.parse::<std::num::NonZeroUsize>().unwrap(),
    );
    decoder.set_graph_cache_capacity(requested_capacity);
    let preparation_enabled = std::env::var_os("ORBITKV_PREPARE_EXECUTION").is_some();
    if preparation_enabled {
        let before = decoder.graph_cache_stats();
        let report = decoder.prepare_execution().unwrap();
        assert_eq!(
            report.buckets.len(),
            requested_capacity.get().min(report.compiled_buckets)
        );
        if requested_capacity.get() == 1 && before.materialized_graphs > 0 {
            assert_eq!(
                report.cache.graph_builds, before.graph_builds,
                "startup preparation evicted and replaced the existing graph"
            );
        }
        eprintln!(
            "ORBITKV_DECODER_PREPARATION {}",
            serde_json::to_string(&report).unwrap()
        );
    }
    let reference_directory = std::env::var_os("ORBITKV_REFERENCE_DIR").map(PathBuf::from);
    for sequence in 0..sequences {
        let request_id = EngineRequestId(u64::try_from(sequence).unwrap() + 1);
        let completion_base = u64::try_from(sequence).unwrap() * 8;
        let capacity = if sequences > 1 && sequence + 1 == sequences {
            // Exercise eviction after the earlier sequences exercised residency.
            orbitkv_executor::model::DEFAULT_GRAPH_CACHE_CAPACITY
        } else {
            requested_capacity
        };
        decoder.set_graph_cache_capacity(capacity);
        if sequence > 0 {
            run.session.acquire_requests(&[request_id]).unwrap();
            let (attention, prepared, _) = prepare_decode_step_for_request(
                &mut run,
                request_id,
                u64::try_from(prompt.len()).unwrap(),
            );
            run.attention = attention;
            run.prepared = prepared;
        }
        let before = decoder.graph_cache_stats();
        let sequence_started = Instant::now();
        let mut warm_decode_seconds = Vec::new();
        eprintln!(
            "ORBITKV_BUCKET_SEQUENCE_START {}",
            serde_json::json!({"sequence": sequence, "capacity": capacity.get()})
        );
        let classes = decoder_class_steps(&run.prepared, &run.attention);
        let state_steps = run
            .prepared
            .fixed_state_requests()
            .map(
                |(request_id, states)| orbitkv_executor::model::DecoderFixedStateStep {
                    request_id,
                    states,
                },
            )
            .collect::<Vec<_>>();
        let execute_started = Instant::now();
        let output = decoder
            .execute_with_fixed_states_and_logits(
                DecoderStep {
                    tokens: &prompt,
                    positions: &positions,
                    classes: &classes,
                },
                &state_steps,
            )
            .unwrap();
        let prefill_seconds = execute_started.elapsed().as_secs_f64();
        eprintln!(
            "Qwen3.8 27B FP8: bounded prefill completed after {:.6}s token={:?}",
            execute_started.elapsed().as_secs_f64(),
            output.token_ids.last()
        );
        assert_eq!(output.token_ids.len(), prompt.len());
        assert_eq!(output.logits.len(), prompt.len() * config.vocabulary_size);
        let prefill_token = if let Some(reference) = &reference_directory {
            assert_reference_logits(
                "prefill",
                row_logits(&output.logits, prompt.len() - 1, config.vocabulary_size),
                &reference.join("prefill-last.f32"),
                Some(5),
            )
        } else {
            *output.token_ids.last().unwrap()
        };
        complete_with_fixed_states(
            &mut run.session,
            &run.prepared,
            &run.arenas,
            &output.fixed_states,
            completion_base + 1,
        );

        let (decode_attention, decode_prepared, _) =
            prepare_decode_step_for_request(&mut run, request_id, 5);
        let decode_classes = decoder_class_steps(&decode_prepared, &decode_attention);
        let decode_states = decode_prepared
            .fixed_state_requests()
            .map(
                |(request_id, states)| orbitkv_executor::model::DecoderFixedStateStep {
                    request_id,
                    states,
                },
            )
            .collect::<Vec<_>>();
        let decode_started = Instant::now();
        let decode = decoder
            .execute_with_fixed_states_and_logits(
                DecoderStep {
                    tokens: &[prefill_token],
                    positions: &[4],
                    classes: &decode_classes,
                },
                &decode_states,
            )
            .unwrap();
        let first_decode_seconds = decode_started.elapsed().as_secs_f64();
        eprintln!(
            "Qwen3.8 27B FP8: bounded decode completed after {:.6}s token={:?}",
            decode_started.elapsed().as_secs_f64(),
            decode.token_ids.first()
        );
        assert_eq!(decode.token_ids.len(), 1);
        assert_eq!(decode.logits.len(), config.vocabulary_size);
        let mut generated_token = if let Some(reference) = &reference_directory {
            assert_reference_logits(
                "decode",
                &decode.logits,
                &reference.join("decode.f32"),
                Some(0),
            )
        } else {
            decode.token_ids[0]
        };
        complete_with_fixed_states(
            &mut run.session,
            &decode_prepared,
            &run.arenas,
            &decode.fixed_states,
            completion_base + 2,
        );

        for generated_index in 2..8_u64 {
            let target_boundary = u64::try_from(prompt.len()).unwrap() + generated_index;
            let (attention, prepared, _) =
                prepare_decode_step_for_request(&mut run, request_id, target_boundary);
            let classes = decoder_class_steps(&prepared, &attention);
            let states = prepared
                .fixed_state_requests()
                .map(
                    |(request_id, states)| orbitkv_executor::model::DecoderFixedStateStep {
                        request_id,
                        states,
                    },
                )
                .collect::<Vec<_>>();
            let step_started = Instant::now();
            let output = decoder
                .execute_with_fixed_states_and_logits(
                    DecoderStep {
                        tokens: &[generated_token],
                        positions: &[u32::try_from(target_boundary - 1).unwrap()],
                        classes: &classes,
                    },
                    &states,
                )
                .unwrap();
            warm_decode_seconds.push(step_started.elapsed().as_secs_f64());
            eprintln!(
                "ORBITKV_DECODER_STEP {}",
                serde_json::json!({
                    "phase": format!("decode-{generated_index}"),
                    "seconds": step_started.elapsed().as_secs_f64(),
                    "diagnostic_logits": true,
                }),
            );
            let next_token = output.token_ids[0];
            let next_input_token = if let Some(reference) = &reference_directory {
                assert_reference_logits(
                    &format!("decode-{generated_index}"),
                    &output.logits,
                    &reference.join(format!("decode-{generated_index}.f32")),
                    None,
                )
            } else {
                next_token
            };
            complete_with_fixed_states(
                &mut run.session,
                &prepared,
                &run.arenas,
                &output.fixed_states,
                completion_base + generated_index + 1,
            );
            generated_token = next_input_token;
        }
        release_and_drain(&mut run.session, request_id);
        assert!(run.session.fixed_state_stats().iter().all(|(_, state)| {
            state.active_owners == 0
                && state.pending_transitions == 0
                && state.pending_retirements == 0
                && state.free_slots == u64::from(state.identity.slot_count)
        }));
        let after = decoder.graph_cache_stats();
        if (sequence > 0 || preparation_enabled)
            && capacity.get() >= decoder.compiled_bucket_count()
        {
            assert_eq!(
                after.graph_builds, before.graph_builds,
                "retained bucket switch rebuilt the complete graph"
            );
        }
        eprintln!(
            "ORBITKV_BUCKET_SEQUENCE {}",
            serde_json::json!({
                "sequence": sequence, "capacity": capacity.get(),
                "prefill_seconds": prefill_seconds, "first_decode_seconds": first_decode_seconds,
                "warm_decode_seconds": warm_decode_seconds,
                "sequence_seconds": sequence_started.elapsed().as_secs_f64(),
                "graph_builds_before": before.graph_builds, "graph_builds_after": after.graph_builds,
                "materialized_graphs_before": before.materialized_graphs,
                "materialized_graphs_after": after.materialized_graphs,
                "parity_steps": if reference_directory.is_some() { 8 } else { 0 },
                "drain_passed": true,
            })
        );
    }
    eprintln!(
        "ORBITKV_DECODER_QUALIFICATION {}",
        serde_json::json!({
            "schema": 1,
            "reference_enabled": reference_directory.is_some(),
            "parity_steps": if reference_directory.is_some() { 8 * sequences } else { 0 },
            "sequences": sequences,
            "drain_passed": true,
            "artifact_mode": if artifact.is_some() { "replay" } else { "search" },
        }),
    );
    if let Some(trace) = stage_trace {
        trace.finish().unwrap();
    }
}
