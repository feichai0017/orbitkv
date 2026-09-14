//! Manifest-driven, teacher-forced decoder traces for independent numerical diagnosis.

use super::*;
use orbitkv_executor::model::{
    DecoderFixedStateStep, DecoderStorage, DecoderTuningProfile, StatefulDecoderDiagnosticOutput,
};
use serde::{Deserialize, Serialize};
use std::io::Write;

const SCHEMA: u32 = 1;
const REQUEST_ID: EngineRequestId = EngineRequestId(1);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    schema: u32,
    model_directory: PathBuf,
    device_index: usize,
    page_tokens: u64,
    kv_dtype_bytes: u64,
    page_counts: Vec<u32>,
    compile: DecoderCompileConfig,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    prompt_token_ids: Vec<u32>,
    continuation_token_ids: Vec<u32>,
}

impl Probe {
    fn validate_cases(&self, vocabulary_size: usize) -> Result<(), &'static str> {
        if self.schema != SCHEMA || self.cases.is_empty() {
            return Err("unsupported probe schema or empty cases");
        }
        let mut ids = std::collections::HashSet::new();
        for case in &self.cases {
            if case.id.is_empty() || !ids.insert(&case.id) {
                return Err("case IDs must be nonempty and unique");
            }
            if case.prompt_token_ids.is_empty()
                || case.continuation_token_ids.is_empty()
                || case.prompt_token_ids.len() > self.compile.maximum_query_tokens
            {
                return Err("empty sequence or prompt exceeds compiled capacity");
            }
            if case
                .prompt_token_ids
                .len()
                .checked_add(case.continuation_token_ids.len())
                .and_then(|length| u32::try_from(length).ok())
                .is_none()
            {
                return Err("positions exceed the decoder token-position ABI");
            }
            if case
                .prompt_token_ids
                .iter()
                .chain(&case.continuation_token_ids)
                .any(|&token| usize::try_from(token).map_or(true, |id| id >= vocabulary_size))
            {
                return Err("token is outside the model vocabulary");
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct StepTrace {
    step: usize,
    input_token_ids: Vec<u32>,
    selected_token_id: u32,
    logits_file: String,
}

#[derive(Serialize)]
struct CaseTrace {
    id: String,
    prompt_token_ids: Vec<u32>,
    steps: Vec<StepTrace>,
    drain_passed: bool,
}

fn environment_path(name: &str) -> PathBuf {
    std::env::var_os(name).map_or_else(|| panic!("{name} is required"), PathBuf::from)
}

#[test]
#[ignore = "requires a local checkpoint, an existing schedule, a probe manifest and CUDA"]
#[allow(clippy::too_many_lines)]
fn decoder_manifest_logits_and_drain() {
    let probe: Probe = serde_json::from_slice(
        &std::fs::read(environment_path("ORBITKV_LOGIT_PROBE_MANIFEST")).unwrap(),
    )
    .unwrap();
    assert_eq!(probe.schema, SCHEMA);
    assert!(!probe.cases.is_empty());
    assert!(probe.page_tokens > 0 && !probe.page_counts.is_empty());
    let output_directory = environment_path("ORBITKV_LOGIT_PROBE_OUTPUT");
    std::fs::create_dir(&output_directory).expect("probe output must be fresh");
    let config_bytes = std::fs::read(probe.model_directory.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&config_bytes).unwrap();
    probe.validate_cases(config.vocabulary_size).unwrap();
    let mut harness = model_harness_with_geometry(
        &config_bytes,
        probe.compile.maximum_batch_size,
        probe.compile.maximum_batch_size,
        probe.compile.maximum_query_tokens,
        PhysicalResidencePolicy::Compiled,
        Some(&probe.page_counts),
        HfRetentionOptions {
            page_tokens: probe.page_tokens,
            kv_dtype_bytes: probe.kv_dtype_bytes,
        },
    );
    let artifact_bytes = std::fs::read(environment_path("ORBITKV_DECODER_ARTIFACT")).unwrap();
    let artifact = DecoderArtifact::from_bytes(&artifact_bytes).unwrap();
    let tuning = DecoderTuningProfile::from_json(
        &std::fs::read(environment_path("ORBITKV_TUNING_PROFILE")).unwrap(),
    )
    .unwrap();
    let context = luminal_cuda_lite::cudarc::driver::CudaContext::new(probe.device_index).unwrap();
    let stream = context.new_stream().unwrap();
    let fixed_states = fixed_state_identities(&harness.session);
    let (mut decoder, _) = CompiledDecoder::compile_or_load_with_tuning(
        &config,
        &harness.executor_plan,
        DecoderStorage::new(&harness.arenas, &fixed_states),
        &stream,
        &weight_files(&probe.model_directory),
        probe.compile,
        &tuning,
        Some(&artifact),
    )
    .expect("strict replay of the supplied schedule must succeed");
    let mut traces = Vec::new();
    let mut completion_value = 0;
    for (case_index, case) in probe.cases.into_iter().enumerate() {
        harness.session.acquire_requests(&[REQUEST_ID]).unwrap();
        let mut steps = Vec::new();
        for step in 0..case.continuation_token_ids.len() {
            let tokens = if step == 0 {
                case.prompt_token_ids.clone()
            } else {
                vec![case.continuation_token_ids[step - 1]]
            };
            let boundary = case.prompt_token_ids.len() + step;
            let positions = (boundary - tokens.len()..boundary)
                .map(|position| u32::try_from(position).unwrap())
                .collect::<Vec<_>>();
            let (attention, prepared) = prepare_model_batch(
                &mut harness,
                &[REQUEST_ID],
                u64::try_from(boundary).unwrap(),
            );
            let classes = decoder_class_steps(&prepared, &attention);
            let states = prepared
                .fixed_state_requests()
                .map(|(request_id, states)| DecoderFixedStateStep { request_id, states })
                .collect::<Vec<_>>();
            let step_input = DecoderStep {
                tokens: &tokens,
                positions: &positions,
                classes: &classes,
            };
            let output = if harness.executor_plan.fixed_states.is_empty() {
                let output = decoder.execute_with_logits(step_input).unwrap();
                StatefulDecoderDiagnosticOutput {
                    token_ids: output.token_ids,
                    logits: output.logits,
                    fixed_states: Box::default(),
                }
            } else {
                decoder
                    .execute_with_fixed_states_and_logits(step_input, &states)
                    .unwrap()
            };
            assert_eq!(output.logits.len(), tokens.len() * config.vocabulary_size);
            let row = row_logits(&output.logits, tokens.len() - 1, config.vocabulary_size);
            assert!(row.iter().all(|value| value.is_finite()));
            let logits_file = format!("case-{case_index}-step-{step}.f32");
            let mut file = std::fs::File::create_new(output_directory.join(&logits_file)).unwrap();
            for value in row {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            steps.push(StepTrace {
                step,
                input_token_ids: tokens,
                selected_token_id: *output.token_ids.last().unwrap(),
                logits_file,
            });
            completion_value += 1;
            complete_with_fixed_states(
                &mut harness.session,
                &prepared,
                &harness.arenas,
                &output.fixed_states,
                completion_value,
            );
        }
        release_and_drain(&mut harness.session, REQUEST_ID);
        assert!(
            harness
                .session
                .fixed_state_stats()
                .iter()
                .all(|(_, state)| state.active_owners == 0
                    && state.pending_transitions == 0
                    && state.pending_retirements == 0
                    && state.free_slots == u64::from(state.identity.slot_count))
        );
        traces.push(CaseTrace {
            id: case.id,
            prompt_token_ids: case.prompt_token_ids,
            steps,
            drain_passed: true,
        });
    }
    assert_eq!(
        std::fs::read(environment_path("ORBITKV_DECODER_ARTIFACT")).unwrap(),
        artifact_bytes
    );
    let trace = serde_json::json!({"schema": SCHEMA, "vocabulary_size": config.vocabulary_size, "teacher_forced": true, "cases": traces});
    std::fs::write(
        output_directory.join("trace.json"),
        serde_json::to_vec_pretty(&trace).unwrap(),
    )
    .unwrap();
    eprintln!(
        "ORBITKV_LOGIT_PROBE_COMPLETE {}",
        output_directory.display()
    );
}

#[test]
fn manifest_rejects_ambiguous_or_unrepresentable_cases_before_device_execution() {
    let mut probe = Probe {
        schema: SCHEMA,
        model_directory: PathBuf::new(),
        device_index: 0,
        page_tokens: 1,
        kv_dtype_bytes: 2,
        page_counts: vec![1],
        compile: DecoderCompileConfig {
            maximum_query_tokens: 2,
            representative_prefill_tokens: 2,
            maximum_batch_size: 1,
            maximum_context_pages: 1,
            representative_context_pages: 1,
            search_graphs: 1,
            search_seed: 0,
        },
        cases: vec![Case {
            id: "input".into(),
            prompt_token_ids: vec![0],
            continuation_token_ids: vec![1],
        }],
    };
    assert!(probe.validate_cases(2).is_ok());
    probe.cases.push(Case {
        id: "input".into(),
        prompt_token_ids: vec![0],
        continuation_token_ids: vec![1],
    });
    assert!(probe.validate_cases(2).is_err());
    probe.cases.pop();
    probe.cases[0].prompt_token_ids = vec![0, 0, 0];
    assert!(probe.validate_cases(2).is_err());
    probe.cases[0].prompt_token_ids = vec![2];
    assert!(probe.validate_cases(2).is_err());
}
