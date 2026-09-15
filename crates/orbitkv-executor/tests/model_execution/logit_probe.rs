//! Manifest-driven, teacher-forced decoder traces for independent numerical diagnosis.

use super::*;
use orbitkv_executor::model::{
    DecoderFixedStateStep, DecoderOutputRows, DecoderStorage, DecoderTuningProfile,
    StatefulDecoderDiagnosticOutput,
};
use serde::{Deserialize, Serialize};

mod batches;
mod execution;

const SCHEMA: u32 = 1;

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
    /// Explicit GPU submissions; omitted manifests run cases sequentially.
    #[serde(default)]
    batches: Option<Vec<Vec<batches::Query>>>,
    #[serde(default)]
    graph_cache_capacity: Option<std::num::NonZeroUsize>,
    #[serde(default)]
    prepare_execution: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    prompt_token_ids: Vec<u32>,
    continuation_token_ids: Vec<u32>,
}

impl Probe {
    fn output_row_count(&self, query_tokens: usize, requests: usize) -> usize {
        match self.compile.output_rows {
            DecoderOutputRows::AllTokens => query_tokens,
            DecoderOutputRows::LastTokenPerRequest => requests,
        }
    }

    fn validate_cases(&self, vocabulary_size: usize) -> Result<(), &'static str> {
        if self.schema != SCHEMA || self.cases.is_empty() {
            return Err("unsupported probe schema or empty cases");
        }
        let mut ids = std::collections::HashSet::new();
        for case in &self.cases {
            if case.id.is_empty() || !ids.insert(&case.id) {
                return Err("case IDs must be nonempty and unique");
            }
            if case.prompt_token_ids.is_empty() || case.continuation_token_ids.is_empty() {
                return Err("empty sequence");
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
fn decoder_manifest_logits_and_drain() {
    run_probe(ScheduleSource::Replay);
}

#[test]
#[ignore = "requires a local checkpoint, a fresh schedule path, a probe manifest and CUDA"]
fn decoder_manifest_compile_logits_and_drain() {
    run_probe(ScheduleSource::Compile);
}

#[derive(Clone, Copy)]
enum ScheduleSource {
    Replay,
    Compile,
}

#[allow(clippy::too_many_lines)]
fn run_probe(source: ScheduleSource) {
    let stage_trace = orbitkv_executor::diagnostics::install_stage_trace_from_env().unwrap();
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
    let batches = probe.plan_batches().unwrap();
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
    let artifact_path = environment_path("ORBITKV_DECODER_ARTIFACT");
    let artifact = match source {
        ScheduleSource::Replay => {
            Some(DecoderArtifact::from_bytes(&std::fs::read(&artifact_path).unwrap()).unwrap())
        }
        ScheduleSource::Compile => {
            assert!(
                !artifact_path.exists(),
                "compile probe requires a fresh schedule path"
            );
            None
        }
    };
    let tuning = DecoderTuningProfile::from_json(
        &std::fs::read(environment_path("ORBITKV_TUNING_PROFILE")).unwrap(),
    )
    .unwrap();
    let context = orbitkv_cuda::cudarc::driver::CudaContext::new(probe.device_index).unwrap();
    let stream = context.new_stream().unwrap();
    let fixed_states = fixed_state_identities(&harness.session);
    let (mut decoder, selected) = CompiledDecoder::compile_or_load_with_tuning(
        &config,
        &harness.executor_plan,
        DecoderStorage::new(&harness.arenas, &fixed_states),
        &stream,
        &weight_files(&probe.model_directory),
        probe.compile,
        &tuning,
        artifact.as_ref(),
    )
    .expect("probe compilation or strict schedule replay must succeed");
    if matches!(source, ScheduleSource::Compile) {
        use std::io::Write;
        std::fs::File::create_new(&artifact_path)
            .unwrap()
            .write_all(&selected.to_bytes().unwrap())
            .unwrap();
    }
    let artifact_bytes = std::fs::read(&artifact_path).unwrap();
    let graph_cache_capacity = probe
        .graph_cache_capacity
        .unwrap_or(orbitkv_executor::model::DEFAULT_GRAPH_CACHE_CAPACITY);
    decoder.set_graph_cache_capacity(graph_cache_capacity);
    let preparation = probe
        .prepare_execution
        .then(|| decoder.prepare_execution().unwrap());
    let (traces, submissions) = execution::run(
        &probe,
        &batches,
        &mut decoder,
        &mut harness,
        config.vocabulary_size,
        &output_directory,
    );
    assert_eq!(
        std::fs::read(environment_path("ORBITKV_DECODER_ARTIFACT")).unwrap(),
        artifact_bytes
    );
    let trace = serde_json::json!({
        "schema": SCHEMA, "vocabulary_size": config.vocabulary_size,
        "teacher_forced": true, "cases": traces, "batches": submissions,
        "output_rows": probe.compile.output_rows,
        "graph_cache_capacity": graph_cache_capacity, "preparation": preparation,
        "final_graph_cache": decoder.graph_cache_stats(),
    });
    std::fs::write(
        output_directory.join("trace.json"),
        serde_json::to_vec_pretty(&trace).unwrap(),
    )
    .unwrap();
    eprintln!(
        "ORBITKV_LOGIT_PROBE_COMPLETE {}",
        output_directory.display()
    );
    if let Some(trace) = stage_trace {
        trace.finish().unwrap();
    }
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
            output_rows: orbitkv_executor::model::DecoderOutputRows::AllTokens,
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
        batches: None,
        graph_cache_capacity: None,
        prepare_execution: false,
    };
    assert!(probe.validate_cases(2).is_ok());
    probe.cases.push(Case {
        id: "input".into(),
        prompt_token_ids: vec![0],
        continuation_token_ids: vec![1],
    });
    assert!(probe.validate_cases(2).is_err());
    probe.cases.pop();
    probe.cases[0].prompt_token_ids.clear();
    assert!(probe.validate_cases(2).is_err());
    probe.cases[0].prompt_token_ids = vec![2];
    assert!(probe.validate_cases(2).is_err());
}
