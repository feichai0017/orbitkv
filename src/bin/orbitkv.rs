use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, ExitCode};

use orbitkv::{
    ApplicabilityReport, CapsuleComponentSpec, CapsuleIdentity, CapsuleManifest, CompiledKvPlan,
    ContentDigest, HfRetentionCompilation, HfRetentionOptions, HfStatePlanOptions,
    HoltCapsuleStore, KvPlanSource, OwnerCommand, PhysicalPlanObjective, PrefixPath,
    RetentionAnalysis, RuntimeCapsuleContract, RuntimeExecutionContract, RuntimeExecutionFrontier,
    RuntimeExecutionMode, RuntimeOwnerTransport, RuntimePrefixContract, RuntimePrefixMode,
    RuntimeStatePlan, RuntimeStatePlanOptions, RuntimeUniformStatePlanMode, SglangOwner,
    SglangPhysicalOptimizationInput, SglangPhysicalPlan, SglangUniformSwaOptions,
    UniformSwaCudaGraphMode, analyze_state, build_capsule_components, compile_hf_config,
    compile_hf_state_plan, compile_retention_program, compile_runtime_state_plan,
    optimize_sglang_physical_plan,
    trace::{read_jsonl, summarize_sglang_trace},
};
use serde::{Deserialize, Serialize};

const EXPECTED_SGLANG_REVISION: &str = "095ec6c997bfdd25d3864cb0ce77a6562a934b96";

#[derive(Serialize)]
struct CompileReport<'a> {
    page_tokens: u64,
    boundary: u64,
    classes: &'a [orbitkv::CompiledKvClass],
    capacity: Vec<orbitkv::ClassCapacity>,
    resident_bytes: u64,
    all_full_baseline_bytes: u64,
    continuation_blocks: std::collections::BTreeMap<String, Vec<orbitkv::plan::BlockRange>>,
}

#[derive(Serialize)]
struct SglangContractReport {
    root: String,
    revision: String,
    expected_revision: &'static str,
    allocator_methods: Vec<String>,
    passed_checks: Vec<&'static str>,
    failed_checks: Vec<&'static str>,
    status: ContractStatus,
}

#[derive(Serialize)]
struct RetentionReport {
    schema: &'static str,
    source_schema: String,
    page_tokens: u64,
    analyses: Vec<RetentionAnalysis>,
    layout: orbitkv::LayoutProgram,
}

#[derive(Serialize)]
struct HfCompileReport {
    compilation: HfRetentionCompilation,
    layout: orbitkv::LayoutProgram,
}

#[derive(Serialize)]
struct HfPhysicalPlanReport {
    schema: &'static str,
    compilation: HfRetentionCompilation,
    layout: orbitkv::LayoutProgram,
    physical_plan: SglangPhysicalPlan,
}

#[derive(Serialize)]
struct HfApplicabilityReport {
    schema: &'static str,
    compilation: HfRetentionCompilation,
    layout: orbitkv::LayoutProgram,
    applicability: ApplicabilityReport,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ContractStatus {
    Pass,
    Fail,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum CapsuleCommand {
    Publish {
        identity: CapsuleIdentity,
        chunk_tokens: u32,
        token_ids: Vec<u32>,
        live_token_count: u64,
        payload_path: String,
        components: Vec<CapsuleComponentSpec>,
        created_unix_ms: u64,
    },
    Restore {
        identity: CapsuleIdentity,
        chunk_tokens: u32,
        token_ids: Vec<u32>,
    },
    Checkpoint,
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum CapsuleResponse {
    Published {
        capsule_id: ContentDigest,
        payload_digest: ContentDigest,
        prefix_token_count: u64,
        payload_bytes: u64,
        created: bool,
        manifest: Box<CapsuleManifest>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefix: Option<Box<orbitkv::PrefixObjectSnapshot>>,
    },
    Restored {
        manifest: Box<CapsuleManifest>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefix: Option<Box<orbitkv::PrefixObjectSnapshot>>,
        payload_path: String,
    },
    Miss,
    Checkpointed,
    Error {
        code: &'static str,
        message: String,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("orbitkv: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("compile-hf-physical-plan") => {
            compile_hf_physical_plan_command(&mut args)?;
        }
        Some("compile-hf-config") => {
            compile_hf_config_command(&mut args)?;
        }
        Some("compile-hf-state-plan") => {
            compile_hf_state_plan_command(&mut args)?;
        }
        Some("compile-runtime-state-plan") => {
            compile_runtime_state_plan_command(&mut args)?;
        }
        Some("validate-runtime-state-plan") => {
            validate_runtime_state_plan_command(&mut args)?;
        }
        Some("analyze-hf-applicability") => {
            analyze_hf_applicability_command(&mut args)?;
        }
        Some("compile") => {
            let plan_path = required(&mut args, "plan path")?;
            require_flag(&mut args, "--boundary")?;
            let boundary = required(&mut args, "boundary")?.parse::<u64>()?;
            require_end(&mut args)?;
            let plan = load_plan(plan_path)?;
            let report = CompileReport {
                page_tokens: plan.page_tokens,
                boundary,
                classes: &plan.classes,
                capacity: plan.capacity_at(boundary)?,
                resident_bytes: plan.resident_bytes_at(boundary)?,
                all_full_baseline_bytes: plan.all_full_baseline_bytes_at(boundary)?,
                continuation_blocks: plan.continuation_ranges(boundary)?,
            };
            write_json(&report)?;
        }
        Some("analyze-retention") => {
            analyze_retention_command(&mut args)?;
        }
        Some("analyze-lifetime-normalization") => {
            analyze_lifetime_normalization_command(&mut args)?;
        }
        Some("analyze-applicability") => {
            analyze_applicability_command(&mut args)?;
        }
        Some("analyze-sglang") => {
            analyze_sglang_command(&mut args)?;
        }
        Some("emit-sglang-policy") => {
            let plan_path = required(&mut args, "plan path")?;
            let plan = load_plan(plan_path)?;
            let policy = match args.next() {
                None => plan.sglang_policy()?,
                Some(flag) if flag == "--eviction-interval" => {
                    let interval = required(&mut args, "eviction interval")?.parse::<u64>()?;
                    require_end(&mut args)?;
                    plan.sglang_policy_with_eviction_interval(interval)?
                }
                Some(argument) => return Err(format!("unexpected argument {argument}").into()),
            };
            write_json(&policy)?;
        }
        Some("emit-layout") => {
            let plan_path = required(&mut args, "plan path")?;
            require_end(&mut args)?;
            write_json(&load_plan(plan_path)?.layout_program()?)?;
        }
        Some("serve-sglang-owner") => {
            let plan_path = required(&mut args, "plan path")?;
            require_end(&mut args)?;
            serve_sglang_owner(&load_owner_plan(plan_path)?)?;
        }
        Some("serve-capsules") => {
            let root = required(&mut args, "capsule store root")?;
            require_end(&mut args)?;
            serve_capsules(Path::new(&root))?;
        }
        Some("check-sglang") => {
            let root = required(&mut args, "SGLang root")?;
            require_end(&mut args)?;
            let report = check_sglang(Path::new(&root))?;
            let valid = matches!(report.status, ContractStatus::Pass);
            write_json(&report)?;
            if !valid {
                return Err("SGLang contract check failed".into());
            }
        }
        _ => {
            return Err(
                "usage: orbitkv <compile-hf-physical-plan|compile-hf-config|compile-hf-state-plan|compile-runtime-state-plan|validate-runtime-state-plan|compile|analyze-hf-applicability|analyze-retention|analyze-lifetime-normalization|analyze-applicability|emit-layout|emit-sglang-policy|serve-sglang-owner|serve-capsules|analyze-sglang|check-sglang> ..."
                    .into(),
            );
        }
    }
    Ok(())
}

fn analyze_sglang_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan_path = required(args, "plan path")?;
    let trace_path = required(args, "trace path")?;
    require_flag(args, "--max-active-requests")?;
    let max_active_requests = required(args, "max active requests")?.parse::<u64>()?;
    require_end(args)?;
    let plan = load_plan(plan_path)?;
    let trace = read_jsonl(BufReader::new(File::open(trace_path)?))?;
    write_json(&summarize_sglang_trace(&trace, &plan, max_active_requests)?)
}

fn validate_runtime_state_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "runtime StatePlan path")?;
    require_end(args)?;
    let artifact = serde_json::from_slice::<RuntimeStatePlan>(&std::fs::read(path)?)?;
    artifact.validate()?;
    write_json(&artifact)
}

fn compile_runtime_state_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan_path = required(args, "plan path")?;
    let eviction_interval_tokens =
        required_flagged_u64(args, "--eviction-interval", "eviction interval")?;
    require_flag(args, "--execution-mode")?;
    let execution_mode = match required(args, "execution mode")?.as_str() {
        "policy" => RuntimeExecutionMode::Policy,
        "owner" => RuntimeExecutionMode::Owner,
        value => return Err(format!("unsupported execution mode {value:?}").into()),
    };
    require_flag(args, "--owner-transport")?;
    let owner_transport = match required(args, "owner transport")?.as_str() {
        "none" => None,
        "ffi" => Some(RuntimeOwnerTransport::Ffi),
        "sidecar" => Some(RuntimeOwnerTransport::Sidecar),
        value => return Err(format!("unsupported owner transport {value:?}").into()),
    };
    require_flag(args, "--capsule-enabled")?;
    let capsule_enabled = parse_bool(&required(args, "capsule enabled")?)?;
    let capsule_chunk_tokens =
        required_flagged_u64(args, "--capsule-chunk-tokens", "Capsule chunk tokens")?;
    let capsule_maximum_payload_bytes =
        required_flagged_u64(args, "--capsule-max-payload-bytes", "Capsule payload limit")?;
    let mut physical_plan = None;
    let mut uniform_state_plan = None;
    let mut uniform_state_plan_mode = None;
    let mut prefix = None;
    let mut execution_frontier = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--physical-plan" => {
                let path = required(args, "physical plan path")?;
                physical_plan = Some(serde_json::from_slice(&std::fs::read(path)?)?);
            }
            "--uniform-state-plan" => {
                let path = required(args, "uniform state plan path")?;
                uniform_state_plan = Some(serde_json::from_slice(&std::fs::read(path)?)?);
            }
            "--uniform-state-plan-mode" => {
                uniform_state_plan_mode =
                    Some(match required(args, "uniform state plan mode")?.as_str() {
                        "execute" => RuntimeUniformStatePlanMode::Execute,
                        "kernel_reference" => RuntimeUniformStatePlanMode::KernelReference,
                        value => {
                            return Err(
                                format!("unsupported uniform state plan mode {value:?}").into()
                            );
                        }
                    });
            }
            "--prefix-mode" => {
                prefix = Some(RuntimePrefixContract {
                    mode: match required(args, "Prefix mode")?.as_str() {
                        "capsule_backed_swa_radix" => RuntimePrefixMode::CapsuleBackedSwaRadix,
                        value => {
                            return Err(format!("unsupported Prefix mode {value:?}").into());
                        }
                    },
                });
            }
            "--execution-frontier" => {
                execution_frontier = Some(match required(args, "execution frontier")?.as_str() {
                    "cuda_event" => RuntimeExecutionFrontier::CudaEvent,
                    value => {
                        return Err(format!("unsupported execution frontier {value:?}").into());
                    }
                });
            }
            argument => return Err(format!("unexpected argument {argument}").into()),
        }
    }
    let artifact = compile_runtime_state_plan(
        load_source(plan_path)?,
        RuntimeStatePlanOptions {
            eviction_interval_tokens,
            physical_plan,
            uniform_state_plan,
            execution: RuntimeExecutionContract {
                mode: execution_mode,
                owner_transport,
                uniform_state_plan_mode,
                frontier: execution_frontier,
            },
            capsule: RuntimeCapsuleContract {
                enabled: capsule_enabled,
                chunk_tokens: capsule_chunk_tokens,
                maximum_payload_bytes: capsule_maximum_payload_bytes,
            },
            prefix,
        },
    )?;
    artifact.validate()?;
    write_json(&artifact)
}

fn parse_bool(value: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("expected true or false, got {value:?}").into()),
    }
}

fn serve_capsules(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let store = HoltCapsuleStore::open(root)?;
    let stdin = std::io::stdin();
    let mut stdout = BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<CapsuleCommand>(&line) {
            Ok(command) => execute_capsule_command(&store, root, command),
            Err(error) => CapsuleResponse::Error {
                code: "invalid_command",
                message: error.to_string(),
            },
        };
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    store.checkpoint()?;
    Ok(())
}

fn execute_capsule_command(
    store: &HoltCapsuleStore,
    root: &Path,
    command: CapsuleCommand,
) -> CapsuleResponse {
    match execute_capsule_command_inner(store, root, command) {
        Ok(response) => response,
        Err(error) => CapsuleResponse::Error {
            code: "capsule_operation_failed",
            message: error.to_string(),
        },
    }
}

fn execute_capsule_command_inner(
    store: &HoltCapsuleStore,
    root: &Path,
    command: CapsuleCommand,
) -> Result<CapsuleResponse, Box<dyn std::error::Error>> {
    match command {
        CapsuleCommand::Publish {
            identity,
            chunk_tokens,
            token_ids,
            live_token_count,
            payload_path,
            components,
            created_unix_ms,
        } => {
            let path = PrefixPath::from_token_ids(identity, chunk_tokens, &token_ids)?;
            let payload = std::fs::read(payload_path)?;
            let components = build_capsule_components(&payload, &components)?;
            let manifest = CapsuleManifest::new(
                &path,
                live_token_count,
                &payload,
                components,
                created_unix_ms,
            )?;
            let publication = store.publish(&path, &manifest, &payload)?;
            let prefix = if manifest.components.iter().all(|component| {
                component.token_start.is_some() && component.token_end_exclusive.is_some()
            }) {
                let mut prefix_runtime = orbitkv::PrefixRuntime::default();
                let object_id = prefix_runtime.register_capsule(&path, &manifest)?;
                Some(Box::new(prefix_runtime.snapshot(object_id)?))
            } else {
                None
            };
            Ok(CapsuleResponse::Published {
                capsule_id: manifest.capsule_id,
                payload_digest: manifest.payload_digest,
                prefix_token_count: manifest.prefix_token_count,
                payload_bytes: manifest.payload_bytes,
                created: matches!(publication, orbitkv::CapsulePublish::Published),
                manifest: Box::new(manifest),
                prefix,
            })
        }
        CapsuleCommand::Restore {
            identity,
            chunk_tokens,
            token_ids,
        } => {
            let path = PrefixPath::from_token_ids(identity, chunk_tokens, &token_ids)?;
            let Some(restored) = store.restore_deepest_state(&path)? else {
                return Ok(CapsuleResponse::Miss);
            };
            let manifest = restored.capsule.manifest;
            let prefix = restored.prefix.map(Box::new);
            let payload_path = capsule_payload_path(root, manifest.payload_digest);
            Ok(CapsuleResponse::Restored {
                manifest: Box::new(manifest),
                prefix,
                payload_path: payload_path.display().to_string(),
            })
        }
        CapsuleCommand::Checkpoint => {
            store.checkpoint()?;
            Ok(CapsuleResponse::Checkpointed)
        }
    }
}

fn capsule_payload_path(root: &Path, digest: ContentDigest) -> std::path::PathBuf {
    let hex = digest.to_hex();
    root.join("objects")
        .join(&hex[..2])
        .join(format!("{}.capsule", &hex[2..]))
}

fn compile_hf_state_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = required(args, "HF config path")?;
    let page_tokens = required_flagged_u64(args, "--page-tokens", "page tokens")?;
    let kv_dtype_bytes = required_flagged_u64(args, "--kv-dtype-bytes", "KV dtype bytes")?;
    let boundary = required_flagged_u64(args, "--boundary", "boundary")?;
    let maximum_running_requests =
        required_flagged_u64(args, "--max-running-requests", "max running requests")?;
    let chunked_prefill_tokens =
        required_flagged_u64(args, "--chunked-prefill-tokens", "chunked prefill tokens")?;
    let eviction_interval_tokens =
        required_flagged_u64(args, "--eviction-interval", "eviction interval")?;
    let decode_headroom_tokens =
        required_flagged_u64(args, "--decode-headroom-tokens", "decode headroom tokens")?;
    require_flag(args, "--cuda-graph-mode")?;
    let cuda_graph_mode = parse_cuda_graph_mode(&required(args, "CUDA Graph mode")?)?;
    require_end(args)?;
    write_json(&compile_hf_state_plan(
        &std::fs::read(config_path)?,
        HfStatePlanOptions {
            retention: HfRetentionOptions {
                page_tokens,
                kv_dtype_bytes,
            },
            boundary_tokens: boundary,
            sglang_uniform_swa: SglangUniformSwaOptions {
                maximum_running_requests,
                chunked_prefill_tokens,
                eviction_interval_tokens,
                decode_headroom_tokens,
                cuda_graph_mode,
            },
        },
    )?)
}

fn parse_cuda_graph_mode(
    value: &str,
) -> Result<UniformSwaCudaGraphMode, Box<dyn std::error::Error>> {
    match value {
        "disabled" => Ok(UniformSwaCudaGraphMode::Disabled),
        "decode" => Ok(UniformSwaCudaGraphMode::Decode),
        _ => Err(format!("unsupported CUDA Graph mode {value:?}").into()),
    }
}

fn analyze_retention_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan_path = required(args, "plan path")?;
    require_end(args)?;
    let source = load_source(plan_path)?;
    let program = source.clone().into_retention_program()?;
    let analyses = program
        .states
        .iter()
        .map(analyze_state)
        .collect::<Result<Vec<_>, _>>()?;
    let plan = source.compile()?;
    write_json(&RetentionReport {
        schema: "orbitkv.retention-analysis.v1",
        source_schema: program.schema,
        page_tokens: program.page_tokens,
        analyses,
        layout: plan.layout_program()?,
    })
}

fn analyze_applicability_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan_path = required(args, "plan path")?;
    let boundary = required_flagged_u64(args, "--boundary", "boundary")?;
    require_end(args)?;
    write_json(&load_plan(plan_path)?.applicability_report(boundary)?)
}

fn analyze_hf_applicability_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = required(args, "HF config path")?;
    let page_tokens = required_flagged_u64(args, "--page-tokens", "page tokens")?;
    let kv_dtype_bytes = required_flagged_u64(args, "--kv-dtype-bytes", "KV dtype bytes")?;
    let boundary = required_flagged_u64(args, "--boundary", "boundary")?;
    require_end(args)?;
    let compilation = compile_hf_config(
        &std::fs::read(config_path)?,
        HfRetentionOptions {
            page_tokens,
            kv_dtype_bytes,
        },
    )?;
    let plan = compile_retention_program(compilation.program.clone())?;
    write_json(&HfApplicabilityReport {
        schema: "orbitkv.hf-applicability-compilation.v1",
        layout: plan.layout_program()?,
        applicability: plan.applicability_report(boundary)?,
        compilation,
    })
}

fn analyze_lifetime_normalization_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan_path = required(args, "plan path")?;
    require_end(args)?;
    write_json(&load_plan(plan_path)?.lifetime_normalization_report()?)
}

fn compile_hf_physical_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = required(args, "HF config path")?;
    let page_tokens = required_flagged_u64(args, "--page-tokens", "page tokens")?;
    let kv_dtype_bytes = required_flagged_u64(args, "--kv-dtype-bytes", "KV dtype bytes")?;
    let available_kv_bytes =
        required_flagged_u64(args, "--available-kv-bytes", "available KV bytes")?;
    let max_running_requests =
        required_flagged_u64(args, "--max-running-requests", "max running requests")?;
    let attention_data_parallel_size =
        required_flagged_u64(args, "--attention-dp-size", "attention DP size")?;
    let chunked_prefill_tokens =
        required_flagged_u64(args, "--chunked-prefill-tokens", "chunked prefill tokens")?;
    let workload_requests = required_flagged_u64(args, "--workload-requests", "workload requests")?;
    let prompt_tokens_per_request = required_flagged_u64(args, "--prompt-tokens", "prompt tokens")?;
    let decode_tokens_per_request = required_flagged_u64(args, "--decode-tokens", "decode tokens")?;
    require_flag(args, "--candidate-intervals")?;
    let candidate_eviction_intervals = parse_intervals(&required(args, "candidate intervals")?)?;
    let maximum_reclamation_calls_per_request =
        required_flagged_u64(args, "--max-reclamation-calls", "max reclamation calls")?;
    let minimum_admitted_requests =
        required_flagged_u64(args, "--min-admitted-requests", "minimum admitted requests")?;
    require_flag(args, "--objective")?;
    let objective = parse_objective(&required(args, "physical-plan objective")?)?;
    require_end(args)?;

    let compilation = compile_hf_config(
        &std::fs::read(config_path)?,
        HfRetentionOptions {
            page_tokens,
            kv_dtype_bytes,
        },
    )?;
    let plan = compile_retention_program(compilation.program.clone())?;
    let layout = plan.layout_program()?;
    let physical_plan = optimize_sglang_physical_plan(
        &plan,
        &SglangPhysicalOptimizationInput {
            available_kv_bytes,
            max_running_requests,
            attention_data_parallel_size,
            chunked_prefill_tokens,
            workload_requests,
            prompt_tokens_per_request,
            decode_tokens_per_request,
            candidate_eviction_intervals,
            maximum_reclamation_calls_per_request: Some(maximum_reclamation_calls_per_request),
            minimum_admitted_requests: Some(minimum_admitted_requests),
            objective,
        },
    )?;
    write_json(&HfPhysicalPlanReport {
        schema: "orbitkv.hf-physical-compilation.v1",
        compilation,
        layout,
        physical_plan,
    })
}

fn compile_hf_config_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = required(args, "HF config path")?;
    require_flag(args, "--page-tokens")?;
    let page_tokens = required(args, "page tokens")?.parse::<u64>()?;
    require_flag(args, "--kv-dtype-bytes")?;
    let kv_dtype_bytes = required(args, "KV dtype bytes")?.parse::<u64>()?;
    require_end(args)?;
    let compilation = compile_hf_config(
        &std::fs::read(config_path)?,
        HfRetentionOptions {
            page_tokens,
            kv_dtype_bytes,
        },
    )?;
    let layout = compile_retention_program(compilation.program.clone())?.layout_program()?;
    write_json(&HfCompileReport {
        compilation,
        layout,
    })
}

fn required_flagged_u64(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
    name: &str,
) -> Result<u64, Box<dyn std::error::Error>> {
    require_flag(args, flag)?;
    Ok(required(args, name)?.parse::<u64>()?)
}

fn parse_intervals(value: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let intervals = value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()?;
    if intervals.is_empty() {
        return Err("candidate intervals must not be empty".into());
    }
    Ok(intervals)
}

fn parse_objective(value: &str) -> Result<PhysicalPlanObjective, Box<dyn std::error::Error>> {
    match value {
        "capacity" => Ok(PhysicalPlanObjective::CapacityUnderReclamationBudget),
        "reclamation" => Ok(PhysicalPlanObjective::ReclamationUnderAdmissionTarget),
        _ => Err(format!("unsupported physical-plan objective {value:?}").into()),
    }
}

fn load_plan(path: impl AsRef<Path>) -> Result<CompiledKvPlan, Box<dyn std::error::Error>> {
    Ok(load_source(path)?.compile()?)
}

fn load_source(path: impl AsRef<Path>) -> Result<KvPlanSource, Box<dyn std::error::Error>> {
    Ok(serde_json::from_reader::<_, KvPlanSource>(BufReader::new(
        File::open(path)?,
    ))?)
}

fn load_owner_plan(path: impl AsRef<Path>) -> Result<CompiledKvPlan, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if let Ok(artifact) = serde_json::from_slice::<RuntimeStatePlan>(&bytes) {
        artifact.validate()?;
        if artifact.execution.mode != RuntimeExecutionMode::Owner {
            return Err("runtime StatePlan execution mode is not owner".into());
        }
        return Ok(artifact.semantic_source.compile()?);
    }
    Ok(serde_json::from_slice::<KvPlanSource>(&bytes)?.compile()?)
}

fn check_sglang(root: &Path) -> Result<SglangContractReport, Box<dyn std::error::Error>> {
    let revision = String::from_utf8(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(root)
            .output()?
            .stdout,
    )?
    .trim()
    .to_owned();
    let plugin_source =
        std::fs::read_to_string(root.join("python/sglang/srt/plugins/__init__.py"))?;
    let hook_source =
        std::fs::read_to_string(root.join("python/sglang/srt/plugins/hook_registry.py"))?;
    let allocator_source =
        std::fs::read_to_string(root.join("python/sglang/srt/mem_cache/allocator/swa.py"))?;
    let methods = ["alloc", "alloc_extend", "alloc_decode", "free", "free_swa"]
        .into_iter()
        .filter(|method| allocator_source.contains(&format!("    def {method}(")))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let plugin_group_present = plugin_source.contains("sglang.srt.plugins")
        && plugin_source.contains("GENERAL_PLUGINS_GROUP");
    let hook_registry_present =
        hook_source.contains("class HookRegistry") && hook_source.contains("class HookType");
    let revision_matches = revision == EXPECTED_SGLANG_REVISION;
    let mut passed_checks = Vec::new();
    let mut failed_checks = Vec::new();
    for (name, passed) in [
        ("revision", revision_matches),
        ("plugin_group", plugin_group_present),
        ("hook_registry", hook_registry_present),
        ("allocator_methods", methods.len() == 5),
    ] {
        if passed {
            passed_checks.push(name);
        } else {
            failed_checks.push(name);
        }
    }
    let status = if failed_checks.is_empty() {
        ContractStatus::Pass
    } else {
        ContractStatus::Fail
    };
    Ok(SglangContractReport {
        root: root.display().to_string(),
        revision,
        expected_revision: EXPECTED_SGLANG_REVISION,
        allocator_methods: methods,
        passed_checks,
        failed_checks,
        status,
    })
}

fn required(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next().ok_or_else(|| format!("missing {name}").into())
}

fn require_flag(
    args: &mut impl Iterator<Item = String>,
    expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = required(args, expected)?;
    if actual != expected {
        return Err(format!("expected {expected}, got {actual}").into());
    }
    Ok(())
}

fn require_end(args: &mut impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(argument) = args.next() {
        return Err(format!("unexpected argument {argument}").into());
    }
    Ok(())
}

fn write_json(value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer_pretty(BufWriter::new(std::io::stdout()), value)?;
    println!();
    Ok(())
}

fn serve_sglang_owner(plan: &CompiledKvPlan) -> Result<(), Box<dyn std::error::Error>> {
    let mut owner = SglangOwner::new(plan)?;
    let stdin = std::io::stdin();
    let mut stdout = BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<OwnerCommand>(&line) {
            Ok(command) => owner.execute(command),
            Err(error) => orbitkv::OwnerResponse::Error {
                code: "invalid_command",
                message: error.to_string(),
            },
        };
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}
