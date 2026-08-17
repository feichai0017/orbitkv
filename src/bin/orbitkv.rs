use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, ExitCode};

use orbitkv::{
    CompiledKvPlan, HfRetentionCompilation, HfRetentionOptions, KvPlanSource, OwnerCommand,
    PhysicalPlanObjective, RetentionAnalysis, SglangOwner, SglangPhysicalOptimizationInput,
    SglangPhysicalPlan, analyze_state, compile_hf_config, compile_retention_program,
    optimize_sglang_physical_plan,
    trace::{read_jsonl, summarize_sglang_trace},
};
use serde::Serialize;

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

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ContractStatus {
    Pass,
    Fail,
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
            let plan_path = required(&mut args, "plan path")?;
            require_end(&mut args)?;
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
            })?;
        }
        Some("analyze-sglang") => {
            let plan_path = required(&mut args, "plan path")?;
            let trace_path = required(&mut args, "trace path")?;
            require_flag(&mut args, "--max-active-requests")?;
            let max_active_requests = required(&mut args, "max active requests")?.parse::<u64>()?;
            require_end(&mut args)?;
            let plan = load_plan(plan_path)?;
            let trace = read_jsonl(BufReader::new(File::open(trace_path)?))?;
            write_json(&summarize_sglang_trace(&trace, &plan, max_active_requests)?)?;
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
            serve_sglang_owner(&load_plan(plan_path)?)?;
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
                "usage: orbitkv <compile-hf-physical-plan|compile-hf-config|compile|analyze-retention|emit-layout|emit-sglang-policy|serve-sglang-owner|analyze-sglang|check-sglang> ..."
                    .into(),
            );
        }
    }
    Ok(())
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
