use std::env;
use std::fs;
use std::process::ExitCode;

use kern_manifest::Verified;
use orbitkv_compiler::compiler::provider::{DeepGemmSm90Capability, Fp8ProjectionShape, render_aot_source};
use orbitkv_compiler::lower::{Qwen38Fp8WeightPlan, lower_deepgemm_projection_probe};
use orbitkv_compiler::oracle::{ManifestInventory, Qwen38Fp8Checkpoint, Qwen38Oracle};

const DEFAULT_ORACLE: &str = "examples/qwen3.8-27b.json";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("oracle") => run_oracle(args),
        Some("provider") => run_provider(args),
        _ => Err(usage().into()),
    }
}

fn run_oracle(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    match args.next().as_deref() {
        Some("inspect") => {
            let path = args.next().unwrap_or_else(|| DEFAULT_ORACLE.into());
            no_more(args)?;
            let json = fs::read_to_string(path)?;
            let verified = Verified::from_json(&json)?;
            println!("{}", serde_json::to_string_pretty(&ManifestInventory::from_verified(&verified))?);
        }
        Some("validate") => {
            let path = args.next().unwrap_or_else(|| DEFAULT_ORACLE.into());
            no_more(args)?;
            let oracle = Qwen38Oracle::from_path(path)?;
            oracle.validate_contract()?;
            println!("{}", serde_json::to_string_pretty(&oracle.report())?);
        }
        Some("diff") => {
            let candidate = args.next().ok_or_else(usage)?;
            let reference = args.next().unwrap_or_else(|| DEFAULT_ORACLE.into());
            no_more(args)?;
            let oracle = Qwen38Oracle::from_path(reference)?;
            oracle.validate_contract()?;
            let candidate = Verified::from_json(&fs::read_to_string(candidate)?)?;
            let diff = oracle.diff(&candidate);
            println!("{}", serde_json::to_string_pretty(&diff)?);
            if !diff.is_empty() {
                return Err("candidate differs from the Qwen3.8 executable oracle".into());
            }
        }
        Some("checkpoint") => {
            let model_dir = args.next().ok_or_else(usage)?;
            no_more(args)?;
            let report = Qwen38Fp8Checkpoint::inspect(model_dir)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Some("weights") => {
            let model_dir = args.next().ok_or_else(usage)?;
            no_more(args)?;
            let checkpoint = Qwen38Fp8Checkpoint::inspect(&model_dir)?;
            let contracts = Qwen38Fp8Checkpoint::expected_text_tensors();
            let plan = Qwen38Fp8WeightPlan::lower();
            plan.validate(&contracts)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "checkpoint": checkpoint,
                    "bound_buffers": plan.bound.len(),
                    "fp8_execution_buffers": plan.fp8_buffers(),
                    "derived_buffers": plan.derived.len(),
                    "load_transforms": plan.transforms.len(),
                }))?
            );
        }
        Some("provider-contract") => {
            let path = args.next().ok_or_else(usage)?;
            no_more(args)?;
            let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
            let source = value
                .pointer("/artifacts/source")
                .and_then(serde_json::Value::as_str)
                .ok_or("qualification record has no artifacts.source")?;
            let cubin = value
                .pointer("/artifacts/cubin")
                .and_then(serde_json::Value::as_str)
                .ok_or("qualification record has no artifacts.cubin")?;
            let contract = value.get("contract").cloned().ok_or("qualification record has no contract")?;
            let contract: orbitkv_compiler::compiler::provider::DeepGemmKernelContract =
                serde_json::from_value(contract)?;
            DeepGemmSm90Capability::h20().validate_contract(&contract)?;
            contract.verify_artifacts(&fs::read(source)?, &fs::read(cubin)?)?;
            println!("{}", serde_json::to_string_pretty(&contract)?);
        }
        Some("projection-manifest") => {
            let path = args.next().ok_or_else(usage)?;
            no_more(args)?;
            let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
            let source = value
                .pointer("/artifacts/source")
                .and_then(serde_json::Value::as_str)
                .ok_or("qualification record has no artifacts.source")?;
            let cubin = value
                .pointer("/artifacts/cubin")
                .and_then(serde_json::Value::as_str)
                .ok_or("qualification record has no artifacts.cubin")?;
            let module_source = std::path::Path::new(cubin)
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("qualification cubin has no UTF-8 file name")?;
            let contract = value.get("contract").cloned().ok_or("qualification record has no contract")?;
            let contract: orbitkv_compiler::compiler::provider::DeepGemmKernelContract =
                serde_json::from_value(contract)?;
            DeepGemmSm90Capability::h20().validate_contract(&contract)?;
            contract.verify_artifacts(&fs::read(source)?, &fs::read(cubin)?)?;
            let artifact = lower_deepgemm_projection_probe(&contract, module_source)?;
            println!("{}", artifact.to_json());
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn run_provider(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    if args.next().as_deref() != Some("deepgemm-h20") {
        return Err(usage().into());
    }
    let operation = args.next().ok_or_else(usage)?;
    let rows = parse_u32(args.next(), "rows")?;
    let output_features = parse_u32(args.next(), "output features")?;
    let input_features = parse_u32(args.next(), "input features")?;
    no_more(args)?;
    let contract = DeepGemmSm90Capability::h20()
        .preferred_candidate(rows, Fp8ProjectionShape { output_features, input_features })?;
    match operation.as_str() {
        "plan" => println!("{}", serde_json::to_string_pretty(&contract)?),
        "source" => print!("{}", render_aot_source(&contract)?),
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn parse_u32(value: Option<String>, name: &str) -> Result<u32, Box<dyn std::error::Error>> {
    value.ok_or_else(usage)?.parse().map_err(|_| format!("invalid {name}").into())
}

fn no_more(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    if args.next().is_some() { Err(usage().into()) } else { Ok(()) }
}

fn usage() -> &'static str {
    "usage: orbitkv oracle inspect [manifest]\n       orbitkv oracle validate [qwen38-bf16-manifest]\n       orbitkv oracle diff <candidate> [reference]\n       orbitkv oracle checkpoint <qwen38-fp8-model-dir>\n       orbitkv oracle weights <qwen38-fp8-model-dir>\n       orbitkv oracle provider-contract <qualification.json>\n       orbitkv oracle projection-manifest <qualification.json>\n       orbitkv provider deepgemm-h20 plan <rows> <output-features> <input-features>\n       orbitkv provider deepgemm-h20 source <rows> <output-features> <input-features>"
}
