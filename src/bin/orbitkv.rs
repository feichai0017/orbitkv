use std::env;
use std::fs;
use std::process::ExitCode;

use kern_manifest::Verified;
use orbitkv_compiler::lower::Qwen38Fp8WeightPlan;
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
    if args.next().as_deref() != Some("oracle") {
        return Err(usage().into());
    }
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
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn no_more(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    if args.next().is_some() { Err(usage().into()) } else { Ok(()) }
}

fn usage() -> &'static str {
    "usage: orbitkv oracle inspect [manifest]\n       orbitkv oracle validate [qwen38-bf16-manifest]\n       orbitkv oracle diff <candidate> [reference]\n       orbitkv oracle checkpoint <qwen38-fp8-model-dir>\n       orbitkv oracle weights <qwen38-fp8-model-dir>"
}
