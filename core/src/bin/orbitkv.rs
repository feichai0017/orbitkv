use std::env;
use std::io::{BufWriter, Read, Write};
use std::process::ExitCode;

use orbitkv::{
    AttentionStatePlanInput, HfRetentionOptions, KvPlanInput, RetentionProgramInput,
    RuntimeManifest, admit_runtime_manifest, compile_attention_state_manager_plan,
    compile_attention_state_plan, compile_hf_attention_state_input,
    compile_hf_attention_state_plan, compile_hf_runtime_manifest, compile_hf_token_manager_plan,
    compile_plan, compile_retention_runtime_manifest, compile_runtime_manifest,
};
use serde::Serialize;

const USAGE: &str = "usage:\n  orbitkv compile-plan <plan.json>\n  orbitkv compile-state-plan <state-plan.json>\n  orbitkv compile-state-manager-plan <state-plan.json>\n  orbitkv compile-runtime-manifest <state-plan.json>\n  orbitkv compile-retention-runtime-manifest <retention-ir.json>\n  orbitkv bind-runtime-manifest <manifest.json>\n  orbitkv check-runtime-manifest <manifest.json>\n  orbitkv compile-hf-state-input <config.json> --page-tokens <tokens> --kv-dtype-bytes <bytes>\n  orbitkv compile-hf-state-plan <config.json> --page-tokens <tokens> --kv-dtype-bytes <bytes>\n  orbitkv compile-hf-token-manager-plan <config.json> --page-tokens <tokens> --kv-dtype-bytes <bytes>\n  orbitkv compile-hf-runtime-manifest <config.json> --page-tokens <tokens> --kv-dtype-bytes <bytes>\n\nnotes:\n  compile-runtime-manifest emits the canonical executable artifact.\n  compile-retention-runtime-manifest compiles Retention IR into the same canonical artifact.\n  bind-runtime-manifest emits a static binding to the packaged SGLang target.\n  check-runtime-manifest validates the same binding without emitting an artifact.\n  compile-hf-runtime-manifest compiles a supported HF config into the canonical artifact.\n  compile-hf-state-input emits declarative attention-state compiler input.\n  compile-hf-state-plan emits compiled backend contracts.\n  compile-hf-token-manager-plan emits only token-addressable state; recurrent and convolution state are omitted.";

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
        Some("compile-plan") => compile_plan_command(&mut args),
        Some("compile-state-plan") => compile_state_plan_command(&mut args),
        Some("compile-state-manager-plan") => compile_state_manager_plan_command(&mut args),
        Some("compile-runtime-manifest") => compile_runtime_manifest_command(&mut args),
        Some("compile-retention-runtime-manifest") => {
            compile_retention_runtime_manifest_command(&mut args)
        }
        Some("bind-runtime-manifest") => bind_runtime_manifest_command(&mut args, true),
        Some("check-runtime-manifest") => bind_runtime_manifest_command(&mut args, false),
        Some("compile-hf-state-input") => compile_hf_state_input_command(&mut args),
        Some("compile-hf-state-plan") => compile_hf_state_plan_command(&mut args),
        Some("compile-hf-runtime-manifest") => compile_hf_runtime_manifest_command(&mut args),
        Some("compile-hf-token-manager-plan") => compile_hf_token_manager_plan_command(&mut args),
        _ => Err(USAGE.into()),
    }
}

fn compile_retention_runtime_manifest_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "retention IR path")?;
    require_end(args)?;
    let input = serde_json::from_slice::<RetentionProgramInput>(&read_bounded(
        &path,
        orbitkv::RUNTIME_MANIFEST_MAX_BYTES,
    )?)?;
    write_json(&compile_retention_runtime_manifest(input)?)
}

fn bind_runtime_manifest_command(
    args: &mut impl Iterator<Item = String>,
    emit: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = required(args, "runtime manifest path")?;
    require_end(args)?;
    let manifest = RuntimeManifest::from_json(&read_bounded(
        &manifest_path,
        orbitkv::RUNTIME_MANIFEST_MAX_BYTES,
    )?)?;
    let binding = admit_runtime_manifest(&manifest)?;
    if emit {
        write_json(&binding)?;
    }
    Ok(())
}

fn read_bounded(path: &str, maximum: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() > u64::try_from(maximum)? {
        return Err(format!("artifact {path:?} exceeds the {maximum}-byte limit").into());
    }
    let limit = u64::try_from(maximum)?
        .checked_add(1)
        .ok_or("artifact size limit overflow")?;
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len())?);
    file.take(limit).read_to_end(&mut bytes)?;
    if bytes.len() > maximum {
        return Err(format!("artifact {path:?} exceeds the {maximum}-byte limit").into());
    }
    Ok(bytes)
}

fn compile_runtime_manifest_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "attention-state plan path")?;
    require_end(args)?;
    let input = serde_json::from_slice::<AttentionStatePlanInput>(&std::fs::read(path)?)?;
    write_json(&compile_runtime_manifest(input)?)
}

fn compile_hf_runtime_manifest_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (path, options) = parse_hf_args(args)?;
    write_json(&compile_hf_runtime_manifest(
        &std::fs::read(path)?,
        options,
    )?)
}

fn compile_hf_state_input_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (path, options) = parse_hf_args(args)?;
    write_json(&compile_hf_attention_state_input(
        &std::fs::read(path)?,
        options,
    )?)
}

fn compile_hf_state_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (path, options) = parse_hf_args(args)?;
    write_json(&compile_hf_attention_state_plan(
        &std::fs::read(path)?,
        options,
    )?)
}

fn compile_state_manager_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "attention-state plan path")?;
    require_end(args)?;
    let input = serde_json::from_slice::<AttentionStatePlanInput>(&std::fs::read(path)?)?;
    write_json(&compile_attention_state_manager_plan(input)?)
}

fn compile_state_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "attention-state plan path")?;
    require_end(args)?;
    let input = serde_json::from_slice::<AttentionStatePlanInput>(&std::fs::read(path)?)?;
    write_json(&compile_attention_state_plan(input)?)
}

fn compile_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "plan path")?;
    require_end(args)?;
    let input = serde_json::from_slice::<KvPlanInput>(&std::fs::read(path)?)?;
    write_json(&compile_plan(input)?)
}

fn compile_hf_token_manager_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (path, options) = parse_hf_args(args)?;
    let input = compile_hf_token_manager_plan(&std::fs::read(path)?, options)?;
    write_json(&input)
}

fn parse_hf_args(
    args: &mut impl Iterator<Item = String>,
) -> Result<(String, HfRetentionOptions), Box<dyn std::error::Error>> {
    let path = required(args, "HF config path")?;
    let mut page_tokens = None;
    let mut kv_dtype_bytes = None;
    while let Some(flag) = args.next() {
        let destination = match flag.as_str() {
            "--page-tokens" => &mut page_tokens,
            "--kv-dtype-bytes" => &mut kv_dtype_bytes,
            _ => return Err(format!("unexpected argument {flag}").into()),
        };
        if destination.is_some() {
            return Err(format!("duplicate argument {flag}").into());
        }
        *destination = Some(required(args, &format!("value for {flag}"))?.parse::<u64>()?);
    }
    Ok((
        path,
        HfRetentionOptions {
            page_tokens: page_tokens.ok_or("missing --page-tokens")?,
            kv_dtype_bytes: kv_dtype_bytes.ok_or("missing --kv-dtype-bytes")?,
        },
    ))
}

fn required(
    args: &mut impl Iterator<Item = String>,
    label: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next().ok_or_else(|| format!("missing {label}").into())
}

fn require_end(args: &mut impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(argument) = args.next() {
        return Err(format!("unexpected argument {argument}").into());
    }
    Ok(())
}

fn write_json(value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let mut output = BufWriter::new(std::io::stdout().lock());
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}
