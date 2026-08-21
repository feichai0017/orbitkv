use std::env;
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use orbitkv::{HfRetentionOptions, KvPlanInput, compile_hf_manager_plan, compile_plan};
use serde::Serialize;

const USAGE: &str = "usage:\n  orbitkv compile-plan <plan.json>\n  orbitkv compile-hf-manager-plan <config.json> --page-tokens <tokens> --kv-dtype-bytes <bytes>";

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
        Some("compile-hf-manager-plan") => compile_hf_manager_plan_command(&mut args),
        _ => Err(USAGE.into()),
    }
}

fn compile_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "plan path")?;
    require_end(args)?;
    let input = serde_json::from_slice::<KvPlanInput>(&std::fs::read(path)?)?;
    write_json(&compile_plan(input)?)
}

fn compile_hf_manager_plan_command(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let options = HfRetentionOptions {
        page_tokens: page_tokens.ok_or("missing --page-tokens")?,
        kv_dtype_bytes: kv_dtype_bytes.ok_or("missing --kv-dtype-bytes")?,
    };
    let input = compile_hf_manager_plan(&std::fs::read(path)?, options)?;
    write_json(&input)
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
