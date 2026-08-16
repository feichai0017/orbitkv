use std::env;
use std::process::ExitCode;

use cudarc::driver::CudaContext;
use orbitkv_cuda::{CudaVmmSlot, probe};
use serde::Serialize;

const REMAP_CYCLES: usize = 64;

#[derive(Serialize)]
struct VmmSmokeReport {
    schema: &'static str,
    device_ordinal: usize,
    device_name: String,
    compute_capability: [i32; 2],
    virtual_address_management: bool,
    minimum_granularity_bytes: usize,
    recommended_granularity_bytes: usize,
    requested_bytes: usize,
    reserved_bytes: usize,
    remap_cycles: usize,
    first_address: u64,
    final_address: u64,
    passed_checks: Vec<&'static str>,
    failed_checks: Vec<&'static str>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("orbitkv-cuda: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("probe") => {
            let device = optional_device(&mut args)?;
            require_end(&mut args)?;
            serde_json::to_writer_pretty(std::io::stdout(), &probe(device)?)?;
            println!();
        }
        Some("vmm-smoke") => {
            let requested_bytes = args
                .next()
                .as_deref()
                .unwrap_or("1048576")
                .parse::<usize>()?;
            let device = optional_device(&mut args)?;
            require_end(&mut args)?;
            vmm_smoke(device, requested_bytes)?;
        }
        _ => return Err("usage: orbitkv-cuda <probe|vmm-smoke [bytes]> [--device N]".into()),
    }
    Ok(())
}

fn vmm_smoke(
    device_ordinal: usize,
    requested_bytes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let capabilities = probe(device_ordinal)?;
    let context = CudaContext::new(device_ordinal)?;
    let mut slot = CudaVmmSlot::reserve(context, requested_bytes)?;
    let first_address = slot.address();

    let sample_bytes = requested_bytes.min(4096);
    let mut all_addresses_stable = true;
    let mut all_patterns_verified = true;
    let mut first_output = Vec::new();
    let mut final_output = Vec::new();
    for cycle in 0..REMAP_CYCLES {
        slot.map_fresh()?;
        all_addresses_stable &= slot.address() == first_address;
        let pattern = (0..sample_bytes)
            .map(|index| u8::try_from((index * 17 + cycle * 29 + 7) % 251))
            .collect::<Result<Vec<_>, _>>()?;
        slot.write(&pattern)?;
        let mut output = vec![0_u8; sample_bytes];
        slot.read(&mut output)?;
        all_patterns_verified &= pattern == output;
        if cycle == 0 {
            first_output.clone_from(&output);
        }
        final_output = output;
        if cycle + 1 < REMAP_CYCLES {
            slot.unmap()?;
        }
    }
    let final_address = slot.address();
    let physical_backing_replaced = first_output != final_output;
    let reserved_bytes = slot.bytes();
    slot.close()?;

    let mut passed_checks = Vec::new();
    let mut failed_checks = Vec::new();
    for (name, passed) in [
        ("stable_virtual_address", all_addresses_stable),
        ("all_patterns_verified", all_patterns_verified),
        ("physical_backing_replaced", physical_backing_replaced),
    ] {
        if passed {
            passed_checks.push(name);
        } else {
            failed_checks.push(name);
        }
    }
    let report = VmmSmokeReport {
        schema: "orbitkv.cuda-vmm-smoke.v1",
        device_ordinal,
        device_name: capabilities.device_name,
        compute_capability: [
            capabilities.compute_capability_major,
            capabilities.compute_capability_minor,
        ],
        virtual_address_management: capabilities.virtual_address_management,
        minimum_granularity_bytes: capabilities.minimum_granularity_bytes,
        recommended_granularity_bytes: capabilities.recommended_granularity_bytes,
        requested_bytes,
        reserved_bytes,
        remap_cycles: REMAP_CYCLES,
        first_address,
        final_address,
        passed_checks,
        failed_checks,
    };
    if !report.failed_checks.is_empty() {
        return Err("CUDA VMM smoke invariants failed".into());
    }
    serde_json::to_writer_pretty(std::io::stdout(), &report)?;
    println!();
    Ok(())
}

fn optional_device(
    args: &mut impl Iterator<Item = String>,
) -> Result<usize, Box<dyn std::error::Error>> {
    match args.next() {
        None => Ok(0),
        Some(flag) if flag == "--device" => Ok(args
            .next()
            .ok_or("missing device ordinal")?
            .parse::<usize>()?),
        Some(argument) => Err(format!("unexpected argument {argument}").into()),
    }
}

fn require_end(args: &mut impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(argument) = args.next() {
        return Err(format!("unexpected argument {argument}").into());
    }
    Ok(())
}
