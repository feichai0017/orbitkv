use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;

use cudarc::driver::{CudaContext, CudaStream, result};
use orbitkv::{
    BlockHandle, BlockManagerConfig, ClassPoolConfig, KvBlockManager, KvClassSpec, KvPlanInput,
    RetentionKind, compile_plan,
};
use orbitkv_cuda::{CudaExecutionFrontier, CudaVmmBlockPool, CudaVmmSlot, probe};
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

#[derive(Serialize)]
struct GenerationLifecycleReport {
    schema: &'static str,
    device_ordinal: usize,
    cycles: u64,
    slot_count: usize,
    slot_bytes: usize,
    stable_addresses: BTreeMap<u64, u64>,
    maximum_physical_generation: u64,
    maximum_temporal_cycle: u64,
    temporal_cycle_checks: u64,
    physical_generation_checks: u64,
    completed_submissions: u64,
    committed_reclamations: u64,
    verified_patterns: u64,
    stale_generation_rejections: u64,
    passed_checks: Vec<&'static str>,
    failed_checks: Vec<&'static str>,
}

struct GenerationState {
    stable_addresses: BTreeMap<u64, u64>,
    maximum_physical_generation: u64,
    maximum_temporal_cycle: u64,
    temporal_cycle_checks: u64,
    physical_generation_checks: u64,
    completed_submissions: u64,
    committed_reclamations: u64,
    verified_patterns: u64,
    stale_generation_rejections: u64,
    previous_handles: BTreeMap<u64, BlockHandle>,
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
        Some("generation-smoke") => {
            let cycles = args.next().as_deref().unwrap_or("32").parse::<u64>()?;
            let device = optional_device(&mut args)?;
            require_end(&mut args)?;
            generation_smoke(device, cycles)?;
        }
        _ => {
            return Err(
                "usage: orbitkv-cuda <probe|vmm-smoke [bytes]|generation-smoke [cycles]> [--device N]"
                    .into(),
            );
        }
    }
    Ok(())
}

fn generation_smoke(device_ordinal: usize, cycles: u64) -> Result<(), Box<dyn std::error::Error>> {
    if cycles == 0 {
        return Err("cycles must be positive".into());
    }
    let mut manager = generation_manager()?;
    manager.register_request("cuda-generation-smoke")?;
    let mut pool = CudaVmmBlockPool::new(device_ordinal, "swa", 2, 1 << 20)?;
    let context = CudaContext::new(device_ordinal)?;
    let stream = context.new_stream()?;
    let mut execution = CudaExecutionFrontier::new();
    let mut state = GenerationState {
        stable_addresses: BTreeMap::new(),
        maximum_physical_generation: 0,
        maximum_temporal_cycle: 0,
        temporal_cycle_checks: 0,
        physical_generation_checks: 0,
        completed_submissions: 0,
        committed_reclamations: 0,
        verified_patterns: 0,
        stale_generation_rejections: 0,
        previous_handles: BTreeMap::new(),
    };

    initialize_generation_lifecycle(&mut manager, &mut pool, &mut state)?;
    for cycle in 0..cycles {
        run_generation_cycle(
            cycle,
            &mut manager,
            &mut pool,
            &stream,
            &mut execution,
            &mut state,
        )?;
    }
    finalize_generation_lifecycle(&mut manager, &mut pool, &mut state)?;
    let manager_empty = manager.stats().resident_blocks == 0;
    let no_pending_events = execution.pending() == 0;
    pool.close()?;
    write_generation_report(
        device_ordinal,
        cycles,
        state,
        manager_empty,
        no_pending_events,
    )
}

fn generation_manager() -> Result<KvBlockManager, Box<dyn std::error::Error>> {
    let plan = compile_plan(KvPlanInput {
        page_tokens: 1,
        classes: vec![KvClassSpec {
            name: "swa".into(),
            layers: vec![0],
            retention: RetentionKind::Sliding,
            bytes_per_token_per_layer: 1,
            window_tokens: Some(2),
        }],
    })?;
    Ok(KvBlockManager::new(
        plan,
        BlockManagerConfig {
            pools: vec![ClassPoolConfig {
                class_name: "swa".into(),
                slot_count: 2,
            }],
        },
    )?)
}

fn initialize_generation_lifecycle(
    manager: &mut KvBlockManager,
    pool: &mut CudaVmmBlockPool,
    state: &mut GenerationState,
) -> Result<(), Box<dyn std::error::Error>> {
    let initial = manager
        .materialize_to("cuda-generation-smoke", 1)?
        .into_iter()
        .next()
        .ok_or("manager did not materialize the initial block")?;
    let initial_address = pool.activate(&initial.physical)?;
    state
        .stable_addresses
        .insert(initial.physical.slot, initial_address.address);
    state
        .previous_handles
        .insert(initial.physical.slot, initial.physical);
    manager.advance_semantic_frontier("cuda-generation-smoke", 1)?;
    Ok(())
}

fn finalize_generation_lifecycle(
    manager: &mut KvBlockManager,
    pool: &mut CudaVmmBlockPool,
    state: &mut GenerationState,
) -> Result<(), Box<dyn std::error::Error>> {
    let final_certificates = manager.release_request("cuda-generation-smoke")?;
    for certificate in final_certificates {
        let receipt = pool.reclaim(&certificate)?;
        manager.commit_reclamation(&receipt)?;
        state.committed_reclamations += 1;
    }
    Ok(())
}

fn write_generation_report(
    device_ordinal: usize,
    cycles: u64,
    state: GenerationState,
    manager_empty: bool,
    no_pending_events: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed_checks = Vec::new();
    let mut failed_checks = Vec::new();
    for (name, passed) in [
        ("stable_virtual_address", state.stable_addresses.len() == 2),
        (
            "temporal_cycle_progress",
            state.temporal_cycle_checks == cycles,
        ),
        (
            "physical_generation_progress",
            state.physical_generation_checks == cycles.saturating_sub(1),
        ),
        (
            "cuda_event_completion",
            state.completed_submissions == cycles,
        ),
        ("receipt_commit", state.committed_reclamations == cycles + 1),
        ("data_patterns", state.verified_patterns == cycles),
        (
            "stale_generation_rejected",
            state.stale_generation_rejections == cycles.saturating_sub(1),
        ),
        ("manager_empty", manager_empty),
        ("no_pending_events", no_pending_events),
    ] {
        if passed {
            passed_checks.push(name);
        } else {
            failed_checks.push(name);
        }
    }
    let report = GenerationLifecycleReport {
        schema: "orbitkv.cuda-generation-lifecycle.v1",
        device_ordinal,
        cycles,
        slot_count: 2,
        slot_bytes: 2 << 20,
        stable_addresses: state.stable_addresses,
        maximum_physical_generation: state.maximum_physical_generation,
        maximum_temporal_cycle: state.maximum_temporal_cycle,
        temporal_cycle_checks: state.temporal_cycle_checks,
        physical_generation_checks: state.physical_generation_checks,
        completed_submissions: state.completed_submissions,
        committed_reclamations: state.committed_reclamations,
        verified_patterns: state.verified_patterns,
        stale_generation_rejections: state.stale_generation_rejections,
        passed_checks,
        failed_checks,
    };
    if !report.failed_checks.is_empty() {
        return Err("generation lifecycle invariants failed".into());
    }
    serde_json::to_writer_pretty(std::io::stdout(), &report)?;
    println!();
    Ok(())
}

fn run_generation_cycle(
    cycle: u64,
    manager: &mut KvBlockManager,
    pool: &mut CudaVmmBlockPool,
    stream: &CudaStream,
    execution: &mut CudaExecutionFrontier,
    state: &mut GenerationState,
) -> Result<(), Box<dyn std::error::Error>> {
    let frontier = cycle + 1;
    let view = manager.submit_view("cuda-generation-smoke")?;
    if view.blocks.len() != 1 {
        return Err(format!(
            "expected one live block at frontier {frontier}, got {}",
            view.blocks.len()
        )
        .into());
    }
    let current = &view.blocks[0];
    let address = pool.address(&current.physical)?;
    let pattern = u8::try_from(cycle % 251)?;
    unsafe {
        result::memset_d8_async(address.address, pattern, address.bytes, stream.cu_stream())?;
    }
    execution.record(view.submission_id, stream)?;

    let next_boundary = frontier + 1;
    let next = manager
        .materialize_to("cuda-generation-smoke", next_boundary)?
        .into_iter()
        .next()
        .ok_or("manager did not materialize the next block")?;
    let next_address = pool.activate(&next.physical)?;
    if let Some(previous) = state
        .previous_handles
        .insert(next.physical.slot, next.physical.clone())
    {
        if next.physical.generation != previous.generation + 1 {
            return Err("physical generation did not advance monotonically".into());
        }
        state.physical_generation_checks += 1;
        if pool.address(&previous).is_ok() {
            return Err("stale physical generation remained addressable".into());
        }
        state.stale_generation_rejections += 1;
    }
    if let Some(expected) = state.stable_addresses.get(&next.physical.slot) {
        if next_address.address != *expected {
            return Err("VMM address changed across physical generations".into());
        }
    } else {
        state
            .stable_addresses
            .insert(next.physical.slot, next_address.address);
    }
    state.maximum_physical_generation = state
        .maximum_physical_generation
        .max(next.physical.generation);
    state.maximum_temporal_cycle = state
        .maximum_temporal_cycle
        .max(next.temporal.version.cycle);
    let expected_cycle = next.logical.ordinal / 2;
    if next.temporal.version.cycle != expected_cycle {
        return Err(format!(
            "temporal cycle {} does not match expected {}",
            next.temporal.version.cycle, expected_cycle
        )
        .into());
    }
    state.temporal_cycle_checks += 1;
    manager.advance_semantic_frontier("cuda-generation-smoke", next_boundary)?;
    execution.wait(view.submission_id)?;
    let mut sample = [0_u8; 64];
    pool.read(&current.physical, &mut sample)?;
    if sample.iter().any(|&value| value != pattern) {
        return Err("CUDA generation data pattern mismatch".into());
    }
    state.verified_patterns += 1;
    let certificates = manager.complete_submission(view.submission_id)?;
    state.completed_submissions += 1;
    if certificates.len() != 1 {
        return Err(format!(
            "expected one retirement certificate, got {}",
            certificates.len()
        )
        .into());
    }
    let receipt = pool.reclaim(&certificates[0])?;
    manager.commit_reclamation(&receipt)?;
    state.committed_reclamations += 1;
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
