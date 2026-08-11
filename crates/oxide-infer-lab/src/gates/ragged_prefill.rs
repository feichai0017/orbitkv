use super::{VALID_GRAPH_REPLAYS, valid_graph_commands};
use crate::comparison::{compare_bf16, compare_f32};
use crate::fixture::deterministic_bf16;
use crate::reporting::GateCase;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use oxide_infer::{Bf16RaggedPrefillSpec, ragged_prefill_bf16_reference};
use oxide_infer_cuda::attention::{
    Bf16RaggedPrefillAlgorithm, Bf16RaggedPrefillArgs, PrefillProvider, RaggedPrefillEnqueueError,
};
use oxide_infer_cuda::command::CommandQueue;
use oxide_infer_cuda::graph::GraphQueue;
use std::error::Error;
use std::sync::Arc;

const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const LSE_MAX_ABS_LIMIT: f32 = 0.01;
const TILED_GRAPH_COMMANDS_PER_STAGE: usize = 2;

fn run_case(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
    name: &str,
    spec: Bf16RaggedPrefillSpec,
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    salt: u64,
) -> Result<(), Box<dyn Error>> {
    spec.validate_metadata(qo_indptr, kv_indptr)?;
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_ragged(spec)?;
    let algorithm = match plan.algorithm() {
        Bf16RaggedPrefillAlgorithm::Direct => "direct",
        Bf16RaggedPrefillAlgorithm::TokenParallel8 => "token_parallel_8warp",
        Bf16RaggedPrefillAlgorithm::TokenParallel16 => "token_parallel_16warp",
        Bf16RaggedPrefillAlgorithm::TiledGqa4 => "tiled_gqa4_mma",
    };
    let query_host = deterministic_bf16(spec.query_numel(), salt);
    let key_host = deterministic_bf16(spec.kv_numel(), salt ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_numel(), salt ^ 0x5641_4c55_4500);
    let mut expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut expected_lse = vec![f32::NAN; spec.lse_numel()];
    ragged_prefill_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        qo_indptr,
        kv_indptr,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;

    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, qo_indptr)?);
    let kv_indptr = Arc::new(DeviceBuffer::from_host(&stream, kv_indptr)?);
    let output = DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.output_numel()])?;
    let lse = DeviceBuffer::from_host(&stream, &vec![f32::NAN; spec.lse_numel()])?;
    let workspace =
        DeviceBuffer::<f32>::zeroed(&stream, usize::max(plan.workspace_required_numel(), 1))?;
    let mut bindings = queue.bindings(8)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let qo_indptr_handle = bindings.bind_read(qo_indptr)?;
    let kv_indptr_handle = bindings.bind_read(kv_indptr)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let mut scope = queue.begin(bindings)?;
    let mut args = Bf16RaggedPrefillArgs::new(
        query_handle,
        key_handle,
        value_handle,
        qo_indptr_handle,
        kv_indptr_handle,
        output_handle.write(),
        lse_handle.write(),
    );
    let expected_commands = if plan.workspace_required_numel() == 0 {
        1
    } else {
        args = args.with_workspace(workspace_handle);
        2
    };
    plan.enqueue_into(&mut scope, args)?;
    let completion = scope.finish();
    if completion.submitted() != expected_commands {
        return Err("ragged prefill completion covered the wrong command count".into());
    }
    bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);

    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    let output_comparison = compare_bf16(&actual_output, &expected_output, "prefill BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "prefill F32 LSE")?;
    if output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT {
        return Err(format!(
            "{name} output max abs {:.9e} exceeds {:.9e}",
            output_comparison.max_abs, OUTPUT_MAX_ABS_LIMIT
        )
        .into());
    }
    if lse_comparison.max_abs > LSE_MAX_ABS_LIMIT {
        return Err(format!(
            "{name} LSE max abs {:.9e} exceeds {:.9e}",
            lse_comparison.max_abs, LSE_MAX_ABS_LIMIT
        )
        .into());
    }

    println!(
        "{} batch_size={} nnz_qo={} nnz_kv={} query_heads={} kv_heads={} \
         group_size={} head_dim={} layout=NHD causal=bottom_right dtype=BF16 \
         accumulation=F32 lse_domain=log2 algorithm={} commands={} stream=non_default \
         output_max_abs={:.9e} output_bit_mismatches={} output_digest={:016x} \
         lse_max_abs={:.9e} lse_bit_mismatches={} lse_digest={:016x}",
        GateCase::new("ragged_prefill_h20", name),
        spec.batch_size(),
        spec.nnz_qo(),
        spec.nnz_kv(),
        spec.num_query_heads(),
        spec.num_kv_heads(),
        spec.gqa_group_size(),
        spec.head_dim(),
        algorithm,
        expected_commands,
        output_comparison.max_abs,
        output_comparison.bit_mismatches,
        output_comparison.digest,
        lse_comparison.max_abs,
        lse_comparison.bit_mismatches,
        lse_comparison.digest,
    );
    Ok(())
}

fn run_short_indptr_case(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RaggedPrefillSpec::new(2, 2, 2, 4, 2, 128)?;
    let plan = provider.plan_bf16_ragged(spec)?;
    let stream = queue.stream().clone();
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_numel())?);
    let value = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_numel())?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 2])?);
    let kv_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1, 2])?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(7)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let kv_handle = bindings.bind_read(kv_indptr)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16RaggedPrefillArgs::new(
                query_handle,
                key_handle,
                value_handle,
                qo_handle,
                kv_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
        .expect_err("short qo_indptr must fail before submission");
    if !matches!(
        error,
        RaggedPrefillEnqueueError::LengthMismatch {
            operand: "qo_indptr",
            expected: 3,
            actual: 2,
        }
    ) {
        return Err(format!("short qo_indptr returned the wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("short qo_indptr reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!(
        "{} qo_indptr=rejected before_ffi=true",
        GateCase::new("ragged_prefill_h20", "short_metadata")
    );
    Ok(())
}

fn run_invalid_metadata_guard(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RaggedPrefillSpec::new(2, 1, 2, 1, 1, 128)?;
    let plan = provider.plan_bf16_ragged(spec)?;
    let stream = queue.stream().clone();
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_numel())?);
    let value = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_numel())?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 2, 1])?);
    let kv_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1, 2])?);
    let output_sentinel = vec![bf16::NAN; spec.output_numel()];
    let lse_sentinel = vec![f32::NAN; spec.lse_numel()];
    let output = DeviceBuffer::from_host(&stream, &output_sentinel)?;
    let lse = DeviceBuffer::from_host(&stream, &lse_sentinel)?;
    let mut bindings = queue.bindings(7)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let kv_handle = bindings.bind_read(kv_indptr)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16RaggedPrefillArgs::new(
            query_handle,
            key_handle,
            value_handle,
            qo_handle,
            kv_handle,
            output_handle.write(),
            lse_handle.write(),
        ),
    )?;
    bindings = scope.finish().wait()?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);
    if output
        .to_host_vec(&stream)?
        .iter()
        .any(|value| !value.to_f32().is_nan())
        || lse
            .to_host_vec(&stream)?
            .iter()
            .any(|value| !value.is_nan())
    {
        return Err("invalid metadata did not preserve output sentinels".into());
    }
    println!(
        "{} nonmonotonic_indptr=guarded sentinel_preserved=true",
        GateCase::new("ragged_prefill_h20", "invalid_metadata_guard")
    );
    Ok(())
}

fn run_missing_workspace_case(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RaggedPrefillSpec::new(1, 1, 256, 4, 1, 128)?;
    let plan = provider.plan_bf16_ragged(spec)?;
    if plan.algorithm() != Bf16RaggedPrefillAlgorithm::TiledGqa4
        || plan.workspace_required_numel() == 0
    {
        return Err("missing-workspace gate did not select tiled GQA4".into());
    }
    let stream = queue.stream().clone();
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_numel())?);
    let value = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_numel())?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1])?);
    let kv_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 256])?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(7)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let kv_handle = bindings.bind_read(kv_indptr)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16RaggedPrefillArgs::new(
                query_handle,
                key_handle,
                value_handle,
                qo_handle,
                kv_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
        .expect_err("tiled GQA4 must require an explicit workspace");
    if !matches!(error, RaggedPrefillEnqueueError::MissingWorkspace) {
        return Err(format!("missing workspace returned the wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("missing workspace reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!(
        "{} workspace=required before_submission=true",
        GateCase::new("ragged_prefill_h20", "missing_workspace")
    );
    Ok(())
}

fn run_tiled_graph_case(
    queue: &mut CommandQueue,
    provider: PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RaggedPrefillSpec::new(2, 96, 1280, 16, 4, 128)?;
    let qo_indptr_host = [0_i32, 32, 96];
    let kv_indptr_host = [0_i32, 256, 1280];
    spec.validate_metadata(&qo_indptr_host, &kv_indptr_host)?;
    let plan = provider.plan_bf16_ragged(spec)?;
    if plan.algorithm() != Bf16RaggedPrefillAlgorithm::TiledGqa4
        || plan.workspace_required_numel() == 0
    {
        return Err("ragged graph gate did not select tiled GQA4".into());
    }

    let stream = queue.stream().clone();
    let query_host = deterministic_bf16(spec.query_numel(), 0x4001);
    let key_host = deterministic_bf16(spec.kv_numel(), 0x4001 ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_numel(), 0x4001 ^ 0x5641_4c55_4500);
    let poison_query_host = vec![bf16::ZERO; spec.query_numel()];
    let poison_key_host = vec![bf16::ZERO; spec.kv_numel()];
    let poison_value_host = vec![bf16::ZERO; spec.kv_numel()];
    let mut poison_expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut poison_expected_lse = vec![f32::NAN; spec.lse_numel()];
    let mut expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut expected_lse = vec![f32::NAN; spec.lse_numel()];
    ragged_prefill_bf16_reference(
        &poison_query_host,
        &poison_key_host,
        &poison_value_host,
        &qo_indptr_host,
        &kv_indptr_host,
        &mut poison_expected_output,
        &mut poison_expected_lse,
        spec,
    )?;
    ragged_prefill_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &qo_indptr_host,
        &kv_indptr_host,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;
    if poison_expected_output
        .iter()
        .zip(&expected_output)
        .all(|(poison, expected)| poison.to_bits() == expected.to_bits())
        || poison_expected_lse
            .iter()
            .zip(&expected_lse)
            .all(|(poison, expected)| poison.to_bits() == expected.to_bits())
    {
        return Err("ragged graph poison does not distinguish every checked output".into());
    }

    let poison_query = Arc::new(DeviceBuffer::from_host(&stream, &poison_query_host)?);
    let poison_key = Arc::new(DeviceBuffer::from_host(&stream, &poison_key_host)?);
    let poison_value = Arc::new(DeviceBuffer::from_host(&stream, &poison_value_host)?);
    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &qo_indptr_host)?);
    let kv_indptr = Arc::new(DeviceBuffer::from_host(&stream, &kv_indptr_host)?);
    let initial_output = vec![bf16::NAN; spec.output_numel()];
    let initial_lse = vec![f32::NAN; spec.lse_numel()];
    let poison_observer_output = DeviceBuffer::from_host(&stream, &initial_output)?;
    let poison_observer_lse = DeviceBuffer::from_host(&stream, &initial_lse)?;
    let output = DeviceBuffer::from_host(&stream, &initial_output)?;
    let lse = DeviceBuffer::from_host(&stream, &initial_lse)?;
    let poison_observer_workspace =
        DeviceBuffer::<f32>::zeroed(&stream, plan.workspace_required_numel())?;
    let target_poison_workspace =
        DeviceBuffer::<f32>::zeroed(&stream, plan.workspace_required_numel())?;
    let real_workspace = DeviceBuffer::<f32>::zeroed(&stream, plan.workspace_required_numel())?;

    let graph_commands = valid_graph_commands(TILED_GRAPH_COMMANDS_PER_STAGE);
    let graph_queue = GraphQueue::new(stream.context(), graph_commands)?;
    let mut bindings = graph_queue.bindings(15)?;
    let poison_query_handle = bindings.bind_read(Arc::clone(&poison_query))?;
    let poison_key_handle = bindings.bind_read(Arc::clone(&poison_key))?;
    let poison_value_handle = bindings.bind_read(Arc::clone(&poison_value))?;
    let query_handle = bindings.bind_read(Arc::clone(&query))?;
    let key_handle = bindings.bind_read(Arc::clone(&key))?;
    let value_handle = bindings.bind_read(Arc::clone(&value))?;
    let qo_indptr_handle = bindings.bind_read(Arc::clone(&qo_indptr))?;
    let kv_indptr_handle = bindings.bind_read(Arc::clone(&kv_indptr))?;
    let poison_observer_output_handle = bindings.bind_read_write(poison_observer_output)?;
    let poison_observer_lse_handle = bindings.bind_read_write(poison_observer_lse)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let poison_observer_workspace_handle = bindings.bind_read_write(poison_observer_workspace)?;
    let target_poison_workspace_handle = bindings.bind_read_write(target_poison_workspace)?;
    let real_workspace_handle = bindings.bind_read_write(real_workspace)?;

    let captured = graph_queue.capture(bindings, |scope| {
        plan.enqueue_into(
            scope,
            Bf16RaggedPrefillArgs::new(
                poison_query_handle,
                poison_key_handle,
                poison_value_handle,
                qo_indptr_handle,
                kv_indptr_handle,
                poison_observer_output_handle.write(),
                poison_observer_lse_handle.write(),
            )
            .with_workspace(poison_observer_workspace_handle),
        )?;
        plan.enqueue_into(
            scope,
            Bf16RaggedPrefillArgs::new(
                poison_query_handle,
                poison_key_handle,
                poison_value_handle,
                qo_indptr_handle,
                kv_indptr_handle,
                output_handle.write(),
                lse_handle.write(),
            )
            .with_workspace(target_poison_workspace_handle),
        )?;
        plan.enqueue_into(
            scope,
            Bf16RaggedPrefillArgs::new(
                query_handle,
                key_handle,
                value_handle,
                qo_indptr_handle,
                kv_indptr_handle,
                output_handle.write(),
                lse_handle.write(),
            )
            .with_workspace(real_workspace_handle),
        )
    })?;
    if captured.commands() != graph_commands {
        return Err("ragged graph captured the wrong command count".into());
    }

    drop(plan);
    drop(provider);
    drop(poison_query);
    drop(poison_key);
    drop(poison_value);
    drop(query);
    drop(key);
    drop(value);
    drop(qo_indptr);
    drop(kv_indptr);

    let mut exec = captured.instantiate()?;
    for expected_launch in 1..=VALID_GRAPH_REPLAYS {
        let mut completion = exec.launch()?;
        if completion.launch_index() != expected_launch {
            return Err("ragged graph completion reported the wrong replay index".into());
        }
        let _ = completion.is_complete()?;
        if expected_launch == 1 {
            completion.wait()?;
        } else {
            drop(completion);
        }
    }
    if exec.launches() != VALID_GRAPH_REPLAYS || exec.commands() != graph_commands {
        return Err("ragged graph accounting changed across replay".into());
    }

    let mut bindings = exec.into_bindings()?;
    let poison_observer_output = bindings.take_read_write(poison_observer_output_handle)?;
    let poison_observer_lse = bindings.take_read_write(poison_observer_lse_handle)?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);
    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    let poison_output_comparison = compare_bf16(
        &poison_observer_output.to_host_vec(&stream)?,
        &poison_expected_output,
        "graph prefill poison BF16",
    )?;
    let poison_lse_comparison = compare_f32(
        &poison_observer_lse.to_host_vec(&stream)?,
        &poison_expected_lse,
        "graph prefill poison F32 LSE",
    )?;
    let output_comparison = compare_bf16(&actual_output, &expected_output, "graph prefill BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "graph prefill F32 LSE")?;
    if poison_output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
    {
        return Err(format!(
            "ragged graph output max abs {:.9e} exceeds {:.9e}",
            poison_output_comparison
                .max_abs
                .max(output_comparison.max_abs),
            OUTPUT_MAX_ABS_LIMIT
        )
        .into());
    }
    if poison_lse_comparison.max_abs > LSE_MAX_ABS_LIMIT
        || lse_comparison.max_abs > LSE_MAX_ABS_LIMIT
    {
        return Err(format!(
            "ragged graph LSE max abs {:.9e} exceeds {:.9e}",
            poison_lse_comparison.max_abs.max(lse_comparison.max_abs),
            LSE_MAX_ABS_LIMIT
        )
        .into());
    }

    println!(
        "{} batch_size=2 nnz_qo=96 nnz_kv=1280 query_heads=16 kv_heads=4 \
         head_dim=128 algorithm=tiled_gqa4_mma commands={} commands_per_stage=2 replays={} \
         replay_stages=independent_poison_then_target_poison_then_real \
         independent_poison_observable=true output_lse_poisoned_each_replay=true \
         independent_workspaces=true initial_outputs=poisoned \
         fixed_bindings=true cross_stream=false external_owners_dropped_before_replay=true \
         completion_queries=2 completion_waits=1 completion_drops=1 \
         poison_output_max_abs={:.9e} poison_output_digest={:016x} \
         poison_lse_max_abs={:.9e} poison_lse_digest={:016x} \
         output_max_abs={:.9e} output_bit_mismatches={} output_digest={:016x} \
         lse_max_abs={:.9e} lse_bit_mismatches={} lse_digest={:016x}",
        GateCase::new("ragged_prefill_h20", "gqa4_graph"),
        graph_commands,
        VALID_GRAPH_REPLAYS,
        poison_output_comparison.max_abs,
        poison_output_comparison.digest,
        poison_lse_comparison.max_abs,
        poison_lse_comparison.digest,
        output_comparison.max_abs,
        output_comparison.bit_mismatches,
        output_comparison.digest,
        lse_comparison.max_abs,
        lse_comparison.bit_mismatches,
        lse_comparison.digest,
    );
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let provider = PrefillProvider::load(&context)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let mut queue = CommandQueue::new(stream, 2, 1)?;

    run_case(
        &mut queue,
        &provider,
        "mha_equal_lengths",
        Bf16RaggedPrefillSpec::new(1, 4, 4, 8, 8, 128)?,
        &[0, 4],
        &[0, 4],
        0x1001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "mqa_append_mixed",
        Bf16RaggedPrefillSpec::new(3, 6, 13, 8, 1, 128)?,
        &[0, 2, 5, 6],
        &[0, 4, 10, 13],
        0x2001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "gqa4_mixed",
        Bf16RaggedPrefillSpec::new(2, 6, 11, 16, 4, 128)?,
        &[0, 4, 6],
        &[0, 7, 11],
        0x4001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "mqa_token_parallel",
        Bf16RaggedPrefillSpec::new(3, 21, 896, 8, 1, 128)?,
        &[0, 1, 5, 21],
        &[0, 128, 384, 896],
        0x2001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "gqa4_token_parallel",
        Bf16RaggedPrefillSpec::new(2, 96, 1280, 16, 4, 128)?,
        &[0, 32, 96],
        &[0, 256, 1280],
        0x4001,
    )?;
    run_short_indptr_case(&mut queue, &provider)?;
    run_invalid_metadata_guard(&mut queue, &provider)?;
    run_missing_workspace_case(&mut queue, &provider)?;
    run_tiled_graph_case(&mut queue, provider)?;
    println!(
        "gate=ragged_prefill_h20 suite=all status=pass output_max_abs_limit={:.9e} \
         lse_max_abs_limit={:.9e}",
        OUTPUT_MAX_ABS_LIMIT, LSE_MAX_ABS_LIMIT
    );
    Ok(())
}
