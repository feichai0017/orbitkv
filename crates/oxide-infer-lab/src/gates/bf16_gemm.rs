use crate::comparison::{Comparison, compare_bf16};
use crate::reporting::GateCase;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use oxide_infer::{
    Bf16DenseGemmSpec, DType, RmsNormSpec, bf16_dense_gemm_reference, rms_norm_bf16_reference,
};
use oxide_infer_cuda::command::{CommandError, CommandQueue};
use oxide_infer_cuda::gemm::{
    Bf16DenseGemmAlgorithm, Bf16DenseGemmEnqueueError, Bf16DenseGemmOperands, Bf16DenseGemmPlan,
    Bf16DenseGemmSelection, GemmPlanner, GemmProviderId, GemmProviderVersion,
};
use oxide_infer_cuda::graph::GraphQueue;
use oxide_infer_cuda::rms_norm::{RmsNormArgs, RmsNormProvider};
use std::error::Error;
use std::sync::Arc;

const LARGE_M: usize = 1;
const LARGE_N: usize = 4096;
const LARGE_K: usize = 4096;

fn require_bit_exact(case: &str, comparison: Comparison) -> Result<(), Box<dyn Error>> {
    if comparison.bit_mismatches == 0 {
        Ok(())
    } else {
        Err(format!(
            "{case} had {} BF16 bit mismatches; max_abs={:.9e}",
            comparison.bit_mismatches, comparison.max_abs
        )
        .into())
    }
}

fn large_fixture(spec: Bf16DenseGemmSpec) -> (Vec<bf16>, Vec<bf16>) {
    let activation = vec![bf16::ONE; spec.a_numel()];
    let mut weight = Vec::with_capacity(spec.weight_numel());
    for row in 0..spec.n() {
        let magnitude = ((row % 16) + 1) as f32 / 256.0;
        let sign = if (row / 16).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        weight.extend(std::iter::repeat_n(
            bf16::from_f32(sign * magnitude),
            spec.k(),
        ));
    }
    (activation, weight)
}

fn workspace_len(plan: &Bf16DenseGemmPlan) -> usize {
    plan.workspace_required_bytes()
}

fn check_standalone(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let spec = plan.spec();
    let (activation_host, weight_host) = large_fixture(spec);
    let mut expected = vec![bf16::ZERO; spec.output_numel()];
    bf16_dense_gemm_reference(&activation_host, &weight_host, &mut expected, spec)?;

    let activation = Arc::new(DeviceBuffer::from_host(&stream, &activation_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(&stream, &weight_host)?);
    let first_output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let second_output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(plan))?;
    let mut bindings = queue.bindings(5)?;
    let activation_handle = bindings.bind_read(Arc::clone(&activation))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let first_output_handle = bindings.bind_read_write(first_output)?;
    let second_output_handle = bindings.bind_read_write(second_output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    for output_handle in [first_output_handle.write(), second_output_handle.write()] {
        let mut scope = queue.begin(bindings)?;
        plan.enqueue_into(
            &mut scope,
            Bf16DenseGemmOperands::new(
                activation_handle,
                weight_handle,
                output_handle,
                workspace_handle.write(),
            ),
        )?;
        let completion = scope.finish();
        if completion.submitted() != 1 {
            return Err("standalone GEMM completion covered the wrong command count".into());
        }
        bindings = completion.wait()?;
    }
    let first_output = bindings.take_read_write(first_output_handle)?;
    let second_output = bindings.take_read_write(second_output_handle)?;
    drop(bindings);

    let first_actual = first_output.to_host_vec(&stream)?;
    let second_actual = second_output.to_host_vec(&stream)?;
    let first_comparison = compare_bf16(&first_actual, &expected, "BF16")?;
    let second_comparison = compare_bf16(&second_actual, &expected, "BF16")?;
    require_bit_exact("first standalone GEMM scope", first_comparison)?;
    require_bit_exact("second standalone GEMM scope", second_comparison)?;
    println!(
        "{} m={} n={} k={} scopes=2 \
         commands_per_scope=1 queue_reused=true bindings_reused=true plan_reused=true \
         first_bit_mismatches={} second_bit_mismatches={} max_abs={:.9e} \
         first_digest={:016x} second_digest={:016x}",
        GateCase::new("bf16_gemm_h20", "standalone"),
        spec.m(),
        spec.n(),
        spec.k(),
        first_comparison.bit_mismatches,
        second_comparison.bit_mismatches,
        first_comparison.max_abs.max(second_comparison.max_abs),
        first_comparison.digest,
        second_comparison.digest,
    );
    Ok(())
}

fn check_row_major_transpose(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let spec = plan.spec();
    let activation_host = [
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
        bf16::from_f32(-2.0),
        bf16::from_f32(0.5),
        bf16::from_f32(1.0),
        bf16::from_f32(-1.0),
    ];
    let weight_host = [
        bf16::from_f32(1.0),
        bf16::from_f32(0.0),
        bf16::from_f32(2.0),
        bf16::from_f32(-1.0),
        bf16::from_f32(-1.0),
        bf16::from_f32(3.0),
        bf16::from_f32(0.5),
        bf16::from_f32(2.0),
        bf16::from_f32(4.0),
        bf16::from_f32(-2.0),
        bf16::from_f32(1.0),
        bf16::from_f32(0.25),
    ];
    let mut expected = vec![bf16::ZERO; spec.output_numel()];
    bf16_dense_gemm_reference(&activation_host, &weight_host, &mut expected, spec)?;

    let activation = Arc::new(DeviceBuffer::from_host(&stream, &activation_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(&stream, &weight_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(plan))?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(Arc::clone(&activation))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16DenseGemmOperands::new(
            activation_handle,
            weight_handle,
            output_handle.write(),
            workspace_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("transpose-sensitive GEMM completion covered the wrong command count".into());
    }
    let mut bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;
    drop(bindings);

    let actual = output.to_host_vec(&stream)?;
    let comparison = compare_bf16(&actual, &expected, "BF16")?;
    require_bit_exact("transpose-sensitive GEMM", comparison)?;
    println!(
        "{} m={} n={} k={} \
         bit_mismatches={} max_abs={:.9e} digest={:016x}",
        GateCase::new("bf16_gemm_h20", "row_major_transpose"),
        spec.m(),
        spec.n(),
        spec.k(),
        comparison.bit_mismatches,
        comparison.max_abs,
        comparison.digest,
    );
    Ok(())
}

fn expect_length_rejection(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
    operand: &'static str,
    activation_len: usize,
    weight_len: usize,
    output_len: usize,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let activation = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, activation_len)?);
    let weight = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, weight_len)?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, output_len)?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(plan))?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(Arc::clone(&activation))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let mut scope = queue.begin(bindings)?;
    let result = plan.enqueue_into(
        &mut scope,
        Bf16DenseGemmOperands::new(
            activation_handle,
            weight_handle,
            output_handle.write(),
            workspace_handle.write(),
        ),
    );
    match result {
        Err(Bf16DenseGemmEnqueueError::LengthMismatch {
            operand: actual, ..
        }) if actual == operand => {}
        other => {
            return Err(
                format!("short {operand} buffer returned the wrong result: {other:?}").into(),
            );
        }
    }
    drop(scope);
    Ok(())
}

fn check_short_buffers(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    let spec = plan.spec();
    expect_length_rejection(
        queue,
        plan,
        "A",
        spec.a_numel() - 1,
        spec.weight_numel(),
        spec.output_numel(),
    )?;
    expect_length_rejection(
        queue,
        plan,
        "W",
        spec.a_numel(),
        spec.weight_numel() - 1,
        spec.output_numel(),
    )?;
    expect_length_rejection(
        queue,
        plan,
        "D",
        spec.a_numel(),
        spec.weight_numel(),
        spec.output_numel() - 1,
    )?;

    let workspace_required = plan.workspace_required_bytes();
    let workspace_gate = if workspace_required == 0 {
        "not_applicable_zero_required"
    } else {
        let stream = queue.stream().clone();
        let activation = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.a_numel())?);
        let weight = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.weight_numel())?);
        let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
        let workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_required.saturating_sub(1))?;
        let mut bindings = queue.bindings(4)?;
        let activation_handle = bindings.bind_read(Arc::clone(&activation))?;
        let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
        let output_handle = bindings.bind_read_write(output)?;
        let workspace_handle = bindings.bind_read_write(workspace)?;
        let mut scope = queue.begin(bindings)?;
        let result = plan.enqueue_into(
            &mut scope,
            Bf16DenseGemmOperands::new(
                activation_handle,
                weight_handle,
                output_handle.write(),
                workspace_handle.write(),
            ),
        );
        match result {
            Err(Bf16DenseGemmEnqueueError::WorkspaceTooSmall { required, actual })
                if required == workspace_required && actual == workspace_required - 1 => {}
            other => {
                return Err(format!("short workspace returned the wrong result: {other:?}").into());
            }
        }
        drop(scope);
        "rejected"
    };

    println!(
        "{} a=rejected w=rejected d=rejected \
         workspace={} workspace_required={}",
        GateCase::new("bf16_gemm_h20", "short_buffers"),
        workspace_gate,
        workspace_required,
    );
    Ok(())
}

fn check_command_capacity(
    stream: &Arc<CudaStream>,
    plan: &Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    let spec = plan.spec();
    let mut queue = CommandQueue::new(stream.clone(), 1, 1)?;
    let activation = Arc::new(DeviceBuffer::<bf16>::zeroed(stream, spec.a_numel())?);
    let weight = Arc::new(DeviceBuffer::<bf16>::zeroed(stream, spec.weight_numel())?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(stream, workspace_len(plan))?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(Arc::clone(&activation))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let operands = Bf16DenseGemmOperands::new(
        activation_handle,
        weight_handle,
        output_handle.write(),
        workspace_handle.write(),
    );
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(&mut scope, operands)?;
    match plan.enqueue_into(&mut scope, operands) {
        Err(Bf16DenseGemmEnqueueError::Command(CommandError::CommandCapacityExceeded {
            capacity: 1,
        })) => {}
        other => return Err(format!("second command returned the wrong result: {other:?}").into()),
    }
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("capacity rejection changed the submitted command count".into());
    }
    drop(completion.wait()?);

    println!(
        "{} capacity=1 first_submitted=true second_rejected_before_ffi=true submitted=1",
        GateCase::new("bf16_gemm_h20", "command_capacity"),
    );
    Ok(())
}

fn check_rms_norm_gemm_chain(
    queue: &mut CommandQueue,
    rms_provider: &RmsNormProvider,
    gemm_plan: &Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let gemm_spec = gemm_plan.spec();
    let rms_spec = RmsNormSpec::new(1, LARGE_K, 1.0e-5, DType::Bf16)?;
    let rms_plan = rms_provider.plan_bf16(rms_spec)?;
    let input_host = vec![bf16::ONE; rms_spec.numel()];
    let norm_weight_host = vec![bf16::ONE; rms_spec.hidden_size()];
    let (_, gemm_weight_host) = large_fixture(gemm_spec);
    let mut intermediate_expected = vec![bf16::ZERO; rms_spec.numel()];
    let mut expected = vec![bf16::ZERO; gemm_spec.output_numel()];
    rms_norm_bf16_reference(
        &input_host,
        &norm_weight_host,
        &mut intermediate_expected,
        rms_spec,
    )?;
    bf16_dense_gemm_reference(
        &intermediate_expected,
        &gemm_weight_host,
        &mut expected,
        gemm_spec,
    )?;

    let input = Arc::new(DeviceBuffer::from_host(&stream, &input_host)?);
    let norm_weight = Arc::new(DeviceBuffer::from_host(&stream, &norm_weight_host)?);
    let intermediate = DeviceBuffer::<bf16>::zeroed(&stream, rms_spec.numel())?;
    let gemm_weight = Arc::new(DeviceBuffer::from_host(&stream, &gemm_weight_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, gemm_spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(gemm_plan))?;
    let mut bindings = queue.bindings(6)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let norm_weight_handle = bindings.bind_read(Arc::clone(&norm_weight))?;
    let intermediate_handle = bindings.bind_read_write(intermediate)?;
    let gemm_weight_handle = bindings.bind_read(Arc::clone(&gemm_weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let mut scope = queue.begin(bindings)?;
    rms_plan.enqueue_into(
        &mut scope,
        RmsNormArgs::new(
            input_handle,
            norm_weight_handle,
            intermediate_handle.write(),
        ),
    )?;
    gemm_plan.enqueue_into(
        &mut scope,
        Bf16DenseGemmOperands::new(
            intermediate_handle.read(),
            gemm_weight_handle,
            output_handle.write(),
            workspace_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 2 {
        return Err("RMSNorm to GEMM completion covered the wrong command count".into());
    }
    let mut bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;
    drop(bindings);

    let actual = output.to_host_vec(&stream)?;
    let comparison = compare_bf16(&actual, &expected, "BF16")?;
    require_bit_exact("RMSNorm to GEMM chain", comparison)?;
    println!(
        "{} rows=1 hidden=4096 \
         m={} n={} k={} commands=2 completion_records=1 intermediate_waits=0 \
         gemm_plan_reused=true bit_mismatches={} max_abs={:.9e} digest={:016x}",
        GateCase::new("bf16_gemm_h20", "rms_norm_gemm_chain"),
        gemm_spec.m(),
        gemm_spec.n(),
        gemm_spec.k(),
        comparison.bit_mismatches,
        comparison.max_abs,
        comparison.digest,
    );
    Ok(())
}

fn check_rms_norm_gemm_graph(
    queue: &mut CommandQueue,
    rms_provider: RmsNormProvider,
    gemm_plan: Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let gemm_spec = gemm_plan.spec();
    let rms_spec = RmsNormSpec::new(1, LARGE_K, 1.0e-5, DType::Bf16)?;
    let rms_plan = rms_provider.plan_bf16(rms_spec)?;
    drop(rms_provider);
    let input_host = vec![bf16::ONE; rms_spec.numel()];
    let norm_weight_host = vec![bf16::ONE; rms_spec.hidden_size()];
    let (_, gemm_weight_host) = large_fixture(gemm_spec);
    let mut intermediate_expected = vec![bf16::ZERO; rms_spec.numel()];
    let mut expected = vec![bf16::ZERO; gemm_spec.output_numel()];
    rms_norm_bf16_reference(
        &input_host,
        &norm_weight_host,
        &mut intermediate_expected,
        rms_spec,
    )?;
    bf16_dense_gemm_reference(
        &intermediate_expected,
        &gemm_weight_host,
        &mut expected,
        gemm_spec,
    )?;

    let input = Arc::new(DeviceBuffer::from_host(&stream, &input_host)?);
    let norm_weight = Arc::new(DeviceBuffer::from_host(&stream, &norm_weight_host)?);
    let intermediate = DeviceBuffer::<bf16>::zeroed(&stream, rms_spec.numel())?;
    let gemm_weight = Arc::new(DeviceBuffer::from_host(&stream, &gemm_weight_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, gemm_spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(&gemm_plan))?;
    let graph_queue = GraphQueue::new(stream.context(), 2)?;
    let mut bindings = graph_queue.bindings(6)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let norm_weight_handle = bindings.bind_read(Arc::clone(&norm_weight))?;
    let intermediate_handle = bindings.bind_read_write(intermediate)?;
    let gemm_weight_handle = bindings.bind_read(Arc::clone(&gemm_weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let captured = graph_queue.capture(bindings, |scope| -> Result<(), Box<dyn Error>> {
        rms_plan.enqueue_into(
            scope,
            RmsNormArgs::new(
                input_handle,
                norm_weight_handle,
                intermediate_handle.write(),
            ),
        )?;
        gemm_plan.enqueue_into(
            scope,
            Bf16DenseGemmOperands::new(
                intermediate_handle.read(),
                gemm_weight_handle,
                output_handle.write(),
                workspace_handle.write(),
            ),
        )?;
        Ok(())
    })?;
    if captured.commands() != 2 {
        return Err("captured graph covered the wrong command count".into());
    }
    drop(rms_plan);
    drop(gemm_plan);
    drop(input);
    drop(norm_weight);
    drop(gemm_weight);

    let mut exec = captured.instantiate()?;
    for expected_launch in 1..=2 {
        let mut completion = exec.launch()?;
        if completion.launch_index() != expected_launch {
            return Err("graph completion reported the wrong replay index".into());
        }
        let _ = completion.is_complete()?;
        if expected_launch == 1 {
            completion.wait()?;
        } else {
            drop(completion);
        }
    }
    if exec.launches() != 2 || exec.commands() != 2 {
        return Err("graph exec accounting changed across replay".into());
    }
    let mut bindings = exec.into_bindings()?;
    let output = bindings.take_read_write(output_handle)?;
    drop(bindings);

    let actual = output.to_host_vec(&stream)?;
    let comparison = compare_bf16(&actual, &expected, "BF16")?;
    require_bit_exact("RMSNorm to GEMM graph", comparison)?;
    println!(
        "{} rows=1 hidden=4096 \
         m={} n={} k={} commands=2 replays=2 fixed_bindings=true cross_stream=false \
         external_owners_dropped_before_replay=true \
         completion_queries=2 completion_waits=1 completion_drops=1 \
         intra_graph_host_waits=0 inter_replay_host_waits=1 \
         bit_mismatches={} max_abs={:.9e} digest={:016x}",
        GateCase::new("bf16_gemm_h20", "rms_norm_gemm_graph"),
        gemm_spec.m(),
        gemm_spec.n(),
        gemm_spec.k(),
        comparison.bit_mismatches,
        comparison.max_abs,
        comparison.digest,
    );
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let rms_provider = RmsNormProvider::load(&context)?;
    let gemm_planner = GemmPlanner::load(&context)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let mut queue = CommandQueue::new(stream, 2, 1)?;

    let large_spec = Bf16DenseGemmSpec::new(LARGE_M, LARGE_N, LARGE_K)?;
    let large_plan = gemm_planner.plan_bf16_dense(large_spec, Bf16DenseGemmSelection::CublasLt)?;
    let small_plan = gemm_planner.plan_bf16_dense(
        Bf16DenseGemmSpec::new(2, 3, 4)?,
        Bf16DenseGemmSelection::CublasLt,
    )?;
    for (name, plan) in [("large", &large_plan), ("small", &small_plan)] {
        let info = plan.plan_info();
        if info.provider() != GemmProviderId::CublasLt
            || info.algorithm() != Bf16DenseGemmAlgorithm::CublasLtHeuristic
            || info.workspace_required_bytes() != plan.workspace_required_bytes()
            || info.tensor_alignment_bytes() != plan.tensor_alignment_bytes()
            || info.workspace_alignment_bytes() != plan.workspace_alignment_bytes()
        {
            return Err(format!("{name} dense GEMM plan reported inconsistent plan info").into());
        }
    }
    let GemmProviderVersion::CublasLt(library_version) =
        gemm_planner.provider_version(GemmProviderId::CublasLt)
    else {
        return Err("cuBLASLt GEMM provider reported the wrong version identity".into());
    };
    let workspace_limit = gemm_planner.workspace_limit_bytes(Bf16DenseGemmSelection::CublasLt);
    let large_waves = large_plan
        .estimated_waves_count()
        .ok_or("cuBLASLt large plan did not report its estimated waves count")?;
    let small_waves = small_plan
        .estimated_waves_count()
        .ok_or("cuBLASLt small plan did not report its estimated waves count")?;
    println!(
        "{} library_version={} \
         workspace_limit={} large_workspace_required={} large_waves={:.6} \
         small_workspace_required={} small_waves={:.6}",
        GateCase::new("bf16_gemm_h20", "plan"),
        library_version,
        workspace_limit,
        large_plan.workspace_required_bytes(),
        large_waves,
        small_plan.workspace_required_bytes(),
        small_waves,
    );

    check_standalone(&mut queue, &large_plan)?;
    check_row_major_transpose(&mut queue, &small_plan)?;
    check_short_buffers(&mut queue, &small_plan)?;
    check_command_capacity(queue.stream(), &small_plan)?;
    check_rms_norm_gemm_chain(&mut queue, &rms_provider, &large_plan)?;
    drop(small_plan);
    drop(gemm_planner);
    check_rms_norm_gemm_graph(&mut queue, rms_provider, large_plan)?;
    println!("gate=bf16_gemm_h20 suite=all status=pass");
    Ok(())
}
