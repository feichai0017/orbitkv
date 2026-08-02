use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use loom_infer::{Bf16GemmSpec, DType, RmsNormSpec, bf16_gemm_reference, rms_norm_bf16_reference};
use loom_infer_cuda::command::{CommandError, CommandQueue};
use loom_infer_cuda::gemm::{Bf16GemmArgs, Bf16GemmEnqueueError, Bf16GemmPlan, CublasLtProvider};
use loom_infer_cuda::rms_norm::{RmsNormArgs, RmsNormProvider};
use std::error::Error;
use std::sync::Arc;

const LARGE_M: usize = 1;
const LARGE_N: usize = 4096;
const LARGE_K: usize = 4096;

#[derive(Clone, Copy, Debug)]
struct Comparison {
    max_abs: f32,
    bit_mismatches: usize,
    digest: u64,
}

fn compare(actual: &[bf16], expected: &[bf16]) -> Result<Comparison, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "comparison length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        )
        .into());
    }

    let mut max_abs = 0.0_f32;
    let mut bit_mismatches = 0_usize;
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.to_f32().is_finite() {
            return Err(format!("non-finite output at index {index}").into());
        }
        max_abs = max_abs.max((actual.to_f32() - expected.to_f32()).abs());
        bit_mismatches += usize::from(actual.to_bits() != expected.to_bits());
        digest ^= u64::from(actual.to_bits());
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }

    Ok(Comparison {
        max_abs,
        bit_mismatches,
        digest,
    })
}

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

fn large_fixture(spec: Bf16GemmSpec) -> (Vec<bf16>, Vec<bf16>) {
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

fn workspace_len(plan: &Bf16GemmPlan) -> usize {
    plan.workspace_required_bytes()
}

fn check_standalone(queue: &mut CommandQueue, plan: &Bf16GemmPlan) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let spec = plan.spec();
    let (activation_host, weight_host) = large_fixture(spec);
    let mut expected = vec![bf16::ZERO; spec.output_numel()];
    bf16_gemm_reference(&activation_host, &weight_host, &mut expected, spec)?;

    let activation = DeviceBuffer::from_host(&stream, &activation_host)?;
    let weight = DeviceBuffer::from_host(&stream, &weight_host)?;
    let mut first_output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let mut second_output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let mut workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(plan))?;
    let mut bindings = queue.bindings(5)?;
    let activation_handle = bindings.bind_read(&activation)?;
    let weight_handle = bindings.bind_read(&weight)?;
    let first_output_handle = bindings.bind_read_write(&mut first_output)?;
    let second_output_handle = bindings.bind_read_write(&mut second_output)?;
    let workspace_handle = bindings.bind_read_write(&mut workspace)?;

    for output_handle in [first_output_handle.write(), second_output_handle.write()] {
        let mut scope = queue.begin(bindings)?;
        plan.enqueue_into(
            &mut scope,
            Bf16GemmArgs::new(
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
    drop(bindings);

    let first_actual = first_output.to_host_vec(&stream)?;
    let second_actual = second_output.to_host_vec(&stream)?;
    let first_comparison = compare(&first_actual, &expected)?;
    let second_comparison = compare(&second_actual, &expected)?;
    require_bit_exact("first standalone GEMM scope", first_comparison)?;
    require_bit_exact("second standalone GEMM scope", second_comparison)?;
    println!(
        "gate=bf16_gemm_h20 case=standalone status=pass m={} n={} k={} scopes=2 \
         commands_per_scope=1 queue_reused=true bindings_reused=true plan_reused=true \
         first_bit_mismatches={} second_bit_mismatches={} max_abs={:.9e} \
         first_digest={:016x} second_digest={:016x}",
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
    plan: &Bf16GemmPlan,
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
    bf16_gemm_reference(&activation_host, &weight_host, &mut expected, spec)?;

    let activation = DeviceBuffer::from_host(&stream, &activation_host)?;
    let weight = DeviceBuffer::from_host(&stream, &weight_host)?;
    let mut output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let mut workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(plan))?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(&activation)?;
    let weight_handle = bindings.bind_read(&weight)?;
    let output_handle = bindings.bind_read_write(&mut output)?;
    let workspace_handle = bindings.bind_read_write(&mut workspace)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16GemmArgs::new(
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
    drop(completion.wait()?);

    let actual = output.to_host_vec(&stream)?;
    let comparison = compare(&actual, &expected)?;
    require_bit_exact("transpose-sensitive GEMM", comparison)?;
    println!(
        "gate=bf16_gemm_h20 case=row_major_transpose status=pass m={} n={} k={} \
         bit_mismatches={} max_abs={:.9e} digest={:016x}",
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
    plan: &Bf16GemmPlan,
    operand: &'static str,
    activation_len: usize,
    weight_len: usize,
    output_len: usize,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let activation = DeviceBuffer::<bf16>::zeroed(&stream, activation_len)?;
    let weight = DeviceBuffer::<bf16>::zeroed(&stream, weight_len)?;
    let mut output = DeviceBuffer::<bf16>::zeroed(&stream, output_len)?;
    let mut workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(plan))?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(&activation)?;
    let weight_handle = bindings.bind_read(&weight)?;
    let output_handle = bindings.bind_read_write(&mut output)?;
    let workspace_handle = bindings.bind_read_write(&mut workspace)?;
    let mut scope = queue.begin(bindings)?;
    let result = plan.enqueue_into(
        &mut scope,
        Bf16GemmArgs::new(
            activation_handle,
            weight_handle,
            output_handle.write(),
            workspace_handle.write(),
        ),
    );
    match result {
        Err(Bf16GemmEnqueueError::LengthMismatch {
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
    plan: &Bf16GemmPlan,
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
        let activation = DeviceBuffer::<bf16>::zeroed(&stream, spec.a_numel())?;
        let weight = DeviceBuffer::<bf16>::zeroed(&stream, spec.weight_numel())?;
        let mut output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
        let mut workspace =
            DeviceBuffer::<u8>::zeroed(&stream, workspace_required.saturating_sub(1))?;
        let mut bindings = queue.bindings(4)?;
        let activation_handle = bindings.bind_read(&activation)?;
        let weight_handle = bindings.bind_read(&weight)?;
        let output_handle = bindings.bind_read_write(&mut output)?;
        let workspace_handle = bindings.bind_read_write(&mut workspace)?;
        let mut scope = queue.begin(bindings)?;
        let result = plan.enqueue_into(
            &mut scope,
            Bf16GemmArgs::new(
                activation_handle,
                weight_handle,
                output_handle.write(),
                workspace_handle.write(),
            ),
        );
        match result {
            Err(Bf16GemmEnqueueError::WorkspaceTooSmall { required, actual })
                if required == workspace_required && actual == workspace_required - 1 => {}
            other => {
                return Err(format!("short workspace returned the wrong result: {other:?}").into());
            }
        }
        drop(scope);
        "rejected"
    };

    println!(
        "gate=bf16_gemm_h20 case=short_buffers status=pass a=rejected w=rejected d=rejected \
         workspace={} workspace_required={}",
        workspace_gate, workspace_required,
    );
    Ok(())
}

fn check_command_capacity(
    stream: &Arc<CudaStream>,
    plan: &Bf16GemmPlan,
) -> Result<(), Box<dyn Error>> {
    let spec = plan.spec();
    let mut queue = CommandQueue::new(stream.clone(), 1)?;
    let activation = DeviceBuffer::<bf16>::zeroed(stream, spec.a_numel())?;
    let weight = DeviceBuffer::<bf16>::zeroed(stream, spec.weight_numel())?;
    let mut output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let mut workspace = DeviceBuffer::<u8>::zeroed(stream, workspace_len(plan))?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(&activation)?;
    let weight_handle = bindings.bind_read(&weight)?;
    let output_handle = bindings.bind_read_write(&mut output)?;
    let workspace_handle = bindings.bind_read_write(&mut workspace)?;
    let args = Bf16GemmArgs::new(
        activation_handle,
        weight_handle,
        output_handle.write(),
        workspace_handle.write(),
    );
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(&mut scope, args)?;
    match plan.enqueue_into(&mut scope, args) {
        Err(Bf16GemmEnqueueError::Command(CommandError::CommandCapacityExceeded {
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
        "gate=bf16_gemm_h20 case=command_capacity status=pass capacity=1 \
         first_submitted=true second_rejected_before_ffi=true submitted=1"
    );
    Ok(())
}

fn check_rms_norm_gemm_chain(
    queue: &mut CommandQueue,
    rms_provider: &RmsNormProvider,
    gemm_plan: &Bf16GemmPlan,
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
    bf16_gemm_reference(
        &intermediate_expected,
        &gemm_weight_host,
        &mut expected,
        gemm_spec,
    )?;

    let input = DeviceBuffer::from_host(&stream, &input_host)?;
    let norm_weight = DeviceBuffer::from_host(&stream, &norm_weight_host)?;
    let mut intermediate = DeviceBuffer::<bf16>::zeroed(&stream, rms_spec.numel())?;
    let gemm_weight = DeviceBuffer::from_host(&stream, &gemm_weight_host)?;
    let mut output = DeviceBuffer::<bf16>::zeroed(&stream, gemm_spec.output_numel())?;
    let mut workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_len(gemm_plan))?;
    let mut bindings = queue.bindings(6)?;
    let input_handle = bindings.bind_read(&input)?;
    let norm_weight_handle = bindings.bind_read(&norm_weight)?;
    let intermediate_handle = bindings.bind_read_write(&mut intermediate)?;
    let gemm_weight_handle = bindings.bind_read(&gemm_weight)?;
    let output_handle = bindings.bind_read_write(&mut output)?;
    let workspace_handle = bindings.bind_read_write(&mut workspace)?;
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
        Bf16GemmArgs::new(
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
    drop(completion.wait()?);

    let actual = output.to_host_vec(&stream)?;
    let comparison = compare(&actual, &expected)?;
    require_bit_exact("RMSNorm to GEMM chain", comparison)?;
    println!(
        "gate=bf16_gemm_h20 case=rms_norm_gemm_chain status=pass rows=1 hidden=4096 \
         m={} n={} k={} commands=2 completion_records=1 intermediate_waits=0 \
         gemm_plan_reused=true bit_mismatches={} max_abs={:.9e} digest={:016x}",
        gemm_spec.m(),
        gemm_spec.n(),
        gemm_spec.k(),
        comparison.bit_mismatches,
        comparison.max_abs,
        comparison.digest,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let rms_provider = RmsNormProvider::load(&context)?;
    let gemm_provider = CublasLtProvider::load(&context)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let mut queue = CommandQueue::new(stream, 2)?;

    let large_spec = Bf16GemmSpec::new(LARGE_M, LARGE_N, LARGE_K)?;
    let large_plan = gemm_provider.plan_bf16(large_spec)?;
    let small_plan = gemm_provider.plan_bf16(Bf16GemmSpec::new(2, 3, 4)?)?;
    println!(
        "gate=bf16_gemm_h20 case=plan status=pass library_version={} \
         workspace_limit={} large_workspace_required={} large_waves={:.6} \
         small_workspace_required={} small_waves={:.6}",
        gemm_provider.library_version(),
        gemm_provider.workspace_limit_bytes(),
        large_plan.workspace_required_bytes(),
        large_plan.heuristic_waves_count(),
        small_plan.workspace_required_bytes(),
        small_plan.heuristic_waves_count(),
    );

    check_standalone(&mut queue, &large_plan)?;
    check_row_major_transpose(&mut queue, &small_plan)?;
    check_short_buffers(&mut queue, &small_plan)?;
    check_command_capacity(queue.stream(), &small_plan)?;
    check_rms_norm_gemm_chain(&mut queue, &rms_provider, &large_plan)?;
    println!("gate=bf16_gemm_h20 suite=all status=pass");
    Ok(())
}
