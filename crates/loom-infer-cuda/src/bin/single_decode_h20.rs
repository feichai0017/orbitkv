use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchContractError};
use half::bf16;
use loom_infer::{Bf16SingleDecodeSpec, SINGLE_DECODE_HEAD_DIM, single_decode_bf16_reference};
use loom_infer_cuda::attention::{
    AttentionProvider, Bf16SingleDecodeArgs, Bf16SingleDecodePlan, SingleDecodeEnqueueError,
};
use loom_infer_cuda::command::{CommandError, CommandQueue};
use std::error::Error;
use std::sync::Arc;

const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const LSE_MAX_ABS_LIMIT: f32 = 0.01;

#[derive(Clone, Copy, Debug)]
struct Comparison {
    max_abs: f32,
    mismatches: usize,
    digest: u64,
}

#[derive(Clone, Copy, Debug)]
enum ShortBuffer {
    Query,
    Key,
    Value,
    Output,
    Lse,
}

fn deterministic_bf16(len: usize, salt: u64) -> Vec<bf16> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let signed = (state % 2001) as i32 - 1000;
        values.push(bf16::from_f32(signed as f32 / 2048.0));
    }
    values
}

fn compare_bf16(actual: &[bf16], expected: &[bf16]) -> Result<Comparison, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "BF16 comparison length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut max_abs = 0.0_f32;
    let mut mismatches = 0_usize;
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual_f32 = actual.to_f32();
        if !actual_f32.is_finite() {
            return Err(format!("non-finite BF16 output at index {index}").into());
        }
        max_abs = max_abs.max((actual_f32 - expected.to_f32()).abs());
        mismatches += usize::from(actual.to_bits() != expected.to_bits());
        digest ^= u64::from(actual.to_bits());
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(Comparison {
        max_abs,
        mismatches,
        digest,
    })
}

fn compare_f32(actual: &[f32], expected: &[f32]) -> Result<Comparison, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "F32 comparison length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut max_abs = 0.0_f32;
    let mut mismatches = 0_usize;
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() {
            return Err(format!("non-finite F32 LSE at index {index}").into());
        }
        max_abs = max_abs.max((actual - expected).abs());
        mismatches += usize::from(actual.to_bits() != expected.to_bits());
        digest ^= u64::from(actual.to_bits());
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(Comparison {
        max_abs,
        mismatches,
        digest,
    })
}

fn run_case(
    queue: &mut CommandQueue,
    provider: &AttentionProvider,
    name: &str,
    spec: Bf16SingleDecodeSpec,
    salt: u64,
) -> Result<(), Box<dyn Error>> {
    let query_host = deterministic_bf16(spec.query_numel(), salt);
    let key_host = deterministic_bf16(spec.kv_numel(), salt ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_numel(), salt ^ 0x5641_4c55_4500);
    run_case_with_inputs(
        queue,
        provider,
        name,
        spec,
        &query_host,
        &key_host,
        &value_host,
    )
}

fn run_case_with_inputs(
    queue: &mut CommandQueue,
    provider: &AttentionProvider,
    name: &str,
    spec: Bf16SingleDecodeSpec,
    query_host: &[bf16],
    key_host: &[bf16],
    value_host: &[bf16],
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16(spec)?;
    let mut expected_output = vec![bf16::ZERO; spec.output_numel()];
    let mut expected_lse = vec![0.0_f32; spec.lse_numel()];
    single_decode_bf16_reference(
        query_host,
        key_host,
        value_host,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;

    let query = Arc::new(DeviceBuffer::from_host(&stream, query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&stream, key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(&stream, value_host)?);
    let output_sentinel = vec![bf16::NAN; spec.output_numel()];
    let lse_sentinel = vec![f32::NAN; spec.lse_numel()];
    let output = DeviceBuffer::from_host(&stream, &output_sentinel)?;
    let lse = DeviceBuffer::from_host(&stream, &lse_sentinel)?;
    let mut bindings = queue.bindings(5)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16SingleDecodeArgs::new(
            query_handle,
            key_handle,
            value_handle,
            output_handle.write(),
            lse_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("single decode completion covered the wrong command count".into());
    }
    bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);

    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    let output_comparison = compare_bf16(&actual_output, &expected_output)?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse)?;
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
        "gate=single_decode_h20 case={} status=pass kv_len={} query_heads={} kv_heads={} \
         group_size={} head_dim={} layout=NHD dtype=BF16 accumulation=F32 lse_domain=log2 \
         output_max_abs={:.9e} output_bit_mismatches={} output_digest={:016x} \
         lse_max_abs={:.9e} lse_bit_mismatches={} lse_digest={:016x}",
        name,
        spec.kv_len(),
        spec.num_query_heads(),
        spec.num_kv_heads(),
        spec.gqa_group_size(),
        spec.head_dim(),
        output_comparison.max_abs,
        output_comparison.mismatches,
        output_comparison.digest,
        lse_comparison.max_abs,
        lse_comparison.mismatches,
        lse_comparison.digest,
    );
    Ok(())
}

fn run_large_logit_case(
    queue: &mut CommandQueue,
    provider: &AttentionProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16SingleDecodeSpec::new(3, 1, 1, SINGLE_DECODE_HEAD_DIM)?;
    let query = vec![bf16::from_f32(64.0); spec.query_numel()];
    let mut key = Vec::with_capacity(spec.kv_numel());
    let mut value = Vec::with_capacity(spec.kv_numel());
    for (key_value, value_value) in [(-64.0, -1.0), (0.0, 0.5), (64.0, 2.0)] {
        key.extend(std::iter::repeat_n(
            bf16::from_f32(key_value),
            SINGLE_DECODE_HEAD_DIM,
        ));
        value.extend(std::iter::repeat_n(
            bf16::from_f32(value_value),
            SINGLE_DECODE_HEAD_DIM,
        ));
    }
    run_case_with_inputs(
        queue,
        provider,
        "large_logits_l3",
        spec,
        &query,
        &key,
        &value,
    )
}

fn run_short_buffer_case(
    queue: &mut CommandQueue,
    plan: &Bf16SingleDecodePlan,
    short: ShortBuffer,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let spec = plan.spec();
    let query_len = spec.query_numel() - usize::from(matches!(short, ShortBuffer::Query));
    let key_len = spec.kv_numel() - usize::from(matches!(short, ShortBuffer::Key));
    let value_len = spec.kv_numel() - usize::from(matches!(short, ShortBuffer::Value));
    let output_len = spec.output_numel() - usize::from(matches!(short, ShortBuffer::Output));
    let lse_len = spec.lse_numel() - usize::from(matches!(short, ShortBuffer::Lse));
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, query_len)?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, key_len)?);
    let value = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, value_len)?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, output_len)?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, lse_len)?;
    let mut bindings = queue.bindings(5)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16SingleDecodeArgs::new(
                query_handle,
                key_handle,
                value_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
        .expect_err("short single-decode buffer must be rejected");
    let expected_relation = match short {
        ShortBuffer::Query => "query.len() == num_query_heads * 128",
        ShortBuffer::Key => "key.len() == kv_len * num_kv_heads * 128",
        ShortBuffer::Value => "value.len() == kv_len * num_kv_heads * 128",
        ShortBuffer::Output => "output.len() == num_query_heads * 128",
        ShortBuffer::Lse => "lse.len() == num_query_heads",
    };
    if !matches!(
        &error,
        SingleDecodeEnqueueError::Launch(LaunchContractError::SizeRequirementViolated {
            relation,
            ..
        }) if *relation == expected_relation
    ) {
        return Err(
            format!("short {short:?} buffer did not violate {expected_relation}: {error}").into(),
        );
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("short buffer reached CUDA submission".into());
    }
    drop(completion.wait()?);
    Ok(())
}

fn check_short_buffers(
    queue: &mut CommandQueue,
    provider: &AttentionProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16SingleDecodeSpec::new(2, 4, 2, SINGLE_DECODE_HEAD_DIM)?;
    let plan = provider.plan_bf16(spec)?;
    for short in [
        ShortBuffer::Query,
        ShortBuffer::Key,
        ShortBuffer::Value,
        ShortBuffer::Output,
        ShortBuffer::Lse,
    ] {
        run_short_buffer_case(queue, &plan, short)?;
    }
    println!(
        "gate=single_decode_h20 case=short_buffers status=pass \
         query=rejected key=rejected value=rejected output=rejected lse=rejected before_ffi=true"
    );
    Ok(())
}

fn check_duplicate_binding(
    queue: &mut CommandQueue,
    provider: &AttentionProvider,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let spec = Bf16SingleDecodeSpec::new(1, 1, 1, SINGLE_DECODE_HEAD_DIM)?;
    let plan = provider.plan_bf16(spec)?;
    let query_and_key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let value = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_numel())?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(4)?;
    let query_and_key_handle = bindings.bind_read(query_and_key)?;
    let value_handle = bindings.bind_read(value)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16SingleDecodeArgs::new(
                query_and_key_handle,
                query_and_key_handle,
                value_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
        .expect_err("one binding slot cannot serve two single-decode operands");
    if !matches!(
        error,
        SingleDecodeEnqueueError::Command(CommandError::DuplicateBindingSlot)
    ) {
        return Err(format!("duplicate binding returned the wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("duplicate binding reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!("gate=single_decode_h20 case=duplicate_binding status=pass rejected_before_ffi=true");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let provider = AttentionProvider::load(&context)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let mut queue = CommandQueue::new(stream, 1)?;

    run_case(
        &mut queue,
        &provider,
        "mha_l1",
        Bf16SingleDecodeSpec::new(1, 8, 8, SINGLE_DECODE_HEAD_DIM)?,
        0x1001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "mqa_l33",
        Bf16SingleDecodeSpec::new(33, 8, 1, SINGLE_DECODE_HEAD_DIM)?,
        0x2001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "gqa4_l127",
        Bf16SingleDecodeSpec::new(127, 16, 4, SINGLE_DECODE_HEAD_DIM)?,
        0x4001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "gqa8_l4096",
        Bf16SingleDecodeSpec::new(4096, 32, 4, SINGLE_DECODE_HEAD_DIM)?,
        0x8001,
    )?;
    run_large_logit_case(&mut queue, &provider)?;
    check_short_buffers(&mut queue, &provider)?;
    check_duplicate_binding(&mut queue, &provider)?;
    println!(
        "gate=single_decode_h20 suite=all status=pass output_max_abs_limit={:.9e} \
         lse_max_abs_limit={:.9e}",
        OUTPUT_MAX_ABS_LIMIT, LSE_MAX_ABS_LIMIT
    );
    Ok(())
}
