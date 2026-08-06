use crate::comparison::compare_bf16;
use crate::fixture::deterministic_bf16;
use crate::reporting::GateCase;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use loom_infer::{
    Bf16RopePagedKvAppendSpec, Bf16RopePagedKvAppendTokensSpec, Bf16RopePosIdsSpec,
    rope_paged_kv_append_bf16_reference, rope_paged_kv_append_tokens_bf16_reference,
    rope_pos_ids_bf16_reference,
};
use loom_infer_cuda::command::CommandQueue;
use loom_infer_cuda::graph::GraphQueue;
use loom_infer_cuda::rope::{
    Bf16RopePagedKvAppendArgs, Bf16RopePagedKvAppendTokensArgs, Bf16RopePosIdsArgs,
    RopeEnqueueError, RopePlanError, RopeProvider,
};
use std::error::Error;
use std::sync::Arc;

const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;

fn run_reference_case(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePosIdsSpec::new(5, 16, 4, 128, 128, 1.0, 10_000.0)?;
    let position_ids_host = [0_i32, 1, 127, 4096, 32_767];
    let query_host = deterministic_bf16(spec.query_numel(), 0x524f_5045);
    let key_host = deterministic_bf16(spec.key_numel(), 0x4b45_5900);
    let mut expected_query = vec![bf16::NAN; spec.query_numel()];
    let mut expected_key = vec![bf16::NAN; spec.key_numel()];
    rope_pos_ids_bf16_reference(
        &query_host,
        &key_host,
        &position_ids_host,
        &mut expected_query,
        &mut expected_key,
        spec,
    )?;

    let plan = provider.plan_bf16_pos_ids(spec)?;
    let stream = queue.stream().clone();
    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let position_ids = Arc::new(DeviceBuffer::from_host(&stream, &position_ids_host)?);
    let query_output = DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.query_numel()])?;
    let key_output = DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.key_numel()])?;
    let mut bindings = queue.bindings(5)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let position_ids_handle = bindings.bind_read(position_ids)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_output_handle = bindings.bind_read_write(key_output)?;

    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16RopePosIdsArgs::new(
            query_handle,
            key_handle,
            position_ids_handle,
            query_output_handle.write(),
            key_output_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("RoPE completion covered the wrong command count".into());
    }
    bindings = completion.wait()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_output = bindings.take_read_write(key_output_handle)?;
    drop(bindings);

    let query_comparison = compare_bf16(
        &query_output.to_host_vec(&stream)?,
        &expected_query,
        "RoPE query BF16",
    )?;
    let key_comparison = compare_bf16(
        &key_output.to_host_vec(&stream)?,
        &expected_key,
        "RoPE key BF16",
    )?;
    if query_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT {
        return Err(format!(
            "RoPE query max abs {:.9e} exceeds {:.9e}",
            query_comparison.max_abs, OUTPUT_MAX_ABS_LIMIT
        )
        .into());
    }
    if key_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT {
        return Err(format!(
            "RoPE key max abs {:.9e} exceeds {:.9e}",
            key_comparison.max_abs, OUTPUT_MAX_ABS_LIMIT
        )
        .into());
    }

    println!(
        "{} tokens=5 query_heads=16 key_heads=4 head_dim=128 rotary_dim=128 \
         layout=NHD style=neox_split_half dtype=BF16 accumulation=F32 \
         rope_scale=1 rope_theta=10000 position_ids=0_1_127_4096_32767 \
         commands=1 stream=non_default query_max_abs={:.9e} \
         query_bit_mismatches={} query_digest={:016x} key_max_abs={:.9e} \
         key_bit_mismatches={} key_digest={:016x}",
        GateCase::new("rope_h20", "bf16_pos_ids"),
        query_comparison.max_abs,
        query_comparison.bit_mismatches,
        query_comparison.digest,
        key_comparison.max_abs,
        key_comparison.bit_mismatches,
        key_comparison.digest,
    );
    Ok(())
}

fn run_short_buffer_case(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePosIdsSpec::new(1, 1, 1, 128, 128, 1.0, 10_000.0)?;
    let plan = provider.plan_bf16_pos_ids(spec)?;
    let stream = queue.stream().clone();
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.query_numel() - 1,
    )?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.key_numel())?);
    let position_ids = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32])?);
    let query_output = DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?;
    let key_output = DeviceBuffer::<bf16>::zeroed(&stream, spec.key_numel())?;
    let mut bindings = queue.bindings(5)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let position_ids_handle = bindings.bind_read(position_ids)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_output_handle = bindings.bind_read_write(key_output)?;
    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16RopePosIdsArgs::new(
                query_handle,
                key_handle,
                position_ids_handle,
                query_output_handle.write(),
                key_output_handle.write(),
            ),
        )
        .expect_err("short query must fail before submission");
    if !matches!(
        error,
        RopeEnqueueError::LengthMismatch {
            operand: "query",
            expected: 128,
            actual: 127,
        }
    ) {
        return Err(format!("short RoPE query returned the wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("short RoPE query reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!(
        "{} query=rejected before_ffi=true",
        GateCase::new("rope_h20", "short_buffer")
    );
    Ok(())
}

fn run_negative_position_guard(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePosIdsSpec::new(1, 1, 1, 128, 128, 1.0, 10_000.0)?;
    let plan = provider.plan_bf16_pos_ids(spec)?;
    let stream = queue.stream().clone();
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.key_numel())?);
    let position_ids = Arc::new(DeviceBuffer::from_host(&stream, &[-1_i32])?);
    let query_sentinel = vec![bf16::NAN; spec.query_numel()];
    let key_sentinel = vec![bf16::NAN; spec.key_numel()];
    let query_output = DeviceBuffer::from_host(&stream, &query_sentinel)?;
    let key_output = DeviceBuffer::from_host(&stream, &key_sentinel)?;
    let mut bindings = queue.bindings(5)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let position_ids_handle = bindings.bind_read(position_ids)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_output_handle = bindings.bind_read_write(key_output)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16RopePosIdsArgs::new(
            query_handle,
            key_handle,
            position_ids_handle,
            query_output_handle.write(),
            key_output_handle.write(),
        ),
    )?;
    bindings = scope.finish().wait()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_output = bindings.take_read_write(key_output_handle)?;
    drop(bindings);
    if query_output
        .to_host_vec(&stream)?
        .iter()
        .any(|value| !value.to_f32().is_nan())
        || key_output
            .to_host_vec(&stream)?
            .iter()
            .any(|value| !value.to_f32().is_nan())
    {
        return Err("negative RoPE position did not preserve output sentinels".into());
    }
    println!(
        "{} negative_position=guarded sentinel_preserved=true",
        GateCase::new("rope_h20", "negative_position_guard")
    );
    Ok(())
}

fn run_plan_scope_case(provider: &RopeProvider) -> Result<(), Box<dyn Error>> {
    let partial = Bf16RopePosIdsSpec::new(1, 1, 1, 128, 64, 1.0, 10_000.0)?;
    let error = match provider.plan_bf16_pos_ids(partial) {
        Ok(_) => return Err("first CUDA RoPE plan accepted partial rotary dimensions".into()),
        Err(error) => error,
    };
    if !matches!(
        error,
        RopePlanError::UnsupportedRotaryDimension {
            expected: 128,
            actual: 64,
        }
    ) {
        return Err(format!("partial RoPE plan returned the wrong error: {error}").into());
    }
    println!(
        "{} rotary_dim_64=rejected before_ffi=true",
        GateCase::new("rope_h20", "plan_scope")
    );
    Ok(())
}

fn run_paged_append_case(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePagedKvAppendSpec::new(4, 8, 16, 4, 128, 16)?;
    let page_indptr_host = [0_i32, 1, 3, 5, 8];
    let page_indices_host = [7_i32, 2, 6, 5, 1, 7, 0, 4];
    let last_page_len_host = [3_i32, 16, 1, 9];
    spec.validate_page_table(&page_indptr_host, &page_indices_host, &last_page_len_host)?;
    let query_host = deterministic_bf16(spec.query_numel(), 0x5150_4147);
    let key_host = deterministic_bf16(spec.key_numel(), 0x4b50_4147);
    let value_host = deterministic_bf16(spec.value_numel(), 0x5650_4147);
    let key_pages_host = deterministic_bf16(spec.kv_pages_numel(), 0x4b43_4143);
    let value_pages_host = deterministic_bf16(spec.kv_pages_numel(), 0x5643_4143);
    let mut expected_query = vec![bf16::NAN; spec.query_output_numel()];
    let mut expected_key_pages = key_pages_host.clone();
    let mut expected_value_pages = value_pages_host.clone();
    rope_paged_kv_append_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &mut expected_query,
        &mut expected_key_pages,
        &mut expected_value_pages,
        spec,
    )?;

    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged_append(spec)?;
    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &last_page_len_host)?);
    let query_output =
        DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.query_output_numel()])?;
    let key_pages = DeviceBuffer::from_host(&stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(&stream, &value_pages_host)?;
    let mut bindings = queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;

    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16RopePagedKvAppendArgs::new(
            query_handle,
            key_handle,
            value_handle,
            page_indptr_handle,
            page_indices_handle,
            last_page_len_handle,
            query_output_handle.write(),
            key_pages_handle.write(),
            value_pages_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("fused RoPE append completion covered the wrong command count".into());
    }
    bindings = completion.wait()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);

    let query_comparison = compare_bf16(
        &query_output.to_host_vec(&stream)?,
        &expected_query,
        "fused RoPE query",
    )?;
    let key_comparison = compare_bf16(
        &key_pages.to_host_vec(&stream)?,
        &expected_key_pages,
        "fused RoPE key pages",
    )?;
    let value_comparison = compare_bf16(
        &value_pages.to_host_vec(&stream)?,
        &expected_value_pages,
        "fused RoPE value pages",
    )?;
    if query_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || key_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || value_comparison.max_abs != 0.0
    {
        return Err("fused RoPE paged append exceeded its correctness limits".into());
    }
    println!(
        "{} batch_size=4 query_heads=16 key_heads=4 head_dim=128 page_size=16 \
         positions=2_31_16_40 physical_slots=7x2_6x15_1x0_4x8 \
         layout=NHD style=neox_split_half dtype=BF16 commands=1 stream=non_default \
         query_max_abs={:.9e} query_bit_mismatches={} query_digest={:016x} \
         key_pages_max_abs={:.9e} key_pages_bit_mismatches={} key_pages_digest={:016x} \
         value_pages_max_abs={:.9e} value_pages_bit_mismatches={} value_pages_digest={:016x}",
        GateCase::new("rope_h20", "paged_append"),
        query_comparison.max_abs,
        query_comparison.bit_mismatches,
        query_comparison.digest,
        key_comparison.max_abs,
        key_comparison.bit_mismatches,
        key_comparison.digest,
        value_comparison.max_abs,
        value_comparison.bit_mismatches,
        value_comparison.digest,
    );
    Ok(())
}

fn run_paged_append_tokens_case(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePagedKvAppendTokensSpec::new(6, 3, 8, 16, 4, 128, 16)?;
    let batch_indices_host = [2_i32, 0, 1, 0, 2, 1];
    let positions_host = [5_i32, 17, 20, 16, 4, 19];
    let page_indptr_host = [0_i32, 2, 4, 5];
    let page_indices_host = [7_i32, 3, 2, 6, 3];
    let last_page_len_host = [2_i32, 5, 6];
    let query_host = deterministic_bf16(spec.query_numel(), 0x5451_4147);
    let key_host = deterministic_bf16(spec.key_numel(), 0x544b_4147);
    let value_host = deterministic_bf16(spec.value_numel(), 0x5456_4147);
    let key_pages_host = deterministic_bf16(spec.kv_pages_numel(), 0x544b_4343);
    let value_pages_host = deterministic_bf16(spec.kv_pages_numel(), 0x5456_4343);
    let mut expected_query = vec![bf16::NAN; spec.query_output_numel()];
    let mut expected_key_pages = key_pages_host.clone();
    let mut expected_value_pages = value_pages_host.clone();
    rope_paged_kv_append_tokens_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &batch_indices_host,
        &positions_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &mut expected_query,
        &mut expected_key_pages,
        &mut expected_value_pages,
        spec,
    )?;

    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged_append_tokens(spec)?;
    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let batch_indices = Arc::new(DeviceBuffer::from_host(&stream, &batch_indices_host)?);
    let positions = Arc::new(DeviceBuffer::from_host(&stream, &positions_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &last_page_len_host)?);
    let query_output =
        DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.query_output_numel()])?;
    let key_pages = DeviceBuffer::from_host(&stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(&stream, &value_pages_host)?;
    let mut bindings = queue.bindings(11)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let batch_indices_handle = bindings.bind_read(batch_indices)?;
    let positions_handle = bindings.bind_read(positions)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;

    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16RopePagedKvAppendTokensArgs::new(
            query_handle,
            key_handle,
            value_handle,
            batch_indices_handle,
            positions_handle,
            page_indptr_handle,
            page_indices_handle,
            last_page_len_handle,
            query_output_handle.write(),
            key_pages_handle.write(),
            value_pages_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("explicit fused RoPE append completion covered the wrong command count".into());
    }
    bindings = completion.wait()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);

    let query_comparison = compare_bf16(
        &query_output.to_host_vec(&stream)?,
        &expected_query,
        "explicit fused RoPE query",
    )?;
    let key_comparison = compare_bf16(
        &key_pages.to_host_vec(&stream)?,
        &expected_key_pages,
        "explicit fused RoPE key pages",
    )?;
    let value_comparison = compare_bf16(
        &value_pages.to_host_vec(&stream)?,
        &expected_value_pages,
        "explicit fused RoPE value pages",
    )?;
    if query_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || key_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || value_comparison.max_abs != 0.0
    {
        return Err("explicit fused RoPE paged append exceeded its correctness limits".into());
    }
    println!(
        "{} tokens=6 batch_size=3 query_heads=16 key_heads=4 head_dim=128 page_size=16 \
         batch_indices=2_0_1_0_2_1 positions=5_17_20_16_4_19 \
         physical_slots=3x5_3x1_6x4_3x0_3x4_6x3 \
         layout=NHD style=neox_split_half dtype=BF16 commands=1 stream=non_default \
         query_max_abs={:.9e} query_bit_mismatches={} query_digest={:016x} \
         key_pages_max_abs={:.9e} key_pages_bit_mismatches={} key_pages_digest={:016x} \
         value_pages_max_abs={:.9e} value_pages_bit_mismatches={} value_pages_digest={:016x}",
        GateCase::new("rope_h20", "paged_append_tokens"),
        query_comparison.max_abs,
        query_comparison.bit_mismatches,
        query_comparison.digest,
        key_comparison.max_abs,
        key_comparison.bit_mismatches,
        key_comparison.digest,
        value_comparison.max_abs,
        value_comparison.bit_mismatches,
        value_comparison.digest,
    );
    Ok(())
}

fn run_paged_append_tokens_limit_case(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePagedKvAppendTokensSpec::new(64, 1, 4, 1, 1, 128, 16)?;
    let batch_indices_host = [0_i32; 64];
    let positions_host = (0_i32..64).collect::<Vec<_>>();
    let page_indptr_host = [0_i32, 4];
    let page_indices_host = [3_i32, 1, 2, 0];
    let last_page_len_host = [16_i32];
    let query_host = deterministic_bf16(spec.query_numel(), 0x4c51_4147);
    let key_host = deterministic_bf16(spec.key_numel(), 0x4c4b_4147);
    let value_host = deterministic_bf16(spec.value_numel(), 0x4c56_4147);
    let key_pages_host = deterministic_bf16(spec.kv_pages_numel(), 0x4c4b_4343);
    let value_pages_host = deterministic_bf16(spec.kv_pages_numel(), 0x4c56_4343);
    let mut expected_query = vec![bf16::NAN; spec.query_output_numel()];
    let mut expected_key_pages = key_pages_host.clone();
    let mut expected_value_pages = value_pages_host.clone();
    rope_paged_kv_append_tokens_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &batch_indices_host,
        &positions_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &mut expected_query,
        &mut expected_key_pages,
        &mut expected_value_pages,
        spec,
    )?;

    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged_append_tokens(spec)?;
    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let batch_indices = Arc::new(DeviceBuffer::from_host(&stream, &batch_indices_host)?);
    let positions = Arc::new(DeviceBuffer::from_host(&stream, &positions_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &last_page_len_host)?);
    let query_output =
        DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.query_output_numel()])?;
    let key_pages = DeviceBuffer::from_host(&stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(&stream, &value_pages_host)?;
    let mut bindings = queue.bindings(11)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let batch_indices_handle = bindings.bind_read(batch_indices)?;
    let positions_handle = bindings.bind_read(positions)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16RopePagedKvAppendTokensArgs::new(
            query_handle,
            key_handle,
            value_handle,
            batch_indices_handle,
            positions_handle,
            page_indptr_handle,
            page_indices_handle,
            last_page_len_handle,
            query_output_handle.write(),
            key_pages_handle.write(),
            value_pages_handle.write(),
        ),
    )?;
    bindings = scope.finish().wait()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);

    let query_comparison = compare_bf16(
        &query_output.to_host_vec(&stream)?,
        &expected_query,
        "64-token explicit fused RoPE query",
    )?;
    let key_comparison = compare_bf16(
        &key_pages.to_host_vec(&stream)?,
        &expected_key_pages,
        "64-token explicit fused RoPE key pages",
    )?;
    let value_comparison = compare_bf16(
        &value_pages.to_host_vec(&stream)?,
        &expected_value_pages,
        "64-token explicit fused RoPE value pages",
    )?;
    if query_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || key_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || value_comparison.max_abs != 0.0
    {
        return Err("64-token fused RoPE append exceeded its correctness limits".into());
    }
    println!(
        "{} tokens=64 batch_size=1 query_heads=1 key_heads=1 head_dim=128 page_size=16 \
         positions=0_to_63 physical_pages=3_1_2_0 both_validation_warps=true \
         query_max_abs={:.9e} query_bit_mismatches={} query_digest={:016x} \
         key_pages_max_abs={:.9e} key_pages_bit_mismatches={} key_pages_digest={:016x} \
         value_pages_max_abs={:.9e} value_pages_bit_mismatches={} value_pages_digest={:016x}",
        GateCase::new("rope_h20", "paged_append_tokens_limit"),
        query_comparison.max_abs,
        query_comparison.bit_mismatches,
        query_comparison.digest,
        key_comparison.max_abs,
        key_comparison.bit_mismatches,
        key_comparison.digest,
        value_comparison.max_abs,
        value_comparison.bit_mismatches,
        value_comparison.digest,
    );
    Ok(())
}

fn run_paged_append_tokens_short_metadata_case(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePagedKvAppendTokensSpec::new(2, 1, 1, 1, 1, 128, 16)?;
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged_append_tokens(spec)?;
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.key_numel())?);
    let value = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.value_numel())?);
    let batch_indices = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32])?);
    let positions = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1])?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1])?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32])?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &[2_i32])?);
    let query_output = DeviceBuffer::<bf16>::zeroed(&stream, spec.query_output_numel())?;
    let key_pages = DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_pages_numel())?;
    let value_pages = DeviceBuffer::<bf16>::zeroed(&stream, spec.kv_pages_numel())?;
    let mut bindings = queue.bindings(11)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let batch_indices_handle = bindings.bind_read(batch_indices)?;
    let positions_handle = bindings.bind_read(positions)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;
    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16RopePagedKvAppendTokensArgs::new(
                query_handle,
                key_handle,
                value_handle,
                batch_indices_handle,
                positions_handle,
                page_indptr_handle,
                page_indices_handle,
                last_page_len_handle,
                query_output_handle.write(),
                key_pages_handle.write(),
                value_pages_handle.write(),
            ),
        )
        .expect_err("short explicit append metadata must fail before submission");
    if !matches!(
        error,
        RopeEnqueueError::LengthMismatch {
            operand: "batch_indices",
            expected: 2,
            actual: 1,
        }
    ) {
        return Err(format!("short explicit append metadata returned wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("short explicit append metadata reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!(
        "{} batch_indices=rejected before_ffi=true",
        GateCase::new("rope_h20", "paged_append_tokens_short_metadata")
    );
    Ok(())
}

fn run_paged_append_tokens_graph_case(context: &Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    let provider = RopeProvider::load(context)?;
    let spec = Bf16RopePagedKvAppendTokensSpec::new(6, 3, 8, 16, 4, 128, 16)?;
    let batch_indices_host = [2_i32, 0, 1, 0, 2, 1];
    let positions_host = [5_i32, 17, 20, 16, 4, 19];
    let page_indptr_host = [0_i32, 2, 4, 5];
    let page_indices_host = [7_i32, 3, 2, 6, 3];
    let last_page_len_host = [2_i32, 5, 6];
    let query_host = deterministic_bf16(spec.query_numel(), 0x5451_4147);
    let key_host = deterministic_bf16(spec.key_numel(), 0x544b_4147);
    let value_host = deterministic_bf16(spec.value_numel(), 0x5456_4147);
    let key_pages_host = deterministic_bf16(spec.kv_pages_numel(), 0x544b_4343);
    let value_pages_host = deterministic_bf16(spec.kv_pages_numel(), 0x5456_4343);
    let mut expected_query = vec![bf16::NAN; spec.query_output_numel()];
    let mut expected_key_pages = key_pages_host.clone();
    let mut expected_value_pages = value_pages_host.clone();
    rope_paged_kv_append_tokens_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &batch_indices_host,
        &positions_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &mut expected_query,
        &mut expected_key_pages,
        &mut expected_value_pages,
        spec,
    )?;

    let upload_stream = context.new_stream()?;
    let plan = provider.plan_bf16_paged_append_tokens(spec)?;
    let query = Arc::new(DeviceBuffer::from_host(&upload_stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&upload_stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(&upload_stream, &value_host)?);
    let batch_indices = Arc::new(DeviceBuffer::from_host(
        &upload_stream,
        &batch_indices_host,
    )?);
    let positions = Arc::new(DeviceBuffer::from_host(&upload_stream, &positions_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&upload_stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&upload_stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(
        &upload_stream,
        &last_page_len_host,
    )?);
    let query_output =
        DeviceBuffer::from_host(&upload_stream, &vec![bf16::NAN; spec.query_output_numel()])?;
    let key_pages = DeviceBuffer::from_host(&upload_stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(&upload_stream, &value_pages_host)?;

    let graph_queue = GraphQueue::new(context, 1)?;
    let mut bindings = graph_queue.bindings(11)?;
    let query_handle = bindings.bind_read(Arc::clone(&query))?;
    let key_handle = bindings.bind_read(Arc::clone(&key))?;
    let value_handle = bindings.bind_read(Arc::clone(&value))?;
    let batch_indices_handle = bindings.bind_read(Arc::clone(&batch_indices))?;
    let positions_handle = bindings.bind_read(Arc::clone(&positions))?;
    let page_indptr_handle = bindings.bind_read(Arc::clone(&page_indptr))?;
    let page_indices_handle = bindings.bind_read(Arc::clone(&page_indices))?;
    let last_page_len_handle = bindings.bind_read(Arc::clone(&last_page_len))?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;
    let captured = graph_queue.capture(bindings, |scope| {
        plan.enqueue_into(
            scope,
            Bf16RopePagedKvAppendTokensArgs::new(
                query_handle,
                key_handle,
                value_handle,
                batch_indices_handle,
                positions_handle,
                page_indptr_handle,
                page_indices_handle,
                last_page_len_handle,
                query_output_handle.write(),
                key_pages_handle.write(),
                value_pages_handle.write(),
            ),
        )
    })?;
    if captured.commands() != 1 {
        return Err("explicit append graph captured the wrong command count".into());
    }

    drop(plan);
    drop(provider);
    drop(query);
    drop(key);
    drop(value);
    drop(batch_indices);
    drop(positions);
    drop(page_indptr);
    drop(page_indices);
    drop(last_page_len);

    let mut exec = captured.instantiate()?;
    for expected_launch in 1..=2 {
        let mut completion = exec.launch()?;
        if completion.launch_index() != expected_launch {
            return Err("explicit append graph completion reported wrong replay index".into());
        }
        let _ = completion.is_complete()?;
        if expected_launch == 1 {
            completion.wait()?;
        } else {
            drop(completion);
        }
    }
    if exec.launches() != 2 || exec.commands() != 1 {
        return Err("explicit append graph accounting changed across replay".into());
    }

    let mut bindings = exec.into_bindings()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);
    let query_comparison = compare_bf16(
        &query_output.to_host_vec(&upload_stream)?,
        &expected_query,
        "explicit append graph query",
    )?;
    let key_comparison = compare_bf16(
        &key_pages.to_host_vec(&upload_stream)?,
        &expected_key_pages,
        "explicit append graph key pages",
    )?;
    let value_comparison = compare_bf16(
        &value_pages.to_host_vec(&upload_stream)?,
        &expected_value_pages,
        "explicit append graph value pages",
    )?;
    if query_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || key_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || value_comparison.max_abs != 0.0
    {
        return Err("explicit append graph exceeded its correctness limits".into());
    }
    println!(
        "{} tokens=6 batch_size=3 commands=1 replays=2 fixed_bindings=true \
         cross_stream=false external_owners_dropped_before_replay=true \
         completion_queries=2 completion_waits=1 completion_drops=1 \
         query_max_abs={:.9e} query_bit_mismatches={} query_digest={:016x} \
         key_pages_max_abs={:.9e} key_pages_bit_mismatches={} key_pages_digest={:016x} \
         value_pages_max_abs={:.9e} value_pages_bit_mismatches={} value_pages_digest={:016x}",
        GateCase::new("rope_h20", "paged_append_tokens_graph"),
        query_comparison.max_abs,
        query_comparison.bit_mismatches,
        query_comparison.digest,
        key_comparison.max_abs,
        key_comparison.bit_mismatches,
        key_comparison.digest,
        value_comparison.max_abs,
        value_comparison.bit_mismatches,
        value_comparison.digest,
    );
    Ok(())
}

fn run_append_metadata_guard(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
    page_indptr_host: &[i32],
    page_indices_host: &[i32],
    last_page_len_host: &[i32],
    failure: &'static str,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePagedKvAppendSpec::new(2, 2, 2, 1, 128, 16)?;
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged_append(spec)?;
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.key_numel())?);
    let value = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.value_numel())?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, last_page_len_host)?);
    let query_sentinel = vec![bf16::NAN; spec.query_output_numel()];
    let key_sentinel = vec![bf16::from_f32(-7.0); spec.kv_pages_numel()];
    let value_sentinel = vec![bf16::from_f32(9.0); spec.kv_pages_numel()];
    let query_output = DeviceBuffer::from_host(&stream, &query_sentinel)?;
    let key_pages = DeviceBuffer::from_host(&stream, &key_sentinel)?;
    let value_pages = DeviceBuffer::from_host(&stream, &value_sentinel)?;
    let mut bindings = queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16RopePagedKvAppendArgs::new(
            query_handle,
            key_handle,
            value_handle,
            page_indptr_handle,
            page_indices_handle,
            last_page_len_handle,
            query_output_handle.write(),
            key_pages_handle.write(),
            value_pages_handle.write(),
        ),
    )?;
    bindings = scope.finish().wait()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);
    if query_output
        .to_host_vec(&stream)?
        .iter()
        .any(|value| !value.to_f32().is_nan())
        || key_pages.to_host_vec(&stream)? != key_sentinel
        || value_pages.to_host_vec(&stream)? != value_sentinel
    {
        return Err(failure.into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_append_tokens_metadata_guard(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
    batch_indices_host: &[i32],
    positions_host: &[i32],
    page_indptr_host: &[i32],
    page_indices_host: &[i32],
    last_page_len_host: &[i32],
    failure: &'static str,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePagedKvAppendTokensSpec::new(2, 2, 4, 2, 1, 128, 16)?;
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged_append_tokens(spec)?;
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.key_numel())?);
    let value = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.value_numel())?);
    let batch_indices = Arc::new(DeviceBuffer::from_host(&stream, batch_indices_host)?);
    let positions = Arc::new(DeviceBuffer::from_host(&stream, positions_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, last_page_len_host)?);
    let query_sentinel = vec![bf16::NAN; spec.query_output_numel()];
    let key_sentinel = vec![bf16::from_f32(-7.0); spec.kv_pages_numel()];
    let value_sentinel = vec![bf16::from_f32(9.0); spec.kv_pages_numel()];
    let query_output = DeviceBuffer::from_host(&stream, &query_sentinel)?;
    let key_pages = DeviceBuffer::from_host(&stream, &key_sentinel)?;
    let value_pages = DeviceBuffer::from_host(&stream, &value_sentinel)?;
    let mut bindings = queue.bindings(11)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let batch_indices_handle = bindings.bind_read(batch_indices)?;
    let positions_handle = bindings.bind_read(positions)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16RopePagedKvAppendTokensArgs::new(
            query_handle,
            key_handle,
            value_handle,
            batch_indices_handle,
            positions_handle,
            page_indptr_handle,
            page_indices_handle,
            last_page_len_handle,
            query_output_handle.write(),
            key_pages_handle.write(),
            value_pages_handle.write(),
        ),
    )?;
    bindings = scope.finish().wait()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);
    if query_output
        .to_host_vec(&stream)?
        .iter()
        .any(|value| !value.to_f32().is_nan())
        || key_pages.to_host_vec(&stream)? != key_sentinel
        || value_pages.to_host_vec(&stream)? != value_sentinel
    {
        return Err(failure.into());
    }
    Ok(())
}

fn run_duplicate_append_guard(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    run_append_metadata_guard(
        queue,
        provider,
        &[0, 1, 2],
        &[1, 1],
        &[4, 4],
        "duplicate append slot did not preserve all output sentinels",
    )?;
    println!(
        "{} duplicate_final_slot=guarded sentinels_preserved=true",
        GateCase::new("rope_h20", "paged_append_duplicate_guard")
    );
    Ok(())
}

fn run_invalid_page_guard(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    run_append_metadata_guard(
        queue,
        provider,
        &[0, 2, 3],
        &[2, 0, 1],
        &[4, 5],
        "invalid non-final page index did not preserve all output sentinels",
    )?;
    println!(
        "{} invalid_non_final_page=guarded sentinels_preserved=true",
        GateCase::new("rope_h20", "paged_append_invalid_page_guard")
    );
    Ok(())
}

fn run_append_tokens_guards(
    queue: &mut CommandQueue,
    provider: &RopeProvider,
) -> Result<(), Box<dyn Error>> {
    let page_indptr = [0, 2, 4];
    let page_indices = [3, 1, 3, 2];
    let last_page_len = [4, 4];
    for (case, batch_indices, positions, indices, failure) in [
        (
            "duplicate_slot",
            [0, 1],
            [2, 2],
            page_indices,
            "duplicate explicit append slot changed output sentinels",
        ),
        (
            "batch_index",
            [0, 2],
            [1, 1],
            page_indices,
            "out-of-range explicit batch index changed output sentinels",
        ),
        (
            "position",
            [0, 1],
            [1, 20],
            page_indices,
            "out-of-range explicit position changed output sentinels",
        ),
        (
            "page_index",
            [0, 1],
            [16, 16],
            [4, 1, 3, 2],
            "invalid explicit non-final page changed output sentinels",
        ),
    ] {
        run_append_tokens_metadata_guard(
            queue,
            provider,
            &batch_indices,
            &positions,
            &page_indptr,
            &indices,
            &last_page_len,
            failure,
        )?;
        println!(
            "{} invalid_case={} sentinels_preserved=true",
            GateCase::new("rope_h20", "paged_append_tokens_guard"),
            case,
        );
    }
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let provider = RopeProvider::load(&context)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let mut queue = CommandQueue::new(stream, 1)?;

    run_reference_case(&mut queue, &provider)?;
    run_short_buffer_case(&mut queue, &provider)?;
    run_negative_position_guard(&mut queue, &provider)?;
    run_plan_scope_case(&provider)?;
    run_paged_append_case(&mut queue, &provider)?;
    run_paged_append_tokens_case(&mut queue, &provider)?;
    run_paged_append_tokens_limit_case(&mut queue, &provider)?;
    run_paged_append_tokens_short_metadata_case(&mut queue, &provider)?;
    run_paged_append_tokens_graph_case(&context)?;
    run_duplicate_append_guard(&mut queue, &provider)?;
    run_invalid_page_guard(&mut queue, &provider)?;
    run_append_tokens_guards(&mut queue, &provider)?;
    println!(
        "gate=rope_h20 suite=all status=pass output_max_abs_limit={:.9e}",
        OUTPUT_MAX_ABS_LIMIT
    );
    Ok(())
}
