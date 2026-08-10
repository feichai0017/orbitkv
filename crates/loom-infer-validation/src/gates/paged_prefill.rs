use crate::comparison::{compare_bf16, compare_f32};
use crate::fixture::deterministic_bf16;
use crate::reporting::GateCase;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use loom_infer::{Bf16PagedPrefillSpec, paged_prefill_bf16_reference};
use loom_infer_cuda::attention::{
    Bf16PagedPrefillAlgorithm, Bf16PagedPrefillArgs, Bf16PagedPrefillPlan,
    PagedPrefillEnqueueError, PrefillProvider,
};
use loom_infer_cuda::command::{CommandError, CommandQueue};
use loom_infer_cuda::graph::GraphQueue;
use std::error::Error;
use std::sync::Arc;

const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const LSE_MAX_ABS_LIMIT: f32 = 0.01;

#[derive(Clone, Copy)]
struct MetadataInput<'a> {
    qo_indptr: &'a [i32],
    page_indptr: &'a [i32],
    page_indices: &'a [i32],
    last_page_len: &'a [i32],
}

fn run_case(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
    name: &str,
    spec: Bf16PagedPrefillSpec,
    metadata: MetadataInput<'_>,
    salt: u64,
) -> Result<(), Box<dyn Error>> {
    let validated = spec.validate_metadata(
        metadata.qo_indptr,
        metadata.page_indptr,
        metadata.page_indices,
        metadata.last_page_len,
    )?;
    let kv_lens = (0..spec.batch_size())
        .map(|request| validated.request_kv_len(request).unwrap().to_string())
        .collect::<Vec<_>>()
        .join(",");
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged(spec)?;
    let algorithm = match plan.algorithm() {
        Bf16PagedPrefillAlgorithm::Direct => "direct_one_warp_per_query_row_head",
        Bf16PagedPrefillAlgorithm::TokenParallel8 => "token_parallel_8warp_block_local_merge",
        Bf16PagedPrefillAlgorithm::TokenParallel16 => "token_parallel_16warp_block_local_merge",
    };
    let query_host = deterministic_bf16(spec.query_numel(), salt);
    let key_host = deterministic_bf16(spec.kv_pages_numel(), salt ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_pages_numel(), salt ^ 0x5641_4c55_4500);
    let mut expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut expected_lse = vec![f32::NAN; spec.lse_numel()];
    paged_prefill_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        metadata.qo_indptr,
        metadata.page_indptr,
        metadata.page_indices,
        metadata.last_page_len,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;

    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key_pages = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value_pages = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, metadata.qo_indptr)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, metadata.page_indptr)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, metadata.page_indices)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, metadata.last_page_len)?);
    let output = DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.output_numel()])?;
    let lse = DeviceBuffer::from_host(&stream, &vec![f32::NAN; spec.lse_numel()])?;
    let mut bindings = queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16PagedPrefillArgs::new(
            query_handle,
            key_handle,
            value_handle,
            qo_handle,
            page_indptr_handle,
            page_indices_handle,
            last_page_len_handle,
            output_handle.write(),
            lse_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("paged prefill completion covered the wrong command count".into());
    }
    bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);

    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    let output_comparison = compare_bf16(&actual_output, &expected_output, "paged prefill BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "paged prefill F32 LSE")?;
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
        "{} batch_size={} nnz_qo={} query_heads={} kv_heads={} group_size={} \
         max_num_pages={} referenced_pages={} page_size={} kv_lens={} layout=NHD \
         causal=bottom_right dtype=BF16 accumulation=F32 lse_domain=log2 \
         algorithm={} commands=1 stream=non_default \
         output_max_abs={:.9e} output_bit_mismatches={} output_digest={:016x} \
         lse_max_abs={:.9e} lse_bit_mismatches={} lse_digest={:016x}",
        GateCase::new("paged_prefill_h20", name),
        spec.batch_size(),
        spec.nnz_qo(),
        spec.num_query_heads(),
        spec.num_kv_heads(),
        spec.gqa_group_size(),
        spec.max_num_pages(),
        metadata.page_indices.len(),
        spec.page_size(),
        kv_lens,
        algorithm,
        output_comparison.max_abs,
        output_comparison.bit_mismatches,
        output_comparison.digest,
        lse_comparison.max_abs,
        lse_comparison.bit_mismatches,
        lse_comparison.digest,
    );
    Ok(())
}

fn run_short_metadata_case(
    queue: &mut CommandQueue,
    plan: &Bf16PagedPrefillPlan,
) -> Result<(), Box<dyn Error>> {
    let spec = plan.spec();
    let stream = queue.stream().clone();
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let value_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let qo_indptr = Arc::new(DeviceBuffer::<i32>::zeroed(
        &stream,
        spec.indptr_numel() - 1,
    )?);
    let page_indptr = Arc::new(DeviceBuffer::<i32>::zeroed(&stream, spec.indptr_numel())?);
    let page_indices = Arc::new(DeviceBuffer::<i32>::zeroed(&stream, spec.batch_size())?);
    let last_page_len = Arc::new(DeviceBuffer::<i32>::zeroed(
        &stream,
        spec.last_page_len_numel(),
    )?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16PagedPrefillArgs::new(
                query_handle,
                key_handle,
                value_handle,
                qo_handle,
                page_indptr_handle,
                page_indices_handle,
                last_page_len_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
        .expect_err("short qo_indptr must fail before submission");
    if !matches!(
        error,
        PagedPrefillEnqueueError::LengthMismatch {
            operand: "qo_indptr",
            expected,
            actual,
        } if expected == spec.indptr_numel() && actual + 1 == expected
    ) {
        return Err(format!("short qo_indptr returned the wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("short paged-prefill metadata reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!(
        "{} qo_indptr=rejected before_ffi=true",
        GateCase::new("paged_prefill_h20", "short_metadata")
    );
    Ok(())
}

fn run_invalid_page_guard(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedPrefillSpec::new(2, 2, 8, 1, 1, 128, 16)?;
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged(spec)?;
    if plan.algorithm() != Bf16PagedPrefillAlgorithm::TokenParallel16 {
        return Err("invalid-page guard did not select token-parallel paged prefill".into());
    }
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let value_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1, 2])?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1, 2])?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 8])?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &[1_i32, 1])?);
    let output = DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.output_numel()])?;
    let lse = DeviceBuffer::from_host(&stream, &vec![f32::NAN; spec.lse_numel()])?;
    let mut bindings = queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16PagedPrefillArgs::new(
            query_handle,
            key_handle,
            value_handle,
            qo_handle,
            page_indptr_handle,
            page_indices_handle,
            last_page_len_handle,
            output_handle.write(),
            lse_handle.write(),
        ),
    )?;
    bindings = scope.finish().wait()?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);
    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    if actual_output[..spec.head_dim()]
        .iter()
        .any(|&value| value != bf16::ZERO)
        || actual_lse[0] != 0.0
    {
        return Err("valid request in paged-prefill guard case produced the wrong output".into());
    }
    if actual_output[spec.head_dim()..]
        .iter()
        .any(|value| !value.to_f32().is_nan())
        || !actual_lse[1].is_nan()
    {
        return Err("invalid page did not preserve paged-prefill sentinels".into());
    }
    println!(
        "{} invalid_physical_page=guarded valid_request=completed invalid_request=sentinel_preserved",
        GateCase::new("paged_prefill_h20", "invalid_page_guard")
    );
    Ok(())
}

fn run_duplicate_binding_case(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedPrefillSpec::new(1, 1, 1, 1, 1, 128, 16)?;
    let plan = provider.plan_bf16_paged(spec)?;
    let stream = queue.stream().clone();
    let query_and_key = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let value_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1])?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1])?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32])?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &[1_i32])?);
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(8)?;
    let query_and_key_handle = bindings.bind_read(query_and_key)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16PagedPrefillArgs::new(
                query_and_key_handle,
                query_and_key_handle,
                value_handle,
                qo_handle,
                page_indptr_handle,
                page_indices_handle,
                last_page_len_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
        .expect_err("duplicate paged-prefill binding must fail before submission");
    if !matches!(
        error,
        PagedPrefillEnqueueError::Command(CommandError::DuplicateBindingSlot)
    ) {
        return Err(format!("duplicate binding returned the wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("duplicate paged-prefill binding reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!(
        "{} rejected_before_ffi=true",
        GateCase::new("paged_prefill_h20", "duplicate_binding")
    );
    Ok(())
}

fn run_graph_case(
    queue: &mut CommandQueue,
    provider: PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedPrefillSpec::new(2, 6, 6, 16, 4, 128, 16)?;
    let qo_indptr_host = [0_i32, 4, 6];
    let page_indptr_host = [0_i32, 2, 4];
    let page_indices_host = [5_i32, 1, 5, 3];
    let last_page_len_host = [7_i32, 2];
    spec.validate_metadata(
        &qo_indptr_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
    )?;
    let plan = provider.plan_bf16_paged(spec)?;
    let stream = queue.stream().clone();
    let query_host = deterministic_bf16(spec.query_numel(), 0x4001);
    let key_host = deterministic_bf16(spec.kv_pages_numel(), 0x4001 ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_pages_numel(), 0x4001 ^ 0x5641_4c55_4500);
    let mut expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut expected_lse = vec![f32::NAN; spec.lse_numel()];
    paged_prefill_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &qo_indptr_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;

    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key_pages = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value_pages = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &qo_indptr_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &last_page_len_host)?);
    let output = DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.output_numel()])?;
    let lse = DeviceBuffer::from_host(&stream, &vec![f32::NAN; spec.lse_numel()])?;

    let graph_queue = GraphQueue::new(stream.context(), 1)?;
    let mut bindings = graph_queue.bindings(9)?;
    let query_handle = bindings.bind_read(Arc::clone(&query))?;
    let key_handle = bindings.bind_read(Arc::clone(&key_pages))?;
    let value_handle = bindings.bind_read(Arc::clone(&value_pages))?;
    let qo_handle = bindings.bind_read(Arc::clone(&qo_indptr))?;
    let page_indptr_handle = bindings.bind_read(Arc::clone(&page_indptr))?;
    let page_indices_handle = bindings.bind_read(Arc::clone(&page_indices))?;
    let last_page_len_handle = bindings.bind_read(Arc::clone(&last_page_len))?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let captured = graph_queue.capture(bindings, |scope| {
        plan.enqueue_into(
            scope,
            Bf16PagedPrefillArgs::new(
                query_handle,
                key_handle,
                value_handle,
                qo_handle,
                page_indptr_handle,
                page_indices_handle,
                last_page_len_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
    })?;
    if captured.commands() != 1 {
        return Err("paged-prefill graph captured the wrong command count".into());
    }

    drop(plan);
    drop(provider);
    drop(query);
    drop(key_pages);
    drop(value_pages);
    drop(qo_indptr);
    drop(page_indptr);
    drop(page_indices);
    drop(last_page_len);

    let mut exec = captured.instantiate()?;
    for expected_launch in 1..=2 {
        let mut completion = exec.launch()?;
        if completion.launch_index() != expected_launch {
            return Err("paged-prefill graph reported the wrong replay index".into());
        }
        let _ = completion.is_complete()?;
        if expected_launch == 1 {
            completion.wait()?;
        } else {
            drop(completion);
        }
    }
    if exec.launches() != 2 || exec.commands() != 1 {
        return Err("paged-prefill graph accounting changed across replay".into());
    }

    let mut bindings = exec.into_bindings()?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);
    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    let output_comparison =
        compare_bf16(&actual_output, &expected_output, "graph paged prefill BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "graph paged prefill F32 LSE")?;
    if output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT {
        return Err(format!(
            "paged-prefill graph output max abs {:.9e} exceeds {:.9e}",
            output_comparison.max_abs, OUTPUT_MAX_ABS_LIMIT
        )
        .into());
    }
    if lse_comparison.max_abs > LSE_MAX_ABS_LIMIT {
        return Err(format!(
            "paged-prefill graph LSE max abs {:.9e} exceeds {:.9e}",
            lse_comparison.max_abs, LSE_MAX_ABS_LIMIT
        )
        .into());
    }

    println!(
        "{} batch_size=2 nnz_qo=6 query_heads=16 kv_heads=4 page_size=16 \
         algorithm=direct_one_warp_per_query_row_head commands=1 replays=2 \
         fixed_bindings=true cross_stream=false external_owners_dropped_before_replay=true \
         completion_queries=2 completion_waits=1 completion_drops=1 \
         output_max_abs={:.9e} output_bit_mismatches={} output_digest={:016x} \
         lse_max_abs={:.9e} lse_bit_mismatches={} lse_digest={:016x}",
        GateCase::new("paged_prefill_h20", "gqa4_graph"),
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
    let mut queue = CommandQueue::new(stream, 1)?;

    run_case(
        &mut queue,
        &provider,
        "mha_equal_lengths",
        Bf16PagedPrefillSpec::new(1, 4, 2, 8, 8, 128, 16)?,
        MetadataInput {
            qo_indptr: &[0, 4],
            page_indptr: &[0, 1],
            page_indices: &[1],
            last_page_len: &[4],
        },
        0x1001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "mqa_append_mixed",
        Bf16PagedPrefillSpec::new(3, 6, 7, 8, 1, 128, 16)?,
        MetadataInput {
            qo_indptr: &[0, 2, 5, 6],
            page_indptr: &[0, 1, 3, 6],
            page_indices: &[4, 6, 1, 5, 0, 3],
            last_page_len: &[4, 6, 3],
        },
        0x2001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "gqa4_reordered_reuse",
        Bf16PagedPrefillSpec::new(2, 6, 6, 16, 4, 128, 16)?,
        MetadataInput {
            qo_indptr: &[0, 4, 6],
            page_indptr: &[0, 2, 4],
            page_indices: &[5, 1, 5, 3],
            last_page_len: &[7, 2],
        },
        0x4001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "mqa_token_parallel",
        Bf16PagedPrefillSpec::new(3, 21, 64, 8, 1, 128, 16)?,
        MetadataInput {
            qo_indptr: &[0, 1, 5, 21],
            page_indptr: &[0, 8, 24, 56],
            page_indices: &[
                7, 2, 11, 5, 13, 3, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 1, 4, 8, 14,
                22, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 0, 6, 9,
                10, 12, 15, 16, 18, 20, 21, 24, 25, 26, 27, 30, 33,
            ],
            last_page_len: &[16, 16, 16],
        },
        0x6001,
    )?;
    run_case(
        &mut queue,
        &provider,
        "gqa4_token_parallel",
        Bf16PagedPrefillSpec::new(2, 96, 96, 16, 4, 128, 16)?,
        MetadataInput {
            qo_indptr: &[0, 32, 96],
            page_indptr: &[0, 16, 80],
            page_indices: &[
                15, 3, 27, 9, 31, 1, 35, 5, 39, 7, 43, 11, 47, 13, 51, 17, 19, 21, 23, 25, 29, 33,
                37, 41, 45, 49, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85,
                87, 89, 91, 93, 95, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
            ],
            last_page_len: &[16, 16],
        },
        0x8001,
    )?;

    let preflight_spec = Bf16PagedPrefillSpec::new(2, 2, 2, 4, 2, 128, 16)?;
    let preflight_plan = provider.plan_bf16_paged(preflight_spec)?;
    run_short_metadata_case(&mut queue, &preflight_plan)?;
    run_invalid_page_guard(&mut queue, &provider)?;
    run_duplicate_binding_case(&mut queue, &provider)?;
    run_graph_case(&mut queue, provider)?;
    println!(
        "gate=paged_prefill_h20 suite=all status=pass output_max_abs_limit={:.9e} \
         lse_max_abs_limit={:.9e}",
        OUTPUT_MAX_ABS_LIMIT, LSE_MAX_ABS_LIMIT
    );
    Ok(())
}
