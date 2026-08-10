use crate::comparison::{compare_bf16, compare_f32};
use crate::fixture::deterministic_bf16;
use crate::reporting::GateCase;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use loom_infer::{
    Bf16PagedBatchDecodeSpec, ContractError, PAGED_BATCH_DECODE_PAGE_SIZE, PagedKvLayout,
    SINGLE_DECODE_HEAD_DIM, paged_batch_decode_bf16_reference,
};
use loom_infer_cuda::attention::{
    Bf16PagedBatchDecodeArgs, Bf16PagedBatchDecodePlan, DecodeProvider,
    PagedBatchDecodeEnqueueError,
};
use loom_infer_cuda::command::{CommandCompletionError, CommandQueue};
use loom_infer_cuda::graph::{GraphBindingsError, GraphError, GraphQueue};
use std::error::Error;
use std::sync::Arc;

const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const LSE_MAX_ABS_LIMIT: f32 = 0.01;

#[derive(Clone, Copy)]
struct PageTableInput<'a> {
    indptr: &'a [i32],
    indices: &'a [i32],
    last_page_len: &'a [i32],
}

fn run_case(
    queue: &mut CommandQueue,
    provider: &DecodeProvider,
    name: &str,
    spec: Bf16PagedBatchDecodeSpec,
    page_table: PageTableInput<'_>,
    salt: u64,
) -> Result<(), Box<dyn Error>> {
    let table = spec.validate_page_table(
        page_table.indptr,
        page_table.indices,
        page_table.last_page_len,
    )?;
    let referenced_pages = table.page_indices().len();
    let layout = match spec.kv_layout() {
        PagedKvLayout::Nhd => "NHD",
        PagedKvLayout::Hnd => "HND",
    };
    let kv_lens = (0..spec.batch_size())
        .map(|request| table.request_kv_len(request).unwrap().to_string())
        .collect::<Vec<_>>()
        .join(",");
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged_batch(spec)?;
    let query_host = deterministic_bf16(spec.query_numel(), salt);
    let key_host = deterministic_bf16(spec.kv_pages_numel(), salt ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_pages_numel(), salt ^ 0x5641_4c55_4500);
    let mut expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut expected_lse = vec![f32::NAN; spec.lse_numel()];
    paged_batch_decode_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        page_table.indptr,
        page_table.indices,
        page_table.last_page_len,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;

    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key_pages = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value_pages = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, page_table.indptr)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, page_table.indices)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, page_table.last_page_len)?);
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&stream, plan.metadata_status_required_numel())?;
    let output = DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.output_numel()])?;
    let lse = DeviceBuffer::from_host(&stream, &vec![f32::NAN; spec.lse_numel()])?;
    let mut bindings = queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let indptr_handle = bindings.bind_read(page_indptr)?;
    let indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16PagedBatchDecodeArgs::new(
            query_handle,
            key_handle,
            value_handle,
            indptr_handle,
            indices_handle,
            last_page_len_handle,
            metadata_status_handle,
            output_handle.write(),
            lse_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 3 {
        return Err("paged batch decode completion covered the wrong command count".into());
    }
    bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);

    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    let output_comparison = compare_bf16(&actual_output, &expected_output, "paged BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "paged F32 LSE")?;
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
        "{} batch_size={} query_heads={} kv_heads={} group_size={} \
         max_num_pages={} referenced_pages={} page_size={} kv_lens={} layout={} \
         dtype=BF16 accumulation=F32 lse_domain=log2 commands=3 \
         validator_kernels=1 attention_kernels=1 status_readbacks=1 stream=non_default \
         output_max_abs={:.9e} output_bit_mismatches={} output_digest={:016x} \
         lse_max_abs={:.9e} lse_bit_mismatches={} lse_digest={:016x}",
        GateCase::new("paged_batch_decode_h20", name),
        spec.batch_size(),
        spec.num_query_heads(),
        spec.num_kv_heads(),
        spec.gqa_group_size(),
        spec.max_num_pages(),
        referenced_pages,
        spec.page_size(),
        kv_lens,
        layout,
        output_comparison.max_abs,
        output_comparison.bit_mismatches,
        output_comparison.digest,
        lse_comparison.max_abs,
        lse_comparison.bit_mismatches,
        lse_comparison.digest,
    );
    Ok(())
}

fn run_short_page_indptr_case(
    queue: &mut CommandQueue,
    plan: &Bf16PagedBatchDecodePlan,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let spec = plan.spec();
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let value_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let page_indptr = Arc::new(DeviceBuffer::<i32>::zeroed(
        &stream,
        spec.page_indptr_numel() - 1,
    )?);
    let page_indices = Arc::new(DeviceBuffer::<i32>::zeroed(&stream, spec.batch_size())?);
    let last_page_len = Arc::new(DeviceBuffer::<i32>::zeroed(
        &stream,
        spec.last_page_len_numel(),
    )?);
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&stream, plan.metadata_status_required_numel())?;
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let indptr_handle = bindings.bind_read(page_indptr)?;
    let indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    let error = plan
        .enqueue_into(
            &mut scope,
            Bf16PagedBatchDecodeArgs::new(
                query_handle,
                key_handle,
                value_handle,
                indptr_handle,
                indices_handle,
                last_page_len_handle,
                metadata_status_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
        .expect_err("short paged indptr must be rejected");
    if !matches!(
        error,
        PagedBatchDecodeEnqueueError::LengthMismatch {
            operand: "page_indptr",
            expected,
            actual,
        } if expected == spec.page_indptr_numel() && actual + 1 == expected
    ) {
        return Err(format!("short page_indptr returned the wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("short paged metadata reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!(
        "{} page_indptr=rejected before_ffi=true",
        GateCase::new("paged_batch_decode_h20", "short_metadata")
    );
    Ok(())
}

fn run_invalid_page_guard_case(
    queue: &mut CommandQueue,
    provider: &DecodeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedBatchDecodeSpec::new(2, 2, 4, 2, 128, 16, PagedKvLayout::Hnd)?;
    let expected_error = ContractError::PageIndexOutOfRange {
        position: 1,
        index: 2,
        max_num_pages: 2,
    };
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged_batch(spec)?;
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, spec.query_numel())?);
    let key_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let value_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.kv_pages_numel(),
    )?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1, 2])?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 2])?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &[16_i32, 16])?);
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&stream, plan.metadata_status_required_numel())?;
    let output_sentinel = vec![bf16::NAN; spec.output_numel()];
    let lse_sentinel = vec![f32::NAN; spec.lse_numel()];
    let output = DeviceBuffer::from_host(&stream, &output_sentinel)?;
    let lse = DeviceBuffer::from_host(&stream, &lse_sentinel)?;
    let mut bindings = queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let indptr_handle = bindings.bind_read(page_indptr)?;
    let indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16PagedBatchDecodeArgs::new(
            query_handle,
            key_handle,
            value_handle,
            indptr_handle,
            indices_handle,
            last_page_len_handle,
            metadata_status_handle,
            output_handle.write(),
            lse_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 3 {
        return Err("invalid-page rejection covered the wrong command count".into());
    }
    bindings = match completion.wait() {
        Err(CommandCompletionError::DeviceRejected(rejection)) => {
            if rejection.error() != expected_error {
                return Err(format!(
                    "paged decode returned the wrong device rejection: expected \
                     {expected_error}, got {}",
                    rejection.error()
                )
                .into());
            }
            rejection.into_parts().1
        }
        Err(error) => return Err(format!("paged decode returned the wrong error: {error}").into()),
        Ok(_) => return Err("invalid paged-decode metadata was not rejected".into()),
    };
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    let scope = queue.begin(bindings)?;
    let recovered = scope.finish();
    if recovered.submitted() != 0 {
        return Err("device rejection recovery submitted an unexpected command".into());
    }
    drop(recovered.wait()?);
    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    if actual_output.iter().any(|value| !value.to_f32().is_nan())
        || actual_lse.iter().any(|value| !value.is_nan())
    {
        return Err("rejected paged decode changed output sentinels".into());
    }
    println!(
        "{} invalid_physical_page=device_rejected bindings_recovered=true \
         queue_reusable=true sentinels_unchanged=true commands=3",
        GateCase::new("paged_batch_decode_h20", "invalid_page_guard")
    );
    Ok(())
}

fn run_invalid_page_graph_rejection_case(
    context: &Arc<CudaContext>,
    provider: &DecodeProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedBatchDecodeSpec::new(2, 2, 4, 2, 128, 16, PagedKvLayout::Hnd)?;
    let expected_error = ContractError::PageIndexOutOfRange {
        position: 1,
        index: 2,
        max_num_pages: 2,
    };
    let upload_stream = context.new_stream()?;
    let plan = provider.plan_bf16_paged_batch(spec)?;
    let query = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &upload_stream,
        spec.query_numel(),
    )?);
    let key_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &upload_stream,
        spec.kv_pages_numel(),
    )?);
    let value_pages = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &upload_stream,
        spec.kv_pages_numel(),
    )?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&upload_stream, &[0_i32, 1, 2])?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&upload_stream, &[0_i32, 2])?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&upload_stream, &[16_i32, 16])?);
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&upload_stream, plan.metadata_status_required_numel())?;
    let output_sentinel = vec![bf16::NAN; spec.output_numel()];
    let lse_sentinel = vec![f32::NAN; spec.lse_numel()];
    let output = DeviceBuffer::from_host(&upload_stream, &output_sentinel)?;
    let lse = DeviceBuffer::from_host(&upload_stream, &lse_sentinel)?;

    let graph_queue = GraphQueue::new(context, 3)?;
    let mut bindings = graph_queue.bindings(9)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let indptr_handle = bindings.bind_read(page_indptr)?;
    let indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let captured = graph_queue.capture(bindings, |scope| {
        plan.enqueue_into(
            scope,
            Bf16PagedBatchDecodeArgs::new(
                query_handle,
                key_handle,
                value_handle,
                indptr_handle,
                indices_handle,
                last_page_len_handle,
                metadata_status_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
    })?;
    if captured.commands() != 3 {
        return Err("rejected paged-decode graph captured the wrong command count".into());
    }

    let mut exec = captured.instantiate()?;
    for expected_launch in 1..=2 {
        let completion = exec.launch()?;
        if completion.launch_index() != expected_launch {
            return Err("rejected paged-decode graph reported the wrong replay index".into());
        }
        match completion.wait() {
            Err(GraphError::DeviceRejected(error)) if error == expected_error => {}
            Err(error) => {
                return Err(format!(
                    "rejected paged-decode graph returned the wrong error: {error}"
                )
                .into());
            }
            Ok(()) => return Err("invalid paged-decode graph metadata was not rejected".into()),
        }
    }
    if exec.launches() != 2 || exec.commands() != 3 {
        return Err("rejected paged-decode graph accounting changed across replay".into());
    }

    let mut bindings = match exec.into_bindings() {
        Err(GraphBindingsError::DeviceRejected(rejection)) => {
            if rejection.error() != expected_error {
                return Err(format!(
                    "rejected paged-decode graph returned the wrong bindings error: expected \
                     {expected_error}, got {}",
                    rejection.error()
                )
                .into());
            }
            rejection.into_parts().1
        }
        Err(error) => {
            return Err(
                format!("rejected paged-decode graph could not recover bindings: {error}").into(),
            );
        }
        Ok(_) => {
            return Err("invalid paged-decode graph returned bindings without rejection".into());
        }
    };
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);
    if output
        .to_host_vec(&upload_stream)?
        .iter()
        .any(|value| !value.to_f32().is_nan())
        || lse
            .to_host_vec(&upload_stream)?
            .iter()
            .any(|value| !value.is_nan())
    {
        return Err("rejected paged-decode graph changed output sentinels".into());
    }

    println!(
        "{} replays=2 fixed_bindings=true device_rejected=true \
         graph_reusable=true bindings_recovered=true sentinels_unchanged=true commands=3",
        GateCase::new("paged_batch_decode_h20", "invalid_page_graph")
    );
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let provider = DecodeProvider::load(&context)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let mut queue = CommandQueue::new(stream, 3)?;

    for layout in [PagedKvLayout::Nhd, PagedKvLayout::Hnd] {
        let layout_name = match layout {
            PagedKvLayout::Nhd => "nhd",
            PagedKvLayout::Hnd => "hnd",
        };
        run_case(
            &mut queue,
            &provider,
            &format!("mha_b1_l1_{layout_name}"),
            Bf16PagedBatchDecodeSpec::new(
                1,
                2,
                8,
                8,
                SINGLE_DECODE_HEAD_DIM,
                PAGED_BATCH_DECODE_PAGE_SIZE,
                layout,
            )?,
            PageTableInput {
                indptr: &[0, 1],
                indices: &[1],
                last_page_len: &[1],
            },
            0x1001,
        )?;
        run_case(
            &mut queue,
            &provider,
            &format!("mqa_b3_mixed_{layout_name}"),
            Bf16PagedBatchDecodeSpec::new(3, 7, 8, 1, 128, 16, layout)?,
            PageTableInput {
                indptr: &[0, 1, 3, 6],
                indices: &[4, 6, 1, 5, 0, 3],
                last_page_len: &[16, 7, 16],
            },
            0x2001,
        )?;
        run_case(
            &mut queue,
            &provider,
            &format!("gqa4_b4_reuse_{layout_name}"),
            Bf16PagedBatchDecodeSpec::new(4, 8, 16, 4, 128, 16, layout)?,
            PageTableInput {
                indptr: &[0, 1, 3, 5, 8],
                indices: &[7, 2, 6, 5, 1, 7, 0, 4],
                last_page_len: &[3, 16, 1, 9],
            },
            0x4001,
        )?;
    }

    let preflight_spec = Bf16PagedBatchDecodeSpec::new(2, 2, 4, 2, 128, 16, PagedKvLayout::Nhd)?;
    let preflight_plan = provider.plan_bf16_paged_batch(preflight_spec)?;
    run_short_page_indptr_case(&mut queue, &preflight_plan)?;
    run_invalid_page_guard_case(&mut queue, &provider)?;
    run_invalid_page_graph_rejection_case(&context, &provider)?;
    println!(
        "gate=paged_batch_decode_h20 suite=all status=pass output_max_abs_limit={:.9e} \
         lse_max_abs_limit={:.9e}",
        OUTPUT_MAX_ABS_LIMIT, LSE_MAX_ABS_LIMIT
    );
    Ok(())
}
