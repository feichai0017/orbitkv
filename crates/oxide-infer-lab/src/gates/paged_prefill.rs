use super::{VALID_GRAPH_REPLAYS, valid_graph_commands};
use crate::comparison::{compare_bf16, compare_f32};
use crate::fixture::deterministic_bf16;
use crate::reporting::GateCase;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use oxide_infer::{Bf16PagedPrefillSpec, ContractError, paged_prefill_bf16_reference};
use oxide_infer_cuda::attention::{
    Bf16PagedPrefillAlgorithm, Bf16PagedPrefillArgs, Bf16PagedPrefillPlan,
    PagedPrefillEnqueueError, PrefillProvider,
};
use oxide_infer_cuda::command::{CommandCompletionError, CommandError, CommandQueue};
use oxide_infer_cuda::graph::{GraphBindingsError, GraphError, GraphQueue};
use std::error::Error;
use std::sync::Arc;

const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const LSE_MAX_ABS_LIMIT: f32 = 0.01;
const PAGED_GRAPH_COMMANDS_PER_STAGE: usize = 4;
const STATUS_POISON: i32 = i32::MIN;
const LONG_GQA4_QO_INDPTR: [i32; 3] = [0, 32, 96];
const LONG_GQA4_PAGE_INDPTR: [i32; 3] = [0, 16, 80];
const LONG_GQA4_PAGE_INDICES: [i32; 80] = [
    15, 3, 27, 9, 31, 1, 35, 5, 39, 7, 43, 11, 47, 13, 51, 17, 19, 21, 23, 25, 29, 33, 37, 41, 45,
    49, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 0,
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
    52, 54, 56, 58, 60, 62,
];
const LONG_GQA4_LAST_PAGE_LEN: [i32; 2] = [16, 16];

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
    requested_algorithm: Bf16PagedPrefillAlgorithm,
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
    let plan = provider.plan_bf16_paged(spec, requested_algorithm)?;
    if plan.algorithm() != requested_algorithm {
        return Err(format!(
            "{name} planner changed the requested algorithm: expected {requested_algorithm:?}, got {:?}",
            plan.algorithm()
        )
        .into());
    }
    let algorithm = match plan.algorithm() {
        Bf16PagedPrefillAlgorithm::Direct => "direct_one_warp_per_query_row_head",
        Bf16PagedPrefillAlgorithm::TokenParallel8 => "token_parallel_8warp_block_local_merge",
        Bf16PagedPrefillAlgorithm::TokenParallel16 => "token_parallel_16warp_block_local_merge",
        Bf16PagedPrefillAlgorithm::TiledGqa4 => "tiled_gqa4_paged_mma_qk_softmax_pv",
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
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&stream, plan.metadata_status_required_numel())?;
    let output = DeviceBuffer::from_host(&stream, &vec![bf16::NAN; spec.output_numel()])?;
    let lse = DeviceBuffer::from_host(&stream, &vec![f32::NAN; spec.lse_numel()])?;
    let workspace =
        DeviceBuffer::<f32>::zeroed(&stream, usize::max(plan.workspace_required_numel(), 1))?;
    let mut bindings = queue.bindings(11)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let mut scope = queue.begin(bindings)?;
    let mut args = Bf16PagedPrefillArgs::new(
        query_handle,
        key_handle,
        value_handle,
        qo_handle,
        page_indptr_handle,
        page_indices_handle,
        last_page_len_handle,
        metadata_status_handle,
        output_handle.write(),
        lse_handle.write(),
    );
    if plan.workspace_required_numel() != 0 {
        args = args.with_workspace(workspace_handle);
    }
    plan.enqueue_into(&mut scope, args)?;
    let completion = scope.finish();
    let expected_commands = if plan.workspace_required_numel() == 0 {
        3
    } else {
        4
    };
    if completion.submitted() != expected_commands {
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
         algorithm={} commands={} validator_kernels=1 attention_kernels={} \
         status_readbacks=1 stream=non_default \
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
        expected_commands,
        expected_commands - 2,
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
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&stream, plan.metadata_status_required_numel())?;
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(10)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
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
                metadata_status_handle,
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

fn run_invalid_query_guard(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedPrefillSpec::new(3, 5, 3, 1, 1, 128, 16)?;
    let expected_error = ContractError::RaggedQueryLongerThanKv {
        request: 1,
        query_len: 2,
        kv_len: 1,
    };
    let stream = queue.stream().clone();
    let plan = provider.plan_bf16_paged(spec, Bf16PagedPrefillAlgorithm::TokenParallel16)?;
    if plan.algorithm() != Bf16PagedPrefillAlgorithm::TokenParallel16 {
        return Err("invalid-query guard did not select token-parallel paged prefill".into());
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
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1, 3, 5])?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1, 2, 3])?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1, 2])?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &[1_i32, 1, 1])?);
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&stream, plan.metadata_status_required_numel())?;
    let output_sentinel = vec![bf16::NAN; spec.output_numel()];
    let lse_sentinel = vec![f32::NAN; spec.lse_numel()];
    let output = DeviceBuffer::from_host(&stream, &output_sentinel)?;
    let lse = DeviceBuffer::from_host(&stream, &lse_sentinel)?;
    let mut bindings = queue.bindings(10)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
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
            metadata_status_handle,
            output_handle.write(),
            lse_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 3 {
        return Err("invalid-query rejection covered the wrong command count".into());
    }
    bindings = match completion.wait() {
        Err(CommandCompletionError::DeviceRejected(rejection)) => {
            if rejection.error() != expected_error {
                return Err(format!(
                    "paged prefill returned the wrong device rejection: expected \
                     {expected_error}, got {}",
                    rejection.error()
                )
                .into());
            }
            rejection.into_parts().1
        }
        Err(error) => return Err(format!("paged prefill returned the wrong error: {error}").into()),
        Ok(_) => return Err("invalid paged-prefill query metadata was not rejected".into()),
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
        return Err("rejected paged prefill changed output sentinels".into());
    }
    println!(
        "{} query_longer_than_kv=device_rejected bindings_recovered=true \
         queue_reusable=true sentinels_unchanged=true commands=3",
        GateCase::new("paged_prefill_h20", "invalid_query_guard")
    );
    Ok(())
}

fn run_duplicate_binding_case(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedPrefillSpec::new(1, 1, 1, 1, 1, 128, 16)?;
    let plan = provider.plan_bf16_paged(spec, Bf16PagedPrefillAlgorithm::Direct)?;
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
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&stream, plan.metadata_status_required_numel())?;
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(9)?;
    let query_and_key_handle = bindings.bind_read(query_and_key)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
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
                metadata_status_handle,
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

fn run_missing_workspace_case(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedPrefillSpec::new(1, 1, 16, 4, 1, 128, 16)?;
    let plan = provider.plan_bf16_paged(spec, Bf16PagedPrefillAlgorithm::TiledGqa4)?;
    if plan.workspace_required_numel() == 0 {
        return Err("missing-workspace gate did not select tiled GQA4".into());
    }
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
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 1])?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &[0_i32, 16])?);
    let page_indices = Arc::new(DeviceBuffer::from_host(
        &stream,
        &(0_i32..16).collect::<Vec<_>>(),
    )?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &[16_i32])?);
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&stream, plan.metadata_status_required_numel())?;
    let output = DeviceBuffer::<bf16>::zeroed(&stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(&stream, spec.lse_numel())?;
    let mut bindings = queue.bindings(10)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
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
                metadata_status_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
        .expect_err("paged tiled GQA4 must require an explicit workspace");
    if !matches!(error, PagedPrefillEnqueueError::MissingWorkspace) {
        return Err(format!("missing workspace returned the wrong error: {error}").into());
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err("missing workspace reached CUDA submission".into());
    }
    drop(completion.wait()?);
    println!(
        "{} workspace=required before_submission=true",
        GateCase::new("paged_prefill_h20", "missing_workspace")
    );
    Ok(())
}

fn run_graph_case(
    queue: &mut CommandQueue,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedPrefillSpec::new(2, 96, 96, 16, 4, 128, 16)?;
    let qo_indptr_host = LONG_GQA4_QO_INDPTR;
    let page_indptr_host = LONG_GQA4_PAGE_INDPTR;
    let page_indices_host = LONG_GQA4_PAGE_INDICES;
    let last_page_len_host = LONG_GQA4_LAST_PAGE_LEN;
    spec.validate_metadata(
        &qo_indptr_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
    )?;
    let plan = provider.plan_bf16_paged(spec, Bf16PagedPrefillAlgorithm::TiledGqa4)?;
    let stream = queue.stream().clone();
    let query_host = deterministic_bf16(spec.query_numel(), 0x8001);
    let key_host = deterministic_bf16(spec.kv_pages_numel(), 0x8001 ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_pages_numel(), 0x8001 ^ 0x5641_4c55_4500);
    let poison_query_host = vec![bf16::ZERO; spec.query_numel()];
    let poison_key_host = vec![bf16::ZERO; spec.kv_pages_numel()];
    let poison_value_host = vec![bf16::ZERO; spec.kv_pages_numel()];
    let mut poison_expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut poison_expected_lse = vec![f32::NAN; spec.lse_numel()];
    let mut expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut expected_lse = vec![f32::NAN; spec.lse_numel()];
    paged_prefill_bf16_reference(
        &poison_query_host,
        &poison_key_host,
        &poison_value_host,
        &qo_indptr_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &mut poison_expected_output,
        &mut poison_expected_lse,
        spec,
    )?;
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
    if poison_expected_output
        .iter()
        .zip(&expected_output)
        .all(|(poison, expected)| poison.to_bits() == expected.to_bits())
        || poison_expected_lse
            .iter()
            .zip(&expected_lse)
            .all(|(poison, expected)| poison.to_bits() == expected.to_bits())
    {
        return Err("paged-prefill graph poison does not distinguish every checked output".into());
    }

    let poison_query = Arc::new(DeviceBuffer::from_host(&stream, &poison_query_host)?);
    let poison_key_pages = Arc::new(DeviceBuffer::from_host(&stream, &poison_key_host)?);
    let poison_value_pages = Arc::new(DeviceBuffer::from_host(&stream, &poison_value_host)?);
    let query = Arc::new(DeviceBuffer::from_host(&stream, &query_host)?);
    let key_pages = Arc::new(DeviceBuffer::from_host(&stream, &key_host)?);
    let value_pages = Arc::new(DeviceBuffer::from_host(&stream, &value_host)?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&stream, &qo_indptr_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(&stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&stream, &last_page_len_host)?);
    let initial_status = vec![STATUS_POISON; plan.metadata_status_required_numel()];
    let poison_observer_status = DeviceBuffer::from_host(&stream, &initial_status)?;
    let target_poison_status = DeviceBuffer::from_host(&stream, &initial_status)?;
    let real_status = DeviceBuffer::from_host(&stream, &initial_status)?;
    let initial_output = vec![bf16::NAN; spec.output_numel()];
    let initial_lse = vec![f32::NAN; spec.lse_numel()];
    let poison_observer_output = DeviceBuffer::from_host(&stream, &initial_output)?;
    let poison_observer_lse = DeviceBuffer::from_host(&stream, &initial_lse)?;
    let output = DeviceBuffer::from_host(&stream, &initial_output)?;
    let lse = DeviceBuffer::from_host(&stream, &initial_lse)?;
    let workspace = DeviceBuffer::<f32>::zeroed(&stream, plan.workspace_required_numel())?;

    let graph_commands = valid_graph_commands(PAGED_GRAPH_COMMANDS_PER_STAGE);
    let graph_queue = GraphQueue::new(stream.context(), graph_commands)?;
    let mut bindings = graph_queue.bindings(18)?;
    let poison_query_handle = bindings.bind_read(Arc::clone(&poison_query))?;
    let poison_key_handle = bindings.bind_read(Arc::clone(&poison_key_pages))?;
    let poison_value_handle = bindings.bind_read(Arc::clone(&poison_value_pages))?;
    let query_handle = bindings.bind_read(Arc::clone(&query))?;
    let key_handle = bindings.bind_read(Arc::clone(&key_pages))?;
    let value_handle = bindings.bind_read(Arc::clone(&value_pages))?;
    let qo_handle = bindings.bind_read(Arc::clone(&qo_indptr))?;
    let page_indptr_handle = bindings.bind_read(Arc::clone(&page_indptr))?;
    let page_indices_handle = bindings.bind_read(Arc::clone(&page_indices))?;
    let last_page_len_handle = bindings.bind_read(Arc::clone(&last_page_len))?;
    let poison_observer_status_handle = bindings.bind_read_write(poison_observer_status)?;
    let target_poison_status_handle = bindings.bind_read_write(target_poison_status)?;
    let real_status_handle = bindings.bind_read_write(real_status)?;
    let poison_observer_output_handle = bindings.bind_read_write(poison_observer_output)?;
    let poison_observer_lse_handle = bindings.bind_read_write(poison_observer_lse)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let captured = graph_queue.capture(bindings, |scope| {
        plan.enqueue_into(
            scope,
            Bf16PagedPrefillArgs::new(
                poison_query_handle,
                poison_key_handle,
                poison_value_handle,
                qo_handle,
                page_indptr_handle,
                page_indices_handle,
                last_page_len_handle,
                poison_observer_status_handle,
                poison_observer_output_handle.write(),
                poison_observer_lse_handle.write(),
            )
            .with_workspace(workspace_handle),
        )?;
        plan.enqueue_into(
            scope,
            Bf16PagedPrefillArgs::new(
                poison_query_handle,
                poison_key_handle,
                poison_value_handle,
                qo_handle,
                page_indptr_handle,
                page_indices_handle,
                last_page_len_handle,
                target_poison_status_handle,
                output_handle.write(),
                lse_handle.write(),
            )
            .with_workspace(workspace_handle),
        )?;
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
                real_status_handle,
                output_handle.write(),
                lse_handle.write(),
            )
            .with_workspace(workspace_handle),
        )
    })?;
    if captured.commands() != graph_commands {
        return Err("paged-prefill graph captured the wrong command count".into());
    }

    drop(plan);
    drop(poison_query);
    drop(poison_key_pages);
    drop(poison_value_pages);
    drop(query);
    drop(key_pages);
    drop(value_pages);
    drop(qo_indptr);
    drop(page_indptr);
    drop(page_indices);
    drop(last_page_len);

    let mut exec = captured.instantiate()?;
    for expected_launch in 1..=VALID_GRAPH_REPLAYS {
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
    if exec.launches() != VALID_GRAPH_REPLAYS || exec.commands() != graph_commands {
        return Err("paged-prefill graph accounting changed across replay".into());
    }

    let mut bindings = exec.into_bindings()?;
    let poison_observer_status = bindings.take_read_write(poison_observer_status_handle)?;
    let target_poison_status = bindings.take_read_write(target_poison_status_handle)?;
    let real_status = bindings.take_read_write(real_status_handle)?;
    let poison_observer_output = bindings.take_read_write(poison_observer_output_handle)?;
    let poison_observer_lse = bindings.take_read_write(poison_observer_lse_handle)?;
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);
    let actual_output = output.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    for (label, status) in [
        (
            "independent poison",
            poison_observer_status.to_host_vec(&stream)?,
        ),
        ("target poison", target_poison_status.to_host_vec(&stream)?),
        ("real", real_status.to_host_vec(&stream)?),
    ] {
        if status.iter().any(|&word| word != 0) {
            return Err(format!("paged-prefill graph {label} status was not rewritten").into());
        }
    }
    let poison_output_comparison = compare_bf16(
        &poison_observer_output.to_host_vec(&stream)?,
        &poison_expected_output,
        "graph paged prefill poison BF16",
    )?;
    let poison_lse_comparison = compare_f32(
        &poison_observer_lse.to_host_vec(&stream)?,
        &poison_expected_lse,
        "graph paged prefill poison F32 LSE",
    )?;
    let output_comparison =
        compare_bf16(&actual_output, &expected_output, "graph paged prefill BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "graph paged prefill F32 LSE")?;
    if poison_output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
    {
        return Err(format!(
            "paged-prefill graph output max abs {:.9e} exceeds {:.9e}",
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
            "paged-prefill graph LSE max abs {:.9e} exceeds {:.9e}",
            poison_lse_comparison.max_abs.max(lse_comparison.max_abs),
            LSE_MAX_ABS_LIMIT
        )
        .into());
    }

    println!(
        "{} batch_size=2 nnz_qo=96 query_heads=16 kv_heads=4 page_size=16 \
         algorithm=tiled_gqa4_paged_mma_qk_softmax_pv commands={} commands_per_stage=4 \
         validator_kernels_per_stage=1 attention_kernels_per_stage=2 \
         status_readbacks_per_stage=1 replays={} \
         replay_stages=independent_poison_then_target_poison_then_real \
         independent_poison_observable=true output_lse_poisoned_each_replay=true \
         independent_status_packets=true status_packets_rewritten=true \
         initial_outputs=poisoned \
         fixed_bindings=true cross_stream=false external_owners_dropped_before_replay=true \
         completion_queries=2 completion_waits=1 completion_drops=1 \
         poison_output_max_abs={:.9e} poison_output_digest={:016x} \
         poison_lse_max_abs={:.9e} poison_lse_digest={:016x} \
         output_max_abs={:.9e} output_bit_mismatches={} output_digest={:016x} \
         lse_max_abs={:.9e} lse_bit_mismatches={} lse_digest={:016x}",
        GateCase::new("paged_prefill_h20", "gqa4_tiled_graph"),
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

fn run_invalid_page_graph_rejection_case(
    context: &Arc<CudaContext>,
    provider: &PrefillProvider,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedPrefillSpec::new(1, 1, 66, 1, 1, 128, 16)?;
    let expected_error = ContractError::PageIndexOutOfRange {
        position: 33,
        index: 66,
        max_num_pages: 66,
    };
    let upload_stream = context.new_stream()?;
    let plan = provider.plan_bf16_paged(spec, Bf16PagedPrefillAlgorithm::Direct)?;
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
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&upload_stream, &[0_i32, 1])?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(&upload_stream, &[0_i32, 66])?);
    let mut page_indices_host = (0_i32..66).collect::<Vec<_>>();
    page_indices_host[33] = 66;
    page_indices_host[65] = -1;
    let page_indices = Arc::new(DeviceBuffer::from_host(&upload_stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(&upload_stream, &[1_i32])?);
    let metadata_status =
        DeviceBuffer::<i32>::zeroed(&upload_stream, plan.metadata_status_required_numel())?;
    let output_sentinel = vec![bf16::NAN; spec.output_numel()];
    let lse_sentinel = vec![f32::NAN; spec.lse_numel()];
    let output = DeviceBuffer::from_host(&upload_stream, &output_sentinel)?;
    let lse = DeviceBuffer::from_host(&upload_stream, &lse_sentinel)?;

    let graph_queue = GraphQueue::new(context, 3)?;
    let mut bindings = graph_queue.bindings(10)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let metadata_status_handle = bindings.bind_read_write(metadata_status)?;
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
                metadata_status_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )
    })?;
    if captured.commands() != 3 {
        return Err("rejected paged-prefill graph captured the wrong command count".into());
    }

    let mut exec = captured.instantiate()?;
    for expected_launch in 1..=2 {
        let completion = exec.launch()?;
        if completion.launch_index() != expected_launch {
            return Err("rejected paged-prefill graph reported the wrong replay index".into());
        }
        match completion.wait() {
            Err(GraphError::DeviceRejected(error)) if error == expected_error => {}
            Err(error) => {
                return Err(format!(
                    "rejected paged-prefill graph returned the wrong error: {error}"
                )
                .into());
            }
            Ok(()) => return Err("invalid paged-prefill graph metadata was not rejected".into()),
        }
    }
    if exec.launches() != 2 || exec.commands() != 3 {
        return Err("rejected paged-prefill graph accounting changed across replay".into());
    }

    let mut bindings = match exec.into_bindings() {
        Err(GraphBindingsError::DeviceRejected(rejection)) => {
            if rejection.error() != expected_error {
                return Err(format!(
                    "rejected paged-prefill graph returned the wrong bindings error: expected \
                     {expected_error}, got {}",
                    rejection.error()
                )
                .into());
            }
            rejection.into_parts().1
        }
        Err(error) => {
            return Err(format!(
                "rejected paged-prefill graph could not recover bindings: {error}"
            )
            .into());
        }
        Ok(_) => {
            return Err("invalid paged-prefill graph returned bindings without rejection".into());
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
        return Err("rejected paged-prefill graph changed output sentinels".into());
    }

    println!(
        "{} replays=2 fixed_bindings=true device_rejected=true graph_reusable=true \
         bindings_recovered=true sentinels_unchanged=true commands=3",
        GateCase::new("paged_prefill_h20", "invalid_page_graph")
    );
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let provider = PrefillProvider::load(&context)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let mut queue = CommandQueue::new(stream, 4, 1)?;

    run_case(
        &mut queue,
        &provider,
        "mha_equal_lengths",
        Bf16PagedPrefillSpec::new(1, 4, 2, 8, 8, 128, 16)?,
        Bf16PagedPrefillAlgorithm::Direct,
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
        "mha_large_pool_short_table_explicit_direct",
        Bf16PagedPrefillSpec::new(1, 4, 96, 8, 8, 128, 16)?,
        Bf16PagedPrefillAlgorithm::Direct,
        MetadataInput {
            qo_indptr: &[0, 4],
            page_indptr: &[0, 1],
            page_indices: &[95],
            last_page_len: &[4],
        },
        0x1002,
    )?;
    run_case(
        &mut queue,
        &provider,
        "mqa_append_mixed",
        Bf16PagedPrefillSpec::new(3, 6, 7, 8, 1, 128, 16)?,
        Bf16PagedPrefillAlgorithm::Direct,
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
        Bf16PagedPrefillAlgorithm::Direct,
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
        Bf16PagedPrefillAlgorithm::TokenParallel16,
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
        "gqa4_tiled",
        Bf16PagedPrefillSpec::new(2, 96, 96, 16, 4, 128, 16)?,
        Bf16PagedPrefillAlgorithm::TiledGqa4,
        MetadataInput {
            qo_indptr: &LONG_GQA4_QO_INDPTR,
            page_indptr: &LONG_GQA4_PAGE_INDPTR,
            page_indices: &LONG_GQA4_PAGE_INDICES,
            last_page_len: &LONG_GQA4_LAST_PAGE_LEN,
        },
        0x8001,
    )?;

    let preflight_spec = Bf16PagedPrefillSpec::new(2, 2, 2, 4, 2, 128, 16)?;
    let preflight_plan =
        provider.plan_bf16_paged(preflight_spec, Bf16PagedPrefillAlgorithm::Direct)?;
    run_short_metadata_case(&mut queue, &preflight_plan)?;
    run_invalid_query_guard(&mut queue, &provider)?;
    run_duplicate_binding_case(&mut queue, &provider)?;
    run_missing_workspace_case(&mut queue, &provider)?;
    run_graph_case(&mut queue, &provider)?;
    run_invalid_page_graph_rejection_case(&context, &provider)?;
    println!(
        "gate=paged_prefill_h20 suite=all status=pass output_max_abs_limit={:.9e} \
         lse_max_abs_limit={:.9e}",
        OUTPUT_MAX_ABS_LIMIT, LSE_MAX_ABS_LIMIT
    );
    Ok(())
}
