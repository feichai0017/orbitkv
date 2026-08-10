use crate::benchmark::BenchmarkRecord;
use crate::comparison::{compare_bf16, digest_bf16};
use crate::fixture::{deterministic_bf16, page_refcounts};
use cuda_core::{CudaContext, DeviceBuffer, sys};
use half::bf16;
use loom_infer::{Bf16RopePagedKvAppendTokensSpec, rope_paged_kv_append_tokens_bf16_reference};
use loom_infer_cuda::graph::GraphQueue;
use loom_infer_cuda::rope::{
    Bf16PagedKvAppendTokensMapArgs, Bf16RopePagedKvAppendMappedArgs, RopeProvider,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::sync::Arc;

const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");
const MEASUREMENT: &str = "fixed_address_cuda_graph_single_replay_event";
const FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_rope_paged_append_tokens_v2";
const CASE: &str = "bf16_rope_paged_append_t6_b3_qh16_kh4_d128_p16";
const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                Err(format!("{name} must be nonzero").into())
            } else {
                Ok(parsed)
            }
        }
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn digest_i32(values: &[i32]) -> u64 {
    values.iter().fold(FNV_OFFSET_BASIS, |digest, &value| {
        (digest ^ u64::from(value as u32)).wrapping_mul(FNV_PRIME)
    })
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let provider_commit = env::var("LOOM_SOURCE_COMMIT")?;
    if provider_commit.len() != 40 || !provider_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("LOOM_SOURCE_COMMIT must be a full 40-character Git commit SHA".into());
    }
    let run_label = env::var("LOOM_BENCH_RUN_LABEL").unwrap_or_else(|_| "unlabeled".to_string());
    let warmup_launches = env_usize("LOOM_BENCH_WARMUP", 200)?;
    let samples = env_usize("LOOM_BENCH_SAMPLES", 100)?;
    if env_usize("LOOM_BENCH_LAUNCHES", 1)? != 1 {
        return Err("Graph benchmark requires LOOM_BENCH_LAUNCHES=1".into());
    }

    let context = CudaContext::new(0)?;
    let provider = RopeProvider::load(&context)?;
    let spec = Bf16RopePagedKvAppendTokensSpec::new(6, 3, 8, 16, 4, 128, 16)?;
    let batch_indices_host = [2_i32, 0, 1, 0, 2, 1];
    let positions_host = [5_i32, 17, 20, 16, 4, 19];
    let page_indptr_host = [0_i32, 2, 4, 5];
    let page_indices_host = [7_i32, 3, 2, 6, 5];
    let last_page_len_host = [2_i32, 5, 6];
    let page_refcounts_host = page_refcounts(spec.max_num_pages(), &page_indices_host);
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
        &page_refcounts_host,
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
    let page_refcounts = Arc::new(DeviceBuffer::from_host(
        &upload_stream,
        &page_refcounts_host,
    )?);
    let query_output =
        DeviceBuffer::from_host(&upload_stream, &vec![bf16::NAN; spec.query_output_numel()])?;
    let key_pages = DeviceBuffer::from_host(&upload_stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(&upload_stream, &value_pages_host)?;
    let workspace = DeviceBuffer::<i32>::zeroed(&upload_stream, plan.workspace_required_numel())?;

    let graph_queue = GraphQueue::new(&context, 3)?;
    let mut bindings = graph_queue.bindings(13)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let batch_indices_handle = bindings.bind_read(batch_indices)?;
    let positions_handle = bindings.bind_read(positions)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let page_refcounts_handle = bindings.bind_read(page_refcounts)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let captured = graph_queue.capture(bindings, |scope| {
        let append_map = plan.enqueue_map_into(
            scope,
            Bf16PagedKvAppendTokensMapArgs::new(
                batch_indices_handle,
                positions_handle,
                page_indptr_handle,
                page_indices_handle,
                last_page_len_handle,
                page_refcounts_handle,
                key_pages_handle.write(),
                value_pages_handle.write(),
                workspace_handle,
            ),
        )?;
        plan.enqueue_mapped_into(
            scope,
            Bf16RopePagedKvAppendMappedArgs::new(
                query_handle,
                key_handle,
                value_handle,
                append_map,
                query_output_handle.write(),
                key_pages_handle.write(),
                value_pages_handle.write(),
            ),
        )
    })?;
    if captured.commands() != 3 {
        return Err("explicit append Graph benchmark captured wrong command count".into());
    }
    let mut exec = captured.instantiate()?;
    for _ in 0..warmup_launches {
        exec.launch()?.wait()?;
    }

    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let mut samples_us = Vec::with_capacity(samples);
    for _ in 0..samples {
        samples_us.push(f64::from(exec.measure_launch_ms(&start, &end)?) * 1000.0);
    }

    let mut bindings = exec.into_bindings()?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);
    let query_comparison = compare_bf16(
        &query_output.to_host_vec(&upload_stream)?,
        &expected_query,
        "Graph benchmark explicit append query",
    )?;
    let key_comparison = compare_bf16(
        &key_pages.to_host_vec(&upload_stream)?,
        &expected_key_pages,
        "Graph benchmark explicit append key pages",
    )?;
    let value_comparison = compare_bf16(
        &value_pages.to_host_vec(&upload_stream)?,
        &expected_value_pages,
        "Graph benchmark explicit append value pages",
    )?;
    if query_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || key_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || value_comparison.max_abs != 0.0
    {
        return Err("explicit append Graph benchmark exceeded correctness limits".into());
    }

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &provider_commit,
        run_label: &run_label,
        measurement: MEASUREMENT,
        operator: "rope_paged_kv_append_tokens",
        case: CASE,
        dtype: "bf16",
        layout: "NHD_D128_neox_split_half_page16",
        execution: json!({
            "algorithm": "validate_compact_then_fused_append_explicit_tokens",
            "graph": "fixed_address_private_stream",
            "graph_nodes": 3,
            "kernels": 2,
            "status_readbacks": 1,
            "completion_event_inside_timed_interval": true,
            "batch_indices": [2, 0, 1, 0, 2, 1],
            "positions": [5, 17, 20, 16, 4, 19],
            "physical_slots": [[5, 5], [3, 1], [6, 4], [3, 0], [5, 4], [6, 3]],
            "correctness": {
                "reference": "loom-infer CPU reference",
                "query_max_abs": query_comparison.max_abs,
                "query_bit_mismatches": query_comparison.bit_mismatches,
                "query_digest": format!("{:016x}", query_comparison.digest),
                "key_pages_max_abs": key_comparison.max_abs,
                "key_pages_bit_mismatches": key_comparison.bit_mismatches,
                "key_pages_digest": format!("{:016x}", key_comparison.digest),
                "value_pages_max_abs": value_comparison.max_abs,
                "value_pages_bit_mismatches": value_comparison.bit_mismatches,
                "value_pages_digest": format!("{:016x}", value_comparison.digest)
            }
        }),
        kernels_per_call: 2,
        shape: json!({
            "tokens": 6,
            "batch_size": 3,
            "max_num_pages": 8,
            "query_heads": 16,
            "key_heads": 4,
            "head_dim": 128,
            "page_size": 16
        }),
        fixture_id: FIXTURE_ID,
        fixture_digests: json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key": format!("{:016x}", digest_bf16(&key_host)),
            "value": format!("{:016x}", digest_bf16(&value_host)),
            "key_pages_initial": format!("{:016x}", digest_bf16(&key_pages_host)),
            "value_pages_initial": format!("{:016x}", digest_bf16(&value_pages_host)),
            "batch_indices": format!("{:016x}", digest_i32(&batch_indices_host)),
            "positions": format!("{:016x}", digest_i32(&positions_host)),
            "page_indptr": format!("{:016x}", digest_i32(&page_indptr_host)),
            "page_indices": format!("{:016x}", digest_i32(&page_indices_host)),
            "last_page_len": format!("{:016x}", digest_i32(&last_page_len_host)),
            "page_refcounts": format!("{:016x}", digest_i32(&page_refcounts_host))
        }),
        warmup_launches,
        launches_per_sample: 1,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}
