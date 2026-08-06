use crate::benchmark::BenchmarkRecord;
use crate::comparison::{compare_bf16, compare_f32, digest_bf16};
use crate::fixture::deterministic_bf16;
use cuda_core::{CudaContext, DeviceBuffer, sys};
use half::bf16;
use loom_infer::{Bf16RaggedPrefillSpec, ragged_prefill_bf16_reference};
use loom_infer_cuda::attention::{Bf16RaggedPrefillArgs, PrefillProvider};
use loom_infer_cuda::graph::GraphQueue;
use serde_json::json;
use std::env;
use std::error::Error;
use std::sync::Arc;

const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");
const MEASUREMENT: &str = "fixed_address_cuda_graph_single_replay_event";
const FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_ragged_indptr_v1";
const CASE: &str = "bf16_ragged_gqa4_b2_q32_64_kv256_1024_qh16_kvh4_d128";
const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const LSE_MAX_ABS_LIMIT: f32 = 0.01;
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
    let provider = PrefillProvider::load(&context)?;
    let spec = Bf16RaggedPrefillSpec::new(2, 96, 1280, 16, 4, 128)?;
    let qo_indptr_host = [0_i32, 32, 96];
    let kv_indptr_host = [0_i32, 256, 1280];
    spec.validate_metadata(&qo_indptr_host, &kv_indptr_host)?;
    let plan = provider.plan_bf16_ragged(spec)?;
    if plan.workspace_required_numel() == 0 {
        return Err("Graph benchmark requires the tiled two-kernel plan".into());
    }

    let query_host = deterministic_bf16(spec.query_numel(), 0x4001);
    let key_host = deterministic_bf16(spec.kv_numel(), 0x4001 ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_numel(), 0x4001 ^ 0x5641_4c55_4500);
    let mut expected_output = vec![bf16::NAN; spec.output_numel()];
    let mut expected_lse = vec![f32::NAN; spec.lse_numel()];
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

    let upload_stream = context.new_stream()?;
    let query = Arc::new(DeviceBuffer::from_host(&upload_stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(&upload_stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(&upload_stream, &value_host)?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(&upload_stream, &qo_indptr_host)?);
    let kv_indptr = Arc::new(DeviceBuffer::from_host(&upload_stream, &kv_indptr_host)?);
    let output = DeviceBuffer::from_host(&upload_stream, &vec![bf16::NAN; spec.output_numel()])?;
    let lse = DeviceBuffer::from_host(&upload_stream, &vec![f32::NAN; spec.lse_numel()])?;
    let workspace = DeviceBuffer::<f32>::zeroed(&upload_stream, plan.workspace_required_numel())?;

    let graph_queue = GraphQueue::new(&context, 2)?;
    let mut bindings = graph_queue.bindings(8)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let qo_indptr_handle = bindings.bind_read(qo_indptr)?;
    let kv_indptr_handle = bindings.bind_read(kv_indptr)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let captured = graph_queue.capture(bindings, |scope| {
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
            .with_workspace(workspace_handle),
        )
    })?;
    if captured.commands() != 2 {
        return Err("Graph benchmark captured the wrong command count".into());
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
    let output = bindings.take_read_write(output_handle)?;
    let lse = bindings.take_read_write(lse_handle)?;
    drop(bindings);
    let output_comparison = compare_bf16(
        &output.to_host_vec(&upload_stream)?,
        &expected_output,
        "Graph benchmark BF16 output",
    )?;
    let lse_comparison = compare_f32(
        &lse.to_host_vec(&upload_stream)?,
        &expected_lse,
        "Graph benchmark F32 LSE",
    )?;
    if output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT {
        return Err(format!(
            "Graph output max abs {:.9e} exceeds {:.9e}",
            output_comparison.max_abs, OUTPUT_MAX_ABS_LIMIT
        )
        .into());
    }
    if lse_comparison.max_abs > LSE_MAX_ABS_LIMIT {
        return Err(format!(
            "Graph LSE max abs {:.9e} exceeds {:.9e}",
            lse_comparison.max_abs, LSE_MAX_ABS_LIMIT
        )
        .into());
    }

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &provider_commit,
        run_label: &run_label,
        measurement: MEASUREMENT,
        operator: "ragged_prefill",
        case: CASE,
        dtype: "bf16",
        layout: "NHD_D128_ragged",
        execution: json!({
            "algorithm": "tiled_gqa4_mma_qk_softmax_pv",
            "causal": "bottom_right",
            "graph": "fixed_address_private_stream",
            "graph_nodes": 2,
            "completion_event_inside_timed_interval": true,
            "correctness": {
                "output_max_abs": output_comparison.max_abs,
                "output_bit_mismatches": output_comparison.bit_mismatches,
                "output_digest": format!("{:016x}", output_comparison.digest),
                "lse_max_abs": lse_comparison.max_abs,
                "lse_bit_mismatches": lse_comparison.bit_mismatches,
                "lse_digest": format!("{:016x}", lse_comparison.digest)
            }
        }),
        kernels_per_call: 2,
        shape: json!({
            "batch_size": 2,
            "nnz_qo": 96,
            "nnz_kv": 1280,
            "request_qo_lens": [32, 64],
            "request_kv_lens": [256, 1024],
            "query_heads": 16,
            "kv_heads": 4,
            "head_dim": 128
        }),
        fixture_id: FIXTURE_ID,
        fixture_digests: json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key": format!("{:016x}", digest_bf16(&key_host)),
            "value": format!("{:016x}", digest_bf16(&value_host)),
            "qo_indptr": format!("{:016x}", digest_i32(&qo_indptr_host)),
            "kv_indptr": format!("{:016x}", digest_i32(&kv_indptr_host))
        }),
        warmup_launches,
        launches_per_sample: 1,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}
