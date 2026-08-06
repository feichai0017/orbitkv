use crate::comparison::digest_bf16;
use crate::fixture::deterministic_bf16;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
use half::bf16;
use loom_infer::{Bf16SingleDecodeSpec, Bf16SingleDecodeSplitKSpec};
use loom_infer_cuda::attention::{
    AttentionProvider, Bf16SingleDecodeArgs, Bf16SingleDecodeSplitKArgs,
};
use loom_infer_cuda::command::{CheckedBindings, CommandQueue};
use serde_json::json;
use std::env;
use std::error::Error;
use std::sync::Arc;

const MEASUREMENT: &str = "eager_stream_batch_cuda_event";
const FIXTURE_ID: &str = "xorshift64_mod2001_bf16_v1";

#[derive(Clone, Copy)]
struct BenchConfig {
    warmup_calls: usize,
    calls_per_sample: usize,
    samples: usize,
}

#[derive(Clone, Copy)]
struct DecodeCase {
    name: &'static str,
    kv_len: usize,
    query_heads: usize,
    kv_heads: usize,
    partitions: &'static [usize],
}

impl BenchConfig {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            warmup_calls: env_usize("LOOM_BENCH_WARMUP", 50)?,
            calls_per_sample: env_usize("LOOM_BENCH_LAUNCHES", 100)?,
            samples: env_usize("LOOM_BENCH_SAMPLES", 20)?,
        })
    }
}

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

fn benchmark_scopes<F>(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    queue: &mut CommandQueue,
    mut bindings: CheckedBindings,
    config: BenchConfig,
    mut enqueue: F,
) -> Result<Vec<f64>, Box<dyn Error>>
where
    F: FnMut(&mut loom_infer_cuda::command::CommandScope<'_>) -> Result<(), Box<dyn Error>>,
{
    for _ in 0..config.warmup_calls {
        let mut scope = queue.begin(bindings)?;
        enqueue(&mut scope)?;
        bindings = scope.finish().wait()?;
    }

    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let mut samples_us = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        start.record(stream)?;
        let mut scope = queue.begin(bindings)?;
        for _ in 0..config.calls_per_sample {
            enqueue(&mut scope)?;
        }
        end.record(stream)?;
        bindings = scope.finish().wait()?;
        samples_us
            .push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / config.calls_per_sample as f64);
    }
    Ok(samples_us)
}

fn write_record(
    case: DecodeCase,
    variant: &str,
    partitions: usize,
    workspace_numel: usize,
    config: BenchConfig,
    fixture_digests: serde_json::Value,
    samples_us: Vec<f64>,
) -> Result<(), serde_json::Error> {
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema_version": 1,
            "source_commit": env::var("LOOM_SOURCE_COMMIT").unwrap_or_else(|_| "unrecorded".into()),
            "source_state": env::var("LOOM_SOURCE_STATE").unwrap_or_else(|_| "working_tree".into()),
            "measurement": MEASUREMENT,
            "operator": "single_decode",
            "case": case.name,
            "variant": variant,
            "partitions": partitions,
            "kernels_per_call": if variant == "direct" { 1 } else { 2 },
            "workspace_numel": workspace_numel,
            "dtype": "bf16",
            "layout": "NHD_D128",
            "shape": {
                "kv_len": case.kv_len,
                "query_heads": case.query_heads,
                "kv_heads": case.kv_heads,
                "head_dim": 128
            },
            "fixture_id": FIXTURE_ID,
            "fixture_digests": fixture_digests,
            "warmup_calls": config.warmup_calls,
            "calls_per_sample": config.calls_per_sample,
            "samples_us": samples_us,
        }))?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn benchmark_direct(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &AttentionProvider,
    case: DecodeCase,
    config: BenchConfig,
    query_host: &[bf16],
    key_host: &[bf16],
    value_host: &[bf16],
    fixture_digests: &serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16SingleDecodeSpec::new(case.kv_len, case.query_heads, case.kv_heads, 128)?;
    let plan = provider.plan_bf16(spec)?;
    let query = Arc::new(DeviceBuffer::from_host(stream, query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(stream, key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(stream, value_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(stream, spec.lse_numel())?;
    let mut queue = CommandQueue::new(stream.clone(), config.calls_per_sample)?;
    let mut bindings = queue.bindings(5)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let samples_us = benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
        plan.enqueue_into(
            scope,
            Bf16SingleDecodeArgs::new(
                query_handle,
                key_handle,
                value_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )?;
        Ok(())
    })?;
    write_record(
        case,
        "direct",
        1,
        0,
        config,
        fixture_digests.clone(),
        samples_us,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn benchmark_split_k(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &AttentionProvider,
    case: DecodeCase,
    partitions: usize,
    config: BenchConfig,
    query_host: &[bf16],
    key_host: &[bf16],
    value_host: &[bf16],
    fixture_digests: &serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let decode = Bf16SingleDecodeSpec::new(case.kv_len, case.query_heads, case.kv_heads, 128)?;
    let spec = Bf16SingleDecodeSplitKSpec::new(decode, partitions)?;
    let plan = provider.plan_bf16_split_k(spec)?;
    let query = Arc::new(DeviceBuffer::from_host(stream, query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(stream, key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(stream, value_host)?);
    let workspace = DeviceBuffer::<f32>::zeroed(stream, spec.workspace_numel())?;
    let output = DeviceBuffer::<bf16>::zeroed(stream, decode.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(stream, decode.lse_numel())?;
    let command_capacity = config
        .calls_per_sample
        .checked_mul(2)
        .ok_or("split-K command capacity overflow")?;
    let mut queue = CommandQueue::new(stream.clone(), command_capacity)?;
    let mut bindings = queue.bindings(6)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let samples_us = benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
        plan.enqueue_into(
            scope,
            Bf16SingleDecodeSplitKArgs::new(
                query_handle,
                key_handle,
                value_handle,
                workspace_handle,
                output_handle.write(),
                lse_handle.write(),
            ),
        )?;
        Ok(())
    })?;
    write_record(
        case,
        "split_k",
        partitions,
        spec.workspace_numel(),
        config,
        fixture_digests.clone(),
        samples_us,
    )?;
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let config = BenchConfig::from_env()?;
    let context = CudaContext::new(0)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let provider = AttentionProvider::load(&context)?;

    for case in [
        DecodeCase {
            name: "bf16_mqa_l33_qh8_kvh1_d128",
            kv_len: 33,
            query_heads: 8,
            kv_heads: 1,
            partitions: &[2, 3, 4, 6, 8, 12, 16],
        },
        DecodeCase {
            name: "bf16_gqa_l127_qh16_kvh4_d128",
            kv_len: 127,
            query_heads: 16,
            kv_heads: 4,
            partitions: &[6, 8, 10, 12, 16, 20],
        },
        DecodeCase {
            name: "bf16_gqa_l4096_qh32_kvh4_d128",
            kv_len: 4096,
            query_heads: 32,
            kv_heads: 4,
            partitions: &[48, 64, 80, 96, 128, 160],
        },
    ] {
        let spec = Bf16SingleDecodeSpec::new(case.kv_len, case.query_heads, case.kv_heads, 128)?;
        let query_host = deterministic_bf16(spec.query_numel(), 0x5155_4552);
        let key_host = deterministic_bf16(spec.kv_numel(), 0x4b45_5900);
        let value_host = deterministic_bf16(spec.kv_numel(), 0x5641_4c55);
        let fixture_digests = json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key": format!("{:016x}", digest_bf16(&key_host)),
            "value": format!("{:016x}", digest_bf16(&value_host))
        });
        benchmark_direct(
            &context,
            &stream,
            &provider,
            case,
            config,
            &query_host,
            &key_host,
            &value_host,
            &fixture_digests,
        )?;
        for &partitions in case.partitions {
            benchmark_split_k(
                &context,
                &stream,
                &provider,
                case,
                partitions,
                config,
                &query_host,
                &key_host,
                &value_host,
                &fixture_digests,
            )?;
        }
    }
    Ok(())
}
