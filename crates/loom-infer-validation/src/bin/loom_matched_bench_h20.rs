use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
use half::bf16;
use loom_infer::{
    Bf16GemmSpec, Bf16SingleDecodeSpec, Bf16SingleDecodeSplitKSpec, DType, RmsNormSpec,
};
use loom_infer_cuda::attention::{
    AttentionProvider, Bf16SingleDecodeArgs, Bf16SingleDecodePlan, Bf16SingleDecodeSplitKArgs,
};
use loom_infer_cuda::command::{CheckedBindings, CommandQueue, Read, ReadWrite};
use loom_infer_cuda::gemm::{Bf16GemmArgs, Bf16GemmPlan, CublasLtProvider};
use loom_infer_cuda::rms_norm::{RmsNormArgs, RmsNormBf16Plan, RmsNormProvider};
use loom_infer_validation::benchmark::BenchmarkRecord;
use loom_infer_validation::comparison::digest_bf16;
use serde_json::json;
use std::env;
use std::error::Error;
use std::sync::Arc;

const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");
const MEASUREMENT: &str = "eager_stream_batch_cuda_event";
const FIXTURE_ID: &str = "xorshift64_mod2001_bf16_v1";

struct RunIdentity {
    provider_commit: String,
    run_label: String,
}

impl RunIdentity {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let provider_commit = env::var("LOOM_SOURCE_COMMIT")?;
        if provider_commit.len() != 40
            || !provider_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("LOOM_SOURCE_COMMIT must be a full 40-character Git commit SHA".into());
        }
        Ok(Self {
            provider_commit,
            run_label: env::var("LOOM_BENCH_RUN_LABEL").unwrap_or_else(|_| "unlabeled".to_string()),
        })
    }
}

#[derive(Clone, Copy)]
struct BenchConfig {
    warmup_launches: usize,
    launches_per_sample: usize,
    samples: usize,
}

#[derive(Clone, Copy)]
struct DecodeCase {
    name: &'static str,
    kv_len: usize,
    query_heads: usize,
    kv_heads: usize,
    partitions: usize,
}

impl BenchConfig {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            warmup_launches: env_usize("LOOM_BENCH_WARMUP", 20)?,
            launches_per_sample: env_usize("LOOM_BENCH_LAUNCHES", 100)?,
            samples: env_usize("LOOM_BENCH_SAMPLES", 30)?,
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
) -> Result<(CheckedBindings, Vec<f64>), Box<dyn Error>>
where
    F: FnMut(&mut loom_infer_cuda::command::CommandScope<'_>) -> Result<(), Box<dyn Error>>,
{
    for _ in 0..config.warmup_launches {
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
        for _ in 0..config.launches_per_sample {
            enqueue(&mut scope)?;
        }
        end.record(stream)?;
        bindings = scope.finish().wait()?;
        samples_us
            .push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / config.launches_per_sample as f64);
    }
    Ok((bindings, samples_us))
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

fn benchmark_rms_norm(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &RmsNormProvider,
    rows: usize,
    hidden_size: usize,
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let spec = RmsNormSpec::new(rows, hidden_size, 1.0e-5, DType::Bf16)?;
    let plan: RmsNormBf16Plan = provider.plan_bf16(spec)?;
    let input_host = deterministic_bf16(spec.numel(), 0x524d_534e);
    let weight_host = deterministic_bf16(spec.hidden_size(), 0x5745_4947);
    let input = Arc::new(DeviceBuffer::from_host(stream, &input_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(stream, &weight_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.numel())?;
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample)?;
    let mut bindings = queue.bindings(3)?;
    let input_handle: Read<bf16> = bindings.bind_read(input)?;
    let weight_handle: Read<bf16> = bindings.bind_read(weight)?;
    let output_handle: ReadWrite<bf16> = bindings.bind_read_write(output)?;

    let (_bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            plan.enqueue_into(
                scope,
                RmsNormArgs::new(input_handle, weight_handle, output_handle.write()),
            )?;
            Ok(())
        })?;

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "rms_norm",
        case: &format!("bf16_r{rows}_h{hidden_size}"),
        dtype: "bf16",
        layout: "contiguous_rows_hidden",
        execution: json!({"algorithm": "packed_or_scalar_by_alignment"}),
        kernels_per_call: 1,
        shape: json!({"rows": rows, "hidden_size": hidden_size}),
        fixture_id: FIXTURE_ID,
        fixture_digests: json!({
            "input": format!("{:016x}", digest_bf16(&input_host)),
            "weight": format!("{:016x}", digest_bf16(&weight_host))
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}

fn benchmark_gemm(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &CublasLtProvider,
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16GemmSpec::new(1, 4096, 4096)?;
    let plan: Bf16GemmPlan = provider.plan_bf16(spec)?;
    let activation_host = deterministic_bf16(spec.a_numel(), 0x4143_5449);
    let weight_host = deterministic_bf16(spec.weight_numel(), 0x4745_4d4d);
    let activation = Arc::new(DeviceBuffer::from_host(stream, &activation_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(stream, &weight_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(stream, plan.workspace_required_bytes())?;
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample)?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(activation)?;
    let weight_handle = bindings.bind_read(weight)?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let (_bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            plan.enqueue_into(
                scope,
                Bf16GemmArgs::new(
                    activation_handle,
                    weight_handle,
                    output_handle.write(),
                    workspace_handle.write(),
                ),
            )?;
            Ok(())
        })?;

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "gemm",
        case: "bf16_m1_n4096_k4096_cublaslt",
        dtype: "bf16",
        layout: "A_row_major_W_row_major_transposed",
        execution: json!({"algorithm": "cublaslt", "tactic": 0}),
        kernels_per_call: 1,
        shape: json!({"m": 1, "n": 4096, "k": 4096}),
        fixture_id: FIXTURE_ID,
        fixture_digests: json!({
            "activation": format!("{:016x}", digest_bf16(&activation_host)),
            "weight_storage": format!("{:016x}", digest_bf16(&weight_host))
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}

fn benchmark_decode_case(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &AttentionProvider,
    case: DecodeCase,
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16SingleDecodeSpec::new(case.kv_len, case.query_heads, case.kv_heads, 128)?;
    let query_host = deterministic_bf16(spec.query_numel(), 0x5155_4552);
    let key_host = deterministic_bf16(spec.kv_numel(), 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_numel(), 0x5641_4c55);
    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(stream, &value_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(stream, spec.lse_numel())?;
    let (samples_us, execution, kernels_per_call) = if case.partitions == 1 {
        let plan: Bf16SingleDecodePlan = provider.plan_bf16(spec)?;
        let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample)?;
        let mut bindings = queue.bindings(5)?;
        let query_handle = bindings.bind_read(query)?;
        let key_handle = bindings.bind_read(key)?;
        let value_handle = bindings.bind_read(value)?;
        let output_handle = bindings.bind_read_write(output)?;
        let lse_handle = bindings.bind_read_write(lse)?;
        let (_bindings, samples_us) =
            benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
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
        (samples_us, json!({"algorithm": "direct"}), 1)
    } else {
        let split_spec = Bf16SingleDecodeSplitKSpec::new(spec, case.partitions)?;
        let plan = provider.plan_bf16_split_k(split_spec)?;
        let workspace = DeviceBuffer::<f32>::zeroed(stream, split_spec.workspace_numel())?;
        let command_capacity = config
            .launches_per_sample
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
        let (_bindings, samples_us) =
            benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
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
        (
            samples_us,
            json!({
                "algorithm": "split_k",
                "partitions": case.partitions,
                "workspace_numel": split_spec.workspace_numel(),
                "workspace_bytes": split_spec.workspace_bytes()
            }),
            2,
        )
    };

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "single_decode",
        case: case.name,
        dtype: "bf16",
        layout: "NHD_D128",
        execution,
        kernels_per_call,
        shape: json!({
            "kv_len": case.kv_len,
            "query_heads": case.query_heads,
            "kv_heads": case.kv_heads,
            "head_dim": 128
        }),
        fixture_id: FIXTURE_ID,
        fixture_digests: json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key": format!("{:016x}", digest_bf16(&key_host)),
            "value": format!("{:016x}", digest_bf16(&value_host))
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = BenchConfig::from_env()?;
    let identity = RunIdentity::from_env()?;
    let context = CudaContext::new(0)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let rms_provider = RmsNormProvider::load(&context)?;
    let gemm_provider = CublasLtProvider::load(&context)?;
    let attention_provider = AttentionProvider::load(&context)?;

    for (rows, hidden_size) in [(1, 4096), (8, 4096), (64, 4096), (16, 8192)] {
        benchmark_rms_norm(
            &context,
            &stream,
            &rms_provider,
            rows,
            hidden_size,
            config,
            &identity,
        )?;
    }
    benchmark_gemm(&context, &stream, &gemm_provider, config, &identity)?;
    for case in [
        DecodeCase {
            name: "bf16_mha_l1_qh8_kvh8_d128",
            kv_len: 1,
            query_heads: 8,
            kv_heads: 8,
            partitions: 1,
        },
        DecodeCase {
            name: "bf16_mqa_l33_qh8_kvh1_d128",
            kv_len: 33,
            query_heads: 8,
            kv_heads: 1,
            partitions: 12,
        },
        DecodeCase {
            name: "bf16_gqa_l127_qh16_kvh4_d128",
            kv_len: 127,
            query_heads: 16,
            kv_heads: 4,
            partitions: 16,
        },
        DecodeCase {
            name: "bf16_gqa_l4096_qh32_kvh4_d128",
            kv_len: 4096,
            query_heads: 32,
            kv_heads: 4,
            partitions: 64,
        },
    ] {
        benchmark_decode_case(
            &context,
            &stream,
            &attention_provider,
            case,
            config,
            &identity,
        )?;
    }
    Ok(())
}
