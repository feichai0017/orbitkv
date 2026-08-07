use crate::benchmark::BenchmarkRecord;
use crate::comparison::{compare_bf16, digest_bf16};
use crate::fixture::deterministic_bf16;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
use half::bf16;
use loom_infer::{
    Bf16GemmSpec, Bf16PagedBatchDecodeSpec, Bf16RaggedPrefillSpec, Bf16RopePagedKvAppendSpec,
    Bf16RopePagedKvAppendTokensSpec, Bf16RopePosIdsSpec, Bf16SingleDecodeSpec,
    Bf16SingleDecodeSplitKSpec, DType, RmsNormSpec, rope_paged_kv_append_bf16_reference,
    rope_paged_kv_append_tokens_bf16_reference, rope_pos_ids_bf16_reference,
};
use loom_infer_cuda::attention::{
    Bf16PagedBatchDecodeAlgorithm, Bf16PagedBatchDecodeArgs, Bf16RaggedPrefillAlgorithm,
    Bf16RaggedPrefillArgs, Bf16SingleDecodeArgs, Bf16SingleDecodePlan, Bf16SingleDecodeSplitKArgs,
    DecodeProvider, PrefillProvider,
};
use loom_infer_cuda::command::{CheckedBindings, CommandQueue, Read, ReadWrite};
use loom_infer_cuda::gemm::{Bf16GemmArgs, Bf16GemmPlan, CublasLtProvider};
use loom_infer_cuda::rms_norm::{RmsNormArgs, RmsNormBf16Plan, RmsNormProvider};
use loom_infer_cuda::rope::{
    Bf16RopePagedKvAppendArgs, Bf16RopePagedKvAppendTokensArgs, Bf16RopePosIdsArgs, RopeProvider,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::sync::Arc;

const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");
const MEASUREMENT: &str = "eager_stream_batch_cuda_event";
const FIXTURE_ID: &str = "xorshift64_mod2001_bf16_v1";
const PAGED_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_page_table_v1";
const RAGGED_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_ragged_indptr_v1";
const ROPE_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_rope_pos_ids_v1";
const ROPE_APPEND_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_rope_paged_append_v1";
const ROPE_APPEND_TOKENS_FIXTURE_ID: &str =
    "xorshift64_mod2001_bf16_i32_rope_paged_append_tokens_v1";
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

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

#[derive(Clone, Copy)]
struct PagedDecodeCase {
    name: &'static str,
    batch_size: usize,
    max_num_pages: usize,
    query_heads: usize,
    kv_heads: usize,
    page_indptr: &'static [i32],
    page_indices: &'static [i32],
    last_page_len: &'static [i32],
    salt: u64,
}

#[derive(Clone, Copy)]
struct RaggedPrefillCase {
    name: &'static str,
    batch_size: usize,
    query_heads: usize,
    kv_heads: usize,
    qo_indptr: &'static [i32],
    kv_indptr: &'static [i32],
    salt: u64,
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

fn digest_i32(values: &[i32]) -> u64 {
    values.iter().fold(FNV_OFFSET_BASIS, |digest, &value| {
        (digest ^ u64::from(value as u32)).wrapping_mul(FNV_PRIME)
    })
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
    provider: &DecodeProvider,
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

fn benchmark_paged_decode_case(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &DecodeProvider,
    case: PagedDecodeCase,
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16PagedBatchDecodeSpec::new(
        case.batch_size,
        case.max_num_pages,
        case.query_heads,
        case.kv_heads,
        128,
        16,
    )?;
    let table =
        spec.validate_page_table(case.page_indptr, case.page_indices, case.last_page_len)?;
    let plan = provider.plan_bf16_paged_batch(spec)?;
    let algorithm = match plan.algorithm() {
        Bf16PagedBatchDecodeAlgorithm::Direct => "direct_one_warp_per_request_head",
        Bf16PagedBatchDecodeAlgorithm::TokenParallel8 => "token_parallel_8warp_block_local_merge",
    };
    let query_host = deterministic_bf16(spec.query_numel(), case.salt);
    let key_host = deterministic_bf16(spec.kv_pages_numel(), case.salt ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_pages_numel(), case.salt ^ 0x5641_4c55_4500);
    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key_pages = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let value_pages = Arc::new(DeviceBuffer::from_host(stream, &value_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(stream, case.page_indptr)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(stream, case.page_indices)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(stream, case.last_page_len)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(stream, spec.lse_numel())?;
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample)?;
    let mut bindings = queue.bindings(8)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let indptr_handle = bindings.bind_read(page_indptr)?;
    let indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;

    let (_bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            plan.enqueue_into(
                scope,
                Bf16PagedBatchDecodeArgs::new(
                    query_handle,
                    key_handle,
                    value_handle,
                    indptr_handle,
                    indices_handle,
                    last_page_len_handle,
                    output_handle.write(),
                    lse_handle.write(),
                ),
            )?;
            Ok(())
        })?;
    let request_kv_lens = (0..spec.batch_size())
        .map(|request| {
            table
                .request_kv_len(request)
                .expect("validated request has a KV length")
        })
        .collect::<Vec<_>>();

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "paged_batch_decode",
        case: case.name,
        dtype: "bf16",
        layout: "NHD_D128_page16",
        execution: json!({
            "algorithm": algorithm,
            "page_table_location": "device"
        }),
        kernels_per_call: 1,
        shape: json!({
            "batch_size": spec.batch_size(),
            "max_num_pages": spec.max_num_pages(),
            "referenced_pages": case.page_indices.len(),
            "request_kv_lens": request_kv_lens,
            "query_heads": spec.num_query_heads(),
            "kv_heads": spec.num_kv_heads(),
            "head_dim": spec.head_dim(),
            "page_size": spec.page_size()
        }),
        fixture_id: PAGED_FIXTURE_ID,
        fixture_digests: json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key_pages": format!("{:016x}", digest_bf16(&key_host)),
            "value_pages": format!("{:016x}", digest_bf16(&value_host)),
            "page_indptr": format!("{:016x}", digest_i32(case.page_indptr)),
            "page_indices": format!("{:016x}", digest_i32(case.page_indices)),
            "last_page_len": format!("{:016x}", digest_i32(case.last_page_len))
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}

fn benchmark_ragged_prefill_case(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &PrefillProvider,
    case: RaggedPrefillCase,
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let nnz_qo = usize::try_from(*case.qo_indptr.last().ok_or("empty qo_indptr")?)?;
    let nnz_kv = usize::try_from(*case.kv_indptr.last().ok_or("empty kv_indptr")?)?;
    let spec = Bf16RaggedPrefillSpec::new(
        case.batch_size,
        nnz_qo,
        nnz_kv,
        case.query_heads,
        case.kv_heads,
        128,
    )?;
    let metadata = spec.validate_metadata(case.qo_indptr, case.kv_indptr)?;
    let plan = provider.plan_bf16_ragged(spec)?;
    let algorithm = match plan.algorithm() {
        Bf16RaggedPrefillAlgorithm::Direct => "direct_one_warp_per_query_row_head",
        Bf16RaggedPrefillAlgorithm::TokenParallel8 => "token_parallel_8warp_block_local_merge",
        Bf16RaggedPrefillAlgorithm::TokenParallel16 => "token_parallel_16warp_block_local_merge",
        Bf16RaggedPrefillAlgorithm::TiledGqa4 => "tiled_gqa4_mma_qk_softmax_pv",
    };
    let query_host = deterministic_bf16(spec.query_numel(), case.salt);
    let key_host = deterministic_bf16(spec.kv_numel(), case.salt ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_numel(), case.salt ^ 0x5641_4c55_4500);
    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(stream, &value_host)?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(stream, case.qo_indptr)?);
    let kv_indptr = Arc::new(DeviceBuffer::from_host(stream, case.kv_indptr)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(stream, spec.lse_numel())?;
    let workspace =
        DeviceBuffer::<f32>::zeroed(stream, usize::max(plan.workspace_required_numel(), 1))?;
    let kernels_per_call = if plan.workspace_required_numel() == 0 {
        1
    } else {
        2
    };
    let command_capacity = config
        .launches_per_sample
        .checked_mul(kernels_per_call)
        .ok_or("ragged benchmark command capacity overflow")?;
    let mut queue = CommandQueue::new(stream.clone(), command_capacity)?;
    let mut bindings = queue.bindings(8)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let qo_indptr_handle = bindings.bind_read(qo_indptr)?;
    let kv_indptr_handle = bindings.bind_read(kv_indptr)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let (_bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            let mut args = Bf16RaggedPrefillArgs::new(
                query_handle,
                key_handle,
                value_handle,
                qo_indptr_handle,
                kv_indptr_handle,
                output_handle.write(),
                lse_handle.write(),
            );
            if plan.workspace_required_numel() != 0 {
                args = args.with_workspace(workspace_handle);
            }
            plan.enqueue_into(scope, args)?;
            Ok(())
        })?;
    let mut request_qo_lens = Vec::with_capacity(spec.batch_size());
    let mut request_kv_lens = Vec::with_capacity(spec.batch_size());
    for request in 0..spec.batch_size() {
        let ((qo_start, qo_end), (kv_start, kv_end)) = metadata
            .request_row_ranges(request)
            .expect("validated request has row ranges");
        request_qo_lens.push(qo_end - qo_start);
        request_kv_lens.push(kv_end - kv_start);
    }

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "ragged_prefill",
        case: case.name,
        dtype: "bf16",
        layout: "NHD_D128_ragged",
        execution: json!({
            "algorithm": algorithm,
            "causal": "bottom_right",
            "indptr_location": "device"
        }),
        kernels_per_call,
        shape: json!({
            "batch_size": spec.batch_size(),
            "nnz_qo": spec.nnz_qo(),
            "nnz_kv": spec.nnz_kv(),
            "request_qo_lens": request_qo_lens,
            "request_kv_lens": request_kv_lens,
            "query_heads": spec.num_query_heads(),
            "kv_heads": spec.num_kv_heads(),
            "head_dim": spec.head_dim()
        }),
        fixture_id: RAGGED_FIXTURE_ID,
        fixture_digests: json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key": format!("{:016x}", digest_bf16(&key_host)),
            "value": format!("{:016x}", digest_bf16(&value_host)),
            "qo_indptr": format!("{:016x}", digest_i32(case.qo_indptr)),
            "kv_indptr": format!("{:016x}", digest_i32(case.kv_indptr))
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}

fn benchmark_rope(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &RopeProvider,
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16RopePosIdsSpec::new(96, 16, 4, 128, 128, 1.0, 10_000.0)?;
    let query_host = deterministic_bf16(spec.query_numel(), 0x524f_5045);
    let key_host = deterministic_bf16(spec.key_numel(), 0x4b45_5900);
    let position_ids_host = (224_i32..256).chain(960_i32..1024).collect::<Vec<_>>();
    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let position_ids = Arc::new(DeviceBuffer::from_host(stream, &position_ids_host)?);
    let query_output = DeviceBuffer::<bf16>::zeroed(stream, spec.query_numel())?;
    let key_output = DeviceBuffer::<bf16>::zeroed(stream, spec.key_numel())?;
    let plan = provider.plan_bf16_pos_ids(spec)?;
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample)?;
    let mut bindings = queue.bindings(5)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let position_ids_handle = bindings.bind_read(position_ids)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_output_handle = bindings.bind_read_write(key_output)?;

    let (mut bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            plan.enqueue_into(
                scope,
                Bf16RopePosIdsArgs::new(
                    query_handle,
                    key_handle,
                    position_ids_handle,
                    query_output_handle.write(),
                    key_output_handle.write(),
                ),
            )?;
            Ok(())
        })?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_output = bindings.take_read_write(key_output_handle)?;
    drop(bindings);
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
    let query_comparison = compare_bf16(
        &query_output.to_host_vec(stream)?,
        &expected_query,
        "benchmark RoPE query",
    )?;
    let key_comparison = compare_bf16(
        &key_output.to_host_vec(stream)?,
        &expected_key,
        "benchmark RoPE key",
    )?;
    if query_comparison.max_abs > 0.015_625 || key_comparison.max_abs > 0.015_625 {
        return Err("benchmark RoPE output exceeded the BF16 correctness limit".into());
    }

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "rope",
        case: "bf16_rope_pos_ids_t96_qh16_kh4_d128_neox",
        dtype: "bf16",
        layout: "NHD_D128_neox_split_half",
        execution: json!({
            "algorithm": "one_64thread_cta_per_token_head",
            "position_mode": "explicit_i32",
            "rotary_dim": 128,
            "rope_scale": 1.0,
            "rope_theta": 10000.0,
            "correctness": {
                "reference": "loom-infer CPU reference",
                "query_max_abs": query_comparison.max_abs,
                "query_bit_mismatches": query_comparison.bit_mismatches,
                "query_digest": format!("{:016x}", query_comparison.digest),
                "query_reference_digest": format!("{:016x}", digest_bf16(&expected_query)),
                "key_max_abs": key_comparison.max_abs,
                "key_bit_mismatches": key_comparison.bit_mismatches,
                "key_digest": format!("{:016x}", key_comparison.digest),
                "key_reference_digest": format!("{:016x}", digest_bf16(&expected_key))
            }
        }),
        kernels_per_call: 1,
        shape: json!({
            "tokens": 96,
            "query_heads": 16,
            "key_heads": 4,
            "head_dim": 128,
            "rotary_dim": 128,
            "position_ranges": [[224, 256], [960, 1024]]
        }),
        fixture_id: ROPE_FIXTURE_ID,
        fixture_digests: json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key": format!("{:016x}", digest_bf16(&key_host)),
            "position_ids": format!("{:016x}", digest_i32(&position_ids_host))
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}

fn benchmark_rope_paged_append(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &RopeProvider,
    config: BenchConfig,
    identity: &RunIdentity,
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

    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(stream, &value_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(stream, &last_page_len_host)?);
    let query_output = DeviceBuffer::<bf16>::zeroed(stream, spec.query_output_numel())?;
    let key_pages = DeviceBuffer::from_host(stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(stream, &value_pages_host)?;
    let plan = provider.plan_bf16_paged_append(spec)?;
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample)?;
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

    let (mut bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            plan.enqueue_into(
                scope,
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
            Ok(())
        })?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);
    let query_comparison = compare_bf16(
        &query_output.to_host_vec(stream)?,
        &expected_query,
        "benchmark fused query",
    )?;
    let key_comparison = compare_bf16(
        &key_pages.to_host_vec(stream)?,
        &expected_key_pages,
        "benchmark fused key pages",
    )?;
    let value_comparison = compare_bf16(
        &value_pages.to_host_vec(stream)?,
        &expected_value_pages,
        "benchmark fused value pages",
    )?;
    if query_comparison.max_abs > 0.015_625
        || key_comparison.max_abs > 0.015_625
        || value_comparison.max_abs != 0.0
    {
        return Err("benchmark fused RoPE append exceeded correctness limits".into());
    }

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "rope_paged_kv_append",
        case: "bf16_rope_paged_append_b4_qh16_kh4_d128_p16",
        dtype: "bf16",
        layout: "NHD_D128_neox_split_half_page16",
        execution: json!({
            "algorithm": "fused_one_kernel",
            "kernels": 1,
            "positions": [2, 31, 16, 40],
            "physical_slots": [[7, 2], [6, 15], [1, 0], [4, 8]],
            "correctness": {
                "reference": "loom-infer CPU reference",
                "query_max_abs": query_comparison.max_abs,
                "query_bit_mismatches": query_comparison.bit_mismatches,
                "query_digest": format!("{:016x}", query_comparison.digest),
                "query_reference_digest": format!("{:016x}", digest_bf16(&expected_query)),
                "key_pages_max_abs": key_comparison.max_abs,
                "key_pages_bit_mismatches": key_comparison.bit_mismatches,
                "key_pages_digest": format!("{:016x}", key_comparison.digest),
                "key_pages_reference_digest": format!("{:016x}", digest_bf16(&expected_key_pages)),
                "value_pages_max_abs": value_comparison.max_abs,
                "value_pages_bit_mismatches": value_comparison.bit_mismatches,
                "value_pages_digest": format!("{:016x}", value_comparison.digest),
                "value_pages_reference_digest": format!("{:016x}", digest_bf16(&expected_value_pages))
            }
        }),
        kernels_per_call: 1,
        shape: json!({
            "batch_size": 4,
            "max_num_pages": 8,
            "query_heads": 16,
            "key_heads": 4,
            "head_dim": 128,
            "page_size": 16
        }),
        fixture_id: ROPE_APPEND_FIXTURE_ID,
        fixture_digests: json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key": format!("{:016x}", digest_bf16(&key_host)),
            "value": format!("{:016x}", digest_bf16(&value_host)),
            "key_pages_initial": format!("{:016x}", digest_bf16(&key_pages_host)),
            "value_pages_initial": format!("{:016x}", digest_bf16(&value_pages_host)),
            "page_indptr": format!("{:016x}", digest_i32(&page_indptr_host)),
            "page_indices": format!("{:016x}", digest_i32(&page_indices_host)),
            "last_page_len": format!("{:016x}", digest_i32(&last_page_len_host))
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}

fn benchmark_rope_paged_append_tokens(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &RopeProvider,
    config: BenchConfig,
    identity: &RunIdentity,
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

    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(stream, &value_host)?);
    let batch_indices = Arc::new(DeviceBuffer::from_host(stream, &batch_indices_host)?);
    let positions = Arc::new(DeviceBuffer::from_host(stream, &positions_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(stream, &last_page_len_host)?);
    let query_output = DeviceBuffer::<bf16>::zeroed(stream, spec.query_output_numel())?;
    let key_pages = DeviceBuffer::from_host(stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(stream, &value_pages_host)?;
    let plan = provider.plan_bf16_paged_append_tokens(spec)?;
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample)?;
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

    let (mut bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
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
            )?;
            Ok(())
        })?;
    let query_output = bindings.take_read_write(query_output_handle)?;
    let key_pages = bindings.take_read_write(key_pages_handle)?;
    let value_pages = bindings.take_read_write(value_pages_handle)?;
    drop(bindings);
    let query_comparison = compare_bf16(
        &query_output.to_host_vec(stream)?,
        &expected_query,
        "benchmark explicit fused query",
    )?;
    let key_comparison = compare_bf16(
        &key_pages.to_host_vec(stream)?,
        &expected_key_pages,
        "benchmark explicit fused key pages",
    )?;
    let value_comparison = compare_bf16(
        &value_pages.to_host_vec(stream)?,
        &expected_value_pages,
        "benchmark explicit fused value pages",
    )?;
    if query_comparison.max_abs > 0.015_625
        || key_comparison.max_abs > 0.015_625
        || value_comparison.max_abs != 0.0
    {
        return Err("benchmark explicit fused RoPE append exceeded correctness limits".into());
    }

    BenchmarkRecord {
        schema_version: 1,
        provider: "loom-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "rope_paged_kv_append_tokens",
        case: "bf16_rope_paged_append_t6_b3_qh16_kh4_d128_p16",
        dtype: "bf16",
        layout: "NHD_D128_neox_split_half_page16",
        execution: json!({
            "algorithm": "fused_one_kernel_explicit_tokens",
            "kernels": 1,
            "batch_indices": [2, 0, 1, 0, 2, 1],
            "positions": [5, 17, 20, 16, 4, 19],
            "physical_slots": [[3, 5], [3, 1], [6, 4], [3, 0], [3, 4], [6, 3]],
            "correctness": {
                "reference": "loom-infer CPU reference",
                "query_max_abs": query_comparison.max_abs,
                "query_bit_mismatches": query_comparison.bit_mismatches,
                "query_digest": format!("{:016x}", query_comparison.digest),
                "query_reference_digest": format!("{:016x}", digest_bf16(&expected_query)),
                "key_pages_max_abs": key_comparison.max_abs,
                "key_pages_bit_mismatches": key_comparison.bit_mismatches,
                "key_pages_digest": format!("{:016x}", key_comparison.digest),
                "key_pages_reference_digest": format!("{:016x}", digest_bf16(&expected_key_pages)),
                "value_pages_max_abs": value_comparison.max_abs,
                "value_pages_bit_mismatches": value_comparison.bit_mismatches,
                "value_pages_digest": format!("{:016x}", value_comparison.digest),
                "value_pages_reference_digest": format!("{:016x}", digest_bf16(&expected_value_pages))
            }
        }),
        kernels_per_call: 1,
        shape: json!({
            "tokens": 6,
            "batch_size": 3,
            "max_num_pages": 8,
            "query_heads": 16,
            "key_heads": 4,
            "head_dim": 128,
            "page_size": 16
        }),
        fixture_id: ROPE_APPEND_TOKENS_FIXTURE_ID,
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
            "last_page_len": format!("{:016x}", digest_i32(&last_page_len_host))
        }),
        warmup_launches: config.warmup_launches,
        launches_per_sample: config.launches_per_sample,
        samples_us,
    }
    .write_json_line()?;
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let config = BenchConfig::from_env()?;
    let identity = RunIdentity::from_env()?;
    let requested = env::var("LOOM_BENCH_OPERATORS").unwrap_or_else(|_| {
        "rms_norm,gemm,single_decode,paged_batch_decode,ragged_prefill,rope,rope_paged_kv_append,rope_paged_kv_append_tokens".to_string()
    });
    let requested = requested.split(',').collect::<Vec<_>>();
    let context = CudaContext::new(0)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let rms_provider = RmsNormProvider::load(&context)?;
    let gemm_provider = CublasLtProvider::load(&context)?;
    let decode_provider = DecodeProvider::load(&context)?;
    let prefill_provider = PrefillProvider::load(&context)?;
    let rope_provider = RopeProvider::load(&context)?;

    if requested.contains(&"rms_norm") {
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
    }
    if requested.contains(&"gemm") {
        benchmark_gemm(&context, &stream, &gemm_provider, config, &identity)?;
    }
    if requested.contains(&"single_decode") {
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
            benchmark_decode_case(&context, &stream, &decode_provider, case, config, &identity)?;
        }
    }
    if requested.contains(&"paged_batch_decode") {
        for case in [
            PagedDecodeCase {
                name: "bf16_paged_mha_b1_l1_qh8_kvh8_d128_p16",
                batch_size: 1,
                max_num_pages: 2,
                query_heads: 8,
                kv_heads: 8,
                page_indptr: &[0, 1],
                page_indices: &[1],
                last_page_len: &[1],
                salt: 0x1001,
            },
            PagedDecodeCase {
                name: "bf16_paged_mqa_b3_l16_23_48_qh8_kvh1_d128_p16",
                batch_size: 3,
                max_num_pages: 7,
                query_heads: 8,
                kv_heads: 1,
                page_indptr: &[0, 1, 3, 6],
                page_indices: &[4, 6, 1, 5, 0, 3],
                last_page_len: &[16, 7, 16],
                salt: 0x2001,
            },
            PagedDecodeCase {
                name: "bf16_paged_gqa4_b4_l3_32_17_41_qh16_kvh4_d128_p16",
                batch_size: 4,
                max_num_pages: 8,
                query_heads: 16,
                kv_heads: 4,
                page_indptr: &[0, 1, 3, 5, 8],
                page_indices: &[7, 2, 6, 5, 1, 7, 0, 4],
                last_page_len: &[3, 16, 1, 9],
                salt: 0x4001,
            },
        ] {
            benchmark_paged_decode_case(
                &context,
                &stream,
                &decode_provider,
                case,
                config,
                &identity,
            )?;
        }
    }
    if requested.contains(&"ragged_prefill") {
        for case in [
            RaggedPrefillCase {
                name: "bf16_ragged_mha_b1_q16_kv16_qh8_kvh8_d128",
                batch_size: 1,
                query_heads: 8,
                kv_heads: 8,
                qo_indptr: &[0, 16],
                kv_indptr: &[0, 16],
                salt: 0x1001,
            },
            RaggedPrefillCase {
                name: "bf16_ragged_mqa_b3_q1_4_16_kv128_256_512_qh8_kvh1_d128",
                batch_size: 3,
                query_heads: 8,
                kv_heads: 1,
                qo_indptr: &[0, 1, 5, 21],
                kv_indptr: &[0, 128, 384, 896],
                salt: 0x2001,
            },
            RaggedPrefillCase {
                name: "bf16_ragged_gqa4_b2_q32_64_kv256_1024_qh16_kvh4_d128",
                batch_size: 2,
                query_heads: 16,
                kv_heads: 4,
                qo_indptr: &[0, 32, 96],
                kv_indptr: &[0, 256, 1280],
                salt: 0x4001,
            },
        ] {
            benchmark_ragged_prefill_case(
                &context,
                &stream,
                &prefill_provider,
                case,
                config,
                &identity,
            )?;
        }
    }
    if requested.contains(&"rope") {
        benchmark_rope(&context, &stream, &rope_provider, config, &identity)?;
    }
    if requested.contains(&"rope_paged_kv_append") {
        benchmark_rope_paged_append(&context, &stream, &rope_provider, config, &identity)?;
    }
    if requested.contains(&"rope_paged_kv_append_tokens") {
        benchmark_rope_paged_append_tokens(&context, &stream, &rope_provider, config, &identity)?;
    }
    Ok(())
}
