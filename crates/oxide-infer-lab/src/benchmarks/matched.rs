use crate::benchmark::BenchmarkRecord;
use crate::comparison::{compare_bf16, digest_bf16};
use crate::fixture::{deterministic_bf16, page_refcounts};
use crate::support::gemm_fixture::{CENSUS_SHAPES, exact_fixture};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
use half::bf16;
use oxide_infer::{
    Bf16DenseGemmSpec, Bf16PagedBatchDecodeSpec, Bf16PagedPrefillSpec, Bf16RaggedPrefillSpec,
    Bf16RopePagedKvAppendSpec, Bf16RopePagedKvAppendTokensSpec, Bf16RopePosIdsSpec,
    Bf16SingleDecodeSpec, Bf16SingleDecodeSplitKSpec, DType, PagedKvLayout, RmsNormSpec,
    bf16_dense_gemm_reference, rope_paged_kv_append_bf16_reference,
    rope_paged_kv_append_tokens_bf16_reference, rope_pos_ids_bf16_reference,
};
use oxide_infer_cuda::attention::{
    Bf16PagedBatchDecodeAlgorithm, Bf16PagedBatchDecodeArgs, Bf16PagedPrefillAlgorithm,
    Bf16PagedPrefillArgs, Bf16RaggedPrefillAlgorithm, Bf16RaggedPrefillArgs, Bf16SingleDecodeArgs,
    Bf16SingleDecodePlan, Bf16SingleDecodeSplitKArgs, DecodeProvider, PrefillProvider,
};
use oxide_infer_cuda::command::{CheckedBindings, CommandQueue, Read, ReadWrite};
use oxide_infer_cuda::gemm::{
    Bf16DenseGemmAlgorithm, Bf16DenseGemmOperands, Bf16DenseGemmPlan, Bf16DenseGemmSelection,
    GemmPlanner, GemmProviderId, GemmProviderVersion,
};
use oxide_infer_cuda::rms_norm::{RmsNormArgs, RmsNormBf16Plan, RmsNormProvider};
use oxide_infer_cuda::rope::{
    Bf16PagedKvAppendMapArgs, Bf16PagedKvAppendTokensMapArgs, Bf16RopePagedKvAppendMappedArgs,
    Bf16RopePosIdsArgs, RopeProvider,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::sync::Arc;

const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");
const MEASUREMENT: &str = "eager_stream_batch_cuda_event";
const FIXTURE_ID: &str = "xorshift64_mod2001_bf16_v1";
const GEMV_FIXTURE_ID: &str = "dyadic_exact_qwen25_15b_gemv_census_v1";
const PAGED_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_page_table_layout_v2";
const PAGED_PREFILL_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_paged_prefill_v1";
const RAGGED_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_ragged_indptr_v1";
const ROPE_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_rope_pos_ids_v1";
const ROPE_APPEND_FIXTURE_ID: &str = "xorshift64_mod2001_bf16_i32_rope_paged_append_v2";
const ROPE_APPEND_TOKENS_FIXTURE_ID: &str =
    "xorshift64_mod2001_bf16_i32_rope_paged_append_tokens_v2";
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const PAGED_DECODE_B8_L96_INDPTR: [i32; 9] = [0, 6, 12, 18, 24, 30, 36, 42, 48];
const PAGED_DECODE_B8_L96_INDICES: [i32; 48] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
];
const PAGED_DECODE_B8_L96_LAST_PAGE_LEN: [i32; 8] = [16; 8];
const PAGED_DECODE_B16_L96_INDPTR: [i32; 17] = [
    0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96,
];
const PAGED_DECODE_B16_L96_INDICES: [i32; 96] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
];
const PAGED_DECODE_B16_L96_LAST_PAGE_LEN: [i32; 16] = [16; 16];

struct RunIdentity {
    provider_commit: String,
    run_label: String,
}

impl RunIdentity {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let provider_commit = env::var("OXIDE_SOURCE_COMMIT")?;
        if provider_commit.len() != 40
            || !provider_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("OXIDE_SOURCE_COMMIT must be a full 40-character Git commit SHA".into());
        }
        Ok(Self {
            provider_commit,
            run_label: env::var("OXIDE_BENCH_RUN_LABEL")
                .unwrap_or_else(|_| "unlabeled".to_string()),
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
    algorithm: Bf16PagedBatchDecodeAlgorithm,
    layout: PagedKvLayout,
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
struct PagedPrefillCase {
    name: &'static str,
    algorithm: Bf16PagedPrefillAlgorithm,
    batch_size: usize,
    max_num_pages: usize,
    query_heads: usize,
    kv_heads: usize,
    qo_indptr: &'static [i32],
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
            warmup_launches: env_usize("OXIDE_BENCH_WARMUP", 20)?,
            launches_per_sample: env_usize("OXIDE_BENCH_LAUNCHES", 100)?,
            samples: env_usize("OXIDE_BENCH_SAMPLES", 30)?,
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
    F: FnMut(&mut oxide_infer_cuda::command::CommandScope<'_>) -> Result<(), Box<dyn Error>>,
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
        let completion = scope.finish();
        end.record(stream)?;
        bindings = completion.wait()?;
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

fn pack_paged_decode_kv(
    logical_nhd: &[bf16],
    max_num_pages: usize,
    kv_heads: usize,
    layout: PagedKvLayout,
) -> Vec<bf16> {
    match layout {
        PagedKvLayout::Nhd => logical_nhd.to_vec(),
        PagedKvLayout::Hnd => {
            let page_size = 16;
            let head_dim = 128;
            let mut packed = vec![bf16::NAN; logical_nhd.len()];
            for page in 0..max_num_pages {
                for token in 0..page_size {
                    for head in 0..kv_heads {
                        let source = ((page * page_size + token) * kv_heads + head) * head_dim;
                        let target = ((page * kv_heads + head) * page_size + token) * head_dim;
                        packed[target..target + head_dim]
                            .copy_from_slice(&logical_nhd[source..source + head_dim]);
                    }
                }
            }
            packed
        }
    }
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
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample, 1)?;
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
        provider: "oxide-infer",
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
    planner: &GemmPlanner,
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16DenseGemmSpec::new(1, 4096, 4096)?;
    let plan: Bf16DenseGemmPlan =
        planner.plan_bf16_dense(spec, Bf16DenseGemmSelection::CublasLt)?;
    let activation_host = deterministic_bf16(spec.a_numel(), 0x4143_5449);
    let weight_host = deterministic_bf16(spec.weight_numel(), 0x4745_4d4d);
    let activation = Arc::new(DeviceBuffer::from_host(stream, &activation_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(stream, &weight_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(stream, plan.workspace_required_bytes())?;
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample, 1)?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(activation)?;
    let weight_handle = bindings.bind_read(weight)?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let (_bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            plan.enqueue_into(
                scope,
                Bf16DenseGemmOperands::new(
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
        provider: "oxide-infer",
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

fn gemv_selection_from_env() -> Result<Bf16DenseGemmSelection, Box<dyn Error>> {
    match env::var("OXIDE_BENCH_GEMV_PROVIDER") {
        Ok(value) if value == "oxide" => Ok(Bf16DenseGemmSelection::Oxide),
        Ok(value) if value == "cublaslt" => Ok(Bf16DenseGemmSelection::CublasLt),
        Ok(value) => {
            Err(format!("OXIDE_BENCH_GEMV_PROVIDER must be oxide or cublaslt, got {value}").into())
        }
        Err(env::VarError::NotPresent) => {
            Err("OXIDE_BENCH_GEMV_PROVIDER is required for gemv_m1".into())
        }
        Err(error) => Err(error.into()),
    }
}

fn gemv_algorithm_name(algorithm: Bf16DenseGemmAlgorithm) -> &'static str {
    match algorithm {
        Bf16DenseGemmAlgorithm::CublasLtHeuristic => "cublaslt_heuristic",
        Bf16DenseGemmAlgorithm::OxideSm90SimtGemvM1N16K64 => "oxide_sm90_simt_gemv_m1_n16_k64",
    }
}

fn gemv_provider_version(planner: &GemmPlanner, provider: GemmProviderId) -> serde_json::Value {
    match planner.provider_version(provider) {
        GemmProviderVersion::CublasLt(version) => json!(version),
        GemmProviderVersion::Oxide(version) => json!(version),
    }
}

fn benchmark_gemv_case(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    planner: &GemmPlanner,
    selection: Bf16DenseGemmSelection,
    dimensions: (usize, usize, usize),
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16DenseGemmSpec::new(dimensions.0, dimensions.1, dimensions.2)?;
    let plan = planner.plan_bf16_dense(spec, selection)?;
    let plan_info = plan.plan_info();
    let (activation_host, weight_host) = exact_fixture(spec);
    let mut expected = vec![bf16::ZERO; spec.output_numel()];
    bf16_dense_gemm_reference(&activation_host, &weight_host, &mut expected, spec)?;

    let activation = Arc::new(DeviceBuffer::from_host(stream, &activation_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(stream, &weight_host)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(stream, plan.workspace_required_bytes())?;
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample, 1)?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(activation)?;
    let weight_handle = bindings.bind_read(weight)?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let (mut bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            plan.enqueue_into(
                scope,
                Bf16DenseGemmOperands::new(
                    activation_handle,
                    weight_handle,
                    output_handle.write(),
                    workspace_handle.write(),
                ),
            )?;
            Ok(())
        })?;
    let output = bindings.take_read_write(output_handle)?;
    let actual = output.to_host_vec(stream)?;
    let comparison = compare_bf16(&actual, &expected, "matched GEMV")?;
    if comparison.bit_mismatches != 0 {
        return Err(format!(
            "matched GEMV provider {} differed from the exact CPU reference: bit_mismatches={} max_abs={}",
            plan_info.provider().name(),
            comparison.bit_mismatches,
            comparison.max_abs,
        )
        .into());
    }

    let case = format!("bf16_gemv_m1_n{}_k{}", spec.n(), spec.k());
    BenchmarkRecord {
        schema_version: 1,
        provider: "oxide-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "gemv",
        case: &case,
        dtype: "bf16",
        layout: "A_row_major_W_row_major_transposed",
        execution: json!({
            "provider": plan_info.provider().name(),
            "provider_version": gemv_provider_version(planner, plan_info.provider()),
            "algorithm": gemv_algorithm_name(plan_info.algorithm()),
            "commands_per_call": 1,
            "workspace_required_bytes": plan_info.workspace_required_bytes(),
            "estimated_waves_count": plan.estimated_waves_count(),
            "correctness": {
                "reference": "oxide-infer CPU F32 sequential dense GEMM",
                "tolerance": "bit_exact_dyadic_fixture",
                "max_abs": comparison.max_abs,
                "bit_mismatches": comparison.bit_mismatches,
                "output_digest": format!("{:016x}", comparison.digest),
                "reference_digest": format!("{:016x}", digest_bf16(&expected)),
            }
        }),
        kernels_per_call: 1,
        shape: json!({"m": spec.m(), "n": spec.n(), "k": spec.k()}),
        fixture_id: GEMV_FIXTURE_ID,
        fixture_digests: json!({
            "activation": format!("{:016x}", digest_bf16(&activation_host)),
            "weight_storage": format!("{:016x}", digest_bf16(&weight_host)),
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
        let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample, 1)?;
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
        let mut queue = CommandQueue::new(stream.clone(), command_capacity, 1)?;
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
        provider: "oxide-infer",
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
        case.layout,
    )?;
    let table =
        spec.validate_page_table(case.page_indptr, case.page_indices, case.last_page_len)?;
    let plan = provider.plan_bf16_paged_batch_with_algorithm(spec, case.algorithm)?;
    let algorithm = match plan.algorithm() {
        Bf16PagedBatchDecodeAlgorithm::Direct => "direct_one_warp_per_request_head",
        Bf16PagedBatchDecodeAlgorithm::TokenParallel8 => "token_parallel_8warp_block_local_merge",
    };
    let query_host = deterministic_bf16(spec.query_numel(), case.salt);
    let logical_key_host = deterministic_bf16(spec.kv_pages_numel(), case.salt ^ 0x4b45_5900);
    let logical_value_host =
        deterministic_bf16(spec.kv_pages_numel(), case.salt ^ 0x5641_4c55_4500);
    let key_host = pack_paged_decode_kv(
        &logical_key_host,
        spec.max_num_pages(),
        spec.num_kv_heads(),
        case.layout,
    );
    let value_host = pack_paged_decode_kv(
        &logical_value_host,
        spec.max_num_pages(),
        spec.num_kv_heads(),
        case.layout,
    );
    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key_pages = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let value_pages = Arc::new(DeviceBuffer::from_host(stream, &value_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(stream, case.page_indptr)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(stream, case.page_indices)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(stream, case.last_page_len)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(stream, spec.lse_numel())?;
    let mut metadata_statuses = Vec::with_capacity(config.launches_per_sample);
    for _ in 0..config.launches_per_sample {
        metadata_statuses.push(DeviceBuffer::<i32>::zeroed(
            stream,
            plan.metadata_status_required_numel(),
        )?);
    }
    let command_capacity = config
        .launches_per_sample
        .checked_mul(3)
        .ok_or("paged-decode command capacity overflowed")?;
    let mut queue = CommandQueue::new(stream.clone(), command_capacity, 1)?;
    let binding_capacity = 8_usize
        .checked_add(config.launches_per_sample)
        .ok_or("paged-decode binding capacity overflowed")?;
    let mut bindings = queue.bindings(binding_capacity)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let indptr_handle = bindings.bind_read(page_indptr)?;
    let indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let mut metadata_status_handles = Vec::with_capacity(config.launches_per_sample);
    for metadata_status in metadata_statuses {
        metadata_status_handles.push(bindings.bind_read_write(metadata_status)?);
    }
    let mut metadata_status_index = 0_usize;

    let (_bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            let metadata_status_handle = metadata_status_handles[metadata_status_index];
            metadata_status_index = (metadata_status_index + 1) % metadata_status_handles.len();
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
        provider: "oxide-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "paged_batch_decode",
        case: case.name,
        dtype: "bf16",
        layout: match case.layout {
            PagedKvLayout::Nhd => "NHD_D128_page16",
            PagedKvLayout::Hnd => "HND_D128_page16",
        },
        execution: json!({
            "algorithm": algorithm,
            "page_table_location": "device",
            "metadata_validation": "device_status",
            "validator_kernels_per_call": 1,
            "attention_kernels_per_call": 1,
            "status_readbacks_per_call": 1,
            "commands_per_call": 3
        }),
        kernels_per_call: 2,
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

fn benchmark_paged_prefill_case(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    provider: &PrefillProvider,
    case: PagedPrefillCase,
    config: BenchConfig,
    identity: &RunIdentity,
) -> Result<(), Box<dyn Error>> {
    let nnz_qo = usize::try_from(*case.qo_indptr.last().ok_or("empty qo_indptr")?)?;
    let spec = Bf16PagedPrefillSpec::new(
        case.batch_size,
        nnz_qo,
        case.max_num_pages,
        case.query_heads,
        case.kv_heads,
        128,
        16,
    )?;
    let metadata = spec.validate_metadata(
        case.qo_indptr,
        case.page_indptr,
        case.page_indices,
        case.last_page_len,
    )?;
    let plan = provider.plan_bf16_paged(spec, case.algorithm)?;
    let algorithm = match plan.algorithm() {
        Bf16PagedPrefillAlgorithm::Direct => "direct_one_warp_per_query_row_head",
        Bf16PagedPrefillAlgorithm::TokenParallel8 => "token_parallel_8warp_block_local_merge",
        Bf16PagedPrefillAlgorithm::TokenParallel16 => "token_parallel_16warp_block_local_merge",
        Bf16PagedPrefillAlgorithm::TiledGqa4 => "tiled_gqa4_paged_mma_qk_softmax_pv",
    };
    let query_host = deterministic_bf16(spec.query_numel(), case.salt);
    let key_host = deterministic_bf16(spec.kv_pages_numel(), case.salt ^ 0x4b45_5900);
    let value_host = deterministic_bf16(spec.kv_pages_numel(), case.salt ^ 0x5641_4c55_4500);
    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key_pages = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let value_pages = Arc::new(DeviceBuffer::from_host(stream, &value_host)?);
    let qo_indptr = Arc::new(DeviceBuffer::from_host(stream, case.qo_indptr)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(stream, case.page_indptr)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(stream, case.page_indices)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(stream, case.last_page_len)?);
    let output = DeviceBuffer::<bf16>::zeroed(stream, spec.output_numel())?;
    let lse = DeviceBuffer::<f32>::zeroed(stream, spec.lse_numel())?;
    let workspace =
        DeviceBuffer::<f32>::zeroed(stream, usize::max(plan.workspace_required_numel(), 1))?;
    let mut metadata_statuses = Vec::with_capacity(config.launches_per_sample);
    for _ in 0..config.launches_per_sample {
        metadata_statuses.push(DeviceBuffer::<i32>::zeroed(
            stream,
            plan.metadata_status_required_numel(),
        )?);
    }
    let attention_kernels_per_call = if plan.workspace_required_numel() == 0 {
        1
    } else {
        2
    };
    let commands_per_call = attention_kernels_per_call + 2;
    let kernels_per_call = attention_kernels_per_call + 1;
    let command_capacity = config
        .launches_per_sample
        .checked_mul(commands_per_call)
        .ok_or("paged-prefill command capacity overflowed")?;
    let mut queue = CommandQueue::new(stream.clone(), command_capacity, 1)?;
    let binding_capacity = 10_usize
        .checked_add(config.launches_per_sample)
        .ok_or("paged-prefill binding capacity overflowed")?;
    let mut bindings = queue.bindings(binding_capacity)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key_pages)?;
    let value_handle = bindings.bind_read(value_pages)?;
    let qo_indptr_handle = bindings.bind_read(qo_indptr)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let output_handle = bindings.bind_read_write(output)?;
    let lse_handle = bindings.bind_read_write(lse)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let mut metadata_status_handles = Vec::with_capacity(config.launches_per_sample);
    for metadata_status in metadata_statuses {
        metadata_status_handles.push(bindings.bind_read_write(metadata_status)?);
    }
    let mut metadata_status_index = 0_usize;

    let (_bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            let metadata_status_handle = metadata_status_handles[metadata_status_index];
            metadata_status_index = (metadata_status_index + 1) % metadata_status_handles.len();
            let mut args = Bf16PagedPrefillArgs::new(
                query_handle,
                key_handle,
                value_handle,
                qo_indptr_handle,
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
            plan.enqueue_into(scope, args)?;
            Ok(())
        })?;
    let mut request_qo_lens = Vec::with_capacity(spec.batch_size());
    let mut request_kv_lens = Vec::with_capacity(spec.batch_size());
    for request in 0..spec.batch_size() {
        let (qo_start, qo_end) = metadata
            .request_query_range(request)
            .expect("validated request has a query range");
        request_qo_lens.push(qo_end - qo_start);
        request_kv_lens.push(
            metadata
                .request_kv_len(request)
                .expect("validated request has a KV length"),
        );
    }

    BenchmarkRecord {
        schema_version: 1,
        provider: "oxide-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "paged_prefill",
        case: case.name,
        dtype: "bf16",
        layout: "NHD_D128_page16",
        execution: json!({
            "algorithm": algorithm,
            "causal": "bottom_right",
            "page_table_location": "device",
            "metadata_validation": "device_status",
            "validator_kernels_per_call": 1,
            "attention_kernels_per_call": attention_kernels_per_call,
            "status_readbacks_per_call": 1,
            "commands_per_call": commands_per_call
        }),
        kernels_per_call,
        shape: json!({
            "batch_size": spec.batch_size(),
            "nnz_qo": spec.nnz_qo(),
            "max_num_pages": spec.max_num_pages(),
            "referenced_pages": case.page_indices.len(),
            "request_qo_lens": request_qo_lens,
            "request_kv_lens": request_kv_lens,
            "query_heads": spec.num_query_heads(),
            "kv_heads": spec.num_kv_heads(),
            "head_dim": spec.head_dim(),
            "page_size": spec.page_size()
        }),
        fixture_id: PAGED_PREFILL_FIXTURE_ID,
        fixture_digests: json!({
            "query": format!("{:016x}", digest_bf16(&query_host)),
            "key_pages": format!("{:016x}", digest_bf16(&key_host)),
            "value_pages": format!("{:016x}", digest_bf16(&value_host)),
            "qo_indptr": format!("{:016x}", digest_i32(case.qo_indptr)),
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
    let mut queue = CommandQueue::new(stream.clone(), command_capacity, 1)?;
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
        provider: "oxide-infer",
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
    let mut queue = CommandQueue::new(stream.clone(), config.launches_per_sample, 1)?;
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
        provider: "oxide-infer",
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
                "reference": "oxide-infer CPU reference",
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
    let page_indices_host = [3_i32, 2, 6, 2, 1, 7, 0, 4];
    let last_page_len_host = [3_i32, 16, 1, 9];
    let page_refcounts_host = page_refcounts(spec.max_num_pages(), &page_indices_host);
    spec.validate_metadata(
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &page_refcounts_host,
    )?;
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
        &page_refcounts_host,
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
    let page_refcounts = Arc::new(DeviceBuffer::from_host(stream, &page_refcounts_host)?);
    let query_output = DeviceBuffer::<bf16>::zeroed(stream, spec.query_output_numel())?;
    let key_pages = DeviceBuffer::from_host(stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(stream, &value_pages_host)?;
    let plan = provider.plan_bf16_paged_append(spec)?;
    let mut workspaces = Vec::with_capacity(config.launches_per_sample);
    for _ in 0..config.launches_per_sample {
        workspaces.push(DeviceBuffer::<i32>::zeroed(
            stream,
            plan.workspace_required_numel(),
        )?);
    }
    let command_capacity = config
        .launches_per_sample
        .checked_mul(3)
        .ok_or("RoPE append command capacity overflowed")?;
    let mut queue = CommandQueue::new(stream.clone(), command_capacity, 1)?;
    let binding_capacity = 10_usize
        .checked_add(config.launches_per_sample)
        .ok_or("RoPE append binding capacity overflowed")?;
    let mut bindings = queue.bindings(binding_capacity)?;
    let query_handle = bindings.bind_read(query)?;
    let key_handle = bindings.bind_read(key)?;
    let value_handle = bindings.bind_read(value)?;
    let page_indptr_handle = bindings.bind_read(page_indptr)?;
    let page_indices_handle = bindings.bind_read(page_indices)?;
    let last_page_len_handle = bindings.bind_read(last_page_len)?;
    let page_refcounts_handle = bindings.bind_read(page_refcounts)?;
    let query_output_handle = bindings.bind_read_write(query_output)?;
    let key_pages_handle = bindings.bind_read_write(key_pages)?;
    let value_pages_handle = bindings.bind_read_write(value_pages)?;
    let mut workspace_handles = Vec::with_capacity(config.launches_per_sample);
    for workspace in workspaces {
        workspace_handles.push(bindings.bind_read_write(workspace)?);
    }
    let mut workspace_index = 0_usize;

    let (mut bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            let workspace_handle = workspace_handles[workspace_index];
            workspace_index = (workspace_index + 1) % workspace_handles.len();
            let append_map = plan.enqueue_map_into(
                scope,
                Bf16PagedKvAppendMapArgs::new(
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
        provider: "oxide-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "rope_paged_kv_append",
        case: "bf16_rope_paged_append_b4_qh16_kh4_d128_p16",
        dtype: "bf16",
        layout: "NHD_D128_neox_split_half_page16",
        execution: json!({
            "algorithm": "validate_compact_then_fused_append",
            "commands": 3,
            "kernels": 2,
            "status_readbacks": 1,
            "positions": [2, 31, 16, 40],
            "physical_slots": [[3, 2], [6, 15], [1, 0], [4, 8]],
            "correctness": {
                "reference": "oxide-infer CPU reference",
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
        kernels_per_call: 2,
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
            "last_page_len": format!("{:016x}", digest_i32(&last_page_len_host)),
            "page_refcounts": format!("{:016x}", digest_i32(&page_refcounts_host))
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

    let query = Arc::new(DeviceBuffer::from_host(stream, &query_host)?);
    let key = Arc::new(DeviceBuffer::from_host(stream, &key_host)?);
    let value = Arc::new(DeviceBuffer::from_host(stream, &value_host)?);
    let batch_indices = Arc::new(DeviceBuffer::from_host(stream, &batch_indices_host)?);
    let positions = Arc::new(DeviceBuffer::from_host(stream, &positions_host)?);
    let page_indptr = Arc::new(DeviceBuffer::from_host(stream, &page_indptr_host)?);
    let page_indices = Arc::new(DeviceBuffer::from_host(stream, &page_indices_host)?);
    let last_page_len = Arc::new(DeviceBuffer::from_host(stream, &last_page_len_host)?);
    let page_refcounts = Arc::new(DeviceBuffer::from_host(stream, &page_refcounts_host)?);
    let query_output = DeviceBuffer::<bf16>::zeroed(stream, spec.query_output_numel())?;
    let key_pages = DeviceBuffer::from_host(stream, &key_pages_host)?;
    let value_pages = DeviceBuffer::from_host(stream, &value_pages_host)?;
    let plan = provider.plan_bf16_paged_append_tokens(spec)?;
    let mut workspaces = Vec::with_capacity(config.launches_per_sample);
    for _ in 0..config.launches_per_sample {
        workspaces.push(DeviceBuffer::<i32>::zeroed(
            stream,
            plan.workspace_required_numel(),
        )?);
    }
    let command_capacity = config
        .launches_per_sample
        .checked_mul(3)
        .ok_or("explicit RoPE append command capacity overflowed")?;
    let mut queue = CommandQueue::new(stream.clone(), command_capacity, 1)?;
    let binding_capacity = 12_usize
        .checked_add(config.launches_per_sample)
        .ok_or("explicit RoPE append binding capacity overflowed")?;
    let mut bindings = queue.bindings(binding_capacity)?;
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
    let mut workspace_handles = Vec::with_capacity(config.launches_per_sample);
    for workspace in workspaces {
        workspace_handles.push(bindings.bind_read_write(workspace)?);
    }
    let mut workspace_index = 0_usize;

    let (mut bindings, samples_us) =
        benchmark_scopes(context, stream, &mut queue, bindings, config, |scope| {
            let workspace_handle = workspace_handles[workspace_index];
            workspace_index = (workspace_index + 1) % workspace_handles.len();
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
        provider: "oxide-infer",
        provider_version: PROVIDER_VERSION,
        provider_commit: &identity.provider_commit,
        run_label: &identity.run_label,
        measurement: MEASUREMENT,
        operator: "rope_paged_kv_append_tokens",
        case: "bf16_rope_paged_append_t6_b3_qh16_kh4_d128_p16",
        dtype: "bf16",
        layout: "NHD_D128_neox_split_half_page16",
        execution: json!({
            "algorithm": "validate_compact_then_fused_append_explicit_tokens",
            "commands": 3,
            "kernels": 2,
            "status_readbacks": 1,
            "batch_indices": [2, 0, 1, 0, 2, 1],
            "positions": [5, 17, 20, 16, 4, 19],
            "physical_slots": [[5, 5], [3, 1], [6, 4], [3, 0], [5, 4], [6, 3]],
            "correctness": {
                "reference": "oxide-infer CPU reference",
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
            "last_page_len": format!("{:016x}", digest_i32(&last_page_len_host)),
            "page_refcounts": format!("{:016x}", digest_i32(&page_refcounts_host))
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
    let requested = env::var("OXIDE_BENCH_OPERATORS").unwrap_or_else(|_| {
        "rms_norm,gemm,single_decode,paged_batch_decode,paged_prefill,ragged_prefill,rope,rope_paged_kv_append,rope_paged_kv_append_tokens".to_string()
    });
    let requested = requested.split(',').collect::<Vec<_>>();
    let context = CudaContext::new(0)?;
    let stream: Arc<CudaStream> = context.new_stream()?;
    let rms_provider = RmsNormProvider::load(&context)?;
    let gemm_planner = GemmPlanner::load(&context)?;
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
        benchmark_gemm(&context, &stream, &gemm_planner, config, &identity)?;
    }
    if requested.contains(&"gemv_m1") {
        let selection = gemv_selection_from_env()?;
        for dimensions in CENSUS_SHAPES {
            benchmark_gemv_case(
                &context,
                &stream,
                &gemm_planner,
                selection,
                dimensions,
                config,
                &identity,
            )?;
        }
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
                name: "bf16_paged_mha_b1_l1_qh8_kvh8_d128_p16_nhd",
                algorithm: Bf16PagedBatchDecodeAlgorithm::Direct,
                layout: PagedKvLayout::Nhd,
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
                name: "bf16_paged_mqa_b3_l16_23_48_qh8_kvh1_d128_p16_nhd",
                algorithm: Bf16PagedBatchDecodeAlgorithm::TokenParallel8,
                layout: PagedKvLayout::Nhd,
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
                name: "bf16_paged_gqa4_b4_l3_32_17_41_qh16_kvh4_d128_p16_nhd",
                algorithm: Bf16PagedBatchDecodeAlgorithm::TokenParallel8,
                layout: PagedKvLayout::Nhd,
                batch_size: 4,
                max_num_pages: 8,
                query_heads: 16,
                kv_heads: 4,
                page_indptr: &[0, 1, 3, 5, 8],
                page_indices: &[7, 2, 6, 5, 1, 7, 0, 4],
                last_page_len: &[3, 16, 1, 9],
                salt: 0x4001,
            },
            PagedDecodeCase {
                name: "bf16_paged_mha_b1_l1_qh8_kvh8_d128_p16_hnd",
                algorithm: Bf16PagedBatchDecodeAlgorithm::Direct,
                layout: PagedKvLayout::Hnd,
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
                name: "bf16_paged_mqa_b3_l16_23_48_qh8_kvh1_d128_p16_hnd",
                algorithm: Bf16PagedBatchDecodeAlgorithm::TokenParallel8,
                layout: PagedKvLayout::Hnd,
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
                name: "bf16_paged_gqa4_b4_l3_32_17_41_qh16_kvh4_d128_p16_hnd",
                algorithm: Bf16PagedBatchDecodeAlgorithm::TokenParallel8,
                layout: PagedKvLayout::Hnd,
                batch_size: 4,
                max_num_pages: 8,
                query_heads: 16,
                kv_heads: 4,
                page_indptr: &[0, 1, 3, 5, 8],
                page_indices: &[7, 2, 6, 5, 1, 7, 0, 4],
                last_page_len: &[3, 16, 1, 9],
                salt: 0x4001,
            },
            PagedDecodeCase {
                name: "bf16_paged_qwen15b_b8_l96_qh12_kvh2_d128_p16_hnd_direct",
                algorithm: Bf16PagedBatchDecodeAlgorithm::Direct,
                layout: PagedKvLayout::Hnd,
                batch_size: 8,
                max_num_pages: 48,
                query_heads: 12,
                kv_heads: 2,
                page_indptr: &PAGED_DECODE_B8_L96_INDPTR,
                page_indices: &PAGED_DECODE_B8_L96_INDICES,
                last_page_len: &PAGED_DECODE_B8_L96_LAST_PAGE_LEN,
                salt: 0x1508,
            },
            PagedDecodeCase {
                name: "bf16_paged_qwen15b_b8_l96_qh12_kvh2_d128_p16_hnd_token_parallel8",
                algorithm: Bf16PagedBatchDecodeAlgorithm::TokenParallel8,
                layout: PagedKvLayout::Hnd,
                batch_size: 8,
                max_num_pages: 48,
                query_heads: 12,
                kv_heads: 2,
                page_indptr: &PAGED_DECODE_B8_L96_INDPTR,
                page_indices: &PAGED_DECODE_B8_L96_INDICES,
                last_page_len: &PAGED_DECODE_B8_L96_LAST_PAGE_LEN,
                salt: 0x1508,
            },
            PagedDecodeCase {
                name: "bf16_paged_qwen15b_b16_l96_qh12_kvh2_d128_p16_hnd_direct",
                algorithm: Bf16PagedBatchDecodeAlgorithm::Direct,
                layout: PagedKvLayout::Hnd,
                batch_size: 16,
                max_num_pages: 96,
                query_heads: 12,
                kv_heads: 2,
                page_indptr: &PAGED_DECODE_B16_L96_INDPTR,
                page_indices: &PAGED_DECODE_B16_L96_INDICES,
                last_page_len: &PAGED_DECODE_B16_L96_LAST_PAGE_LEN,
                salt: 0x1516,
            },
            PagedDecodeCase {
                name: "bf16_paged_qwen15b_b16_l96_qh12_kvh2_d128_p16_hnd_token_parallel8",
                algorithm: Bf16PagedBatchDecodeAlgorithm::TokenParallel8,
                layout: PagedKvLayout::Hnd,
                batch_size: 16,
                max_num_pages: 96,
                query_heads: 12,
                kv_heads: 2,
                page_indptr: &PAGED_DECODE_B16_L96_INDPTR,
                page_indices: &PAGED_DECODE_B16_L96_INDICES,
                last_page_len: &PAGED_DECODE_B16_L96_LAST_PAGE_LEN,
                salt: 0x1516,
            },
            PagedDecodeCase {
                name: "bf16_paged_qwen7b_b8_l96_qh28_kvh4_d128_p16_hnd_direct",
                algorithm: Bf16PagedBatchDecodeAlgorithm::Direct,
                layout: PagedKvLayout::Hnd,
                batch_size: 8,
                max_num_pages: 48,
                query_heads: 28,
                kv_heads: 4,
                page_indptr: &PAGED_DECODE_B8_L96_INDPTR,
                page_indices: &PAGED_DECODE_B8_L96_INDICES,
                last_page_len: &PAGED_DECODE_B8_L96_LAST_PAGE_LEN,
                salt: 0x7008,
            },
            PagedDecodeCase {
                name: "bf16_paged_qwen7b_b8_l96_qh28_kvh4_d128_p16_hnd_token_parallel8",
                algorithm: Bf16PagedBatchDecodeAlgorithm::TokenParallel8,
                layout: PagedKvLayout::Hnd,
                batch_size: 8,
                max_num_pages: 48,
                query_heads: 28,
                kv_heads: 4,
                page_indptr: &PAGED_DECODE_B8_L96_INDPTR,
                page_indices: &PAGED_DECODE_B8_L96_INDICES,
                last_page_len: &PAGED_DECODE_B8_L96_LAST_PAGE_LEN,
                salt: 0x7008,
            },
            PagedDecodeCase {
                name: "bf16_paged_qwen7b_b16_l96_qh28_kvh4_d128_p16_hnd_direct",
                algorithm: Bf16PagedBatchDecodeAlgorithm::Direct,
                layout: PagedKvLayout::Hnd,
                batch_size: 16,
                max_num_pages: 96,
                query_heads: 28,
                kv_heads: 4,
                page_indptr: &PAGED_DECODE_B16_L96_INDPTR,
                page_indices: &PAGED_DECODE_B16_L96_INDICES,
                last_page_len: &PAGED_DECODE_B16_L96_LAST_PAGE_LEN,
                salt: 0x7016,
            },
            PagedDecodeCase {
                name: "bf16_paged_qwen7b_b16_l96_qh28_kvh4_d128_p16_hnd_token_parallel8",
                algorithm: Bf16PagedBatchDecodeAlgorithm::TokenParallel8,
                layout: PagedKvLayout::Hnd,
                batch_size: 16,
                max_num_pages: 96,
                query_heads: 28,
                kv_heads: 4,
                page_indptr: &PAGED_DECODE_B16_L96_INDPTR,
                page_indices: &PAGED_DECODE_B16_L96_INDICES,
                last_page_len: &PAGED_DECODE_B16_L96_LAST_PAGE_LEN,
                salt: 0x7016,
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
    if requested.contains(&"paged_prefill") {
        for case in [
            PagedPrefillCase {
                name: "bf16_paged_prefill_mha_b1_q4_kv4_qh8_kvh8_d128_p16",
                algorithm: Bf16PagedPrefillAlgorithm::Direct,
                batch_size: 1,
                max_num_pages: 2,
                query_heads: 8,
                kv_heads: 8,
                qo_indptr: &[0, 4],
                page_indptr: &[0, 1],
                page_indices: &[1],
                last_page_len: &[4],
                salt: 0x1001,
            },
            PagedPrefillCase {
                name: "bf16_paged_prefill_mqa_b3_q2_3_1_kv4_22_35_qh8_kvh1_d128_p16",
                algorithm: Bf16PagedPrefillAlgorithm::Direct,
                batch_size: 3,
                max_num_pages: 7,
                query_heads: 8,
                kv_heads: 1,
                qo_indptr: &[0, 2, 5, 6],
                page_indptr: &[0, 1, 3, 6],
                page_indices: &[4, 6, 1, 5, 0, 3],
                last_page_len: &[4, 6, 3],
                salt: 0x2001,
            },
            PagedPrefillCase {
                name: "bf16_paged_prefill_gqa4_b2_q4_2_kv23_18_qh16_kvh4_d128_p16",
                algorithm: Bf16PagedPrefillAlgorithm::Direct,
                batch_size: 2,
                max_num_pages: 6,
                query_heads: 16,
                kv_heads: 4,
                qo_indptr: &[0, 4, 6],
                page_indptr: &[0, 2, 4],
                page_indices: &[5, 1, 5, 3],
                last_page_len: &[7, 2],
                salt: 0x4001,
            },
            PagedPrefillCase {
                name: "bf16_paged_prefill_mqa_b3_q1_4_16_kv128_256_512_qh8_kvh1_d128_p16",
                algorithm: Bf16PagedPrefillAlgorithm::TokenParallel16,
                batch_size: 3,
                max_num_pages: 64,
                query_heads: 8,
                kv_heads: 1,
                qo_indptr: &[0, 1, 5, 21],
                page_indptr: &[0, 8, 24, 56],
                page_indices: &[
                    7, 2, 11, 5, 13, 3, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 1, 4, 8,
                    14, 22, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 0,
                    6, 9, 10, 12, 15, 16, 18, 20, 21, 24, 25, 26, 27, 30, 33,
                ],
                last_page_len: &[16, 16, 16],
                salt: 0x6001,
            },
            PagedPrefillCase {
                name: "bf16_paged_prefill_gqa4_b2_q32_64_kv256_1024_qh16_kvh4_d128_p16",
                algorithm: Bf16PagedPrefillAlgorithm::TiledGqa4,
                batch_size: 2,
                max_num_pages: 96,
                query_heads: 16,
                kv_heads: 4,
                qo_indptr: &[0, 32, 96],
                page_indptr: &[0, 16, 80],
                page_indices: &[
                    15, 3, 27, 9, 31, 1, 35, 5, 39, 7, 43, 11, 47, 13, 51, 17, 19, 21, 23, 25, 29,
                    33, 37, 41, 45, 49, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81,
                    83, 85, 87, 89, 91, 93, 95, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,
                    28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
                ],
                last_page_len: &[16, 16],
                salt: 0x8001,
            },
        ] {
            benchmark_paged_prefill_case(
                &context,
                &stream,
                &prefill_provider,
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
