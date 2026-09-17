//! Versioned DeepGEMM SM90 contract for block-scaled FP8 projections.
//!
//! This module contains no CUDA loading and no model names. It turns physical
//! matrix shapes plus target facts into auditable kernel descriptors. A
//! descriptor is executable only after an AOT image with the same identity has
//! passed numerical qualification.

use std::collections::BTreeMap;
use std::fmt;

use kern_manifest::types::{Buffer, DType, Dim};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DEEPGEMM_REVISION: &str = "559d79fb6994a58b8a15b4b93bf13ccc16edf247";
pub const PACKED_ACTIVATION_ABI: &str = "fp8-e4m3-row128-f32-kmajor-align4-rne-reciprocal-v2";
pub const LOW_LATENCY_ROW_BUCKETS: [u32; 4] = [1, 2, 4, 8];
const AOT_WRAPPER: &str = include_str!("deepgemm_sm90/wrapper.cu");
const QUANTIZER: &str = include_str!("deepgemm_sm90/quantizer.cu");

const SCALE_BLOCK: u32 = 128;
const SCALE_ROW_ALIGNMENT: u32 = 4;
const TMA_ALIGNMENT_BYTES: u32 = 16;
const QUANTIZER_THREADS: u32 = 32;
const MAX_QUANTIZER_ROWS: u32 = u16::MAX as u32;
const FP8_MAX_FINITE: u32 = 448;
const QUANTIZATION_AMAX_FLOOR: &str = "1e-4";

const MAX_CLUSTER_SIZE: u32 = 2;
const SM90_SHARED_MEMORY_BYTES: u32 = 232_448;
const WARP_GROUP_THREADS: u32 = 128;
const WGMMA_M_ROWS: u32 = 64;
const SHARED_STORE_ALIGNMENT: u32 = 1_024;
const SHARED_SCALE_ALIGNMENT: u32 = 128;
const BARRIER_BYTES: u32 = 8;
const BARRIERS_PER_STAGE: u32 = 2;
const MAX_PIPELINE_STAGES: u32 = 16;
const TILE_N_QUANTUM: u32 = 16;
const MAX_TILE_N: u32 = 192;
const MIN_PIPELINE_STAGES: u32 = 3;
const SMALL_TILE_MIN_STAGES: u32 = 4;
const LARGE_TILE_ELEMENTS: u32 = 128 * 192;
const L2_BYTES_PER_SM_CYCLE: u64 = 64;
const L1_BYTES_PER_SM_CYCLE: u64 = 128;
const NOMINAL_L2_BYTES_PER_MICROSECOND: f64 = 8_000_000.0;
const NOMINAL_CLOCK_MHZ: f64 = 1_300.0;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fp8ProjectionShape {
    pub output_features: u32,
    pub input_features: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fp8ProjectionFamily {
    pub shape: Fp8ProjectionShape,
    pub uses: usize,
    pub buffers: Vec<String>,
}

impl Fp8ProjectionFamily {
    /// Collapse all rank-two FP8 weight buffers into model-neutral shape
    /// families. Names remain diagnostic evidence and never enter selection.
    pub fn from_weight_buffers(buffers: &BTreeMap<String, Buffer>) -> Result<Vec<Self>, ProviderContractError> {
        let mut families = BTreeMap::<Fp8ProjectionShape, Vec<String>>::new();
        for (name, buffer) in buffers.iter().filter(|(_, buffer)| buffer.dtype == DType::Fp8E4m3) {
            let [Dim::Const(output_features), Dim::Const(input_features)] = buffer.shape.as_slice() else {
                return Err(ProviderContractError::new(format!(
                    "FP8 projection `{name}` must have a static rank-two [N,K] shape"
                )));
            };
            let shape = Fp8ProjectionShape {
                output_features: u32::try_from(*output_features)
                    .map_err(|_| ProviderContractError::new(format!("FP8 projection `{name}` N exceeds u32")))?,
                input_features: u32::try_from(*input_features)
                    .map_err(|_| ProviderContractError::new(format!("FP8 projection `{name}` K exceeds u32")))?,
            };
            families.entry(shape).or_default().push(name.clone());
        }
        Ok(families.into_iter().map(|(shape, buffers)| Self { shape, uses: buffers.len(), buffers }).collect())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedActivationScratch {
    pub quantized_bytes: u64,
    pub scale_offset: u64,
    pub scale_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fp8ProjectionLayout {
    pub input_bf16: [u32; 2],
    pub weight_fp8_e4m3: [u32; 2],
    pub weight_scale_f32: [u32; 2],
    pub quantized_activation_fp8_e4m3: [u32; 2],
    /// K-major: `[K / 128, align4(M)]`.
    pub activation_scale_f32: [u32; 2],
    pub output_bf16: [u32; 2],
    pub scratch: PackedActivationScratch,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeepGemmTile {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub cluster_m: u32,
    pub cluster_n: u32,
    pub swizzle_d: u32,
    pub stages: u32,
    pub shared_memory_bytes: u32,
    pub math_threads: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum QualificationState {
    Unqualified,
    Legal,
    Benchmarked { source_sha256: String, cubin_sha256: String, median_nanoseconds: u64, samples: u32 },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeepGemmKernelContract {
    pub contract_version: u32,
    pub provider: String,
    pub provider_revision: String,
    pub target: String,
    pub numerical_abi: String,
    pub row_limit: u32,
    pub shape: Fp8ProjectionShape,
    pub layout: Fp8ProjectionLayout,
    pub tile: DeepGemmTile,
    pub num_sms: u32,
    pub qualification: QualificationState,
}

impl DeepGemmKernelContract {
    pub fn record_benchmark(
        mut self,
        source_sha256: impl Into<String>,
        cubin_sha256: impl Into<String>,
        median_nanoseconds: u64,
        samples: u32,
    ) -> Result<Self, ProviderContractError> {
        if self.qualification != QualificationState::Legal {
            return Err(ProviderContractError::new("only a legal kernel may record a benchmark"));
        }
        if median_nanoseconds == 0 || samples == 0 {
            return Err(ProviderContractError::new("benchmark evidence must contain non-zero timing and samples"));
        }
        let source_sha256 = source_sha256.into();
        let cubin_sha256 = cubin_sha256.into();
        validate_sha256(&source_sha256)?;
        validate_sha256(&cubin_sha256)?;
        self.qualification =
            QualificationState::Benchmarked { source_sha256, cubin_sha256, median_nanoseconds, samples };
        Ok(self)
    }

    pub fn verify_artifacts(&self, source: &[u8], cubin: &[u8]) -> Result<(), ProviderContractError> {
        let QualificationState::Benchmarked { source_sha256, cubin_sha256, .. } = &self.qualification else {
            return Err(ProviderContractError::new("only a benchmarked contract carries admitted artifact identities"));
        };
        if source != render_aot_source(self)?.as_bytes() {
            return Err(ProviderContractError::new(
                "AOT source does not match the compiler rendering for this contract",
            ));
        }
        if sha256(source) != *source_sha256 {
            return Err(ProviderContractError::new("AOT source does not match the qualified SHA-256"));
        }
        if sha256(cubin) != *cubin_sha256 {
            return Err(ProviderContractError::new("cubin does not match the qualified SHA-256"));
        }
        Ok(())
    }
}

pub fn render_aot_source(contract: &DeepGemmKernelContract) -> Result<String, ProviderContractError> {
    DeepGemmSm90Capability::h20().validate_contract(contract)?;
    let tile = contract.tile;
    let mut source = AOT_WRAPPER.replace("@QUANTIZER@", QUANTIZER);
    for (placeholder, value) in [
        ("@PROVIDER_REVISION@", contract.provider_revision.to_owned()),
        ("@NUMERICAL_ABI@", contract.numerical_abi.to_owned()),
        ("@ROW_LIMIT@", contract.row_limit.to_string()),
        ("@N@", contract.shape.output_features.to_string()),
        ("@K@", contract.shape.input_features.to_string()),
        ("@BLOCK_M@", tile.block_m.to_string()),
        ("@BLOCK_N@", tile.block_n.to_string()),
        ("@BLOCK_K@", tile.block_k.to_string()),
        ("@CLUSTER_M@", tile.cluster_m.to_string()),
        ("@CLUSTER_N@", tile.cluster_n.to_string()),
        ("@CLUSTER_SIZE@", (tile.cluster_m * tile.cluster_n).to_string()),
        ("@MULTICAST_ON_A@", (tile.cluster_n > 1).to_string()),
        ("@SWIZZLE_D@", tile.swizzle_d.to_string()),
        ("@STAGES@", tile.stages.to_string()),
        ("@SMEM_BYTES@", tile.shared_memory_bytes.to_string()),
        ("@MATH_THREADS@", tile.math_threads.to_string()),
        ("@NUM_SMS@", contract.num_sms.to_string()),
    ] {
        source = source.replace(placeholder, &value);
    }
    if source.contains('@') {
        return Err(ProviderContractError::new("AOT wrapper contains unresolved placeholders"));
    }
    Ok(source)
}

/// Facts recovered from a legacy generated source file. They help migrate and
/// compare old kernels, but deliberately contain no artifact digest and cannot
/// be converted into an admitted [`DeepGemmKernelContract`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalDeepGemmSourceEvidence {
    pub provider_revision: String,
    pub numerical_abi: String,
    pub shape: Fp8ProjectionShape,
    pub tile: DeepGemmTile,
    pub num_sms: u32,
}

impl HistoricalDeepGemmSourceEvidence {
    pub fn parse(source: &str) -> Result<Self, ProviderContractError> {
        let provider_revision = quoted_after(source, "kDeepGemmRevision[] =")?;
        let numerical_abi = source
            .lines()
            .find_map(|line| line.trim().strip_prefix("// Packed activation ABI:"))
            .map(str::trim)
            .filter(|abi| !abi.is_empty())
            .ok_or_else(|| ProviderContractError::new("legacy source has no packed activation ABI"))?
            .to_owned();

        let marker = "using GemmKernel = decltype(&sm90_fp8_gemm_1d2d_impl<";
        let body = source
            .split_once(marker)
            .map(|(_, body)| body)
            .ok_or_else(|| ProviderContractError::new("legacy source has no DeepGEMM kernel instantiation"))?;
        let lines: Vec<_> = body.lines().map(str::trim).filter(|line| !line.is_empty()).collect();
        let shape_values = csv_u32(
            lines.get(1).ok_or_else(|| ProviderContractError::new("legacy source has no [N,K] template line"))?,
        )?;
        let tile_values = csv_u32(
            lines.get(3).ok_or_else(|| ProviderContractError::new("legacy source has no tile template line"))?,
        )?;
        let swizzle_values = csv_u32(
            lines.get(4).ok_or_else(|| ProviderContractError::new("legacy source has no swizzle template line"))?,
        )?;
        let stages = first_u32(
            lines
                .get(5)
                .ok_or_else(|| ProviderContractError::new("legacy source has no pipeline-stage template line"))?,
        )?;
        let thread_values = csv_u32(
            lines.get(6).ok_or_else(|| ProviderContractError::new("legacy source has no thread template line"))?,
        )?;
        let cluster = first_u32_before_comma(
            lines.get(7).ok_or_else(|| ProviderContractError::new("legacy source has no cluster template line"))?,
        )?;
        let num_sms = first_u32_before_comma(
            lines.get(8).ok_or_else(|| ProviderContractError::new("legacy source has no launch template line"))?,
        )?;
        if shape_values.len() < 3 || tile_values.len() < 3 || swizzle_values.len() < 3 || thread_values.len() < 2 {
            return Err(ProviderContractError::new("legacy DeepGEMM template descriptor is incomplete"));
        }
        let shared_memory_bytes = u32_after(
            source,
            "cudaFuncAttributeMaxDynamicSharedMemorySize,",
            "legacy source has no dynamic shared-memory size",
        )?;
        let cluster_n = if cluster == 2 { 2 } else { 1 };
        let evidence = Self {
            provider_revision,
            numerical_abi,
            shape: Fp8ProjectionShape { output_features: shape_values[1], input_features: shape_values[2] },
            tile: DeepGemmTile {
                block_m: tile_values[0],
                block_n: tile_values[1],
                block_k: tile_values[2],
                cluster_m: 1,
                cluster_n,
                swizzle_d: swizzle_values[2],
                stages,
                shared_memory_bytes,
                math_threads: thread_values[1],
            },
            num_sms,
        };
        if evidence.tile.cluster_m * evidence.tile.cluster_n != cluster {
            return Err(ProviderContractError::new("legacy source uses an unsupported cluster decomposition"));
        }
        Ok(evidence)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DeepGemmSm90Capability {
    pub target: &'static str,
    pub compute_capability: [u8; 2],
    pub num_sms: u32,
    pub provider_revision: &'static str,
    pub numerical_abi: &'static str,
    pub scale_block: u32,
    pub scale_row_alignment: u32,
    pub quantizer_threads: u32,
    pub fp8_max_finite: u32,
    pub quantization_amax_floor: &'static str,
}

impl DeepGemmSm90Capability {
    pub const fn h20() -> Self {
        Self {
            target: "nvidia-h20-sm90a",
            compute_capability: [9, 0],
            num_sms: 78,
            provider_revision: DEEPGEMM_REVISION,
            numerical_abi: PACKED_ACTIVATION_ABI,
            scale_block: SCALE_BLOCK,
            scale_row_alignment: SCALE_ROW_ALIGNMENT,
            quantizer_threads: QUANTIZER_THREADS,
            fp8_max_finite: FP8_MAX_FINITE,
            quantization_amax_floor: QUANTIZATION_AMAX_FLOOR,
        }
    }

    pub fn layout(self, rows: u32, shape: Fp8ProjectionShape) -> Result<Fp8ProjectionLayout, ProviderContractError> {
        validate_shape(rows, shape)?;
        let aligned_rows = align(rows, SCALE_ROW_ALIGNMENT)?;
        let quantized_bytes = checked_mul(rows, shape.input_features, "quantized activation")?;
        let scale_elements = checked_mul(aligned_rows, shape.input_features / SCALE_BLOCK, "activation scales")?;
        let scale_bytes = checked_mul_u64(scale_elements, 4, "activation scale bytes")?;
        let total_bytes = quantized_bytes
            .checked_add(scale_bytes)
            .ok_or_else(|| ProviderContractError::new("packed activation byte count overflows u64"))?;
        if quantized_bytes % u64::from(TMA_ALIGNMENT_BYTES) != 0 {
            return Err(ProviderContractError::new("activation scale plane is not 16-byte aligned"));
        }
        Ok(Fp8ProjectionLayout {
            input_bf16: [rows, shape.input_features],
            weight_fp8_e4m3: [shape.output_features, shape.input_features],
            weight_scale_f32: [shape.output_features.div_ceil(SCALE_BLOCK), shape.input_features.div_ceil(SCALE_BLOCK)],
            quantized_activation_fp8_e4m3: [rows, shape.input_features],
            activation_scale_f32: [shape.input_features / SCALE_BLOCK, aligned_rows],
            output_bf16: [rows, shape.output_features],
            scratch: PackedActivationScratch {
                quantized_bytes,
                scale_offset: quantized_bytes,
                scale_bytes,
                total_bytes,
            },
        })
    }

    /// Produce legal candidates in deterministic heuristic order. This order
    /// is only a compilation prior; qualification must measure the candidates.
    pub fn legal_candidates(
        self,
        row_limit: u32,
        shape: Fp8ProjectionShape,
    ) -> Result<Vec<DeepGemmKernelContract>, ProviderContractError> {
        let layout = self.layout(row_limit, shape)?;
        let mut block_ms = vec![64, 128];
        block_ms.extend([16, 32].into_iter().filter(|block_m| row_limit <= *block_m));
        block_ms.push(256);

        let mut candidates = Vec::new();
        for cluster_m in 1..=MAX_CLUSTER_SIZE {
            for cluster_n in 1..=MAX_CLUSTER_SIZE {
                let cluster_size = cluster_m * cluster_n;
                if cluster_size > MAX_CLUSTER_SIZE || !self.num_sms.is_multiple_of(cluster_size) {
                    continue;
                }
                for &block_m in &block_ms {
                    for block_n in (TILE_N_QUANTUM..=MAX_TILE_N).step_by(TILE_N_QUANTUM as usize) {
                        let Some(tile) = tile(shape, self.num_sms, block_m, block_n, cluster_m, cluster_n) else {
                            continue;
                        };
                        if block_m < WGMMA_M_ROWS && row_limit > block_m {
                            continue;
                        }
                        candidates.push((
                            estimated_cycles(tile, row_limit, shape, self.num_sms),
                            DeepGemmKernelContract {
                                contract_version: 1,
                                provider: "deepgemm".into(),
                                provider_revision: self.provider_revision.into(),
                                target: self.target.into(),
                                numerical_abi: self.numerical_abi.into(),
                                row_limit,
                                shape,
                                layout,
                                tile,
                                num_sms: self.num_sms,
                                qualification: QualificationState::Legal,
                            },
                        ));
                    }
                }
            }
        }
        candidates.sort_by_key(|(cycles, contract)| {
            (*cycles, contract.tile.block_m, contract.tile.block_n, contract.tile.cluster_m, contract.tile.cluster_n)
        });
        Ok(candidates.into_iter().map(|(_, contract)| contract).collect())
    }

    pub fn preferred_candidate(
        self,
        row_limit: u32,
        shape: Fp8ProjectionShape,
    ) -> Result<DeepGemmKernelContract, ProviderContractError> {
        self.legal_candidates(row_limit, shape)?
            .into_iter()
            .next()
            .ok_or_else(|| ProviderContractError::new("no legal DeepGEMM SM90 tile for projection"))
    }

    pub fn validate_contract(self, contract: &DeepGemmKernelContract) -> Result<(), ProviderContractError> {
        if contract.contract_version != 1
            || contract.provider != "deepgemm"
            || contract.provider_revision != self.provider_revision
            || contract.numerical_abi != self.numerical_abi
            || contract.target != self.target
            || contract.num_sms != self.num_sms
        {
            return Err(ProviderContractError::new("kernel identity does not match this DeepGEMM capability"));
        }
        if contract.layout != self.layout(contract.row_limit, contract.shape)? {
            return Err(ProviderContractError::new("kernel layout does not match its shape and row limit"));
        }
        if tile(
            contract.shape,
            self.num_sms,
            contract.tile.block_m,
            contract.tile.block_n,
            contract.tile.cluster_m,
            contract.tile.cluster_n,
        ) != Some(contract.tile)
            || (contract.tile.block_m < WGMMA_M_ROWS && contract.row_limit > contract.tile.block_m)
        {
            return Err(ProviderContractError::new("kernel tile is not legal for its shape and row limit"));
        }
        match &contract.qualification {
            QualificationState::Unqualified => {
                return Err(ProviderContractError::new("unqualified kernels cannot be admitted"));
            }
            QualificationState::Legal => {}
            QualificationState::Benchmarked { source_sha256, cubin_sha256, median_nanoseconds, samples }
                if *median_nanoseconds > 0
                    && *samples > 0
                    && validate_sha256(source_sha256).is_ok()
                    && validate_sha256(cubin_sha256).is_ok() => {}
            QualificationState::Benchmarked { .. } => {
                return Err(ProviderContractError::new("benchmark evidence must contain non-zero timing and samples"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderContractError(String);

impl ProviderContractError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProviderContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProviderContractError {}

fn validate_shape(rows: u32, shape: Fp8ProjectionShape) -> Result<(), ProviderContractError> {
    if rows == 0 || rows > MAX_QUANTIZER_ROWS {
        return Err(ProviderContractError::new("DeepGEMM rows exceed quantizer launch bounds"));
    }
    if shape.output_features == 0 || shape.input_features == 0 {
        return Err(ProviderContractError::new("DeepGEMM dimensions must be positive"));
    }
    if !shape.output_features.is_multiple_of(TMA_ALIGNMENT_BYTES / DType::Bf16.bytes() as u32) {
        return Err(ProviderContractError::new("BF16 output row stride must be 16-byte aligned"));
    }
    if !shape.input_features.is_multiple_of(SCALE_BLOCK) {
        return Err(ProviderContractError::new("DeepGEMM K must be a multiple of 128"));
    }
    Ok(())
}

fn tile(
    shape: Fp8ProjectionShape,
    num_sms: u32,
    block_m: u32,
    block_n: u32,
    cluster_m: u32,
    cluster_n: u32,
) -> Option<DeepGemmTile> {
    if ![16, 32, 64, 128, 256].contains(&block_m)
        || !(TILE_N_QUANTUM..=MAX_TILE_N).contains(&block_n)
        || !block_n.is_multiple_of(TILE_N_QUANTUM)
        || !(1..=MAX_CLUSTER_SIZE).contains(&cluster_m)
        || !(1..=MAX_CLUSTER_SIZE).contains(&cluster_n)
        || cluster_m * cluster_n > MAX_CLUSTER_SIZE
        || !num_sms.is_multiple_of(cluster_m * cluster_n)
    {
        return None;
    }
    let block_k = SCALE_BLOCK;
    if block_n > block_k && !block_n.is_multiple_of(block_n - block_k) && !block_k.is_multiple_of(block_n - block_k) {
        return None;
    }
    if block_m > 128 && block_n > 128 {
        return None;
    }
    let swizzle_d =
        [128, 64, 32, 16].into_iter().find(|mode| (block_n * DType::Bf16.bytes() as u32).is_multiple_of(*mode))?;
    let smem_d =
        align(block_m.checked_mul(block_n)?.checked_mul(DType::Bf16.bytes() as u32)?, SHARED_STORE_ALIGNMENT).ok()?;
    let smem_barriers = MAX_PIPELINE_STAGES * BARRIER_BYTES * BARRIERS_PER_STAGE;
    let smem_a = block_m.checked_mul(block_k)?;
    let smem_b = block_n.checked_mul(block_k)?;
    let smem_sfa = align(block_m.checked_mul(DType::F32.bytes() as u32)?, SHARED_SCALE_ALIGNMENT).ok()?;
    let uniform_sfb = u32::from(!block_k.is_multiple_of(block_n)) + 1;
    let smem_sfb = align(
        shape.input_features.div_ceil(block_k).checked_mul(DType::F32.bytes() as u32)?.checked_mul(uniform_sfb)?,
        BARRIER_BYTES,
    )
    .ok()?;
    let extra = smem_d.checked_add(smem_barriers)?.checked_add(smem_sfb)?;
    let per_stage = smem_a.checked_add(smem_b)?.checked_add(smem_sfa)?;
    let stages = ((SM90_SHARED_MEMORY_BYTES.saturating_sub(extra)) / per_stage).min(MAX_PIPELINE_STAGES);
    if stages < MIN_PIPELINE_STAGES || (block_m * block_n < LARGE_TILE_ELEMENTS && stages < SMALL_TILE_MIN_STAGES) {
        return None;
    }
    Some(DeepGemmTile {
        block_m,
        block_n,
        block_k,
        cluster_m,
        cluster_n,
        swizzle_d,
        stages,
        shared_memory_bytes: extra + stages * per_stage,
        math_threads: if block_m <= WGMMA_M_ROWS { WARP_GROUP_THREADS } else { 2 * WARP_GROUP_THREADS },
    })
}

fn estimated_cycles(tile: DeepGemmTile, rows: u32, shape: Fp8ProjectionShape, num_sms: u32) -> u64 {
    let blocks = u64::from(rows.div_ceil(tile.block_m)) * u64::from(shape.output_features.div_ceil(tile.block_n));
    let waves = blocks.div_ceil(u64::from(num_sms));
    let l2_bandwidth =
        (L2_BYTES_PER_SM_CYCLE * u64::from(num_sms)).min((NOMINAL_L2_BYTES_PER_MICROSECOND / NOMINAL_CLOCK_MHZ) as u64);
    let l1_bandwidth = L1_BYTES_PER_SM_CYCLE * u64::from(num_sms);
    let k = u64::from(shape.input_features);
    let block_m = u64::from(tile.block_m);
    let block_n = u64::from(tile.block_n);
    let l2_bytes = k * (block_m / u64::from(tile.cluster_n) + block_n / u64::from(tile.cluster_m))
        + block_m * block_n * DType::Bf16.bytes();
    let l1_bytes = k * (block_m + block_n)
        + k * (u64::from(WGMMA_M_ROWS).max(block_m) + block_n)
        + block_m * block_n * DType::Bf16.bytes();
    let cycles = (u128::from(l2_bytes) * u128::from(blocks) / u128::from(l2_bandwidth))
        .max(u128::from(l1_bytes) * u128::from(blocks) / u128::from(l1_bandwidth));
    if tile.cluster_m * tile.cluster_n > 1 && waves <= 1 {
        return u64::MAX;
    }
    let efficiency = blocks as f64 / (waves * u64::from(num_sms)) as f64;
    (cycles as f64 / efficiency) as u64
}

fn align(value: u32, alignment: u32) -> Result<u32, ProviderContractError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or_else(|| ProviderContractError::new("alignment calculation overflows u32"))
}

fn checked_mul(left: u32, right: u32, name: &str) -> Result<u64, ProviderContractError> {
    u64::from(left)
        .checked_mul(u64::from(right))
        .ok_or_else(|| ProviderContractError::new(format!("{name} byte count overflows u64")))
}

fn checked_mul_u64(left: u64, right: u64, name: &str) -> Result<u64, ProviderContractError> {
    left.checked_mul(right).ok_or_else(|| ProviderContractError::new(format!("{name} byte count overflows u64")))
}

fn validate_sha256(digest: &str) -> Result<(), ProviderContractError> {
    if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        Ok(())
    } else {
        Err(ProviderContractError::new("artifact SHA-256 must be 64 lowercase hexadecimal characters"))
    }
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn quoted_after(source: &str, marker: &str) -> Result<String, ProviderContractError> {
    let suffix = source
        .split_once(marker)
        .map(|(_, suffix)| suffix)
        .ok_or_else(|| ProviderContractError::new("legacy source has no provider revision"))?;
    let start =
        suffix.find('\"').ok_or_else(|| ProviderContractError::new("legacy provider revision is not quoted"))? + 1;
    let end = suffix[start..]
        .find('\"')
        .ok_or_else(|| ProviderContractError::new("legacy provider revision quote is not closed"))?
        + start;
    Ok(suffix[start..end].to_owned())
}

fn csv_u32(line: &str) -> Result<Vec<u32>, ProviderContractError> {
    line.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value.parse().map_err(|_| ProviderContractError::new(format!("legacy template value `{value}` is not u32")))
        })
        .collect()
}

fn first_u32(line: &str) -> Result<u32, ProviderContractError> {
    line.trim_end_matches(',')
        .trim()
        .parse()
        .map_err(|_| ProviderContractError::new(format!("legacy template value `{line}` is not u32")))
}

fn first_u32_before_comma(line: &str) -> Result<u32, ProviderContractError> {
    line.split(',')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ProviderContractError::new("legacy template line has no integer"))?
        .parse()
        .map_err(|_| ProviderContractError::new(format!("legacy template value `{line}` does not start with u32")))
}

fn u32_after(source: &str, marker: &str, error: &str) -> Result<u32, ProviderContractError> {
    let suffix =
        source.split_once(marker).map(|(_, suffix)| suffix).ok_or_else(|| ProviderContractError::new(error))?;
    let value = suffix
        .split(|character: char| !character.is_ascii_digit())
        .find(|value| !value.is_empty())
        .ok_or_else(|| ProviderContractError::new(error))?;
    value.parse().map_err(|_| ProviderContractError::new(error))
}
