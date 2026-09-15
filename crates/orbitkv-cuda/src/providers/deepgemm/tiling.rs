//! SM90 1D2D kernel legality and the initial ordering of tile candidates.
//!
//! This is a port of the pinned provider's heuristic, not measured device
//! performance. The graph search profiles candidates after this initial
//! ordering. Hardware constraints and heuristic priors live here; JIT source
//! generation and loading are separate in `jit`.

use serde::{Deserialize, Serialize};

use super::contract::{self, BF16_BYTES, SCALE_BLOCK, SCALE_BYTES};

// SM90 kernel/ABI requirements used by the pinned 1D2D implementation.
const MAX_CLUSTER_SIZE: usize = 2;
const SM90_SHARED_MEMORY_BYTES: usize = 232_448;
const WARP_GROUP_THREADS: usize = 128;
const WGMMA_M_ROWS: usize = 64;
const SHARED_STORE_ALIGNMENT: usize = 1024;
const SHARED_SCALE_ALIGNMENT: usize = 128;
const BARRIER_BYTES: usize = std::mem::size_of::<u64>();
const BARRIERS_PER_STAGE: usize = 2; // Producer/consumer arrival barriers.

// Upstream search priors, used only when egglog generates candidates.
// These are assumptions for ordering, not hardware measurements or legality.
// In particular, the nominal clock/bandwidth are not queried device properties.
const MAX_PIPELINE_STAGES: usize = 16;
const TILE_N_QUANTUM: usize = 16;
const MAX_TILE_N: usize = 192;
const BASE_M_TILES: [usize; 2] = [64, 128];
const SMALL_M_TILES: [usize; 2] = [16, 32];
const LARGE_M_TILE: usize = 256;
const MIN_PIPELINE_STAGES: usize = 3;
const SMALL_TILE_MIN_STAGES: usize = 4;
const LARGE_TILE_ELEMENTS: usize = 128 * 192;
const L2_BYTES_PER_SM_CYCLE: usize = 64;
const L1_BYTES_PER_SM_CYCLE: usize = 128;
const NOMINAL_L2_BYTES_PER_MICROSECOND: f64 = 8_000_000.0;
const NOMINAL_CLOCK_MHZ: f64 = 1_300.0;

#[derive(Clone, Copy, Debug, Default, Deserialize, Hash, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub(super) output_features: usize,
    pub(super) input_features: usize,
    pub(super) block_m: usize,
    pub(super) block_n: usize,
    pub(super) block_k: usize,
    pub(super) cluster_m: usize,
    pub(super) cluster_n: usize,
    pub(super) swizzle_d: usize,
    pub(super) stages: usize,
    pub(super) smem_bytes: usize,
    pub(super) math_threads: usize,
    pub(super) num_sms: usize,
}

impl Config {
    pub(super) fn validate_shape(m: usize, n: usize, k: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            m > 0 && n > 0 && k > 0,
            "DeepGEMM dimensions must be positive"
        );
        let output_row_alignment = contract::TMA_ALIGNMENT_BYTES / BF16_BYTES;
        anyhow::ensure!(
            n.is_multiple_of(output_row_alignment),
            "DeepGEMM BF16 output row stride must be TMA aligned"
        );
        anyhow::ensure!(
            k.is_multiple_of(SCALE_BLOCK),
            "DeepGEMM K must be a multiple of 128"
        );
        anyhow::ensure!(
            m <= contract::MAX_QUANTIZER_ROWS && n <= i32::MAX as usize && k <= i32::MAX as usize,
            "DeepGEMM dimensions exceed the quantizer launch or C ABI limits"
        );
        Ok(())
    }

    /// Validate a serialized tile without re-running candidate ranking.
    pub(super) fn validate(self, row_limit: usize) -> anyhow::Result<()> {
        Self::validate_shape(row_limit, self.output_features, self.input_features)?;
        anyhow::ensure!(
            self.num_sms > 0 && self.num_sms <= i32::MAX as usize,
            "DeepGEMM requires a positive SM count"
        );
        anyhow::ensure!(
            tile_config(
                self.output_features,
                self.input_features,
                self.num_sms,
                self.block_m,
                self.block_n,
                self.cluster_m,
                self.cluster_n,
            ) == Some(self),
            "invalid serialized DeepGEMM tile: {self:?}"
        );
        anyhow::ensure!(
            self.block_m >= WGMMA_M_ROWS || row_limit <= self.block_m,
            "DeepGEMM small-M tile exceeds its admitted row limit"
        );
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct RankedConfig {
    config: Config,
    cycles: u64,
}

pub(super) fn candidates(m: usize, n: usize, k: usize, num_sms: usize) -> Vec<Config> {
    if Config::validate_shape(m, n, k).is_err() || num_sms == 0 || num_sms > i32::MAX as usize {
        return vec![];
    }
    let mut block_ms = BASE_M_TILES.to_vec();
    block_ms.extend(SMALL_M_TILES.into_iter().filter(|tile| m <= *tile));
    block_ms.push(LARGE_M_TILE);

    let mut ranked = Vec::new();
    for cluster_m in 1_usize..=MAX_CLUSTER_SIZE {
        for cluster_n in 1_usize..=MAX_CLUSTER_SIZE {
            let cluster = cluster_m * cluster_n;
            if cluster > MAX_CLUSTER_SIZE || !num_sms.is_multiple_of(cluster) {
                continue;
            }
            for &block_m in &block_ms {
                for block_n in (TILE_N_QUANTUM..=MAX_TILE_N).step_by(TILE_N_QUANTUM) {
                    let Some(config) =
                        tile_config(n, k, num_sms, block_m, block_n, cluster_m, cluster_n)
                    else {
                        continue;
                    };
                    ranked.push(RankedConfig {
                        cycles: estimated_cycles(config, m, n, k),
                        config,
                    });
                }
            }
        }
    }
    ranked.sort_by_key(|candidate| candidate.cycles);
    ranked
        .into_iter()
        .map(|candidate| candidate.config)
        .collect()
}

fn tile_config(
    n: usize,
    k: usize,
    num_sms: usize,
    block_m: usize,
    block_n: usize,
    cluster_m: usize,
    cluster_n: usize,
) -> Option<Config> {
    if !(SMALL_M_TILES.contains(&block_m)
        || BASE_M_TILES.contains(&block_m)
        || block_m == LARGE_M_TILE)
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
    if block_n > block_k
        && !block_n.is_multiple_of(block_n - block_k)
        && !block_k.is_multiple_of(block_n - block_k)
    {
        return None;
    }
    if block_m > 128 && block_n > 128 {
        return None;
    }
    let swizzle_d = [128, 64, 32, 16]
        .into_iter()
        .find(|mode| (block_n * BF16_BYTES).is_multiple_of(*mode))
        .unwrap();
    let smem_d = align(block_m * block_n * BF16_BYTES, SHARED_STORE_ALIGNMENT);
    let smem_barriers = MAX_PIPELINE_STAGES * BARRIER_BYTES * BARRIERS_PER_STAGE;
    let smem_a = block_m * block_k;
    let smem_b = block_n * block_k;
    let smem_sfa = align(block_m * SCALE_BYTES, SHARED_SCALE_ALIGNMENT);
    let uniform_sfb = usize::from(!block_k.is_multiple_of(block_n)) + 1;
    let smem_sfb = align(
        k.div_ceil(block_k) * SCALE_BYTES * uniform_sfb,
        BARRIER_BYTES,
    );
    let extra = smem_d + smem_barriers + smem_sfb;
    let per_stage = smem_a + smem_b + smem_sfa;
    let stages =
        ((SM90_SHARED_MEMORY_BYTES.saturating_sub(extra)) / per_stage).min(MAX_PIPELINE_STAGES);
    if stages < MIN_PIPELINE_STAGES
        || (block_m * block_n < LARGE_TILE_ELEMENTS && stages < SMALL_TILE_MIN_STAGES)
    {
        return None;
    }
    Some(Config {
        output_features: n,
        input_features: k,
        block_m,
        block_n,
        block_k,
        cluster_m,
        cluster_n,
        swizzle_d,
        stages,
        smem_bytes: extra + stages * per_stage,
        math_threads: if block_m <= WGMMA_M_ROWS {
            WARP_GROUP_THREADS
        } else {
            2 * WARP_GROUP_THREADS
        },
        num_sms,
    })
}

fn estimated_cycles(config: Config, m: usize, n: usize, k: usize) -> u64 {
    let blocks = m.div_ceil(config.block_m) * n.div_ceil(config.block_n);
    let waves = blocks.div_ceil(config.num_sms);
    let l2_bandwidth = (L2_BYTES_PER_SM_CYCLE * config.num_sms)
        .min((NOMINAL_L2_BYTES_PER_MICROSECOND / NOMINAL_CLOCK_MHZ) as usize);
    let l1_bandwidth = L1_BYTES_PER_SM_CYCLE * config.num_sms;
    let l2_bytes = k * (config.block_m / config.cluster_n + config.block_n / config.cluster_m)
        + config.block_m * config.block_n * BF16_BYTES;
    let l1_bytes = k * (config.block_m + config.block_n)
        + k * (WGMMA_M_ROWS.max(config.block_m) + config.block_n)
        + config.block_m * config.block_n * BF16_BYTES;
    let cycles = (l2_bytes as u128 * blocks as u128 / l2_bandwidth as u128)
        .max(l1_bytes as u128 * blocks as u128 / l1_bandwidth as u128);
    if config.cluster_m * config.cluster_n > 1 && waves <= 1 {
        return u64::MAX;
    }
    let efficiency = blocks as f64 / (waves * config.num_sms) as f64;
    (cycles as f64 / efficiency) as u64
}

fn align(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}
