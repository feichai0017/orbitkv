//! cuda-oxide provider for BF16 single-request decode attention.

use crate::command::{CommandError, CommandPermit, CommandScope, Read, ReadWrite, Write};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, convert, cuda_module, float, kernel, launch_bounds, launch_contract, tcgen05,
    thread, warp,
};
use half::bf16;
use loom_infer::{
    Bf16SingleDecodeSpec, Bf16SingleDecodeSplitKSpec, SINGLE_DECODE_HEAD_DIM,
    SINGLE_DECODE_PARTIAL_STATE_WIDTH,
};
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

const WARP_THREADS: u32 = 32;
const BF16_PAIRS_PER_HEAD: usize = SINGLE_DECODE_HEAD_DIM / 2;
const BF16_PAIRS_PER_LANE: usize = BF16_PAIRS_PER_HEAD / WARP_THREADS as usize;

const _: () = {
    assert!(SINGLE_DECODE_HEAD_DIM == 128);
    assert!(BF16_PAIRS_PER_LANE == 2);
    assert!(core::mem::size_of::<bf16>() == core::mem::size_of::<u16>());
    assert!(core::mem::align_of::<bf16>() == core::mem::align_of::<u16>());
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(32)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            kv_len >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == num_query_heads * 128,
            key.len() == kv_len * num_kv_heads * 128,
            value.len() == kv_len * num_kv_heads * 128,
            output.len() == num_query_heads * 128,
            lse.len() == num_query_heads,
        ),
    )]
    pub fn single_decode_bf16_nhd(
        kv_len: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let query_head = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        if query_head >= num_query_heads || lane >= WARP_THREADS as usize {
            return;
        }

        let group_size = num_query_heads / num_kv_heads;
        let kv_head = query_head / group_size;
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key.as_ptr().cast::<u32>();
        let value_pairs = value.as_ptr().cast::<u32>();
        let output_pairs = output.as_mut_ptr().cast::<u32>();
        let first_pair = query_head * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;

        // SAFETY: the host plan validates four-byte alignment. The launch
        // contract proves both packed query reads are inside the exact span.
        let (query_0, query_1, query_2, query_3) = unsafe {
            let (query_0, query_1) = convert::cvt_f32x2_bf16x2(query_pairs.add(first_pair).read());
            let (query_2, query_3) = convert::cvt_f32x2_bf16x2(query_pairs.add(second_pair).read());
            (query_0, query_1, query_2, query_3)
        };

        let mut output_0 = 0.0_f32;
        let mut output_1 = 0.0_f32;
        let mut output_2 = 0.0_f32;
        let mut output_3 = 0.0_f32;
        let mut max_score_log2 = 0.0_f32;
        let mut normalizer = 0.0_f32;
        let mut token = 0_usize;

        while token < kv_len {
            let kv_pair_offset = (token * num_kv_heads + kv_head) * BF16_PAIRS_PER_HEAD + lane;
            // SAFETY: packed NHD offsets cover two disjoint pairs per lane.
            // Exact spans and four-byte base alignment were checked on host.
            let (key_0, key_1, key_2, key_3, value_0, value_1, value_2, value_3) = unsafe {
                let (key_0, key_1) =
                    convert::cvt_f32x2_bf16x2(key_pairs.add(kv_pair_offset).read());
                let (key_2, key_3) = convert::cvt_f32x2_bf16x2(
                    key_pairs.add(kv_pair_offset + WARP_THREADS as usize).read(),
                );
                let (value_0, value_1) =
                    convert::cvt_f32x2_bf16x2(value_pairs.add(kv_pair_offset).read());
                let (value_2, value_3) = convert::cvt_f32x2_bf16x2(
                    value_pairs
                        .add(kv_pair_offset + WARP_THREADS as usize)
                        .read(),
                );
                (
                    key_0, key_1, key_2, key_3, value_0, value_1, value_2, value_3,
                )
            };

            let mut dot = 0.0_f32;
            dot = float::fma_rn_f32(query_0, key_0, dot);
            dot = float::fma_rn_f32(query_1, key_1, dot);
            dot = float::fma_rn_f32(query_2, key_2, dot);
            dot = float::fma_rn_f32(query_3, key_3, dot);
            let score_log2 = warp::reduce_sum_f32(dot) * softmax_scale_log2;

            let mut previous_weight = 0.0_f32;
            let mut current_weight = 0.0_f32;
            if lane == 0 {
                if token == 0 {
                    max_score_log2 = score_log2;
                    normalizer = 1.0;
                    current_weight = 1.0;
                } else {
                    let next_max = f32::max(max_score_log2, score_log2);
                    previous_weight = float::ex2_approx_f32(max_score_log2 - next_max);
                    current_weight = float::ex2_approx_f32(score_log2 - next_max);
                    normalizer = normalizer * previous_weight + current_weight;
                    max_score_log2 = next_max;
                }
            }
            previous_weight = warp::shuffle_f32(previous_weight, 0);
            current_weight = warp::shuffle_f32(current_weight, 0);

            output_0 = float::fma_rn_f32(value_0, current_weight, output_0 * previous_weight);
            output_1 = float::fma_rn_f32(value_1, current_weight, output_1 * previous_weight);
            output_2 = float::fma_rn_f32(value_2, current_weight, output_2 * previous_weight);
            output_3 = float::fma_rn_f32(value_3, current_weight, output_3 * previous_weight);
            token += 1;
        }

        let mut inverse_normalizer = 0.0_f32;
        if lane == 0 {
            inverse_normalizer = float::div_rn_f32(1.0, normalizer);
            // SAFETY: only lane zero writes this query-head slot.
            unsafe {
                *lse.get_unchecked_mut(query_head) =
                    max_score_log2 + float::lg2_approx_f32(normalizer);
            }
        }
        inverse_normalizer = warp::shuffle_f32(inverse_normalizer, 0);

        // SAFETY: each lane owns two packed output pairs. The output base is
        // four-byte aligned and the launch contract proves the exact span.
        unsafe {
            output_pairs
                .add(first_pair)
                .write(tcgen05::cvt_f32x2_bf16x2(
                    output_0 * inverse_normalizer,
                    output_1 * inverse_normalizer,
                ));
            output_pairs
                .add(second_pair)
                .write(tcgen05::cvt_f32x2_bf16x2(
                    output_2 * inverse_normalizer,
                    output_3 * inverse_normalizer,
                ));
        }
    }

    #[kernel]
    #[launch_bounds(32)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            kv_len >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            partitions >= 1,
            partitions <= kv_len,
            query.len() == num_query_heads * 128,
            key.len() == kv_len * num_kv_heads * 128,
            value.len() == kv_len * num_kv_heads * 128,
            workspace.len() == num_query_heads * partitions * 130,
        ),
    )]
    pub fn single_decode_bf16_nhd_split_k_partials(
        kv_len: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        partitions: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        mut workspace: DisjointSlice<f32>,
    ) {
        let state_index = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let state_count = num_query_heads * partitions;
        if state_index >= state_count || lane >= WARP_THREADS as usize {
            return;
        }

        let query_head = state_index / partitions;
        let partition = state_index % partitions;
        let group_size = num_query_heads / num_kv_heads;
        let kv_head = query_head / group_size;
        let base_tokens = kv_len / partitions;
        let remainder = kv_len % partitions;
        let extra_before = if partition < remainder {
            partition
        } else {
            remainder
        };
        let token_start = partition * base_tokens + extra_before;
        let token_end = token_start + base_tokens + if partition < remainder { 1 } else { 0 };
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key.as_ptr().cast::<u32>();
        let value_pairs = value.as_ptr().cast::<u32>();
        let first_pair = query_head * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;

        // SAFETY: the host plan validates four-byte alignment. The launch
        // contract proves both packed query reads are inside the exact span.
        let (query_0, query_1, query_2, query_3) = unsafe {
            let (query_0, query_1) = convert::cvt_f32x2_bf16x2(query_pairs.add(first_pair).read());
            let (query_2, query_3) = convert::cvt_f32x2_bf16x2(query_pairs.add(second_pair).read());
            (query_0, query_1, query_2, query_3)
        };

        let mut output_0 = 0.0_f32;
        let mut output_1 = 0.0_f32;
        let mut output_2 = 0.0_f32;
        let mut output_3 = 0.0_f32;
        let mut max_score_log2 = 0.0_f32;
        let mut normalizer = 0.0_f32;
        let mut token = token_start;
        while token < token_end {
            let kv_pair_offset = (token * num_kv_heads + kv_head) * BF16_PAIRS_PER_HEAD + lane;
            // SAFETY: packed NHD offsets cover two disjoint pairs per lane.
            // Exact spans and four-byte base alignment were checked on host.
            let (key_0, key_1, key_2, key_3, value_0, value_1, value_2, value_3) = unsafe {
                let (key_0, key_1) =
                    convert::cvt_f32x2_bf16x2(key_pairs.add(kv_pair_offset).read());
                let (key_2, key_3) = convert::cvt_f32x2_bf16x2(
                    key_pairs.add(kv_pair_offset + WARP_THREADS as usize).read(),
                );
                let (value_0, value_1) =
                    convert::cvt_f32x2_bf16x2(value_pairs.add(kv_pair_offset).read());
                let (value_2, value_3) = convert::cvt_f32x2_bf16x2(
                    value_pairs
                        .add(kv_pair_offset + WARP_THREADS as usize)
                        .read(),
                );
                (
                    key_0, key_1, key_2, key_3, value_0, value_1, value_2, value_3,
                )
            };

            let mut dot = 0.0_f32;
            dot = float::fma_rn_f32(query_0, key_0, dot);
            dot = float::fma_rn_f32(query_1, key_1, dot);
            dot = float::fma_rn_f32(query_2, key_2, dot);
            dot = float::fma_rn_f32(query_3, key_3, dot);
            let score_log2 = warp::reduce_sum_f32(dot) * softmax_scale_log2;

            let mut previous_weight = 0.0_f32;
            let mut current_weight = 0.0_f32;
            if lane == 0 {
                if token == token_start {
                    max_score_log2 = score_log2;
                    normalizer = 1.0;
                    current_weight = 1.0;
                } else {
                    let next_max = f32::max(max_score_log2, score_log2);
                    previous_weight = float::ex2_approx_f32(max_score_log2 - next_max);
                    current_weight = float::ex2_approx_f32(score_log2 - next_max);
                    normalizer = normalizer * previous_weight + current_weight;
                    max_score_log2 = next_max;
                }
            }
            previous_weight = warp::shuffle_f32(previous_weight, 0);
            current_weight = warp::shuffle_f32(current_weight, 0);

            output_0 = float::fma_rn_f32(value_0, current_weight, output_0 * previous_weight);
            output_1 = float::fma_rn_f32(value_1, current_weight, output_1 * previous_weight);
            output_2 = float::fma_rn_f32(value_2, current_weight, output_2 * previous_weight);
            output_3 = float::fma_rn_f32(value_3, current_weight, output_3 * previous_weight);
            token += 1;
        }

        let state_offset = state_index * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
        // SAFETY: each block owns one state, lane zero owns its two header
        // slots, and every lane owns four distinct weighted-value slots.
        unsafe {
            if lane == 0 {
                *workspace.get_unchecked_mut(state_offset) = max_score_log2;
                *workspace.get_unchecked_mut(state_offset + 1) = normalizer;
            }
            *workspace.get_unchecked_mut(state_offset + 2 + lane * 2) = output_0;
            *workspace.get_unchecked_mut(state_offset + 3 + lane * 2) = output_1;
            *workspace
                .get_unchecked_mut(state_offset + 2 + SINGLE_DECODE_HEAD_DIM / 2 + lane * 2) =
                output_2;
            *workspace
                .get_unchecked_mut(state_offset + 3 + SINGLE_DECODE_HEAD_DIM / 2 + lane * 2) =
                output_3;
        }
    }

    #[kernel]
    #[launch_bounds(32)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            num_query_heads >= 1,
            partitions >= 1,
            workspace.len() == num_query_heads * partitions * 130,
            output.len() == num_query_heads * 128,
            lse.len() == num_query_heads,
        ),
    )]
    pub fn single_decode_bf16_nhd_split_k_merge(
        num_query_heads: usize,
        partitions: usize,
        workspace: &[f32],
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let query_head = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        if query_head >= num_query_heads || lane >= WARP_THREADS as usize {
            return;
        }

        let first_component = lane * 2;
        let second_component = SINGLE_DECODE_HEAD_DIM / 2 + lane * 2;
        let mut output_0 = 0.0_f32;
        let mut output_1 = 0.0_f32;
        let mut output_2 = 0.0_f32;
        let mut output_3 = 0.0_f32;
        let mut merged_max_log2 = 0.0_f32;
        let mut merged_normalizer = 0.0_f32;
        let mut partition = 0_usize;
        while partition < partitions {
            let state_offset =
                (query_head * partitions + partition) * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
            let partition_max_log2 = workspace[state_offset];
            let partition_normalizer = workspace[state_offset + 1];
            let value_0 = workspace[state_offset + 2 + first_component];
            let value_1 = workspace[state_offset + 3 + first_component];
            let value_2 = workspace[state_offset + 2 + second_component];
            let value_3 = workspace[state_offset + 3 + second_component];

            if partition == 0 {
                merged_max_log2 = partition_max_log2;
                merged_normalizer = partition_normalizer;
                output_0 = value_0;
                output_1 = value_1;
                output_2 = value_2;
                output_3 = value_3;
            } else {
                let next_max = f32::max(merged_max_log2, partition_max_log2);
                let merged_weight = float::ex2_approx_f32(merged_max_log2 - next_max);
                let partition_weight = float::ex2_approx_f32(partition_max_log2 - next_max);
                merged_normalizer =
                    merged_normalizer * merged_weight + partition_normalizer * partition_weight;
                output_0 = float::fma_rn_f32(value_0, partition_weight, output_0 * merged_weight);
                output_1 = float::fma_rn_f32(value_1, partition_weight, output_1 * merged_weight);
                output_2 = float::fma_rn_f32(value_2, partition_weight, output_2 * merged_weight);
                output_3 = float::fma_rn_f32(value_3, partition_weight, output_3 * merged_weight);
                merged_max_log2 = next_max;
            }
            partition += 1;
        }

        let inverse_normalizer = float::div_rn_f32(1.0, merged_normalizer);
        if lane == 0 {
            // SAFETY: only lane zero writes this query-head slot.
            unsafe {
                *lse.get_unchecked_mut(query_head) =
                    merged_max_log2 + float::lg2_approx_f32(merged_normalizer);
            }
        }

        let output_pairs = output.as_mut_ptr().cast::<u32>();
        let first_pair = query_head * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;
        // SAFETY: each lane owns two packed output pairs. The output base is
        // four-byte aligned and the launch contract proves the exact span.
        unsafe {
            output_pairs
                .add(first_pair)
                .write(tcgen05::cvt_f32x2_bf16x2(
                    output_0 * inverse_normalizer,
                    output_1 * inverse_normalizer,
                ));
            output_pairs
                .add(second_pair)
                .write(tcgen05::cvt_f32x2_bf16x2(
                    output_2 * inverse_normalizer,
                    output_3 * inverse_normalizer,
                ));
        }
    }
}

/// Loaded cuda-oxide module for attention kernels.
#[derive(Clone, Debug)]
pub struct AttentionProvider {
    module: kernels::LoadedModule,
}

impl AttentionProvider {
    /// Loads the embedded attention artifact into one CUDA context.
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, cuda_host::EmbeddedModuleError> {
        // SAFETY: this crate owns the package-named device bundle and the
        // inline module defines the admitted attention entry point.
        let module = unsafe { kernels::load(context)? };
        Ok(Self { module })
    }

    /// Creates one immutable BF16 NHD single-decode launch plan.
    pub fn plan_bf16(
        &self,
        spec: Bf16SingleDecodeSpec,
    ) -> Result<Bf16SingleDecodePlan, SingleDecodePlanError> {
        let query_heads = u32::try_from(spec.num_query_heads())
            .map_err(|_| SingleDecodePlanError::QueryHeadCountOutOfRange(spec.num_query_heads()))?;
        let launch = self
            .module
            .prepare_single_decode_bf16_nhd(LaunchConfig1D::new(query_heads, WARP_THREADS, 0))?;
        Ok(Bf16SingleDecodePlan {
            spec,
            module: self.module.clone(),
            launch,
        })
    }

    /// Creates one immutable split-K BF16 NHD launch plan.
    pub fn plan_bf16_split_k(
        &self,
        spec: Bf16SingleDecodeSplitKSpec,
    ) -> Result<Bf16SingleDecodeSplitKPlan, SingleDecodePlanError> {
        let query_heads = u32::try_from(spec.decode().num_query_heads()).map_err(|_| {
            SingleDecodePlanError::QueryHeadCountOutOfRange(spec.decode().num_query_heads())
        })?;
        let partial_states = u32::try_from(spec.partial_state_count()).map_err(|_| {
            SingleDecodePlanError::PartialStateCountOutOfRange(spec.partial_state_count())
        })?;
        let partial_launch = self
            .module
            .prepare_single_decode_bf16_nhd_split_k_partials(LaunchConfig1D::new(
                partial_states,
                WARP_THREADS,
                0,
            ))?;
        let merge_launch =
            self.module
                .prepare_single_decode_bf16_nhd_split_k_merge(LaunchConfig1D::new(
                    query_heads,
                    WARP_THREADS,
                    0,
                ))?;
        Ok(Bf16SingleDecodeSplitKPlan {
            spec,
            module: self.module.clone(),
            partial_launch,
            merge_launch,
        })
    }
}

/// Immutable launch plan for the first single-decode contract.
#[derive(Clone)]
pub struct Bf16SingleDecodePlan {
    spec: Bf16SingleDecodeSpec,
    module: kernels::LoadedModule,
    launch: PreparedLaunch<kernels::__single_decode_bf16_nhd_CudaKernel>,
}

impl Bf16SingleDecodePlan {
    pub const fn spec(&self) -> Bf16SingleDecodeSpec {
        self.spec
    }

    /// Enqueues the fixed plan into a checked command scope.
    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16SingleDecodeArgs,
    ) -> Result<(), SingleDecodeEnqueueError> {
        let permit = scope.prepare_command()?;
        let (function, launch_result) = {
            let resolved =
                scope.resolve_rrrww(args.query, args.key, args.value, args.output, args.lse)?;
            for (operand, address) in [
                ("Q", resolved.first.cu_deviceptr()),
                ("K", resolved.second.cu_deviceptr()),
                ("V", resolved.third.cu_deviceptr()),
                ("O", resolved.fourth.cu_deviceptr()),
            ] {
                require_packed_alignment(operand, address)?;
            }
            let result = self.module.single_decode_bf16_nhd(
                resolved.stream,
                &self.launch,
                self.spec.kv_len(),
                self.spec.num_query_heads(),
                self.spec.num_kv_heads(),
                self.spec.softmax_scale() * core::f32::consts::LOG2_E,
                resolved.first,
                resolved.second,
                resolved.third,
                resolved.fourth,
                resolved.fifth,
            );
            (self.launch.function().clone(), result)
        };
        record_launch(scope, permit, function, launch_result)
    }
}

/// Immutable partial-state and merge launches for split-K single decode.
#[derive(Clone)]
pub struct Bf16SingleDecodeSplitKPlan {
    spec: Bf16SingleDecodeSplitKSpec,
    module: kernels::LoadedModule,
    partial_launch: PreparedLaunch<kernels::__single_decode_bf16_nhd_split_k_partials_CudaKernel>,
    merge_launch: PreparedLaunch<kernels::__single_decode_bf16_nhd_split_k_merge_CudaKernel>,
}

impl Bf16SingleDecodeSplitKPlan {
    pub const fn spec(&self) -> Bf16SingleDecodeSplitKSpec {
        self.spec
    }

    pub const fn workspace_required_numel(&self) -> usize {
        self.spec.workspace_numel()
    }

    pub const fn workspace_required_bytes(&self) -> usize {
        self.spec.workspace_bytes()
    }

    /// Enqueues partial-state and merge kernels into one checked scope.
    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16SingleDecodeSplitKArgs,
    ) -> Result<(), SingleDecodeEnqueueError> {
        scope.require_command_capacity(2)?;
        let partial_permit = scope.prepare_command()?;
        let (partial_function, partial_result) = {
            let resolved = scope.resolve_rrrwww(
                args.query,
                args.key,
                args.value,
                args.workspace.write(),
                args.output,
                args.lse,
            )?;
            let decode = self.spec.decode();
            require_exact_len("Q", resolved.first.len(), decode.query_numel())?;
            require_exact_len("K", resolved.second.len(), decode.kv_numel())?;
            require_exact_len("V", resolved.third.len(), decode.kv_numel())?;
            require_exact_len(
                "workspace",
                resolved.fourth.len(),
                self.spec.workspace_numel(),
            )?;
            require_exact_len("O", resolved.fifth.len(), decode.output_numel())?;
            require_exact_len("LSE", resolved.sixth.len(), decode.lse_numel())?;
            for (operand, address) in [
                ("Q", resolved.first.cu_deviceptr()),
                ("K", resolved.second.cu_deviceptr()),
                ("V", resolved.third.cu_deviceptr()),
                ("O", resolved.fifth.cu_deviceptr()),
            ] {
                require_packed_alignment(operand, address)?;
            }
            let result = self.module.single_decode_bf16_nhd_split_k_partials(
                resolved.stream,
                &self.partial_launch,
                decode.kv_len(),
                decode.num_query_heads(),
                decode.num_kv_heads(),
                self.spec.partitions(),
                decode.softmax_scale() * core::f32::consts::LOG2_E,
                resolved.first,
                resolved.second,
                resolved.third,
                resolved.fourth,
            );
            (self.partial_launch.function().clone(), result)
        };
        record_launch(scope, partial_permit, partial_function, partial_result)?;

        let merge_permit = scope.prepare_command()?;
        let (merge_function, merge_result) = {
            let resolved = scope.resolve_rww(args.workspace.read(), args.output, args.lse)?;
            let result = self.module.single_decode_bf16_nhd_split_k_merge(
                resolved.stream,
                &self.merge_launch,
                self.spec.decode().num_query_heads(),
                self.spec.partitions(),
                resolved.first,
                resolved.second,
                resolved.third,
            );
            (self.merge_launch.function().clone(), result)
        };
        record_launch(scope, merge_permit, merge_function, merge_result)
    }
}

/// Checked handles for one single-decode launch.
#[derive(Clone, Copy, Debug)]
pub struct Bf16SingleDecodeArgs {
    query: Read<bf16>,
    key: Read<bf16>,
    value: Read<bf16>,
    output: Write<bf16>,
    lse: Write<f32>,
}

impl Bf16SingleDecodeArgs {
    pub const fn new(
        query: Read<bf16>,
        key: Read<bf16>,
        value: Read<bf16>,
        output: Write<bf16>,
        lse: Write<f32>,
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
            lse,
        }
    }
}

/// Checked handles for one split-K partial plus merge command pair.
#[derive(Clone, Copy, Debug)]
pub struct Bf16SingleDecodeSplitKArgs {
    query: Read<bf16>,
    key: Read<bf16>,
    value: Read<bf16>,
    workspace: ReadWrite<f32>,
    output: Write<bf16>,
    lse: Write<f32>,
}

impl Bf16SingleDecodeSplitKArgs {
    pub const fn new(
        query: Read<bf16>,
        key: Read<bf16>,
        value: Read<bf16>,
        workspace: ReadWrite<f32>,
        output: Write<bf16>,
        lse: Write<f32>,
    ) -> Self {
        Self {
            query,
            key,
            value,
            workspace,
            output,
            lse,
        }
    }
}

#[derive(Debug, Error)]
pub enum SingleDecodePlanError {
    #[error("single-decode query-head count {0} exceeds the CUDA grid range")]
    QueryHeadCountOutOfRange(usize),
    #[error("single-decode partial-state count {0} exceeds the CUDA grid range")]
    PartialStateCountOutOfRange(usize),
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
}

#[derive(Debug, Error)]
pub enum SingleDecodeEnqueueError {
    #[error("{operand} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
    #[error(
        "packed single decode requires {operand} to be {alignment}-byte aligned, got {address:#x}"
    )]
    MisalignedBuffer {
        operand: &'static str,
        address: u64,
        alignment: u64,
    },
}

fn require_packed_alignment(
    operand: &'static str,
    address: u64,
) -> Result<(), SingleDecodeEnqueueError> {
    const ALIGNMENT: u64 = size_of::<u32>() as u64;
    if address.is_multiple_of(ALIGNMENT) {
        Ok(())
    } else {
        Err(SingleDecodeEnqueueError::MisalignedBuffer {
            operand,
            address,
            alignment: ALIGNMENT,
        })
    }
}

fn require_exact_len(
    operand: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SingleDecodeEnqueueError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SingleDecodeEnqueueError::LengthMismatch {
            operand,
            expected,
            actual,
        })
    }
}

fn record_launch(
    scope: &mut CommandScope<'_>,
    permit: CommandPermit,
    function: CudaFunction,
    result: Result<(), LaunchContractError>,
) -> Result<(), SingleDecodeEnqueueError> {
    match result {
        Ok(()) => {
            scope.record_cuda_submission(permit, function);
            Ok(())
        }
        Err(error) => {
            if let LaunchContractError::Driver(driver_error) = &error {
                scope.record_failed_cuda_submission(permit, function, *driver_error);
            }
            Err(error.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_alignment_gate_accepts_four_byte_boundaries() {
        assert!(require_packed_alignment("Q", 0x1000).is_ok());
    }

    #[test]
    fn packed_alignment_gate_rejects_two_byte_offsets() {
        let error = require_packed_alignment("K", 0x1002).unwrap_err();
        assert!(matches!(
            error,
            SingleDecodeEnqueueError::MisalignedBuffer {
                operand: "K",
                address: 0x1002,
                alignment: 4,
            }
        ));
    }
}
