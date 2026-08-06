//! cuda-oxide provider for BF16 ragged causal prefill attention.

use crate::command::{CommandError, CommandPermit, CommandScope, Read, Write};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, SharedArray, convert, cuda_module, float, kernel, launch_bounds,
    launch_contract, tcgen05, thread, warp,
};
use half::bf16;
use loom_infer::{
    Bf16RaggedPrefillSpec, SINGLE_DECODE_HEAD_DIM, SINGLE_DECODE_PARTIAL_STATE_WIDTH,
};
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

const WARP_THREADS: u32 = 32;
const TOKEN_PARALLEL_8_WARPS: usize = 8;
const TOKEN_PARALLEL_8_THREADS: u32 = WARP_THREADS * TOKEN_PARALLEL_8_WARPS as u32;
const TOKEN_PARALLEL_8_SHARED_NUMEL: usize =
    TOKEN_PARALLEL_8_WARPS * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
const TOKEN_PARALLEL_16_WARPS: usize = 16;
const TOKEN_PARALLEL_16_THREADS: u32 = WARP_THREADS * TOKEN_PARALLEL_16_WARPS as u32;
const TOKEN_PARALLEL_16_SHARED_NUMEL: usize =
    TOKEN_PARALLEL_16_WARPS * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
const TOKEN_PARALLEL_MIN_AVERAGE_KV_LEN: usize = 64;
const BF16_PAIRS_PER_HEAD: usize = SINGLE_DECODE_HEAD_DIM / 2;

const _: () = {
    assert!(TOKEN_PARALLEL_8_THREADS == 256);
    assert!(TOKEN_PARALLEL_16_THREADS == 512);
    assert!(SINGLE_DECODE_HEAD_DIM == 128);
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
            batch_size >= 1,
            nnz_qo >= 1,
            nnz_kv >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == nnz_qo * num_query_heads * 128,
            key.len() == nnz_kv * num_kv_heads * 128,
            value.len() == nnz_kv * num_kv_heads * 128,
            qo_indptr.len() == batch_size + 1,
            kv_indptr.len() == batch_size + 1,
            output.len() == nnz_qo * num_query_heads * 128,
            lse.len() == nnz_qo * num_query_heads,
        ),
    )]
    pub fn ragged_prefill_bf16_nhd_causal(
        batch_size: usize,
        nnz_qo: usize,
        nnz_kv: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let state_index = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let state_count = nnz_qo * num_query_heads;
        if state_index >= state_count || lane >= WARP_THREADS as usize {
            return;
        }

        if qo_indptr[0] != 0
            || kv_indptr[0] != 0
            || qo_indptr[batch_size] != nnz_qo as i32
            || kv_indptr[batch_size] != nnz_kv as i32
        {
            return;
        }

        let query_row = state_index / num_query_heads;
        let query_head = state_index % num_query_heads;
        let mut request = 0_usize;
        while request < batch_size && query_row >= qo_indptr[request + 1] as usize {
            request += 1;
        }
        if request >= batch_size {
            return;
        }

        let qo_start = qo_indptr[request];
        let qo_end = qo_indptr[request + 1];
        let kv_start = kv_indptr[request];
        let kv_end = kv_indptr[request + 1];
        if qo_start < 0
            || qo_end <= qo_start
            || kv_start < 0
            || kv_end <= kv_start
            || qo_end as usize > nnz_qo
            || kv_end as usize > nnz_kv
            || query_row < qo_start as usize
            || query_row >= qo_end as usize
        {
            return;
        }
        let qo_len = (qo_end - qo_start) as usize;
        let kv_len = (kv_end - kv_start) as usize;
        if qo_len > kv_len {
            return;
        }

        let query_index = query_row - qo_start as usize;
        let causal_kv_end = kv_start as usize + kv_len - qo_len + query_index + 1;
        let group_size = num_query_heads / num_kv_heads;
        let kv_head = query_head / group_size;
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key.as_ptr().cast::<u32>();
        let value_pairs = value.as_ptr().cast::<u32>();
        let output_pairs = output.as_mut_ptr().cast::<u32>();
        let first_pair = state_index * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;

        // SAFETY: host validation proves exact spans and four-byte alignment.
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
        let mut token = kv_start as usize;
        while token < causal_kv_end {
            let kv_pair_offset = (token * num_kv_heads + kv_head) * BF16_PAIRS_PER_HEAD + lane;
            // SAFETY: validated request ranges and launch spans prove both
            // packed K/V pairs are in bounds.
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
                if token == kv_start as usize {
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
            // SAFETY: only lane zero writes this state slot.
            unsafe {
                *lse.get_unchecked_mut(state_index) =
                    max_score_log2 + float::lg2_approx_f32(normalizer);
            }
        }
        inverse_normalizer = warp::shuffle_f32(inverse_normalizer, 0);
        // SAFETY: each lane owns two packed output pairs for this state.
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

    #[allow(clippy::too_many_arguments)]
    fn ragged_prefill_bf16_nhd_causal_token_parallel_impl(
        warps: usize,
        partial_states: *mut f32,
        batch_size: usize,
        nnz_qo: usize,
        nnz_kv: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let state_index = thread::blockIdx_x() as usize;
        let thread_in_block = thread::threadIdx_x() as usize;
        let warp_in_block = thread_in_block / WARP_THREADS as usize;
        let lane = thread_in_block % WARP_THREADS as usize;
        let state_count = nnz_qo * num_query_heads;
        if state_index >= state_count {
            return;
        }

        let query_row = state_index / num_query_heads;
        let query_head = state_index % num_query_heads;
        let mut request = 0_usize;
        while request < batch_size && query_row >= qo_indptr[request + 1] as usize {
            request += 1;
        }
        if request >= batch_size {
            return;
        }

        let qo_start = qo_indptr[request];
        let qo_end = qo_indptr[request + 1];
        let kv_start = kv_indptr[request];
        let kv_end = kv_indptr[request + 1];
        let metadata_valid = qo_indptr[0] == 0
            && kv_indptr[0] == 0
            && qo_indptr[batch_size] == nnz_qo as i32
            && kv_indptr[batch_size] == nnz_kv as i32
            && qo_start >= 0
            && qo_end > qo_start
            && kv_start >= 0
            && kv_end > kv_start
            && qo_end as usize <= nnz_qo
            && kv_end as usize <= nnz_kv
            && query_row >= qo_start as usize
            && query_row < qo_end as usize
            && qo_end - qo_start <= kv_end - kv_start;
        if !metadata_valid {
            return;
        }

        let qo_len = (qo_end - qo_start) as usize;
        let kv_len = (kv_end - kv_start) as usize;
        let query_index = query_row - qo_start as usize;
        let causal_token_count = kv_len - qo_len + query_index + 1;
        let causal_kv_start = kv_start as usize;
        let group_size = num_query_heads / num_kv_heads;
        let kv_head = query_head / group_size;
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key.as_ptr().cast::<u32>();
        let value_pairs = value.as_ptr().cast::<u32>();
        let first_pair = state_index * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;

        // SAFETY: the host validates alignment and the launch contract proves
        // the exact query span. Every warp reads the same query-row/head.
        let (query_0, query_1, query_2, query_3) = unsafe {
            let (query_0, query_1) = convert::cvt_f32x2_bf16x2(query_pairs.add(first_pair).read());
            let (query_2, query_3) = convert::cvt_f32x2_bf16x2(query_pairs.add(second_pair).read());
            (query_0, query_1, query_2, query_3)
        };

        let token_start = causal_kv_start + warp_in_block * causal_token_count / warps;
        let token_end = causal_kv_start + (warp_in_block + 1) * causal_token_count / warps;
        let mut output_0 = 0.0_f32;
        let mut output_1 = 0.0_f32;
        let mut output_2 = 0.0_f32;
        let mut output_3 = 0.0_f32;
        let mut max_score_log2 = f32::NEG_INFINITY;
        let mut normalizer = 0.0_f32;
        let mut token = token_start;
        while token < token_end {
            let kv_pair_offset = (token * num_kv_heads + kv_head) * BF16_PAIRS_PER_HEAD + lane;
            // SAFETY: metadata validation and the causal partition prove both
            // packed K/V pairs are in the exact launch spans.
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

        let first_component = lane * 2;
        let second_component = SINGLE_DECODE_HEAD_DIM / 2 + lane * 2;
        let state_offset = warp_in_block * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
        // SAFETY: lane zero owns the two header slots and every lane owns four
        // distinct weighted-value slots within its warp's partial state.
        unsafe {
            if lane == 0 {
                partial_states.add(state_offset).write(max_score_log2);
                partial_states.add(state_offset + 1).write(normalizer);
            }
            partial_states
                .add(state_offset + 2 + first_component)
                .write(output_0);
            partial_states
                .add(state_offset + 3 + first_component)
                .write(output_1);
            partial_states
                .add(state_offset + 2 + second_component)
                .write(output_2);
            partial_states
                .add(state_offset + 3 + second_component)
                .write(output_3);
        }
        thread::sync_threads();

        if warp_in_block != 0 {
            return;
        }

        let mut merged_max_log2 = f32::NEG_INFINITY;
        let mut merged_normalizer = 0.0_f32;
        let mut merged_output_0 = 0.0_f32;
        let mut merged_output_1 = 0.0_f32;
        let mut merged_output_2 = 0.0_f32;
        let mut merged_output_3 = 0.0_f32;
        let mut partial = 0_usize;
        while partial < warps {
            let partial_offset = partial * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
            // SAFETY: all warps completed their disjoint state before the
            // barrier and warp zero owns the merge reads.
            let (partial_max, partial_normalizer, value_0, value_1, value_2, value_3) = unsafe {
                (
                    partial_states.add(partial_offset).read(),
                    partial_states.add(partial_offset + 1).read(),
                    partial_states
                        .add(partial_offset + 2 + first_component)
                        .read(),
                    partial_states
                        .add(partial_offset + 3 + first_component)
                        .read(),
                    partial_states
                        .add(partial_offset + 2 + second_component)
                        .read(),
                    partial_states
                        .add(partial_offset + 3 + second_component)
                        .read(),
                )
            };
            if partial_normalizer != 0.0 {
                if merged_normalizer == 0.0 {
                    merged_max_log2 = partial_max;
                    merged_normalizer = partial_normalizer;
                    merged_output_0 = value_0;
                    merged_output_1 = value_1;
                    merged_output_2 = value_2;
                    merged_output_3 = value_3;
                } else {
                    let next_max = f32::max(merged_max_log2, partial_max);
                    let merged_weight = float::ex2_approx_f32(merged_max_log2 - next_max);
                    let partial_weight = float::ex2_approx_f32(partial_max - next_max);
                    merged_normalizer =
                        merged_normalizer * merged_weight + partial_normalizer * partial_weight;
                    merged_output_0 =
                        float::fma_rn_f32(value_0, partial_weight, merged_output_0 * merged_weight);
                    merged_output_1 =
                        float::fma_rn_f32(value_1, partial_weight, merged_output_1 * merged_weight);
                    merged_output_2 =
                        float::fma_rn_f32(value_2, partial_weight, merged_output_2 * merged_weight);
                    merged_output_3 =
                        float::fma_rn_f32(value_3, partial_weight, merged_output_3 * merged_weight);
                    merged_max_log2 = next_max;
                }
            }
            partial += 1;
        }

        let inverse_normalizer = float::div_rn_f32(1.0, merged_normalizer);
        if lane == 0 {
            // SAFETY: only lane zero writes this query-row/head LSE slot.
            unsafe {
                *lse.get_unchecked_mut(state_index) =
                    merged_max_log2 + float::lg2_approx_f32(merged_normalizer);
            }
        }
        let output_pairs = output.as_mut_ptr().cast::<u32>();
        // SAFETY: warp zero owns all packed output pairs for this state.
        unsafe {
            output_pairs
                .add(first_pair)
                .write(tcgen05::cvt_f32x2_bf16x2(
                    merged_output_0 * inverse_normalizer,
                    merged_output_1 * inverse_normalizer,
                ));
            output_pairs
                .add(second_pair)
                .write(tcgen05::cvt_f32x2_bf16x2(
                    merged_output_2 * inverse_normalizer,
                    merged_output_3 * inverse_normalizer,
                ));
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            batch_size >= 1,
            nnz_qo >= 1,
            nnz_kv >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == nnz_qo * num_query_heads * 128,
            key.len() == nnz_kv * num_kv_heads * 128,
            value.len() == nnz_kv * num_kv_heads * 128,
            qo_indptr.len() == batch_size + 1,
            kv_indptr.len() == batch_size + 1,
            output.len() == nnz_qo * num_query_heads * 128,
            lse.len() == nnz_qo * num_query_heads,
        ),
    )]
    pub fn ragged_prefill_bf16_nhd_causal_token_parallel8(
        batch_size: usize,
        nnz_qo: usize,
        nnz_kv: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        output: DisjointSlice<bf16>,
        lse: DisjointSlice<f32>,
    ) {
        static mut PARTIAL_STATES: SharedArray<f32, TOKEN_PARALLEL_8_SHARED_NUMEL> =
            SharedArray::UNINIT;
        // SAFETY: each warp owns one disjoint state and the implementation
        // orders cross-warp reads with one block barrier.
        let partial_states = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL_STATES) };
        ragged_prefill_bf16_nhd_causal_token_parallel_impl(
            TOKEN_PARALLEL_8_WARPS,
            partial_states,
            batch_size,
            nnz_qo,
            nnz_kv,
            num_query_heads,
            num_kv_heads,
            softmax_scale_log2,
            query,
            key,
            value,
            qo_indptr,
            kv_indptr,
            output,
            lse,
        );
    }

    #[kernel]
    #[launch_bounds(512)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            batch_size >= 1,
            nnz_qo >= 1,
            nnz_kv >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == nnz_qo * num_query_heads * 128,
            key.len() == nnz_kv * num_kv_heads * 128,
            value.len() == nnz_kv * num_kv_heads * 128,
            qo_indptr.len() == batch_size + 1,
            kv_indptr.len() == batch_size + 1,
            output.len() == nnz_qo * num_query_heads * 128,
            lse.len() == nnz_qo * num_query_heads,
        ),
    )]
    pub fn ragged_prefill_bf16_nhd_causal_token_parallel16(
        batch_size: usize,
        nnz_qo: usize,
        nnz_kv: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        output: DisjointSlice<bf16>,
        lse: DisjointSlice<f32>,
    ) {
        static mut PARTIAL_STATES: SharedArray<f32, TOKEN_PARALLEL_16_SHARED_NUMEL> =
            SharedArray::UNINIT;
        // SAFETY: each warp owns one disjoint state and the implementation
        // orders cross-warp reads with one block barrier.
        let partial_states = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL_STATES) };
        ragged_prefill_bf16_nhd_causal_token_parallel_impl(
            TOKEN_PARALLEL_16_WARPS,
            partial_states,
            batch_size,
            nnz_qo,
            nnz_kv,
            num_query_heads,
            num_kv_heads,
            softmax_scale_log2,
            query,
            key,
            value,
            qo_indptr,
            kv_indptr,
            output,
            lse,
        );
    }
}

#[derive(Clone, Debug)]
pub struct PrefillProvider {
    module: kernels::LoadedModule,
}

impl PrefillProvider {
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, cuda_host::EmbeddedModuleError> {
        // SAFETY: this crate owns the package-named prefill artifact bundle.
        let module = unsafe { kernels::load(context)? };
        Ok(Self { module })
    }

    pub fn plan_bf16_ragged(
        &self,
        spec: Bf16RaggedPrefillSpec,
    ) -> Result<Bf16RaggedPrefillPlan, RaggedPrefillPlanError> {
        let states = spec
            .nnz_qo()
            .checked_mul(spec.num_query_heads())
            .ok_or(RaggedPrefillPlanError::StateCountOutOfRange(usize::MAX))?;
        let states = u32::try_from(states)
            .map_err(|_| RaggedPrefillPlanError::StateCountOutOfRange(states))?;
        let algorithm = ragged_prefill_algorithm(spec);
        let launch = match algorithm {
            Bf16RaggedPrefillAlgorithm::Direct => Bf16RaggedPrefillLaunch::Direct(
                self.module
                    .prepare_ragged_prefill_bf16_nhd_causal(LaunchConfig1D::new(
                        states,
                        WARP_THREADS,
                        0,
                    ))?,
            ),
            Bf16RaggedPrefillAlgorithm::TokenParallel8 => {
                Bf16RaggedPrefillLaunch::TokenParallel8(
                    self.module
                        .prepare_ragged_prefill_bf16_nhd_causal_token_parallel8(
                            LaunchConfig1D::new(states, TOKEN_PARALLEL_8_THREADS, 0),
                        )?,
                )
            }
            Bf16RaggedPrefillAlgorithm::TokenParallel16 => {
                Bf16RaggedPrefillLaunch::TokenParallel16(
                    self.module
                        .prepare_ragged_prefill_bf16_nhd_causal_token_parallel16(
                            LaunchConfig1D::new(states, TOKEN_PARALLEL_16_THREADS, 0),
                        )?,
                )
            }
        };
        Ok(Bf16RaggedPrefillPlan {
            spec,
            algorithm,
            module: self.module.clone(),
            launch,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16RaggedPrefillAlgorithm {
    Direct,
    TokenParallel8,
    TokenParallel16,
}

const fn ragged_prefill_algorithm(spec: Bf16RaggedPrefillSpec) -> Bf16RaggedPrefillAlgorithm {
    if spec.nnz_kv() / spec.batch_size() < TOKEN_PARALLEL_MIN_AVERAGE_KV_LEN {
        Bf16RaggedPrefillAlgorithm::Direct
    } else if spec.num_kv_heads() == 1 {
        Bf16RaggedPrefillAlgorithm::TokenParallel16
    } else {
        Bf16RaggedPrefillAlgorithm::TokenParallel8
    }
}

#[derive(Clone)]
enum Bf16RaggedPrefillLaunch {
    Direct(PreparedLaunch<kernels::__ragged_prefill_bf16_nhd_causal_CudaKernel>),
    TokenParallel8(
        PreparedLaunch<kernels::__ragged_prefill_bf16_nhd_causal_token_parallel8_CudaKernel>,
    ),
    TokenParallel16(
        PreparedLaunch<kernels::__ragged_prefill_bf16_nhd_causal_token_parallel16_CudaKernel>,
    ),
}

#[derive(Clone)]
pub struct Bf16RaggedPrefillPlan {
    spec: Bf16RaggedPrefillSpec,
    algorithm: Bf16RaggedPrefillAlgorithm,
    module: kernels::LoadedModule,
    launch: Bf16RaggedPrefillLaunch,
}

impl Bf16RaggedPrefillPlan {
    pub const fn spec(&self) -> Bf16RaggedPrefillSpec {
        self.spec
    }

    pub const fn algorithm(&self) -> Bf16RaggedPrefillAlgorithm {
        self.algorithm
    }

    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16RaggedPrefillArgs,
    ) -> Result<(), RaggedPrefillEnqueueError> {
        let permit = scope.prepare_command()?;
        let (function, launch_result) = {
            let resolved = scope.resolve_rrrrrww(
                args.query,
                args.key,
                args.value,
                args.qo_indptr,
                args.kv_indptr,
                args.output,
                args.lse,
            )?;
            require_exact_len("Q", resolved.first.len(), self.spec.query_numel())?;
            require_exact_len("K", resolved.second.len(), self.spec.kv_numel())?;
            require_exact_len("V", resolved.third.len(), self.spec.kv_numel())?;
            require_exact_len("qo_indptr", resolved.fourth.len(), self.spec.indptr_numel())?;
            require_exact_len("kv_indptr", resolved.fifth.len(), self.spec.indptr_numel())?;
            require_exact_len("O", resolved.sixth.len(), self.spec.output_numel())?;
            require_exact_len("LSE", resolved.seventh.len(), self.spec.lse_numel())?;
            for (operand, address) in [
                ("Q", resolved.first.cu_deviceptr()),
                ("K", resolved.second.cu_deviceptr()),
                ("V", resolved.third.cu_deviceptr()),
                ("O", resolved.sixth.cu_deviceptr()),
            ] {
                require_packed_alignment(operand, address)?;
            }
            let common = (
                self.spec.batch_size(),
                self.spec.nnz_qo(),
                self.spec.nnz_kv(),
                self.spec.num_query_heads(),
                self.spec.num_kv_heads(),
                self.spec.softmax_scale() * core::f32::consts::LOG2_E,
            );
            match &self.launch {
                Bf16RaggedPrefillLaunch::Direct(launch) => {
                    let result = self.module.ragged_prefill_bf16_nhd_causal(
                        resolved.stream,
                        launch,
                        common.0,
                        common.1,
                        common.2,
                        common.3,
                        common.4,
                        common.5,
                        resolved.first,
                        resolved.second,
                        resolved.third,
                        resolved.fourth,
                        resolved.fifth,
                        resolved.sixth,
                        resolved.seventh,
                    );
                    (launch.function().clone(), result)
                }
                Bf16RaggedPrefillLaunch::TokenParallel8(launch) => {
                    let result = self.module.ragged_prefill_bf16_nhd_causal_token_parallel8(
                        resolved.stream,
                        launch,
                        common.0,
                        common.1,
                        common.2,
                        common.3,
                        common.4,
                        common.5,
                        resolved.first,
                        resolved.second,
                        resolved.third,
                        resolved.fourth,
                        resolved.fifth,
                        resolved.sixth,
                        resolved.seventh,
                    );
                    (launch.function().clone(), result)
                }
                Bf16RaggedPrefillLaunch::TokenParallel16(launch) => {
                    let result = self.module.ragged_prefill_bf16_nhd_causal_token_parallel16(
                        resolved.stream,
                        launch,
                        common.0,
                        common.1,
                        common.2,
                        common.3,
                        common.4,
                        common.5,
                        resolved.first,
                        resolved.second,
                        resolved.third,
                        resolved.fourth,
                        resolved.fifth,
                        resolved.sixth,
                        resolved.seventh,
                    );
                    (launch.function().clone(), result)
                }
            }
        };
        record_launch(scope, permit, function, launch_result)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bf16RaggedPrefillArgs {
    query: Read<bf16>,
    key: Read<bf16>,
    value: Read<bf16>,
    qo_indptr: Read<i32>,
    kv_indptr: Read<i32>,
    output: Write<bf16>,
    lse: Write<f32>,
}

impl Bf16RaggedPrefillArgs {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        query: Read<bf16>,
        key: Read<bf16>,
        value: Read<bf16>,
        qo_indptr: Read<i32>,
        kv_indptr: Read<i32>,
        output: Write<bf16>,
        lse: Write<f32>,
    ) -> Self {
        Self {
            query,
            key,
            value,
            qo_indptr,
            kv_indptr,
            output,
            lse,
        }
    }
}

#[derive(Debug, Error)]
pub enum RaggedPrefillPlanError {
    #[error("ragged prefill state count {0} exceeds the CUDA grid range")]
    StateCountOutOfRange(usize),
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
}

#[derive(Debug, Error)]
pub enum RaggedPrefillEnqueueError {
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
        "packed ragged prefill requires {operand} to be {alignment}-byte aligned, got {address:#x}"
    )]
    MisalignedBuffer {
        operand: &'static str,
        address: u64,
        alignment: u64,
    },
}

fn require_exact_len(
    operand: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), RaggedPrefillEnqueueError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RaggedPrefillEnqueueError::LengthMismatch {
            operand,
            expected,
            actual,
        })
    }
}

fn require_packed_alignment(
    operand: &'static str,
    address: u64,
) -> Result<(), RaggedPrefillEnqueueError> {
    const ALIGNMENT: u64 = size_of::<u32>() as u64;
    if address.is_multiple_of(ALIGNMENT) {
        Ok(())
    } else {
        Err(RaggedPrefillEnqueueError::MisalignedBuffer {
            operand,
            address,
            alignment: ALIGNMENT,
        })
    }
}

fn record_launch(
    scope: &mut CommandScope<'_>,
    permit: CommandPermit,
    function: CudaFunction,
    result: Result<(), LaunchContractError>,
) -> Result<(), RaggedPrefillEnqueueError> {
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
    fn algorithm_keeps_short_kv_direct_and_parallelizes_long_kv() {
        let short = Bf16RaggedPrefillSpec::new(1, 16, 16, 8, 8, 128).unwrap();
        let long = Bf16RaggedPrefillSpec::new(3, 21, 896, 8, 1, 128).unwrap();

        assert_eq!(
            ragged_prefill_algorithm(short),
            Bf16RaggedPrefillAlgorithm::Direct
        );
        assert_eq!(
            ragged_prefill_algorithm(long),
            Bf16RaggedPrefillAlgorithm::TokenParallel16
        );
        let grouped = Bf16RaggedPrefillSpec::new(2, 96, 1280, 16, 4, 128).unwrap();
        assert_eq!(
            ragged_prefill_algorithm(grouped),
            Bf16RaggedPrefillAlgorithm::TokenParallel8
        );
    }
}
