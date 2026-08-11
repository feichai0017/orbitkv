//! cuda-oxide providers for BF16 ragged and paged causal prefill attention.

use crate::command::{
    CommandError, CommandPermit, CommandScope, DeviceStatusReservation, Read, ReadWrite, Write,
};
use crate::device_status::{
    DeviceStatusDecoder, STATUS_ELEMENT_COUNT_OVERFLOW, STATUS_EMPTY_PAGED_REQUEST,
    STATUS_EMPTY_QO_REQUEST, STATUS_INVALID_LAST_PAGE_LENGTH, STATUS_INVALID_PAGE_INDPTR_START,
    STATUS_INVALID_QO_INDPTR_START, STATUS_NON_MONOTONIC_PAGE_INDPTR,
    STATUS_NON_MONOTONIC_QO_INDPTR, STATUS_PACKET_WORDS, STATUS_PAGE_INDEX_OUT_OF_RANGE,
    STATUS_PAGE_INDICES_LENGTH_MISMATCH, STATUS_QO_INDPTR_LENGTH_MISMATCH,
    STATUS_RAGGED_QUERY_LONGER_THAN_KV, STATUS_SUCCESS,
};
use crate::memory::{DeviceRegionLaunchError, enqueue_region_launch};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, SharedArray, async_copy, convert, cuda_module, float, kernel, launch_bounds,
    launch_contract, tcgen05, thread, warp, wmma,
};
use half::bf16;
use oxide_infer::{
    Bf16PagedPrefillSpec, Bf16RaggedPrefillSpec, PAGED_PREFILL_PAGE_SIZE, SINGLE_DECODE_HEAD_DIM,
    SINGLE_DECODE_PARTIAL_STATE_WIDTH,
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
const TILED_GQA_GROUP_SIZE: usize = 4;
const TILED_QUERY_ROWS: usize = 16;
const TILED_PACKED_QUERY_ROWS: usize = TILED_QUERY_ROWS * TILED_GQA_GROUP_SIZE;
const TILED_WARPS: usize = 4;
const TILED_THREADS: u32 = WARP_THREADS * TILED_WARPS as u32;
const TILED_KV_ROWS: usize = 64;
const TILED_KV_SUBTILES: usize = TILED_KV_ROWS / 16;
const TILED_KV_SHARED_PAIRS: usize = TILED_KV_ROWS * BF16_PAIRS_PER_HEAD;
const TILED_KV_COPY_BYTES: usize = 16;
const TILED_KV_COPY_PAIRS: usize = TILED_KV_COPY_BYTES / size_of::<u32>();
const TILED_KV_COPIES: usize = TILED_KV_SHARED_PAIRS / TILED_KV_COPY_PAIRS;
const TILED_PARTITIONS: usize = 8;
const TILED_MIN_AVERAGE_KV_LEN: usize = 256;

const _: () = {
    assert!(TOKEN_PARALLEL_8_THREADS == 256);
    assert!(TOKEN_PARALLEL_16_THREADS == 512);
    assert!(TILED_PACKED_QUERY_ROWS == TILED_WARPS * 16);
    assert!(TILED_THREADS == 128);
    assert!(TILED_KV_SUBTILES == 4);
    assert!(TILED_KV_COPIES == TILED_THREADS as usize * 8);
    assert!(TILED_PARTITIONS == 8);
    assert!(SINGLE_DECODE_HEAD_DIM == 128);
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(1)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (1, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            batch_size >= 1,
            nnz_qo >= 1,
            max_num_pages >= 1,
            qo_indptr.len() == batch_size + 1,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
        ),
    )]
    pub fn validate_paged_prefill_metadata(
        batch_size: usize,
        nnz_qo: usize,
        max_num_pages: usize,
        qo_indptr: &[i32],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        mut metadata_status: DisjointSlice<i32>,
    ) {
        if thread::blockIdx_x() != 0 || thread::threadIdx_x() != 0 {
            return;
        }
        let output = metadata_status.as_mut_ptr();
        // SAFETY: the launch contract proves the five-word status span.
        unsafe { write_status(output, STATUS_SUCCESS, 0, 0, 0, 0) };

        if qo_indptr[0] != 0 {
            // SAFETY: as above.
            unsafe {
                write_status(
                    output,
                    STATUS_INVALID_QO_INDPTR_START,
                    qo_indptr[0],
                    0,
                    0,
                    0,
                )
            };
            return;
        }
        let mut request = 0_usize;
        while request < batch_size {
            let start = qo_indptr[request];
            let end = qo_indptr[request + 1];
            if end < start {
                // SAFETY: the launch contract proves the status span.
                unsafe {
                    write_status(
                        output,
                        STATUS_NON_MONOTONIC_QO_INDPTR,
                        request as i32,
                        start,
                        end,
                        0,
                    )
                };
                return;
            }
            if end == start {
                // SAFETY: as above.
                unsafe { write_status(output, STATUS_EMPTY_QO_REQUEST, request as i32, 0, 0, 0) };
                return;
            }
            request += 1;
        }
        let qo_terminal = qo_indptr[batch_size];
        if qo_terminal < 0 {
            // SAFETY: the launch contract proves the status span.
            unsafe { write_status(output, STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 0) };
            return;
        }
        if qo_terminal as usize != nnz_qo {
            // SAFETY: as above.
            unsafe {
                write_status(
                    output,
                    STATUS_QO_INDPTR_LENGTH_MISMATCH,
                    qo_terminal,
                    0,
                    0,
                    0,
                )
            };
            return;
        }

        if page_indptr[0] != 0 {
            // SAFETY: the launch contract proves the status span.
            unsafe {
                write_status(
                    output,
                    STATUS_INVALID_PAGE_INDPTR_START,
                    page_indptr[0],
                    0,
                    0,
                    0,
                )
            };
            return;
        }
        request = 0;
        while request < batch_size {
            let page_start = page_indptr[request];
            let page_end = page_indptr[request + 1];
            if page_end < page_start {
                // SAFETY: the launch contract proves the status span.
                unsafe {
                    write_status(
                        output,
                        STATUS_NON_MONOTONIC_PAGE_INDPTR,
                        request as i32,
                        page_start,
                        page_end,
                        0,
                    )
                };
                return;
            }
            if page_end == page_start {
                // SAFETY: as above.
                unsafe {
                    write_status(output, STATUS_EMPTY_PAGED_REQUEST, request as i32, 0, 0, 0)
                };
                return;
            }
            let tail_len = last_page_len[request];
            if !(1..=PAGED_PREFILL_PAGE_SIZE as i32).contains(&tail_len) {
                // SAFETY: as above.
                unsafe {
                    write_status(
                        output,
                        STATUS_INVALID_LAST_PAGE_LENGTH,
                        request as i32,
                        tail_len,
                        0,
                        0,
                    )
                };
                return;
            }
            let qo_len = (qo_indptr[request + 1] - qo_indptr[request]) as usize;
            let page_count = (page_end - page_start) as usize;
            let Some(kv_len) = (page_count - 1)
                .checked_mul(PAGED_PREFILL_PAGE_SIZE)
                .and_then(|tokens| tokens.checked_add(tail_len as usize))
            else {
                // SAFETY: the launch contract proves the status span.
                unsafe { write_status(output, STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 0) };
                return;
            };
            if qo_len > kv_len {
                // The failing lengths are bounded by the validated i32 query domain.
                // SAFETY: the launch contract proves the status span.
                unsafe {
                    write_status(
                        output,
                        STATUS_RAGGED_QUERY_LONGER_THAN_KV,
                        request as i32,
                        qo_len as i32,
                        kv_len as i32,
                        0,
                    )
                };
                return;
            }
            request += 1;
        }

        let page_terminal = page_indptr[batch_size];
        if page_terminal < 0 {
            // SAFETY: the launch contract proves the status span.
            unsafe { write_status(output, STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 0) };
            return;
        }
        if page_terminal as usize != page_indices.len() {
            // SAFETY: as above.
            unsafe {
                write_status(
                    output,
                    STATUS_PAGE_INDICES_LENGTH_MISMATCH,
                    page_terminal,
                    0,
                    0,
                    0,
                )
            };
            return;
        }
        let mut position = 0_usize;
        while position < page_indices.len() {
            let physical_page = page_indices[position];
            if physical_page < 0 || physical_page as usize >= max_num_pages {
                // SAFETY: the launch contract proves the status span.
                unsafe {
                    write_status(
                        output,
                        STATUS_PAGE_INDEX_OUT_OF_RANGE,
                        position as i32,
                        physical_page,
                        0,
                        0,
                    )
                };
                return;
            }
            position += 1;
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
            max_num_pages >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == nnz_qo * num_query_heads * 128,
            key_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            value_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            qo_indptr.len() == batch_size + 1,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
            output.len() == nnz_qo * num_query_heads * 128,
            lse.len() == nnz_qo * num_query_heads,
        ),
    )]
    pub fn paged_prefill_bf16_nhd_causal(
        batch_size: usize,
        nnz_qo: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        qo_indptr: &[i32],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let state_index = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let state_count = nnz_qo * num_query_heads;
        if state_index >= state_count
            || lane >= WARP_THREADS as usize
            || metadata_status[0] != STATUS_SUCCESS
        {
            return;
        }
        let _ = (batch_size, max_num_pages);

        let query_row = state_index / num_query_heads;
        let query_head = state_index % num_query_heads;
        let mut request = 0_usize;
        while query_row >= qo_indptr[request + 1] as usize {
            request += 1;
        }

        let qo_start = qo_indptr[request] as usize;
        let qo_end = qo_indptr[request + 1] as usize;
        let page_start = page_indptr[request];
        let page_end = page_indptr[request + 1];
        let tail_len = last_page_len[request];

        let qo_len = qo_end - qo_start;
        let page_count = (page_end - page_start) as usize;
        let kv_len = (page_count - 1) * PAGED_PREFILL_PAGE_SIZE + tail_len as usize;
        let query_index = query_row - qo_start;
        let causal_kv_end = kv_len - qo_len + query_index + 1;
        let group_size = num_query_heads / num_kv_heads;
        let kv_head = query_head / group_size;
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key_pages.as_ptr().cast::<u32>();
        let value_pairs = value_pages.as_ptr().cast::<u32>();
        let output_pairs = output.as_mut_ptr().cast::<u32>();
        let first_pair = state_index * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;

        // SAFETY: the host validates alignment and the launch contract proves
        // both packed query reads are inside the exact span.
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
        let mut logical_token = 0_usize;
        while logical_token < causal_kv_end {
            let logical_page = logical_token / PAGED_PREFILL_PAGE_SIZE;
            let page_offset = logical_token % PAGED_PREFILL_PAGE_SIZE;
            let physical_page = page_indices[page_start as usize + logical_page] as usize;
            let kv_pair_offset = (((physical_page * PAGED_PREFILL_PAGE_SIZE + page_offset)
                * num_kv_heads
                + kv_head)
                * BF16_PAIRS_PER_HEAD)
                + lane;

            // SAFETY: the preceding stream-ordered validator proved every
            // referenced page is in range before this K/V pointer arithmetic.
            // The launch contract proves the exact page-pool spans.
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
                if logical_token == 0 {
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
            logical_token += 1;
        }

        let mut inverse_normalizer = 0.0_f32;
        if lane == 0 {
            inverse_normalizer = float::div_rn_f32(1.0, normalizer);
            // SAFETY: only lane zero writes this query-row/head state.
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
    fn paged_prefill_bf16_nhd_causal_token_parallel_impl(
        warps: usize,
        partial_states: *mut f32,
        batch_size: usize,
        nnz_qo: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        qo_indptr: &[i32],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let state_index = thread::blockIdx_x() as usize;
        let thread_in_block = thread::threadIdx_x() as usize;
        let warp_in_block = thread_in_block / WARP_THREADS as usize;
        let lane = thread_in_block % WARP_THREADS as usize;
        let state_count = nnz_qo * num_query_heads;
        if state_index >= state_count || metadata_status[0] != STATUS_SUCCESS {
            return;
        }
        let _ = (batch_size, max_num_pages);

        let query_row = state_index / num_query_heads;
        let query_head = state_index % num_query_heads;
        let mut request = 0_usize;
        while query_row >= qo_indptr[request + 1] as usize {
            request += 1;
        }

        let qo_start = qo_indptr[request] as usize;
        let qo_end = qo_indptr[request + 1] as usize;
        let page_start = page_indptr[request];
        let page_end = page_indptr[request + 1];
        let tail_len = last_page_len[request];

        let qo_len = qo_end - qo_start;
        let page_count = (page_end - page_start) as usize;
        let kv_len = (page_count - 1) * PAGED_PREFILL_PAGE_SIZE + tail_len as usize;
        let query_index = query_row - qo_start;
        let causal_token_count = kv_len - qo_len + query_index + 1;
        let group_size = num_query_heads / num_kv_heads;
        let kv_head = query_head / group_size;
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key_pages.as_ptr().cast::<u32>();
        let value_pairs = value_pages.as_ptr().cast::<u32>();
        let first_pair = state_index * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;

        // SAFETY: the host validates alignment and the launch contract proves
        // the exact query span. Every warp reads the same query-row/head.
        let (query_0, query_1, query_2, query_3) = unsafe {
            let (query_0, query_1) = convert::cvt_f32x2_bf16x2(query_pairs.add(first_pair).read());
            let (query_2, query_3) = convert::cvt_f32x2_bf16x2(query_pairs.add(second_pair).read());
            (query_0, query_1, query_2, query_3)
        };

        let token_start = warp_in_block * causal_token_count / warps;
        let token_end = (warp_in_block + 1) * causal_token_count / warps;
        let mut output_0 = 0.0_f32;
        let mut output_1 = 0.0_f32;
        let mut output_2 = 0.0_f32;
        let mut output_3 = 0.0_f32;
        let mut max_score_log2 = f32::NEG_INFINITY;
        let mut normalizer = 0.0_f32;
        let mut logical_token = token_start;
        while logical_token < token_end {
            let logical_page = logical_token / PAGED_PREFILL_PAGE_SIZE;
            let page_offset = logical_token % PAGED_PREFILL_PAGE_SIZE;
            let physical_page = page_indices[page_start as usize + logical_page] as usize;
            let kv_pair_offset = (((physical_page * PAGED_PREFILL_PAGE_SIZE + page_offset)
                * num_kv_heads
                + kv_head)
                * BF16_PAIRS_PER_HEAD)
                + lane;

            // SAFETY: the preceding stream-ordered validator proved every
            // referenced page is in range before this K/V pointer arithmetic.
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
                if logical_token == token_start {
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
            logical_token += 1;
        }

        let first_component = lane * 2;
        let second_component = SINGLE_DECODE_HEAD_DIM / 2 + lane * 2;
        let state_offset = warp_in_block * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
        // SAFETY: lane zero owns the headers and every lane owns four distinct
        // weighted-value slots within its warp's partial state.
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
            // SAFETY: all warps completed their disjoint states before the
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
            max_num_pages >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == nnz_qo * num_query_heads * 128,
            key_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            value_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            qo_indptr.len() == batch_size + 1,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
            output.len() == nnz_qo * num_query_heads * 128,
            lse.len() == nnz_qo * num_query_heads,
        ),
    )]
    pub fn paged_prefill_bf16_nhd_causal_token_parallel8(
        batch_size: usize,
        nnz_qo: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        qo_indptr: &[i32],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        output: DisjointSlice<bf16>,
        lse: DisjointSlice<f32>,
    ) {
        static mut PARTIAL_STATES: SharedArray<f32, TOKEN_PARALLEL_8_SHARED_NUMEL> =
            SharedArray::UNINIT;
        // SAFETY: each block owns this array, every warp owns one partial
        // state, and barriers order all cross-warp reads.
        let partial_states = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL_STATES) };
        paged_prefill_bf16_nhd_causal_token_parallel_impl(
            TOKEN_PARALLEL_8_WARPS,
            partial_states,
            batch_size,
            nnz_qo,
            max_num_pages,
            num_query_heads,
            num_kv_heads,
            softmax_scale_log2,
            query,
            key_pages,
            value_pages,
            qo_indptr,
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
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
            max_num_pages >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == nnz_qo * num_query_heads * 128,
            key_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            value_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            qo_indptr.len() == batch_size + 1,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
            output.len() == nnz_qo * num_query_heads * 128,
            lse.len() == nnz_qo * num_query_heads,
        ),
    )]
    pub fn paged_prefill_bf16_nhd_causal_token_parallel16(
        batch_size: usize,
        nnz_qo: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        qo_indptr: &[i32],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        output: DisjointSlice<bf16>,
        lse: DisjointSlice<f32>,
    ) {
        static mut PARTIAL_STATES: SharedArray<f32, TOKEN_PARALLEL_16_SHARED_NUMEL> =
            SharedArray::UNINIT;
        // SAFETY: each block owns this array, every warp owns one partial
        // state, and barriers order all cross-warp reads.
        let partial_states = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL_STATES) };
        paged_prefill_bf16_nhd_causal_token_parallel_impl(
            TOKEN_PARALLEL_16_WARPS,
            partial_states,
            batch_size,
            nnz_qo,
            max_num_pages,
            num_query_heads,
            num_kv_heads,
            softmax_scale_log2,
            query,
            key_pages,
            value_pages,
            qo_indptr,
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
            output,
            lse,
        );
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
    #[launch_bounds(128)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
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

    #[inline(always)]
    unsafe fn tiled_qk_mma(
        accumulator: [f32; 4],
        query_fragment: [u32; 4],
        kv_tile: *const u32,
        token_row: usize,
        component_pair_base: usize,
        lane_in_group: usize,
    ) -> [f32; 4] {
        let pair_base = token_row * BF16_PAIRS_PER_HEAD + component_pair_base;
        // SAFETY: the caller initialized the complete shared K tile and all
        // lanes call this helper with offsets inside that tile.
        let key_fragment = unsafe {
            [
                kv_tile.add(pair_base + lane_in_group).read(),
                kv_tile.add(pair_base + 4 + lane_in_group).read(),
            ]
        };
        // SAFETY: the caller invokes this uniformly across the full warp with
        // fragments in the documented m16n8k16 BF16 layout.
        unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, query_fragment, key_fragment) }
    }

    macro_rules! load_tiled_kv_async {
        (
            $shared_tile:expr,
            $global_pairs:expr,
            $thread_in_block:expr,
            $kv_start:expr,
            $kv_tile_start:expr,
            $partition_token_end:expr,
            $num_kv_heads:expr,
            $kv_head:expr $(,)?
        ) => {{
            let token_lane = $thread_in_block / 16;
            let pair_in_head = ($thread_in_block % 16) * TILED_KV_COPY_PAIRS;
            macro_rules! issue_copy {
                ($iteration:literal) => {{
                    let token = $kv_tile_start + token_lane + $iteration * 8;
                    let source_token = usize::min(token, $partition_token_end - 1);
                    let source_pair = (($kv_start + source_token) * $num_kv_heads + $kv_head)
                        * BF16_PAIRS_PER_HEAD
                        + pair_in_head;
                    let destination_pair = ($thread_in_block + $iteration * TILED_THREADS as usize)
                        * TILED_KV_COPY_PAIRS;
                    let src_size = if token < $partition_token_end {
                        TILED_KV_COPY_BYTES as u32
                    } else {
                        0
                    };
                    // SAFETY: shared destinations are disjoint 16-byte regions.
                    // Valid sources are inside K/V; padded rows use zero-fill.
                    unsafe {
                        async_copy::cp_async_cg_zfill_16(
                            $shared_tile.add(destination_pair),
                            $global_pairs.add(source_pair).cast(),
                            src_size,
                        );
                    }
                }};
            }
            issue_copy!(0);
            issue_copy!(1);
            issue_copy!(2);
            issue_copy!(3);
            issue_copy!(4);
            issue_copy!(5);
            issue_copy!(6);
            issue_copy!(7);
            // SAFETY: all copies issued by this thread belong to this group.
            unsafe {
                async_copy::cp_async_commit_group();
                async_copy::cp_async_wait_all();
            }
        }};
    }

    #[inline(always)]
    fn tiled_mask_score(
        score: [f32; 4],
        token_base: usize,
        causal_end_0: usize,
        causal_end_1: usize,
        query_valid_0: bool,
        query_valid_1: bool,
        softmax_scale_log2: f32,
    ) -> [f32; 4] {
        [
            if query_valid_0 && token_base < causal_end_0 {
                score[0] * softmax_scale_log2
            } else {
                f32::NEG_INFINITY
            },
            if query_valid_0 && token_base + 1 < causal_end_0 {
                score[1] * softmax_scale_log2
            } else {
                f32::NEG_INFINITY
            },
            if query_valid_1 && token_base < causal_end_1 {
                score[2] * softmax_scale_log2
            } else {
                f32::NEG_INFINITY
            },
            if query_valid_1 && token_base + 1 < causal_end_1 {
                score[3] * softmax_scale_log2
            } else {
                f32::NEG_INFINITY
            },
        ]
    }

    #[inline(always)]
    fn tiled_score_max_0(score: [f32; 4]) -> f32 {
        f32::max(score[0], score[1])
    }

    #[inline(always)]
    fn tiled_score_max_1(score: [f32; 4]) -> f32 {
        f32::max(score[2], score[3])
    }

    #[inline(always)]
    fn tiled_softmax_score(
        score: [f32; 4],
        row_max_0: f32,
        row_max_1: f32,
        query_valid_0: bool,
        query_valid_1: bool,
    ) -> [f32; 4] {
        [
            if query_valid_0 && score[0] != f32::NEG_INFINITY {
                float::ex2_approx_f32(score[0] - row_max_0)
            } else {
                0.0
            },
            if query_valid_0 && score[1] != f32::NEG_INFINITY {
                float::ex2_approx_f32(score[1] - row_max_0)
            } else {
                0.0
            },
            if query_valid_1 && score[2] != f32::NEG_INFINITY {
                float::ex2_approx_f32(score[2] - row_max_1)
            } else {
                0.0
            },
            if query_valid_1 && score[3] != f32::NEG_INFINITY {
                float::ex2_approx_f32(score[3] - row_max_1)
            } else {
                0.0
            },
        ]
    }

    #[inline(always)]
    fn tiled_scale_output(output: [f32; 4], scale_0: f32, scale_1: f32) -> [f32; 4] {
        [
            output[0] * scale_0,
            output[1] * scale_0,
            output[2] * scale_1,
            output[3] * scale_1,
        ]
    }

    #[inline(always)]
    fn tiled_weight_fragment(score_0: [f32; 4], score_1: [f32; 4]) -> [u32; 4] {
        [
            tcgen05::cvt_f32x2_bf16x2(score_0[0], score_0[1]),
            tcgen05::cvt_f32x2_bf16x2(score_0[2], score_0[3]),
            tcgen05::cvt_f32x2_bf16x2(score_1[0], score_1[1]),
            tcgen05::cvt_f32x2_bf16x2(score_1[2], score_1[3]),
        ]
    }

    #[inline(always)]
    unsafe fn tiled_pv_mma(
        accumulator: [f32; 4],
        weight_fragment: [u32; 4],
        shared_values: *const u16,
        token_base: usize,
        dimension: usize,
        lane_in_group: usize,
    ) -> [f32; 4] {
        let value_row_0 = token_base + lane_in_group * 2;
        let value_row_1 = value_row_0 + 1;
        let value_row_8 = value_row_0 + 8;
        let value_row_9 = value_row_0 + 9;
        // SAFETY: the caller initialized all 64 shared V rows and supplies a
        // dimension in 0..128.
        let value_fragment = unsafe {
            let lo_0 = shared_values
                .add(value_row_0 * SINGLE_DECODE_HEAD_DIM + dimension)
                .read();
            let hi_0 = shared_values
                .add(value_row_1 * SINGLE_DECODE_HEAD_DIM + dimension)
                .read();
            let lo_1 = shared_values
                .add(value_row_8 * SINGLE_DECODE_HEAD_DIM + dimension)
                .read();
            let hi_1 = shared_values
                .add(value_row_9 * SINGLE_DECODE_HEAD_DIM + dimension)
                .read();
            [
                u32::from(lo_0) | (u32::from(hi_0) << 16),
                u32::from(lo_1) | (u32::from(hi_1) << 16),
            ]
        };
        // SAFETY: the caller invokes this uniformly across the full warp.
        unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, weight_fragment, value_fragment) }
    }

    macro_rules! scale_tiled_outputs {
        ($scale_0:expr, $scale_1:expr; $($output:ident),+ $(,)?) => {
            $(
                $output = tiled_scale_output($output, $scale_0, $scale_1);
            )+
        };
    }

    macro_rules! tiled_pv_outputs {
        (
            $weight:expr,
            $shared_values:expr,
            $token_base:expr,
            $lane_group:expr,
            $lane_in_group:expr;
            $($tile:literal => $output:ident),+ $(,)?
        ) => {
            $(
                // SAFETY: the complete shared V tile is initialized and all
                // lanes execute each expanded MMA uniformly.
                $output = unsafe {
                    tiled_pv_mma(
                        $output,
                        $weight,
                        $shared_values,
                        $token_base,
                        $tile * 8 + $lane_group,
                        $lane_in_group,
                    )
                };
            )+
        };
    }

    #[kernel]
    #[launch_bounds(128)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            batch_size >= 1,
            nnz_qo >= 1,
            nnz_kv >= 1,
            num_query_heads == num_kv_heads * 4,
            num_kv_heads >= 1,
            query.len() == nnz_qo * num_query_heads * 128,
            key.len() == nnz_kv * num_kv_heads * 128,
            value.len() == nnz_kv * num_kv_heads * 128,
            qo_indptr.len() == batch_size + 1,
            kv_indptr.len() == batch_size + 1,
            workspace.len() == nnz_qo * num_query_heads * 8 * 130,
        ),
    )]
    pub fn ragged_prefill_bf16_nhd_causal_tiled_gqa4(
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
        mut workspace: DisjointSlice<f32>,
    ) {
        static mut KV_TILE: SharedArray<u32, TILED_KV_SHARED_PAIRS, 16> = SharedArray::UNINIT;

        let block = thread::blockIdx_x() as usize;
        let thread_in_block = thread::threadIdx_x() as usize;
        let warp_in_block = thread_in_block / WARP_THREADS as usize;
        let lane = thread_in_block % WARP_THREADS as usize;
        let partition = block % TILED_PARTITIONS;
        let kv_head = block / TILED_PARTITIONS % num_kv_heads;
        let mut tile_index = block / (TILED_PARTITIONS * num_kv_heads);

        if qo_indptr[0] != 0
            || kv_indptr[0] != 0
            || qo_indptr[batch_size] != nnz_qo as i32
            || kv_indptr[batch_size] != nnz_kv as i32
        {
            return;
        }

        let mut request = 0_usize;
        let mut request_tile = 0_usize;
        let mut qo_start = 0_usize;
        let mut qo_len = 0_usize;
        let mut kv_start = 0_usize;
        let mut kv_len = 0_usize;
        while request < batch_size {
            let request_qo_start = qo_indptr[request];
            let request_qo_end = qo_indptr[request + 1];
            let request_kv_start = kv_indptr[request];
            let request_kv_end = kv_indptr[request + 1];
            if request_qo_start < 0
                || request_qo_end <= request_qo_start
                || request_kv_start < 0
                || request_kv_end <= request_kv_start
                || request_qo_end as usize > nnz_qo
                || request_kv_end as usize > nnz_kv
                || request_qo_end - request_qo_start > request_kv_end - request_kv_start
            {
                return;
            }
            let request_qo_len = (request_qo_end - request_qo_start) as usize;
            let request_tiles = request_qo_len.div_ceil(TILED_QUERY_ROWS);
            if tile_index < request_tiles {
                request_tile = tile_index;
                qo_start = request_qo_start as usize;
                qo_len = request_qo_len;
                kv_start = request_kv_start as usize;
                kv_len = (request_kv_end - request_kv_start) as usize;
                break;
            }
            tile_index -= request_tiles;
            request += 1;
        }
        if request >= batch_size {
            return;
        }

        let query_tile_start = request_tile * TILED_QUERY_ROWS;
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key.as_ptr().cast::<u32>();
        let value_pairs = value.as_ptr().cast::<u32>();
        // SAFETY: one block owns this shared tile and all cross-warp reuse is
        // ordered by block barriers below.
        let kv_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KV_TILE) };

        let mut output_0 = [0.0_f32; 4];
        let mut output_1 = [0.0_f32; 4];
        let mut output_2 = [0.0_f32; 4];
        let mut output_3 = [0.0_f32; 4];
        let mut output_4 = [0.0_f32; 4];
        let mut output_5 = [0.0_f32; 4];
        let mut output_6 = [0.0_f32; 4];
        let mut output_7 = [0.0_f32; 4];
        let mut output_8 = [0.0_f32; 4];
        let mut output_9 = [0.0_f32; 4];
        let mut output_10 = [0.0_f32; 4];
        let mut output_11 = [0.0_f32; 4];
        let mut output_12 = [0.0_f32; 4];
        let mut output_13 = [0.0_f32; 4];
        let mut output_14 = [0.0_f32; 4];
        let mut output_15 = [0.0_f32; 4];
        let mut row_max_0 = f32::NEG_INFINITY;
        let mut row_max_1 = f32::NEG_INFINITY;
        let mut row_sum_0 = 0.0_f32;
        let mut row_sum_1 = 0.0_f32;
        let lane_group = lane / 4;
        let lane_in_group = lane % 4;
        let packed_row_0 = warp_in_block * 16 + lane_group;
        let packed_row_1 = packed_row_0 + 8;
        let query_in_request_0 = query_tile_start + packed_row_0 / TILED_GQA_GROUP_SIZE;
        let query_in_request_1 = query_tile_start + packed_row_1 / TILED_GQA_GROUP_SIZE;
        let query_valid_0 = query_in_request_0 < qo_len;
        let query_valid_1 = query_in_request_1 < qo_len;

        let final_query = usize::min(query_tile_start + TILED_QUERY_ROWS, qo_len);
        let causal_token_end = kv_len - qo_len + final_query;
        let partition_token_start = partition * causal_token_end / TILED_PARTITIONS;
        let partition_token_end = (partition + 1) * causal_token_end / TILED_PARTITIONS;
        let mut kv_tile_start = partition_token_start;
        while kv_tile_start < partition_token_end {
            // SAFETY: metadata validation proves the request range, and the
            // helper zfills the final partial tile without reading past K.
            load_tiled_kv_async!(
                kv_tile,
                key_pairs,
                thread_in_block,
                kv_start,
                kv_tile_start,
                partition_token_end,
                num_kv_heads,
                kv_head,
            );
            thread::sync_threads();

            let mut score_0 = [0.0_f32; 4];
            let mut score_1 = [0.0_f32; 4];
            let mut score_2 = [0.0_f32; 4];
            let mut score_3 = [0.0_f32; 4];
            let mut score_4 = [0.0_f32; 4];
            let mut score_5 = [0.0_f32; 4];
            let mut score_6 = [0.0_f32; 4];
            let mut score_7 = [0.0_f32; 4];
            let mut component_tile = 0_usize;
            while component_tile < SINGLE_DECODE_HEAD_DIM / 16 {
                let component_pair_base = component_tile * 8;
                let mut query_fragment = [0_u32; 4];
                if query_valid_0 {
                    let query_head =
                        kv_head * TILED_GQA_GROUP_SIZE + packed_row_0 % TILED_GQA_GROUP_SIZE;
                    let query_state =
                        (qo_start + query_in_request_0) * num_query_heads + query_head;
                    let pair_base = query_state * BF16_PAIRS_PER_HEAD + component_pair_base;
                    // SAFETY: the valid packed row and fixed component tile
                    // prove both query pairs are in the exact Q span.
                    unsafe {
                        query_fragment[0] = query_pairs.add(pair_base + lane_in_group).read();
                        query_fragment[2] = query_pairs.add(pair_base + 4 + lane_in_group).read();
                    }
                }
                if query_valid_1 {
                    let query_head =
                        kv_head * TILED_GQA_GROUP_SIZE + packed_row_1 % TILED_GQA_GROUP_SIZE;
                    let query_state =
                        (qo_start + query_in_request_1) * num_query_heads + query_head;
                    let pair_base = query_state * BF16_PAIRS_PER_HEAD + component_pair_base;
                    // SAFETY: as above, for the second MMA row owned by this lane.
                    unsafe {
                        query_fragment[1] = query_pairs.add(pair_base + lane_in_group).read();
                        query_fragment[3] = query_pairs.add(pair_base + 4 + lane_in_group).read();
                    }
                }

                // SAFETY: the complete shared K tile is initialized and all
                // lanes execute these eight MMA instructions uniformly.
                unsafe {
                    score_0 = tiled_qk_mma(
                        score_0,
                        query_fragment,
                        kv_tile,
                        lane_group,
                        component_pair_base,
                        lane_in_group,
                    );
                    score_1 = tiled_qk_mma(
                        score_1,
                        query_fragment,
                        kv_tile,
                        8 + lane_group,
                        component_pair_base,
                        lane_in_group,
                    );
                    score_2 = tiled_qk_mma(
                        score_2,
                        query_fragment,
                        kv_tile,
                        16 + lane_group,
                        component_pair_base,
                        lane_in_group,
                    );
                    score_3 = tiled_qk_mma(
                        score_3,
                        query_fragment,
                        kv_tile,
                        24 + lane_group,
                        component_pair_base,
                        lane_in_group,
                    );
                    score_4 = tiled_qk_mma(
                        score_4,
                        query_fragment,
                        kv_tile,
                        32 + lane_group,
                        component_pair_base,
                        lane_in_group,
                    );
                    score_5 = tiled_qk_mma(
                        score_5,
                        query_fragment,
                        kv_tile,
                        40 + lane_group,
                        component_pair_base,
                        lane_in_group,
                    );
                    score_6 = tiled_qk_mma(
                        score_6,
                        query_fragment,
                        kv_tile,
                        48 + lane_group,
                        component_pair_base,
                        lane_in_group,
                    );
                    score_7 = tiled_qk_mma(
                        score_7,
                        query_fragment,
                        kv_tile,
                        56 + lane_group,
                        component_pair_base,
                        lane_in_group,
                    );
                }
                component_tile += 1;
            }

            let causal_end_0 = usize::min(
                kv_len - qo_len + query_in_request_0 + 1,
                partition_token_end,
            );
            let causal_end_1 = usize::min(
                kv_len - qo_len + query_in_request_1 + 1,
                partition_token_end,
            );
            score_0 = tiled_mask_score(
                score_0,
                kv_tile_start + lane_in_group * 2,
                causal_end_0,
                causal_end_1,
                query_valid_0,
                query_valid_1,
                softmax_scale_log2,
            );
            score_1 = tiled_mask_score(
                score_1,
                kv_tile_start + 8 + lane_in_group * 2,
                causal_end_0,
                causal_end_1,
                query_valid_0,
                query_valid_1,
                softmax_scale_log2,
            );
            score_2 = tiled_mask_score(
                score_2,
                kv_tile_start + 16 + lane_in_group * 2,
                causal_end_0,
                causal_end_1,
                query_valid_0,
                query_valid_1,
                softmax_scale_log2,
            );
            score_3 = tiled_mask_score(
                score_3,
                kv_tile_start + 24 + lane_in_group * 2,
                causal_end_0,
                causal_end_1,
                query_valid_0,
                query_valid_1,
                softmax_scale_log2,
            );
            score_4 = tiled_mask_score(
                score_4,
                kv_tile_start + 32 + lane_in_group * 2,
                causal_end_0,
                causal_end_1,
                query_valid_0,
                query_valid_1,
                softmax_scale_log2,
            );
            score_5 = tiled_mask_score(
                score_5,
                kv_tile_start + 40 + lane_in_group * 2,
                causal_end_0,
                causal_end_1,
                query_valid_0,
                query_valid_1,
                softmax_scale_log2,
            );
            score_6 = tiled_mask_score(
                score_6,
                kv_tile_start + 48 + lane_in_group * 2,
                causal_end_0,
                causal_end_1,
                query_valid_0,
                query_valid_1,
                softmax_scale_log2,
            );
            score_7 = tiled_mask_score(
                score_7,
                kv_tile_start + 56 + lane_in_group * 2,
                causal_end_0,
                causal_end_1,
                query_valid_0,
                query_valid_1,
                softmax_scale_log2,
            );
            let mut tile_max_0 = f32::max(
                f32::max(tiled_score_max_0(score_0), tiled_score_max_0(score_1)),
                f32::max(tiled_score_max_0(score_2), tiled_score_max_0(score_3)),
            );
            tile_max_0 = f32::max(
                tile_max_0,
                f32::max(
                    f32::max(tiled_score_max_0(score_4), tiled_score_max_0(score_5)),
                    f32::max(tiled_score_max_0(score_6), tiled_score_max_0(score_7)),
                ),
            );
            let mut tile_max_1 = f32::max(
                f32::max(tiled_score_max_1(score_0), tiled_score_max_1(score_1)),
                f32::max(tiled_score_max_1(score_2), tiled_score_max_1(score_3)),
            );
            tile_max_1 = f32::max(
                tile_max_1,
                f32::max(
                    f32::max(tiled_score_max_1(score_4), tiled_score_max_1(score_5)),
                    f32::max(tiled_score_max_1(score_6), tiled_score_max_1(score_7)),
                ),
            );
            tile_max_0 = f32::max(tile_max_0, warp::shuffle_xor_f32(tile_max_0, 1));
            tile_max_0 = f32::max(tile_max_0, warp::shuffle_xor_f32(tile_max_0, 2));
            tile_max_1 = f32::max(tile_max_1, warp::shuffle_xor_f32(tile_max_1, 1));
            tile_max_1 = f32::max(tile_max_1, warp::shuffle_xor_f32(tile_max_1, 2));

            let next_max_0 = if query_valid_0 {
                f32::max(row_max_0, tile_max_0)
            } else {
                row_max_0
            };
            let next_max_1 = if query_valid_1 {
                f32::max(row_max_1, tile_max_1)
            } else {
                row_max_1
            };
            let previous_scale_0 = if row_sum_0 == 0.0 {
                0.0
            } else {
                float::ex2_approx_f32(row_max_0 - next_max_0)
            };
            let previous_scale_1 = if row_sum_1 == 0.0 {
                0.0
            } else {
                float::ex2_approx_f32(row_max_1 - next_max_1)
            };
            row_sum_0 *= previous_scale_0;
            row_sum_1 *= previous_scale_1;
            row_max_0 = next_max_0;
            row_max_1 = next_max_1;
            scale_tiled_outputs!(
                previous_scale_0,
                previous_scale_1;
                output_0,
                output_1,
                output_2,
                output_3,
                output_4,
                output_5,
                output_6,
                output_7,
                output_8,
                output_9,
                output_10,
                output_11,
                output_12,
                output_13,
                output_14,
                output_15,
            );

            score_0 =
                tiled_softmax_score(score_0, row_max_0, row_max_1, query_valid_0, query_valid_1);
            score_1 =
                tiled_softmax_score(score_1, row_max_0, row_max_1, query_valid_0, query_valid_1);
            score_2 =
                tiled_softmax_score(score_2, row_max_0, row_max_1, query_valid_0, query_valid_1);
            score_3 =
                tiled_softmax_score(score_3, row_max_0, row_max_1, query_valid_0, query_valid_1);
            score_4 =
                tiled_softmax_score(score_4, row_max_0, row_max_1, query_valid_0, query_valid_1);
            score_5 =
                tiled_softmax_score(score_5, row_max_0, row_max_1, query_valid_0, query_valid_1);
            score_6 =
                tiled_softmax_score(score_6, row_max_0, row_max_1, query_valid_0, query_valid_1);
            score_7 =
                tiled_softmax_score(score_7, row_max_0, row_max_1, query_valid_0, query_valid_1);
            let mut tile_sum_0 = score_0[0]
                + score_0[1]
                + score_1[0]
                + score_1[1]
                + score_2[0]
                + score_2[1]
                + score_3[0]
                + score_3[1]
                + score_4[0]
                + score_4[1]
                + score_5[0]
                + score_5[1]
                + score_6[0]
                + score_6[1]
                + score_7[0]
                + score_7[1];
            let mut tile_sum_1 = score_0[2]
                + score_0[3]
                + score_1[2]
                + score_1[3]
                + score_2[2]
                + score_2[3]
                + score_3[2]
                + score_3[3]
                + score_4[2]
                + score_4[3]
                + score_5[2]
                + score_5[3]
                + score_6[2]
                + score_6[3]
                + score_7[2]
                + score_7[3];
            tile_sum_0 += warp::shuffle_xor_f32(tile_sum_0, 1);
            tile_sum_0 += warp::shuffle_xor_f32(tile_sum_0, 2);
            tile_sum_1 += warp::shuffle_xor_f32(tile_sum_1, 1);
            tile_sum_1 += warp::shuffle_xor_f32(tile_sum_1, 2);
            row_sum_0 += tile_sum_0;
            row_sum_1 += tile_sum_1;

            thread::sync_threads();
            // SAFETY: as above, for the matching V tile.
            load_tiled_kv_async!(
                kv_tile,
                value_pairs,
                thread_in_block,
                kv_start,
                kv_tile_start,
                partition_token_end,
                num_kv_heads,
                kv_head,
            );
            thread::sync_threads();

            let shared_values = kv_tile.cast::<u16>();
            let weight_0 = tiled_weight_fragment(score_0, score_1);
            let weight_1 = tiled_weight_fragment(score_2, score_3);
            let weight_2 = tiled_weight_fragment(score_4, score_5);
            let weight_3 = tiled_weight_fragment(score_6, score_7);
            tiled_pv_outputs!(
                weight_0,
                shared_values,
                0,
                lane_group,
                lane_in_group;
                0 => output_0,
                1 => output_1,
                2 => output_2,
                3 => output_3,
                4 => output_4,
                5 => output_5,
                6 => output_6,
                7 => output_7,
                8 => output_8,
                9 => output_9,
                10 => output_10,
                11 => output_11,
                12 => output_12,
                13 => output_13,
                14 => output_14,
                15 => output_15,
            );
            tiled_pv_outputs!(
                weight_1,
                shared_values,
                16,
                lane_group,
                lane_in_group;
                0 => output_0,
                1 => output_1,
                2 => output_2,
                3 => output_3,
                4 => output_4,
                5 => output_5,
                6 => output_6,
                7 => output_7,
                8 => output_8,
                9 => output_9,
                10 => output_10,
                11 => output_11,
                12 => output_12,
                13 => output_13,
                14 => output_14,
                15 => output_15,
            );
            tiled_pv_outputs!(
                weight_2,
                shared_values,
                32,
                lane_group,
                lane_in_group;
                0 => output_0,
                1 => output_1,
                2 => output_2,
                3 => output_3,
                4 => output_4,
                5 => output_5,
                6 => output_6,
                7 => output_7,
                8 => output_8,
                9 => output_9,
                10 => output_10,
                11 => output_11,
                12 => output_12,
                13 => output_13,
                14 => output_14,
                15 => output_15,
            );
            tiled_pv_outputs!(
                weight_3,
                shared_values,
                48,
                lane_group,
                lane_in_group;
                0 => output_0,
                1 => output_1,
                2 => output_2,
                3 => output_3,
                4 => output_4,
                5 => output_5,
                6 => output_6,
                7 => output_7,
                8 => output_8,
                9 => output_9,
                10 => output_10,
                11 => output_11,
                12 => output_12,
                13 => output_13,
                14 => output_14,
                15 => output_15,
            );
            thread::sync_threads();
            kv_tile_start += TILED_KV_ROWS;
        }

        let query_head_0 = kv_head * TILED_GQA_GROUP_SIZE + packed_row_0 % TILED_GQA_GROUP_SIZE;
        let query_head_1 = kv_head * TILED_GQA_GROUP_SIZE + packed_row_1 % TILED_GQA_GROUP_SIZE;
        let state_0 = (qo_start + query_in_request_0) * num_query_heads + query_head_0;
        let state_1 = (qo_start + query_in_request_1) * num_query_heads + query_head_1;
        let partial_0 =
            (state_0 * TILED_PARTITIONS + partition) * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
        let partial_1 =
            (state_1 * TILED_PARTITIONS + partition) * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
        let first_component = lane_in_group * 2;
        let second_component = SINGLE_DECODE_HEAD_DIM / 2 + lane_in_group * 2;
        macro_rules! write_partial_output {
            ($output:ident, $component:expr) => {
                if query_valid_0 {
                    // SAFETY: each packed query row and partition owns one
                    // disjoint partial state; lanes own disjoint components.
                    unsafe {
                        *workspace.get_unchecked_mut(partial_0 + 2 + $component) = $output[0];
                        *workspace.get_unchecked_mut(partial_0 + 3 + $component) = $output[1];
                    }
                }
                if query_valid_1 {
                    // SAFETY: the second packed row owns a different state.
                    unsafe {
                        *workspace.get_unchecked_mut(partial_1 + 2 + $component) = $output[2];
                        *workspace.get_unchecked_mut(partial_1 + 3 + $component) = $output[3];
                    }
                }
            };
        }
        write_partial_output!(output_0, first_component);
        write_partial_output!(output_1, first_component + 8);
        write_partial_output!(output_2, first_component + 16);
        write_partial_output!(output_3, first_component + 24);
        write_partial_output!(output_4, first_component + 32);
        write_partial_output!(output_5, first_component + 40);
        write_partial_output!(output_6, first_component + 48);
        write_partial_output!(output_7, first_component + 56);
        write_partial_output!(output_8, second_component);
        write_partial_output!(output_9, second_component + 8);
        write_partial_output!(output_10, second_component + 16);
        write_partial_output!(output_11, second_component + 24);
        write_partial_output!(output_12, second_component + 32);
        write_partial_output!(output_13, second_component + 40);
        write_partial_output!(output_14, second_component + 48);
        write_partial_output!(output_15, second_component + 56);
        if lane_in_group == 0 {
            if query_valid_0 {
                // SAFETY: one lane owns this partial state's header.
                unsafe {
                    *workspace.get_unchecked_mut(partial_0) = row_max_0;
                    *workspace.get_unchecked_mut(partial_0 + 1) = row_sum_0;
                }
            }
            if query_valid_1 {
                // SAFETY: one lane owns the second partial state's header.
                unsafe {
                    *workspace.get_unchecked_mut(partial_1) = row_max_1;
                    *workspace.get_unchecked_mut(partial_1 + 1) = row_sum_1;
                }
            }
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
            nnz_qo >= 1,
            num_query_heads >= 1,
            workspace.len() == nnz_qo * num_query_heads * 8 * 130,
            output.len() == nnz_qo * num_query_heads * 128,
            lse.len() == nnz_qo * num_query_heads,
        ),
    )]
    pub fn ragged_prefill_bf16_nhd_causal_tiled_gqa4_merge(
        nnz_qo: usize,
        num_query_heads: usize,
        workspace: &[f32],
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let state = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        if state >= nnz_qo * num_query_heads {
            return;
        }

        let first_component = lane * 2;
        let second_component = SINGLE_DECODE_HEAD_DIM / 2 + lane * 2;
        let mut merged_max = f32::NEG_INFINITY;
        let mut merged_sum = 0.0_f32;
        let mut output_0 = 0.0_f32;
        let mut output_1 = 0.0_f32;
        let mut output_2 = 0.0_f32;
        let mut output_3 = 0.0_f32;
        let mut partition = 0_usize;
        while partition < TILED_PARTITIONS {
            let partial =
                (state * TILED_PARTITIONS + partition) * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
            let partial_max = workspace[partial];
            let partial_sum = workspace[partial + 1];
            if partial_sum != 0.0 {
                let value_0 = workspace[partial + 2 + first_component];
                let value_1 = workspace[partial + 3 + first_component];
                let value_2 = workspace[partial + 2 + second_component];
                let value_3 = workspace[partial + 3 + second_component];
                if merged_sum == 0.0 {
                    merged_max = partial_max;
                    merged_sum = partial_sum;
                    output_0 = value_0;
                    output_1 = value_1;
                    output_2 = value_2;
                    output_3 = value_3;
                } else {
                    let next_max = f32::max(merged_max, partial_max);
                    let merged_weight = float::ex2_approx_f32(merged_max - next_max);
                    let partial_weight = float::ex2_approx_f32(partial_max - next_max);
                    merged_sum = merged_sum * merged_weight + partial_sum * partial_weight;
                    output_0 = float::fma_rn_f32(value_0, partial_weight, output_0 * merged_weight);
                    output_1 = float::fma_rn_f32(value_1, partial_weight, output_1 * merged_weight);
                    output_2 = float::fma_rn_f32(value_2, partial_weight, output_2 * merged_weight);
                    output_3 = float::fma_rn_f32(value_3, partial_weight, output_3 * merged_weight);
                    merged_max = next_max;
                }
            }
            partition += 1;
        }

        let inverse_sum = float::div_rn_f32(1.0, merged_sum);
        if lane == 0 {
            // SAFETY: lane zero owns this state LSE.
            unsafe {
                *lse.get_unchecked_mut(state) = merged_max + float::lg2_approx_f32(merged_sum);
            }
        }
        let output_pairs = output.as_mut_ptr().cast::<u32>();
        let first_pair = state * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;
        // SAFETY: each lane owns two packed pairs for this state.
        unsafe {
            output_pairs
                .add(first_pair)
                .write(tcgen05::cvt_f32x2_bf16x2(
                    output_0 * inverse_sum,
                    output_1 * inverse_sum,
                ));
            output_pairs
                .add(second_pair)
                .write(tcgen05::cvt_f32x2_bf16x2(
                    output_2 * inverse_sum,
                    output_3 * inverse_sum,
                ));
        }
    }

    #[inline(always)]
    unsafe fn write_status(
        output: *mut i32,
        code: i32,
        detail0: i32,
        detail1: i32,
        detail2: i32,
        detail3: i32,
    ) {
        // SAFETY: the caller guarantees five writable status words.
        unsafe {
            output.write(code);
            output.add(1).write(detail0);
            output.add(2).write(detail1);
            output.add(3).write(detail2);
            output.add(4).write(detail3);
        }
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

    /// Creates a paged-prefill plan for the selected kernel algorithm.
    pub fn plan_bf16_paged(
        &self,
        spec: Bf16PagedPrefillSpec,
        algorithm: Bf16PagedPrefillAlgorithm,
    ) -> Result<Bf16PagedPrefillPlan, PagedPrefillPlanError> {
        let states = spec
            .nnz_qo()
            .checked_mul(spec.num_query_heads())
            .ok_or(PagedPrefillPlanError::StateCountOutOfRange(usize::MAX))?;
        let states = u32::try_from(states)
            .map_err(|_| PagedPrefillPlanError::StateCountOutOfRange(states))?;
        let metadata_launch = self
            .module
            .prepare_validate_paged_prefill_metadata(LaunchConfig1D::new(1, 1, 0))?;
        let launch = match algorithm {
            Bf16PagedPrefillAlgorithm::Direct => {
                Bf16PagedPrefillLaunch::Direct(self.module.prepare_paged_prefill_bf16_nhd_causal(
                    LaunchConfig1D::new(states, WARP_THREADS, 0),
                )?)
            }
            Bf16PagedPrefillAlgorithm::TokenParallel8 => {
                Bf16PagedPrefillLaunch::TokenParallel8(
                    self.module
                        .prepare_paged_prefill_bf16_nhd_causal_token_parallel8(
                            LaunchConfig1D::new(states, TOKEN_PARALLEL_8_THREADS, 0),
                        )?,
                )
            }
            Bf16PagedPrefillAlgorithm::TokenParallel16 => {
                Bf16PagedPrefillLaunch::TokenParallel16(
                    self.module
                        .prepare_paged_prefill_bf16_nhd_causal_token_parallel16(
                            LaunchConfig1D::new(states, TOKEN_PARALLEL_16_THREADS, 0),
                        )?,
                )
            }
        };
        Ok(Bf16PagedPrefillPlan {
            spec,
            algorithm,
            module: self.module.clone(),
            metadata_launch,
            launch,
        })
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
        let workspace_numel = if algorithm == Bf16RaggedPrefillAlgorithm::TiledGqa4 {
            (states as usize)
                .checked_mul(TILED_PARTITIONS)
                .and_then(|states| states.checked_mul(SINGLE_DECODE_PARTIAL_STATE_WIDTH))
                .ok_or(RaggedPrefillPlanError::WorkspaceElementCountOutOfRange)?
        } else {
            0
        };
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
            Bf16RaggedPrefillAlgorithm::TiledGqa4 => {
                let tile_upper_bound = spec
                    .nnz_qo()
                    .div_ceil(TILED_QUERY_ROWS)
                    .checked_add(spec.batch_size() - 1)
                    .and_then(|tiles| tiles.checked_mul(spec.num_kv_heads()))
                    .and_then(|tiles| tiles.checked_mul(TILED_PARTITIONS))
                    .ok_or(RaggedPrefillPlanError::StateCountOutOfRange(usize::MAX))?;
                let tile_upper_bound = u32::try_from(tile_upper_bound)
                    .map_err(|_| RaggedPrefillPlanError::StateCountOutOfRange(tile_upper_bound))?;
                let partial = self
                    .module
                    .prepare_ragged_prefill_bf16_nhd_causal_tiled_gqa4(LaunchConfig1D::new(
                        tile_upper_bound,
                        TILED_THREADS,
                        0,
                    ))?;
                let merge = self
                    .module
                    .prepare_ragged_prefill_bf16_nhd_causal_tiled_gqa4_merge(
                        LaunchConfig1D::new(states, WARP_THREADS, 0),
                    )?;
                Bf16RaggedPrefillLaunch::TiledGqa4 { partial, merge }
            }
        };
        Ok(Bf16RaggedPrefillPlan {
            spec,
            algorithm,
            workspace_numel,
            module: self.module.clone(),
            launch,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16PagedPrefillAlgorithm {
    Direct,
    TokenParallel8,
    TokenParallel16,
}

#[derive(Clone)]
enum Bf16PagedPrefillLaunch {
    Direct(PreparedLaunch<kernels::__paged_prefill_bf16_nhd_causal_CudaKernel>),
    TokenParallel8(
        PreparedLaunch<kernels::__paged_prefill_bf16_nhd_causal_token_parallel8_CudaKernel>,
    ),
    TokenParallel16(
        PreparedLaunch<kernels::__paged_prefill_bf16_nhd_causal_token_parallel16_CudaKernel>,
    ),
}

#[derive(Clone)]
pub struct Bf16PagedPrefillPlan {
    spec: Bf16PagedPrefillSpec,
    algorithm: Bf16PagedPrefillAlgorithm,
    module: kernels::LoadedModule,
    metadata_launch: PreparedLaunch<kernels::__validate_paged_prefill_metadata_CudaKernel>,
    launch: Bf16PagedPrefillLaunch,
}

impl Bf16PagedPrefillPlan {
    pub const fn spec(&self) -> Bf16PagedPrefillSpec {
        self.spec
    }

    pub const fn algorithm(&self) -> Bf16PagedPrefillAlgorithm {
        self.algorithm
    }

    pub const fn metadata_status_required_numel(&self) -> usize {
        STATUS_PACKET_WORDS
    }

    pub const fn metadata_status_required_bytes(&self) -> usize {
        STATUS_PACKET_WORDS * size_of::<i32>()
    }

    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16PagedPrefillArgs,
    ) -> Result<(), PagedPrefillEnqueueError> {
        let page_indices_len = {
            let resolved = scope.resolve_rrrrrrrrww(
                args.query,
                args.key_pages,
                args.value_pages,
                args.qo_indptr,
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.metadata_status.read(),
                args.output,
                args.lse,
            )?;
            require_paged_exact_len("Q", resolved.first.len(), self.spec.query_numel())?;
            require_paged_exact_len("K_pages", resolved.second.len(), self.spec.kv_pages_numel())?;
            require_paged_exact_len("V_pages", resolved.third.len(), self.spec.kv_pages_numel())?;
            require_paged_exact_len("qo_indptr", resolved.fourth.len(), self.spec.indptr_numel())?;
            require_paged_exact_len(
                "page_indptr",
                resolved.fifth.len(),
                self.spec.indptr_numel(),
            )?;
            if resolved.sixth.len() < self.spec.batch_size() {
                return Err(PagedPrefillEnqueueError::PageIndicesTooShort {
                    minimum: self.spec.batch_size(),
                    actual: resolved.sixth.len(),
                });
            }
            require_paged_exact_len(
                "last_page_len",
                resolved.seventh.len(),
                self.spec.last_page_len_numel(),
            )?;
            require_paged_exact_len(
                "metadata_status",
                resolved.eighth.len(),
                STATUS_PACKET_WORDS,
            )?;
            require_paged_exact_len("O", resolved.ninth.len(), self.spec.output_numel())?;
            require_paged_exact_len("LSE", resolved.tenth.len(), self.spec.lse_numel())?;
            for (operand, address) in [
                ("Q", resolved.first.cu_deviceptr()),
                ("K_pages", resolved.second.cu_deviceptr()),
                ("V_pages", resolved.third.cu_deviceptr()),
                ("O", resolved.ninth.cu_deviceptr()),
            ] {
                require_paged_alignment(operand, address)?;
            }
            resolved.sixth.len()
        };

        scope.require_command_capacity(3)?;
        let status = scope.reserve_device_status(
            args.metadata_status.read(),
            DeviceStatusDecoder::paged_prefill(
                self.spec.batch_size(),
                self.spec.nnz_qo(),
                self.spec.max_num_pages(),
                page_indices_len,
                PAGED_PREFILL_PAGE_SIZE,
            ),
        )?;
        let permit = scope.prepare_command()?;
        let (function, validation_result) = {
            let resolved = scope.resolve_rrrrw(
                args.qo_indptr,
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.metadata_status.write(),
            )?;
            let operation = self.module.validate_paged_prefill_metadata_async(
                &self.metadata_launch,
                self.spec.batch_size(),
                self.spec.nnz_qo(),
                self.spec.max_num_pages(),
                resolved.first,
                resolved.second,
                resolved.third,
                resolved.fourth,
                resolved.fifth,
            );
            let result = enqueue_region_launch(resolved.stream, operation);
            (self.metadata_launch.function().clone(), result)
        };
        record_paged_metadata_launch(scope, status, permit, function, validation_result)?;

        let permit = scope.prepare_command()?;
        let (function, launch_result) = {
            let resolved = scope.resolve_rrrrrrrrww(
                args.query,
                args.key_pages,
                args.value_pages,
                args.qo_indptr,
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.metadata_status.read(),
                args.output,
                args.lse,
            )?;
            let common = (
                self.spec.batch_size(),
                self.spec.nnz_qo(),
                self.spec.max_num_pages(),
                self.spec.num_query_heads(),
                self.spec.num_kv_heads(),
                self.spec.softmax_scale() * core::f32::consts::LOG2_E,
            );
            match &self.launch {
                Bf16PagedPrefillLaunch::Direct(launch) => {
                    let operation = self.module.paged_prefill_bf16_nhd_causal_async(
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
                        resolved.eighth,
                        resolved.ninth,
                        resolved.tenth,
                    );
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
                Bf16PagedPrefillLaunch::TokenParallel8(launch) => {
                    let operation = self
                        .module
                        .paged_prefill_bf16_nhd_causal_token_parallel8_async(
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
                            resolved.eighth,
                            resolved.ninth,
                            resolved.tenth,
                        );
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
                Bf16PagedPrefillLaunch::TokenParallel16(launch) => {
                    let operation = self
                        .module
                        .paged_prefill_bf16_nhd_causal_token_parallel16_async(
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
                            resolved.eighth,
                            resolved.ninth,
                            resolved.tenth,
                        );
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
            }
        };
        record_paged_launch(scope, permit, function, launch_result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16RaggedPrefillAlgorithm {
    Direct,
    TokenParallel8,
    TokenParallel16,
    TiledGqa4,
}

const fn ragged_prefill_algorithm(spec: Bf16RaggedPrefillSpec) -> Bf16RaggedPrefillAlgorithm {
    let average_kv_len = spec.nnz_kv() / spec.batch_size();
    if spec.gqa_group_size() == TILED_GQA_GROUP_SIZE && average_kv_len >= TILED_MIN_AVERAGE_KV_LEN {
        Bf16RaggedPrefillAlgorithm::TiledGqa4
    } else if average_kv_len < TOKEN_PARALLEL_MIN_AVERAGE_KV_LEN {
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
    TiledGqa4 {
        partial: PreparedLaunch<kernels::__ragged_prefill_bf16_nhd_causal_tiled_gqa4_CudaKernel>,
        merge:
            PreparedLaunch<kernels::__ragged_prefill_bf16_nhd_causal_tiled_gqa4_merge_CudaKernel>,
    },
}

#[derive(Clone)]
pub struct Bf16RaggedPrefillPlan {
    spec: Bf16RaggedPrefillSpec,
    algorithm: Bf16RaggedPrefillAlgorithm,
    workspace_numel: usize,
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

    pub const fn workspace_required_numel(&self) -> usize {
        self.workspace_numel
    }

    pub const fn workspace_required_bytes(&self) -> usize {
        self.workspace_required_numel() * size_of::<f32>()
    }

    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16RaggedPrefillArgs,
    ) -> Result<(), RaggedPrefillEnqueueError> {
        if let Bf16RaggedPrefillLaunch::TiledGqa4 { partial, merge } = &self.launch {
            return self.enqueue_tiled_gqa4(scope, args, partial, merge);
        }

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
                    let operation = self.module.ragged_prefill_bf16_nhd_causal_async(
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
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
                Bf16RaggedPrefillLaunch::TokenParallel8(launch) => {
                    let operation = self
                        .module
                        .ragged_prefill_bf16_nhd_causal_token_parallel8_async(
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
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
                Bf16RaggedPrefillLaunch::TokenParallel16(launch) => {
                    let operation = self
                        .module
                        .ragged_prefill_bf16_nhd_causal_token_parallel16_async(
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
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
                Bf16RaggedPrefillLaunch::TiledGqa4 { .. } => unreachable!(),
            }
        };
        record_launch(scope, permit, function, launch_result)
    }

    fn enqueue_tiled_gqa4(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16RaggedPrefillArgs,
        partial: &PreparedLaunch<kernels::__ragged_prefill_bf16_nhd_causal_tiled_gqa4_CudaKernel>,
        merge: &PreparedLaunch<
            kernels::__ragged_prefill_bf16_nhd_causal_tiled_gqa4_merge_CudaKernel,
        >,
    ) -> Result<(), RaggedPrefillEnqueueError> {
        let workspace = args
            .workspace
            .ok_or(RaggedPrefillEnqueueError::MissingWorkspace)?;
        scope.require_command_capacity(2)?;

        {
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
        }

        let partial_permit = scope.prepare_command()?;
        let (partial_function, partial_result) = {
            let resolved = scope.resolve_rrrrrw(
                args.query,
                args.key,
                args.value,
                args.qo_indptr,
                args.kv_indptr,
                workspace.write(),
            )?;
            require_exact_len(
                "workspace",
                resolved.sixth.len(),
                self.workspace_required_numel(),
            )?;
            let operation = self.module.ragged_prefill_bf16_nhd_causal_tiled_gqa4_async(
                partial,
                self.spec.batch_size(),
                self.spec.nnz_qo(),
                self.spec.nnz_kv(),
                self.spec.num_query_heads(),
                self.spec.num_kv_heads(),
                self.spec.softmax_scale() * core::f32::consts::LOG2_E,
                resolved.first,
                resolved.second,
                resolved.third,
                resolved.fourth,
                resolved.fifth,
                resolved.sixth,
            );
            let result = enqueue_region_launch(resolved.stream, operation);
            (partial.function().clone(), result)
        };
        record_launch(scope, partial_permit, partial_function, partial_result)?;

        let merge_permit = scope.prepare_command()?;
        let (merge_function, merge_result) = {
            let resolved = scope.resolve_rww(workspace.read(), args.output, args.lse)?;
            let operation = self
                .module
                .ragged_prefill_bf16_nhd_causal_tiled_gqa4_merge_async(
                    merge,
                    self.spec.nnz_qo(),
                    self.spec.num_query_heads(),
                    resolved.first,
                    resolved.second,
                    resolved.third,
                );
            let result = enqueue_region_launch(resolved.stream, operation);
            (merge.function().clone(), result)
        };
        record_launch(scope, merge_permit, merge_function, merge_result)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bf16PagedPrefillArgs {
    query: Read<bf16>,
    key_pages: Read<bf16>,
    value_pages: Read<bf16>,
    qo_indptr: Read<i32>,
    page_indptr: Read<i32>,
    page_indices: Read<i32>,
    last_page_len: Read<i32>,
    metadata_status: ReadWrite<i32>,
    output: Write<bf16>,
    lse: Write<f32>,
}

impl Bf16PagedPrefillArgs {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        query: Read<bf16>,
        key_pages: Read<bf16>,
        value_pages: Read<bf16>,
        qo_indptr: Read<i32>,
        page_indptr: Read<i32>,
        page_indices: Read<i32>,
        last_page_len: Read<i32>,
        metadata_status: ReadWrite<i32>,
        output: Write<bf16>,
        lse: Write<f32>,
    ) -> Self {
        Self {
            query,
            key_pages,
            value_pages,
            qo_indptr,
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
            output,
            lse,
        }
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
    workspace: Option<ReadWrite<f32>>,
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
            workspace: None,
        }
    }

    pub const fn with_workspace(mut self, workspace: ReadWrite<f32>) -> Self {
        self.workspace = Some(workspace);
        self
    }
}

#[derive(Debug, Error)]
pub enum PagedPrefillPlanError {
    #[error("paged prefill state count {0} exceeds the CUDA grid range")]
    StateCountOutOfRange(usize),
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
}

#[derive(Debug, Error)]
pub enum PagedPrefillEnqueueError {
    #[error("{operand} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("page_indices requires at least {minimum} entries, got {actual}")]
    PageIndicesTooShort { minimum: usize, actual: usize },
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Launch(#[from] DeviceRegionLaunchError),
    #[error(
        "packed paged prefill requires {operand} to be {alignment}-byte aligned, got {address:#x}"
    )]
    MisalignedBuffer {
        operand: &'static str,
        address: u64,
        alignment: u64,
    },
}

#[derive(Debug, Error)]
pub enum RaggedPrefillPlanError {
    #[error("ragged prefill state count {0} exceeds the CUDA grid range")]
    StateCountOutOfRange(usize),
    #[error("ragged prefill workspace element count exceeds usize")]
    WorkspaceElementCountOutOfRange,
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
    #[error("ragged prefill algorithm requires an explicit F32 workspace binding")]
    MissingWorkspace,
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Launch(#[from] DeviceRegionLaunchError),
    #[error(
        "packed ragged prefill requires {operand} to be {alignment}-byte aligned, got {address:#x}"
    )]
    MisalignedBuffer {
        operand: &'static str,
        address: u64,
        alignment: u64,
    },
}

fn require_paged_exact_len(
    operand: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), PagedPrefillEnqueueError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PagedPrefillEnqueueError::LengthMismatch {
            operand,
            expected,
            actual,
        })
    }
}

fn require_paged_alignment(
    operand: &'static str,
    address: u64,
) -> Result<(), PagedPrefillEnqueueError> {
    const ALIGNMENT: u64 = size_of::<u32>() as u64;
    if address.is_multiple_of(ALIGNMENT) {
        Ok(())
    } else {
        Err(PagedPrefillEnqueueError::MisalignedBuffer {
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

fn record_paged_launch(
    scope: &mut CommandScope<'_>,
    permit: CommandPermit,
    function: CudaFunction,
    result: Result<(), DeviceRegionLaunchError>,
) -> Result<(), PagedPrefillEnqueueError> {
    match result {
        Ok(()) => {
            scope.record_cuda_submission(permit, function);
            Ok(())
        }
        Err(error) => {
            if let Some(driver_error) = error.driver_error() {
                scope.record_failed_cuda_submission(permit, function, driver_error);
            }
            Err(error.into())
        }
    }
}

fn record_paged_metadata_launch(
    scope: &mut CommandScope<'_>,
    status: DeviceStatusReservation,
    permit: CommandPermit,
    function: CudaFunction,
    result: Result<(), DeviceRegionLaunchError>,
) -> Result<(), PagedPrefillEnqueueError> {
    match result {
        Ok(()) => {
            scope.record_cuda_submission(permit, function);
            Ok(())
        }
        Err(error) => {
            if let Some(driver_error) = error.driver_error() {
                scope.record_failed_cuda_submission(permit, function, driver_error);
            } else {
                scope.cancel_device_status(status);
            }
            Err(error.into())
        }
    }
}

fn record_launch(
    scope: &mut CommandScope<'_>,
    permit: CommandPermit,
    function: CudaFunction,
    result: Result<(), DeviceRegionLaunchError>,
) -> Result<(), RaggedPrefillEnqueueError> {
    match result {
        Ok(()) => {
            scope.record_cuda_submission(permit, function);
            Ok(())
        }
        Err(error) => {
            if let Some(driver_error) = error.driver_error() {
                scope.record_failed_cuda_submission(permit, function, driver_error);
            }
            Err(error.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_planner_requires_an_explicit_algorithm() {
        let _planner: fn(
            &PrefillProvider,
            Bf16PagedPrefillSpec,
            Bf16PagedPrefillAlgorithm,
        ) -> Result<Bf16PagedPrefillPlan, PagedPrefillPlanError> = PrefillProvider::plan_bf16_paged;
    }

    #[test]
    fn ragged_algorithm_keeps_short_kv_direct_and_parallelizes_long_kv() {
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
            Bf16RaggedPrefillAlgorithm::TiledGqa4
        );
        let expected_workspace =
            grouped.nnz_qo() * grouped.num_query_heads() * 8 * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
        assert_eq!(expected_workspace, 1_597_440);
    }
}
