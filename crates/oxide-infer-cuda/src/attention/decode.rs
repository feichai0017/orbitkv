//! cuda-oxide provider for BF16 single-request decode attention.

use crate::command::{
    CommandError, CommandPermit, CommandScope, DeviceStatusReservation, Read, ReadWrite, Write,
};
use crate::device_status::{
    DeviceStatusDecoder, STATUS_ELEMENT_COUNT_OVERFLOW, STATUS_EMPTY_PAGED_REQUEST,
    STATUS_INVALID_LAST_PAGE_LENGTH, STATUS_INVALID_PAGE_INDPTR_START,
    STATUS_NON_MONOTONIC_PAGE_INDPTR, STATUS_PACKET_WORDS, STATUS_PAGE_INDEX_OUT_OF_RANGE,
    STATUS_PAGE_INDICES_LENGTH_MISMATCH, STATUS_SUCCESS,
};
use crate::memory::{DeviceRegionLaunchError, enqueue_region_launch};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, SharedArray, convert, cuda_module, device, float, kernel, launch_bounds,
    launch_contract, tcgen05, thread, warp,
};
use half::bf16;
use oxide_infer::{
    Bf16PagedBatchDecodeSpec, Bf16SingleDecodeSpec, Bf16SingleDecodeSplitKSpec,
    PAGED_BATCH_DECODE_PAGE_SIZE, PagedKvLayout, SINGLE_DECODE_HEAD_DIM,
    SINGLE_DECODE_PARTIAL_STATE_WIDTH,
};
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

const WARP_THREADS: u32 = 32;
const PAGED_BATCH_DECODE_WARPS_PER_BLOCK: usize = 8;
const PAGED_BATCH_DECODE_BLOCK_THREADS: u32 =
    WARP_THREADS * PAGED_BATCH_DECODE_WARPS_PER_BLOCK as u32;
const PAGED_BATCH_DECODE_SHARED_NUMEL: usize =
    PAGED_BATCH_DECODE_WARPS_PER_BLOCK * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
const SPLIT_K_MERGE_WARPS_PER_BLOCK: usize = 8;
const SPLIT_K_MERGE_BLOCK_THREADS: u32 = WARP_THREADS * SPLIT_K_MERGE_WARPS_PER_BLOCK as u32;
const SPLIT_K_MERGE_SHARED_NUMEL: usize =
    SPLIT_K_MERGE_WARPS_PER_BLOCK * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
const BF16_PAIRS_PER_HEAD: usize = SINGLE_DECODE_HEAD_DIM / 2;
const BF16_PAIRS_PER_LANE: usize = BF16_PAIRS_PER_HEAD / WARP_THREADS as usize;

const _: () = {
    assert!(SINGLE_DECODE_HEAD_DIM == 128);
    assert!(PAGED_BATCH_DECODE_PAGE_SIZE == 16);
    assert!(PAGED_BATCH_DECODE_BLOCK_THREADS == 256);
    assert!(SPLIT_K_MERGE_BLOCK_THREADS == 256);
    assert!(BF16_PAIRS_PER_LANE == 2);
    assert!(core::mem::size_of::<bf16>() == core::mem::size_of::<u16>());
    assert!(core::mem::align_of::<bf16>() == core::mem::align_of::<u16>());
};

#[cuda_module]
#[allow(clippy::too_many_arguments)]
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
            batch_size >= 1,
            max_num_pages >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == batch_size * num_query_heads * 128,
            key_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            value_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
            output.len() == batch_size * num_query_heads * 128,
            lse.len() == batch_size * num_query_heads,
        ),
    )]
    pub fn paged_batch_decode_bf16_nhd(
        batch_size: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        metadata_is_trusted: bool,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        output: DisjointSlice<bf16>,
        lse: DisjointSlice<f32>,
    ) {
        let _ = max_num_pages;
        paged_batch_decode_bf16_impl::<false>(
            batch_size,
            num_query_heads,
            num_kv_heads,
            softmax_scale_log2,
            metadata_is_trusted,
            query,
            key_pages,
            value_pages,
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
            output,
            lse,
        );
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
            max_num_pages >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == batch_size * num_query_heads * 128,
            key_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            value_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
            output.len() == batch_size * num_query_heads * 128,
            lse.len() == batch_size * num_query_heads,
        ),
    )]
    pub fn paged_batch_decode_bf16_hnd(
        batch_size: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        metadata_is_trusted: bool,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        output: DisjointSlice<bf16>,
        lse: DisjointSlice<f32>,
    ) {
        let _ = max_num_pages;
        paged_batch_decode_bf16_impl::<true>(
            batch_size,
            num_query_heads,
            num_kv_heads,
            softmax_scale_log2,
            metadata_is_trusted,
            query,
            key_pages,
            value_pages,
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
            output,
            lse,
        );
    }

    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(
        domain = 1,
        block = (1, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            batch_size >= 1,
            max_num_pages >= 1,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
        ),
    )]
    pub fn validate_paged_batch_decode_metadata(
        batch_size: usize,
        max_num_pages: usize,
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        mut metadata_status: DisjointSlice<i32>,
    ) {
        if thread::blockIdx_x() != 0 || thread::threadIdx_x() != 0 {
            return;
        }
        let output = metadata_status.as_mut_ptr();
        // SAFETY: the launch contract proves the status packet span.
        unsafe { write_status(output, STATUS_SUCCESS, 0, 0, 0, 0) };
        if page_indptr[0] != 0 {
            // SAFETY: as above.
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

        let mut request = 0_usize;
        while request < batch_size {
            let start = page_indptr[request];
            let end = page_indptr[request + 1];
            if end < start {
                // SAFETY: the launch contract proves the status packet span.
                unsafe {
                    write_status(
                        output,
                        STATUS_NON_MONOTONIC_PAGE_INDPTR,
                        request as i32,
                        start,
                        end,
                        0,
                    )
                };
                return;
            }
            if end == start {
                // SAFETY: the launch contract proves the status packet span.
                unsafe {
                    write_status(output, STATUS_EMPTY_PAGED_REQUEST, request as i32, 0, 0, 0)
                };
                return;
            }
            let tail = last_page_len[request];
            if !(1..=PAGED_BATCH_DECODE_PAGE_SIZE as i32).contains(&tail) {
                // SAFETY: the launch contract proves the status packet span.
                unsafe {
                    write_status(
                        output,
                        STATUS_INVALID_LAST_PAGE_LENGTH,
                        request as i32,
                        tail,
                        0,
                        0,
                    )
                };
                return;
            }
            request += 1;
        }

        let terminal = page_indptr[batch_size];
        if terminal < 0 {
            // SAFETY: the launch contract proves the status packet span.
            unsafe { write_status(output, STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 0) };
            return;
        }
        if terminal as usize != page_indices.len() {
            // SAFETY: the launch contract proves the status packet span.
            unsafe {
                write_status(
                    output,
                    STATUS_PAGE_INDICES_LENGTH_MISMATCH,
                    terminal,
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
                // SAFETY: the launch contract proves the status packet span.
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

    #[device]
    fn paged_batch_decode_bf16_impl<const HND: bool>(
        batch_size: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        metadata_is_trusted: bool,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let state_index = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let state_count = batch_size * num_query_heads;
        if state_index >= state_count
            || lane >= WARP_THREADS as usize
            || (!metadata_is_trusted && metadata_status[0] != STATUS_SUCCESS)
        {
            return;
        }
        let request = state_index / num_query_heads;
        let query_head = state_index % num_query_heads;
        let page_start = page_indptr[request];
        let page_end = page_indptr[request + 1];
        let tail_len = last_page_len[request];

        let num_pages = (page_end - page_start) as usize;
        let kv_len = (num_pages - 1) * PAGED_BATCH_DECODE_PAGE_SIZE + tail_len as usize;
        let group_size = num_query_heads / num_kv_heads;
        let kv_head = query_head / group_size;
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key_pages.as_ptr().cast::<u32>();
        let value_pairs = value_pages.as_ptr().cast::<u32>();
        let output_pairs = output.as_mut_ptr().cast::<u32>();
        let first_pair = state_index * BF16_PAIRS_PER_HEAD + lane;
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
            let page_slot = token / PAGED_BATCH_DECODE_PAGE_SIZE;
            let page_offset = token % PAGED_BATCH_DECODE_PAGE_SIZE;
            let physical_page = page_indices[page_start as usize + page_slot] as usize;
            let kv_pair_offset = if HND {
                (((physical_page * num_kv_heads + kv_head) * PAGED_BATCH_DECODE_PAGE_SIZE
                    + page_offset)
                    * BF16_PAIRS_PER_HEAD)
                    + lane
            } else {
                (((physical_page * PAGED_BATCH_DECODE_PAGE_SIZE + page_offset) * num_kv_heads
                    + kv_head)
                    * BF16_PAIRS_PER_HEAD)
                    + lane
            };

            // SAFETY: the preceding stream-ordered validator proved the page
            // index is in range. The layout is fixed by this entry point. Each
            // lane reads two disjoint pairs from the page-pool span.
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
            // SAFETY: only lane zero writes this request and query-head slot.
            unsafe {
                *lse.get_unchecked_mut(state_index) =
                    max_score_log2 + float::lg2_approx_f32(normalizer);
            }
        }
        inverse_normalizer = warp::shuffle_f32(inverse_normalizer, 0);

        // SAFETY: each lane owns two packed output pairs. The host validates
        // four-byte alignment and the launch contract proves the exact span.
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

    #[device]
    fn paged_batch_decode_bf16_token_parallel_impl<const HND: bool>(
        batch_size: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        metadata_is_trusted: bool,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        partial_states: *mut f32,
        mut output: DisjointSlice<bf16>,
        mut lse: DisjointSlice<f32>,
    ) {
        let state_index = thread::blockIdx_x() as usize;
        let thread_in_block = thread::threadIdx_x() as usize;
        let warp_in_block = thread_in_block / WARP_THREADS as usize;
        let lane = thread_in_block % WARP_THREADS as usize;
        let state_count = batch_size * num_query_heads;
        if state_index >= state_count
            || (!metadata_is_trusted && metadata_status[0] != STATUS_SUCCESS)
        {
            return;
        }
        let request = state_index / num_query_heads;
        let query_head = state_index % num_query_heads;
        let page_start = page_indptr[request];
        let page_end = page_indptr[request + 1];
        let tail_len = last_page_len[request];

        let num_pages = (page_end - page_start) as usize;
        let kv_len = (num_pages - 1) * PAGED_BATCH_DECODE_PAGE_SIZE + tail_len as usize;
        let group_size = num_query_heads / num_kv_heads;
        let kv_head = query_head / group_size;
        let query_pairs = query.as_ptr().cast::<u32>();
        let key_pairs = key_pages.as_ptr().cast::<u32>();
        let value_pairs = value_pages.as_ptr().cast::<u32>();
        let first_pair = state_index * BF16_PAIRS_PER_HEAD + lane;
        let second_pair = first_pair + WARP_THREADS as usize;

        // SAFETY: the host validates alignment and the launch contract proves
        // the exact query span. Every warp reads the same query head.
        let (query_0, query_1, query_2, query_3) = unsafe {
            let (query_0, query_1) = convert::cvt_f32x2_bf16x2(query_pairs.add(first_pair).read());
            let (query_2, query_3) = convert::cvt_f32x2_bf16x2(query_pairs.add(second_pair).read());
            (query_0, query_1, query_2, query_3)
        };

        let token_start = warp_in_block * kv_len / PAGED_BATCH_DECODE_WARPS_PER_BLOCK;
        let token_end = (warp_in_block + 1) * kv_len / PAGED_BATCH_DECODE_WARPS_PER_BLOCK;
        let mut output_0 = 0.0_f32;
        let mut output_1 = 0.0_f32;
        let mut output_2 = 0.0_f32;
        let mut output_3 = 0.0_f32;
        let mut max_score_log2 = f32::NEG_INFINITY;
        let mut normalizer = 0.0_f32;
        let mut token = token_start;
        while token < token_end {
            let page_slot = token / PAGED_BATCH_DECODE_PAGE_SIZE;
            let page_offset = token % PAGED_BATCH_DECODE_PAGE_SIZE;
            let physical_page = page_indices[page_start as usize + page_slot] as usize;
            let kv_pair_offset = if HND {
                (((physical_page * num_kv_heads + kv_head) * PAGED_BATCH_DECODE_PAGE_SIZE
                    + page_offset)
                    * BF16_PAIRS_PER_HEAD)
                    + lane
            } else {
                (((physical_page * PAGED_BATCH_DECODE_PAGE_SIZE + page_offset) * num_kv_heads
                    + kv_head)
                    * BF16_PAIRS_PER_HEAD)
                    + lane
            };

            // SAFETY: the preceding stream-ordered validator proved every
            // referenced physical page is in range.
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
        // SAFETY: each CUDA block owns this shared allocation. Each warp
        // writes one disjoint state before the block barrier.
        // SAFETY: each warp owns one complete 130-float partial state.
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

        let mut merged_max_log2 = 0.0_f32;
        let mut merged_normalizer = 0.0_f32;
        let mut merged_output_0 = 0.0_f32;
        let mut merged_output_1 = 0.0_f32;
        let mut merged_output_2 = 0.0_f32;
        let mut merged_output_3 = 0.0_f32;
        let mut partial = 0_usize;
        while partial < PAGED_BATCH_DECODE_WARPS_PER_BLOCK {
            let partial_offset = partial * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
            // SAFETY: all warps initialized their partial states before the
            // barrier and warp zero only reads after it.
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
            if partial == 0 {
                merged_max_log2 = partial_max;
                merged_normalizer = partial_normalizer;
                merged_output_0 = value_0;
                merged_output_1 = value_1;
                merged_output_2 = value_2;
                merged_output_3 = value_3;
            } else if partial_normalizer != 0.0 {
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
            partial += 1;
        }

        let inverse_normalizer = float::div_rn_f32(1.0, merged_normalizer);
        if lane == 0 {
            // SAFETY: only lane zero writes this request and query-head slot.
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
            max_num_pages >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == batch_size * num_query_heads * 128,
            key_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            value_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
            output.len() == batch_size * num_query_heads * 128,
            lse.len() == batch_size * num_query_heads,
        ),
    )]
    pub fn paged_batch_decode_bf16_nhd_token_parallel(
        batch_size: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        metadata_is_trusted: bool,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        output: DisjointSlice<bf16>,
        lse: DisjointSlice<f32>,
    ) {
        static mut PARTIAL_STATES: SharedArray<f32, PAGED_BATCH_DECODE_SHARED_NUMEL> =
            SharedArray::UNINIT;
        let _ = max_num_pages;
        // SAFETY: this kernel entry owns its block-local shared allocation.
        let partial_states = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL_STATES) };
        paged_batch_decode_bf16_token_parallel_impl::<false>(
            batch_size,
            num_query_heads,
            num_kv_heads,
            softmax_scale_log2,
            metadata_is_trusted,
            query,
            key_pages,
            value_pages,
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
            partial_states,
            output,
            lse,
        );
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
            max_num_pages >= 1,
            num_query_heads >= 1,
            num_kv_heads >= 1,
            query.len() == batch_size * num_query_heads * 128,
            key_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            value_pages.len() == max_num_pages * 16 * num_kv_heads * 128,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            metadata_status.len() == 5,
            output.len() == batch_size * num_query_heads * 128,
            lse.len() == batch_size * num_query_heads,
        ),
    )]
    pub fn paged_batch_decode_bf16_hnd_token_parallel(
        batch_size: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        softmax_scale_log2: f32,
        metadata_is_trusted: bool,
        query: &[bf16],
        key_pages: &[bf16],
        value_pages: &[bf16],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        metadata_status: &[i32],
        output: DisjointSlice<bf16>,
        lse: DisjointSlice<f32>,
    ) {
        static mut PARTIAL_STATES: SharedArray<f32, PAGED_BATCH_DECODE_SHARED_NUMEL> =
            SharedArray::UNINIT;
        let _ = max_num_pages;
        // SAFETY: this kernel entry owns its block-local shared allocation.
        let partial_states = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL_STATES) };
        paged_batch_decode_bf16_token_parallel_impl::<true>(
            batch_size,
            num_query_heads,
            num_kv_heads,
            softmax_scale_log2,
            metadata_is_trusted,
            query,
            key_pages,
            value_pages,
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
            partial_states,
            output,
            lse,
        );
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
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
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
        static mut MERGE_STATES: SharedArray<f32, SPLIT_K_MERGE_SHARED_NUMEL> = SharedArray::UNINIT;

        let query_head = thread::blockIdx_x() as usize;
        let thread_in_block = thread::threadIdx_x() as usize;
        let warp_in_block = thread_in_block / WARP_THREADS as usize;
        let lane = thread_in_block % WARP_THREADS as usize;
        if query_head >= num_query_heads {
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
        let active_warps = usize::min(partitions, SPLIT_K_MERGE_WARPS_PER_BLOCK);
        let partition_start = warp_in_block * partitions / active_warps;
        let partition_end = (warp_in_block + 1) * partitions / active_warps;
        let mut partition = partition_start;
        while warp_in_block < active_warps && partition < partition_end {
            let state_offset =
                (query_head * partitions + partition) * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
            let partition_max_log2 = workspace[state_offset];
            let partition_normalizer = workspace[state_offset + 1];
            let value_0 = workspace[state_offset + 2 + first_component];
            let value_1 = workspace[state_offset + 3 + first_component];
            let value_2 = workspace[state_offset + 2 + second_component];
            let value_3 = workspace[state_offset + 3 + second_component];

            if partition == partition_start {
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

        // SAFETY: every active warp owns one disjoint 130-float state. Lane
        // zero writes its header and every lane writes four distinct values.
        let merge_states = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MERGE_STATES) };
        if warp_in_block < active_warps {
            let state_offset = warp_in_block * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
            // SAFETY: this active warp owns the state at `state_offset`.
            // Lane zero writes its header and every lane writes four distinct
            // weighted-value elements.
            unsafe {
                if lane == 0 {
                    merge_states.add(state_offset).write(merged_max_log2);
                    merge_states.add(state_offset + 1).write(merged_normalizer);
                }
                merge_states
                    .add(state_offset + 2 + first_component)
                    .write(output_0);
                merge_states
                    .add(state_offset + 3 + first_component)
                    .write(output_1);
                merge_states
                    .add(state_offset + 2 + second_component)
                    .write(output_2);
                merge_states
                    .add(state_offset + 3 + second_component)
                    .write(output_3);
            }
        }
        thread::sync_threads();

        if warp_in_block != 0 {
            return;
        }

        let mut warp_state = 0_usize;
        while warp_state < active_warps {
            let state_offset = warp_state * SINGLE_DECODE_PARTIAL_STATE_WIDTH;
            // SAFETY: all active warps initialized their complete states
            // before the block barrier, and warp zero only reads afterward.
            let (partition_max_log2, partition_normalizer, value_0, value_1, value_2, value_3) = unsafe {
                (
                    merge_states.add(state_offset).read(),
                    merge_states.add(state_offset + 1).read(),
                    merge_states.add(state_offset + 2 + first_component).read(),
                    merge_states.add(state_offset + 3 + first_component).read(),
                    merge_states.add(state_offset + 2 + second_component).read(),
                    merge_states.add(state_offset + 3 + second_component).read(),
                )
            };
            if warp_state == 0 {
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
            warp_state += 1;
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

/// Loaded cuda-oxide module for decode kernels.
#[derive(Clone, Debug)]
pub struct DecodeProvider {
    module: kernels::LoadedModule,
}

impl DecodeProvider {
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
                    SPLIT_K_MERGE_BLOCK_THREADS,
                    0,
                ))?;
        Ok(Bf16SingleDecodeSplitKPlan {
            spec,
            module: self.module.clone(),
            partial_launch,
            merge_launch,
        })
    }

    /// Creates one immutable BF16 paged batch-decode launch plan.
    pub fn plan_bf16_paged_batch(
        &self,
        spec: Bf16PagedBatchDecodeSpec,
    ) -> Result<Bf16PagedBatchDecodePlan, PagedBatchDecodePlanError> {
        self.plan_bf16_paged_batch_with_algorithm(spec, paged_batch_decode_algorithm(spec))
    }

    /// Creates one immutable BF16 paged batch-decode launch plan with an explicit algorithm.
    pub fn plan_bf16_paged_batch_with_algorithm(
        &self,
        spec: Bf16PagedBatchDecodeSpec,
        algorithm: Bf16PagedBatchDecodeAlgorithm,
    ) -> Result<Bf16PagedBatchDecodePlan, PagedBatchDecodePlanError> {
        let states = spec
            .batch_size()
            .checked_mul(spec.num_query_heads())
            .ok_or(PagedBatchDecodePlanError::StateCountOutOfRange(usize::MAX))?;
        let states = u32::try_from(states)
            .map_err(|_| PagedBatchDecodePlanError::StateCountOutOfRange(states))?;
        let metadata_launch = self
            .module
            .prepare_validate_paged_batch_decode_metadata(LaunchConfig1D::new(1, 1, 0))?;
        let launch =
            match (algorithm, spec.kv_layout()) {
                (Bf16PagedBatchDecodeAlgorithm::Direct, PagedKvLayout::Nhd) => {
                    Bf16PagedBatchDecodeLaunch::DirectNhd(
                        self.module
                            .prepare_paged_batch_decode_bf16_nhd(LaunchConfig1D::new(
                                states,
                                WARP_THREADS,
                                0,
                            ))?,
                    )
                }
                (Bf16PagedBatchDecodeAlgorithm::Direct, PagedKvLayout::Hnd) => {
                    Bf16PagedBatchDecodeLaunch::DirectHnd(
                        self.module
                            .prepare_paged_batch_decode_bf16_hnd(LaunchConfig1D::new(
                                states,
                                WARP_THREADS,
                                0,
                            ))?,
                    )
                }
                (Bf16PagedBatchDecodeAlgorithm::TokenParallel8, PagedKvLayout::Nhd) => {
                    Bf16PagedBatchDecodeLaunch::TokenParallel8Nhd(
                        self.module
                            .prepare_paged_batch_decode_bf16_nhd_token_parallel(
                                LaunchConfig1D::new(states, PAGED_BATCH_DECODE_BLOCK_THREADS, 0),
                            )?,
                    )
                }
                (Bf16PagedBatchDecodeAlgorithm::TokenParallel8, PagedKvLayout::Hnd) => {
                    Bf16PagedBatchDecodeLaunch::TokenParallel8Hnd(
                        self.module
                            .prepare_paged_batch_decode_bf16_hnd_token_parallel(
                                LaunchConfig1D::new(states, PAGED_BATCH_DECODE_BLOCK_THREADS, 0),
                            )?,
                    )
                }
            };
        Ok(Bf16PagedBatchDecodePlan {
            spec,
            algorithm,
            module: self.module.clone(),
            metadata_launch,
            launch,
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
            let operation = self.module.single_decode_bf16_nhd_async(
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
            let result = enqueue_region_launch(resolved.stream, operation);
            (self.launch.function().clone(), result)
        };
        record_launch(scope, permit, function, launch_result)
    }
}

/// Immutable launch plan for BF16 paged batch decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16PagedBatchDecodeAlgorithm {
    Direct,
    TokenParallel8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PagedBatchDecodeHostProfile {
    preflight_host_nanoseconds: u64,
    metadata_host_nanoseconds: u64,
    attention_host_nanoseconds: u64,
}

impl PagedBatchDecodeHostProfile {
    pub(crate) const fn preflight_host_nanoseconds(self) -> u64 {
        self.preflight_host_nanoseconds
    }

    pub(crate) const fn metadata_host_nanoseconds(self) -> u64 {
        self.metadata_host_nanoseconds
    }

    pub(crate) const fn attention_host_nanoseconds(self) -> u64 {
        self.attention_host_nanoseconds
    }
}

const fn paged_batch_decode_algorithm(
    spec: Bf16PagedBatchDecodeSpec,
) -> Bf16PagedBatchDecodeAlgorithm {
    if spec.num_query_heads() == spec.num_kv_heads() {
        Bf16PagedBatchDecodeAlgorithm::Direct
    } else {
        Bf16PagedBatchDecodeAlgorithm::TokenParallel8
    }
}

#[derive(Clone)]
enum Bf16PagedBatchDecodeLaunch {
    DirectNhd(PreparedLaunch<kernels::__paged_batch_decode_bf16_nhd_CudaKernel>),
    DirectHnd(PreparedLaunch<kernels::__paged_batch_decode_bf16_hnd_CudaKernel>),
    TokenParallel8Nhd(
        PreparedLaunch<kernels::__paged_batch_decode_bf16_nhd_token_parallel_CudaKernel>,
    ),
    TokenParallel8Hnd(
        PreparedLaunch<kernels::__paged_batch_decode_bf16_hnd_token_parallel_CudaKernel>,
    ),
}

#[derive(Clone)]
pub struct Bf16PagedBatchDecodePlan {
    spec: Bf16PagedBatchDecodeSpec,
    algorithm: Bf16PagedBatchDecodeAlgorithm,
    module: kernels::LoadedModule,
    metadata_launch: PreparedLaunch<kernels::__validate_paged_batch_decode_metadata_CudaKernel>,
    launch: Bf16PagedBatchDecodeLaunch,
}

impl Bf16PagedBatchDecodePlan {
    pub const fn spec(&self) -> Bf16PagedBatchDecodeSpec {
        self.spec
    }

    pub const fn algorithm(&self) -> Bf16PagedBatchDecodeAlgorithm {
        self.algorithm
    }

    pub const fn metadata_status_required_numel(&self) -> usize {
        STATUS_PACKET_WORDS
    }

    pub const fn metadata_status_required_bytes(&self) -> usize {
        STATUS_PACKET_WORDS * size_of::<i32>()
    }

    /// Enqueues the fixed plan into a checked command scope.
    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16PagedBatchDecodeArgs,
    ) -> Result<(), PagedBatchDecodeEnqueueError> {
        self.enqueue_into_impl(scope, args, false, None)
    }

    pub(crate) fn enqueue_into_profiled(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16PagedBatchDecodeArgs,
    ) -> Result<PagedBatchDecodeHostProfile, PagedBatchDecodeEnqueueError> {
        let mut profile = PagedBatchDecodeHostProfile::default();
        self.enqueue_into_impl(scope, args, false, Some(&mut profile))?;
        Ok(profile)
    }

    /// Enqueues after a trusted adapter has established metadata validity.
    pub fn enqueue_trusted_metadata_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: TrustedBf16PagedBatchDecodeArgs,
    ) -> Result<(), PagedBatchDecodeEnqueueError> {
        self.require_trusted_spec(args.spec)?;
        self.enqueue_into_impl(scope, args.args, true, None)
    }

    pub(crate) fn enqueue_trusted_metadata_into_profiled(
        &self,
        scope: &mut CommandScope<'_>,
        args: TrustedBf16PagedBatchDecodeArgs,
    ) -> Result<PagedBatchDecodeHostProfile, PagedBatchDecodeEnqueueError> {
        self.require_trusted_spec(args.spec)?;
        let mut profile = PagedBatchDecodeHostProfile::default();
        self.enqueue_into_impl(scope, args.args, true, Some(&mut profile))?;
        Ok(profile)
    }

    fn require_trusted_spec(
        &self,
        trusted_spec: Bf16PagedBatchDecodeSpec,
    ) -> Result<(), PagedBatchDecodeEnqueueError> {
        if trusted_spec == self.spec {
            Ok(())
        } else {
            Err(PagedBatchDecodeEnqueueError::TrustedMetadataPlanMismatch)
        }
    }

    fn enqueue_into_impl(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16PagedBatchDecodeArgs,
        metadata_is_trusted: bool,
        mut profile: Option<&mut PagedBatchDecodeHostProfile>,
    ) -> Result<(), PagedBatchDecodeEnqueueError> {
        let preflight_started = profile.is_some().then(std::time::Instant::now);
        let page_indices_len = {
            let resolved = scope.resolve_rrrrrrrww(
                args.query,
                args.key_pages,
                args.value_pages,
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
            require_paged_exact_len(
                "page_indptr",
                resolved.fourth.len(),
                self.spec.page_indptr_numel(),
            )?;
            if resolved.fifth.len() < self.spec.batch_size() {
                return Err(PagedBatchDecodeEnqueueError::PageIndicesTooShort {
                    minimum: self.spec.batch_size(),
                    actual: resolved.fifth.len(),
                });
            }
            require_paged_exact_len(
                "last_page_len",
                resolved.sixth.len(),
                self.spec.last_page_len_numel(),
            )?;
            require_paged_exact_len(
                "metadata_status",
                resolved.seventh.len(),
                STATUS_PACKET_WORDS,
            )?;
            require_paged_exact_len("O", resolved.eighth.len(), self.spec.output_numel())?;
            require_paged_exact_len("LSE", resolved.ninth.len(), self.spec.lse_numel())?;
            for (operand, address) in [
                ("Q", resolved.first.cu_deviceptr()),
                ("K_pages", resolved.second.cu_deviceptr()),
                ("V_pages", resolved.third.cu_deviceptr()),
                ("O", resolved.eighth.cu_deviceptr()),
            ] {
                require_paged_packed_alignment(operand, address)?;
            }
            resolved.fifth.len()
        };

        scope.require_command_capacity(if metadata_is_trusted { 1 } else { 3 })?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.preflight_host_nanoseconds = elapsed_nanoseconds(preflight_started);
        }
        if !metadata_is_trusted {
            let metadata_started = profile.is_some().then(std::time::Instant::now);
            let status = scope.reserve_device_status(
                args.metadata_status.read(),
                DeviceStatusDecoder::paged_batch_decode(
                    self.spec.batch_size(),
                    self.spec.max_num_pages(),
                    page_indices_len,
                    PAGED_BATCH_DECODE_PAGE_SIZE,
                ),
            )?;
            let permit = scope.prepare_command()?;
            let (function, validation_result) = {
                let resolved = scope.resolve_rrrw(
                    args.page_indptr,
                    args.page_indices,
                    args.last_page_len,
                    args.metadata_status.write(),
                )?;
                let operation = self.module.validate_paged_batch_decode_metadata_async(
                    &self.metadata_launch,
                    self.spec.batch_size(),
                    self.spec.max_num_pages(),
                    resolved.first,
                    resolved.second,
                    resolved.third,
                    resolved.fourth,
                );
                let result = enqueue_region_launch(resolved.stream, operation);
                (self.metadata_launch.function().clone(), result)
            };
            record_paged_metadata_launch(scope, status, permit, function, validation_result)?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.metadata_host_nanoseconds = elapsed_nanoseconds(metadata_started);
            }
        }

        let attention_started = profile.is_some().then(std::time::Instant::now);
        let permit = scope.prepare_command()?;
        let (function, launch_result) = {
            let resolved = scope.resolve_rrrrrrrww(
                args.query,
                args.key_pages,
                args.value_pages,
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.metadata_status.read(),
                args.output,
                args.lse,
            )?;
            let common = (
                self.spec.batch_size(),
                self.spec.max_num_pages(),
                self.spec.num_query_heads(),
                self.spec.num_kv_heads(),
                self.spec.softmax_scale() * core::f32::consts::LOG2_E,
            );
            match &self.launch {
                Bf16PagedBatchDecodeLaunch::DirectNhd(launch) => {
                    let operation = self.module.paged_batch_decode_bf16_nhd_async(
                        launch,
                        common.0,
                        common.1,
                        common.2,
                        common.3,
                        common.4,
                        metadata_is_trusted,
                        resolved.first,
                        resolved.second,
                        resolved.third,
                        resolved.fourth,
                        resolved.fifth,
                        resolved.sixth,
                        resolved.seventh,
                        resolved.eighth,
                        resolved.ninth,
                    );
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
                Bf16PagedBatchDecodeLaunch::DirectHnd(launch) => {
                    let operation = self.module.paged_batch_decode_bf16_hnd_async(
                        launch,
                        common.0,
                        common.1,
                        common.2,
                        common.3,
                        common.4,
                        metadata_is_trusted,
                        resolved.first,
                        resolved.second,
                        resolved.third,
                        resolved.fourth,
                        resolved.fifth,
                        resolved.sixth,
                        resolved.seventh,
                        resolved.eighth,
                        resolved.ninth,
                    );
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
                Bf16PagedBatchDecodeLaunch::TokenParallel8Nhd(launch) => {
                    let operation = self
                        .module
                        .paged_batch_decode_bf16_nhd_token_parallel_async(
                            launch,
                            common.0,
                            common.1,
                            common.2,
                            common.3,
                            common.4,
                            metadata_is_trusted,
                            resolved.first,
                            resolved.second,
                            resolved.third,
                            resolved.fourth,
                            resolved.fifth,
                            resolved.sixth,
                            resolved.seventh,
                            resolved.eighth,
                            resolved.ninth,
                        );
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
                Bf16PagedBatchDecodeLaunch::TokenParallel8Hnd(launch) => {
                    let operation = self
                        .module
                        .paged_batch_decode_bf16_hnd_token_parallel_async(
                            launch,
                            common.0,
                            common.1,
                            common.2,
                            common.3,
                            common.4,
                            metadata_is_trusted,
                            resolved.first,
                            resolved.second,
                            resolved.third,
                            resolved.fourth,
                            resolved.fifth,
                            resolved.sixth,
                            resolved.seventh,
                            resolved.eighth,
                            resolved.ninth,
                        );
                    let result = enqueue_region_launch(resolved.stream, operation);
                    (launch.function().clone(), result)
                }
            }
        };
        let result = record_paged_launch(scope, permit, function, launch_result);
        if let Some(profile) = profile {
            profile.attention_host_nanoseconds = elapsed_nanoseconds(attention_started);
        }
        result
    }
}

fn elapsed_nanoseconds(started: Option<std::time::Instant>) -> u64 {
    started
        .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or_default()
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
            let operation = self.module.single_decode_bf16_nhd_split_k_partials_async(
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
            let result = enqueue_region_launch(resolved.stream, operation);
            (self.partial_launch.function().clone(), result)
        };
        record_launch(scope, partial_permit, partial_function, partial_result)?;

        let merge_permit = scope.prepare_command()?;
        let (merge_function, merge_result) = {
            let resolved = scope.resolve_rww(args.workspace.read(), args.output, args.lse)?;
            let operation = self.module.single_decode_bf16_nhd_split_k_merge_async(
                &self.merge_launch,
                self.spec.decode().num_query_heads(),
                self.spec.partitions(),
                resolved.first,
                resolved.second,
                resolved.third,
            );
            let result = enqueue_region_launch(resolved.stream, operation);
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

/// Checked handles for one paged batch-decode launch.
#[derive(Clone, Copy, Debug)]
pub struct Bf16PagedBatchDecodeArgs {
    query: Read<bf16>,
    key_pages: Read<bf16>,
    value_pages: Read<bf16>,
    page_indptr: Read<i32>,
    page_indices: Read<i32>,
    last_page_len: Read<i32>,
    metadata_status: ReadWrite<i32>,
    output: Write<bf16>,
    lse: Write<f32>,
}

impl Bf16PagedBatchDecodeArgs {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        query: Read<bf16>,
        key_pages: Read<bf16>,
        value_pages: Read<bf16>,
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
            page_indptr,
            page_indices,
            last_page_len,
            metadata_status,
            output,
            lse,
        }
    }
}

/// Checked handles whose paged metadata validity is guaranteed by the adapter.
#[derive(Clone, Copy, Debug)]
pub struct TrustedBf16PagedBatchDecodeArgs {
    spec: Bf16PagedBatchDecodeSpec,
    args: Bf16PagedBatchDecodeArgs,
}

impl TrustedBf16PagedBatchDecodeArgs {
    /// Marks paged metadata as valid for the complete lifetime of this command.
    ///
    /// # Safety
    ///
    /// The bound CSR page table must satisfy the paged-decode contract for
    /// `spec`: indptr starts at zero and is monotonic, every request has at
    /// least one page, each last-page length is within the page size, the
    /// terminal indptr equals the exposed page-index length, and every page
    /// index is below `spec.max_num_pages()`. The adapter must prevent mutation
    /// of these buffers until command completion.
    pub unsafe fn assume_metadata_valid(
        spec: Bf16PagedBatchDecodeSpec,
        args: Bf16PagedBatchDecodeArgs,
    ) -> Self {
        Self { spec, args }
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
pub enum PagedBatchDecodePlanError {
    #[error("paged batch-decode state count {0} exceeds the CUDA grid range")]
    StateCountOutOfRange(usize),
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
    Launch(#[from] DeviceRegionLaunchError),
    #[error(
        "packed single decode requires {operand} to be {alignment}-byte aligned, got {address:#x}"
    )]
    MisalignedBuffer {
        operand: &'static str,
        address: u64,
        alignment: u64,
    },
}

#[derive(Debug, Error)]
pub enum PagedBatchDecodeEnqueueError {
    #[error("{operand} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("page_indices requires at least {minimum} entries, got {actual}")]
    PageIndicesTooShort { minimum: usize, actual: usize },
    #[error("trusted paged metadata belongs to a different decode plan")]
    TrustedMetadataPlanMismatch,
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Launch(#[from] DeviceRegionLaunchError),
    #[error(
        "packed paged batch decode requires {operand} to be {alignment}-byte aligned, got {address:#x}"
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

fn require_paged_packed_alignment(
    operand: &'static str,
    address: u64,
) -> Result<(), PagedBatchDecodeEnqueueError> {
    const ALIGNMENT: u64 = size_of::<u32>() as u64;
    if address.is_multiple_of(ALIGNMENT) {
        Ok(())
    } else {
        Err(PagedBatchDecodeEnqueueError::MisalignedBuffer {
            operand,
            address,
            alignment: ALIGNMENT,
        })
    }
}

fn require_paged_exact_len(
    operand: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), PagedBatchDecodeEnqueueError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PagedBatchDecodeEnqueueError::LengthMismatch {
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
    result: Result<(), DeviceRegionLaunchError>,
) -> Result<(), SingleDecodeEnqueueError> {
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

fn record_paged_launch(
    scope: &mut CommandScope<'_>,
    permit: CommandPermit,
    function: CudaFunction,
    result: Result<(), DeviceRegionLaunchError>,
) -> Result<(), PagedBatchDecodeEnqueueError> {
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
) -> Result<(), PagedBatchDecodeEnqueueError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_algorithm_keeps_mha_direct_and_parallelizes_grouped_heads() {
        let mha = Bf16PagedBatchDecodeSpec::new(1, 1, 8, 8, 128, 16, PagedKvLayout::Nhd).unwrap();
        let mqa = Bf16PagedBatchDecodeSpec::new(1, 1, 8, 1, 128, 16, PagedKvLayout::Hnd).unwrap();
        let gqa = Bf16PagedBatchDecodeSpec::new(1, 1, 16, 4, 128, 16, PagedKvLayout::Nhd).unwrap();

        assert_eq!(
            paged_batch_decode_algorithm(mha),
            Bf16PagedBatchDecodeAlgorithm::Direct
        );
        assert_eq!(
            paged_batch_decode_algorithm(mqa),
            Bf16PagedBatchDecodeAlgorithm::TokenParallel8
        );
        assert_eq!(
            paged_batch_decode_algorithm(gqa),
            Bf16PagedBatchDecodeAlgorithm::TokenParallel8
        );
    }

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
