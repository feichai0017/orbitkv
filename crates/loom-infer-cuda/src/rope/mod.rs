//! cuda-oxide provider for standard BF16 NeoX rotary position embedding.

use crate::command::{
    CommandError, CommandPermit, CommandScope, DeviceStatusReservation, Read, ReadWrite, Write,
};
use crate::device_status::{
    AppendMapKind, DeviceStatusDecoder, STATUS_APPEND_BATCH_INDEX_OUT_OF_RANGE,
    STATUS_APPEND_POSITION_OUT_OF_RANGE, STATUS_DUPLICATE_APPEND_SLOT,
    STATUS_ELEMENT_COUNT_OVERFLOW, STATUS_EMPTY_PAGED_REQUEST, STATUS_INVALID_LAST_PAGE_LENGTH,
    STATUS_INVALID_PAGE_INDPTR_START, STATUS_NON_EXCLUSIVE_APPEND_TARGET,
    STATUS_NON_MONOTONIC_PAGE_INDPTR, STATUS_PACKET_WORDS, STATUS_PAGE_INDEX_OUT_OF_RANGE,
    STATUS_PAGE_INDICES_LENGTH_MISMATCH, STATUS_PAGE_REFERENCE_COUNT_TOO_SMALL, STATUS_SUCCESS,
};
use crate::memory::{DeviceRegionLaunchError, enqueue_region_launch};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, convert, cuda_module, kernel, launch_bounds, launch_contract, tcgen05, thread,
};
use half::bf16;
use loom_infer::{
    Bf16RopePagedKvAppendSpec, Bf16RopePagedKvAppendTokensSpec, Bf16RopePosIdsSpec,
    PAGED_BATCH_DECODE_PAGE_SIZE,
};
use std::sync::Arc;
use thiserror::Error;

const HEAD_DIM: usize = 128;
const ROTARY_DIM: usize = 128;
const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
const BLOCK_THREADS: u32 = ROTARY_PAIRS as u32;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(64)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            tokens >= 1,
            query_heads >= 1,
            key_heads >= 1,
            query.len() == tokens * query_heads * 128,
            key.len() == tokens * key_heads * 128,
            position_ids.len() == tokens,
            query_output.len() == tokens * query_heads * 128,
            key_output.len() == tokens * key_heads * 128,
        ),
    )]
    pub fn rope_pos_ids_bf16_neox_d128(
        tokens: usize,
        query_heads: usize,
        key_heads: usize,
        rope_scale: f32,
        rope_theta: f32,
        query: &[bf16],
        key: &[bf16],
        position_ids: &[i32],
        mut query_output: DisjointSlice<bf16>,
        mut key_output: DisjointSlice<bf16>,
    ) {
        let state = thread::blockIdx_x() as usize;
        let pair = thread::threadIdx_x() as usize;
        let heads = query_heads + key_heads;
        if state >= tokens * heads || pair >= ROTARY_PAIRS {
            return;
        }

        let token = state / heads;
        let combined_head = state % heads;
        let position = position_ids[token];
        if position < 0 {
            return;
        }

        let exponent = pair as f32 / ROTARY_PAIRS as f32;
        let inverse_frequency =
            core::intrinsics::powf32(1.0_f32 / rope_theta, exponent) / rope_scale;
        let angle = position as f32 * inverse_frequency;
        let sin = core::intrinsics::sinf32(angle);
        let cos = core::intrinsics::cosf32(angle);

        if combined_head < query_heads {
            let base = (token * query_heads + combined_head) * HEAD_DIM;
            let input = query.as_ptr().cast::<u16>();
            let output = query_output.as_mut_ptr().cast::<u16>();
            // SAFETY: the launch contract proves the full Q spans. Every
            // thread owns one pair in one token/head state.
            unsafe {
                rotate_pair_to(input, output, base, base, pair, sin, cos);
            }
        } else {
            let key_head = combined_head - query_heads;
            let base = (token * key_heads + key_head) * HEAD_DIM;
            let input = key.as_ptr().cast::<u16>();
            let output = key_output.as_mut_ptr().cast::<u16>();
            // SAFETY: as above, for the disjoint K state.
            unsafe {
                rotate_pair_to(input, output, base, base, pair, sin, cos);
            }
        }
    }

    #[kernel]
    #[launch_bounds(1)]
    #[allow(clippy::too_many_arguments)]
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
            page_refcounts.len() == max_num_pages,
            workspace.len() == 5 + batch_size * 3 + max_num_pages,
        ),
    )]
    pub fn build_paged_append_map(
        batch_size: usize,
        max_num_pages: usize,
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        page_refcounts: &[i32],
        mut workspace: DisjointSlice<i32>,
    ) {
        if thread::blockIdx_x() != 0 || thread::threadIdx_x() != 0 {
            return;
        }
        let output = workspace.as_mut_ptr();
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
        // SAFETY: the launch contract proves the complete workspace span.
        if unsafe { validate_page_indices(output, max_num_pages, page_indices) } {
            return;
        }

        request = 0;
        while request < batch_size {
            let start = page_indptr[request] as usize;
            let end = page_indptr[request + 1] as usize;
            let tail = last_page_len[request] as usize;
            let page_slot = end - start - 1;
            let position = page_slot * PAGED_BATCH_DECODE_PAGE_SIZE + tail - 1;
            if position > i32::MAX as usize {
                // SAFETY: the launch contract proves the status packet span.
                unsafe { write_status(output, STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 0) };
                return;
            }
            let physical_page = page_indices[end - 1] as usize;
            // SAFETY: the workspace has one descriptor for every request.
            unsafe {
                write_descriptor(output, request, position, physical_page, tail - 1);
            }
            request += 1;
        }
        // SAFETY: every request descriptor was initialized above.
        if unsafe { validate_duplicate_slots(output, batch_size) } {
            return;
        }
        // SAFETY: the workspace includes the descriptor and page-count spans.
        let _ = unsafe {
            validate_page_ownership(
                output,
                batch_size,
                max_num_pages,
                page_indices,
                page_refcounts,
            )
        };
    }

    #[kernel]
    #[launch_bounds(1)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (1, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            tokens >= 1,
            tokens <= 64,
            batch_size >= 1,
            max_num_pages >= 1,
            batch_indices.len() == tokens,
            positions.len() == tokens,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            page_refcounts.len() == max_num_pages,
            workspace.len() == 5 + tokens * 3 + max_num_pages,
        ),
    )]
    pub fn build_paged_append_tokens_map(
        tokens: usize,
        batch_size: usize,
        max_num_pages: usize,
        batch_indices: &[i32],
        positions: &[i32],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        page_refcounts: &[i32],
        mut workspace: DisjointSlice<i32>,
    ) {
        if thread::blockIdx_x() != 0 || thread::threadIdx_x() != 0 {
            return;
        }
        let output = workspace.as_mut_ptr();
        // SAFETY: the launch contract proves the status packet span.
        unsafe { write_status(output, STATUS_SUCCESS, 0, 0, 0, 0) };
        if page_indptr[0] != 0 {
            // SAFETY: the launch contract proves the status packet span.
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
        // SAFETY: the launch contract proves the complete workspace span.
        if unsafe { validate_page_indices(output, max_num_pages, page_indices) } {
            return;
        }

        let mut token = 0_usize;
        while token < tokens {
            let request_value = batch_indices[token];
            if request_value < 0 || request_value as usize >= batch_size {
                // SAFETY: the launch contract proves the status packet span.
                unsafe {
                    write_status(
                        output,
                        STATUS_APPEND_BATCH_INDEX_OUT_OF_RANGE,
                        token as i32,
                        request_value,
                        0,
                        0,
                    )
                };
                return;
            }
            let request = request_value as usize;
            let start = page_indptr[request] as usize;
            let end = page_indptr[request + 1] as usize;
            let kv_len =
                (end - start - 1) * PAGED_BATCH_DECODE_PAGE_SIZE + last_page_len[request] as usize;
            if kv_len > i32::MAX as usize {
                // SAFETY: the launch contract proves the status packet span.
                unsafe { write_status(output, STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 0) };
                return;
            }
            let position_value = positions[token];
            if position_value < 0 || position_value as usize >= kv_len {
                // SAFETY: the launch contract proves the status packet span.
                unsafe {
                    write_status(
                        output,
                        STATUS_APPEND_POSITION_OUT_OF_RANGE,
                        token as i32,
                        request as i32,
                        position_value,
                        kv_len as i32,
                    )
                };
                return;
            }
            let position = position_value as usize;
            let page_slot = position / PAGED_BATCH_DECODE_PAGE_SIZE;
            let page_offset = position % PAGED_BATCH_DECODE_PAGE_SIZE;
            let physical_page = page_indices[start + page_slot] as usize;
            // SAFETY: the workspace has one descriptor for every token.
            unsafe {
                write_descriptor(output, token, position, physical_page, page_offset);
            }
            token += 1;
        }
        // SAFETY: every token descriptor was initialized above.
        if unsafe { validate_duplicate_slots(output, tokens) } {
            return;
        }
        // SAFETY: the workspace includes the descriptor and page-count spans.
        let _ = unsafe {
            validate_page_ownership(output, tokens, max_num_pages, page_indices, page_refcounts)
        };
    }

    #[kernel]
    #[launch_bounds(64)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            items >= 1,
            max_num_pages >= 1,
            query_heads >= 1,
            key_heads >= 1,
            query.len() == items * query_heads * 128,
            key.len() == items * key_heads * 128,
            value.len() == items * key_heads * 128,
            append_map.len() >= 5 + items * 3,
            query_output.len() == items * query_heads * 128,
            key_pages.len() == max_num_pages * 16 * key_heads * 128,
            value_pages.len() == max_num_pages * 16 * key_heads * 128,
        ),
    )]
    pub fn rope_paged_kv_append_mapped_bf16_neox_d128(
        items: usize,
        max_num_pages: usize,
        query_heads: usize,
        key_heads: usize,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        append_map: &[i32],
        mut query_output: DisjointSlice<bf16>,
        mut key_pages: DisjointSlice<bf16>,
        mut value_pages: DisjointSlice<bf16>,
    ) {
        let state = thread::blockIdx_x() as usize;
        let pair = thread::threadIdx_x() as usize;
        let heads = query_heads + key_heads;
        if state >= items * heads || pair >= ROTARY_PAIRS || append_map[0] != STATUS_SUCCESS {
            return;
        }
        let item = state / heads;
        let combined_head = state % heads;
        let descriptor = STATUS_PACKET_WORDS + item * 3;
        let position = append_map[descriptor] as usize;
        let physical_page = append_map[descriptor + 1] as usize;
        let page_offset = append_map[descriptor + 2] as usize;
        if physical_page >= max_num_pages || page_offset >= PAGED_BATCH_DECODE_PAGE_SIZE {
            return;
        }
        let exponent = pair as f32 / ROTARY_PAIRS as f32;
        let inverse_frequency = core::intrinsics::powf32(1.0_f32 / 10_000.0, exponent);
        let angle = position as f32 * inverse_frequency;
        let sin = core::intrinsics::sinf32(angle);
        let cos = core::intrinsics::cosf32(angle);
        if combined_head < query_heads {
            let base = (item * query_heads + combined_head) * HEAD_DIM;
            // SAFETY: the launch contract proves the Q spans, and each thread
            // owns one pair in one item/head state.
            unsafe {
                rotate_pair_to(
                    query.as_ptr().cast(),
                    query_output.as_mut_ptr().cast(),
                    base,
                    base,
                    pair,
                    sin,
                    cos,
                );
            }
        } else {
            let key_head = combined_head - query_heads;
            let source = (item * key_heads + key_head) * HEAD_DIM;
            let destination = ((physical_page * PAGED_BATCH_DECODE_PAGE_SIZE + page_offset)
                * key_heads
                + key_head)
                * HEAD_DIM;
            // SAFETY: the validated map bounds the private destination page,
            // and each thread owns one K/V pair in that state.
            unsafe {
                rotate_pair_to(
                    key.as_ptr().cast(),
                    key_pages.as_mut_ptr().cast(),
                    source,
                    destination,
                    pair,
                    sin,
                    cos,
                );
                copy_pair_to(
                    value.as_ptr().cast(),
                    value_pages.as_mut_ptr().cast(),
                    source,
                    destination,
                    pair,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn validate_page_indices(
        output: *mut i32,
        max_num_pages: usize,
        page_indices: &[i32],
    ) -> bool {
        let mut position = 0_usize;
        while position < page_indices.len() {
            let index = page_indices[position];
            if index < 0 || index as usize >= max_num_pages {
                // SAFETY: the caller guarantees a writable status packet.
                unsafe {
                    write_status(
                        output,
                        STATUS_PAGE_INDEX_OUT_OF_RANGE,
                        position as i32,
                        index,
                        0,
                        0,
                    )
                };
                return true;
            }
            position += 1;
        }
        false
    }

    #[inline(always)]
    unsafe fn validate_duplicate_slots(output: *mut i32, items: usize) -> bool {
        let mut first = 0_usize;
        while first < items {
            let first_descriptor = STATUS_PACKET_WORDS + first * 3;
            // SAFETY: the caller guarantees `items` initialized descriptors.
            let first_page = unsafe { output.add(first_descriptor + 1).read() };
            // SAFETY: as above, for this descriptor's offset.
            let first_offset = unsafe { output.add(first_descriptor + 2).read() };
            let mut second = first + 1;
            while second < items {
                let second_descriptor = STATUS_PACKET_WORDS + second * 3;
                // SAFETY: the caller guarantees `items` initialized descriptors.
                let second_page = unsafe { output.add(second_descriptor + 1).read() };
                // SAFETY: as above, for this descriptor's offset.
                let second_offset = unsafe { output.add(second_descriptor + 2).read() };
                if first_page == second_page && first_offset == second_offset {
                    // SAFETY: the caller guarantees a writable status packet.
                    unsafe {
                        write_status(
                            output,
                            STATUS_DUPLICATE_APPEND_SLOT,
                            first as i32,
                            second as i32,
                            first_page,
                            first_offset,
                        )
                    };
                    return true;
                }
                second += 1;
            }
            first += 1;
        }
        false
    }

    #[inline(always)]
    unsafe fn validate_page_ownership(
        output: *mut i32,
        items: usize,
        max_num_pages: usize,
        page_indices: &[i32],
        page_refcounts: &[i32],
    ) -> bool {
        let counts_offset = STATUS_PACKET_WORDS + items * 3;
        let mut page = 0_usize;
        while page < max_num_pages {
            // SAFETY: the caller guarantees `max_num_pages` count slots.
            unsafe { output.add(counts_offset + page).write(0) };
            page += 1;
        }
        let mut position = 0_usize;
        while position < page_indices.len() {
            let page = page_indices[position] as usize;
            // SAFETY: page indices were validated against `max_num_pages`.
            let counter = unsafe { output.add(counts_offset + page) };
            // SAFETY: `counter` points into the initialized count span.
            let count = unsafe { counter.read() };
            if count == i32::MAX {
                // SAFETY: the caller guarantees a writable status packet.
                unsafe { write_status(output, STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 0) };
                return true;
            }
            // SAFETY: `counter` points into the writable count span.
            unsafe { counter.write(count + 1) };
            position += 1;
        }
        page = 0;
        while page < max_num_pages {
            // SAFETY: the caller guarantees `max_num_pages` count slots.
            let minimum = unsafe { output.add(counts_offset + page).read() };
            let actual = page_refcounts[page];
            if actual < 0 || actual < minimum {
                // SAFETY: the caller guarantees a writable status packet.
                unsafe {
                    write_status(
                        output,
                        STATUS_PAGE_REFERENCE_COUNT_TOO_SMALL,
                        page as i32,
                        minimum,
                        actual,
                        0,
                    )
                };
                return true;
            }
            page += 1;
        }
        let mut item = 0_usize;
        while item < items {
            let descriptor = STATUS_PACKET_WORDS + item * 3;
            // SAFETY: the caller guarantees `items` initialized descriptors.
            let page = unsafe { output.add(descriptor + 1).read() } as usize;
            let reference_count = page_refcounts[page];
            if reference_count != 1 {
                // SAFETY: the caller guarantees a writable status packet.
                unsafe {
                    write_status(
                        output,
                        STATUS_NON_EXCLUSIVE_APPEND_TARGET,
                        page as i32,
                        reference_count,
                        0,
                        0,
                    )
                };
                return true;
            }
            item += 1;
        }
        false
    }

    #[inline(always)]
    unsafe fn write_descriptor(
        output: *mut i32,
        item: usize,
        position: usize,
        physical_page: usize,
        page_offset: usize,
    ) {
        let descriptor = STATUS_PACKET_WORDS + item * 3;
        // SAFETY: the caller guarantees the descriptor span for `item`.
        unsafe {
            output.add(descriptor).write(position as i32);
            output.add(descriptor + 1).write(physical_page as i32);
            output.add(descriptor + 2).write(page_offset as i32);
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

    #[inline(always)]
    unsafe fn rotate_pair_to(
        input: *const u16,
        output: *mut u16,
        source: usize,
        destination: usize,
        pair: usize,
        sin: f32,
        cos: f32,
    ) {
        // SAFETY: the caller proves both split-half indices are in one exact
        // D128 state and uniquely owned by this thread.
        let first_bits = unsafe { input.add(source + pair).read() };
        // SAFETY: as above, for the paired second-half component.
        let second_bits = unsafe { input.add(source + ROTARY_PAIRS + pair).read() };
        let first = convert::cvt_f32_bf16x2_lo(first_bits as u32);
        let second = convert::cvt_f32_bf16x2_lo(second_bits as u32);
        let rotated =
            tcgen05::cvt_f32x2_bf16x2(first * cos - second * sin, second * cos + first * sin);
        // SAFETY: both output components are in the exact D128 state and no
        // other thread writes this pair.
        unsafe {
            output.add(destination + pair).write(rotated as u16);
            output
                .add(destination + ROTARY_PAIRS + pair)
                .write((rotated >> 16) as u16);
        }
    }

    #[inline(always)]
    unsafe fn copy_pair_to(
        input: *const u16,
        output: *mut u16,
        source: usize,
        destination: usize,
        pair: usize,
    ) {
        // SAFETY: caller proves source/destination spans and unique ownership.
        let first = unsafe { input.add(source + pair).read() };
        // SAFETY: as above, for the second-half component.
        let second = unsafe { input.add(source + ROTARY_PAIRS + pair).read() };
        // SAFETY: this thread uniquely owns both output components.
        unsafe {
            output.add(destination + pair).write(first);
            output.add(destination + ROTARY_PAIRS + pair).write(second);
        }
    }
}

/// Loaded standard RoPE CUDA module.
#[derive(Clone, Debug)]
pub struct RopeProvider {
    module: kernels::LoadedModule,
}

impl RopeProvider {
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, cuda_host::EmbeddedModuleError> {
        // SAFETY: this crate owns the package-named cuda-oxide artifact bundle.
        let module = unsafe { kernels::load(context)? };
        Ok(Self { module })
    }

    pub fn plan_bf16_pos_ids(
        &self,
        spec: Bf16RopePosIdsSpec,
    ) -> Result<Bf16RopePosIdsPlan, RopePlanError> {
        if spec.head_dim() != HEAD_DIM {
            return Err(RopePlanError::UnsupportedHeadDimension {
                expected: HEAD_DIM,
                actual: spec.head_dim(),
            });
        }
        if spec.rotary_dim() != ROTARY_DIM {
            return Err(RopePlanError::UnsupportedRotaryDimension {
                expected: ROTARY_DIM,
                actual: spec.rotary_dim(),
            });
        }
        let states = spec
            .tokens()
            .checked_mul(
                spec.query_heads()
                    .checked_add(spec.key_heads())
                    .ok_or(RopePlanError::StateCountOverflow)?,
            )
            .ok_or(RopePlanError::StateCountOverflow)?;
        let blocks =
            u32::try_from(states).map_err(|_| RopePlanError::StateCountOutOfRange(states))?;
        let launch = self
            .module
            .prepare_rope_pos_ids_bf16_neox_d128(LaunchConfig1D::new(blocks, BLOCK_THREADS, 0))?;
        Ok(Bf16RopePosIdsPlan {
            spec,
            module: self.module.clone(),
            launch,
        })
    }

    pub fn plan_bf16_paged_append(
        &self,
        spec: Bf16RopePagedKvAppendSpec,
    ) -> Result<Bf16RopePagedKvAppendPlan, RopePlanError> {
        let workspace_numel = append_map_workspace_numel(spec.batch_size(), spec.max_num_pages())?;
        let states = spec
            .batch_size()
            .checked_mul(
                spec.num_query_heads()
                    .checked_add(spec.num_kv_heads())
                    .ok_or(RopePlanError::StateCountOverflow)?,
            )
            .ok_or(RopePlanError::StateCountOverflow)?;
        let blocks =
            u32::try_from(states).map_err(|_| RopePlanError::StateCountOutOfRange(states))?;
        let map_launch = self
            .module
            .prepare_build_paged_append_map(LaunchConfig1D::new(1, 1, 0))?;
        let append_launch = self
            .module
            .prepare_rope_paged_kv_append_mapped_bf16_neox_d128(LaunchConfig1D::new(
                blocks,
                BLOCK_THREADS,
                0,
            ))?;
        Ok(Bf16RopePagedKvAppendPlan {
            spec,
            workspace_numel,
            module: self.module.clone(),
            map_launch,
            append_launch,
        })
    }

    pub fn plan_bf16_paged_append_tokens(
        &self,
        spec: Bf16RopePagedKvAppendTokensSpec,
    ) -> Result<Bf16RopePagedKvAppendTokensPlan, RopePlanError> {
        let workspace_numel = append_map_workspace_numel(spec.tokens(), spec.max_num_pages())?;
        let states = spec
            .tokens()
            .checked_mul(
                spec.num_query_heads()
                    .checked_add(spec.num_kv_heads())
                    .ok_or(RopePlanError::StateCountOverflow)?,
            )
            .ok_or(RopePlanError::StateCountOverflow)?;
        let blocks =
            u32::try_from(states).map_err(|_| RopePlanError::StateCountOutOfRange(states))?;
        let map_launch = self
            .module
            .prepare_build_paged_append_tokens_map(LaunchConfig1D::new(1, 1, 0))?;
        let append_launch = self
            .module
            .prepare_rope_paged_kv_append_mapped_bf16_neox_d128(LaunchConfig1D::new(
                blocks,
                BLOCK_THREADS,
                0,
            ))?;
        Ok(Bf16RopePagedKvAppendTokensPlan {
            spec,
            workspace_numel,
            module: self.module.clone(),
            map_launch,
            append_launch,
        })
    }
}

/// Immutable prepared launch for standard BF16 RoPE with explicit positions.
#[derive(Clone)]
pub struct Bf16RopePosIdsPlan {
    spec: Bf16RopePosIdsSpec,
    module: kernels::LoadedModule,
    launch: PreparedLaunch<kernels::__rope_pos_ids_bf16_neox_d128_CudaKernel>,
}

impl Bf16RopePosIdsPlan {
    pub const fn spec(&self) -> Bf16RopePosIdsSpec {
        self.spec
    }

    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16RopePosIdsArgs,
    ) -> Result<(), RopeEnqueueError> {
        let permit = scope.prepare_command()?;
        let (function, result) = {
            let resolved = scope.resolve_rrrww(
                args.query,
                args.key,
                args.position_ids,
                args.query_output,
                args.key_output,
            )?;
            require_exact_len("query", resolved.first.len(), self.spec.query_numel())?;
            require_exact_len("key", resolved.second.len(), self.spec.key_numel())?;
            require_exact_len(
                "position_ids",
                resolved.third.len(),
                self.spec.position_numel(),
            )?;
            require_exact_len(
                "query_output",
                resolved.fourth.len(),
                self.spec.query_numel(),
            )?;
            require_exact_len("key_output", resolved.fifth.len(), self.spec.key_numel())?;
            let operation = self.module.rope_pos_ids_bf16_neox_d128_async(
                &self.launch,
                self.spec.tokens(),
                self.spec.query_heads(),
                self.spec.key_heads(),
                self.spec.rope_scale(),
                self.spec.rope_theta(),
                resolved.first,
                resolved.second,
                resolved.third,
                resolved.fourth,
                resolved.fifth,
            );
            let result = enqueue_region_launch(resolved.stream, operation);
            (self.launch.function().clone(), result)
        };
        record_launch(scope, permit, function, result)
    }
}

/// Immutable plan for a reusable one-token-per-request append map and RoPE append.
#[derive(Clone)]
pub struct Bf16RopePagedKvAppendPlan {
    spec: Bf16RopePagedKvAppendSpec,
    workspace_numel: usize,
    module: kernels::LoadedModule,
    map_launch: PreparedLaunch<kernels::__build_paged_append_map_CudaKernel>,
    append_launch: PreparedLaunch<kernels::__rope_paged_kv_append_mapped_bf16_neox_d128_CudaKernel>,
}

impl Bf16RopePagedKvAppendPlan {
    pub const fn spec(&self) -> Bf16RopePagedKvAppendSpec {
        self.spec
    }

    pub const fn workspace_required_numel(&self) -> usize {
        self.workspace_numel
    }

    pub fn enqueue_map_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16PagedKvAppendMapArgs,
    ) -> Result<Bf16PagedKvAppendMap, RopeEnqueueError> {
        let page_indices_len = {
            let resolved = scope.resolve_rrrrw(
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.page_refcounts,
                args.workspace.write(),
            )?;
            require_exact_len(
                "page_indptr",
                resolved.first.len(),
                self.spec.page_indptr_numel(),
            )?;
            if resolved.second.len() < self.spec.batch_size() {
                return Err(RopeEnqueueError::PageIndicesTooShort {
                    minimum: self.spec.batch_size(),
                    actual: resolved.second.len(),
                });
            }
            require_exact_len(
                "last_page_len",
                resolved.third.len(),
                self.spec.last_page_len_numel(),
            )?;
            require_exact_len(
                "page_refcounts",
                resolved.fourth.len(),
                self.spec.page_refcounts_numel(),
            )?;
            require_exact_len("workspace", resolved.fifth.len(), self.workspace_numel)?;
            resolved.second.len()
        };
        scope.require_command_capacity(2)?;
        let status = scope.reserve_device_status(
            args.workspace.read(),
            DeviceStatusDecoder::paged_append(
                AppendMapKind::Requests,
                self.spec.batch_size(),
                self.spec.batch_size(),
                self.spec.max_num_pages(),
                page_indices_len,
                PAGED_BATCH_DECODE_PAGE_SIZE,
            ),
        )?;
        let permit = scope.prepare_command()?;
        let (function, result) = {
            let resolved = scope.resolve_rrrrw(
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.page_refcounts,
                args.workspace.write(),
            )?;
            let operation = self.module.build_paged_append_map_async(
                &self.map_launch,
                self.spec.batch_size(),
                self.spec.max_num_pages(),
                resolved.first,
                resolved.second,
                resolved.third,
                resolved.fourth,
                resolved.fifth,
            );
            let result = enqueue_region_launch(resolved.stream, operation);
            (self.map_launch.function().clone(), result)
        };
        record_map_launch(scope, status, permit, function, result)?;
        Ok(Bf16PagedKvAppendMap {
            scope_id: scope.scope_id(),
            kind: AppendMapKind::Requests,
            items: self.spec.batch_size(),
            max_num_pages: self.spec.max_num_pages(),
            workspace: args.workspace.read(),
            key_pages: args.key_pages,
            value_pages: args.value_pages,
        })
    }

    pub fn enqueue_mapped_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16RopePagedKvAppendMappedArgs,
    ) -> Result<(), RopeEnqueueError> {
        require_append_map(
            scope,
            args.append_map,
            AppendMapKind::Requests,
            self.spec.batch_size(),
            self.spec.max_num_pages(),
            args.key_pages,
            args.value_pages,
        )?;
        enqueue_mapped_append(
            scope,
            &self.module,
            &self.append_launch,
            self.spec.batch_size(),
            self.spec.max_num_pages(),
            self.spec.num_query_heads(),
            self.spec.num_kv_heads(),
            self.spec.query_numel(),
            self.spec.key_numel(),
            self.spec.value_numel(),
            self.spec.query_output_numel(),
            self.spec.kv_pages_numel(),
            self.workspace_numel,
            args,
        )
    }
}

/// Immutable plan for an explicit-token append map and reusable RoPE append.
#[derive(Clone)]
pub struct Bf16RopePagedKvAppendTokensPlan {
    spec: Bf16RopePagedKvAppendTokensSpec,
    workspace_numel: usize,
    module: kernels::LoadedModule,
    map_launch: PreparedLaunch<kernels::__build_paged_append_tokens_map_CudaKernel>,
    append_launch: PreparedLaunch<kernels::__rope_paged_kv_append_mapped_bf16_neox_d128_CudaKernel>,
}

impl Bf16RopePagedKvAppendTokensPlan {
    pub const fn spec(&self) -> Bf16RopePagedKvAppendTokensSpec {
        self.spec
    }

    pub const fn workspace_required_numel(&self) -> usize {
        self.workspace_numel
    }

    pub fn enqueue_map_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16PagedKvAppendTokensMapArgs,
    ) -> Result<Bf16PagedKvAppendMap, RopeEnqueueError> {
        let page_indices_len = {
            let resolved = scope.resolve_rrrrrrw(
                args.batch_indices,
                args.positions,
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.page_refcounts,
                args.workspace.write(),
            )?;
            require_exact_len(
                "batch_indices",
                resolved.first.len(),
                self.spec.batch_indices_numel(),
            )?;
            require_exact_len(
                "positions",
                resolved.second.len(),
                self.spec.positions_numel(),
            )?;
            require_exact_len(
                "page_indptr",
                resolved.third.len(),
                self.spec.page_indptr_numel(),
            )?;
            if resolved.fourth.len() < self.spec.batch_size() {
                return Err(RopeEnqueueError::PageIndicesTooShort {
                    minimum: self.spec.batch_size(),
                    actual: resolved.fourth.len(),
                });
            }
            require_exact_len(
                "last_page_len",
                resolved.fifth.len(),
                self.spec.last_page_len_numel(),
            )?;
            require_exact_len(
                "page_refcounts",
                resolved.sixth.len(),
                self.spec.page_refcounts_numel(),
            )?;
            require_exact_len("workspace", resolved.seventh.len(), self.workspace_numel)?;
            resolved.fourth.len()
        };
        scope.require_command_capacity(2)?;
        let status = scope.reserve_device_status(
            args.workspace.read(),
            DeviceStatusDecoder::paged_append(
                AppendMapKind::ExplicitTokens,
                self.spec.tokens(),
                self.spec.batch_size(),
                self.spec.max_num_pages(),
                page_indices_len,
                PAGED_BATCH_DECODE_PAGE_SIZE,
            ),
        )?;
        let permit = scope.prepare_command()?;
        let (function, result) = {
            let resolved = scope.resolve_rrrrrrw(
                args.batch_indices,
                args.positions,
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.page_refcounts,
                args.workspace.write(),
            )?;
            let operation = self.module.build_paged_append_tokens_map_async(
                &self.map_launch,
                self.spec.tokens(),
                self.spec.batch_size(),
                self.spec.max_num_pages(),
                resolved.first,
                resolved.second,
                resolved.third,
                resolved.fourth,
                resolved.fifth,
                resolved.sixth,
                resolved.seventh,
            );
            let result = enqueue_region_launch(resolved.stream, operation);
            (self.map_launch.function().clone(), result)
        };
        record_map_launch(scope, status, permit, function, result)?;
        Ok(Bf16PagedKvAppendMap {
            scope_id: scope.scope_id(),
            kind: AppendMapKind::ExplicitTokens,
            items: self.spec.tokens(),
            max_num_pages: self.spec.max_num_pages(),
            workspace: args.workspace.read(),
            key_pages: args.key_pages,
            value_pages: args.value_pages,
        })
    }

    pub fn enqueue_mapped_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16RopePagedKvAppendMappedArgs,
    ) -> Result<(), RopeEnqueueError> {
        require_append_map(
            scope,
            args.append_map,
            AppendMapKind::ExplicitTokens,
            self.spec.tokens(),
            self.spec.max_num_pages(),
            args.key_pages,
            args.value_pages,
        )?;
        enqueue_mapped_append(
            scope,
            &self.module,
            &self.append_launch,
            self.spec.tokens(),
            self.spec.max_num_pages(),
            self.spec.num_query_heads(),
            self.spec.num_kv_heads(),
            self.spec.query_numel(),
            self.spec.key_numel(),
            self.spec.value_numel(),
            self.spec.query_output_numel(),
            self.spec.kv_pages_numel(),
            self.workspace_numel,
            args,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bf16RopePosIdsArgs {
    query: Read<bf16>,
    key: Read<bf16>,
    position_ids: Read<i32>,
    query_output: Write<bf16>,
    key_output: Write<bf16>,
}

impl Bf16RopePosIdsArgs {
    pub const fn new(
        query: Read<bf16>,
        key: Read<bf16>,
        position_ids: Read<i32>,
        query_output: Write<bf16>,
        key_output: Write<bf16>,
    ) -> Self {
        Self {
            query,
            key,
            position_ids,
            query_output,
            key_output,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bf16PagedKvAppendMapArgs {
    page_indptr: Read<i32>,
    page_indices: Read<i32>,
    last_page_len: Read<i32>,
    page_refcounts: Read<i32>,
    key_pages: Write<bf16>,
    value_pages: Write<bf16>,
    workspace: ReadWrite<i32>,
}

impl Bf16PagedKvAppendMapArgs {
    pub const fn new(
        page_indptr: Read<i32>,
        page_indices: Read<i32>,
        last_page_len: Read<i32>,
        page_refcounts: Read<i32>,
        key_pages: Write<bf16>,
        value_pages: Write<bf16>,
        workspace: ReadWrite<i32>,
    ) -> Self {
        Self {
            page_indptr,
            page_indices,
            last_page_len,
            page_refcounts,
            key_pages,
            value_pages,
            workspace,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bf16PagedKvAppendTokensMapArgs {
    batch_indices: Read<i32>,
    positions: Read<i32>,
    page_indptr: Read<i32>,
    page_indices: Read<i32>,
    last_page_len: Read<i32>,
    page_refcounts: Read<i32>,
    key_pages: Write<bf16>,
    value_pages: Write<bf16>,
    workspace: ReadWrite<i32>,
}

impl Bf16PagedKvAppendTokensMapArgs {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        batch_indices: Read<i32>,
        positions: Read<i32>,
        page_indptr: Read<i32>,
        page_indices: Read<i32>,
        last_page_len: Read<i32>,
        page_refcounts: Read<i32>,
        key_pages: Write<bf16>,
        value_pages: Write<bf16>,
        workspace: ReadWrite<i32>,
    ) -> Self {
        Self {
            batch_indices,
            positions,
            page_indptr,
            page_indices,
            last_page_len,
            page_refcounts,
            key_pages,
            value_pages,
            workspace,
        }
    }
}

/// Scope-bound compact append mapping produced by one device validator.
#[derive(Clone, Copy, Debug)]
pub struct Bf16PagedKvAppendMap {
    scope_id: u64,
    kind: AppendMapKind,
    items: usize,
    max_num_pages: usize,
    workspace: Read<i32>,
    key_pages: Write<bf16>,
    value_pages: Write<bf16>,
}

#[derive(Clone, Copy, Debug)]
pub struct Bf16RopePagedKvAppendMappedArgs {
    query: Read<bf16>,
    key: Read<bf16>,
    value: Read<bf16>,
    append_map: Bf16PagedKvAppendMap,
    query_output: Write<bf16>,
    key_pages: Write<bf16>,
    value_pages: Write<bf16>,
}

impl Bf16RopePagedKvAppendMappedArgs {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        query: Read<bf16>,
        key: Read<bf16>,
        value: Read<bf16>,
        append_map: Bf16PagedKvAppendMap,
        query_output: Write<bf16>,
        key_pages: Write<bf16>,
        value_pages: Write<bf16>,
    ) -> Self {
        Self {
            query,
            key,
            value,
            append_map,
            query_output,
            key_pages,
            value_pages,
        }
    }
}

#[derive(Debug, Error)]
pub enum RopePlanError {
    #[error("standard RoPE requires head dimension {expected}, got {actual}")]
    UnsupportedHeadDimension { expected: usize, actual: usize },
    #[error("standard RoPE requires rotary dimension {expected}, got {actual}")]
    UnsupportedRotaryDimension { expected: usize, actual: usize },
    #[error("RoPE state count overflowed")]
    StateCountOverflow,
    #[error("RoPE state count {0} exceeds the CUDA grid range")]
    StateCountOutOfRange(usize),
    #[error("paged append map workspace size overflowed")]
    WorkspaceSizeOverflow,
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
}

#[derive(Debug, Error)]
pub enum RopeEnqueueError {
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Launch(#[from] DeviceRegionLaunchError),
    #[error("{operand} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("page_indices requires at least {minimum} entries, got {actual}")]
    PageIndicesTooShort { minimum: usize, actual: usize },
    #[error("the append map was created by a different command scope")]
    AppendMapScopeMismatch,
    #[error("the append map does not match this append plan")]
    AppendMapSpecMismatch,
    #[error("the append map belongs to a different paged KV cache binding")]
    AppendMapCacheMismatch,
}

fn append_map_workspace_numel(items: usize, max_num_pages: usize) -> Result<usize, RopePlanError> {
    items
        .checked_mul(3)
        .and_then(|descriptors| descriptors.checked_add(STATUS_PACKET_WORDS))
        .and_then(|fixed| fixed.checked_add(max_num_pages))
        .ok_or(RopePlanError::WorkspaceSizeOverflow)
}

fn require_append_map(
    scope: &CommandScope<'_>,
    append_map: Bf16PagedKvAppendMap,
    kind: AppendMapKind,
    items: usize,
    max_num_pages: usize,
    key_pages: Write<bf16>,
    value_pages: Write<bf16>,
) -> Result<(), RopeEnqueueError> {
    if append_map.scope_id != scope.scope_id() {
        return Err(RopeEnqueueError::AppendMapScopeMismatch);
    }
    if append_map.kind != kind
        || append_map.items != items
        || append_map.max_num_pages != max_num_pages
    {
        return Err(RopeEnqueueError::AppendMapSpecMismatch);
    }
    if append_map.key_pages != key_pages || append_map.value_pages != value_pages {
        return Err(RopeEnqueueError::AppendMapCacheMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn enqueue_mapped_append(
    scope: &mut CommandScope<'_>,
    module: &kernels::LoadedModule,
    launch: &PreparedLaunch<kernels::__rope_paged_kv_append_mapped_bf16_neox_d128_CudaKernel>,
    items: usize,
    max_num_pages: usize,
    query_heads: usize,
    key_heads: usize,
    query_numel: usize,
    key_numel: usize,
    value_numel: usize,
    query_output_numel: usize,
    kv_pages_numel: usize,
    workspace_numel: usize,
    args: Bf16RopePagedKvAppendMappedArgs,
) -> Result<(), RopeEnqueueError> {
    let permit = scope.prepare_command()?;
    let (function, result) = {
        let resolved = scope.resolve_rrrrwww(
            args.query,
            args.key,
            args.value,
            args.append_map.workspace,
            args.query_output,
            args.key_pages,
            args.value_pages,
        )?;
        require_exact_len("query", resolved.first.len(), query_numel)?;
        require_exact_len("key", resolved.second.len(), key_numel)?;
        require_exact_len("value", resolved.third.len(), value_numel)?;
        require_exact_len("append_map", resolved.fourth.len(), workspace_numel)?;
        require_exact_len("query_output", resolved.fifth.len(), query_output_numel)?;
        require_exact_len("key_pages", resolved.sixth.len(), kv_pages_numel)?;
        require_exact_len("value_pages", resolved.seventh.len(), kv_pages_numel)?;
        let operation = module.rope_paged_kv_append_mapped_bf16_neox_d128_async(
            launch,
            items,
            max_num_pages,
            query_heads,
            key_heads,
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
    };
    record_launch(scope, permit, function, result)
}

fn require_exact_len(
    operand: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), RopeEnqueueError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RopeEnqueueError::LengthMismatch {
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
) -> Result<(), RopeEnqueueError> {
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

fn record_map_launch(
    scope: &mut CommandScope<'_>,
    status: DeviceStatusReservation,
    permit: CommandPermit,
    function: CudaFunction,
    result: Result<(), DeviceRegionLaunchError>,
) -> Result<(), RopeEnqueueError> {
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
