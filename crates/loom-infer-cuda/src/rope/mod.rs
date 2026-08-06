//! cuda-oxide provider for standard BF16 NeoX rotary position embedding.

use crate::command::{CommandError, CommandPermit, CommandScope, Read, Write};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract,
    tcgen05, thread, warp,
};
use half::bf16;
use loom_infer::{
    Bf16RopePagedKvAppendSpec, Bf16RopePagedKvAppendTokensSpec, Bf16RopePosIdsSpec,
    PAGED_BATCH_DECODE_PAGE_SIZE, ROPE_PAGED_KV_APPEND_MAX_TOKENS,
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
        let inverse_frequency = (1.0_f32 / rope_theta).powf(exponent) / rope_scale;
        let angle = position as f32 * inverse_frequency;
        let (sin, cos) = angle.sin_cos();

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
    #[launch_bounds(64)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            batch_size >= 1,
            max_num_pages >= 1,
            query_heads >= 1,
            key_heads >= 1,
            query.len() == batch_size * query_heads * 128,
            key.len() == batch_size * key_heads * 128,
            value.len() == batch_size * key_heads * 128,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            query_output.len() == batch_size * query_heads * 128,
            key_pages.len() == max_num_pages * 16 * key_heads * 128,
            value_pages.len() == max_num_pages * 16 * key_heads * 128,
        ),
    )]
    pub fn rope_paged_kv_append_bf16_neox_d128(
        batch_size: usize,
        max_num_pages: usize,
        query_heads: usize,
        key_heads: usize,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        mut query_output: DisjointSlice<bf16>,
        mut key_pages: DisjointSlice<bf16>,
        mut value_pages: DisjointSlice<bf16>,
    ) {
        let state = thread::blockIdx_x() as usize;
        let pair = thread::threadIdx_x() as usize;
        let heads = query_heads + key_heads;
        if state >= batch_size * heads || pair >= ROTARY_PAIRS {
            return;
        }
        if page_indptr[0] != 0 || page_indptr[batch_size] as usize != page_indices.len() {
            return;
        }
        let mut page = 0_usize;
        while page < page_indices.len() {
            let physical_page = page_indices[page];
            if physical_page < 0 || physical_page as usize >= max_num_pages {
                return;
            }
            page += 1;
        }

        let request = state / heads;
        let combined_head = state % heads;
        let Some((position, physical_page, page_offset)) = append_slot(
            request,
            batch_size,
            max_num_pages,
            page_indptr,
            page_indices,
            last_page_len,
        ) else {
            return;
        };
        let mut other = 0_usize;
        while other < batch_size {
            if other != request {
                let Some((_, other_page, other_offset)) = append_slot(
                    other,
                    batch_size,
                    max_num_pages,
                    page_indptr,
                    page_indices,
                    last_page_len,
                ) else {
                    return;
                };
                if physical_page == other_page && page_offset == other_offset {
                    return;
                }
            }
            other += 1;
        }

        let exponent = pair as f32 / ROTARY_PAIRS as f32;
        let inverse_frequency = (1.0_f32 / 10_000.0).powf(exponent);
        let angle = position as f32 * inverse_frequency;
        let (sin, cos) = angle.sin_cos();
        if combined_head < query_heads {
            let base = (request * query_heads + combined_head) * HEAD_DIM;
            // SAFETY: this CTA/thread uniquely owns one Q split-half pair.
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
            let source = (request * key_heads + key_head) * HEAD_DIM;
            let destination = ((physical_page * PAGED_BATCH_DECODE_PAGE_SIZE + page_offset)
                * key_heads
                + key_head)
                * HEAD_DIM;
            // SAFETY: duplicate-slot validation and one CTA per request/head
            // prove unique K/V cache destinations.
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

    #[kernel]
    #[launch_bounds(64)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            tokens >= 1,
            tokens <= 64,
            batch_size >= 1,
            max_num_pages >= 1,
            query_heads >= 1,
            key_heads >= 1,
            query.len() == tokens * query_heads * 128,
            key.len() == tokens * key_heads * 128,
            value.len() == tokens * key_heads * 128,
            batch_indices.len() == tokens,
            positions.len() == tokens,
            page_indptr.len() == batch_size + 1,
            page_indices.len() >= batch_size,
            last_page_len.len() == batch_size,
            query_output.len() == tokens * query_heads * 128,
            key_pages.len() == max_num_pages * 16 * key_heads * 128,
            value_pages.len() == max_num_pages * 16 * key_heads * 128,
        ),
    )]
    pub fn rope_paged_kv_append_tokens_bf16_neox_d128(
        tokens: usize,
        batch_size: usize,
        max_num_pages: usize,
        query_heads: usize,
        key_heads: usize,
        query: &[bf16],
        key: &[bf16],
        value: &[bf16],
        batch_indices: &[i32],
        positions: &[i32],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
        mut query_output: DisjointSlice<bf16>,
        mut key_pages: DisjointSlice<bf16>,
        mut value_pages: DisjointSlice<bf16>,
    ) {
        static mut VALIDATION_FLAGS: SharedArray<u32, 2> = SharedArray::UNINIT;
        let state = thread::blockIdx_x() as usize;
        let pair = thread::threadIdx_x() as usize;
        let heads = query_heads + key_heads;
        if state >= tokens * heads
            || pair >= ROTARY_PAIRS
            || tokens > ROPE_PAGED_KV_APPEND_MAX_TOKENS
        {
            return;
        }
        // SAFETY: each CTA owns this static shared allocation. Access is
        // partitioned by warp leader and synchronized in `block_has_invalid`.
        let validation_flags = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALIDATION_FLAGS) };
        let page_table_invalid = page_table_partition_invalid(
            pair,
            batch_size,
            max_num_pages,
            page_indptr,
            page_indices,
            last_page_len,
        );
        // SAFETY: all 64 threads participate, each warp leader owns one flag,
        // and the block barrier makes both writes visible before reads.
        if unsafe { block_has_invalid(page_table_invalid, pair, validation_flags) } {
            return;
        }

        let mut token_mapping_invalid = false;
        if pair < tokens {
            if let Some((_, _, first_page, first_offset)) = explicit_append_slot(
                pair,
                batch_size,
                batch_indices,
                positions,
                page_indptr,
                page_indices,
                last_page_len,
            ) {
                let mut second_token = pair + 1;
                while second_token < tokens {
                    let Some((_, _, second_page, second_offset)) = explicit_append_slot(
                        second_token,
                        batch_size,
                        batch_indices,
                        positions,
                        page_indptr,
                        page_indices,
                        last_page_len,
                    ) else {
                        token_mapping_invalid = true;
                        break;
                    };
                    if first_page == second_page && first_offset == second_offset {
                        token_mapping_invalid = true;
                        break;
                    }
                    second_token += 1;
                }
            } else {
                token_mapping_invalid = true;
            }
        }
        // SAFETY: as above. This second stage overwrites both flags before the
        // barrier, so no reset or atomic operation is required.
        if unsafe { block_has_invalid(token_mapping_invalid, pair, validation_flags) } {
            return;
        }

        let token = state / heads;
        let combined_head = state % heads;
        let Some((_, position, physical_page, page_offset)) = explicit_append_slot(
            token,
            batch_size,
            batch_indices,
            positions,
            page_indptr,
            page_indices,
            last_page_len,
        ) else {
            return;
        };
        let exponent = pair as f32 / ROTARY_PAIRS as f32;
        let inverse_frequency = (1.0_f32 / 10_000.0).powf(exponent);
        let angle = position as f32 * inverse_frequency;
        let (sin, cos) = angle.sin_cos();
        if combined_head < query_heads {
            let base = (token * query_heads + combined_head) * HEAD_DIM;
            // SAFETY: global metadata validation and one CTA per token/head
            // prove exact spans and unique Q output ownership.
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
            let source = (token * key_heads + key_head) * HEAD_DIM;
            let destination = ((physical_page * PAGED_BATCH_DECODE_PAGE_SIZE + page_offset)
                * key_heads
                + key_head)
                * HEAD_DIM;
            // SAFETY: global physical-slot uniqueness and one CTA per
            // token/KV-head prove unique K/V destinations.
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
    fn page_table_partition_invalid(
        pair: usize,
        batch_size: usize,
        max_num_pages: usize,
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
    ) -> bool {
        let mut invalid = false;
        if pair == 0 {
            let final_indptr = page_indptr[batch_size];
            invalid = page_indptr[0] != 0
                || final_indptr < 0
                || final_indptr as usize != page_indices.len();
        }
        let mut request = pair;
        while request < batch_size {
            let start = page_indptr[request];
            let end = page_indptr[request + 1];
            let tail = last_page_len[request];
            if start < 0
                || end <= start
                || end as usize > page_indices.len()
                || !(1..=PAGED_BATCH_DECODE_PAGE_SIZE as i32).contains(&tail)
            {
                invalid = true;
            }
            request += BLOCK_THREADS as usize;
        }
        let mut page = pair;
        while page < page_indices.len() {
            let physical_page = page_indices[page];
            if physical_page < 0 || physical_page as usize >= max_num_pages {
                invalid = true;
            }
            page += BLOCK_THREADS as usize;
        }
        invalid
    }

    #[inline(always)]
    unsafe fn block_has_invalid(invalid: bool, pair: usize, flags: *mut u32) -> bool {
        let warp_id = pair / 32;
        let lane = pair % 32;
        let warp_invalid = warp::any(invalid);
        if lane == 0 {
            // SAFETY: one leader per warp writes its disjoint flag.
            unsafe {
                flags.add(warp_id).write(warp_invalid as u32);
            }
        }
        thread::sync_threads();
        // SAFETY: the barrier completed both leader writes.
        let invalid = unsafe { flags.read() != 0 || flags.add(1).read() != 0 };
        // Prevent the next validation stage from overwriting either flag
        // while another thread is still reading this stage's consensus.
        thread::sync_threads();
        invalid
    }

    #[inline(always)]
    fn explicit_append_slot(
        token: usize,
        batch_size: usize,
        batch_indices: &[i32],
        positions: &[i32],
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
    ) -> Option<(usize, usize, usize, usize)> {
        let request = batch_indices[token];
        let position = positions[token];
        if request < 0 || request as usize >= batch_size || position < 0 {
            return None;
        }
        let request = request as usize;
        let page_start = page_indptr[request] as usize;
        let page_end = page_indptr[request + 1] as usize;
        let page_count = page_end - page_start;
        let kv_len =
            (page_count - 1) * PAGED_BATCH_DECODE_PAGE_SIZE + last_page_len[request] as usize;
        let position = position as usize;
        if position >= kv_len {
            return None;
        }
        let page_slot = position / PAGED_BATCH_DECODE_PAGE_SIZE;
        let page_offset = position % PAGED_BATCH_DECODE_PAGE_SIZE;
        let physical_page = page_indices[page_start + page_slot] as usize;
        Some((request, position, physical_page, page_offset))
    }

    #[inline(always)]
    fn append_slot(
        request: usize,
        batch_size: usize,
        max_num_pages: usize,
        page_indptr: &[i32],
        page_indices: &[i32],
        last_page_len: &[i32],
    ) -> Option<(usize, usize, usize)> {
        if request >= batch_size {
            return None;
        }
        let page_start = page_indptr[request];
        let page_end = page_indptr[request + 1];
        let tail = last_page_len[request];
        if page_start < 0
            || page_end <= page_start
            || page_end as usize > page_indices.len()
            || !(1..=PAGED_BATCH_DECODE_PAGE_SIZE as i32).contains(&tail)
        {
            return None;
        }
        let page_slot = (page_end - page_start - 1) as usize;
        let physical_page = page_indices[page_start as usize + page_slot];
        if physical_page < 0 || physical_page as usize >= max_num_pages {
            return None;
        }
        let position = page_slot * PAGED_BATCH_DECODE_PAGE_SIZE + tail as usize - 1;
        Some((position, physical_page as usize, tail as usize - 1))
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
        let launch =
            self.module
                .prepare_rope_paged_kv_append_bf16_neox_d128(LaunchConfig1D::new(
                    blocks,
                    BLOCK_THREADS,
                    0,
                ))?;
        Ok(Bf16RopePagedKvAppendPlan {
            spec,
            module: self.module.clone(),
            launch,
        })
    }

    pub fn plan_bf16_paged_append_tokens(
        &self,
        spec: Bf16RopePagedKvAppendTokensSpec,
    ) -> Result<Bf16RopePagedKvAppendTokensPlan, RopePlanError> {
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
        let launch = self
            .module
            .prepare_rope_paged_kv_append_tokens_bf16_neox_d128(LaunchConfig1D::new(
                blocks,
                BLOCK_THREADS,
                0,
            ))?;
        Ok(Bf16RopePagedKvAppendTokensPlan {
            spec,
            module: self.module.clone(),
            launch,
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
            let result = self.module.rope_pos_ids_bf16_neox_d128(
                resolved.stream,
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
            (self.launch.function().clone(), result)
        };
        record_launch(scope, permit, function, result)
    }
}

/// Immutable prepared launch for fused RoPE plus paged KV append.
#[derive(Clone)]
pub struct Bf16RopePagedKvAppendPlan {
    spec: Bf16RopePagedKvAppendSpec,
    module: kernels::LoadedModule,
    launch: PreparedLaunch<kernels::__rope_paged_kv_append_bf16_neox_d128_CudaKernel>,
}

impl Bf16RopePagedKvAppendPlan {
    pub const fn spec(&self) -> Bf16RopePagedKvAppendSpec {
        self.spec
    }

    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16RopePagedKvAppendArgs,
    ) -> Result<(), RopeEnqueueError> {
        let permit = scope.prepare_command()?;
        let (function, result) = {
            let resolved = scope.resolve_rrrrrrwww(
                args.query,
                args.key,
                args.value,
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.query_output,
                args.key_pages,
                args.value_pages,
            )?;
            require_exact_len("query", resolved.first.len(), self.spec.query_numel())?;
            require_exact_len("key", resolved.second.len(), self.spec.key_numel())?;
            require_exact_len("value", resolved.third.len(), self.spec.value_numel())?;
            require_exact_len(
                "page_indptr",
                resolved.fourth.len(),
                self.spec.page_indptr_numel(),
            )?;
            if resolved.fifth.len() < self.spec.batch_size() {
                return Err(RopeEnqueueError::PageIndicesTooShort {
                    minimum: self.spec.batch_size(),
                    actual: resolved.fifth.len(),
                });
            }
            require_exact_len(
                "last_page_len",
                resolved.sixth.len(),
                self.spec.last_page_len_numel(),
            )?;
            require_exact_len(
                "query_output",
                resolved.seventh.len(),
                self.spec.query_output_numel(),
            )?;
            require_exact_len(
                "key_pages",
                resolved.eighth.len(),
                self.spec.kv_pages_numel(),
            )?;
            require_exact_len(
                "value_pages",
                resolved.ninth.len(),
                self.spec.kv_pages_numel(),
            )?;
            let result = self.module.rope_paged_kv_append_bf16_neox_d128(
                resolved.stream,
                &self.launch,
                self.spec.batch_size(),
                self.spec.max_num_pages(),
                self.spec.num_query_heads(),
                self.spec.num_kv_heads(),
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
            (self.launch.function().clone(), result)
        };
        record_launch(scope, permit, function, result)
    }
}

/// Immutable prepared launch for explicit multi-token fused RoPE append.
#[derive(Clone)]
pub struct Bf16RopePagedKvAppendTokensPlan {
    spec: Bf16RopePagedKvAppendTokensSpec,
    module: kernels::LoadedModule,
    launch: PreparedLaunch<kernels::__rope_paged_kv_append_tokens_bf16_neox_d128_CudaKernel>,
}

impl Bf16RopePagedKvAppendTokensPlan {
    pub const fn spec(&self) -> Bf16RopePagedKvAppendTokensSpec {
        self.spec
    }

    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        args: Bf16RopePagedKvAppendTokensArgs,
    ) -> Result<(), RopeEnqueueError> {
        let permit = scope.prepare_command()?;
        let (function, result) = {
            let resolved = scope.resolve_rrrrrrrrwww(
                args.query,
                args.key,
                args.value,
                args.batch_indices,
                args.positions,
                args.page_indptr,
                args.page_indices,
                args.last_page_len,
                args.query_output,
                args.key_pages,
                args.value_pages,
            )?;
            require_exact_len("query", resolved.first.len(), self.spec.query_numel())?;
            require_exact_len("key", resolved.second.len(), self.spec.key_numel())?;
            require_exact_len("value", resolved.third.len(), self.spec.value_numel())?;
            require_exact_len(
                "batch_indices",
                resolved.fourth.len(),
                self.spec.batch_indices_numel(),
            )?;
            require_exact_len(
                "positions",
                resolved.fifth.len(),
                self.spec.positions_numel(),
            )?;
            require_exact_len(
                "page_indptr",
                resolved.sixth.len(),
                self.spec.page_indptr_numel(),
            )?;
            if resolved.seventh.len() < self.spec.batch_size() {
                return Err(RopeEnqueueError::PageIndicesTooShort {
                    minimum: self.spec.batch_size(),
                    actual: resolved.seventh.len(),
                });
            }
            require_exact_len(
                "last_page_len",
                resolved.eighth.len(),
                self.spec.last_page_len_numel(),
            )?;
            require_exact_len(
                "query_output",
                resolved.ninth.len(),
                self.spec.query_output_numel(),
            )?;
            require_exact_len(
                "key_pages",
                resolved.tenth.len(),
                self.spec.kv_pages_numel(),
            )?;
            require_exact_len(
                "value_pages",
                resolved.eleventh.len(),
                self.spec.kv_pages_numel(),
            )?;
            let result = self.module.rope_paged_kv_append_tokens_bf16_neox_d128(
                resolved.stream,
                &self.launch,
                self.spec.tokens(),
                self.spec.batch_size(),
                self.spec.max_num_pages(),
                self.spec.num_query_heads(),
                self.spec.num_kv_heads(),
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
                resolved.eleventh,
            );
            (self.launch.function().clone(), result)
        };
        record_launch(scope, permit, function, result)
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
pub struct Bf16RopePagedKvAppendArgs {
    query: Read<bf16>,
    key: Read<bf16>,
    value: Read<bf16>,
    page_indptr: Read<i32>,
    page_indices: Read<i32>,
    last_page_len: Read<i32>,
    query_output: Write<bf16>,
    key_pages: Write<bf16>,
    value_pages: Write<bf16>,
}

impl Bf16RopePagedKvAppendArgs {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        query: Read<bf16>,
        key: Read<bf16>,
        value: Read<bf16>,
        page_indptr: Read<i32>,
        page_indices: Read<i32>,
        last_page_len: Read<i32>,
        query_output: Write<bf16>,
        key_pages: Write<bf16>,
        value_pages: Write<bf16>,
    ) -> Self {
        Self {
            query,
            key,
            value,
            page_indptr,
            page_indices,
            last_page_len,
            query_output,
            key_pages,
            value_pages,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bf16RopePagedKvAppendTokensArgs {
    query: Read<bf16>,
    key: Read<bf16>,
    value: Read<bf16>,
    batch_indices: Read<i32>,
    positions: Read<i32>,
    page_indptr: Read<i32>,
    page_indices: Read<i32>,
    last_page_len: Read<i32>,
    query_output: Write<bf16>,
    key_pages: Write<bf16>,
    value_pages: Write<bf16>,
}

impl Bf16RopePagedKvAppendTokensArgs {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        query: Read<bf16>,
        key: Read<bf16>,
        value: Read<bf16>,
        batch_indices: Read<i32>,
        positions: Read<i32>,
        page_indptr: Read<i32>,
        page_indices: Read<i32>,
        last_page_len: Read<i32>,
        query_output: Write<bf16>,
        key_pages: Write<bf16>,
        value_pages: Write<bf16>,
    ) -> Self {
        Self {
            query,
            key,
            value,
            batch_indices,
            positions,
            page_indptr,
            page_indices,
            last_page_len,
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
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
}

#[derive(Debug, Error)]
pub enum RopeEnqueueError {
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
    #[error("{operand} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("page_indices requires at least {minimum} entries, got {actual}")]
    PageIndicesTooShort { minimum: usize, actual: usize },
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
    result: Result<(), LaunchContractError>,
) -> Result<(), RopeEnqueueError> {
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
