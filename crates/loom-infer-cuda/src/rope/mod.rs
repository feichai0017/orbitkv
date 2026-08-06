//! cuda-oxide provider for standard BF16 NeoX rotary position embedding.

use crate::command::{CommandError, CommandPermit, CommandScope, Read, Write};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, convert, cuda_module, kernel, launch_bounds, launch_contract, tcgen05, thread,
};
use half::bf16;
use loom_infer::Bf16RopePosIdsSpec;
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
                rotate_pair(input, output, base, pair, sin, cos);
            }
        } else {
            let key_head = combined_head - query_heads;
            let base = (token * key_heads + key_head) * HEAD_DIM;
            let input = key.as_ptr().cast::<u16>();
            let output = key_output.as_mut_ptr().cast::<u16>();
            // SAFETY: as above, for the disjoint K state.
            unsafe {
                rotate_pair(input, output, base, pair, sin, cos);
            }
        }
    }

    #[inline(always)]
    unsafe fn rotate_pair(
        input: *const u16,
        output: *mut u16,
        base: usize,
        pair: usize,
        sin: f32,
        cos: f32,
    ) {
        // SAFETY: the caller proves both split-half indices are in one exact
        // D128 state and uniquely owned by this thread.
        let first_bits = unsafe { input.add(base + pair).read() };
        // SAFETY: as above, for the paired second-half component.
        let second_bits = unsafe { input.add(base + ROTARY_PAIRS + pair).read() };
        let first = convert::cvt_f32_bf16x2_lo(first_bits as u32);
        let second = convert::cvt_f32_bf16x2_lo(second_bits as u32);
        let rotated =
            tcgen05::cvt_f32x2_bf16x2(first * cos - second * sin, second * cos + first * sin);
        // SAFETY: both output components are in the exact D128 state and no
        // other thread writes this pair.
        unsafe {
            output.add(base + pair).write(rotated as u16);
            output
                .add(base + ROTARY_PAIRS + pair)
                .write((rotated >> 16) as u16);
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
