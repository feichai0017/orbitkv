//! cuda-oxide provider for BF16 ragged causal prefill attention.

use crate::command::{CommandError, CommandPermit, CommandScope, Read, Write};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, convert, cuda_module, float, kernel, launch_bounds, launch_contract, tcgen05,
    thread, warp,
};
use half::bf16;
use loom_infer::{Bf16RaggedPrefillSpec, SINGLE_DECODE_HEAD_DIM};
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

const WARP_THREADS: u32 = 32;
const BF16_PAIRS_PER_HEAD: usize = SINGLE_DECODE_HEAD_DIM / 2;

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
        let launch = self
            .module
            .prepare_ragged_prefill_bf16_nhd_causal(LaunchConfig1D::new(states, WARP_THREADS, 0))?;
        Ok(Bf16RaggedPrefillPlan {
            spec,
            module: self.module.clone(),
            launch,
        })
    }
}

#[derive(Clone)]
pub struct Bf16RaggedPrefillPlan {
    spec: Bf16RaggedPrefillSpec,
    module: kernels::LoadedModule,
    launch: PreparedLaunch<kernels::__ragged_prefill_bf16_nhd_causal_CudaKernel>,
}

impl Bf16RaggedPrefillPlan {
    pub const fn spec(&self) -> Bf16RaggedPrefillSpec {
        self.spec
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
            let result = self.module.ragged_prefill_bf16_nhd_causal(
                resolved.stream,
                &self.launch,
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
                resolved.seventh,
            );
            (self.launch.function().clone(), result)
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
