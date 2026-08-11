//! SM90a SIMT kernels for the Oxide GEMM provider.

use crate::command::{CommandPermit, CommandScope, ResolvedRrww};
use crate::gemm::plan::{Bf16DenseGemmEnqueueError, Bf16DenseGemmOperands, Bf16DenseGemmPlanError};
use crate::memory::{DeviceRegionLaunchError, enqueue_region_launch};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, PreparedLaunch};
use cuda_device::{
    DisjointSlice, convert, cuda_module, kernel, launch_bounds, launch_contract, tcgen05, thread,
    warp,
};
use half::bf16;
use oxide_infer::Bf16DenseGemmSpec;
use std::mem::size_of;
use std::sync::Arc;

const BLOCK_THREADS: u32 = 256;
const WARPS_PER_BLOCK: usize = 8;
const OUTPUTS_PER_WARP: usize = 2;
const OUTPUTS_PER_BLOCK: usize = WARPS_PER_BLOCK * OUTPUTS_PER_WARP;
const N_MULTIPLE: usize = 16;
const K_MULTIPLE: usize = 64;
const TENSOR_ALIGNMENT_BYTES: u64 = size_of::<u32>() as u64;
const WORKSPACE_ALIGNMENT_BYTES: u64 = 1;

const _: () = {
    assert!(OUTPUTS_PER_BLOCK == N_MULTIPLE);
    assert!(core::mem::size_of::<bf16>() == core::mem::size_of::<u16>());
    assert!(core::mem::align_of::<bf16>() == core::mem::align_of::<u16>());
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        min_compute_capability = (9, 0),
        requires = (
            n >= 16,
            k >= 64,
            activation.len() == k,
            weight.len() == n * k,
            output.len() == n,
        ),
    )]
    pub fn oxide_sm90_simt_gemv_m1_n16_k64_bf16(
        n: usize,
        k: usize,
        activation: &[bf16],
        weight: &[bf16],
        mut output: DisjointSlice<bf16>,
    ) {
        let lane = warp::lane_id() as usize;
        let warp_id = warp::warp_id() as usize;
        let first_output =
            thread::blockIdx_x() as usize * OUTPUTS_PER_BLOCK + warp_id * OUTPUTS_PER_WARP;
        if first_output + 1 >= n {
            return;
        }

        let pairs_per_row = k / 2;
        let activation_pairs = activation.as_ptr().cast::<u32>();
        let weight_pairs = weight.as_ptr().cast::<u32>();
        let output_pairs = output.as_mut_ptr().cast::<u32>();
        let first_weight_row = first_output * pairs_per_row;
        let second_weight_row = first_weight_row + pairs_per_row;

        let mut first_sum = 0.0_f32;
        let mut second_sum = 0.0_f32;
        let mut pair = lane;
        while pair < pairs_per_row {
            // SAFETY: explicit planning proves K is a multiple of 64 and all
            // buffers are four-byte aligned. Each lane walks its own packed
            // BF16 pairs through the complete K reduction.
            unsafe {
                let (a_lo, a_hi) = convert::cvt_f32x2_bf16x2(activation_pairs.add(pair).read());
                let (first_w_lo, first_w_hi) =
                    convert::cvt_f32x2_bf16x2(weight_pairs.add(first_weight_row + pair).read());
                let (second_w_lo, second_w_hi) =
                    convert::cvt_f32x2_bf16x2(weight_pairs.add(second_weight_row + pair).read());
                first_sum += a_lo * first_w_lo;
                first_sum += a_hi * first_w_hi;
                second_sum += a_lo * second_w_lo;
                second_sum += a_hi * second_w_hi;
            }
            pair += 32;
        }

        first_sum = warp::reduce_sum_f32(first_sum);
        second_sum = warp::reduce_sum_f32(second_sum);
        if lane == 0 {
            // SAFETY: N is a multiple of 16, so every launched warp owns one
            // in-range adjacent output pair. No other warp writes this pair.
            unsafe {
                output_pairs
                    .add(first_output / OUTPUTS_PER_WARP)
                    .write(tcgen05::cvt_f32x2_bf16x2(first_sum, second_sum));
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct Sm90Provider {
    module: kernels::LoadedModule,
}

impl Sm90Provider {
    pub(super) fn load(context: &Arc<CudaContext>) -> Result<Self, cuda_host::EmbeddedModuleError> {
        // SAFETY: the named package artifact is inspected and fixed to
        // sm_90a before this loader is reached. This module owns the entry ABI.
        let module = unsafe { kernels::load(context)? };
        Ok(Self { module })
    }

    pub(super) fn plan_bf16_dense(
        &self,
        spec: Bf16DenseGemmSpec,
    ) -> Result<OxideBf16DensePlan, Bf16DenseGemmPlanError> {
        let blocks = checked_grid_blocks(spec.n())?;
        let launch =
            self.module
                .prepare_oxide_sm90_simt_gemv_m1_n16_k64_bf16(LaunchConfig1D::new(
                    blocks,
                    BLOCK_THREADS,
                    0,
                ))?;
        Ok(OxideBf16DensePlan {
            spec,
            module: self.module.clone(),
            launch,
        })
    }
}

/// One immutable prepared Oxide SM90a GEMV launch.
#[derive(Clone)]
pub(crate) struct OxideBf16DensePlan {
    spec: Bf16DenseGemmSpec,
    module: kernels::LoadedModule,
    launch: PreparedLaunch<kernels::__oxide_sm90_simt_gemv_m1_n16_k64_bf16_CudaKernel>,
}

impl OxideBf16DensePlan {
    pub(crate) const fn spec(&self) -> Bf16DenseGemmSpec {
        self.spec
    }

    pub(crate) const fn workspace_required_bytes(&self) -> usize {
        0
    }

    pub(crate) const fn tensor_alignment_bytes(&self) -> u64 {
        TENSOR_ALIGNMENT_BYTES
    }

    pub(crate) const fn workspace_alignment_bytes(&self) -> u64 {
        WORKSPACE_ALIGNMENT_BYTES
    }

    pub(crate) fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        operands: Bf16DenseGemmOperands,
    ) -> Result<(), Bf16DenseGemmEnqueueError> {
        let permit = scope.prepare_command()?;
        let launch_result = {
            let resolved = scope.resolve_rrww(
                operands.activation(),
                operands.weight(),
                operands.output(),
                operands.workspace(),
            )?;
            self.validate_resolved(&resolved)?;
            let operation = self.module.oxide_sm90_simt_gemv_m1_n16_k64_bf16_async(
                &self.launch,
                self.spec.n(),
                self.spec.k(),
                resolved.first,
                resolved.second,
                resolved.third,
            );
            enqueue_region_launch(resolved.stream, operation)
        };
        record_launch(scope, permit, self.launch.function().clone(), launch_result)
    }

    fn validate_resolved(
        &self,
        resolved: &ResolvedRrww<'_, bf16, bf16, bf16, u8>,
    ) -> Result<(), Bf16DenseGemmEnqueueError> {
        require_exact_len("A", resolved.first.len(), self.spec.a_numel())?;
        require_exact_len("W", resolved.second.len(), self.spec.weight_numel())?;
        require_exact_len("D", resolved.third.len(), self.spec.output_numel())?;
        require_alignment("A", resolved.first.cu_deviceptr())?;
        require_alignment("W", resolved.second.cu_deviceptr())?;
        require_alignment("D", resolved.third.cu_deviceptr())?;

        let plan_context = self.module.as_cuda_module().context();
        let stream_context = resolved.stream.context();
        if plan_context.cu_ctx() != stream_context.cu_ctx() {
            return Err(Bf16DenseGemmEnqueueError::ContextMismatch {
                plan_device: plan_context.ordinal(),
                stream_device: stream_context.ordinal(),
            });
        }
        Ok(())
    }
}

pub(super) fn validate_spec(spec: Bf16DenseGemmSpec) -> Result<(), Bf16DenseGemmPlanError> {
    if spec.m() != 1 {
        return Err(Bf16DenseGemmPlanError::OxideMNotOne { m: spec.m() });
    }
    if !spec.n().is_multiple_of(N_MULTIPLE) {
        return Err(Bf16DenseGemmPlanError::OxideNNotMultipleOf16 { n: spec.n() });
    }
    if !spec.k().is_multiple_of(K_MULTIPLE) {
        return Err(Bf16DenseGemmPlanError::OxideKNotMultipleOf64 { k: spec.k() });
    }
    checked_grid_blocks(spec.n())?;
    Ok(())
}

fn checked_grid_blocks(n: usize) -> Result<u32, Bf16DenseGemmPlanError> {
    let blocks = n / OUTPUTS_PER_BLOCK;
    u32::try_from(blocks)
        .map_err(|_| Bf16DenseGemmPlanError::OxideGridDimensionOutOfRange { blocks })
}

fn require_exact_len(
    operand: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), Bf16DenseGemmEnqueueError> {
    if actual == expected {
        Ok(())
    } else {
        Err(Bf16DenseGemmEnqueueError::LengthMismatch {
            operand,
            expected,
            actual,
        })
    }
}

fn require_alignment(operand: &'static str, address: u64) -> Result<(), Bf16DenseGemmEnqueueError> {
    if address.is_multiple_of(TENSOR_ALIGNMENT_BYTES) {
        Ok(())
    } else {
        Err(Bf16DenseGemmEnqueueError::MisalignedBuffer {
            operand,
            address,
            alignment: TENSOR_ALIGNMENT_BYTES,
        })
    }
}

fn record_launch(
    scope: &mut CommandScope<'_>,
    permit: CommandPermit,
    function: CudaFunction,
    result: Result<(), DeviceRegionLaunchError>,
) -> Result<(), Bf16DenseGemmEnqueueError> {
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

    fn spec(m: usize, n: usize, k: usize) -> Bf16DenseGemmSpec {
        Bf16DenseGemmSpec::new(m, n, k).unwrap()
    }

    #[test]
    fn admission_accepts_every_census_m1_shape() {
        for (n, k) in [
            (1536, 1536),
            (256, 1536),
            (17920, 1536),
            (1536, 8960),
            (151936, 1536),
        ] {
            validate_spec(spec(1, n, k)).unwrap();
        }
    }

    #[test]
    fn admission_rejects_non_m1() {
        assert!(matches!(
            validate_spec(spec(2, 16, 64)),
            Err(Bf16DenseGemmPlanError::OxideMNotOne { m: 2 })
        ));
    }

    #[test]
    fn admission_rejects_n_outside_tile_contract() {
        assert!(matches!(
            validate_spec(spec(1, 17, 64)),
            Err(Bf16DenseGemmPlanError::OxideNNotMultipleOf16 { n: 17 })
        ));
    }

    #[test]
    fn admission_rejects_k_outside_tile_contract() {
        assert!(matches!(
            validate_spec(spec(1, 16, 65)),
            Err(Bf16DenseGemmPlanError::OxideKNotMultipleOf64 { k: 65 })
        ));
    }

    #[test]
    fn grid_gate_rejects_more_than_u32_blocks() {
        let blocks = u32::MAX as usize + 1;
        let n = blocks * OUTPUTS_PER_BLOCK;
        assert!(matches!(
            validate_spec(spec(1, n, K_MULTIPLE)),
            Err(Bf16DenseGemmPlanError::OxideGridDimensionOutOfRange {
                blocks: actual
            }) if actual == blocks
        ));
    }
}
