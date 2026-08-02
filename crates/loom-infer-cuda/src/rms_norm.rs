//! cuda-oxide implementation and prepared launch for RMSNorm.

use crate::command::{BindingElement, CommandError, CommandScope, Read, Write};
use cuda_core::{CudaContext, CudaFunction, LaunchConfig1D, LaunchContractError, PreparedLaunch};
use cuda_device::{
    DisjointSlice, SharedArray, convert, cuda_module, float, kernel, launch_bounds,
    launch_contract, tcgen05, thread, warp,
};
use half::{bf16, f16};
use loom_infer::{DType, RmsNormSpec};
use std::sync::Arc;
use thiserror::Error;

const BLOCK_THREADS: u32 = 256;
const WARP_COUNT: usize = 8;

const _: () = {
    assert!(core::mem::size_of::<f16>() == core::mem::size_of::<u16>());
    assert!(core::mem::align_of::<f16>() == core::mem::align_of::<u16>());
    assert!(core::mem::size_of::<bf16>() == core::mem::size_of::<u16>());
    assert!(core::mem::align_of::<bf16>() == core::mem::align_of::<u16>());
};

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn reduce_inverse_rms_with_epsilon(
        mut square_sum: f32,
        hidden_size: usize,
        epsilon: f32,
        warp_sums: *mut f32,
    ) -> f32 {
        let thread_id = thread::threadIdx_x() as usize;
        let lane_id = warp::lane_id();
        let warp_id = warp::warp_id() as usize;

        square_sum = warp::reduce_sum_f32(square_sum);
        if lane_id == 0 {
            // SAFETY: each warp owns one slot. The block barrier publishes all
            // eight writes before warp zero starts the second reduction.
            unsafe { warp_sums.add(warp_id).write(square_sum) };
        }
        thread::sync_threads();

        if warp_id == 0 {
            // SAFETY: the first barrier initialized all eight shared slots.
            let partial = if thread_id < WARP_COUNT {
                unsafe { warp_sums.add(thread_id).read() }
            } else {
                0.0
            };
            let block_sum = warp::reduce_sum_f32(partial);
            if lane_id == 0 {
                let mean_square = block_sum / hidden_size as f32;
                let inverse_rms = 1.0 / float::sqrt_rn_f32(mean_square + epsilon);
                // SAFETY: lane zero is the only writer. The next barrier
                // publishes the value before any thread reads it.
                unsafe { warp_sums.write(inverse_rms) };
            }
        }
        thread::sync_threads();

        // SAFETY: the second barrier initialized slot zero. No thread writes
        // shared memory after this read.
        unsafe { warp_sums.read() }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            rows >= 1,
            hidden_size >= 1,
            input.len() == rows * hidden_size,
            weight.len() == hidden_size,
            output.len() == rows * hidden_size,
        ),
    )]
    pub fn rms_norm_f32(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        input: &[f32],
        weight: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, WARP_COUNT> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        if row >= rows {
            return;
        }

        let thread_id = thread::threadIdx_x() as usize;
        let row_offset = row * hidden_size;
        let warp_sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_SUMS) };

        let mut square_sum = 0.0_f32;
        let mut column = thread_id;
        while column < hidden_size {
            let value = input[row_offset + column];
            square_sum += value * value;
            column += BLOCK_THREADS as usize;
        }
        let inverse_rms =
            reduce_inverse_rms_with_epsilon(square_sum, hidden_size, epsilon, warp_sums);
        column = thread_id;
        while column < hidden_size {
            let index = row_offset + column;
            // SAFETY: the launch contract proves every buffer span. Columns
            // assigned modulo 256 give each output element one writer.
            unsafe {
                *output.get_unchecked_mut(index) = input[index] * inverse_rms * weight[column];
            }
            column += BLOCK_THREADS as usize;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        min_compute_capability = (8, 0),
        requires = (
            rows >= 1,
            hidden_size >= 1,
            input.len() == rows * hidden_size,
            weight.len() == hidden_size,
            output.len() == rows * hidden_size,
        ),
    )]
    pub fn rms_norm_f16_scalar(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        input: &[f16],
        weight: &[f16],
        mut output: DisjointSlice<f16>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, WARP_COUNT> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        if row >= rows {
            return;
        }

        let thread_id = thread::threadIdx_x() as usize;
        let row_offset = row * hidden_size;
        let warp_sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_SUMS) };
        let input_bits = input.as_ptr().cast::<u16>();
        let weight_bits = weight.as_ptr().cast::<u16>();
        let output_bits = output.as_mut_ptr().cast::<u16>();

        let mut square_sum = 0.0_f32;
        let mut column = thread_id;
        while column < hidden_size {
            let index = row_offset + column;
            // SAFETY: the launch contract proves the element span, and f16 is
            // repr-transparent over u16 with identical alignment.
            let bits = unsafe { input_bits.add(index).read() };
            let value = convert::cvt_f32_f16x2_lo(bits as u32);
            square_sum += value * value;
            column += BLOCK_THREADS as usize;
        }
        let inverse_rms =
            reduce_inverse_rms_with_epsilon(square_sum, hidden_size, epsilon, warp_sums);

        column = thread_id;
        while column < hidden_size {
            let index = row_offset + column;
            // SAFETY: all reads are in the launch-contract spans. Columns
            // assigned modulo 256 give each output element one writer.
            unsafe {
                let value = convert::cvt_f32_f16x2_lo(input_bits.add(index).read() as u32);
                let scale = convert::cvt_f32_f16x2_lo(weight_bits.add(column).read() as u32);
                let packed = convert::cvt_f16x2_f32(value * inverse_rms * scale, 0.0);
                output_bits.add(index).write(packed as u16);
            }
            column += BLOCK_THREADS as usize;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        min_compute_capability = (8, 0),
        requires = (
            rows >= 1,
            hidden_size >= 2,
            input.len() == rows * hidden_size,
            weight.len() == hidden_size,
            output.len() == rows * hidden_size,
        ),
    )]
    pub fn rms_norm_f16_packed2(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        input: &[f16],
        weight: &[f16],
        mut output: DisjointSlice<f16>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, WARP_COUNT> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        if row >= rows {
            return;
        }

        let thread_id = thread::threadIdx_x() as usize;
        let pairs_per_row = hidden_size / 2;
        let row_pair_offset = row * pairs_per_row;
        let warp_sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_SUMS) };
        let input_pairs = input.as_ptr().cast::<u32>();
        let weight_pairs = weight.as_ptr().cast::<u32>();
        let output_pairs = output.as_mut_ptr().cast::<u32>();

        let mut square_sum = 0.0_f32;
        let mut pair_column = thread_id;
        while pair_column < pairs_per_row {
            // SAFETY: the plan selects this kernel only for even hidden sizes
            // and validates 4-byte alignment for all three buffers.
            let packed = unsafe { input_pairs.add(row_pair_offset + pair_column).read() };
            let (lo, hi) = convert::cvt_f32x2_f16x2(packed);
            square_sum += lo * lo + hi * hi;
            pair_column += BLOCK_THREADS as usize;
        }
        let inverse_rms =
            reduce_inverse_rms_with_epsilon(square_sum, hidden_size, epsilon, warp_sums);

        pair_column = thread_id;
        while pair_column < pairs_per_row {
            // SAFETY: pair indices cover the exact contract-proven spans, and
            // each pair is owned by one thread.
            unsafe {
                let input_pair = input_pairs.add(row_pair_offset + pair_column).read();
                let weight_pair = weight_pairs.add(pair_column).read();
                let (input_lo, input_hi) = convert::cvt_f32x2_f16x2(input_pair);
                let (weight_lo, weight_hi) = convert::cvt_f32x2_f16x2(weight_pair);
                let packed = convert::cvt_f16x2_f32(
                    input_lo * inverse_rms * weight_lo,
                    input_hi * inverse_rms * weight_hi,
                );
                output_pairs
                    .add(row_pair_offset + pair_column)
                    .write(packed);
            }
            pair_column += BLOCK_THREADS as usize;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        min_compute_capability = (8, 0),
        requires = (
            rows >= 1,
            hidden_size >= 1,
            input.len() == rows * hidden_size,
            weight.len() == hidden_size,
            output.len() == rows * hidden_size,
        ),
    )]
    pub fn rms_norm_bf16_scalar(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        input: &[bf16],
        weight: &[bf16],
        mut output: DisjointSlice<bf16>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, WARP_COUNT> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        if row >= rows {
            return;
        }

        let thread_id = thread::threadIdx_x() as usize;
        let row_offset = row * hidden_size;
        let warp_sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_SUMS) };
        let input_bits = input.as_ptr().cast::<u16>();
        let weight_bits = weight.as_ptr().cast::<u16>();
        let output_bits = output.as_mut_ptr().cast::<u16>();

        let mut square_sum = 0.0_f32;
        let mut column = thread_id;
        while column < hidden_size {
            let index = row_offset + column;
            // SAFETY: the launch contract proves the element span, and bf16 is
            // repr-transparent over u16 with identical alignment.
            let bits = unsafe { input_bits.add(index).read() };
            let value = convert::cvt_f32_bf16x2_lo(bits as u32);
            square_sum += value * value;
            column += BLOCK_THREADS as usize;
        }
        let inverse_rms =
            reduce_inverse_rms_with_epsilon(square_sum, hidden_size, epsilon, warp_sums);

        column = thread_id;
        while column < hidden_size {
            let index = row_offset + column;
            // SAFETY: all reads are in the launch-contract spans. Columns
            // assigned modulo 256 give each output element one writer.
            unsafe {
                let value = convert::cvt_f32_bf16x2_lo(input_bits.add(index).read() as u32);
                let scale = convert::cvt_f32_bf16x2_lo(weight_bits.add(column).read() as u32);
                let packed = tcgen05::cvt_f32x2_bf16x2(value * inverse_rms * scale, 0.0);
                output_bits.add(index).write(packed as u16);
            }
            column += BLOCK_THREADS as usize;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        min_compute_capability = (8, 0),
        requires = (
            rows >= 1,
            hidden_size >= 2,
            input.len() == rows * hidden_size,
            weight.len() == hidden_size,
            output.len() == rows * hidden_size,
        ),
    )]
    pub fn rms_norm_bf16_packed2(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        input: &[bf16],
        weight: &[bf16],
        mut output: DisjointSlice<bf16>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, WARP_COUNT> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        if row >= rows {
            return;
        }

        let thread_id = thread::threadIdx_x() as usize;
        let pairs_per_row = hidden_size / 2;
        let row_pair_offset = row * pairs_per_row;
        let warp_sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_SUMS) };
        let input_pairs = input.as_ptr().cast::<u32>();
        let weight_pairs = weight.as_ptr().cast::<u32>();
        let output_pairs = output.as_mut_ptr().cast::<u32>();

        let mut square_sum = 0.0_f32;
        let mut pair_column = thread_id;
        while pair_column < pairs_per_row {
            // SAFETY: the plan selects this kernel only for even hidden sizes
            // and validates 4-byte alignment for all three buffers.
            let packed = unsafe { input_pairs.add(row_pair_offset + pair_column).read() };
            let (lo, hi) = convert::cvt_f32x2_bf16x2(packed);
            square_sum += lo * lo + hi * hi;
            pair_column += BLOCK_THREADS as usize;
        }
        let inverse_rms =
            reduce_inverse_rms_with_epsilon(square_sum, hidden_size, epsilon, warp_sums);

        pair_column = thread_id;
        while pair_column < pairs_per_row {
            // SAFETY: pair indices cover the exact contract-proven spans, and
            // each pair is owned by one thread.
            unsafe {
                let input_pair = input_pairs.add(row_pair_offset + pair_column).read();
                let weight_pair = weight_pairs.add(pair_column).read();
                let (input_lo, input_hi) = convert::cvt_f32x2_bf16x2(input_pair);
                let (weight_lo, weight_hi) = convert::cvt_f32x2_bf16x2(weight_pair);
                let packed = tcgen05::cvt_f32x2_bf16x2(
                    input_lo * inverse_rms * weight_lo,
                    input_hi * inverse_rms * weight_hi,
                );
                output_pairs
                    .add(row_pair_offset + pair_column)
                    .write(packed);
            }
            pair_column += BLOCK_THREADS as usize;
        }
    }
}

/// A loaded RMSNorm CUDA module.
#[derive(Clone, Debug)]
pub struct RmsNormProvider {
    module: kernels::LoadedModule,
}

impl RmsNormProvider {
    /// Loads the embedded cuda-oxide artifact into `context`.
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, cuda_host::EmbeddedModuleError> {
        // SAFETY: this crate owns one package-named device bundle. The module
        // above defines its only entry point.
        let module = unsafe { kernels::load(context)? };
        Ok(Self { module })
    }

    /// Creates an immutable F32 launch plan.
    pub fn plan_f32(&self, spec: RmsNormSpec) -> Result<RmsNormF32Plan, RmsNormPlanError> {
        if spec.dtype() != DType::F32 {
            return Err(RmsNormPlanError::UnsupportedDType {
                expected: DType::F32,
                actual: spec.dtype(),
            });
        }
        let rows = checked_rows(spec)?;
        let launch =
            self.module
                .prepare_rms_norm_f32(LaunchConfig1D::new(rows, BLOCK_THREADS, 0))?;

        Ok(RmsNormF32Plan {
            spec,
            module: self.module.clone(),
            launch,
        })
    }

    /// Creates an immutable FP16 launch plan.
    pub fn plan_f16(&self, spec: RmsNormSpec) -> Result<RmsNormF16Plan, RmsNormPlanError> {
        if spec.dtype() != DType::F16 {
            return Err(RmsNormPlanError::UnsupportedDType {
                expected: DType::F16,
                actual: spec.dtype(),
            });
        }
        let config = LaunchConfig1D::new(checked_rows(spec)?, BLOCK_THREADS, 0);
        let launch = if spec.hidden_size().is_multiple_of(2) {
            RmsNormF16Launch::Packed2(self.module.prepare_rms_norm_f16_packed2(config)?)
        } else {
            RmsNormF16Launch::Scalar(self.module.prepare_rms_norm_f16_scalar(config)?)
        };

        Ok(RmsNormF16Plan {
            spec,
            module: self.module.clone(),
            launch,
        })
    }

    /// Creates an immutable BF16 launch plan.
    pub fn plan_bf16(&self, spec: RmsNormSpec) -> Result<RmsNormBf16Plan, RmsNormPlanError> {
        if spec.dtype() != DType::Bf16 {
            return Err(RmsNormPlanError::UnsupportedDType {
                expected: DType::Bf16,
                actual: spec.dtype(),
            });
        }
        let config = LaunchConfig1D::new(checked_rows(spec)?, BLOCK_THREADS, 0);
        let launch = if spec.hidden_size().is_multiple_of(2) {
            RmsNormBf16Launch::Packed2(self.module.prepare_rms_norm_bf16_packed2(config)?)
        } else {
            RmsNormBf16Launch::Scalar(self.module.prepare_rms_norm_bf16_scalar(config)?)
        };

        Ok(RmsNormBf16Plan {
            spec,
            module: self.module.clone(),
            launch,
        })
    }
}

/// An immutable, prepared F32 RMSNorm launch.
#[derive(Clone)]
pub struct RmsNormF32Plan {
    spec: RmsNormSpec,
    module: kernels::LoadedModule,
    launch: PreparedLaunch<kernels::__rms_norm_f32_CudaKernel>,
}

impl RmsNormF32Plan {
    pub const fn spec(&self) -> RmsNormSpec {
        self.spec
    }

    /// Enqueues this prepared launch into a checked command scope.
    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_, '_>,
        args: RmsNormArgs<f32>,
    ) -> Result<(), RmsNormEnqueueError> {
        scope.ensure_launch_capacity()?;
        let launch_result = {
            let resolved = scope.resolve_triplet(args.input, args.weight, args.output)?;
            self.module.rms_norm_f32(
                resolved.stream,
                &self.launch,
                self.spec.rows(),
                self.spec.hidden_size(),
                self.spec.epsilon(),
                resolved.input,
                resolved.weight,
                resolved.output,
            )
        };
        record_launch(scope, self.launch.function().clone(), launch_result)
    }
}

/// The low-precision kernel selected when a plan is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmsNormKernelPath {
    /// One 16-bit element per load and store. Used for odd hidden sizes.
    Scalar,
    /// Two adjacent 16-bit elements per 32-bit load and store.
    Packed2,
}

#[derive(Clone)]
enum RmsNormF16Launch {
    Scalar(PreparedLaunch<kernels::__rms_norm_f16_scalar_CudaKernel>),
    Packed2(PreparedLaunch<kernels::__rms_norm_f16_packed2_CudaKernel>),
}

/// An immutable, prepared FP16 RMSNorm launch.
#[derive(Clone)]
pub struct RmsNormF16Plan {
    spec: RmsNormSpec,
    module: kernels::LoadedModule,
    launch: RmsNormF16Launch,
}

impl RmsNormF16Plan {
    pub const fn spec(&self) -> RmsNormSpec {
        self.spec
    }

    pub const fn kernel_path(&self) -> RmsNormKernelPath {
        match self.launch {
            RmsNormF16Launch::Scalar(_) => RmsNormKernelPath::Scalar,
            RmsNormF16Launch::Packed2(_) => RmsNormKernelPath::Packed2,
        }
    }

    /// Enqueues this prepared launch into a checked command scope.
    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_, '_>,
        args: RmsNormArgs<f16>,
    ) -> Result<(), RmsNormEnqueueError> {
        scope.ensure_launch_capacity()?;
        let (function, launch_result) = match &self.launch {
            RmsNormF16Launch::Scalar(launch) => {
                let result = {
                    let resolved = scope.resolve_triplet(args.input, args.weight, args.output)?;
                    self.module.rms_norm_f16_scalar(
                        resolved.stream,
                        launch,
                        self.spec.rows(),
                        self.spec.hidden_size(),
                        self.spec.epsilon(),
                        resolved.input,
                        resolved.weight,
                        resolved.output,
                    )
                };
                (launch.function().clone(), result)
            }
            RmsNormF16Launch::Packed2(launch) => {
                let result = {
                    let resolved = scope.resolve_triplet(args.input, args.weight, args.output)?;
                    require_packed_alignment("input", resolved.input.cu_deviceptr())?;
                    require_packed_alignment("weight", resolved.weight.cu_deviceptr())?;
                    require_packed_alignment("output", resolved.output.cu_deviceptr())?;
                    self.module.rms_norm_f16_packed2(
                        resolved.stream,
                        launch,
                        self.spec.rows(),
                        self.spec.hidden_size(),
                        self.spec.epsilon(),
                        resolved.input,
                        resolved.weight,
                        resolved.output,
                    )
                };
                (launch.function().clone(), result)
            }
        };
        record_launch(scope, function, launch_result)
    }
}

#[derive(Clone)]
enum RmsNormBf16Launch {
    Scalar(PreparedLaunch<kernels::__rms_norm_bf16_scalar_CudaKernel>),
    Packed2(PreparedLaunch<kernels::__rms_norm_bf16_packed2_CudaKernel>),
}

/// An immutable, prepared BF16 RMSNorm launch.
#[derive(Clone)]
pub struct RmsNormBf16Plan {
    spec: RmsNormSpec,
    module: kernels::LoadedModule,
    launch: RmsNormBf16Launch,
}

impl RmsNormBf16Plan {
    pub const fn spec(&self) -> RmsNormSpec {
        self.spec
    }

    pub const fn kernel_path(&self) -> RmsNormKernelPath {
        match self.launch {
            RmsNormBf16Launch::Scalar(_) => RmsNormKernelPath::Scalar,
            RmsNormBf16Launch::Packed2(_) => RmsNormKernelPath::Packed2,
        }
    }

    /// Enqueues this prepared launch into a checked command scope.
    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_, '_>,
        args: RmsNormArgs<bf16>,
    ) -> Result<(), RmsNormEnqueueError> {
        scope.ensure_launch_capacity()?;
        let (function, launch_result) = match &self.launch {
            RmsNormBf16Launch::Scalar(launch) => {
                let result = {
                    let resolved = scope.resolve_triplet(args.input, args.weight, args.output)?;
                    self.module.rms_norm_bf16_scalar(
                        resolved.stream,
                        launch,
                        self.spec.rows(),
                        self.spec.hidden_size(),
                        self.spec.epsilon(),
                        resolved.input,
                        resolved.weight,
                        resolved.output,
                    )
                };
                (launch.function().clone(), result)
            }
            RmsNormBf16Launch::Packed2(launch) => {
                let result = {
                    let resolved = scope.resolve_triplet(args.input, args.weight, args.output)?;
                    require_packed_alignment("input", resolved.input.cu_deviceptr())?;
                    require_packed_alignment("weight", resolved.weight.cu_deviceptr())?;
                    require_packed_alignment("output", resolved.output.cu_deviceptr())?;
                    self.module.rms_norm_bf16_packed2(
                        resolved.stream,
                        launch,
                        self.spec.rows(),
                        self.spec.hidden_size(),
                        self.spec.epsilon(),
                        resolved.input,
                        resolved.weight,
                        resolved.output,
                    )
                };
                (launch.function().clone(), result)
            }
        };
        record_launch(scope, function, launch_result)
    }
}

/// Checked resource handles for one RMSNorm launch.
#[derive(Clone, Copy, Debug)]
pub struct RmsNormArgs<T: BindingElement> {
    input: Read<T>,
    weight: Read<T>,
    output: Write<T>,
}

impl<T: BindingElement> RmsNormArgs<T> {
    pub const fn new(input: Read<T>, weight: Read<T>, output: Write<T>) -> Self {
        Self {
            input,
            weight,
            output,
        }
    }
}

#[derive(Debug, Error)]
pub enum RmsNormPlanError {
    #[error("{expected:?} RMSNorm plan cannot accept {actual:?}")]
    UnsupportedDType { expected: DType, actual: DType },
    #[error("RMSNorm row count {0} exceeds the CUDA grid range")]
    RowCountOutOfRange(usize),
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
}

#[derive(Debug, Error)]
pub enum RmsNormEnqueueError {
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
    #[error("packed RMSNorm requires {operand} to be {alignment}-byte aligned, got {address:#x}")]
    MisalignedBuffer {
        operand: &'static str,
        address: u64,
        alignment: u64,
    },
}

fn checked_rows(spec: RmsNormSpec) -> Result<u32, RmsNormPlanError> {
    u32::try_from(spec.rows()).map_err(|_| RmsNormPlanError::RowCountOutOfRange(spec.rows()))
}

fn require_packed_alignment(
    operand: &'static str,
    address: u64,
) -> Result<(), RmsNormEnqueueError> {
    const ALIGNMENT: u64 = size_of::<u32>() as u64;
    if address.is_multiple_of(ALIGNMENT) {
        Ok(())
    } else {
        Err(RmsNormEnqueueError::MisalignedBuffer {
            operand,
            address,
            alignment: ALIGNMENT,
        })
    }
}

fn record_launch(
    scope: &mut CommandScope<'_, '_>,
    function: CudaFunction,
    result: Result<(), LaunchContractError>,
) -> Result<(), RmsNormEnqueueError> {
    match result {
        Ok(()) => {
            scope.retain_launch(function);
            Ok(())
        }
        Err(error) => {
            if let LaunchContractError::Driver(driver_error) = &error {
                scope.retain_failed_launch(function, *driver_error);
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
        assert!(require_packed_alignment("input", 0x1000).is_ok());
    }

    #[test]
    fn packed_alignment_gate_rejects_two_byte_offsets() {
        let error = require_packed_alignment("weight", 0x1002).unwrap_err();
        assert!(matches!(
            error,
            RmsNormEnqueueError::MisalignedBuffer {
                operand: "weight",
                address: 0x1002,
                alignment: 4,
            }
        ));
    }
}
