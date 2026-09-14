use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::prelude::*;

pub mod argmax;
pub mod conv2d;
pub mod cuda_graph;
pub mod fusion;
pub mod gemv;
pub mod generic_matmul;
pub mod hlir;
pub mod matmul2d;
pub mod other_ops;
pub mod quant_f8;
pub mod recurrent_state;
pub mod rms_norm;
pub mod rope;
pub mod sequence_state;
pub mod swiglu;
pub mod topk;

pub use conv2d::KernelConv2D;
pub use cuda_graph::*;
pub use generic_matmul::GenericMatmul;
pub use matmul2d::{
    Matmul2DCustom, Matmul2DKernel, linear_bias, linear_no_bias_bf16_w, matmul_2d, matmul_2d_t,
    matmul_3d, matmul_3d_t,
};
pub use rms_norm::{RMSNormCustom, RMSNormKernel, fused_rms_norm};
pub use rope::{RoPECustom, RoPEKernel, apply_rope};
pub use sequence_state::{
    PackedConvolutionOutput, PackedConvolutionPlan, PackedConvolutionSpec, PackedDeltaScanOutput,
    PackedDeltaScanPlan, PackedDeltaScanSpec, packed_causal_convolution, packed_delta_scan,
};

pub type Ops = (
    hlir::Ops,
    argmax::KernelArgmax,
    gemv::KernelGemv,
    rms_norm::KernelRMSNorm,
    rope::RoPEHalfKernel,
    rope::RoPEScatterKernel,
    rope::KernelRoPE,
    swiglu::KernelSwiglu,
    topk::KernelStableSortIdx,
    quant_f8::KernelQuantF8,
    recurrent_state::KernelDeltaStateUpdate,
    sequence_state::SequenceStateRules,
    other_ops::Ops,
    conv2d::KernelConv2D,
    GenericMatmul,
    fusion::Ops,
);

pub trait KernelOp: std::fmt::Debug + as_any::AsAny {
    /// Compile the kernel and return its function/module, exact generated CUDA
    /// source, launch expressions, dynamic shared memory, and constants. The
    /// source string is part of the runtime contract: it is used to detect the
    /// optional `dyn_dims` parameter and enforce compilation-resource budgets.
    #[allow(clippy::type_complexity)]
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    );

    /// Returns the output buffer size in elements.
    fn output_size(&self) -> Expression;

    /// Adds all dynamic variables used by this kernel (for grid dims, strides,
    /// etc.) to an existing set. Reusing the set is important during search,
    /// where graph materialization visits millions of kernels. Override if the
    /// kernel has dynamic variables in expressions not captured by
    /// `output_size` (for example, KernelScatter's index_shape).
    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.output_size().collect_dyn_vars_into(vars);
    }

    /// Convenience wrapper for individual kernel compilation paths.
    fn all_dyn_vars(&self) -> FxHashSet<Symbol> {
        let mut vars = FxHashSet::default();
        self.collect_dyn_vars_into(&mut vars);
        vars
    }

    /// Returns the output buffer size in bytes (accounts for dtype).
    fn output_bytes(&self) -> Expression;

    /// Returns the DType of this kernel's output buffer.
    /// Default: F32 (most kernels output float).
    fn output_dtype(&self) -> DType {
        DType::F32
    }

    /// Returns the number of bytes this kernel will load from global memory.
    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    /// Returns the number of bytes this kernel will store to global memory.
    fn bytes_stored(&self) -> Expression {
        0.into()
    }

    /// Returns the number of floating point operations this kernel performs.
    fn flops(&self) -> Expression {
        0.into()
    }

    /// Returns the name of this kernel for profiling display.
    fn kernel_name(&self) -> &'static str {
        "Unknown"
    }

    /// Allocate internal buffers this kernel needs. Called once during graph building.
    /// Default: no internal buffers.
    fn allocate_internal_buffers(
        &self,
        _stream: &Arc<CudaStream>,
        _dyn_map: &DynMap,
    ) -> Vec<CudaSlice<u8>> {
        vec![]
    }

    /// Returns the set of dynamic dimensions that affect internal buffer sizes.
    /// When any of these dimensions change, internal buffers should be reallocated.
    /// Default: empty set (no dimensions affect internal buffers).
    fn internal_buffer_dyn_dims(&self) -> FxHashSet<Symbol> {
        FxHashSet::default()
    }

    /// Build kernel parameters. Returns the u64 values to pass to the kernel.
    /// Default: [output_ptr, input_ptrs..., dyn_dims_ptr (if non-zero)]
    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        let mut params = vec![output_ptr];
        params.extend_from_slice(input_ptrs);
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    /// Number of 64-bit launch parameters produced by `build_params` for a
    /// kernel with `input_count` graph inputs. Resource preflight uses this to
    /// enforce CUDA's kernel-parameter ABI limit without allocating buffers.
    ///
    /// The default matches the standard parameter contract: one output pointer
    /// unless output aliases an input, one pointer per input, and an optional
    /// dynamic-dimension pointer. An override of `build_params` that changes
    /// that count must override this method as well.
    fn kernel_parameter_count(&self, input_count: usize, has_dyn_dims_param: bool) -> usize {
        input_count
            + usize::from(self.output_aliases_input().is_none())
            + usize::from(has_dyn_dims_param)
    }

    /// Called before each kernel execution. Update internal state if needed.
    /// `all_buffer_ptrs` contains pointers for all buffers this kernel might use.
    /// `constants` are device constants returned by compile() that may need updating.
    fn pre_execute(
        &self,
        _stream: &Arc<CudaStream>,
        _internal_bufs: &mut [CudaSlice<u8>],
        _constants: &mut FxHashMap<Symbol, CudaSlice<u8>>,
        _all_buffer_ptrs: &FxHashMap<NodeIndex, u64>,
        _dyn_map: &DynMap,
    ) {
    }

    /// If this kernel's output aliases one of its inputs (i.e., writes in-place),
    /// return the input index. Used to propagate buffer pointers in CUDA graphs.
    /// Pure aliases must also emit the `cuda-scatter-alias` egglog fact (see
    /// FusionStart), so late scatter reuse cannot mistake a view for owned
    /// storage. Mutating specializations must inherit an exclusive-use proof.
    fn output_aliases_input(&self) -> Option<usize> {
        None
    }

    /// Whether aliasing the output also mutates the aliased input buffer.
    /// Aliases are conservatively treated as mutations by default so a new
    /// in-place kernel cannot silently bypass candidate-local hazard checks.
    /// Proven identity/view markers such as FusionStart override this to false.
    fn mutates_aliased_input(&self) -> bool {
        self.output_aliases_input().is_some()
    }

    /// If this kernel's output is derived from one of its inputs (copy-then-modify
    /// or in-place write), return that input index. Used by `resolve_data_node` to
    /// trace buffer ownership back to HLIR inputs for the remove_buffer/set_buffer
    /// roundtrip pattern.
    ///
    /// Defaults to `output_aliases_input()`. Override for copy-then-modify ops
    /// (like Scatter which copies dest→output then scatters into it).
    fn output_data_input(&self) -> Option<usize> {
        self.output_aliases_input()
    }

    /// Returns indices of internal buffers containing timing data, if any.
    /// Returns (timings_idx, start_times_idx, sm_count).
    fn timing_buffer_indices(&self) -> Option<(usize, usize, usize)> {
        None
    }
}

orbitkv_compiler::impl_into_ops!(KernelOp);

// Kernel to host op compilation
mod to_host;
pub(crate) use to_host::{
    CompiledFunctionResourceCache, CudaGraphArenaOrdering, PreparedKernelToHostPlan,
    kernel_to_host_with_prepared, prepare_kernel_to_host_plan,
    prepare_kernel_to_host_plan_with_topo_and_source_cache,
};
pub use to_host::{CudaGraphDebugSummary, CudaGraphOp, kernel_to_host};
