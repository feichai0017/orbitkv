//! Buffer, resource and capture contracts for host-orchestrated GPU operations.

use crate::cudarc::driver::{CudaStream, DriverError, result};
use crate::resource::{HostDeviceMemoryPlan, ResourceViolation};
use orbitkv_compiler::{op::EgglogOp, prelude::*};
use std::{fmt::Debug, sync::Arc};

/// Allocation owners retained by the exact graph that captured them.
pub type CudaGraphCaptureResource = Arc<dyn std::any::Any + Send + Sync>;

/// Non-owning device buffer handle used by host operations.
///
/// Runtime-owned intermediates may be a whole `CudaSlice`, a subregion inside
/// the reusable arena, or an external pointer. Host ops only need the pointer
/// and the logical byte length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    ptr: u64,
    /// Logical bytes belonging to the current dynamic shape.
    len: usize,
    /// Physically writable bytes at `ptr`. Arena-backed buffers can retain a
    /// larger bucket/high-water allocation after their logical shape shrinks.
    capacity: usize,
    host_ptr: u64,
    host_len: usize,
}

impl DeviceBuffer {
    pub fn new(ptr: u64, len: usize) -> Self {
        Self {
            ptr,
            len,
            capacity: len,
            host_ptr: 0,
            host_len: 0,
        }
    }

    /// Attach an authoritative host mirror for the duration of one HostOp
    /// call. The runtime owns the bytes and guarantees they outlive the
    /// temporary buffer map passed to `execute`.
    pub(crate) fn with_host_bytes(mut self, bytes: &[u8]) -> Self {
        self.host_ptr = bytes.as_ptr() as u64;
        self.host_len = bytes.len();
        self
    }

    pub fn ptr(self) -> u64 {
        self.ptr
    }

    pub fn len(self) -> usize {
        self.len
    }

    pub fn capacity(self) -> usize {
        self.capacity
    }

    pub(crate) fn with_capacity(mut self, capacity: usize) -> Self {
        assert!(capacity >= self.len);
        self.capacity = capacity;
        self
    }

    pub(crate) fn with_logical_len(mut self, len: usize) -> Self {
        assert!(len <= self.capacity);
        self.len = len;
        self
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Host-side contents supplied by an opt-in mirrored input binding.
    pub fn host_bytes(&self) -> Option<&[u8]> {
        (self.host_ptr != 0).then(|| unsafe {
            // SAFETY: only the runtime constructs mirrored DeviceBuffers, and
            // its owning Vec remains borrowed for the complete HostOp call.
            std::slice::from_raw_parts(self.host_ptr as *const u8, self.host_len)
        })
    }

    pub fn clone_dtoh(self, stream: &Arc<CudaStream>) -> Result<Vec<u8>, DriverError> {
        let mut host = vec![0u8; self.len];
        unsafe {
            result::memcpy_dtoh_async(&mut host, self.ptr, stream.cu_stream())?;
        }
        stream.synchronize()?;
        Ok(host)
    }
}

/// Host operations that execute on the CPU but orchestrate GPU work.
///
/// This includes operations like cuBLAS calls and CUDA graph executions.
/// `Debug` participates in LLIR program identity: it must describe semantic
/// parameters only. Runtime handles, planner caches and allocation owners must
/// not change that representation when a provider is prepared or executed.
pub trait HostOp: Debug + as_any::AsAny + EgglogOp {
    /// Native providers used by this operation, including nested launches.
    /// Artifact provenance queries this after selection, outside timed execution.
    fn provider_dependencies(&self) -> Vec<super::registry::ProviderId> {
        Vec::new()
    }

    /// Prepare provider code for one concrete profiling shape. This runs while
    /// a candidate is being compiled, before device timing starts. JIT-backed
    /// providers must use this hook rather than charging compilation to their
    /// first profiling sample.
    fn prepare_compilation(
        &self,
        _stream: &Arc<CudaStream>,
        _dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Execute the operation with access to buffers via a map.
    ///
    /// # Arguments
    /// * `stream` - The CUDA stream to execute on
    /// * `self_node` - The NodeIndex of this op in the llir_graph (used as output buffer)
    /// * `inputs` - NodeIndices of input nodes (in edge order from the graph)
    /// * `buffers` - Map from NodeIndex to device buffer for all allocated nodes
    /// * `dyn_map` - Dynamic dimension values
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()>;

    /// Execute with an identifier shared by every host op in one runtime
    /// invocation. Operations that coordinate per-invocation preparation can
    /// override this hook; existing `HostOp` implementations remain source
    /// compatible through the default delegation to [`HostOp::execute`].
    fn execute_with_id(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        _execution_id: u64,
    ) -> anyhow::Result<()> {
        self.execute(stream, self_node, inputs, buffers, dyn_map)
    }

    /// Returns the output buffer size in elements.
    /// Return 0 if this op doesn't have a single output buffer (e.g., CudaGraphOp).
    fn output_size(&self) -> Expression;

    /// Returns the output buffer size in bytes (accounts for dtype).
    fn output_bytes(&self) -> Expression;

    /// Storage dtype of the graph-visible output. Host operations are not
    /// intrinsically F32: cuBLASLt, for example, can write BF16/F16 while
    /// accumulating in F32. Runtime buffer metadata must describe the bytes
    /// actually written so strict readback and downstream kernels agree.
    fn output_dtype(&self) -> DType {
        DType::F32
    }

    /// Number of graph inputs when this operation opts into capture as a
    /// child of a larger CUDA graph. Returning `None` keeps the ordinary
    /// standalone HostOp boundary.
    ///
    /// The graph compiler consumes this contract without depending on the
    /// concrete provider type.
    fn cuda_graph_capture_arity(&self) -> Option<usize> {
        None
    }

    /// Identify mutable device state shared by captured instances of this
    /// operation. Instances naming the same resource can share one parent
    /// graph only when their equivalence keys match; otherwise the compiler
    /// retains their ordinary standalone HostOp boundaries.
    ///
    /// This is a correctness contract, not an extraction preference. It lets
    /// superset backends describe coordinated library planner state without
    /// coupling the graph compiler to their concrete operation types.
    fn cuda_graph_capture_shared_state(
        &self,
        _inputs: &[NodeIndex],
    ) -> Option<CudaGraphCaptureSharedState> {
        None
    }

    /// Input indices whose pointer identity is baked into a captured child
    /// graph. `None` conservatively tracks every graph input. Operations may
    /// omit planner-only buffers that are not consumed by captured kernels.
    fn cuda_graph_capture_pointer_inputs(&self) -> Option<&'static [usize]> {
        None
    }

    /// Input indices whose logical shapes are baked into captured launch
    /// geometry. `None` conservatively inspects the output and every input.
    /// This is separate from pointer tracking so payload-sized buffers can
    /// grow without forcing recapture when kernels consume their live length
    /// or contents from device metadata.
    fn cuda_graph_capture_shape_inputs(&self) -> Option<&'static [usize]> {
        None
    }

    /// Dynamic dimensions baked into captured launch geometry but not
    /// necessarily recoverable from graph-visible buffer sizes.
    fn cuda_graph_capture_dyn_dims(&self) -> Vec<Symbol> {
        vec![]
    }

    /// Prepare stable allocations and metadata before child-graph capture.
    fn prepare_cuda_graph_capture(
        &self,
        _stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        _buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        _dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        anyhow::bail!("HostOp did not implement CUDA graph capture preparation")
    }

    /// Snapshot allocation owners after preparation and before capture. Each
    /// materialized child graph retains its snapshot until its executable and
    /// source graphs have retired, including inactive resident variants.
    fn cuda_graph_capture_resources(&self) -> Vec<CudaGraphCaptureResource> {
        vec![]
    }

    /// Refresh execution-specific metadata immediately before launching a
    /// captured parent graph. Most captured operations need no refresh.
    fn prepare_cuda_graph_execution(
        &self,
        _stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        _buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        _dyn_map: &DynMap,
        _execution_id: u64,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Returns additional nodes (beyond graph edges) that this op needs buffers for.
    ///
    /// For most ops, this returns empty (buffers determined by graph edges).
    /// For CudaGraphOp, this returns all internal kernel nodes.
    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        vec![]
    }

    /// Returns relative lifetimes for extra buffer nodes within this host op.
    ///
    /// The tuple is `(node, first_step, last_step)`, where steps are local to
    /// this host op's execution. Returning `None` tells the runtime to treat
    /// every extra buffer as live for the whole host op.
    fn extra_buffer_lifetimes(&self) -> Option<Vec<(NodeIndex, usize, usize)>> {
        None
    }

    /// Returns buffer size requirements for extra nodes (node -> size in elements).
    ///
    /// Called during buffer allocation to ensure all required buffers exist.
    /// For CudaGraphOp, this returns sizes for all internal kernel output buffers.
    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        FxHashMap::default()
    }

    /// Device allocations owned by, temporarily created by, or globally
    /// shared by this host operation beyond its graph-visible output and the
    /// runtime intermediate arena. Planning is pointer-free: `buffer_lengths`
    /// contains logical byte lengths only, and implementations must not
    /// allocate device memory or read device contents during this call.
    fn device_memory_plan(
        &self,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        _buffer_lengths: &FxHashMap<NodeIndex, usize>,
        _dyn_map: &DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        Ok(HostDeviceMemoryPlan::default())
    }

    /// Graph-visible buffers whose logical byte lengths can change the result
    /// of `device_memory_plan`.
    ///
    /// Pointer identity and payload contents are never resource facts. Most
    /// HostOps derive their resource requirements entirely from `dyn_map` and
    /// therefore return no nodes here. An op that consults `buffer_lengths`
    /// must return exactly those inputs so the runtime can invalidate its
    /// hard-resource validation cache when (and only when) their lengths
    /// change.
    fn resource_buffer_nodes(&self, _inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        vec![]
    }

    /// Whether the operation is a deployable provider implementation.
    /// Semantic fallbacks and correctness oracles return false so search can
    /// use them as e-class roots without installing them in a runtime.
    fn deployment_eligible(&self) -> bool {
        true
    }

    /// Returns the name of this host op for stats reporting, or None if not reportable.
    fn stats_name(&self) -> Option<&'static str> {
        None
    }
}

/// Compatibility identity for mutable state shared by captured HostOps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaGraphCaptureSharedState {
    pub resource: &'static str,
    pub equivalence_key: Vec<usize>,
}

#[cfg(test)]
#[path = "../../tests/unit/providers/mod.rs"]
mod tests;
