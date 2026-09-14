//! Host-operation contract for compiled CUDA Graphs.

use super::CudaGraphOp;
use crate::{
    providers::{DeviceBuffer, HostOp},
    resource::{HostDeviceMemoryPlan, ResourceViolation},
};
use cudarc::driver::CudaStream;
use itertools::Itertools;
use orbitkv_compiler::prelude::{DynMap, Expression, FxHashMap, NodeIndex};
use std::sync::Arc;

impl HostOp for CudaGraphOp {
    fn provider_dependencies(&self) -> Vec<crate::providers::registry::ProviderId> {
        use crate::providers::registry::ProviderId;
        let state = self.state.borrow();
        let mut dependencies = state
            .captured_host_ops
            .iter()
            .flat_map(|op| op.host_op.provider_dependencies())
            .collect::<Vec<_>>();
        if !state.cublaslt_ops.is_empty() {
            dependencies.push(ProviderId::CublasLt);
        }
        if !state.flashinfer_ops.is_empty() {
            dependencies.push(ProviderId::FlashInfer);
        }
        dependencies
    }

    fn prepare_compilation(
        &self,
        stream: &Arc<CudaStream>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        for op in &self.state.borrow().captured_host_ops {
            op.host_op.prepare_compilation(stream, dyn_map)?;
        }
        Ok(())
    }

    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        self.execute_internal(stream, buffers, dyn_map, 0)
    }

    fn execute_with_id(
        &self,
        stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        execution_id: u64,
    ) -> anyhow::Result<()> {
        self.execute_internal(stream, buffers, dyn_map, execution_id)
    }

    fn output_size(&self) -> Expression {
        // CudaGraphOp doesn't have a single output - individual kernels have outputs
        0.into()
    }

    fn output_bytes(&self) -> Expression {
        // CudaGraphOp doesn't have a single output - individual kernels have outputs
        0.into()
    }

    fn device_memory_plan(
        &self,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        self.host_device_memory_plan(buffer_lengths, dyn_map)
    }

    fn resource_buffer_nodes(&self, _inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        // CudaGraphOp absorbs HostOps, so its graph-visible `inputs` argument
        // does not describe the internal FlashInfer inputs. Preserve the
        // dependency explicitly from the compiled island metadata.
        let state = self.state.borrow();
        state
            .flashinfer_ops
            .iter()
            .flat_map(|op| op.inputs.get(1..3).unwrap_or_default().iter().copied())
            .chain(
                state
                    .captured_host_ops
                    .iter()
                    .flat_map(|op| op.host_op.resource_buffer_nodes(&op.inputs).into_iter()),
            )
            .unique()
            .collect()
    }

    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        // Only return nodes that actually have buffers
        // Filter out nodes in buffer_sizes with size 0 (like MegakernelOps)
        // Keep nodes not in buffer_sizes (external inputs that have their own buffers)
        self.buffer_nodes
            .iter()
            .filter(|n| {
                match self.buffer_sizes.get(n) {
                    Some(size) => size.exec(&FxHashMap::default()).unwrap_or(1) != 0,
                    None => true, // Not a kernel output, might be an external input
                }
            })
            .copied()
            .collect()
    }

    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        self.buffer_sizes.clone()
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("CudaGraph")
    }
}
