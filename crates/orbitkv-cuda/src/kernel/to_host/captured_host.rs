//! Native HostOp child graphs and the allocations retained by each capture.

use super::*;

pub(super) struct CompiledCapturedHost {
    pub(super) node: NodeIndex,
    pub(super) inputs: Vec<NodeIndex>,
    pub(super) host_op: Arc<Box<dyn HostOp>>,
    pub(super) child_graph: Option<CudaGraphHandle>,
    pub(super) capture_resources: Vec<CudaGraphCaptureResource>,
    pub(super) graph_node: Option<CUgraphNode>,
}

// Field order retires the source child before the allocations it captured.
pub(super) struct PreparedHostCapture {
    pub(super) graph: CudaGraphHandle,
    pub(super) resources: Vec<CudaGraphCaptureResource>,
}

impl CompiledCapturedHost {
    pub(super) fn captured_pointer_nodes(&self) -> Vec<NodeIndex> {
        let host = self.host_op.as_ref().as_ref();
        let inputs: Box<dyn Iterator<Item = NodeIndex> + '_> =
            match host.cuda_graph_capture_pointer_inputs() {
                Some(indices) => Box::new(indices.iter().map(|&index| self.inputs[index])),
                None => Box::new(self.inputs.iter().copied()),
            };
        std::iter::once(self.node).chain(inputs).collect()
    }

    pub(super) fn prepare_graph_capture(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let host = self.host_op.as_ref().as_ref();
        stream.context().check_err().map_err(|error| {
            anyhow::anyhow!("deferred CUDA error before HostOp capture preparation: {error:?}")
        })?;
        host.prepare_cuda_graph_capture(stream, self.node, &self.inputs, buffers, dyn_map)?;
        stream.context().check_err().map_err(|error| {
            anyhow::anyhow!("deferred CUDA error after HostOp capture preparation: {error:?}")
        })
    }
}
