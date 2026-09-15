//! Captured FlashInfer plans and content-dependent metadata validity.

use super::*;
use crate::providers::flashinfer::FlashInferMetadataCache;

impl CompiledFlashInferDecode {
    pub(super) fn new(
        node: NodeIndex,
        inputs: Vec<NodeIndex>,
        host_op: Arc<Box<dyn HostOp>>,
    ) -> Self {
        Self {
            node,
            inputs,
            host_op,
            entry_node: None,
            exit_node: None,
            captured_nodes: Vec::new(),
            prepared: None,
            ptrs: None,
            signature: None,
            recapture_count: 0,
        }
    }

    pub(super) fn flashinfer(&self) -> &FlashInferAttention {
        self.host_op
            .as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<FlashInferAttention>()
            .expect("CompiledFlashInferDecode only stores FlashInfer host ops")
    }

    pub(super) fn has_runtime_metadata(&self) -> bool {
        self.ptrs
            .is_some_and(|pointers| pointers.explicit_kv_indptr.is_some())
    }

    pub(super) fn enqueue_prepared(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("FlashInfer step is not prepared"))?;
        let resolved =
            self.flashinfer()
                .resolve_for_graph(self.node, &self.inputs, buffers, dyn_map)?;
        let signature = resolved.signature_for_graph_plan(prepared.plan_c());
        anyhow::ensure!(
            self.signature
                .as_ref()
                .is_some_and(|old| old.spec == signature.spec),
            "FlashInfer shape changed after warmup"
        );
        prepared.enqueue(stream, signature.ptrs, true)
    }
}

pub(super) fn changed_metadata(
    state: &CudaGraphOpState,
    stream: &Arc<CudaStream>,
    bindings_changed: bool,
) -> anyhow::Result<FxHashSet<usize>> {
    let mut changed = FxHashSet::default();
    let mut metadata = FlashInferMetadataCache::default();
    for (index, op) in state.flashinfer_ops.iter().enumerate() {
        let Some(pointers) = op.ptrs.filter(|ptrs| ptrs.explicit_kv_indptr.is_some()) else {
            continue;
        };
        // Changed dimensions/bindings already invalidate the plan. Only read
        // its recorded row count while that validated geometry remains current.
        if bindings_changed
            || !op.prepared.as_ref().unwrap().graph_metadata_matches(
                stream,
                pointers,
                &mut metadata,
            )?
        {
            changed.insert(index);
        }
    }
    Ok(changed)
}
