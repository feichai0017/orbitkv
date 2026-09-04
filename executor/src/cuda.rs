//! Direct Luminal graph boundary for OrbitKV-managed paged attention.

use std::collections::{BTreeMap, BTreeSet};

use luminal::{
    dtype::DType,
    prelude::{Expression, Graph, GraphTensor},
};
use luminal_cuda_lite::{
    cudarc::driver::CudaEvent,
    host::flashinfer::{PagedAttentionPlan, PagedAttentionSpec, paged_attention_with_plan},
    runtime::{CudaRuntime, DeviceCopyError, DeviceCopyPlan, DeviceCopyRange},
};
use orbitkv::EngineRelocationExecutionEvidence;
use thiserror::Error;

use crate::{
    AttentionBatch, AttentionClass, AttentionVisibility, ExecutorError, RelocationBatch,
    relocation::RelocationByteRange,
};

/// Kernel geometry shared by every layer in one compiled attention class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionKernel {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub softmax_scale: f64,
}

/// Tensors and dynamic dimensions consumed by one paged-attention node.
#[derive(Clone, Copy)]
pub struct PagedAttentionInputs {
    pub q: GraphTensor,
    pub k_cache: GraphTensor,
    pub v_cache: GraphTensor,
    pub query_tokens: Expression,
    pub context_pages: Expression,
}

/// Graph inputs that carry one OrbitKV-authored CSR page plan.
#[derive(Clone, Copy)]
pub struct PagedAttentionMetadata {
    class_id: u16,
    pub page_indices: GraphTensor,
    pub query_indptr: GraphTensor,
    pub page_indptr: GraphTensor,
    pub last_page_len: GraphTensor,
}

/// Persistent K/V tensors backing one compiled attention layer.
#[derive(Clone, Copy)]
pub struct KvCacheBinding {
    pub class_id: u16,
    pub layer: u32,
    pub key: GraphTensor,
    pub value: GraphTensor,
}

#[derive(Debug, Error)]
pub enum CudaRelocationError {
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error("relocation cache bindings do not exactly cover the requested classes and layers")]
    InvalidBindings,
    #[error(transparent)]
    Device(#[from] DeviceCopyError),
}

impl CudaRelocationError {
    /// Whether some relocation copy may already have reached the device.
    #[must_use]
    pub const fn may_have_enqueued_work(&self) -> bool {
        match self {
            Self::Device(error) => error.may_have_enqueued_work(),
            Self::Executor(_) | Self::InvalidBindings => false,
        }
    }
}

/// Device event gating canonical relocation submission.
pub struct PendingRelocationCopy {
    event: CudaEvent,
    batch: RelocationBatch,
}

impl PendingRelocationCopy {
    /// Waits for every K/V copy before exposing success evidence.
    ///
    /// # Errors
    ///
    /// Returns the CUDA driver error without manufacturing success evidence.
    pub fn wait(self) -> Result<EngineRelocationExecutionEvidence, CudaRelocationError> {
        self.event
            .synchronize()
            .map_err(DeviceCopyError::from)
            .map_err(CudaRelocationError::from)?;
        Ok(self.batch.execution_evidence_after_success())
    }
}

impl RelocationBatch {
    /// Enqueues every manager-selected token move for all affected K/V layers.
    ///
    /// Structural errors are rejected before any copy is enqueued. A returned
    /// device error may be ambiguous and should quarantine the core operation.
    ///
    /// # Errors
    ///
    /// Rejects missing, duplicate, or unexpected cache bindings, byte-range
    /// overflow, unresolved runtime tensors, and CUDA submission failures.
    pub fn enqueue(
        &self,
        runtime: &CudaRuntime,
        bindings: &[KvCacheBinding],
    ) -> Result<PendingRelocationCopy, CudaRelocationError> {
        let required = self
            .requests()
            .iter()
            .filter(|request| !request.copies.is_empty())
            .flat_map(|request| {
                request
                    .layers
                    .iter()
                    .map(move |&layer| (request.class_id, layer))
            })
            .collect::<BTreeSet<_>>();
        let mut provided = BTreeMap::new();
        let mut tensor_ids = BTreeSet::new();
        let mut graph_ref = None;
        for binding in bindings {
            if binding.key.id == binding.value.id
                || binding.key.graph_ref != binding.value.graph_ref
                || graph_ref
                    .replace(binding.key.graph_ref)
                    .is_some_and(|expected| expected != binding.key.graph_ref)
                || !tensor_ids.insert(binding.key.id)
                || !tensor_ids.insert(binding.value.id)
                || provided
                    .insert((binding.class_id, binding.layer), *binding)
                    .is_some()
            {
                return Err(CudaRelocationError::InvalidBindings);
            }
        }
        if provided.keys().copied().collect::<BTreeSet<_>>() != required {
            return Err(CudaRelocationError::InvalidBindings);
        }
        let mut plans = BTreeMap::new();
        for request in self.requests() {
            if request.copies.is_empty() {
                continue;
            }
            let key_ranges = request.key_ranges()?;
            let value_ranges = request.value_ranges()?;
            for &layer in &request.layers {
                let binding = provided
                    .get(&(request.class_id, layer))
                    .ok_or(CudaRelocationError::InvalidBindings)?;
                extend_copy_plan(&mut plans, &binding.key, &key_ranges);
                extend_copy_plan(&mut plans, &binding.value, &value_ranges);
            }
        }
        let plans = plans
            .into_iter()
            .map(|(tensor, ranges)| DeviceCopyPlan {
                tensor,
                ranges: ranges.into_boxed_slice(),
            })
            .collect::<Vec<_>>();
        let event = runtime.copy_output_ranges(&plans)?;
        Ok(PendingRelocationCopy {
            event,
            batch: self.clone(),
        })
    }
}

fn extend_copy_plan(
    plans: &mut BTreeMap<luminal::prelude::NodeIndex, Vec<DeviceCopyRange>>,
    tensor: &GraphTensor,
    ranges: &[RelocationByteRange],
) {
    let target = plans.entry(tensor.id).or_default();
    target.extend(ranges.iter().map(|range| DeviceCopyRange {
        source_offset: range.source_offset,
        destination_offset: range.destination_offset,
        bytes: range.bytes,
    }));
}

impl PagedAttentionMetadata {
    /// Creates named graph inputs for one attention class.
    #[must_use]
    pub fn new(
        graph: &mut Graph,
        class_id: u16,
        request_count: Expression,
        context_pages: Expression,
    ) -> Self {
        let rows = request_count + 1;
        Self {
            class_id,
            page_indices: graph
                .named_tensor(format!("attention.{class_id}.page_indices"), context_pages)
                .as_dtype(DType::Int),
            query_indptr: graph
                .named_tensor(format!("attention.{class_id}.query_indptr"), rows)
                .as_dtype(DType::Int),
            page_indptr: graph
                .named_tensor(format!("attention.{class_id}.page_indptr"), rows)
                .as_dtype(DType::Int),
            last_page_len: graph
                .named_tensor(format!("attention.{class_id}.last_page_len"), request_count)
                .as_dtype(DType::Int),
        }
    }

    /// Uploads one validated host plan into the graph inputs.
    ///
    /// # Errors
    ///
    /// Rejects a plan for another class or malformed CSR row cardinality.
    pub fn upload(
        self,
        runtime: &mut CudaRuntime,
        batch: &AttentionBatch,
    ) -> Result<(), ExecutorError> {
        let rows = batch
            .query_indptr
            .len()
            .checked_sub(1)
            .ok_or(ExecutorError::InvalidRequestGeometry)?;
        if batch.class_id != self.class_id
            || batch.page_indptr.len() != rows + 1
            || batch.last_page_len.len() != rows
            || batch.page_indptr.last().copied() != i32::try_from(batch.page_indices.len()).ok()
        {
            return Err(ExecutorError::InvalidRequestGeometry);
        }
        runtime.set_data(self.page_indices, batch.page_indices.to_vec());
        runtime.set_data(self.query_indptr, batch.query_indptr.to_vec());
        runtime.set_data(self.page_indptr, batch.page_indptr.to_vec());
        runtime.set_data(self.last_page_len, batch.last_page_len.to_vec());
        Ok(())
    }
}

/// Adds one fused paged-attention node that consumes `OrbitKV` metadata.
///
/// The K/V buffers remain Luminal tensors, but their page identities and
/// lifetime are controlled by `RuntimeSession`.
///
/// # Errors
///
/// Rejects invalid head geometry, dtype, page size, graph ownership, or a
/// metadata input belonging to another attention class.
pub fn paged_attention(
    inputs: PagedAttentionInputs,
    metadata: PagedAttentionMetadata,
    class: &AttentionClass,
    kernel: AttentionKernel,
) -> Result<GraphTensor, ExecutorError> {
    if metadata.class_id != class.class_id
        || class.page_tokens == 0
        || kernel.query_heads == 0
        || kernel.kv_heads == 0
        || kernel.head_dim == 0
        || !kernel.query_heads.is_multiple_of(kernel.kv_heads)
        || !matches!(kernel.dtype, DType::F16 | DType::Bf16)
        || inputs.q.dtype != kernel.dtype
        || inputs.k_cache.dtype != kernel.dtype
        || inputs.v_cache.dtype != kernel.dtype
        || [
            inputs.k_cache,
            inputs.v_cache,
            metadata.page_indices,
            metadata.query_indptr,
            metadata.page_indptr,
            metadata.last_page_len,
        ]
        .iter()
        .any(|input| input.graph_ref != inputs.q.graph_ref)
    {
        return Err(ExecutorError::InvalidKernelGeometry);
    }
    let window_left = match class.visibility {
        AttentionVisibility::Full | AttentionVisibility::Chunked { .. } => None,
        AttentionVisibility::Sliding { window_tokens } => Some(
            usize::try_from(window_tokens.saturating_sub(1))
                .map_err(|_| ExecutorError::InvalidKernelGeometry)?,
        ),
    };
    Ok(paged_attention_with_plan(
        PagedAttentionPlan {
            query: inputs.q,
            key_cache: inputs.k_cache,
            value_cache: inputs.v_cache,
            page_indices: metadata.page_indices,
            query_indptr: metadata.query_indptr,
            page_indptr: metadata.page_indptr,
            last_page_len: metadata.last_page_len,
        },
        PagedAttentionSpec {
            num_qo_heads: kernel.query_heads,
            num_kv_heads: kernel.kv_heads,
            head_dim: kernel.head_dim,
            page_size: class.page_tokens as usize,
            query_tokens: inputs.query_tokens,
            context_pages: inputs.context_pages,
            dtype: kernel.dtype,
            sm_scale: kernel.softmax_scale,
            window_left,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_external_page_plan_node() {
        let mut graph = Graph::default();
        let query_tokens = Expression::from('s');
        let context_pages = Expression::from('c');
        let q = graph
            .named_tensor("q", (4, query_tokens, 64))
            .as_dtype(DType::Bf16);
        let k = graph
            .named_tensor("k", (8, 16, 1, 64))
            .as_dtype(DType::Bf16);
        let v = graph
            .named_tensor("v", (8, 16, 1, 64))
            .as_dtype(DType::Bf16);
        let metadata = PagedAttentionMetadata::new(&mut graph, 0, 2.into(), context_pages);
        let output = paged_attention(
            PagedAttentionInputs {
                q,
                k_cache: k,
                v_cache: v,
                query_tokens,
                context_pages,
            },
            metadata,
            &AttentionClass {
                class_id: 0,
                name: "attention".into(),
                layers: vec![0].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 128,
                value_bytes_per_token_per_layer: 128,
                token_relocatable: true,
                visibility: AttentionVisibility::Sliding { window_tokens: 64 },
            },
            AttentionKernel {
                query_heads: 4,
                kv_heads: 1,
                head_dim: 64,
                dtype: DType::Bf16,
                softmax_scale: 0.0,
            },
        )
        .unwrap();
        assert_eq!(output.dims(), &[4.into(), query_tokens, 64.into()]);
        assert_eq!(graph.get_sources(output.id).len(), 7);
    }
}
