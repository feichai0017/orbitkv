//! Direct `OrbitKV` graph boundary for `OrbitKV`-managed paged attention.

use orbitkv_compiler::{
    dtype::DType,
    prelude::{Expression, Graph, GraphTensor},
};
use orbitkv_cuda::runtime::CudaRuntime;
use orbitkv_ops::ops::attention::{
    AttentionInputs, AttentionMask, AttentionSpec, KvView, PagedKvLayout, PagedKvView,
    attention as compile_attention,
};

use crate::{AttentionBatch, AttentionClass, AttentionVisibility, ExecutorError};

/// Mathematical geometry shared by every layer in one compiled attention class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionGeometry {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
    /// Explicit positive finite scale; automatic scaling is resolved by the frontend.
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

/// Graph inputs that carry one `OrbitKV`-authored CSR page plan.
#[derive(Clone, Copy)]
pub struct PagedAttentionMetadata {
    class_id: u16,
    pub page_indices: GraphTensor,
    pub query_indptr: GraphTensor,
    pub page_indptr: GraphTensor,
    pub last_page_len: GraphTensor,
}

/// Persistent K/V graph inputs backing one compiled attention layer.
#[derive(Clone, Copy)]
pub struct KvCacheBinding {
    pub class_id: u16,
    pub layer: u32,
    pub key: GraphTensor,
    pub value: GraphTensor,
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
        let query_indptr = graph
            .named_tensor(
                format!("attention.{class_id}.query_indptr"),
                request_count + 1,
            )
            .as_dtype(DType::Int);
        Self::with_query_indptr(graph, class_id, request_count, context_pages, query_indptr)
    }

    /// Creates per-class metadata around a shared decoder-level query
    /// segmentation tensor.
    ///
    /// # Panics
    ///
    /// Panics when the shared tensor belongs to another graph or does not have
    /// the required integer row-pointer shape.
    #[must_use]
    pub fn with_query_indptr(
        graph: &mut Graph,
        class_id: u16,
        request_count: Expression,
        context_pages: Expression,
        query_indptr: GraphTensor,
    ) -> Self {
        let rows = request_count + 1;
        assert_eq!(query_indptr.graph_ref, std::ptr::from_mut::<Graph>(graph));
        assert_eq!(query_indptr.dtype, DType::Int);
        assert_eq!(query_indptr.dims(), [rows]);
        Self {
            class_id,
            page_indices: graph
                .named_tensor(format!("attention.{class_id}.page_indices"), context_pages)
                .as_dtype(DType::Int),
            query_indptr,
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
        self.upload_class_metadata(runtime, batch)?;
        runtime.set_data(self.query_indptr, batch.query_indptr.to_vec());
        Ok(())
    }

    /// Uploads class-local page metadata when query segmentation is shared by
    /// the complete decoder graph and was uploaded once by its caller.
    ///
    /// # Errors
    ///
    /// Rejects a plan for another class or malformed CSR row cardinality.
    pub fn upload_class_metadata(
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
        runtime.set_data(self.page_indptr, batch.page_indptr.to_vec());
        runtime.set_data(self.last_page_len, batch.last_page_len.to_vec());
        Ok(())
    }
}

/// Adds logical attention with an explicit `OrbitKV` page view.
///
/// Q must be a contiguous `(query_tokens, heads, head_dim)` NHD tensor,
/// with an explicit NHD storage contract. The result stays heads-first for the surrounding
/// graph. The K/V buffers remain `OrbitKV` tensors, but their page identities
/// and lifetime are controlled by `RuntimeSession`.
///
/// # Errors
///
/// Rejects invalid head geometry, dtype, page size, graph ownership, or a
/// metadata input belonging to another attention class.
pub fn paged_attention(
    inputs: PagedAttentionInputs,
    metadata: PagedAttentionMetadata,
    class: &AttentionClass,
    geometry: AttentionGeometry,
) -> Result<GraphTensor, ExecutorError> {
    if metadata.class_id != class.class_id
        || class.page_tokens == 0
        || geometry.query_heads == 0
        || geometry.kv_heads == 0
        || geometry.head_dim == 0
        || !geometry.softmax_scale.is_finite()
        || geometry.softmax_scale <= 0.0
        || !geometry.query_heads.is_multiple_of(geometry.kv_heads)
        || !matches!(geometry.dtype, DType::F16 | DType::Bf16)
        || inputs.q.dtype != geometry.dtype
        || inputs.q.dims()
            != [
                inputs.query_tokens,
                geometry.query_heads.into(),
                geometry.head_dim.into(),
            ]
        || inputs.k_cache.dtype != geometry.dtype
        || inputs.v_cache.dtype != geometry.dtype
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
    let mask = match class.visibility {
        AttentionVisibility::Full | AttentionVisibility::Chunked { .. } => AttentionMask::Causal,
        AttentionVisibility::Sliding { window_tokens } => AttentionMask::Sliding {
            window_left: usize::try_from(window_tokens.saturating_sub(1))
                .map_err(|_| ExecutorError::InvalidKernelGeometry)?,
        },
    };
    if metadata.page_indices.dims() != [inputs.context_pages] {
        return Err(ExecutorError::InvalidKernelGeometry);
    }
    compile_attention(
        AttentionInputs {
            query: inputs.q,
            query_indptr: metadata.query_indptr,
            kv: KvView::Paged(PagedKvView {
                state_class_id: class.class_id,
                key: inputs.k_cache,
                value: inputs.v_cache,
                page_size: class.page_tokens as usize,
                layout: PagedKvLayout::TokenMajor,
                page_indices: metadata.page_indices,
                page_indptr: metadata.page_indptr,
                last_page_len: metadata.last_page_len,
            }),
        },
        AttentionSpec {
            query_heads: geometry.query_heads,
            kv_heads: geometry.kv_heads,
            query_key_dim: geometry.head_dim,
            value_dim: geometry.head_dim,
            dtype: geometry.dtype,
            scale: geometry.softmax_scale,
            mask,
        },
    )
    .map_err(|_| ExecutorError::InvalidKernelGeometry)
}

#[cfg(test)]
#[path = "../tests/unit/cuda/mod.rs"]
mod tests;
