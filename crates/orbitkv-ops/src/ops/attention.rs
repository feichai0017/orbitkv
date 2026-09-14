//! Scaled dot-product attention semantics and explicit, externally owned KV views.
//!
//! A view describes storage; it does not choose a provider or grant permission
//! to publish, mutate or retire that storage. Backends lower supported semantic
//! and view combinations through equivalence rules.

mod kv_view;
mod semantics;

pub use kv_view::{KvView, PagedKvLayout, PagedKvView};
pub use semantics::{AttentionError, AttentionMask, AttentionSpec};

use orbitkv_compiler::{
    egglog_utils::api::{SortDef, sort},
    op::{CustomOp, EgglogOp},
    prelude::{Expression, GraphTensor},
};

use kv_view::KvViewSpec;

/// Packed queries and the KV history visible to them.
#[derive(Clone, Copy)]
pub struct AttentionInputs {
    /// Contiguous [query_tokens, query_heads, query_key_dim] values.
    pub query: GraphTensor,
    /// CSR segmentation of queries. Each active request owns at least one query.
    /// Causal queries are the suffix of that request's visible KV history.
    pub query_indptr: GraphTensor,
    pub kv: KvView,
}

#[derive(Clone, Copy, Debug)]
struct Attention {
    spec: AttentionSpec,
    query_tokens: Expression,
    requests: Expression,
    kv: KvViewSpec,
}

/// Logical computation and physical storage are separate compiler facts.
pub const ATTENTION_DECLARATIONS: &str = "
(datatype AttentionMask (CausalAttention) (SlidingAttention i64) (UnmaskedAttention))
(datatype PagedKvLayout (TokenMajorKv) (HeadMajorKv))
(relation attention-op (i64 Expression Expression Expression Expression Expression Expression DType f64 AttentionMask))
(relation paged-kv-view (i64 i64 Expression Expression PagedKvLayout))
(relation persistent-state-attention-op (i64 i64))";

/// Registers the semantic vocabulary. This is not an executable provider.
#[derive(Debug, Default)]
pub struct AttentionSemantics;

impl EgglogOp for AttentionSemantics {
    fn sort(&self) -> SortDef {
        sort(
            orbitkv_compiler::egglog_utils::base::OP_KIND,
            "AttentionSemantics",
            &[],
        )
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![ATTENTION_DECLARATIONS.to_owned()]
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl CustomOp for Attention {
    fn compiler_declarations(&self) -> &'static str {
        ATTENTION_DECLARATIONS
    }

    fn compiler_facts(&self, custom_op_id: usize) -> String {
        format!(
            "(attention-op {custom_op_id} {} {} {} {} {} {} ({:?}) {:?} {})\n{}",
            Expression::from(self.spec.query_heads).to_egglog(),
            Expression::from(self.spec.kv_heads).to_egglog(),
            Expression::from(self.spec.query_key_dim).to_egglog(),
            Expression::from(self.spec.value_dim).to_egglog(),
            self.query_tokens.to_egglog(),
            self.requests.to_egglog(),
            self.spec.dtype,
            self.spec.scale,
            self.spec.mask.to_egglog(),
            self.kv.compiler_facts(custom_op_id),
        )
    }
}

/// Builds attention without selecting its algorithm, provider or execution phase.
///
/// The output is [query_heads, query_tokens, value_dim]. Unsupported backend
/// combinations remain unlowered; they cannot be extracted as executable work.
/// Runtime CSR contents and page ownership must be validated by the state owner.
pub fn attention(
    inputs: AttentionInputs,
    spec: AttentionSpec,
) -> Result<GraphTensor, AttentionError> {
    spec.validate()?;
    let AttentionInputs {
        query,
        query_indptr,
        kv,
    } = inputs;
    let dimensions = query.dims();
    if dimensions.len() != 3
        || dimensions[1] != Expression::from(spec.query_heads)
        || dimensions[2] != Expression::from(spec.query_key_dim)
    {
        return Err(AttentionError::Shape("query"));
    }
    let query_tokens = dimensions[0];
    if query_tokens.to_usize() == Some(0) {
        return Err(AttentionError::Geometry("query tokens"));
    }
    let (kv_spec, requests, [key, value, indices, page_indptr, last]) =
        kv.validate(query, query_indptr, spec)?;
    if let (Some(tokens), Some(requests)) = (query_tokens.to_usize(), requests.to_usize())
        && tokens < requests
    {
        return Err(AttentionError::Geometry("each request requires a query"));
    }
    Ok(query
        .graph()
        .custom_op(
            Attention {
                spec,
                query_tokens,
                requests,
                kv: kv_spec,
            },
            vec![query, key, value, indices, query_indptr, page_indptr, last],
            (spec.query_heads, query_tokens, spec.value_dim),
            spec.dtype,
        )
        .output())
}
