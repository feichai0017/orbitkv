use orbitkv_compiler::{
    dtype::DType,
    prelude::{Expression, GraphTensor},
};

use super::{AttentionError, AttentionSpec};

/// Contiguous element order within a page. The allocation's tensor may be flat;
/// this explicit view defines its interpretation without moving any elements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagedKvLayout {
    /// [page, token, head, dimension] (NHD).
    TokenMajor,
    /// [page, head, token, dimension] (HND).
    HeadMajor,
}

impl PagedKvLayout {
    pub fn to_egglog(self) -> &'static str {
        match self {
            Self::TokenMajor => "(TokenMajorKv)",
            Self::HeadMajor => "(HeadMajorKv)",
        }
    }
}

/// CSR page traversal authored by an external state manager. K and V are
/// separate, unquantized allocations with the same page count. Read visibility
/// does not transfer allocation ownership to the attention implementation.
#[derive(Clone, Copy)]
pub struct PagedKvView {
    pub state_class_id: u16,
    pub key: GraphTensor,
    pub value: GraphTensor,
    pub page_size: usize,
    pub layout: PagedKvLayout,
    pub page_indices: GraphTensor,
    pub page_indptr: GraphTensor,
    pub last_page_len: GraphTensor,
}

/// Physical KV representations with implemented graph contracts. New variants
/// require their own metadata, validation and backend lowering; a latent cache
/// must never be reinterpreted as an ordinary K/V pair.
#[derive(Clone, Copy)]
pub enum KvView {
    Paged(PagedKvView),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum KvViewSpec {
    Paged {
        state_class_id: u16,
        page_size: usize,
        context_pages: Expression,
        layout: PagedKvLayout,
    },
}

impl KvViewSpec {
    pub(super) fn compiler_facts(self, id: usize) -> String {
        match self {
            Self::Paged {
                state_class_id,
                page_size,
                context_pages,
                layout,
            } => format!(
                "(paged-kv-view {id} {state_class_id} {} {} {})\n(persistent-state-attention-op {id} {state_class_id})",
                Expression::from(page_size).to_egglog(),
                context_pages.to_egglog(),
                layout.to_egglog(),
            ),
        }
    }
}

impl KvView {
    pub(super) fn validate(
        self,
        query: GraphTensor,
        query_indptr: GraphTensor,
        spec: AttentionSpec,
    ) -> Result<(KvViewSpec, Expression, [GraphTensor; 5]), AttentionError> {
        let Self::Paged(view) = self;
        let tensors = [
            query,
            view.key,
            view.value,
            view.page_indices,
            query_indptr,
            view.page_indptr,
            view.last_page_len,
        ];
        if tensors
            .iter()
            .any(|tensor| tensor.graph_ref != query.graph_ref)
        {
            return Err(AttentionError::GraphOwnership);
        }
        if tensors.iter().any(|tensor| !tensor.shape.is_contiguous()) {
            return Err(AttentionError::Layout("views require contiguous storage"));
        }
        if [query, view.key, view.value]
            .iter()
            .any(|tensor| tensor.dtype != spec.dtype)
        {
            return Err(AttentionError::DType("query/K/V encoding"));
        }
        for tensor in [
            view.page_indices,
            query_indptr,
            view.page_indptr,
            view.last_page_len,
        ] {
            if tensor.dtype != DType::Int {
                return Err(AttentionError::DType("CSR metadata must be Int"));
            }
            if tensor.dims().len() != 1 {
                return Err(AttentionError::Shape(
                    "CSR metadata must be one-dimensional",
                ));
            }
        }
        let requests = view.last_page_len.dims()[0];
        if requests.to_usize() == Some(0) || view.page_size == 0 {
            return Err(AttentionError::Geometry("request count/page size"));
        }
        if query_indptr.dims()[0] != requests + 1 || view.page_indptr.dims()[0] != requests + 1 {
            return Err(AttentionError::Shape("query/page CSR row cardinality"));
        }
        let mut capacities = Vec::new();
        for (tensor, dimension) in [(view.key, spec.query_key_dim), (view.value, spec.value_dim)] {
            let page_elements = view
                .page_size
                .checked_mul(spec.kv_heads)
                .and_then(|value| value.checked_mul(dimension))
                .ok_or(AttentionError::Geometry("page element count overflow"))?;
            let elements = tensor.dims().into_iter().product::<Expression>();
            if let Some(elements) = elements.to_usize() {
                if elements == 0 || !elements.is_multiple_of(page_elements) {
                    return Err(AttentionError::Shape(
                        "K/V storage must contain complete pages",
                    ));
                }
                capacities.push(elements / page_elements);
            }
        }
        if capacities.len() == 2 && capacities[0] != capacities[1] {
            return Err(AttentionError::Shape("K/V page capacities differ"));
        }
        Ok((
            KvViewSpec::Paged {
                state_class_id: view.state_class_id,
                page_size: view.page_size,
                context_pages: view.page_indices.dims()[0],
                layout: view.layout,
            },
            requests,
            [
                view.key,
                view.value,
                view.page_indices,
                view.page_indptr,
                view.last_page_len,
            ],
        ))
    }
}
