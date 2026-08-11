//! Paged causal prefill contract, metadata view, and CPU reference.

use super::SINGLE_DECODE_HEAD_DIM;
use crate::ContractError;
use crate::error::require_len;
use half::bf16;

/// Page size admitted by the first paged-prefill contract.
pub const PAGED_PREFILL_PAGE_SIZE: usize = 16;

/// BF16 ragged-query prefill over a FlashInfer-compatible paged NHD KV cache.
///
/// Query and output use `[nnz_qo, query_heads, 128]`. Key and value pages use
/// `[max_num_pages, 16, kv_heads, 128]`. Query rows are partitioned by
/// `qo_indptr`; KV pages use `page_indptr`, physical page indices, and
/// per-request last-page lengths. The causal mask is bottom-right aligned:
///
/// `kv_index <= kv_len - qo_len + query_index`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16PagedPrefillSpec {
    batch_size: usize,
    nnz_qo: usize,
    max_num_pages: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    softmax_scale: f32,
}

impl Bf16PagedPrefillSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_size: usize,
        nnz_qo: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
    ) -> Result<Self, ContractError> {
        if batch_size == 0
            || nnz_qo == 0
            || max_num_pages == 0
            || num_query_heads == 0
            || num_kv_heads == 0
            || head_dim == 0
            || page_size == 0
        {
            return Err(ContractError::ZeroDimension);
        }
        if head_dim != SINGLE_DECODE_HEAD_DIM {
            return Err(ContractError::UnsupportedHeadDimension {
                expected: SINGLE_DECODE_HEAD_DIM,
                actual: head_dim,
            });
        }
        if page_size != PAGED_PREFILL_PAGE_SIZE {
            return Err(ContractError::UnsupportedPageSize {
                expected: PAGED_PREFILL_PAGE_SIZE,
                actual: page_size,
            });
        }
        if !num_query_heads.is_multiple_of(num_kv_heads) {
            return Err(ContractError::InvalidHeadMapping {
                query_heads: num_query_heads,
                kv_heads: num_kv_heads,
            });
        }
        i32::try_from(nnz_qo).map_err(|_| ContractError::ElementCountOverflow)?;

        batch_size
            .checked_add(1)
            .ok_or(ContractError::ElementCountOverflow)?;
        nnz_qo
            .checked_mul(num_query_heads)
            .and_then(|heads| heads.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;
        max_num_pages
            .checked_mul(page_size)
            .and_then(|tokens| tokens.checked_mul(num_kv_heads))
            .and_then(|heads| heads.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            batch_size,
            nnz_qo,
            max_num_pages,
            num_query_heads,
            num_kv_heads,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub const fn batch_size(self) -> usize {
        self.batch_size
    }

    pub const fn nnz_qo(self) -> usize {
        self.nnz_qo
    }

    pub const fn max_num_pages(self) -> usize {
        self.max_num_pages
    }

    pub const fn num_query_heads(self) -> usize {
        self.num_query_heads
    }

    pub const fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    pub const fn head_dim(self) -> usize {
        SINGLE_DECODE_HEAD_DIM
    }

    pub const fn page_size(self) -> usize {
        PAGED_PREFILL_PAGE_SIZE
    }

    pub const fn softmax_scale(self) -> f32 {
        self.softmax_scale
    }

    pub const fn gqa_group_size(self) -> usize {
        self.num_query_heads / self.num_kv_heads
    }

    pub const fn query_numel(self) -> usize {
        self.nnz_qo * self.num_query_heads * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn kv_pages_numel(self) -> usize {
        self.max_num_pages * PAGED_PREFILL_PAGE_SIZE * self.num_kv_heads * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn output_numel(self) -> usize {
        self.query_numel()
    }

    pub const fn lse_numel(self) -> usize {
        self.nnz_qo * self.num_query_heads
    }

    pub const fn indptr_numel(self) -> usize {
        self.batch_size + 1
    }

    pub const fn last_page_len_numel(self) -> usize {
        self.batch_size
    }

    pub const fn kv_head_for_query_head(self, query_head: usize) -> Option<usize> {
        if query_head < self.num_query_heads {
            Some(query_head / self.gqa_group_size())
        } else {
            None
        }
    }

    pub fn validate_metadata<'a>(
        self,
        qo_indptr: &'a [i32],
        page_indptr: &'a [i32],
        page_indices: &'a [i32],
        last_page_len: &'a [i32],
    ) -> Result<Bf16PagedPrefillMetadata<'a>, ContractError> {
        require_len("qo_indptr", qo_indptr.len(), self.indptr_numel())?;
        require_len("page_indptr", page_indptr.len(), self.indptr_numel())?;
        require_len(
            "last_page_len",
            last_page_len.len(),
            self.last_page_len_numel(),
        )?;
        validate_query_indptr(qo_indptr, self.nnz_qo)?;
        if page_indptr[0] != 0 {
            return Err(ContractError::InvalidPageIndptrStart {
                actual: page_indptr[0],
            });
        }

        for request in 0..self.batch_size {
            let page_start = page_indptr[request];
            let page_end = page_indptr[request + 1];
            if page_end < page_start {
                return Err(ContractError::NonMonotonicPageIndptr {
                    request,
                    start: page_start,
                    end: page_end,
                });
            }
            if page_end == page_start {
                return Err(ContractError::EmptyPagedRequest { request });
            }
            let tail_len = last_page_len[request];
            if !(1..=PAGED_PREFILL_PAGE_SIZE as i32).contains(&tail_len) {
                return Err(ContractError::InvalidLastPageLength {
                    request,
                    length: tail_len,
                    page_size: PAGED_PREFILL_PAGE_SIZE,
                });
            }
            let qo_len = (qo_indptr[request + 1] - qo_indptr[request]) as usize;
            let page_count = (page_end - page_start) as usize;
            let kv_len = (page_count - 1)
                .checked_mul(PAGED_PREFILL_PAGE_SIZE)
                .and_then(|tokens| tokens.checked_add(tail_len as usize))
                .ok_or(ContractError::ElementCountOverflow)?;
            if qo_len > kv_len {
                return Err(ContractError::RaggedQueryLongerThanKv {
                    request,
                    query_len: qo_len,
                    kv_len,
                });
            }
        }

        let expected_indices = usize::try_from(page_indptr[self.batch_size])
            .map_err(|_| ContractError::ElementCountOverflow)?;
        require_len("page_indices", page_indices.len(), expected_indices)?;
        for (position, &index) in page_indices.iter().enumerate() {
            let valid = usize::try_from(index).is_ok_and(|index| index < self.max_num_pages);
            if !valid {
                return Err(ContractError::PageIndexOutOfRange {
                    position,
                    index,
                    max_num_pages: self.max_num_pages,
                });
            }
        }

        Ok(Bf16PagedPrefillMetadata {
            spec: self,
            qo_indptr,
            page_indptr,
            page_indices,
            last_page_len,
        })
    }
}

fn validate_query_indptr(indptr: &[i32], expected_total: usize) -> Result<(), ContractError> {
    if indptr[0] != 0 {
        return Err(ContractError::InvalidIndptrStart {
            buffer: "qo_indptr",
            actual: indptr[0],
        });
    }
    for request in 0..indptr.len() - 1 {
        let start = indptr[request];
        let end = indptr[request + 1];
        if end < start {
            return Err(ContractError::NonMonotonicIndptr {
                buffer: "qo_indptr",
                request,
                start,
                end,
            });
        }
        if end == start {
            return Err(ContractError::EmptyRaggedRequest {
                buffer: "qo_indptr",
                request,
            });
        }
    }
    let actual_total = usize::try_from(indptr[indptr.len() - 1])
        .map_err(|_| ContractError::ElementCountOverflow)?;
    require_len("qo_indptr", actual_total, expected_total)
}

/// Validated dynamic metadata for one paged-prefill invocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16PagedPrefillMetadata<'a> {
    spec: Bf16PagedPrefillSpec,
    qo_indptr: &'a [i32],
    page_indptr: &'a [i32],
    page_indices: &'a [i32],
    last_page_len: &'a [i32],
}

impl<'a> Bf16PagedPrefillMetadata<'a> {
    pub const fn spec(self) -> Bf16PagedPrefillSpec {
        self.spec
    }

    pub const fn qo_indptr(self) -> &'a [i32] {
        self.qo_indptr
    }

    pub const fn page_indptr(self) -> &'a [i32] {
        self.page_indptr
    }

    pub const fn page_indices(self) -> &'a [i32] {
        self.page_indices
    }

    pub const fn last_page_len(self) -> &'a [i32] {
        self.last_page_len
    }

    pub fn request_query_range(self, request: usize) -> Option<(usize, usize)> {
        if request >= self.spec.batch_size {
            return None;
        }
        Some((
            self.qo_indptr[request] as usize,
            self.qo_indptr[request + 1] as usize,
        ))
    }

    pub fn request_page_range(self, request: usize) -> Option<(usize, usize)> {
        if request >= self.spec.batch_size {
            return None;
        }
        Some((
            self.page_indptr[request] as usize,
            self.page_indptr[request + 1] as usize,
        ))
    }

    pub fn request_kv_len(self, request: usize) -> Option<usize> {
        let (start, end) = self.request_page_range(request)?;
        (end - start - 1)
            .checked_mul(PAGED_PREFILL_PAGE_SIZE)?
            .checked_add(self.last_page_len[request] as usize)
    }

    pub fn causal_kv_end(self, request: usize, query_index: usize) -> Option<usize> {
        let (qo_start, qo_end) = self.request_query_range(request)?;
        let qo_len = qo_end - qo_start;
        let kv_len = self.request_kv_len(request)?;
        if query_index >= qo_len {
            return None;
        }
        Some(kv_len - qo_len + query_index + 1)
    }

    pub fn physical_page_for_token(
        self,
        request: usize,
        logical_token: usize,
    ) -> Option<(usize, usize)> {
        let kv_len = self.request_kv_len(request)?;
        if logical_token >= kv_len {
            return None;
        }
        let (page_start, _) = self.request_page_range(request)?;
        let page_slot = logical_token / PAGED_PREFILL_PAGE_SIZE;
        let page_offset = logical_token % PAGED_PREFILL_PAGE_SIZE;
        Some((
            self.page_indices[page_start + page_slot] as usize,
            page_offset,
        ))
    }
}

/// Computes BF16 paged causal prefill with F32 online softmax.
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_bf16_reference(
    query: &[bf16],
    key_pages: &[bf16],
    value_pages: &[bf16],
    qo_indptr: &[i32],
    page_indptr: &[i32],
    page_indices: &[i32],
    last_page_len: &[i32],
    output: &mut [bf16],
    lse: &mut [f32],
    spec: Bf16PagedPrefillSpec,
) -> Result<(), ContractError> {
    require_len("Q", query.len(), spec.query_numel())?;
    require_len("K_pages", key_pages.len(), spec.kv_pages_numel())?;
    require_len("V_pages", value_pages.len(), spec.kv_pages_numel())?;
    require_len("O", output.len(), spec.output_numel())?;
    require_len("LSE", lse.len(), spec.lse_numel())?;
    let metadata = spec.validate_metadata(qo_indptr, page_indptr, page_indices, last_page_len)?;

    for request in 0..spec.batch_size() {
        let (qo_start, qo_end) = metadata
            .request_query_range(request)
            .expect("enumerated request has a validated query range");
        for query_row in qo_start..qo_end {
            let query_index = query_row - qo_start;
            let causal_kv_end = metadata
                .causal_kv_end(request, query_index)
                .expect("enumerated query row has a causal KV bound");
            for query_head in 0..spec.num_query_heads() {
                let query_offset =
                    (query_row * spec.num_query_heads() + query_head) * SINGLE_DECODE_HEAD_DIM;
                let kv_head = spec
                    .kv_head_for_query_head(query_head)
                    .expect("enumerated query head is inside the validated specification");
                let mut output_state = [0.0_f32; SINGLE_DECODE_HEAD_DIM];
                let mut max_score = 0.0_f32;
                let mut normalizer = 0.0_f32;

                for logical_token in 0..causal_kv_end {
                    let (physical_page, page_offset) = metadata
                        .physical_page_for_token(request, logical_token)
                        .expect("causal token is inside the validated request");
                    let kv_offset = ((physical_page * PAGED_PREFILL_PAGE_SIZE + page_offset)
                        * spec.num_kv_heads()
                        + kv_head)
                        * SINGLE_DECODE_HEAD_DIM;
                    let dot = (0..SINGLE_DECODE_HEAD_DIM).fold(0.0_f32, |sum, component| {
                        query[query_offset + component]
                            .to_f32()
                            .mul_add(key_pages[kv_offset + component].to_f32(), sum)
                    });
                    let score = dot * spec.softmax_scale();
                    if logical_token == 0 {
                        max_score = score;
                        normalizer = 1.0;
                        for component in 0..SINGLE_DECODE_HEAD_DIM {
                            output_state[component] = value_pages[kv_offset + component].to_f32();
                        }
                        continue;
                    }

                    let next_max = max_score.max(score);
                    let previous_weight = (max_score - next_max).exp();
                    let current_weight = (score - next_max).exp();
                    normalizer = normalizer * previous_weight + current_weight;
                    for component in 0..SINGLE_DECODE_HEAD_DIM {
                        output_state[component] = output_state[component] * previous_weight
                            + value_pages[kv_offset + component].to_f32() * current_weight;
                    }
                    max_score = next_max;
                }

                let output_offset =
                    (query_row * spec.num_query_heads() + query_head) * SINGLE_DECODE_HEAD_DIM;
                for component in 0..SINGLE_DECODE_HEAD_DIM {
                    output[output_offset + component] =
                        bf16::from_f32(output_state[component] / normalizer);
                }
                lse[query_row * spec.num_query_heads() + query_head] =
                    (max_score + normalizer.ln()) * core::f32::consts::LOG2_E;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
