//! Ragged causal prefill contract, metadata view, and CPU reference.

use super::SINGLE_DECODE_HEAD_DIM;
use crate::ContractError;
use crate::error::require_len;
use half::bf16;

/// BF16 ragged prefill over contiguous NHD query, key, and value storage.
///
/// Query and output use `[nnz_qo, query_heads, 128]`. Key and value use
/// `[nnz_kv, kv_heads, 128]`. Two I32 indptr arrays partition those flattened
/// rows into requests. The causal mask is bottom-right aligned, matching
/// FlashInfer batch ragged prefill:
///
/// `kv_index <= kv_len - qo_len + query_index`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16RaggedPrefillSpec {
    batch_size: usize,
    nnz_qo: usize,
    nnz_kv: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    softmax_scale: f32,
}

impl Bf16RaggedPrefillSpec {
    pub fn new(
        batch_size: usize,
        nnz_qo: usize,
        nnz_kv: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ContractError> {
        if batch_size == 0
            || nnz_qo == 0
            || nnz_kv == 0
            || num_query_heads == 0
            || num_kv_heads == 0
            || head_dim == 0
        {
            return Err(ContractError::ZeroDimension);
        }
        if head_dim != SINGLE_DECODE_HEAD_DIM {
            return Err(ContractError::UnsupportedHeadDimension {
                expected: SINGLE_DECODE_HEAD_DIM,
                actual: head_dim,
            });
        }
        if !num_query_heads.is_multiple_of(num_kv_heads) {
            return Err(ContractError::InvalidHeadMapping {
                query_heads: num_query_heads,
                kv_heads: num_kv_heads,
            });
        }
        i32::try_from(nnz_qo).map_err(|_| ContractError::ElementCountOverflow)?;
        i32::try_from(nnz_kv).map_err(|_| ContractError::ElementCountOverflow)?;

        batch_size
            .checked_add(1)
            .ok_or(ContractError::ElementCountOverflow)?;
        nnz_qo
            .checked_mul(num_query_heads)
            .and_then(|heads| heads.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;
        nnz_kv
            .checked_mul(num_kv_heads)
            .and_then(|heads| heads.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            batch_size,
            nnz_qo,
            nnz_kv,
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

    pub const fn nnz_kv(self) -> usize {
        self.nnz_kv
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

    pub const fn softmax_scale(self) -> f32 {
        self.softmax_scale
    }

    pub const fn gqa_group_size(self) -> usize {
        self.num_query_heads / self.num_kv_heads
    }

    pub const fn query_numel(self) -> usize {
        self.nnz_qo * self.num_query_heads * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn kv_numel(self) -> usize {
        self.nnz_kv * self.num_kv_heads * SINGLE_DECODE_HEAD_DIM
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
        kv_indptr: &'a [i32],
    ) -> Result<Bf16RaggedPrefillMetadata<'a>, ContractError> {
        require_len("qo_indptr", qo_indptr.len(), self.indptr_numel())?;
        require_len("kv_indptr", kv_indptr.len(), self.indptr_numel())?;
        validate_indptr("qo_indptr", qo_indptr, self.nnz_qo)?;
        validate_indptr("kv_indptr", kv_indptr, self.nnz_kv)?;

        for request in 0..self.batch_size {
            let qo_len = (qo_indptr[request + 1] - qo_indptr[request]) as usize;
            let kv_len = (kv_indptr[request + 1] - kv_indptr[request]) as usize;
            if qo_len > kv_len {
                return Err(ContractError::RaggedQueryLongerThanKv {
                    request,
                    query_len: qo_len,
                    kv_len,
                });
            }
        }

        Ok(Bf16RaggedPrefillMetadata {
            spec: self,
            qo_indptr,
            kv_indptr,
        })
    }
}

fn validate_indptr(
    buffer: &'static str,
    indptr: &[i32],
    expected_total: usize,
) -> Result<(), ContractError> {
    if indptr[0] != 0 {
        return Err(ContractError::InvalidIndptrStart {
            buffer,
            actual: indptr[0],
        });
    }
    for request in 0..indptr.len() - 1 {
        let start = indptr[request];
        let end = indptr[request + 1];
        if end < start {
            return Err(ContractError::NonMonotonicIndptr {
                buffer,
                request,
                start,
                end,
            });
        }
        if end == start {
            return Err(ContractError::EmptyRaggedRequest { buffer, request });
        }
    }
    let actual_total = usize::try_from(indptr[indptr.len() - 1])
        .map_err(|_| ContractError::ElementCountOverflow)?;
    require_len(buffer, actual_total, expected_total)
}

/// Validated ragged request metadata for one prefill invocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16RaggedPrefillMetadata<'a> {
    spec: Bf16RaggedPrefillSpec,
    qo_indptr: &'a [i32],
    kv_indptr: &'a [i32],
}

impl<'a> Bf16RaggedPrefillMetadata<'a> {
    pub const fn spec(self) -> Bf16RaggedPrefillSpec {
        self.spec
    }

    pub const fn qo_indptr(self) -> &'a [i32] {
        self.qo_indptr
    }

    pub const fn kv_indptr(self) -> &'a [i32] {
        self.kv_indptr
    }

    pub fn request_row_ranges(self, request: usize) -> Option<((usize, usize), (usize, usize))> {
        if request >= self.spec.batch_size {
            return None;
        }
        Some((
            (
                self.qo_indptr[request] as usize,
                self.qo_indptr[request + 1] as usize,
            ),
            (
                self.kv_indptr[request] as usize,
                self.kv_indptr[request + 1] as usize,
            ),
        ))
    }

    pub fn causal_kv_end(self, request: usize, query_index: usize) -> Option<usize> {
        let ((qo_start, qo_end), (kv_start, kv_end)) = self.request_row_ranges(request)?;
        let qo_len = qo_end - qo_start;
        let kv_len = kv_end - kv_start;
        if query_index >= qo_len {
            return None;
        }
        Some(kv_start + kv_len - qo_len + query_index + 1)
    }
}

/// Computes BF16 ragged causal prefill with F32 online softmax.
#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_reference(
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    output: &mut [bf16],
    lse: &mut [f32],
    spec: Bf16RaggedPrefillSpec,
) -> Result<(), ContractError> {
    require_len("Q", query.len(), spec.query_numel())?;
    require_len("K", key.len(), spec.kv_numel())?;
    require_len("V", value.len(), spec.kv_numel())?;
    require_len("O", output.len(), spec.output_numel())?;
    require_len("LSE", lse.len(), spec.lse_numel())?;
    let metadata = spec.validate_metadata(qo_indptr, kv_indptr)?;

    for request in 0..spec.batch_size() {
        let ((qo_start, qo_end), (kv_start, _)) = metadata
            .request_row_ranges(request)
            .expect("enumerated request has validated row ranges");
        for query_row in qo_start..qo_end {
            let query_index = query_row - qo_start;
            let kv_end = metadata
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

                for token in kv_start..kv_end {
                    let kv_offset =
                        (token * spec.num_kv_heads() + kv_head) * SINGLE_DECODE_HEAD_DIM;
                    let dot = (0..SINGLE_DECODE_HEAD_DIM).fold(0.0_f32, |sum, component| {
                        query[query_offset + component]
                            .to_f32()
                            .mul_add(key[kv_offset + component].to_f32(), sum)
                    });
                    let score = dot * spec.softmax_scale();
                    if token == kv_start {
                        max_score = score;
                        normalizer = 1.0;
                        for component in 0..SINGLE_DECODE_HEAD_DIM {
                            output_state[component] = value[kv_offset + component].to_f32();
                        }
                        continue;
                    }

                    let next_max = max_score.max(score);
                    let previous_weight = (max_score - next_max).exp();
                    let current_weight = (score - next_max).exp();
                    normalizer = normalizer * previous_weight + current_weight;
                    for component in 0..SINGLE_DECODE_HEAD_DIM {
                        output_state[component] = output_state[component] * previous_weight
                            + value[kv_offset + component].to_f32() * current_weight;
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
#[path = "ragged_prefill/tests.rs"]
mod tests;
