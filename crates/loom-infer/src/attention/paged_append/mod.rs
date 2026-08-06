//! Standard RoPE plus one-token-per-request paged KV append.

use super::{
    Bf16PagedBatchDecodePageTable, Bf16PagedBatchDecodeSpec, PAGED_BATCH_DECODE_PAGE_SIZE,
    SINGLE_DECODE_HEAD_DIM,
};
use crate::error::require_len;
use crate::{Bf16RopePosIdsSpec, ContractError, rope_pos_ids_bf16_reference};
use half::bf16;

/// Maximum token count admitted by the first explicit append contract.
pub const ROPE_PAGED_KV_APPEND_MAX_TOKENS: usize = 64;

/// BF16 standard RoPE followed by one append into each request's final page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16RopePagedKvAppendSpec {
    paged: Bf16PagedBatchDecodeSpec,
}

impl Bf16RopePagedKvAppendSpec {
    /// Creates the first fused D128, page-size-16, NeoX split-half contract.
    pub fn new(
        batch_size: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
    ) -> Result<Self, ContractError> {
        Ok(Self {
            paged: Bf16PagedBatchDecodeSpec::new(
                batch_size,
                max_num_pages,
                num_query_heads,
                num_kv_heads,
                head_dim,
                page_size,
            )?,
        })
    }

    pub const fn batch_size(self) -> usize {
        self.paged.batch_size()
    }

    pub const fn max_num_pages(self) -> usize {
        self.paged.max_num_pages()
    }

    pub const fn num_query_heads(self) -> usize {
        self.paged.num_query_heads()
    }

    pub const fn num_kv_heads(self) -> usize {
        self.paged.num_kv_heads()
    }

    pub const fn head_dim(self) -> usize {
        SINGLE_DECODE_HEAD_DIM
    }

    pub const fn page_size(self) -> usize {
        PAGED_BATCH_DECODE_PAGE_SIZE
    }

    pub const fn query_numel(self) -> usize {
        self.paged.query_numel()
    }

    pub const fn key_numel(self) -> usize {
        self.batch_size() * self.num_kv_heads() * self.head_dim()
    }

    pub const fn value_numel(self) -> usize {
        self.key_numel()
    }

    pub const fn query_output_numel(self) -> usize {
        self.query_numel()
    }

    pub const fn kv_pages_numel(self) -> usize {
        self.paged.kv_pages_numel()
    }

    pub const fn page_indptr_numel(self) -> usize {
        self.paged.page_indptr_numel()
    }

    pub const fn last_page_len_numel(self) -> usize {
        self.paged.last_page_len_numel()
    }

    /// Validates the extended page table and rejects duplicate final slots.
    pub fn validate_page_table<'a>(
        self,
        page_indptr: &'a [i32],
        page_indices: &'a [i32],
        last_page_len: &'a [i32],
    ) -> Result<Bf16PagedBatchDecodePageTable<'a>, ContractError> {
        let table = self
            .paged
            .validate_page_table(page_indptr, page_indices, last_page_len)?;
        for first_request in 0..self.batch_size() {
            let first_position = table
                .request_kv_len(first_request)
                .expect("validated request has a KV length")
                - 1;
            let first_slot = table
                .physical_page_for_token(first_request, first_position)
                .expect("validated final token maps to a physical slot");
            for second_request in first_request + 1..self.batch_size() {
                let second_position = table
                    .request_kv_len(second_request)
                    .expect("validated request has a KV length")
                    - 1;
                let second_slot = table
                    .physical_page_for_token(second_request, second_position)
                    .expect("validated final token maps to a physical slot");
                if first_slot == second_slot {
                    return Err(ContractError::DuplicatePageAppendSlot {
                        first_request,
                        second_request,
                        physical_page: first_slot.0,
                        offset: first_slot.1,
                    });
                }
            }
        }
        Ok(table)
    }
}

/// BF16 standard RoPE followed by explicit multi-token paged KV append.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16RopePagedKvAppendTokensSpec {
    paged: Bf16PagedBatchDecodeSpec,
    tokens: usize,
}

impl Bf16RopePagedKvAppendTokensSpec {
    /// Creates the explicit D128, page-size-16, NeoX split-half contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokens: usize,
        batch_size: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
    ) -> Result<Self, ContractError> {
        if tokens == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if tokens > ROPE_PAGED_KV_APPEND_MAX_TOKENS {
            return Err(ContractError::UnsupportedAppendTokenCount {
                maximum: ROPE_PAGED_KV_APPEND_MAX_TOKENS,
                actual: tokens,
            });
        }
        let paged = Bf16PagedBatchDecodeSpec::new(
            batch_size,
            max_num_pages,
            num_query_heads,
            num_kv_heads,
            head_dim,
            page_size,
        )?;
        tokens
            .checked_mul(num_query_heads)
            .and_then(|heads| heads.checked_mul(head_dim))
            .and_then(|_| tokens.checked_mul(num_kv_heads))
            .and_then(|heads| heads.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self { paged, tokens })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn batch_size(self) -> usize {
        self.paged.batch_size()
    }

    pub const fn max_num_pages(self) -> usize {
        self.paged.max_num_pages()
    }

    pub const fn num_query_heads(self) -> usize {
        self.paged.num_query_heads()
    }

    pub const fn num_kv_heads(self) -> usize {
        self.paged.num_kv_heads()
    }

    pub const fn head_dim(self) -> usize {
        SINGLE_DECODE_HEAD_DIM
    }

    pub const fn page_size(self) -> usize {
        PAGED_BATCH_DECODE_PAGE_SIZE
    }

    pub const fn query_numel(self) -> usize {
        self.tokens * self.num_query_heads() * self.head_dim()
    }

    pub const fn key_numel(self) -> usize {
        self.tokens * self.num_kv_heads() * self.head_dim()
    }

    pub const fn value_numel(self) -> usize {
        self.key_numel()
    }

    pub const fn query_output_numel(self) -> usize {
        self.query_numel()
    }

    pub const fn kv_pages_numel(self) -> usize {
        self.paged.kv_pages_numel()
    }

    pub const fn batch_indices_numel(self) -> usize {
        self.tokens
    }

    pub const fn positions_numel(self) -> usize {
        self.tokens
    }

    pub const fn page_indptr_numel(self) -> usize {
        self.paged.page_indptr_numel()
    }

    pub const fn last_page_len_numel(self) -> usize {
        self.paged.last_page_len_numel()
    }

    /// Validates page metadata, explicit mappings, and physical write uniqueness.
    pub fn validate_metadata<'a>(
        self,
        batch_indices: &'a [i32],
        positions: &'a [i32],
        page_indptr: &'a [i32],
        page_indices: &'a [i32],
        last_page_len: &'a [i32],
    ) -> Result<Bf16RopePagedKvAppendTokensMetadata<'a>, ContractError> {
        require_len(
            "batch_indices",
            batch_indices.len(),
            self.batch_indices_numel(),
        )?;
        require_len("positions", positions.len(), self.positions_numel())?;
        let table = self
            .paged
            .validate_page_table(page_indptr, page_indices, last_page_len)?;
        let mut slots = Vec::with_capacity(self.tokens);
        for token in 0..self.tokens {
            let request = usize::try_from(batch_indices[token]).map_err(|_| {
                ContractError::AppendBatchIndexOutOfRange {
                    token,
                    index: batch_indices[token],
                    batch_size: self.batch_size(),
                }
            })?;
            if request >= self.batch_size() {
                return Err(ContractError::AppendBatchIndexOutOfRange {
                    token,
                    index: batch_indices[token],
                    batch_size: self.batch_size(),
                });
            }
            let kv_len = table
                .request_kv_len(request)
                .expect("validated request has a KV length");
            let position = usize::try_from(positions[token]).map_err(|_| {
                ContractError::AppendPositionOutOfRange {
                    token,
                    request,
                    position: positions[token],
                    kv_len,
                }
            })?;
            let slot = table.physical_page_for_token(request, position).ok_or(
                ContractError::AppendPositionOutOfRange {
                    token,
                    request,
                    position: positions[token],
                    kv_len,
                },
            )?;
            if let Some((first_token, _)) = slots
                .iter()
                .enumerate()
                .find(|(_, first_slot)| **first_slot == slot)
            {
                return Err(ContractError::DuplicatePageAppendTokenSlot {
                    first_token,
                    second_token: token,
                    physical_page: slot.0,
                    offset: slot.1,
                });
            }
            slots.push(slot);
        }
        Ok(Bf16RopePagedKvAppendTokensMetadata {
            spec: self,
            table,
            batch_indices,
            positions,
            slots,
        })
    }
}

/// Validated explicit token-to-request and token-to-physical-slot mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct Bf16RopePagedKvAppendTokensMetadata<'a> {
    spec: Bf16RopePagedKvAppendTokensSpec,
    table: Bf16PagedBatchDecodePageTable<'a>,
    batch_indices: &'a [i32],
    positions: &'a [i32],
    slots: Vec<(usize, usize)>,
}

impl<'a> Bf16RopePagedKvAppendTokensMetadata<'a> {
    pub const fn spec(&self) -> Bf16RopePagedKvAppendTokensSpec {
        self.spec
    }

    pub const fn page_table(&self) -> Bf16PagedBatchDecodePageTable<'a> {
        self.table
    }

    pub const fn batch_indices(&self) -> &'a [i32] {
        self.batch_indices
    }

    pub const fn positions(&self) -> &'a [i32] {
        self.positions
    }

    pub fn request_for_token(&self, token: usize) -> Option<usize> {
        self.batch_indices
            .get(token)
            .map(|&request| request as usize)
    }

    pub fn physical_slot_for_token(&self, token: usize) -> Option<(usize, usize)> {
        self.slots.get(token).copied()
    }
}

/// Rotates Q/K and appends rotated K plus unmodified V into paged NHD storage.
#[allow(clippy::too_many_arguments)]
pub fn rope_paged_kv_append_bf16_reference(
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    page_indptr: &[i32],
    page_indices: &[i32],
    last_page_len: &[i32],
    query_output: &mut [bf16],
    key_pages: &mut [bf16],
    value_pages: &mut [bf16],
    spec: Bf16RopePagedKvAppendSpec,
) -> Result<(), ContractError> {
    require_len("query", query.len(), spec.query_numel())?;
    require_len("key", key.len(), spec.key_numel())?;
    require_len("value", value.len(), spec.value_numel())?;
    require_len(
        "query_output",
        query_output.len(),
        spec.query_output_numel(),
    )?;
    require_len("key_pages", key_pages.len(), spec.kv_pages_numel())?;
    require_len("value_pages", value_pages.len(), spec.kv_pages_numel())?;
    let table = spec.validate_page_table(page_indptr, page_indices, last_page_len)?;

    let positions = (0..spec.batch_size())
        .map(|request| {
            i32::try_from(
                table
                    .request_kv_len(request)
                    .expect("validated request has a KV length")
                    - 1,
            )
            .map_err(|_| ContractError::ElementCountOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rope = Bf16RopePosIdsSpec::new(
        spec.batch_size(),
        spec.num_query_heads(),
        spec.num_kv_heads(),
        spec.head_dim(),
        spec.head_dim(),
        1.0,
        10_000.0,
    )?;
    let mut rotated_key = vec![bf16::NAN; spec.key_numel()];
    rope_pos_ids_bf16_reference(query, key, &positions, query_output, &mut rotated_key, rope)?;

    for (request, &position) in positions.iter().enumerate() {
        let logical_position = position as usize;
        let (physical_page, page_offset) = table
            .physical_page_for_token(request, logical_position)
            .expect("validated final token maps to a physical slot");
        for kv_head in 0..spec.num_kv_heads() {
            let source = (request * spec.num_kv_heads() + kv_head) * SINGLE_DECODE_HEAD_DIM;
            let destination = ((physical_page * PAGED_BATCH_DECODE_PAGE_SIZE + page_offset)
                * spec.num_kv_heads()
                + kv_head)
                * SINGLE_DECODE_HEAD_DIM;
            key_pages[destination..destination + SINGLE_DECODE_HEAD_DIM]
                .copy_from_slice(&rotated_key[source..source + SINGLE_DECODE_HEAD_DIM]);
            value_pages[destination..destination + SINGLE_DECODE_HEAD_DIM]
                .copy_from_slice(&value[source..source + SINGLE_DECODE_HEAD_DIM]);
        }
    }
    Ok(())
}

/// Rotates explicit Q/K tokens and appends K/V into paged NHD storage.
#[allow(clippy::too_many_arguments)]
pub fn rope_paged_kv_append_tokens_bf16_reference(
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    batch_indices: &[i32],
    positions: &[i32],
    page_indptr: &[i32],
    page_indices: &[i32],
    last_page_len: &[i32],
    query_output: &mut [bf16],
    key_pages: &mut [bf16],
    value_pages: &mut [bf16],
    spec: Bf16RopePagedKvAppendTokensSpec,
) -> Result<(), ContractError> {
    require_len("query", query.len(), spec.query_numel())?;
    require_len("key", key.len(), spec.key_numel())?;
    require_len("value", value.len(), spec.value_numel())?;
    require_len(
        "query_output",
        query_output.len(),
        spec.query_output_numel(),
    )?;
    require_len("key_pages", key_pages.len(), spec.kv_pages_numel())?;
    require_len("value_pages", value_pages.len(), spec.kv_pages_numel())?;
    let metadata = spec.validate_metadata(
        batch_indices,
        positions,
        page_indptr,
        page_indices,
        last_page_len,
    )?;
    let rope = Bf16RopePosIdsSpec::new(
        spec.tokens(),
        spec.num_query_heads(),
        spec.num_kv_heads(),
        spec.head_dim(),
        spec.head_dim(),
        1.0,
        10_000.0,
    )?;
    let mut rotated_key = vec![bf16::NAN; spec.key_numel()];
    rope_pos_ids_bf16_reference(query, key, positions, query_output, &mut rotated_key, rope)?;

    for token in 0..spec.tokens() {
        let (physical_page, page_offset) = metadata
            .physical_slot_for_token(token)
            .expect("validated token has a physical slot");
        for kv_head in 0..spec.num_kv_heads() {
            let source = (token * spec.num_kv_heads() + kv_head) * SINGLE_DECODE_HEAD_DIM;
            let destination = ((physical_page * PAGED_BATCH_DECODE_PAGE_SIZE + page_offset)
                * spec.num_kv_heads()
                + kv_head)
                * SINGLE_DECODE_HEAD_DIM;
            key_pages[destination..destination + SINGLE_DECODE_HEAD_DIM]
                .copy_from_slice(&rotated_key[source..source + SINGLE_DECODE_HEAD_DIM]);
            value_pages[destination..destination + SINGLE_DECODE_HEAD_DIM]
                .copy_from_slice(&value[source..source + SINGLE_DECODE_HEAD_DIM]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
