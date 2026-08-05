//! Single-request decode-attention contracts and CPU references.

use crate::ContractError;
use crate::error::require_len;
use half::bf16;

/// Head dimension admitted by the first single-decode contract.
pub const SINGLE_DECODE_HEAD_DIM: usize = 128;

/// Page size admitted by the first paged batch-decode contract.
pub const PAGED_BATCH_DECODE_PAGE_SIZE: usize = 16;

/// F32 values in one unnormalized split-K softmax state.
///
/// The layout is `[max_score_log2, normalizer, weighted_value[128]]`.
pub const SINGLE_DECODE_PARTIAL_STATE_WIDTH: usize = SINGLE_DECODE_HEAD_DIM + 2;

/// BF16 single-request decode over contiguous NHD key and value caches.
///
/// Query and output use `[query_heads, 128]`. Key and value use
/// `[kv_len, kv_heads, 128]`. Scores and online-softmax state use F32. LSE is
/// `log2(sum(exp(scaled_logit)))` to match FlashInfer's returned state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16SingleDecodeSpec {
    kv_len: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    softmax_scale: f32,
}

impl Bf16SingleDecodeSpec {
    /// Creates the first fixed BF16, NHD, full-window decode contract.
    pub fn new(
        kv_len: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ContractError> {
        if kv_len == 0 || num_query_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
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

        num_query_heads
            .checked_mul(head_dim)
            .ok_or(ContractError::ElementCountOverflow)?;
        let elements_per_token = num_kv_heads
            .checked_mul(head_dim)
            .ok_or(ContractError::ElementCountOverflow)?;
        kv_len
            .checked_mul(elements_per_token)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            kv_len,
            num_query_heads,
            num_kv_heads,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub const fn kv_len(self) -> usize {
        self.kv_len
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
        self.num_query_heads * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn kv_numel(self) -> usize {
        self.kv_len * self.num_kv_heads * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn output_numel(self) -> usize {
        self.query_numel()
    }

    pub const fn lse_numel(self) -> usize {
        self.num_query_heads
    }

    pub const fn kv_head_for_query_head(self, query_head: usize) -> Option<usize> {
        if query_head < self.num_query_heads {
            Some(query_head / self.gqa_group_size())
        } else {
            None
        }
    }
}

/// BF16 batch decode over a FlashInfer-compatible paged NHD KV cache.
///
/// Query and output use `[batch_size, query_heads, 128]`. Key and value pages
/// use `[max_num_pages, 16, kv_heads, 128]`. Each request contributes one query
/// token and at least one KV page. This first contract uses a full attention
/// window with no positional encoding or logits soft cap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16PagedBatchDecodeSpec {
    batch_size: usize,
    max_num_pages: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    softmax_scale: f32,
}

impl Bf16PagedBatchDecodeSpec {
    /// Creates the fixed BF16, NHD, D128, page-size-16 batch-decode contract.
    pub fn new(
        batch_size: usize,
        max_num_pages: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
    ) -> Result<Self, ContractError> {
        if batch_size == 0
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
        if page_size != PAGED_BATCH_DECODE_PAGE_SIZE {
            return Err(ContractError::UnsupportedPageSize {
                expected: PAGED_BATCH_DECODE_PAGE_SIZE,
                actual: page_size,
            });
        }
        if !num_query_heads.is_multiple_of(num_kv_heads) {
            return Err(ContractError::InvalidHeadMapping {
                query_heads: num_query_heads,
                kv_heads: num_kv_heads,
            });
        }

        batch_size
            .checked_add(1)
            .and_then(|_| batch_size.checked_mul(num_query_heads))
            .and_then(|heads| heads.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;
        max_num_pages
            .checked_mul(page_size)
            .and_then(|tokens| tokens.checked_mul(num_kv_heads))
            .and_then(|heads| heads.checked_mul(head_dim))
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            batch_size,
            max_num_pages,
            num_query_heads,
            num_kv_heads,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub const fn batch_size(self) -> usize {
        self.batch_size
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
        PAGED_BATCH_DECODE_PAGE_SIZE
    }

    pub const fn softmax_scale(self) -> f32 {
        self.softmax_scale
    }

    pub const fn gqa_group_size(self) -> usize {
        self.num_query_heads / self.num_kv_heads
    }

    pub const fn query_numel(self) -> usize {
        self.batch_size * self.num_query_heads * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn kv_pages_numel(self) -> usize {
        self.max_num_pages
            * PAGED_BATCH_DECODE_PAGE_SIZE
            * self.num_kv_heads
            * SINGLE_DECODE_HEAD_DIM
    }

    pub const fn output_numel(self) -> usize {
        self.query_numel()
    }

    pub const fn lse_numel(self) -> usize {
        self.batch_size * self.num_query_heads
    }

    pub const fn page_indptr_numel(self) -> usize {
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

    /// Validates dynamic page-table metadata and returns an indexed view.
    pub fn validate_page_table<'a>(
        self,
        page_indptr: &'a [i32],
        page_indices: &'a [i32],
        last_page_len: &'a [i32],
    ) -> Result<Bf16PagedBatchDecodePageTable<'a>, ContractError> {
        require_len("page_indptr", page_indptr.len(), self.page_indptr_numel())?;
        require_len(
            "last_page_len",
            last_page_len.len(),
            self.last_page_len_numel(),
        )?;
        if page_indptr[0] != 0 {
            return Err(ContractError::InvalidPageIndptrStart {
                actual: page_indptr[0],
            });
        }

        for request in 0..self.batch_size {
            let start = page_indptr[request];
            let end = page_indptr[request + 1];
            if end < start {
                return Err(ContractError::NonMonotonicPageIndptr {
                    request,
                    start,
                    end,
                });
            }
            if end == start {
                return Err(ContractError::EmptyPagedRequest { request });
            }
            let length = last_page_len[request];
            if !(1..=PAGED_BATCH_DECODE_PAGE_SIZE as i32).contains(&length) {
                return Err(ContractError::InvalidLastPageLength {
                    request,
                    length,
                    page_size: PAGED_BATCH_DECODE_PAGE_SIZE,
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

        Ok(Bf16PagedBatchDecodePageTable {
            spec: self,
            page_indptr,
            page_indices,
            last_page_len,
        })
    }
}

/// Validated dynamic metadata for one paged batch-decode invocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16PagedBatchDecodePageTable<'a> {
    spec: Bf16PagedBatchDecodeSpec,
    page_indptr: &'a [i32],
    page_indices: &'a [i32],
    last_page_len: &'a [i32],
}

impl<'a> Bf16PagedBatchDecodePageTable<'a> {
    pub const fn spec(self) -> Bf16PagedBatchDecodeSpec {
        self.spec
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
            .checked_mul(PAGED_BATCH_DECODE_PAGE_SIZE)?
            .checked_add(self.last_page_len[request] as usize)
    }

    /// Maps a logical request token to `(physical_page, offset_in_page)`.
    pub fn physical_page_for_token(
        self,
        request: usize,
        logical_token: usize,
    ) -> Option<(usize, usize)> {
        let kv_len = self.request_kv_len(request)?;
        if logical_token >= kv_len {
            return None;
        }
        let (page_start, _) = self
            .request_page_range(request)
            .expect("validated request has a page range");
        let page_slot = logical_token / PAGED_BATCH_DECODE_PAGE_SIZE;
        let page_offset = logical_token % PAGED_BATCH_DECODE_PAGE_SIZE;
        Some((
            self.page_indices[page_start + page_slot] as usize,
            page_offset,
        ))
    }
}

/// Computes BF16 paged batch decode with F32 online softmax.
///
/// Requests, query heads, logical KV tokens, and head components are visited
/// in ascending order. Physical pages may appear in any validated order.
/// Output rounds once after normalization and LSE uses base-two logarithms.
#[allow(clippy::too_many_arguments)]
pub fn paged_batch_decode_bf16_reference(
    query: &[bf16],
    key_pages: &[bf16],
    value_pages: &[bf16],
    page_indptr: &[i32],
    page_indices: &[i32],
    last_page_len: &[i32],
    output: &mut [bf16],
    lse: &mut [f32],
    spec: Bf16PagedBatchDecodeSpec,
) -> Result<(), ContractError> {
    require_len("Q", query.len(), spec.query_numel())?;
    require_len("K_pages", key_pages.len(), spec.kv_pages_numel())?;
    require_len("V_pages", value_pages.len(), spec.kv_pages_numel())?;
    require_len("O", output.len(), spec.output_numel())?;
    require_len("LSE", lse.len(), spec.lse_numel())?;
    let page_table = spec.validate_page_table(page_indptr, page_indices, last_page_len)?;

    for request in 0..spec.batch_size() {
        let kv_len = page_table
            .request_kv_len(request)
            .expect("enumerated request has validated non-empty pages");
        for query_head in 0..spec.num_query_heads() {
            let query_offset =
                (request * spec.num_query_heads() + query_head) * SINGLE_DECODE_HEAD_DIM;
            let kv_head = spec
                .kv_head_for_query_head(query_head)
                .expect("enumerated query head is inside the validated specification");
            let mut output_state = [0.0_f32; SINGLE_DECODE_HEAD_DIM];
            let mut max_score = 0.0_f32;
            let mut normalizer = 0.0_f32;

            for token in 0..kv_len {
                let (physical_page, page_offset) = page_table
                    .physical_page_for_token(request, token)
                    .expect("enumerated token is inside the validated request");
                let kv_offset = ((physical_page * PAGED_BATCH_DECODE_PAGE_SIZE + page_offset)
                    * spec.num_kv_heads()
                    + kv_head)
                    * SINGLE_DECODE_HEAD_DIM;
                let dot = (0..SINGLE_DECODE_HEAD_DIM).fold(0.0_f32, |sum, component| {
                    query[query_offset + component]
                        .to_f32()
                        .mul_add(key_pages[kv_offset + component].to_f32(), sum)
                });
                let score = dot * spec.softmax_scale();

                if token == 0 {
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
                (request * spec.num_query_heads() + query_head) * SINGLE_DECODE_HEAD_DIM;
            for component in 0..SINGLE_DECODE_HEAD_DIM {
                output[output_offset + component] =
                    bf16::from_f32(output_state[component] / normalizer);
            }
            lse[request * spec.num_query_heads() + query_head] =
                (max_score + normalizer.ln()) * core::f32::consts::LOG2_E;
        }
    }

    Ok(())
}

/// Split-K execution contract for the admitted single-decode specification.
///
/// The KV range is divided into non-empty, contiguous, balanced partitions.
/// Each `(query_head, partition)` writes one F32 partial state. The workspace
/// layout is `[query_heads, partitions, 130]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16SingleDecodeSplitKSpec {
    decode: Bf16SingleDecodeSpec,
    partitions: usize,
}

impl Bf16SingleDecodeSplitKSpec {
    pub fn new(decode: Bf16SingleDecodeSpec, partitions: usize) -> Result<Self, ContractError> {
        if partitions == 0 || partitions > decode.kv_len() {
            return Err(ContractError::InvalidPartitionCount {
                partitions,
                kv_len: decode.kv_len(),
            });
        }
        decode
            .num_query_heads()
            .checked_mul(partitions)
            .and_then(|states| states.checked_mul(SINGLE_DECODE_PARTIAL_STATE_WIDTH))
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self { decode, partitions })
    }

    pub const fn decode(self) -> Bf16SingleDecodeSpec {
        self.decode
    }

    pub const fn partitions(self) -> usize {
        self.partitions
    }

    pub const fn partial_state_width(self) -> usize {
        SINGLE_DECODE_PARTIAL_STATE_WIDTH
    }

    pub const fn partial_state_count(self) -> usize {
        self.decode.num_query_heads() * self.partitions
    }

    pub const fn workspace_numel(self) -> usize {
        self.partial_state_count() * SINGLE_DECODE_PARTIAL_STATE_WIDTH
    }

    pub const fn workspace_bytes(self) -> usize {
        self.workspace_numel() * core::mem::size_of::<f32>()
    }

    pub const fn partition_token_range(self, partition: usize) -> Option<(usize, usize)> {
        if partition >= self.partitions {
            return None;
        }
        let base = self.decode.kv_len() / self.partitions;
        let remainder = self.decode.kv_len() % self.partitions;
        let extra_before = if partition < remainder {
            partition
        } else {
            remainder
        };
        let start = partition * base + extra_before;
        let len = base + if partition < remainder { 1 } else { 0 };
        Some((start, start + len))
    }

    pub const fn partial_state_offset(self, query_head: usize, partition: usize) -> Option<usize> {
        if query_head >= self.decode.num_query_heads() || partition >= self.partitions {
            None
        } else {
            Some((query_head * self.partitions + partition) * SINGLE_DECODE_PARTIAL_STATE_WIDTH)
        }
    }
}

/// Computes BF16 single-request decode with F32 online softmax.
///
/// Dot products visit head components in ascending order. Softmax visits KV
/// tokens in ascending order. Output rounds once after normalization.
pub fn single_decode_bf16_reference(
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    output: &mut [bf16],
    lse: &mut [f32],
    spec: Bf16SingleDecodeSpec,
) -> Result<(), ContractError> {
    require_len("Q", query.len(), spec.query_numel())?;
    require_len("K", key.len(), spec.kv_numel())?;
    require_len("V", value.len(), spec.kv_numel())?;
    require_len("O", output.len(), spec.output_numel())?;
    require_len("LSE", lse.len(), spec.lse_numel())?;

    for (query_head, lse_slot) in lse.iter_mut().enumerate() {
        let query_offset = query_head * spec.head_dim();
        let kv_head = spec
            .kv_head_for_query_head(query_head)
            .expect("enumerated query head is inside the validated specification");
        let mut output_state = [0.0_f32; SINGLE_DECODE_HEAD_DIM];
        let mut max_score = 0.0_f32;
        let mut normalizer = 0.0_f32;

        for token in 0..spec.kv_len() {
            let kv_offset = (token * spec.num_kv_heads() + kv_head) * SINGLE_DECODE_HEAD_DIM;
            let dot = (0..SINGLE_DECODE_HEAD_DIM).fold(0.0_f32, |sum, component| {
                query[query_offset + component]
                    .to_f32()
                    .mul_add(key[kv_offset + component].to_f32(), sum)
            });
            let score = dot * spec.softmax_scale();

            if token == 0 {
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

        let output_offset = query_head * SINGLE_DECODE_HEAD_DIM;
        for component in 0..SINGLE_DECODE_HEAD_DIM {
            output[output_offset + component] =
                bf16::from_f32(output_state[component] / normalizer);
        }
        *lse_slot = (max_score + normalizer.ln()) * core::f32::consts::LOG2_E;
    }

    Ok(())
}

/// Computes unnormalized F32 softmax states for every split-K partition.
pub fn single_decode_bf16_split_k_partials_reference(
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    workspace: &mut [f32],
    spec: Bf16SingleDecodeSplitKSpec,
) -> Result<(), ContractError> {
    let decode = spec.decode();
    require_len("Q", query.len(), decode.query_numel())?;
    require_len("K", key.len(), decode.kv_numel())?;
    require_len("V", value.len(), decode.kv_numel())?;
    require_len("workspace", workspace.len(), spec.workspace_numel())?;

    for query_head in 0..decode.num_query_heads() {
        let query_offset = query_head * decode.head_dim();
        let kv_head = decode
            .kv_head_for_query_head(query_head)
            .expect("enumerated query head is inside the validated specification");
        for partition in 0..spec.partitions() {
            let (token_start, token_end) = spec
                .partition_token_range(partition)
                .expect("enumerated partition is inside the validated specification");
            let state_offset = spec
                .partial_state_offset(query_head, partition)
                .expect("enumerated state is inside the validated specification");
            let (header, weighted_value) = workspace
                [state_offset..state_offset + SINGLE_DECODE_PARTIAL_STATE_WIDTH]
                .split_at_mut(2);
            weighted_value.fill(0.0);

            let mut max_score_log2 = 0.0_f32;
            let mut normalizer = 0.0_f32;
            for token in token_start..token_end {
                let kv_offset = (token * decode.num_kv_heads() + kv_head) * SINGLE_DECODE_HEAD_DIM;
                let dot = (0..SINGLE_DECODE_HEAD_DIM).fold(0.0_f32, |sum, component| {
                    query[query_offset + component]
                        .to_f32()
                        .mul_add(key[kv_offset + component].to_f32(), sum)
                });
                let score_log2 = dot * decode.softmax_scale() * core::f32::consts::LOG2_E;

                if token == token_start {
                    max_score_log2 = score_log2;
                    normalizer = 1.0;
                    for component in 0..SINGLE_DECODE_HEAD_DIM {
                        weighted_value[component] = value[kv_offset + component].to_f32();
                    }
                    continue;
                }

                let next_max = max_score_log2.max(score_log2);
                let previous_weight = (max_score_log2 - next_max).exp2();
                let current_weight = (score_log2 - next_max).exp2();
                normalizer = normalizer * previous_weight + current_weight;
                for component in 0..SINGLE_DECODE_HEAD_DIM {
                    weighted_value[component] = weighted_value[component] * previous_weight
                        + value[kv_offset + component].to_f32() * current_weight;
                }
                max_score_log2 = next_max;
            }
            header[0] = max_score_log2;
            header[1] = normalizer;
        }
    }
    Ok(())
}

/// Merges split-K F32 states and writes final BF16 output plus log2-LSE.
pub fn single_decode_bf16_split_k_merge_reference(
    workspace: &[f32],
    output: &mut [bf16],
    lse: &mut [f32],
    spec: Bf16SingleDecodeSplitKSpec,
) -> Result<(), ContractError> {
    let decode = spec.decode();
    require_len("workspace", workspace.len(), spec.workspace_numel())?;
    require_len("O", output.len(), decode.output_numel())?;
    require_len("LSE", lse.len(), decode.lse_numel())?;

    for (query_head, lse_slot) in lse.iter_mut().enumerate() {
        let mut merged_value = [0.0_f32; SINGLE_DECODE_HEAD_DIM];
        let mut merged_max = 0.0_f32;
        let mut merged_normalizer = 0.0_f32;
        for partition in 0..spec.partitions() {
            let state_offset = spec
                .partial_state_offset(query_head, partition)
                .expect("enumerated state is inside the validated specification");
            let state = &workspace[state_offset..state_offset + SINGLE_DECODE_PARTIAL_STATE_WIDTH];
            let partition_max_log2 = state[0];
            let partition_normalizer = state[1];
            if partition == 0 {
                merged_max = partition_max_log2;
                merged_normalizer = partition_normalizer;
                merged_value.copy_from_slice(&state[2..]);
                continue;
            }

            let next_max = merged_max.max(partition_max_log2);
            let merged_weight = (merged_max - next_max).exp2();
            let partition_weight = (partition_max_log2 - next_max).exp2();
            merged_normalizer =
                merged_normalizer * merged_weight + partition_normalizer * partition_weight;
            for component in 0..SINGLE_DECODE_HEAD_DIM {
                merged_value[component] = merged_value[component] * merged_weight
                    + state[component + 2] * partition_weight;
            }
            merged_max = next_max;
        }

        let output_offset = query_head * SINGLE_DECODE_HEAD_DIM;
        for component in 0..SINGLE_DECODE_HEAD_DIM {
            output[output_offset + component] =
                bf16::from_f32(merged_value[component] / merged_normalizer);
        }
        *lse_slot = merged_max + merged_normalizer.log2();
    }
    Ok(())
}

/// Runs the split-K partial and merge references with caller-owned workspace.
pub fn single_decode_bf16_split_k_reference(
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    workspace: &mut [f32],
    output: &mut [bf16],
    lse: &mut [f32],
    spec: Bf16SingleDecodeSplitKSpec,
) -> Result<(), ContractError> {
    single_decode_bf16_split_k_partials_reference(query, key, value, workspace, spec)?;
    single_decode_bf16_split_k_merge_reference(workspace, output, lse, spec)
}

#[cfg(test)]
#[path = "attention/tests.rs"]
mod tests;
