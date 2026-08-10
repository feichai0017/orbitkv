//! Attention contracts and CPU references.
//!
//! The public module is a stable facade. Implementation modules follow the
//! operator domains used by the upstream comparison surface without exposing
//! FlashInfer's Python wrapper structure.

mod paged_append;
mod paged_decode;
mod paged_prefill;
mod ragged_prefill;
mod single_decode;

pub use paged_append::{
    Bf16RopePagedKvAppendMetadata, Bf16RopePagedKvAppendSpec, Bf16RopePagedKvAppendTokensMetadata,
    Bf16RopePagedKvAppendTokensSpec, ROPE_PAGED_KV_APPEND_MAX_TOKENS,
    rope_paged_kv_append_bf16_reference, rope_paged_kv_append_tokens_bf16_reference,
};
pub use paged_decode::{
    Bf16PagedBatchDecodePageTable, Bf16PagedBatchDecodeSpec, PAGED_BATCH_DECODE_PAGE_SIZE,
    paged_batch_decode_bf16_reference,
};
pub use paged_prefill::{
    Bf16PagedPrefillMetadata, Bf16PagedPrefillSpec, PAGED_PREFILL_PAGE_SIZE,
    paged_prefill_bf16_reference,
};
pub use ragged_prefill::{
    Bf16RaggedPrefillMetadata, Bf16RaggedPrefillSpec, ragged_prefill_bf16_reference,
};
pub use single_decode::{
    Bf16SingleDecodeSpec, Bf16SingleDecodeSplitKSpec, SINGLE_DECODE_HEAD_DIM,
    SINGLE_DECODE_PARTIAL_STATE_WIDTH, single_decode_bf16_reference,
    single_decode_bf16_split_k_merge_reference, single_decode_bf16_split_k_partials_reference,
    single_decode_bf16_split_k_reference,
};
