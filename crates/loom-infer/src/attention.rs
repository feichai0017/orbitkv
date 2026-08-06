//! Attention contracts and CPU references.
//!
//! The public module is a stable facade. Implementation modules follow the
//! operator domains used by the upstream comparison surface without exposing
//! FlashInfer's Python wrapper structure.

mod paged_decode;
mod single_decode;

pub use paged_decode::{
    Bf16PagedBatchDecodePageTable, Bf16PagedBatchDecodeSpec, PAGED_BATCH_DECODE_PAGE_SIZE,
    paged_batch_decode_bf16_reference,
};
pub use single_decode::{
    Bf16SingleDecodeSpec, Bf16SingleDecodeSplitKSpec, SINGLE_DECODE_HEAD_DIM,
    SINGLE_DECODE_PARTIAL_STATE_WIDTH, single_decode_bf16_reference,
    single_decode_bf16_split_k_merge_reference, single_decode_bf16_split_k_partials_reference,
    single_decode_bf16_split_k_reference,
};

#[cfg(test)]
#[path = "attention/tests.rs"]
mod tests;
