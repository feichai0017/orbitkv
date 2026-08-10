//! Backend-independent contracts and CPU references for Loom Infer.

#![forbid(unsafe_code)]

mod dtype;
mod error;

pub mod attention;
pub mod gemm;
pub mod rms_norm;
pub mod rope;

pub use attention::{
    Bf16PagedBatchDecodePageTable, Bf16PagedBatchDecodeSpec, Bf16PagedPrefillMetadata,
    Bf16PagedPrefillSpec, Bf16RaggedPrefillMetadata, Bf16RaggedPrefillSpec,
    Bf16RopePagedKvAppendMetadata, Bf16RopePagedKvAppendSpec, Bf16RopePagedKvAppendTokensMetadata,
    Bf16RopePagedKvAppendTokensSpec, Bf16SingleDecodeSpec, Bf16SingleDecodeSplitKSpec,
    PAGED_BATCH_DECODE_PAGE_SIZE, PAGED_PREFILL_PAGE_SIZE, PagedKvLayout,
    ROPE_PAGED_KV_APPEND_MAX_TOKENS, SINGLE_DECODE_HEAD_DIM, SINGLE_DECODE_PARTIAL_STATE_WIDTH,
    paged_batch_decode_bf16_reference, paged_prefill_bf16_reference, ragged_prefill_bf16_reference,
    rope_paged_kv_append_bf16_reference, rope_paged_kv_append_tokens_bf16_reference,
    single_decode_bf16_reference, single_decode_bf16_split_k_merge_reference,
    single_decode_bf16_split_k_partials_reference, single_decode_bf16_split_k_reference,
};
pub use dtype::DType;
pub use error::ContractError;
pub use gemm::{Bf16GemmSpec, bf16_gemm_reference};
pub use rms_norm::{
    RmsNormSpec, rms_norm_bf16_reference, rms_norm_f16_reference, rms_norm_f32_reference,
};
pub use rope::{Bf16RopePosIdsSpec, rope_pos_ids_bf16_reference};
