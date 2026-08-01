#ifndef LOOM_CUDA_H_
#define LOOM_CUDA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum LoomCudaStatus {
  LOOM_CUDA_SUCCESS = 0,
  LOOM_CUDA_INVALID_ARGUMENT = 1,
  LOOM_CUDA_UNSUPPORTED = 2,
  LOOM_CUDA_LAUNCH_ERROR = 3,
  LOOM_CUDA_UNAVAILABLE = 4,
};

enum LoomCudaKvCacheEncoding {
  LOOM_CUDA_KV_CACHE_NATIVE = 0,
  LOOM_CUDA_KV_CACHE_FP8_E4M3 = 1,
};

const char* loom_cuda_status_string(int status);

// F32 bring-up implementation of RMSNorm over contiguous [rows, hidden_size]
// input/output tensors and one contiguous [hidden_size] weight vector. The
// launch is asynchronous with respect to the supplied CUDA stream.
int loom_cuda_rms_norm_f32(const float* input, const float* weight,
                           float* output, uint32_t rows,
                           uint32_t hidden_size, float epsilon, void* stream);

// Pair-vectorized FP16 and BF16 implementations. Odd hidden sizes use a
// scalar fallback so row starts never violate four-byte pair alignment.
int loom_cuda_rms_norm_f16(const uint16_t* input, const uint16_t* weight,
                           uint16_t* output, uint32_t rows,
                           uint32_t hidden_size, float epsilon, void* stream);

int loom_cuda_rms_norm_bf16(const uint16_t* input, const uint16_t* weight,
                            uint16_t* output, uint32_t rows,
                            uint32_t hidden_size, float epsilon, void* stream);

// RMSNorm followed by dynamic per-token FP8 E4M3FN quantization. A non-null
// residual is updated with the storage-rounded input+residual sum before
// normalization. Output holds raw FP8 storage bytes and scales has one F32
// value per row, with approximately normalized_value = fp8(output) * scale.
// Low-precision inputs follow the input scalar arithmetic boundaries for the
// residual sum, normalization, and weight multiplication.
int loom_cuda_rms_norm_dynamic_fp8_f32(
    const float* input, const float* weight, float* residual, uint8_t* output,
    float* scales, uint32_t rows, uint32_t hidden_size, float epsilon,
    void* stream);

int loom_cuda_rms_norm_dynamic_fp8_f16(
    const uint16_t* input, const uint16_t* weight, uint16_t* residual,
    uint8_t* output, float* scales, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream);

int loom_cuda_rms_norm_dynamic_fp8_bf16(
    const uint16_t* input, const uint16_t* weight, uint16_t* residual,
    uint8_t* output, float* scales, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream);

// RMSNorm followed by symmetric dynamic per-token INT8 quantization. A
// non-null residual follows the same update semantics as the FP8 path. Scales
// has one F32 value per row and uses absmax / 127; an all-zero row has scale
// zero and an all-zero output.
int loom_cuda_rms_norm_dynamic_int8_f32(
    const float* input, const float* weight, float* residual, int8_t* output,
    float* scales, uint32_t rows, uint32_t hidden_size, float epsilon,
    void* stream);

int loom_cuda_rms_norm_dynamic_int8_f16(
    const uint16_t* input, const uint16_t* weight, uint16_t* residual,
    int8_t* output, float* scales, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream);

int loom_cuda_rms_norm_dynamic_int8_bf16(
    const uint16_t* input, const uint16_t* weight, uint16_t* residual,
    int8_t* output, float* scales, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream);

// Fused residual addition and RMSNorm over contiguous tensors. Both input and
// residual are updated in place:
//   residual = input + residual
//   input = RMSNorm(residual, weight, epsilon)
// input, residual, and weight must point to non-overlapping allocations.
int loom_cuda_add_rms_norm_f32(float* input, float* residual,
                               const float* weight, uint32_t rows,
                               uint32_t hidden_size, float epsilon,
                               void* stream);

// FP16 and BF16 use 128-bit/eight-element vectors when pointer/row alignment
// permits, then two-element vectors, then a scalar fallback. The materialized
// sum is rounded to the storage dtype before computing the RMS statistic.
int loom_cuda_add_rms_norm_f16(uint16_t* input, uint16_t* residual,
                               const uint16_t* weight, uint32_t rows,
                               uint32_t hidden_size, float epsilon,
                               void* stream);

int loom_cuda_add_rms_norm_bf16(uint16_t* input, uint16_t* residual,
                                const uint16_t* weight, uint32_t rows,
                                uint32_t hidden_size, float epsilon,
                                void* stream);

// Fused split-half SwiGLU activation over input [rows, 2 * width] and output
// [rows, width]: output = silu(input[:, :width]) * input[:, width:]. Low-
// precision activation values are rounded to their storage dtype before the
// multiply, matching vLLM. Input/output storage ranges must not overlap.
int loom_cuda_silu_and_mul_f32(const float* input, float* output,
                               uint32_t rows, uint32_t width, void* stream);

int loom_cuda_silu_and_mul_f16(const uint16_t* input, uint16_t* output,
                               uint32_t rows, uint32_t width, void* stream);

int loom_cuda_silu_and_mul_bf16(const uint16_t* input, uint16_t* output,
                                uint32_t rows, uint32_t width, void* stream);

// Fused SwiGLU and dynamic per-block FP8 E4M3FN quantization. FP16/BF16
// inputs use [rows, 2 * width]; output holds [rows, width] raw FP8 bytes.
// Scales have logical shape [rows, width / group_size] and may use row-major
// or group-major storage. group_size must be 64 or 128. scale_ub may be null;
// activation and multiplication remain in F32 until direct FP8 conversion.
int loom_cuda_silu_and_mul_dynamic_fp8_f16(
    const uint16_t* input, uint8_t* output, float* scales, uint32_t rows,
    uint32_t width, uint32_t group_size, const float* scale_ub,
    uint32_t scales_transposed, void* stream);

int loom_cuda_silu_and_mul_dynamic_fp8_bf16(
    const uint16_t* input, uint8_t* output, float* scales, uint32_t rows,
    uint32_t width, uint32_t group_size, const float* scale_ub,
    uint32_t scales_transposed, void* stream);

// Fused SwiGLU and symmetric dynamic per-token INT8 quantization. FP16/BF16
// inputs use [rows, 2 * width], signed INT8 output uses [rows, width], and
// scales use one F32 value per row. Low-precision storage rounding matches the
// materialized vLLM activation path. GEMM is deliberately outside this ABI.
int loom_cuda_silu_and_mul_dynamic_int8_f16(
    const uint16_t* input, int8_t* output, float* scales, uint32_t rows,
    uint32_t width, void* stream);

int loom_cuda_silu_and_mul_dynamic_int8_bf16(
    const uint16_t* input, int8_t* output, float* scales, uint32_t rows,
    uint32_t width, void* stream);

// Stable expert-major activation permutation around vendor grouped GEMM.
// topk_ids is contiguous int32 [tokens, top_k]. expert_map is either null
// (all experts local) or int32 [num_experts], where -1 marks a remote expert.
// Valid assignments are grouped by local expert while preserving flattened
// token/route order. Remote assignments follow local assignments, are grouped
// by global expert, and preserve flattened order within that expert. Their
// activation rows are zero-filled at the output tail.
// expert_offsets is int64 [num_local_experts + 1], inverse_permutation and
// permuted_assignment_ids are int32 [tokens * top_k]. The latter uses
// tokens * top_k as its invalid sentinel. Workspace is caller-owned bytes.
int loom_cuda_moe_permute_workspace_size(uint32_t assignments,
                                         uint32_t num_experts,
                                         uint64_t* workspace_bytes);

int loom_cuda_moe_permute_f32(
    const float* hidden_states, const int32_t* topk_ids,
    const int32_t* expert_map, float* permuted_hidden_states,
    int64_t* expert_offsets, int32_t* inverse_permutation,
    int32_t* permuted_assignment_ids, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t tokens, uint32_t hidden_size,
    uint32_t top_k, uint32_t num_experts, uint32_t num_local_experts,
    void* stream);

int loom_cuda_moe_permute_f16(
    const uint16_t* hidden_states, const int32_t* topk_ids,
    const int32_t* expert_map, uint16_t* permuted_hidden_states,
    int64_t* expert_offsets, int32_t* inverse_permutation,
    int32_t* permuted_assignment_ids, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t tokens, uint32_t hidden_size,
    uint32_t top_k, uint32_t num_experts, uint32_t num_local_experts,
    void* stream);

int loom_cuda_moe_permute_bf16(
    const uint16_t* hidden_states, const int32_t* topk_ids,
    const int32_t* expert_map, uint16_t* permuted_hidden_states,
    int64_t* expert_offsets, int32_t* inverse_permutation,
    int32_t* permuted_assignment_ids, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t tokens, uint32_t hidden_size,
    uint32_t top_k, uint32_t num_experts, uint32_t num_local_experts,
    void* stream);

int loom_cuda_moe_permute_fp8_e4m3fn(
    const uint8_t* hidden_states, const int32_t* topk_ids,
    const int32_t* expert_map, uint8_t* permuted_hidden_states,
    int64_t* expert_offsets, int32_t* inverse_permutation,
    int32_t* permuted_assignment_ids, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t tokens, uint32_t hidden_size,
    uint32_t top_k, uint32_t num_experts, uint32_t num_local_experts,
    void* stream);

// Weighted inverse permutation after vendor grouped GEMM. routing_weights is
// contiguous F32 [tokens, top_k]. expert_offsets[-1] supplies the valid local
// assignment count so expert-parallel remote routes contribute zero.
int loom_cuda_moe_combine_f32(
    const float* expert_outputs, const float* routing_weights,
    const int32_t* inverse_permutation, const int64_t* expert_offsets,
    float* output, uint32_t tokens, uint32_t hidden_size, uint32_t top_k,
    uint32_t num_local_experts, void* stream);

int loom_cuda_moe_combine_f16(
    const uint16_t* expert_outputs, const float* routing_weights,
    const int32_t* inverse_permutation, const int64_t* expert_offsets,
    uint16_t* output, uint32_t tokens, uint32_t hidden_size, uint32_t top_k,
    uint32_t num_local_experts, void* stream);

int loom_cuda_moe_combine_bf16(
    const uint16_t* expert_outputs, const float* routing_weights,
    const int32_t* inverse_permutation, const int64_t* expert_offsets,
    uint16_t* output, uint32_t tokens, uint32_t hidden_size, uint32_t top_k,
    uint32_t num_local_experts, void* stream);

// Deterministically sample one token from every contiguous normalized F32
// probability row. rng_state is mutable int64 [rows, 2] `(seed, counter)`
// state and token_ids is int64 [rows]. A valid row advances its counter once.
// Device values must satisfy the public contract; the kernel defensively
// leaves a row unchanged if its state or probabilities are invalid.
int loom_cuda_categorical_sample_f32(
    const float* probabilities, int64_t* rng_state, int64_t* token_ids,
    uint32_t rows, uint32_t vocab_size, void* stream);

// Fused greedy argmax and sampled-token logprob over logical
// [rows, vocab_size] logits with a unit vocabulary stride and explicit row
// stride. Token IDs use first-index tie breaking, logprobs are F32, and ranks
// are int64 counts of tokens tied at the maximum, matching vLLM's greater-than
// or-equal rank semantics. Logits must be finite.
int loom_cuda_greedy_sample_logprobs_f32(
    const float* logits, int32_t* token_ids, float* logprobs, int64_t* ranks,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride, void* stream);

int loom_cuda_greedy_sample_logprobs_f16(
    const uint16_t* logits, int32_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

int loom_cuda_greedy_sample_logprobs_bf16(
    const uint16_t* logits, int32_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

// Compute only the normalized logprob and tie-aware rank of one caller-selected
// token per row. This preserves engine-owned sampling policies while avoiding
// a full [rows, vocab_size] F32 log-softmax output. token_ids are int64 engine
// metadata and must be in [0, vocab_size); logits must be finite.
int loom_cuda_selected_token_logprobs_f32(
    const float* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

int loom_cuda_selected_token_logprobs_f16(
    const uint16_t* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

int loom_cuda_selected_token_logprobs_bf16(
    const uint16_t* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

// Return sampled-token plus top-k normalized logprobs without materializing a
// full-vocabulary F32 log-softmax. Each output row has width top_k + 1 and
// starts with the sampled token. Top-k ties use ascending token IDs; sampled
// ranks count logits greater than or equal to the sampled value. Workspace
// must be aligned to at least four bytes.
int loom_cuda_topk_sampled_logprobs_f32(
    const float* logits, const int64_t* sampled_token_ids,
    int32_t* output_token_ids, float* output_logprobs,
    int64_t* sampled_token_ranks, uint32_t rows, uint32_t vocab_size,
    uint32_t top_k, uint64_t row_stride, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t partitions, void* stream);

int loom_cuda_topk_sampled_logprobs_f16(
    const uint16_t* logits, const int64_t* sampled_token_ids,
    int32_t* output_token_ids, float* output_logprobs,
    int64_t* sampled_token_ranks, uint32_t rows, uint32_t vocab_size,
    uint32_t top_k, uint64_t row_stride, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t partitions, void* stream);

int loom_cuda_topk_sampled_logprobs_bf16(
    const uint16_t* logits, const int64_t* sampled_token_ids,
    int32_t* output_token_ids, float* output_logprobs,
    int64_t* sampled_token_ranks, uint32_t rows, uint32_t vocab_size,
    uint32_t top_k, uint64_t row_stride, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t partitions, void* stream);

// Apply an exact per-row top-k threshold to logits in place. top_ks is one
// contiguous int32 value per row in [1, vocab_size]. Values strictly below
// the kth largest value become negative infinity; threshold ties are retained.
// Device metadata values are trusted and logits must not contain NaNs.
// workspace contains partition-sorted uint32 radix keys and threshold keys.
int loom_cuda_top_k_filter_f32(
    float* logits, const int32_t* top_ks, uint32_t* workspace,
    uint64_t workspace_elements, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, uint32_t partitions, void* stream);

int loom_cuda_top_k_filter_f16(
    uint16_t* logits, const int32_t* top_ks, uint32_t* workspace,
    uint64_t workspace_elements, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, uint32_t partitions, void* stream);

int loom_cuda_top_k_filter_bf16(
    uint16_t* logits, const int32_t* top_ks, uint32_t* workspace,
    uint64_t workspace_elements, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, uint32_t partitions, void* stream);

// Apply exact per-row top-p filtering and return F32 probabilities
// renormalized over the retained set. Ties use descending token IDs. top_ps
// contains one trusted value in (0, 1] per row. Logits may contain -infinity,
// but each row must contain a finite value and neither NaN nor +infinity.
int loom_cuda_top_p_renorm_f32(
    float* logits, const float* top_ps, float* probabilities,
    uint8_t* workspace, uint64_t workspace_bytes, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, uint32_t partitions,
    void* stream);

int loom_cuda_top_p_renorm_f16(
    uint16_t* logits, const float* top_ps, float* probabilities,
    uint8_t* workspace, uint64_t workspace_bytes, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, uint32_t partitions,
    void* stream);

int loom_cuda_top_p_renorm_bf16(
    uint16_t* logits, const float* top_ps, float* probabilities,
    uint8_t* workspace, uint64_t workspace_bytes, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, uint32_t partitions,
    void* stream);

// Apply repetition once to the prompt/output union, then frequency and
// presence penalties from output counts. Token IDs outside [0, vocab_size)
// are padding. workspace is [rows, workspace_capacity] packed uint64 hash
// storage; capacity must be a power of two at least twice the combined padded
// prompt/output width.
int loom_cuda_apply_token_penalties_f32(
    float* logits, const int64_t* prompt_token_ids,
    const int64_t* output_token_ids, const float* presence_penalties,
    const float* frequency_penalties, const float* repetition_penalties,
    uint64_t* workspace, uint32_t rows, uint32_t vocab_size,
    uint32_t prompt_tokens, uint32_t output_tokens,
    uint32_t workspace_capacity, uint64_t logits_row_stride,
    uint64_t prompt_row_stride, uint64_t output_row_stride,
    uint64_t workspace_row_stride, void* stream);

// Deterministic greedy speculative verification over flattened ragged draft
// tokens. cumulative_draft_lengths is inclusive and has one int32 entry per
// request. Output is contiguous [requests, max_draft_tokens + 1], padded with
// -1. Accepted and emitted lengths contain one int32 value per request.
int loom_cuda_greedy_speculative_verify(
    const int32_t* draft_token_ids, const int64_t* target_token_ids,
    const int32_t* bonus_token_ids,
    const int32_t* cumulative_draft_lengths, int32_t* output_token_ids,
    int32_t* accepted_lengths, int32_t* emitted_lengths, uint32_t requests,
    uint32_t draft_tokens, uint32_t max_draft_tokens, void* stream);

// Apply one F32 logits-preprocessing pass in place. blocked_mask is optional,
// contiguous [rows, vocab_size] uint8 storage where nonzero means suppressed.
// Sparse bias arrays are an optional equal-length triplet; sparse suppression
// arrays are an optional equal-length pair. Sparse row/token values and unique
// bias targets are trusted. Temperatures below 1e-5 use a divisor of one.
int loom_cuda_logits_preprocess_f32(
    float* logits, const float* temperatures, const uint8_t* blocked_mask,
    const int32_t* bias_row_ids, const int32_t* bias_token_ids,
    const float* bias_values, uint32_t bias_count,
    const int32_t* suppressed_row_ids,
    const int32_t* suppressed_token_ids, uint32_t suppression_count,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride, void* stream);

int loom_cuda_min_p_filter_f32(float* logits, const float* min_p,
                               uint32_t rows, uint32_t vocab_size,
                               uint64_t row_stride, void* stream);
int loom_cuda_min_p_filter_f16(uint16_t* logits, const float* min_p,
                               uint32_t rows, uint32_t vocab_size,
                               uint64_t row_stride, void* stream);
int loom_cuda_min_p_filter_bf16(uint16_t* logits, const float* min_p,
                                uint32_t rows, uint32_t vocab_size,
                                uint64_t row_stride, void* stream);

// Base paged MQA/GQA decode attention for one query token per sequence.
// Query/output are contiguous [sequences, query_heads, dim]; native K/V
// caches have dense inner NHD [block_size, kv_heads, dim] dimensions and an
// explicit element stride between blocks. This accepts both separate caches
// and K/V views of vLLM's interleaved [blocks, 2, block_size, kv_heads, dim]
// storage. Block tables and sequence lengths are contiguous int32 engine
// metadata. Sequence lengths include the current token and are trusted to be
// in [1, max_sequence_length]; active block IDs are trusted to be in
// [0, num_blocks). This kernel family is intentionally limited to
// max_sequence_length <= 1024 and does not implement ALiBi, sliding windows,
// soft caps, quantized KV, or multi-token queries.
int loom_cuda_paged_decode_attention_f32(
    const float* query, const float* key_cache, const float* value_cache,
    const int32_t* block_tables, const int32_t* sequence_lengths,
    float* output, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream);

int loom_cuda_paged_decode_attention_f16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, uint32_t sequences,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t num_blocks, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream);

int loom_cuda_paged_decode_attention_bf16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, uint32_t sequences,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t num_blocks, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream);

// Optional long-context split-K path. The sizing function returns zero when
// the shape should use the base ABI above. Otherwise the caller owns an F32
// workspace with at least the returned element count for the complete pair of
// partial and stable log-sum-exp merge kernels. The original entry points stay
// allocation-free and ABI-compatible.
uint64_t loom_cuda_paged_decode_attention_split_k_workspace_elements(
    uint32_t sequences, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size,
    uint32_t max_sequence_length);

int loom_cuda_paged_decode_attention_split_k_f32(
    const float* query, const float* key_cache, const float* value_cache,
    const int32_t* block_tables, const int32_t* sequence_lengths,
    float* output, float* workspace, uint64_t workspace_elements,
    uint32_t sequences, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size, uint32_t num_blocks,
    uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream);

int loom_cuda_paged_decode_attention_split_k_f16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, float* workspace,
    uint64_t workspace_elements, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream);

int loom_cuda_paged_decode_attention_split_k_bf16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, float* workspace,
    uint64_t workspace_elements, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream);

// Fused in-place RoPE and paged K/V cache write. Query, key, and value have
// logical [tokens, heads, dim] dimensions, a unit dim stride, and explicit
// token/head element strides so packed-QKV views do not need materialization.
// The cosine/sine cache is contiguous [max_position, rotary_dim], with cosine
// then sine halves. Key/value cache tensors have logical
// [blocks, block_size, kv_heads, dim] dimensions; their element strides make
// both vLLM NHD and HND physical layouts expressible. cache_tokens may be less
// than tokens when the engine pads Q/K/V but not slot_mapping. FP8 E4M3 cache
// storage uses one scale for all heads (scale_stride=0) or one scale per KV
// head (scale_stride=1). Negative slots skip the cache write while RoPE still
// updates Q/K. Positions and non-negative slots are trusted engine metadata
// and must be in range.
int loom_cuda_rope_paged_kv_write_f32(
    float* query, float* key, const float* value, const int64_t* positions,
    const float* cos_sin_cache, void* key_cache, void* value_cache,
    const float* key_scales, const float* value_scales,
    const int64_t* slot_mapping, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint32_t cache_encoding,
    uint32_t scale_stride, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream);

int loom_cuda_rope_paged_kv_write_f16(
    uint16_t* query, uint16_t* key, const uint16_t* value,
    const int64_t* positions, const uint16_t* cos_sin_cache,
    void* key_cache, void* value_cache, const float* key_scales,
    const float* value_scales, const int64_t* slot_mapping, uint32_t tokens,
    uint32_t cache_tokens, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size, uint32_t rotary_dim,
    uint32_t max_position, uint32_t num_blocks, uint32_t block_size,
    uint32_t cache_encoding, uint32_t scale_stride,
    uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream);

int loom_cuda_rope_paged_kv_write_bf16(
    uint16_t* query, uint16_t* key, const uint16_t* value,
    const int64_t* positions, const uint16_t* cos_sin_cache,
    void* key_cache, void* value_cache, const float* key_scales,
    const float* value_scales, const int64_t* slot_mapping, uint32_t tokens,
    uint32_t cache_tokens, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size, uint32_t rotary_dim,
    uint32_t max_position, uint32_t num_blocks, uint32_t block_size,
    uint32_t cache_encoding, uint32_t scale_stride,
    uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_CUDA_H_
