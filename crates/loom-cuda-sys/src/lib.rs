//! Raw bindings to the dependency-light Loom Kernels CUDA C ABI.

use std::ffi::c_int;
#[cfg(feature = "cuda")]
use std::ffi::{c_char, c_void};

pub const LOOM_CUDA_SUCCESS: c_int = 0;
pub const LOOM_CUDA_INVALID_ARGUMENT: c_int = 1;
pub const LOOM_CUDA_UNSUPPORTED: c_int = 2;
pub const LOOM_CUDA_LAUNCH_ERROR: c_int = 3;
pub const LOOM_CUDA_UNAVAILABLE: c_int = 4;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_STREAM_NON_BLOCKING: u32 = 1;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn loom_cuda_status_string(status: c_int) -> *const c_char;

    pub fn loom_cuda_rms_norm_f32(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_f16(
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_bf16(
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_fp8_f32(
        input: *const f32,
        weight: *const f32,
        residual: *mut f32,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_fp8_f16(
        input: *const u16,
        weight: *const u16,
        residual: *mut u16,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_fp8_bf16(
        input: *const u16,
        weight: *const u16,
        residual: *mut u16,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_int8_f32(
        input: *const f32,
        weight: *const f32,
        residual: *mut f32,
        output: *mut i8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_int8_f16(
        input: *const u16,
        weight: *const u16,
        residual: *mut u16,
        output: *mut i8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rms_norm_dynamic_int8_bf16(
        input: *const u16,
        weight: *const u16,
        residual: *mut u16,
        output: *mut i8,
        scales: *mut f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_add_rms_norm_f32(
        input: *mut f32,
        residual: *mut f32,
        weight: *const f32,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_add_rms_norm_f16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_add_rms_norm_bf16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_f16(
        input: *const u16,
        output: *mut u16,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_bf16(
        input: *const u16,
        output: *mut u16,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_dynamic_fp8_f16(
        input: *const u16,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        width: u32,
        group_size: u32,
        scale_ub: *const f32,
        scales_transposed: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_dynamic_fp8_bf16(
        input: *const u16,
        output: *mut u8,
        scales: *mut f32,
        rows: u32,
        width: u32,
        group_size: u32,
        scale_ub: *const f32,
        scales_transposed: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_dynamic_int8_f16(
        input: *const u16,
        output: *mut i8,
        scales: *mut f32,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_silu_and_mul_dynamic_int8_bf16(
        input: *const u16,
        output: *mut i8,
        scales: *mut f32,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_moe_permute_workspace_size(
        assignments: u32,
        num_experts: u32,
        workspace_bytes: *mut u64,
    ) -> c_int;

    pub fn loom_cuda_moe_permute_f32(
        hidden_states: *const f32,
        topk_ids: *const i32,
        expert_map: *const i32,
        permuted_hidden_states: *mut f32,
        expert_offsets: *mut i64,
        inverse_permutation: *mut i32,
        permuted_assignment_ids: *mut i32,
        workspace: *mut u8,
        workspace_bytes: u64,
        tokens: u32,
        hidden_size: u32,
        top_k: u32,
        num_experts: u32,
        num_local_experts: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_moe_permute_f16(
        hidden_states: *const u16,
        topk_ids: *const i32,
        expert_map: *const i32,
        permuted_hidden_states: *mut u16,
        expert_offsets: *mut i64,
        inverse_permutation: *mut i32,
        permuted_assignment_ids: *mut i32,
        workspace: *mut u8,
        workspace_bytes: u64,
        tokens: u32,
        hidden_size: u32,
        top_k: u32,
        num_experts: u32,
        num_local_experts: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_moe_permute_bf16(
        hidden_states: *const u16,
        topk_ids: *const i32,
        expert_map: *const i32,
        permuted_hidden_states: *mut u16,
        expert_offsets: *mut i64,
        inverse_permutation: *mut i32,
        permuted_assignment_ids: *mut i32,
        workspace: *mut u8,
        workspace_bytes: u64,
        tokens: u32,
        hidden_size: u32,
        top_k: u32,
        num_experts: u32,
        num_local_experts: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_moe_permute_fp8_e4m3fn(
        hidden_states: *const u8,
        topk_ids: *const i32,
        expert_map: *const i32,
        permuted_hidden_states: *mut u8,
        expert_offsets: *mut i64,
        inverse_permutation: *mut i32,
        permuted_assignment_ids: *mut i32,
        workspace: *mut u8,
        workspace_bytes: u64,
        tokens: u32,
        hidden_size: u32,
        top_k: u32,
        num_experts: u32,
        num_local_experts: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_moe_combine_f32(
        expert_outputs: *const f32,
        routing_weights: *const f32,
        inverse_permutation: *const i32,
        expert_offsets: *const i64,
        output: *mut f32,
        tokens: u32,
        hidden_size: u32,
        top_k: u32,
        num_local_experts: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_moe_combine_f16(
        expert_outputs: *const u16,
        routing_weights: *const f32,
        inverse_permutation: *const i32,
        expert_offsets: *const i64,
        output: *mut u16,
        tokens: u32,
        hidden_size: u32,
        top_k: u32,
        num_local_experts: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_moe_combine_bf16(
        expert_outputs: *const u16,
        routing_weights: *const f32,
        inverse_permutation: *const i32,
        expert_offsets: *const i64,
        output: *mut u16,
        tokens: u32,
        hidden_size: u32,
        top_k: u32,
        num_local_experts: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_categorical_sample_f32(
        probabilities: *const f32,
        rng_state: *mut i64,
        token_ids: *mut i64,
        rows: u32,
        vocab_size: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_greedy_sample_logprobs_f32(
        logits: *const f32,
        token_ids: *mut i32,
        logprobs: *mut f32,
        ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_greedy_sample_logprobs_f16(
        logits: *const u16,
        token_ids: *mut i32,
        logprobs: *mut f32,
        ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_greedy_sample_logprobs_bf16(
        logits: *const u16,
        token_ids: *mut i32,
        logprobs: *mut f32,
        ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_selected_token_logprobs_f32(
        logits: *const f32,
        token_ids: *const i64,
        logprobs: *mut f32,
        ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_selected_token_logprobs_f16(
        logits: *const u16,
        token_ids: *const i64,
        logprobs: *mut f32,
        ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_selected_token_logprobs_bf16(
        logits: *const u16,
        token_ids: *const i64,
        logprobs: *mut f32,
        ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_topk_sampled_logprobs_f32(
        logits: *const f32,
        sampled_token_ids: *const i64,
        output_token_ids: *mut i32,
        output_logprobs: *mut f32,
        sampled_token_ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        top_k: u32,
        row_stride: u64,
        workspace: *mut u8,
        workspace_bytes: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_topk_sampled_logprobs_f16(
        logits: *const u16,
        sampled_token_ids: *const i64,
        output_token_ids: *mut i32,
        output_logprobs: *mut f32,
        sampled_token_ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        top_k: u32,
        row_stride: u64,
        workspace: *mut u8,
        workspace_bytes: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_topk_sampled_logprobs_bf16(
        logits: *const u16,
        sampled_token_ids: *const i64,
        output_token_ids: *mut i32,
        output_logprobs: *mut f32,
        sampled_token_ranks: *mut i64,
        rows: u32,
        vocab_size: u32,
        top_k: u32,
        row_stride: u64,
        workspace: *mut u8,
        workspace_bytes: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_top_k_filter_f32(
        logits: *mut f32,
        top_ks: *const i32,
        workspace: *mut u32,
        workspace_elements: u64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_top_k_filter_f16(
        logits: *mut u16,
        top_ks: *const i32,
        workspace: *mut u32,
        workspace_elements: u64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_top_k_filter_bf16(
        logits: *mut u16,
        top_ks: *const i32,
        workspace: *mut u32,
        workspace_elements: u64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_top_p_renorm_f32(
        logits: *mut f32,
        top_ps: *const f32,
        probabilities: *mut f32,
        workspace: *mut u8,
        workspace_bytes: u64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_top_p_renorm_f16(
        logits: *mut u16,
        top_ps: *const f32,
        probabilities: *mut f32,
        workspace: *mut u8,
        workspace_bytes: u64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_top_p_renorm_bf16(
        logits: *mut u16,
        top_ps: *const f32,
        probabilities: *mut f32,
        workspace: *mut u8,
        workspace_bytes: u64,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        partitions: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_apply_token_penalties_f32(
        logits: *mut f32,
        prompt_token_ids: *const i64,
        output_token_ids: *const i64,
        presence_penalties: *const f32,
        frequency_penalties: *const f32,
        repetition_penalties: *const f32,
        workspace: *mut u64,
        rows: u32,
        vocab_size: u32,
        prompt_tokens: u32,
        output_tokens: u32,
        workspace_capacity: u32,
        logits_row_stride: u64,
        prompt_row_stride: u64,
        output_row_stride: u64,
        workspace_row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_greedy_speculative_verify(
        draft_token_ids: *const i32,
        target_token_ids: *const i64,
        bonus_token_ids: *const i32,
        cumulative_draft_lengths: *const i32,
        output_token_ids: *mut i32,
        accepted_lengths: *mut i32,
        emitted_lengths: *mut i32,
        requests: u32,
        draft_tokens: u32,
        max_draft_tokens: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_logits_preprocess_f32(
        logits: *mut f32,
        temperatures: *const f32,
        blocked_mask: *const u8,
        bias_row_ids: *const i32,
        bias_token_ids: *const i32,
        bias_values: *const f32,
        bias_count: u32,
        suppressed_row_ids: *const i32,
        suppressed_token_ids: *const i32,
        suppression_count: u32,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_min_p_filter_f32(
        logits: *mut f32,
        min_p: *const f32,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_min_p_filter_f16(
        logits: *mut u16,
        min_p: *const f32,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_min_p_filter_bf16(
        logits: *mut u16,
        min_p: *const f32,
        rows: u32,
        vocab_size: u32,
        row_stride: u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_paged_decode_attention_f32(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        block_tables: *const i32,
        sequence_lengths: *const i32,
        output: *mut f32,
        sequences: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        num_blocks: u32,
        block_size: u32,
        key_block_stride: u64,
        value_block_stride: u64,
        max_blocks_per_sequence: u32,
        max_sequence_length: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_paged_decode_attention_f16(
        query: *const u16,
        key_cache: *const u16,
        value_cache: *const u16,
        block_tables: *const i32,
        sequence_lengths: *const i32,
        output: *mut u16,
        sequences: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        num_blocks: u32,
        block_size: u32,
        key_block_stride: u64,
        value_block_stride: u64,
        max_blocks_per_sequence: u32,
        max_sequence_length: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_paged_decode_attention_bf16(
        query: *const u16,
        key_cache: *const u16,
        value_cache: *const u16,
        block_tables: *const i32,
        sequence_lengths: *const i32,
        output: *mut u16,
        sequences: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        num_blocks: u32,
        block_size: u32,
        key_block_stride: u64,
        value_block_stride: u64,
        max_blocks_per_sequence: u32,
        max_sequence_length: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_paged_decode_attention_split_k_workspace_elements(
        sequences: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        max_sequence_length: u32,
    ) -> u64;

    pub fn loom_cuda_paged_decode_attention_split_k_f32(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        block_tables: *const i32,
        sequence_lengths: *const i32,
        output: *mut f32,
        workspace: *mut f32,
        workspace_elements: u64,
        sequences: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        num_blocks: u32,
        block_size: u32,
        key_block_stride: u64,
        value_block_stride: u64,
        max_blocks_per_sequence: u32,
        max_sequence_length: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_paged_decode_attention_split_k_f16(
        query: *const u16,
        key_cache: *const u16,
        value_cache: *const u16,
        block_tables: *const i32,
        sequence_lengths: *const i32,
        output: *mut u16,
        workspace: *mut f32,
        workspace_elements: u64,
        sequences: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        num_blocks: u32,
        block_size: u32,
        key_block_stride: u64,
        value_block_stride: u64,
        max_blocks_per_sequence: u32,
        max_sequence_length: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_paged_decode_attention_split_k_bf16(
        query: *const u16,
        key_cache: *const u16,
        value_cache: *const u16,
        block_tables: *const i32,
        sequence_lengths: *const i32,
        output: *mut u16,
        workspace: *mut f32,
        workspace_elements: u64,
        sequences: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        num_blocks: u32,
        block_size: u32,
        key_block_stride: u64,
        value_block_stride: u64,
        max_blocks_per_sequence: u32,
        max_sequence_length: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rope_paged_kv_write_f32(
        query: *mut f32,
        key: *mut f32,
        value: *const f32,
        positions: *const i64,
        cos_sin_cache: *const f32,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        key_scales: *const f32,
        value_scales: *const f32,
        slot_mapping: *const i64,
        tokens: u32,
        cache_tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        rotary_dim: u32,
        max_position: u32,
        num_blocks: u32,
        block_size: u32,
        cache_encoding: u32,
        scale_stride: u32,
        query_token_stride: u64,
        query_head_stride: u64,
        key_token_stride: u64,
        key_head_stride: u64,
        value_token_stride: u64,
        value_head_stride: u64,
        key_cache_block_stride: u64,
        key_cache_page_stride: u64,
        key_cache_head_stride: u64,
        value_cache_block_stride: u64,
        value_cache_page_stride: u64,
        value_cache_head_stride: u64,
        is_neox: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rope_paged_kv_write_f16(
        query: *mut u16,
        key: *mut u16,
        value: *const u16,
        positions: *const i64,
        cos_sin_cache: *const u16,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        key_scales: *const f32,
        value_scales: *const f32,
        slot_mapping: *const i64,
        tokens: u32,
        cache_tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        rotary_dim: u32,
        max_position: u32,
        num_blocks: u32,
        block_size: u32,
        cache_encoding: u32,
        scale_stride: u32,
        query_token_stride: u64,
        query_head_stride: u64,
        key_token_stride: u64,
        key_head_stride: u64,
        value_token_stride: u64,
        value_head_stride: u64,
        key_cache_block_stride: u64,
        key_cache_page_stride: u64,
        key_cache_head_stride: u64,
        value_cache_block_stride: u64,
        value_cache_page_stride: u64,
        value_cache_head_stride: u64,
        is_neox: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_rope_paged_kv_write_bf16(
        query: *mut u16,
        key: *mut u16,
        value: *const u16,
        positions: *const i64,
        cos_sin_cache: *const u16,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        key_scales: *const f32,
        value_scales: *const f32,
        slot_mapping: *const i64,
        tokens: u32,
        cache_tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        head_size: u32,
        value_head_size: u32,
        rotary_dim: u32,
        max_position: u32,
        num_blocks: u32,
        block_size: u32,
        cache_encoding: u32,
        scale_stride: u32,
        query_token_stride: u64,
        query_head_stride: u64,
        key_token_stride: u64,
        key_head_stride: u64,
        value_token_stride: u64,
        value_head_stride: u64,
        key_cache_block_stride: u64,
        key_cache_page_stride: u64,
        key_cache_head_stride: u64,
        value_cache_block_stride: u64,
        value_cache_page_stride: u64,
        value_cache_head_stride: u64,
        is_neox: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    pub fn cudaFree(pointer: *mut c_void) -> c_int;
    pub fn cudaMemcpy(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        kind: c_int,
    ) -> c_int;
    pub fn cudaGetErrorString(error: c_int) -> *const c_char;
    pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> c_int;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    pub fn cudaEventCreate(event: *mut *mut c_void) -> c_int;
    pub fn cudaEventDestroy(event: *mut c_void) -> c_int;
    pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    pub fn cudaEventSynchronize(event: *mut c_void) -> c_int;
    pub fn cudaEventElapsedTime(
        milliseconds: *mut f32,
        start: *mut c_void,
        end: *mut c_void,
    ) -> c_int;
}

pub const fn compiled_with_cuda() -> bool {
    cfg!(feature = "cuda")
}
