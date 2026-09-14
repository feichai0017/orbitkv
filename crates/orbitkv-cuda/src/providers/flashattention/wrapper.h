#pragma once

#include <stdint.h>

// Provider-owned scratch is distinct from graph inputs and OrbitKV storage.
struct OrbitKVFlashAttentionLaunch {
    const void* query;
    const void* key;
    const void* value;
    const int32_t* page_indices;
    const int32_t* query_indptr;
    const int32_t* page_indptr;
    const int32_t* last_page_len;
    void* output;
    int32_t* page_table;
    int32_t* kv_lengths;
    int32_t* split_counts;
    int32_t* query_tiles;
    int32_t* batch_order;
    int32_t* head_swizzle;
    int32_t* tile_counter;
    float* lse;
    int32_t query_tokens;
    int32_t requests;
    int32_t query_heads;
    int32_t kv_heads;
    int32_t page_size;
    int32_t context_pages;
    int32_t cache_pages;
    int32_t num_sm;
    float scale;
    int32_t window_left;
};

extern "C" int orbitkv_flashattention_run(const OrbitKVFlashAttentionLaunch*, void* stream) noexcept;
extern "C" const char* orbitkv_flashattention_last_error() noexcept;
