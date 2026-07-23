#include "common.h"

namespace loom_kernels::torch_adapter {

void check_rope_paged_kv_write_contract(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& positions, const Tensor& cos_sin_cache,
    const Tensor& key_cache, const Tensor& value_cache,
    const Tensor& key_scales, const Tensor& value_scales,
    const Tensor& slot_mapping) {
  STD_TORCH_CHECK(query.is_cuda(), "Loom RoPE+paged-KV query must be CUDA");
  STD_TORCH_CHECK(key.device() == query.device() &&
                  value.device() == query.device() &&
                  positions.device() == query.device() &&
                  cos_sin_cache.device() == query.device() &&
                  key_cache.device() == query.device() &&
                  value_cache.device() == query.device() &&
                  key_scales.device() == query.device() &&
                  value_scales.device() == query.device() &&
                  slot_mapping.device() == query.device(),
              "Loom RoPE+paged-KV tensors must be on one CUDA device");
  STD_TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == cos_sin_cache.scalar_type(),
              "Loom RoPE+paged-KV Q/K/V and cos/sin cache must share a dtype");
  STD_TORCH_CHECK(query.scalar_type() == ScalarType::Float ||
                  query.scalar_type() == ScalarType::Half ||
                  query.scalar_type() == ScalarType::BFloat16,
              "Loom RoPE+paged-KV supports F32, FP16, and BF16 sources");
  STD_TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "Loom RoPE+paged-KV Q/K/V must have rank 3");
  const bool native_cache =
      key_cache.scalar_type() == query.scalar_type() &&
      value_cache.scalar_type() == query.scalar_type();
  const bool fp8_cache =
      key_cache.scalar_type() == ScalarType::Byte &&
      value_cache.scalar_type() == ScalarType::Byte;
  STD_TORCH_CHECK(native_cache || fp8_cache,
              "Loom paged K/V caches must use the source dtype or uint8 "
              "FP8 E4M3 storage");
  STD_TORCH_CHECK(key_scales.scalar_type() == ScalarType::Float &&
                  value_scales.scalar_type() == ScalarType::Float &&
                  key_scales.is_contiguous() && value_scales.is_contiguous() &&
                  key_scales.numel() == value_scales.numel() &&
                  (key_scales.numel() == 1 ||
                   key_scales.numel() == key.size(1)),
              "Loom paged K/V cache scales must be contiguous float32 "
              "tensors with one element or one element per KV head");
  STD_TORCH_CHECK(positions.scalar_type() == ScalarType::Long &&
                  slot_mapping.scalar_type() == ScalarType::Long,
              "Loom RoPE+paged-KV positions and slot mapping must be int64");
  STD_TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0,
              "Loom RoPE+paged-KV query must be non-empty");
  STD_TORCH_CHECK(key.size(0) == query.size(0) &&
                  value.size(0) == query.size(0),
              "Loom RoPE+paged-KV Q/K/V token counts must match");
  STD_TORCH_CHECK(key.size(1) > 0 && key.size(1) == value.size(1),
              "Loom RoPE+paged-KV K/V head counts must match");
  STD_TORCH_CHECK(key.size(2) == query.size(2),
              "Loom RoPE+paged-KV Q/K head sizes must match");
  STD_TORCH_CHECK(value.size(2) > 0,
              "Loom RoPE+paged-KV value head size must be positive");
  STD_TORCH_CHECK(query.stride(2) == 1 && key.stride(2) == 1 &&
                  value.stride(2) == 1 && query.stride(0) > 0 &&
                  query.stride(1) > 0 && key.stride(0) > 0 &&
                  key.stride(1) > 0 && value.stride(0) > 0 &&
                  value.stride(1) > 0 && positions.is_contiguous() &&
                  cos_sin_cache.is_contiguous() &&
                  slot_mapping.is_contiguous(),
              "Loom RoPE+paged-KV sources require unit dim stride and positive "
              "token/head strides; metadata must be contiguous");
  STD_TORCH_CHECK(positions.dim() == 1 &&
                  positions.numel() == query.size(0) &&
                  slot_mapping.dim() == 1 &&
                  slot_mapping.numel() <= query.size(0),
              "Loom RoPE positions must cover every token and slot_mapping "
              "must not exceed the padded token count");
  STD_TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(0) > 0 &&
                  cos_sin_cache.size(1) > 0 &&
                  cos_sin_cache.size(1) % 2 == 0 &&
                  cos_sin_cache.size(1) <= query.size(2),
              "Loom RoPE+paged-KV cos/sin cache must be "
              "[max_position, even rotary_dim <= head_size]");
  STD_TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "Loom paged K/V cache views must have rank 4");
  STD_TORCH_CHECK(key_cache.size(0) > 0 && key_cache.size(1) > 0 &&
                  key_cache.size(2) == key.size(1) &&
                  key_cache.size(3) == key.size(2),
              "Loom key cache must have logical shape "
              "[blocks, block_size, kv_heads, head_size]");
  STD_TORCH_CHECK(value_cache.size(0) == key_cache.size(0) &&
                  value_cache.size(1) == key_cache.size(1) &&
                  value_cache.size(2) == value.size(1) &&
                  value_cache.size(3) == value.size(2),
              "Loom value cache must have logical shape "
              "[blocks, block_size, kv_heads, value_head_size]");
  STD_TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
                  key_cache.stride(0) > 0 && key_cache.stride(1) > 0 &&
                  key_cache.stride(2) > 0 && value_cache.stride(0) > 0 &&
                  value_cache.stride(1) > 0 && value_cache.stride(2) > 0,
              "Loom paged K/V caches require unit dim stride and positive "
              "block/page/head strides");
}

void launch_rope_paged_kv_write(
    Tensor query, Tensor key, const Tensor& value,
    const Tensor& positions, const Tensor& cos_sin_cache,
    Tensor key_cache, Tensor value_cache,
    const Tensor& key_scales, const Tensor& value_scales,
    const Tensor& slot_mapping, bool is_neox) {
  const int64_t limits[] = {
      query.size(0),       query.size(1),       key.size(1),
      query.size(2),       value.size(2),       cos_sin_cache.size(1),
      cos_sin_cache.size(0), key_cache.size(0), key_cache.size(1),
  };
  for (const int64_t value_to_check : limits) {
    STD_TORCH_CHECK(value_to_check > 0 &&
                    value_to_check <= std::numeric_limits<uint32_t>::max(),
                "Loom RoPE+paged-KV shape exceeds the CUDA ABI");
  }

  const CudaDeviceGuard device_guard(query.device());
  const auto stream = current_cuda_stream(query.device().index());
  const auto tokens = static_cast<uint32_t>(query.size(0));
  const auto cache_tokens = static_cast<uint32_t>(slot_mapping.numel());
  const auto query_heads = static_cast<uint32_t>(query.size(1));
  const auto kv_heads = static_cast<uint32_t>(key.size(1));
  const auto head_size = static_cast<uint32_t>(query.size(2));
  const auto value_head_size = static_cast<uint32_t>(value.size(2));
  const auto rotary_dim = static_cast<uint32_t>(cos_sin_cache.size(1));
  const auto max_position = static_cast<uint32_t>(cos_sin_cache.size(0));
  const auto num_blocks = static_cast<uint32_t>(key_cache.size(0));
  const auto block_size = static_cast<uint32_t>(key_cache.size(1));
  const auto query_token_stride = static_cast<uint64_t>(query.stride(0));
  const auto query_head_stride = static_cast<uint64_t>(query.stride(1));
  const auto key_token_stride = static_cast<uint64_t>(key.stride(0));
  const auto key_head_stride = static_cast<uint64_t>(key.stride(1));
  const auto value_token_stride = static_cast<uint64_t>(value.stride(0));
  const auto value_head_stride = static_cast<uint64_t>(value.stride(1));
  const auto key_block_stride = static_cast<uint64_t>(key_cache.stride(0));
  const auto key_page_stride = static_cast<uint64_t>(key_cache.stride(1));
  const auto key_cache_head_stride =
      static_cast<uint64_t>(key_cache.stride(2));
  const auto value_block_stride =
      static_cast<uint64_t>(value_cache.stride(0));
  const auto value_page_stride =
      static_cast<uint64_t>(value_cache.stride(1));
  const auto value_cache_head_stride =
      static_cast<uint64_t>(value_cache.stride(2));
  const uint32_t cache_encoding =
      key_cache.scalar_type() == ScalarType::Byte
          ? LOOM_CUDA_BRIDGE_KV_CACHE_FP8_E4M3
          : LOOM_CUDA_BRIDGE_KV_CACHE_NATIVE;

  const int status = loom_cuda_bridge_rope_paged_kv_write(
      bridge_dtype(query), cache_encoding, query.mutable_data_ptr(),
      storage_span_elements(query), key.mutable_data_ptr(),
      storage_span_elements(key), value.const_data_ptr(),
      storage_span_elements(value), positions.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(positions.numel()),
      cos_sin_cache.const_data_ptr(),
      static_cast<uint64_t>(cos_sin_cache.numel()),
      key_cache.mutable_data_ptr(), storage_span_elements(key_cache),
      value_cache.mutable_data_ptr(), storage_span_elements(value_cache),
      key_scales.const_data_ptr<float>(),
      static_cast<uint64_t>(key_scales.numel()),
      value_scales.const_data_ptr<float>(),
      static_cast<uint64_t>(value_scales.numel()),
      slot_mapping.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(slot_mapping.numel()), tokens, cache_tokens,
      query_heads, kv_heads, head_size, value_head_size, rotary_dim,
      max_position, num_blocks, block_size, query_token_stride,
      query_head_stride, key_token_stride, key_head_stride,
      value_token_stride, value_head_stride, key_block_stride,
      key_page_stride, key_cache_head_stride, value_block_stride,
      value_page_stride, value_cache_head_stride, is_neox ? 1U : 0U,
      stream.stream());
  check_bridge_status(status, "RoPE+paged-KV");
}

void rope_paged_kv_write(
    Tensor query, Tensor key, const Tensor& value,
    const Tensor& positions, const Tensor& cos_sin_cache,
    Tensor key_cache, Tensor value_cache,
    const Tensor& key_scales, const Tensor& value_scales,
    const Tensor& slot_mapping, bool is_neox) {
  check_rope_paged_kv_write_contract(query, key, value, positions,
                                     cos_sin_cache, key_cache, value_cache,
                                     key_scales, value_scales, slot_mapping);
  launch_rope_paged_kv_write(query, key, value, positions, cos_sin_cache,
                             key_cache, value_cache, key_scales, value_scales,
                             slot_mapping, is_neox);
}


}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "rope_paged_kv_write_",
      TORCH_BOX(&loom_kernels::torch_adapter::rope_paged_kv_write));
}
