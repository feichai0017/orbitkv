#include "common.h"

namespace loom_kernels::torch_adapter {

void check_paged_decode_attention_contract(
    const Tensor& query, const Tensor& key_cache,
    const Tensor& value_cache, const Tensor& block_tables,
    const Tensor& sequence_lengths, const Tensor& output,
    int64_t max_sequence_length, double scale) {
  STD_TORCH_CHECK(query.is_cuda(), "Loom paged decode query must be CUDA");
  STD_TORCH_CHECK(key_cache.device() == query.device() &&
                  value_cache.device() == query.device() &&
                  block_tables.device() == query.device() &&
                  sequence_lengths.device() == query.device() &&
                  output.device() == query.device(),
              "Loom paged decode tensors must be on one CUDA device");
  STD_TORCH_CHECK(query.scalar_type() == ScalarType::Float ||
                  query.scalar_type() == ScalarType::Half ||
                  query.scalar_type() == ScalarType::BFloat16,
              "Loom paged decode supports F32, FP16, and BF16 native caches");
  STD_TORCH_CHECK(key_cache.scalar_type() == query.scalar_type() &&
                  value_cache.scalar_type() == query.scalar_type() &&
                  output.scalar_type() == query.scalar_type(),
              "Loom paged decode data tensors must share a dtype");
  STD_TORCH_CHECK(block_tables.scalar_type() == ScalarType::Int &&
                  sequence_lengths.scalar_type() == ScalarType::Int,
              "Loom paged decode metadata must use int32");
  STD_TORCH_CHECK(query.dim() == 3 && key_cache.dim() == 4 &&
                  value_cache.dim() == 4 && block_tables.dim() == 2 &&
                  sequence_lengths.dim() == 1 && output.dim() == 3,
              "Loom paged decode requires rank-3 query/output, rank-4 K/V "
              "caches, rank-2 block tables, and rank-1 sequence lengths");
  STD_TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0 &&
                  key_cache.size(0) > 0 && key_cache.size(1) > 0 &&
                  key_cache.size(2) > 0 && value_cache.size(3) > 0,
              "Loom paged decode dimensions must be positive");
  STD_TORCH_CHECK(key_cache.size(3) == query.size(2),
              "Loom paged decode Q/K head sizes must match");
  STD_TORCH_CHECK(value_cache.size(0) == key_cache.size(0) &&
                  value_cache.size(1) == key_cache.size(1) &&
                  value_cache.size(2) == key_cache.size(2),
              "Loom paged decode K/V cache prefixes must match");
  STD_TORCH_CHECK(query.size(1) % key_cache.size(2) == 0,
              "Loom paged decode query heads must be divisible by KV heads");
  STD_TORCH_CHECK(block_tables.size(0) == query.size(0) &&
                  block_tables.size(1) > 0 &&
                  sequence_lengths.size(0) == query.size(0),
              "Loom paged decode metadata batch dimensions must match query");
  STD_TORCH_CHECK(output.size(0) == query.size(0) &&
                  output.size(1) == query.size(1) &&
                  output.size(2) == value_cache.size(3),
              "Loom paged decode output must have shape [B, Hq, Dv]");
  STD_TORCH_CHECK(query.is_contiguous() &&
                  has_dense_nhd_inner_strides(key_cache) &&
                  has_dense_nhd_inner_strides(value_cache) &&
                  block_tables.is_contiguous() &&
                  sequence_lengths.is_contiguous() && output.is_contiguous(),
              "Loom paged decode requires contiguous query/output/metadata "
              "and dense-inner NHD caches with an optional block stride");
  STD_TORCH_CHECK(max_sequence_length > 0 && max_sequence_length <= 1024 &&
                  max_sequence_length <=
                      block_tables.size(1) * key_cache.size(1),
              "Loom paged decode max_sequence_length must be within table "
              "capacity and the first-kernel limit 1024");
  STD_TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "Loom paged decode scale must be finite and positive");
  STD_TORCH_CHECK(!byte_ranges_overlap(output, query) &&
                  !byte_ranges_overlap(output, key_cache) &&
                  !byte_ranges_overlap(output, value_cache) &&
                  !byte_ranges_overlap(output, block_tables) &&
                  !byte_ranges_overlap(output, sequence_lengths),
              "Loom paged decode output storage must not overlap inputs");

  const int64_t limits[] = {
      query.size(0),       query.size(1),      key_cache.size(2),
      query.size(2),       value_cache.size(3), key_cache.size(0),
      key_cache.size(1),   block_tables.size(1), max_sequence_length,
  };
  for (const int64_t value_to_check : limits) {
    STD_TORCH_CHECK(value_to_check > 0 &&
                    value_to_check <= std::numeric_limits<uint32_t>::max(),
                "Loom paged decode shape exceeds the CUDA ABI");
  }
  STD_TORCH_CHECK(query.size(0) <=
                  std::numeric_limits<int32_t>::max() / query.size(1),
              "Loom paged decode grid exceeds the CUDA ABI");
}

void launch_paged_decode_attention(
    const Tensor& query, const Tensor& key_cache,
    const Tensor& value_cache, const Tensor& block_tables,
    const Tensor& sequence_lengths, Tensor output,
    int64_t max_sequence_length, double scale) {
  const auto sequences = static_cast<uint32_t>(query.size(0));
  const auto query_heads = static_cast<uint32_t>(query.size(1));
  const auto kv_heads = static_cast<uint32_t>(key_cache.size(2));
  const auto head_size = static_cast<uint32_t>(query.size(2));
  const auto value_head_size = static_cast<uint32_t>(value_cache.size(3));
  const auto num_blocks = static_cast<uint32_t>(key_cache.size(0));
  const auto block_size = static_cast<uint32_t>(key_cache.size(1));
  const auto key_block_stride =
      static_cast<uint64_t>(key_cache.stride(0));
  const auto value_block_stride =
      static_cast<uint64_t>(value_cache.stride(0));
  const auto max_blocks_per_sequence =
      static_cast<uint32_t>(block_tables.size(1));
  const auto max_context = static_cast<uint32_t>(max_sequence_length);
  const auto scale_f32 = static_cast<float>(scale);
  const CudaDeviceGuard device_guard(query.device());
  const auto stream = current_cuda_stream(query.device().index());
  uint64_t split_k_workspace_elements = 0;
  int status = loom_cuda_bridge_paged_decode_workspace_elements(
      bridge_dtype(query), sequences, query_heads, kv_heads, head_size,
      value_head_size, num_blocks, block_size, max_blocks_per_sequence,
      max_context, scale_f32, &split_k_workspace_elements);
  check_bridge_status(status, "paged decode workspace query");
  STD_TORCH_CHECK(split_k_workspace_elements <=
                  static_cast<uint64_t>(
                      std::numeric_limits<int64_t>::max()),
              "Loom paged decode split-K workspace exceeds PyTorch limits");
  Tensor split_k_workspace;
  if (split_k_workspace_elements != 0U) {
    split_k_workspace = new_empty(
        query,
        {static_cast<int64_t>(split_k_workspace_elements)},
        ScalarType::Float);
  }
  float* split_k_workspace_pointer =
      split_k_workspace.defined()
          ? split_k_workspace.mutable_data_ptr<float>()
                                  : nullptr;

  status = loom_cuda_bridge_paged_decode_attention(
      bridge_dtype(query), query.const_data_ptr(),
      static_cast<uint64_t>(query.numel()), key_cache.const_data_ptr(),
      storage_span_elements(key_cache), value_cache.const_data_ptr(),
      storage_span_elements(value_cache),
      block_tables.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(block_tables.numel()),
      sequence_lengths.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(sequence_lengths.numel()),
      output.mutable_data_ptr(),
      static_cast<uint64_t>(output.numel()), split_k_workspace_pointer,
      split_k_workspace_elements, sequences, query_heads, kv_heads, head_size,
      value_head_size, num_blocks, block_size, key_block_stride,
      value_block_stride, max_blocks_per_sequence, max_context, scale_f32,
      stream.stream());
  check_bridge_status(status, "paged decode attention");
}

void paged_decode_attention(
    const Tensor& query, const Tensor& key_cache,
    const Tensor& value_cache, const Tensor& block_tables,
    const Tensor& sequence_lengths, Tensor output,
    int64_t max_sequence_length, double scale) {
  check_paged_decode_attention_contract(
      query, key_cache, value_cache, block_tables, sequence_lengths, output,
      max_sequence_length, scale);
  launch_paged_decode_attention(query, key_cache, value_cache, block_tables,
                                sequence_lengths, output,
                                max_sequence_length, scale);
}


}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "paged_decode_attention",
      TORCH_BOX(&loom_kernels::torch_adapter::paged_decode_attention));
}
