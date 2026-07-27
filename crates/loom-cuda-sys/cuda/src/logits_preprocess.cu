#include "loom_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kItemsPerThread = 4U;
constexpr uint32_t kPartitionSize = kThreads * kItemsPerThread;

__global__ void logits_preprocess_kernel(
    float* logits, const float* temperatures, const uint8_t* blocked_mask,
    const int32_t* bias_row_ids, const int32_t* bias_token_ids,
    const float* bias_values, uint32_t bias_count,
    const int32_t* suppressed_row_ids,
    const int32_t* suppressed_token_ids, uint32_t suppression_count,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    uint32_t partitions) {
  const uint32_t row = blockIdx.x / partitions;
  const uint32_t partition = blockIdx.x % partitions;
  if (row >= rows) {
    return;
  }

  const uint64_t partition_start =
      static_cast<uint64_t>(partition) * kPartitionSize;
  const uint64_t partition_end =
      min(partition_start + kPartitionSize,
          static_cast<uint64_t>(vocab_size));
  const size_t row_offset = static_cast<size_t>(row) * row_stride;
  const size_t mask_row_offset =
      static_cast<size_t>(row) * static_cast<size_t>(vocab_size);

#pragma unroll
  for (uint32_t item = 0; item < kItemsPerThread; ++item) {
    const uint64_t column =
        partition_start + threadIdx.x + item * kThreads;
    if (column < partition_end && blocked_mask != nullptr &&
        blocked_mask[mask_row_offset + static_cast<size_t>(column)] != 0U) {
      logits[row_offset + static_cast<size_t>(column)] = -CUDART_INF_F;
    }
  }
  __syncthreads();

  for (uint32_t entry = threadIdx.x; entry < bias_count;
       entry += blockDim.x) {
    const int32_t target_row = bias_row_ids[entry];
    const int32_t target_token = bias_token_ids[entry];
    if (target_row == static_cast<int32_t>(row) && target_token >= 0 &&
        static_cast<uint64_t>(target_token) >= partition_start &&
        static_cast<uint64_t>(target_token) < partition_end) {
      atomicAdd(logits + row_offset + static_cast<size_t>(target_token),
                bias_values[entry]);
    }
  }
  __syncthreads();

  for (uint32_t entry = threadIdx.x; entry < suppression_count;
       entry += blockDim.x) {
    const int32_t target_row = suppressed_row_ids[entry];
    const int32_t target_token = suppressed_token_ids[entry];
    if (target_row == static_cast<int32_t>(row) && target_token >= 0 &&
        static_cast<uint64_t>(target_token) >= partition_start &&
        static_cast<uint64_t>(target_token) < partition_end) {
      atomicExch(
          reinterpret_cast<int*>(
              logits + row_offset + static_cast<size_t>(target_token)),
          __float_as_int(-CUDART_INF_F));
    }
  }
  __syncthreads();

  const float temperature = temperatures[row];
  const float divisor = temperature < 1.0e-5F ? 1.0F : temperature;
#pragma unroll
  for (uint32_t item = 0; item < kItemsPerThread; ++item) {
    const uint64_t column =
        partition_start + threadIdx.x + item * kThreads;
    if (column < partition_end) {
      const size_t index = row_offset + static_cast<size_t>(column);
      logits[index] /= divisor;
    }
  }
}

bool optional_group_is_valid(uint32_t count, const void* first,
                             const void* second, const void* third) {
  if (count == 0U) {
    return first == nullptr && second == nullptr && third == nullptr;
  }
  return first != nullptr && second != nullptr && third != nullptr;
}

bool optional_pair_is_valid(uint32_t count, const void* first,
                            const void* second) {
  if (count == 0U) {
    return first == nullptr && second == nullptr;
  }
  return first != nullptr && second != nullptr;
}

}  // namespace

extern "C" int loom_cuda_logits_preprocess_f32(
    float* logits, const float* temperatures, const uint8_t* blocked_mask,
    const int32_t* bias_row_ids, const int32_t* bias_token_ids,
    const float* bias_values, uint32_t bias_count,
    const int32_t* suppressed_row_ids,
    const int32_t* suppressed_token_ids, uint32_t suppression_count,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride, void* stream) {
  const bool bias_group_valid = optional_group_is_valid(
      bias_count, bias_row_ids, bias_token_ids, bias_values);
  const bool suppression_group_valid = optional_pair_is_valid(
      suppression_count, suppressed_row_ids, suppressed_token_ids);
  if (logits == nullptr || temperatures == nullptr || !bias_group_valid ||
      !suppression_group_valid || rows == 0U || vocab_size == 0U ||
      row_stride < vocab_size ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  const uint32_t partitions =
      (vocab_size - 1U) / kPartitionSize + 1U;
  const uint64_t blocks =
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(partitions);
  if (blocks > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return LOOM_CUDA_UNSUPPORTED;
  }

  logits_preprocess_kernel<<<static_cast<uint32_t>(blocks), kThreads, 0,
                             static_cast<cudaStream_t>(stream)>>>(
      logits, temperatures, blocked_mask, bias_row_ids, bias_token_ids,
      bias_values, bias_count, suppressed_row_ids, suppressed_token_ids,
      suppression_count, rows, vocab_size, row_stride, partitions);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}
