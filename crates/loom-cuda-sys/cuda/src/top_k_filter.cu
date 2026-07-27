#include "loom_cuda.h"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kItemsPerThread = 16U;
constexpr uint32_t kItemsPerPartition = kThreads * kItemsPerThread;

struct FloatOps {
  using Scalar = float;

  __device__ static float to_float(float value) { return value; }
  __device__ static float from_float(float value) { return value; }
};

struct HalfOps {
  using Scalar = __half;

  __device__ static float to_float(__half value) { return __half2float(value); }
  __device__ static __half from_float(float value) {
    return __float2half_rn(value);
  }
};

struct Bfloat16Ops {
  using Scalar = __nv_bfloat16;

  __device__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
  }
};

__device__ uint32_t ordered_float_key(float value) {
  uint32_t bits = __float_as_uint(value);
  // PyTorch compares -0 and +0 as equal. Canonicalize both before radix
  // selection so a zero-valued threshold never filters the other encoding.
  if ((bits << 1U) == 0U) {
    bits = 0U;
  }
  return (bits & 0x80000000U) != 0U ? ~bits : bits ^ 0x80000000U;
}

template <typename Ops>
__global__ void local_sorted_keys_kernel(
    const typename Ops::Scalar* logits, const int32_t* top_ks,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    uint32_t partitions, uint32_t* sorted_keys) {
  const uint32_t block = blockIdx.x;
  const uint32_t row = block / partitions;
  const uint32_t partition = block - row * partitions;
  if (row >= rows) {
    return;
  }
  const int32_t requested_top_k = top_ks[row];
  if (requested_top_k <= 0 ||
      static_cast<uint32_t>(requested_top_k) >= vocab_size) {
    return;
  }

  uint32_t keys[kItemsPerThread];
  const uint32_t partition_start = partition * kItemsPerPartition;
  const size_t row_offset = static_cast<size_t>(row) * row_stride;
#pragma unroll
  for (uint32_t item = 0; item < kItemsPerThread; ++item) {
    const uint32_t column =
        partition_start + threadIdx.x * kItemsPerThread + item;
    keys[item] =
        column < vocab_size
            ? ordered_float_key(Ops::to_float(logits[row_offset + column]))
            : 0U;
  }

  using BlockRadixSort =
      cub::BlockRadixSort<uint32_t, kThreads, kItemsPerThread>;
  __shared__ typename BlockRadixSort::TempStorage sort_storage;
  BlockRadixSort(sort_storage).SortDescending(keys);

  const size_t output_offset =
      (static_cast<size_t>(row) * partitions + partition) *
      kItemsPerPartition;
#pragma unroll
  for (uint32_t item = 0; item < kItemsPerThread; ++item) {
    sorted_keys[output_offset + threadIdx.x * kItemsPerThread + item] =
        keys[item];
  }
}

template <int Threads>
__global__ void select_threshold_kernel(
    const int32_t* top_ks, uint32_t vocab_size, uint32_t partitions,
    const uint32_t* sorted_keys, uint32_t* threshold_keys) {
  const uint32_t row = blockIdx.x;
  const int32_t requested_top_k = top_ks[row];
  if (requested_top_k <= 0 ||
      static_cast<uint32_t>(requested_top_k) > vocab_size ||
      static_cast<uint32_t>(requested_top_k) == vocab_size) {
    if (threadIdx.x == 0) {
      threshold_keys[row] = 0U;
    }
    return;
  }

  using BlockReduce = cub::BlockReduce<uint64_t, Threads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ uint32_t lower;
  __shared__ uint32_t upper;
  __shared__ uint32_t midpoint;
  __shared__ uint64_t count_at_midpoint;
  if (threadIdx.x == 0) {
    lower = 0U;
    upper = std::numeric_limits<uint32_t>::max();
  }
  __syncthreads();

  // Find the greatest ordered key whose greater-or-equal count still reaches
  // the requested rank. Each count uses binary search over every independently
  // sorted 4096-key partition, so all top-k values share one exact path.
  for (int bit = 0; bit < 32; ++bit) {
    if (threadIdx.x == 0) {
      midpoint = lower + static_cast<uint32_t>(
                             (static_cast<uint64_t>(upper) - lower + 1U) /
                             2U);
    }
    __syncthreads();

    uint64_t local_count = 0U;
    for (uint32_t partition = threadIdx.x; partition < partitions;
         partition += Threads) {
      const uint32_t partition_start = partition * kItemsPerPartition;
      const uint32_t partition_size =
          min(kItemsPerPartition, vocab_size - partition_start);
      const size_t partition_offset =
          (static_cast<size_t>(row) * partitions + partition) *
          kItemsPerPartition;
      uint32_t first_below = 0U;
      uint32_t end = partition_size;
      while (first_below < end) {
        const uint32_t center = first_below + (end - first_below) / 2U;
        if (sorted_keys[partition_offset + center] >= midpoint) {
          first_below = center + 1U;
        } else {
          end = center;
        }
      }
      local_count += first_below;
    }

    const uint64_t total_count = BlockReduce(reduce_storage).Sum(local_count);
    if (threadIdx.x == 0) {
      count_at_midpoint = total_count;
      if (count_at_midpoint >= static_cast<uint32_t>(requested_top_k)) {
        lower = midpoint;
      } else {
        upper = midpoint - 1U;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    threshold_keys[row] = lower;
  }
}

template <typename Ops>
__global__ void apply_threshold_kernel(
    typename Ops::Scalar* logits, const int32_t* top_ks,
    const uint32_t* threshold_keys, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, uint32_t partitions) {
  const uint32_t block = blockIdx.x;
  const uint32_t row = block / partitions;
  const uint32_t partition = block - row * partitions;
  if (row >= rows) {
    return;
  }
  const int32_t requested_top_k = top_ks[row];
  if (requested_top_k <= 0 ||
      static_cast<uint32_t>(requested_top_k) >= vocab_size) {
    return;
  }

  const uint32_t threshold = threshold_keys[row];
  const uint32_t partition_start = partition * kItemsPerPartition;
  const size_t row_offset = static_cast<size_t>(row) * row_stride;
  for (uint32_t column = partition_start + threadIdx.x;
       column < vocab_size && column < partition_start + kItemsPerPartition;
       column += blockDim.x) {
    const size_t index = row_offset + column;
    const uint32_t key = ordered_float_key(Ops::to_float(logits[index]));
    if (key < threshold) {
      logits[index] = Ops::from_float(-CUDART_INF_F);
    }
  }
}

template <typename Ops>
int launch_top_k_filter(typename Ops::Scalar* logits, const int32_t* top_ks,
                        uint32_t* workspace, uint64_t workspace_elements,
                        uint32_t rows, uint32_t vocab_size,
                        uint64_t row_stride, uint32_t partitions,
                        void* stream) {
  const uint64_t expected_partitions =
      (static_cast<uint64_t>(vocab_size) + kItemsPerPartition - 1U) /
      kItemsPerPartition;
  const uint64_t sorted_key_elements =
      static_cast<uint64_t>(rows) * partitions * kItemsPerPartition;
  const uint64_t required_workspace = sorted_key_elements + rows;
  const uint64_t blocks = static_cast<uint64_t>(rows) * partitions;
  if (logits == nullptr || top_ks == nullptr || workspace == nullptr ||
      rows == 0 || vocab_size == 0 || row_stride < vocab_size ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      vocab_size >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      partitions != expected_partitions || blocks == 0 ||
      blocks > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      workspace_elements < required_workspace ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  uint32_t* threshold_keys = workspace + sorted_key_elements;
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  local_sorted_keys_kernel<Ops>
      <<<static_cast<uint32_t>(blocks), kThreads, 0, cuda_stream>>>(
          logits, top_ks, rows, vocab_size, row_stride, partitions, workspace);
  if (cudaGetLastError() != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }
  select_threshold_kernel<kThreads>
      <<<rows, kThreads, 0, cuda_stream>>>(
          top_ks, vocab_size, partitions, workspace, threshold_keys);
  if (cudaGetLastError() != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }
  apply_threshold_kernel<Ops>
      <<<static_cast<uint32_t>(blocks), kThreads, 0, cuda_stream>>>(
          logits, top_ks, threshold_keys, rows, vocab_size, row_stride,
          partitions);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_top_k_filter_f32(
    float* logits, const int32_t* top_ks, uint32_t* workspace,
    uint64_t workspace_elements, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, uint32_t partitions, void* stream) {
  return launch_top_k_filter<FloatOps>(
      logits, top_ks, workspace, workspace_elements, rows, vocab_size,
      row_stride, partitions, stream);
}

extern "C" int loom_cuda_top_k_filter_f16(
    uint16_t* logits, const int32_t* top_ks, uint32_t* workspace,
    uint64_t workspace_elements, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, uint32_t partitions, void* stream) {
  return launch_top_k_filter<HalfOps>(
      reinterpret_cast<__half*>(logits), top_ks, workspace,
      workspace_elements, rows, vocab_size, row_stride, partitions, stream);
}

extern "C" int loom_cuda_top_k_filter_bf16(
    uint16_t* logits, const int32_t* top_ks, uint32_t* workspace,
    uint64_t workspace_elements, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, uint32_t partitions, void* stream) {
  return launch_top_k_filter<Bfloat16Ops>(
      reinterpret_cast<__nv_bfloat16*>(logits), top_ks, workspace,
      workspace_elements, rows, vocab_size, row_stride, partitions, stream);
}
