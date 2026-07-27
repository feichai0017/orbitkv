#include "loom_cuda.h"

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
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

struct Maximum {
  __device__ float operator()(float left, float right) const {
    return fmaxf(left, right);
  }
};

__device__ uint32_t ordered_float_key(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits << 1U) == 0U) {
    bits = 0U;
  }
  return (bits & 0x80000000U) != 0U ? ~bits : bits ^ 0x80000000U;
}

__device__ float float_from_ordered_key(uint32_t key) {
  const uint32_t bits =
      (key & 0x80000000U) != 0U ? key ^ 0x80000000U : ~key;
  return __uint_as_float(bits);
}

__device__ uint64_t ordered_token_key(float value, uint32_t token_id) {
  return (static_cast<uint64_t>(ordered_float_key(value)) << 32U) |
         token_id;
}

template <typename Ops, int Threads>
__global__ void row_maximum_kernel(const typename Ops::Scalar* logits,
                                   uint32_t rows, uint32_t vocab_size,
                                   uint64_t row_stride, float* maxima) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float local_maximum = -CUDART_INF_F;
  const size_t row_offset = static_cast<size_t>(row) * row_stride;
  for (uint32_t column = threadIdx.x; column < vocab_size;
       column += Threads) {
    local_maximum =
        fmaxf(local_maximum, Ops::to_float(logits[row_offset + column]));
  }
  using BlockReduce = cub::BlockReduce<float, Threads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const float maximum =
      BlockReduce(reduce_storage).Reduce(local_maximum, Maximum{});
  if (threadIdx.x == 0) {
    maxima[row] = maximum;
  }
}

template <typename Ops>
__global__ void local_sorted_prefix_kernel(
    const typename Ops::Scalar* logits, const float* maxima, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, uint32_t partitions,
    uint64_t* sorted_keys, float* prefix_masses) {
  const uint32_t block = blockIdx.x;
  const uint32_t row = block / partitions;
  const uint32_t partition = block - row * partitions;
  if (row >= rows) {
    return;
  }

  uint64_t keys[kItemsPerThread];
  const uint32_t partition_start = partition * kItemsPerPartition;
  const size_t row_offset = static_cast<size_t>(row) * row_stride;
#pragma unroll
  for (uint32_t item = 0; item < kItemsPerThread; ++item) {
    const uint32_t column =
        partition_start + threadIdx.x * kItemsPerThread + item;
    keys[item] =
        column < vocab_size
            ? ordered_token_key(
                  Ops::to_float(logits[row_offset + column]), column)
            : std::numeric_limits<uint64_t>::max();
  }

  using BlockRadixSort =
      cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
  using BlockScan = cub::BlockScan<float, kThreads>;
  union TempStorage {
    typename BlockRadixSort::TempStorage sort;
    typename BlockScan::TempStorage scan;
  };
  __shared__ TempStorage temp_storage;
  BlockRadixSort(temp_storage.sort).Sort(keys);
  __syncthreads();

  float masses[kItemsPerThread];
  float prefixes[kItemsPerThread];
  const float maximum = maxima[row];
#pragma unroll
  for (uint32_t item = 0; item < kItemsPerThread; ++item) {
    const uint32_t rank = threadIdx.x * kItemsPerThread + item;
    const uint32_t column = partition_start + rank;
    const uint32_t logit_key = static_cast<uint32_t>(keys[item] >> 32U);
    masses[item] =
        column < vocab_size
            ? expf(float_from_ordered_key(logit_key) - maximum)
            : 0.0F;
  }
  BlockScan(temp_storage.scan).InclusiveSum(masses, prefixes);

  const size_t output_offset =
      (static_cast<size_t>(row) * partitions + partition) *
      kItemsPerPartition;
#pragma unroll
  for (uint32_t item = 0; item < kItemsPerThread; ++item) {
    const uint32_t rank = threadIdx.x * kItemsPerThread + item;
    sorted_keys[output_offset + rank] = keys[item];
    prefix_masses[output_offset + rank] = prefixes[item];
  }
}

template <int Threads>
__global__ void select_top_p_threshold_kernel(
    const float* top_ps, uint32_t vocab_size, uint32_t partitions,
    const uint64_t* sorted_keys, const float* prefix_masses,
    uint64_t* threshold_keys, float* retained_sums) {
  const uint32_t row = blockIdx.x;
  using BlockReduce = cub::BlockReduce<float, Threads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ uint64_t lower;
  __shared__ uint64_t upper;
  __shared__ uint64_t midpoint;
  __shared__ float target_mass;
  __shared__ bool keep_all;

  float local_total = 0.0F;
  for (uint32_t partition = threadIdx.x; partition < partitions;
       partition += Threads) {
    const uint32_t partition_start = partition * kItemsPerPartition;
    const uint32_t remaining = vocab_size - partition_start;
    const uint32_t partition_size =
        remaining < kItemsPerPartition ? remaining : kItemsPerPartition;
    const size_t offset =
        (static_cast<size_t>(row) * partitions + partition) *
        kItemsPerPartition;
    local_total += prefix_masses[offset + partition_size - 1U];
  }
  const float total_mass = BlockReduce(reduce_storage).Sum(local_total);
  if (threadIdx.x == 0) {
    lower = 0ULL;
    upper = std::numeric_limits<uint64_t>::max();
    target_mass = (1.0F - top_ps[row]) * total_mass;
    keep_all = top_ps[row] == 1.0F;
  }
  __syncthreads();
  if (keep_all) {
    if (threadIdx.x == 0) {
      threshold_keys[row] = 0ULL;
      retained_sums[row] = total_mass;
    }
    return;
  }

  // Match vLLM's small-batch PyTorch path: accumulate the low-probability tail
  // in ascending (logit, token ID) order and discard entries while its mass is
  // at most 1 - top_p. The resulting first-kept key makes threshold ties retain
  // the greatest token IDs.
  for (int bit = 0; bit < 64; ++bit) {
    if (threadIdx.x == 0) {
      const uint64_t difference = upper - lower;
      midpoint = lower + (difference >> 1U) + (difference & 1ULL);
    }
    __syncthreads();

    float local_mass = 0.0F;
    for (uint32_t partition = threadIdx.x; partition < partitions;
         partition += Threads) {
      const uint32_t partition_start = partition * kItemsPerPartition;
      const uint32_t remaining = vocab_size - partition_start;
      const uint32_t partition_size =
          remaining < kItemsPerPartition ? remaining : kItemsPerPartition;
      const size_t offset =
          (static_cast<size_t>(row) * partitions + partition) *
          kItemsPerPartition;
      uint32_t first_not_below = 0U;
      uint32_t end = partition_size;
      while (first_not_below < end) {
        const uint32_t center =
            first_not_below + (end - first_not_below) / 2U;
        if (sorted_keys[offset + center] < midpoint) {
          first_not_below = center + 1U;
        } else {
          end = center;
        }
      }
      if (first_not_below != 0U) {
        local_mass += prefix_masses[offset + first_not_below - 1U];
      }
    }
    const float mass = BlockReduce(reduce_storage).Sum(local_mass);
    if (threadIdx.x == 0) {
      if (mass <= target_mass) {
        lower = midpoint;
      } else {
        upper = midpoint - 1ULL;
      }
    }
    __syncthreads();
  }

  float local_retained = 0.0F;
  for (uint32_t partition = threadIdx.x; partition < partitions;
       partition += Threads) {
    const uint32_t partition_start = partition * kItemsPerPartition;
    const uint32_t remaining = vocab_size - partition_start;
    const uint32_t partition_size =
        remaining < kItemsPerPartition ? remaining : kItemsPerPartition;
    const size_t offset =
        (static_cast<size_t>(row) * partitions + partition) *
        kItemsPerPartition;
    uint32_t first_kept = 0U;
    uint32_t end = partition_size;
    while (first_kept < end) {
      const uint32_t center = first_kept + (end - first_kept) / 2U;
      if (sorted_keys[offset + center] < lower) {
        first_kept = center + 1U;
      } else {
        end = center;
      }
    }
    const float partition_total =
        prefix_masses[offset + partition_size - 1U];
    const float excluded =
        first_kept == 0U ? 0.0F
                         : prefix_masses[offset + first_kept - 1U];
    local_retained += partition_total - excluded;
  }
  const float retained = BlockReduce(reduce_storage).Sum(local_retained);
  if (threadIdx.x == 0) {
    threshold_keys[row] = lower;
    retained_sums[row] = retained;
  }
}

template <typename Ops>
__global__ void apply_top_p_renorm_kernel(
    typename Ops::Scalar* logits, float* probabilities,
    const float* maxima, const uint64_t* threshold_keys,
    const float* retained_sums, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, uint32_t partitions) {
  const uint32_t block = blockIdx.x;
  const uint32_t row = block / partitions;
  const uint32_t partition = block - row * partitions;
  if (row >= rows) {
    return;
  }

  const uint64_t threshold = threshold_keys[row];
  const float maximum = maxima[row];
  const float inverse_sum = 1.0F / retained_sums[row];
  const uint32_t partition_start = partition * kItemsPerPartition;
  const size_t input_offset = static_cast<size_t>(row) * row_stride;
  const size_t output_offset = static_cast<size_t>(row) * vocab_size;
  for (uint32_t column = partition_start + threadIdx.x;
       column < vocab_size && column < partition_start + kItemsPerPartition;
       column += blockDim.x) {
    const size_t input_index = input_offset + column;
    const float value = Ops::to_float(logits[input_index]);
    const bool keep = ordered_token_key(value, column) >= threshold;
    probabilities[output_offset + column] =
        keep ? expf(value - maximum) * inverse_sum : 0.0F;
    if (!keep) {
      logits[input_index] = Ops::from_float(-CUDART_INF_F);
    }
  }
}

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

template <typename Ops>
int launch_top_p_renorm(
    typename Ops::Scalar* logits, const float* top_ps, float* probabilities,
    uint8_t* workspace, uint64_t workspace_bytes, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, uint32_t partitions,
    void* stream) {
  const uint64_t expected_partitions =
      (static_cast<uint64_t>(vocab_size) + kItemsPerPartition - 1U) /
      kItemsPerPartition;
  if (partitions == 0U ||
      static_cast<uint64_t>(rows) >
          std::numeric_limits<uint64_t>::max() / partitions ||
      static_cast<uint64_t>(rows) * partitions >
          std::numeric_limits<uint64_t>::max() / kItemsPerPartition) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const uint64_t sorted_elements = static_cast<uint64_t>(rows) * partitions *
                                   kItemsPerPartition;
  if (sorted_elements >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 12U)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const size_t key_bytes = static_cast<size_t>(sorted_elements) * sizeof(uint64_t);
  const size_t prefix_bytes =
      static_cast<size_t>(sorted_elements) * sizeof(float);
  const size_t maxima_offset = key_bytes + prefix_bytes;
  const size_t threshold_offset =
      align_up(maxima_offset + static_cast<size_t>(rows) * sizeof(float),
               alignof(uint64_t));
  const size_t retained_offset =
      threshold_offset + static_cast<size_t>(rows) * sizeof(uint64_t);
  const size_t required_workspace =
      retained_offset + static_cast<size_t>(rows) * sizeof(float);
  const uint64_t blocks = static_cast<uint64_t>(rows) * partitions;
  if (logits == nullptr || top_ps == nullptr || probabilities == nullptr ||
      workspace == nullptr || rows == 0 || vocab_size == 0 ||
      row_stride < vocab_size || partitions != expected_partitions ||
      blocks == 0 ||
      blocks > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      workspace_bytes < required_workspace ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  uint64_t* sorted_keys = reinterpret_cast<uint64_t*>(workspace);
  float* prefix_masses =
      reinterpret_cast<float*>(workspace + key_bytes);
  float* maxima = reinterpret_cast<float*>(workspace + maxima_offset);
  uint64_t* threshold_keys =
      reinterpret_cast<uint64_t*>(workspace + threshold_offset);
  float* retained_sums =
      reinterpret_cast<float*>(workspace + retained_offset);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

  row_maximum_kernel<Ops, kThreads>
      <<<rows, kThreads, 0, cuda_stream>>>(
          logits, rows, vocab_size, row_stride, maxima);
  if (cudaGetLastError() != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }
  local_sorted_prefix_kernel<Ops>
      <<<static_cast<uint32_t>(blocks), kThreads, 0, cuda_stream>>>(
          logits, maxima, rows, vocab_size, row_stride, partitions,
          sorted_keys, prefix_masses);
  if (cudaGetLastError() != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }
  select_top_p_threshold_kernel<kThreads>
      <<<rows, kThreads, 0, cuda_stream>>>(
          top_ps, vocab_size, partitions, sorted_keys, prefix_masses,
          threshold_keys, retained_sums);
  if (cudaGetLastError() != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }
  apply_top_p_renorm_kernel<Ops>
      <<<static_cast<uint32_t>(blocks), kThreads, 0, cuda_stream>>>(
          logits, probabilities, maxima, threshold_keys, retained_sums, rows,
          vocab_size, row_stride, partitions);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_top_p_renorm_f32(
    float* logits, const float* top_ps, float* probabilities,
    uint8_t* workspace, uint64_t workspace_bytes, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, uint32_t partitions,
    void* stream) {
  return launch_top_p_renorm<FloatOps>(
      logits, top_ps, probabilities, workspace, workspace_bytes, rows,
      vocab_size, row_stride, partitions, stream);
}

extern "C" int loom_cuda_top_p_renorm_f16(
    uint16_t* logits, const float* top_ps, float* probabilities,
    uint8_t* workspace, uint64_t workspace_bytes, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, uint32_t partitions,
    void* stream) {
  return launch_top_p_renorm<HalfOps>(
      reinterpret_cast<__half*>(logits), top_ps, probabilities, workspace,
      workspace_bytes, rows, vocab_size, row_stride, partitions, stream);
}

extern "C" int loom_cuda_top_p_renorm_bf16(
    uint16_t* logits, const float* top_ps, float* probabilities,
    uint8_t* workspace, uint64_t workspace_bytes, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, uint32_t partitions,
    void* stream) {
  return launch_top_p_renorm<Bfloat16Ops>(
      reinterpret_cast<__nv_bfloat16*>(logits), top_ps, probabilities,
      workspace, workspace_bytes, rows, vocab_size, row_stride, partitions,
      stream);
}
