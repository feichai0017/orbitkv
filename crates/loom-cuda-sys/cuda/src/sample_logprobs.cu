// Greedy, selected-token, and top-k sampled-logprob kernels.
#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

struct FloatOps {
  using Scalar = float;
  __device__ static float to_float(Scalar value) { return value; }
};

struct HalfOps {
  using Scalar = __half;
  __device__ static float to_float(Scalar value) {
    return __half2float(value);
  }
};

struct Bfloat16Ops {
  using Scalar = __nv_bfloat16;
  __device__ static float to_float(Scalar value) {
    return __bfloat162float(value);
  }
};

struct LogSumExpState {
  float maximum;
  float exponential_sum;
  uint32_t maximum_index;
  uint32_t maximum_count;
};

struct CombineLogSumExp {
  __device__ LogSumExpState operator()(const LogSumExpState& left,
                                       const LogSumExpState& right) const {
    if (left.maximum > right.maximum) {
      return {left.maximum,
              left.exponential_sum +
                  right.exponential_sum *
                      expf(right.maximum - left.maximum),
              left.maximum_index, left.maximum_count};
    }
    if (right.maximum > left.maximum) {
      return {right.maximum,
              right.exponential_sum +
                  left.exponential_sum * expf(left.maximum - right.maximum),
              right.maximum_index, right.maximum_count};
    }
    return {left.maximum, left.exponential_sum + right.exponential_sum,
            min(left.maximum_index, right.maximum_index),
            left.maximum_count + right.maximum_count};
  }
};

struct SelectedLogprobState {
  float maximum;
  float exponential_sum;
  uint32_t rank;
};

struct TopKCandidate {
  float value;
  uint32_t index;
};

static_assert(sizeof(SelectedLogprobState) == 12);
static_assert(alignof(SelectedLogprobState) == alignof(float));
static_assert(sizeof(TopKCandidate) == 8);
static_assert(alignof(TopKCandidate) == alignof(float));

__device__ __forceinline__ bool candidate_precedes(
    const TopKCandidate& left, const TopKCandidate& right) {
  return left.value > right.value ||
         (left.value == right.value && left.index < right.index);
}

__device__ __forceinline__ uint32_t topk_value_key(float value) {
  const float canonical = value == 0.0F ? 0.0F : value;
  const uint32_t bits = __float_as_uint(canonical);
  return (bits & 0x80000000U) != 0U ? ~bits
                                    : bits ^ 0x80000000U;
}

struct SelectTopKCandidate {
  __device__ TopKCandidate operator()(const TopKCandidate& left,
                                      const TopKCandidate& right) const {
    return candidate_precedes(left, right) ? left : right;
  }
};

struct SelectMinimumIndex {
  __device__ uint32_t operator()(uint32_t left,
                                 uint32_t right) const {
    return min(left, right);
  }
};

struct CombineSelectedLogprob {
  __device__ SelectedLogprobState operator()(
      const SelectedLogprobState& left,
      const SelectedLogprobState& right) const {
    if (left.maximum > right.maximum) {
      return {left.maximum,
              left.exponential_sum +
                  right.exponential_sum *
                      expf(right.maximum - left.maximum),
              left.rank + right.rank};
    }
    if (right.maximum > left.maximum) {
      return {right.maximum,
              right.exponential_sum +
                  left.exponential_sum * expf(left.maximum - right.maximum),
              left.rank + right.rank};
    }
    return {left.maximum, left.exponential_sum + right.exponential_sum,
            left.rank + right.rank};
  }
};

__device__ void update_state(LogSumExpState* state, float value,
                             uint32_t index) {
  if (value > state->maximum) {
    state->exponential_sum =
        state->exponential_sum * expf(state->maximum - value) + 1.0F;
    state->maximum = value;
    state->maximum_index = index;
    state->maximum_count = 1U;
  } else if (value == state->maximum) {
    state->exponential_sum += 1.0F;
    state->maximum_index = min(state->maximum_index, index);
    state->maximum_count += 1U;
  } else {
    state->exponential_sum += expf(value - state->maximum);
  }
}

template <typename Ops, int Threads>
__global__ __launch_bounds__(Threads) void greedy_sample_logprobs_kernel(
    const typename Ops::Scalar* logits, int32_t* token_ids,
    float* logprobs, int64_t* ranks, uint32_t vocab_size,
    uint64_t row_stride) {
  const size_t row_offset = static_cast<size_t>(blockIdx.x) * row_stride;
  LogSumExpState local = {-FLT_MAX, 0.0F, 0xffffffffU, 0U};
  for (uint32_t column = threadIdx.x; column < vocab_size;
       column += blockDim.x) {
    update_state(&local, Ops::to_float(logits[row_offset + column]), column);
  }

  using BlockReduce = cub::BlockReduce<LogSumExpState, Threads>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  const LogSumExpState row =
      BlockReduce(reduction_storage).Reduce(local, CombineLogSumExp{});
  if (threadIdx.x == 0) {
    token_ids[blockIdx.x] = static_cast<int32_t>(row.maximum_index);
    logprobs[blockIdx.x] = -logf(row.exponential_sum);
    // vLLM defines the sampled-token rank as the number of logprobs greater
    // than or equal to the sampled value. For greedy sampling this is the
    // number of tokens tied at the maximum, rather than always one.
    ranks[blockIdx.x] = static_cast<int64_t>(row.maximum_count);
  }
}

template <typename Ops>
int launch_greedy_sample_logprobs(const typename Ops::Scalar* logits,
                                  int32_t* token_ids, float* logprobs,
                                  int64_t* ranks, uint32_t rows,
                                  uint32_t vocab_size, uint64_t row_stride,
                                  void* stream) {
  if (logits == nullptr || token_ids == nullptr || logprobs == nullptr ||
      ranks == nullptr || rows == 0 || vocab_size == 0 ||
      row_stride < vocab_size ||
      vocab_size > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (vocab_size >= 65536U) {
    greedy_sample_logprobs_kernel<Ops, 1024>
        <<<rows, 1024, 0, cuda_stream>>>(logits, token_ids, logprobs, ranks,
                                        vocab_size, row_stride);
  } else {
    greedy_sample_logprobs_kernel<Ops, 256>
        <<<rows, 256, 0, cuda_stream>>>(logits, token_ids, logprobs, ranks,
                                       vocab_size, row_stride);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

template <typename Ops, int Threads>
__global__ __launch_bounds__(Threads) void selected_token_logprobs_kernel(
    const typename Ops::Scalar* logits, const int64_t* token_ids,
    float* logprobs, int64_t* ranks, uint32_t vocab_size,
    uint64_t row_stride) {
  const int64_t selected_index = token_ids[blockIdx.x];
  if (selected_index < 0 ||
      selected_index >= static_cast<int64_t>(vocab_size)) {
    if (threadIdx.x == 0) {
      logprobs[blockIdx.x] = __int_as_float(0x7fffffff);
      ranks[blockIdx.x] = 0;
    }
    return;
  }

  const size_t row_offset = static_cast<size_t>(blockIdx.x) * row_stride;
  const float selected = Ops::to_float(
      logits[row_offset + static_cast<size_t>(selected_index)]);
  SelectedLogprobState local = {-FLT_MAX, 0.0F, 0U};
  for (uint32_t column = threadIdx.x; column < vocab_size;
       column += blockDim.x) {
    const float value = Ops::to_float(logits[row_offset + column]);
    local.rank += static_cast<uint32_t>(value >= selected);
    if (value > local.maximum) {
      local.exponential_sum =
          local.exponential_sum * expf(local.maximum - value) + 1.0F;
      local.maximum = value;
    } else {
      local.exponential_sum += expf(value - local.maximum);
    }
  }

  using BlockReduce = cub::BlockReduce<SelectedLogprobState, Threads>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  const SelectedLogprobState row = BlockReduce(reduction_storage)
                                       .Reduce(local, CombineSelectedLogprob{});
  if (threadIdx.x == 0) {
    logprobs[blockIdx.x] =
        selected - row.maximum - logf(row.exponential_sum);
    ranks[blockIdx.x] = static_cast<int64_t>(row.rank);
  }
}

template <typename Ops>
int launch_selected_token_logprobs(
    const typename Ops::Scalar* logits, const int64_t* token_ids,
    float* logprobs, int64_t* ranks, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, void* stream) {
  if (logits == nullptr || token_ids == nullptr || logprobs == nullptr ||
      ranks == nullptr || rows == 0 || vocab_size == 0 ||
      row_stride < vocab_size ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (vocab_size >= 65536U) {
    selected_token_logprobs_kernel<Ops, 1024>
        <<<rows, 1024, 0, cuda_stream>>>(logits, token_ids, logprobs, ranks,
                                        vocab_size, row_stride);
  } else {
    selected_token_logprobs_kernel<Ops, 256>
        <<<rows, 256, 0, cuda_stream>>>(logits, token_ids, logprobs, ranks,
                                       vocab_size, row_stride);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

template <typename Ops, int Threads, int ItemsPerThread>
__global__ __launch_bounds__(Threads)
    void topk_sampled_logprobs_partials_kernel(
        const typename Ops::Scalar* logits,
        const int64_t* sampled_token_ids,
        SelectedLogprobState* partial_states,
        TopKCandidate* partial_candidates, uint32_t vocab_size,
        uint32_t top_k, uint32_t partitions, uint64_t row_stride) {
  const uint32_t partial_index = blockIdx.x;
  const uint32_t row_index = partial_index / partitions;
  const uint32_t partition_index =
      partial_index - row_index * partitions;
  const int64_t sampled_index = sampled_token_ids[row_index];
  if (sampled_index < 0 ||
      sampled_index >= static_cast<int64_t>(vocab_size)) {
    return;
  }

  const uint32_t column_begin = static_cast<uint32_t>(
      static_cast<uint64_t>(vocab_size) * partition_index /
      partitions);
  const uint32_t column_end = static_cast<uint32_t>(
      static_cast<uint64_t>(vocab_size) * (partition_index + 1U) /
      partitions);
  const size_t row_offset =
      static_cast<size_t>(row_index) * row_stride;
  const float selected = Ops::to_float(
      logits[row_offset + static_cast<size_t>(sampled_index)]);
  SelectedLogprobState local_state = {-FLT_MAX, 0.0F, 0U};
  uint32_t keys[ItemsPerThread];
  uint32_t indices[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const uint32_t column =
        column_begin + threadIdx.x + item * Threads;
    if (column < column_end) {
      const float value = Ops::to_float(logits[row_offset + column]);
      local_state.rank += static_cast<uint32_t>(value >= selected);
      if (value > local_state.maximum) {
        local_state.exponential_sum =
            local_state.exponential_sum *
                expf(local_state.maximum - value) +
            1.0F;
        local_state.maximum = value;
      } else {
        local_state.exponential_sum +=
            expf(value - local_state.maximum);
      }
      keys[item] = topk_value_key(value);
      indices[item] = column;
    } else {
      keys[item] = 0U;
      indices[item] = 0xffffffffU;
    }
  }

  using StateReduce =
      cub::BlockReduce<SelectedLogprobState, Threads>;
  using IndexReduce = cub::BlockReduce<uint32_t, Threads>;
  using BlockSort =
      cub::BlockRadixSort<uint32_t, Threads, ItemsPerThread,
                          uint32_t>;
  union SharedStorage {
    typename StateReduce::TempStorage state;
    typename BlockSort::TempStorage sort;
    typename IndexReduce::TempStorage index;
  };
  __shared__ SharedStorage shared;
  __shared__ uint32_t threshold_key;
  __shared__ uint32_t greater_count;
  __shared__ uint32_t minimum_threshold_index;

  const SelectedLogprobState partial =
      StateReduce(shared.state)
          .Reduce(local_state, CombineSelectedLogprob{});
  const uint64_t candidate_offset =
      static_cast<uint64_t>(partial_index) * top_k;
  if (threadIdx.x == 0) {
    partial_states[partial_index] = partial;
  }
  __syncthreads();

  BlockSort(shared.sort).SortDescending(keys, indices);
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const uint32_t rank = threadIdx.x * ItemsPerThread + item;
    if (rank == top_k - 1U) {
      threshold_key = keys[item];
    }
  }
  __syncthreads();

  uint32_t local_greater_count = 0;
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    local_greater_count +=
        static_cast<uint32_t>(keys[item] > threshold_key);
  }
  const uint32_t block_greater_count =
      IndexReduce(shared.index).Sum(local_greater_count);
  if (threadIdx.x == 0) {
    greater_count = block_greater_count;
    minimum_threshold_index = 0U;
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const uint32_t rank = threadIdx.x * ItemsPerThread + item;
    if (keys[item] > threshold_key) {
      partial_candidates[candidate_offset + rank] = {
          Ops::to_float(logits[row_offset + indices[item]]),
          indices[item]};
    }
  }

  for (uint32_t slot = greater_count; slot < top_k; ++slot) {
    uint32_t local_minimum = 0xffffffffU;
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
      if (keys[item] == threshold_key &&
          indices[item] >= minimum_threshold_index) {
        local_minimum = min(local_minimum, indices[item]);
      }
    }
    const uint32_t selected_index =
        IndexReduce(shared.index)
            .Reduce(local_minimum, SelectMinimumIndex{});
    if (threadIdx.x == 0) {
      partial_candidates[candidate_offset + slot] = {
          Ops::to_float(logits[row_offset + selected_index]),
          selected_index};
      minimum_threshold_index = selected_index + 1U;
    }
    __syncthreads();
  }
}

template <typename Ops, int Threads>
__global__ __launch_bounds__(Threads)
    void topk_sampled_logprobs_merge_kernel(
        const typename Ops::Scalar* logits,
        const int64_t* sampled_token_ids,
        const SelectedLogprobState* partial_states,
        const TopKCandidate* partial_candidates,
        int32_t* output_token_ids, float* output_logprobs,
        int64_t* sampled_token_ranks, uint32_t vocab_size,
        uint32_t top_k, uint32_t partitions, uint64_t row_stride) {
  const uint32_t row_index = blockIdx.x;
  const int64_t sampled_index = sampled_token_ids[row_index];
  const uint64_t output_width = static_cast<uint64_t>(top_k) + 1U;
  const uint64_t output_offset =
      static_cast<uint64_t>(row_index) * output_width;
  if (sampled_index < 0 ||
      sampled_index >= static_cast<int64_t>(vocab_size)) {
    for (uint32_t slot = threadIdx.x; slot < output_width;
         slot += Threads) {
      output_token_ids[output_offset + slot] = -1;
      output_logprobs[output_offset + slot] =
          __int_as_float(0x7fffffff);
    }
    if (threadIdx.x == 0) {
      sampled_token_ranks[row_index] = 0;
    }
    return;
  }

  SelectedLogprobState local_state = {-FLT_MAX, 0.0F, 0U};
  const uint64_t partial_offset =
      static_cast<uint64_t>(row_index) * partitions;
  for (uint32_t partition = threadIdx.x; partition < partitions;
       partition += Threads) {
    local_state = CombineSelectedLogprob{}(
        local_state, partial_states[partial_offset + partition]);
  }

  using StateReduce =
      cub::BlockReduce<SelectedLogprobState, Threads>;
  using CandidateReduce =
      cub::BlockReduce<TopKCandidate, Threads>;
  __shared__ typename StateReduce::TempStorage
      state_reduction_storage;
  __shared__ typename CandidateReduce::TempStorage
      candidate_reduction_storage;
  __shared__ float log_normalizer;
  __shared__ TopKCandidate previous;

  const SelectedLogprobState row =
      StateReduce(state_reduction_storage)
          .Reduce(local_state, CombineSelectedLogprob{});
  if (threadIdx.x == 0) {
    const size_t row_offset =
        static_cast<size_t>(row_index) * row_stride;
    const float selected = Ops::to_float(
        logits[row_offset + static_cast<size_t>(sampled_index)]);
    log_normalizer = row.maximum + logf(row.exponential_sum);
    output_token_ids[output_offset] =
        static_cast<int32_t>(sampled_index);
    output_logprobs[output_offset] = selected - log_normalizer;
    sampled_token_ranks[row_index] =
        static_cast<int64_t>(row.rank);
  }
  __syncthreads();

  const uint32_t candidate_count = partitions * top_k;
  const uint64_t candidate_offset = partial_offset * top_k;
  for (uint32_t slot = 0; slot < top_k; ++slot) {
    TopKCandidate local = {-FLT_MAX, 0xffffffffU};
    for (uint32_t index = threadIdx.x; index < candidate_count;
         index += Threads) {
      const TopKCandidate candidate =
          partial_candidates[candidate_offset + index];
      if ((slot == 0 ||
           candidate_precedes(previous, candidate)) &&
          candidate_precedes(candidate, local)) {
        local = candidate;
      }
    }
    const TopKCandidate best =
        CandidateReduce(candidate_reduction_storage)
            .Reduce(local, SelectTopKCandidate{});
    if (threadIdx.x == 0) {
      output_token_ids[output_offset + slot + 1U] =
          static_cast<int32_t>(best.index);
      output_logprobs[output_offset + slot + 1U] =
          best.value - log_normalizer;
      previous = best;
    }
    __syncthreads();
  }
}

template <typename Ops>
int launch_topk_sampled_logprobs(
    const typename Ops::Scalar* logits,
    const int64_t* sampled_token_ids, int32_t* output_token_ids,
    float* output_logprobs, int64_t* sampled_token_ranks,
    uint32_t rows, uint32_t vocab_size, uint32_t top_k,
    uint64_t row_stride, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t partitions, void* stream) {
  const uint64_t partial_count =
      static_cast<uint64_t>(rows) * partitions;
  const uint32_t max_partition_columns =
      partitions == 0
          ? 0U
          : static_cast<uint32_t>(
                (static_cast<uint64_t>(vocab_size) + partitions - 1U) /
                partitions);
  const uint64_t bytes_per_partial =
      sizeof(SelectedLogprobState) +
      static_cast<uint64_t>(top_k) * sizeof(TopKCandidate);
  if (logits == nullptr || sampled_token_ids == nullptr ||
      output_token_ids == nullptr || output_logprobs == nullptr ||
      sampled_token_ranks == nullptr || workspace == nullptr ||
      rows == 0 || vocab_size == 0 || top_k == 0 ||
      top_k > 32U || top_k > vocab_size || partitions == 0 ||
      partitions > vocab_size ||
      max_partition_columns > 4096U ||
      partial_count >
          static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      bytes_per_partial >
          std::numeric_limits<uint64_t>::max() / partial_count ||
      workspace_bytes < partial_count * bytes_per_partial ||
      row_stride < vocab_size ||
      vocab_size >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  auto* partial_states =
      reinterpret_cast<SelectedLogprobState*>(workspace);
  auto* partial_candidates = reinterpret_cast<TopKCandidate*>(
      workspace + partial_count * sizeof(SelectedLogprobState));
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  const uint32_t partial_blocks =
      static_cast<uint32_t>(partial_count);
  if (max_partition_columns <= 256U) {
    topk_sampled_logprobs_partials_kernel<Ops, 256, 1>
        <<<partial_blocks, 256, 0, cuda_stream>>>(
            logits, sampled_token_ids, partial_states,
            partial_candidates, vocab_size, top_k, partitions,
            row_stride);
  } else if (max_partition_columns <= 512U) {
    topk_sampled_logprobs_partials_kernel<Ops, 256, 2>
        <<<partial_blocks, 256, 0, cuda_stream>>>(
            logits, sampled_token_ids, partial_states,
            partial_candidates, vocab_size, top_k, partitions,
            row_stride);
  } else if (max_partition_columns <= 1024U) {
    topk_sampled_logprobs_partials_kernel<Ops, 256, 4>
        <<<partial_blocks, 256, 0, cuda_stream>>>(
            logits, sampled_token_ids, partial_states,
            partial_candidates, vocab_size, top_k, partitions,
            row_stride);
  } else if (max_partition_columns <= 2048U) {
    topk_sampled_logprobs_partials_kernel<Ops, 256, 8>
        <<<partial_blocks, 256, 0, cuda_stream>>>(
            logits, sampled_token_ids, partial_states,
            partial_candidates, vocab_size, top_k, partitions,
            row_stride);
  } else {
    topk_sampled_logprobs_partials_kernel<Ops, 256, 16>
        <<<partial_blocks, 256, 0, cuda_stream>>>(
            logits, sampled_token_ids, partial_states,
            partial_candidates, vocab_size, top_k, partitions,
            row_stride);
  }
  if (cudaGetLastError() != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }
  topk_sampled_logprobs_merge_kernel<Ops, 256>
      <<<rows, 256, 0, cuda_stream>>>(
          logits, sampled_token_ids, partial_states,
          partial_candidates, output_token_ids, output_logprobs,
          sampled_token_ranks, vocab_size, top_k, partitions,
          row_stride);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_greedy_sample_logprobs_f32(
    const float* logits, int32_t* token_ids, float* logprobs, int64_t* ranks,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride, void* stream) {
  return launch_greedy_sample_logprobs<FloatOps>(
      logits, token_ids, logprobs, ranks, rows, vocab_size, row_stride,
      stream);
}

extern "C" int loom_cuda_greedy_sample_logprobs_f16(
    const uint16_t* logits, int32_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_greedy_sample_logprobs<HalfOps>(
      reinterpret_cast<const __half*>(logits), token_ids, logprobs, ranks,
      rows, vocab_size, row_stride, stream);
}

extern "C" int loom_cuda_greedy_sample_logprobs_bf16(
    const uint16_t* logits, int32_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_greedy_sample_logprobs<Bfloat16Ops>(
      reinterpret_cast<const __nv_bfloat16*>(logits), token_ids, logprobs,
      ranks, rows, vocab_size, row_stride, stream);
}

extern "C" int loom_cuda_selected_token_logprobs_f32(
    const float* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_selected_token_logprobs<FloatOps>(
      logits, token_ids, logprobs, ranks, rows, vocab_size, row_stride,
      stream);
}

extern "C" int loom_cuda_selected_token_logprobs_f16(
    const uint16_t* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_selected_token_logprobs<HalfOps>(
      reinterpret_cast<const __half*>(logits), token_ids, logprobs, ranks,
      rows, vocab_size, row_stride, stream);
}

extern "C" int loom_cuda_selected_token_logprobs_bf16(
    const uint16_t* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_selected_token_logprobs<Bfloat16Ops>(
      reinterpret_cast<const __nv_bfloat16*>(logits), token_ids, logprobs,
      ranks, rows, vocab_size, row_stride, stream);
}

extern "C" int loom_cuda_topk_sampled_logprobs_f32(
    const float* logits, const int64_t* sampled_token_ids,
    int32_t* output_token_ids, float* output_logprobs,
    int64_t* sampled_token_ranks, uint32_t rows, uint32_t vocab_size,
    uint32_t top_k, uint64_t row_stride, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t partitions, void* stream) {
  return launch_topk_sampled_logprobs<FloatOps>(
      logits, sampled_token_ids, output_token_ids, output_logprobs,
      sampled_token_ranks, rows, vocab_size, top_k, row_stride,
      workspace, workspace_bytes, partitions, stream);
}

extern "C" int loom_cuda_topk_sampled_logprobs_f16(
    const uint16_t* logits, const int64_t* sampled_token_ids,
    int32_t* output_token_ids, float* output_logprobs,
    int64_t* sampled_token_ranks, uint32_t rows, uint32_t vocab_size,
    uint32_t top_k, uint64_t row_stride, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t partitions, void* stream) {
  return launch_topk_sampled_logprobs<HalfOps>(
      reinterpret_cast<const __half*>(logits), sampled_token_ids,
      output_token_ids, output_logprobs, sampled_token_ranks, rows,
      vocab_size, top_k, row_stride, workspace, workspace_bytes,
      partitions, stream);
}

extern "C" int loom_cuda_topk_sampled_logprobs_bf16(
    const uint16_t* logits, const int64_t* sampled_token_ids,
    int32_t* output_token_ids, float* output_logprobs,
    int64_t* sampled_token_ranks, uint32_t rows, uint32_t vocab_size,
    uint32_t top_k, uint64_t row_stride, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t partitions, void* stream) {
  return launch_topk_sampled_logprobs<Bfloat16Ops>(
      reinterpret_cast<const __nv_bfloat16*>(logits),
      sampled_token_ids, output_token_ids, output_logprobs,
      sampled_token_ranks, rows, vocab_size, top_k, row_stride,
      workspace, workspace_bytes, partitions, stream);
}
