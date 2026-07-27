#include "loom_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kWarps = 32U;
constexpr uint32_t kThreads = kWarpSize * kWarps;
constexpr double kProbabilitySumTolerance = 1.0e-5;
constexpr double kUint32Unit = 1.0 / 4294967296.0;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ __forceinline__ uint32_t warp_max(uint32_t value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = max(value,
                __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_inclusive_sum(float value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float preceding =
        __shfl_up_sync(0xffffffffU, value, offset);
    if (static_cast<int>(threadIdx.x & 31U) >= offset) {
      value += preceding;
    }
  }
  return value;
}

__device__ __forceinline__ uint32_t philox4x32_10_first(uint64_t seed,
                                                        uint64_t counter) {
  constexpr uint32_t kMultiplier0 = 0xD2511F53U;
  constexpr uint32_t kMultiplier1 = 0xCD9E8D57U;
  constexpr uint32_t kWeyl0 = 0x9E3779B9U;
  constexpr uint32_t kWeyl1 = 0xBB67AE85U;

  uint4 value = make_uint4(static_cast<uint32_t>(counter),
                           static_cast<uint32_t>(counter >> 32U), 0U, 0U);
  uint2 key = make_uint2(static_cast<uint32_t>(seed),
                         static_cast<uint32_t>(seed >> 32U));
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint32_t high0 = __umulhi(kMultiplier0, value.x);
    const uint32_t low0 = kMultiplier0 * value.x;
    const uint32_t high1 = __umulhi(kMultiplier1, value.z);
    const uint32_t low1 = kMultiplier1 * value.z;
    value = make_uint4(high1 ^ value.y ^ key.x, low1,
                       high0 ^ value.w ^ key.y, low0);
    key.x += kWeyl0;
    key.y += kWeyl1;
  }
  return value.x;
}

__global__ __launch_bounds__(kThreads) void categorical_sample_kernel(
    const float* probabilities, int64_t* rng_state, int64_t* token_ids,
    uint32_t vocab_size) {
  const uint32_t row = blockIdx.x;
  const uint32_t thread = threadIdx.x;
  const uint32_t warp = thread / kWarpSize;
  const uint32_t lane = thread & (kWarpSize - 1U);
  const size_t probability_offset =
      static_cast<size_t>(row) * vocab_size;
  const size_t state_offset = static_cast<size_t>(row) * 2U;

  __shared__ float warp_sums[kWarps];
  __shared__ float warp_prefixes[kWarps];
  __shared__ float row_sum;
  __shared__ double uniform;
  __shared__ uint32_t candidate_token;
  __shared__ uint32_t target_warp;
  __shared__ uint32_t last_positive_token;
  __shared__ uint32_t invalid_probability;
  __shared__ uint32_t has_positive_probability;
  __shared__ uint32_t valid_state;

  if (thread == 0U) {
    row_sum = 0.0;
    uniform = 0.5;
    candidate_token = vocab_size;
    target_warp = kWarps;
    last_positive_token = 0U;
    invalid_probability = 0U;
    has_positive_probability = 0U;
    const int64_t seed = rng_state[state_offset];
    const int64_t counter = rng_state[state_offset + 1U];
    valid_state =
        seed >= 0 && counter >= 0 &&
                counter < std::numeric_limits<int64_t>::max()
            ? 1U
            : 0U;
    if (valid_state != 0U) {
      const uint32_t random_word = philox4x32_10_first(
          static_cast<uint64_t>(seed), static_cast<uint64_t>(counter));
      uniform = (static_cast<double>(random_word) + 0.5) * kUint32Unit;
    }
  }
  __syncthreads();

  const uint32_t chunks_per_warp =
      (vocab_size + kThreads - 1U) / kThreads;
  const uint32_t warp_start =
      warp * chunks_per_warp * kWarpSize;
  const uint32_t warp_end =
      min(vocab_size, warp_start + chunks_per_warp * kWarpSize);

  float local_sum = 0.0F;
  bool local_invalid = false;
  bool local_positive = false;
  uint32_t local_last_positive = 0U;
  for (uint32_t token = warp_start + lane; token < warp_end;
       token += kWarpSize) {
    const float probability = probabilities[probability_offset + token];
    const bool finite_non_negative =
        isfinite(probability) && probability >= 0.0F;
    local_invalid |= !finite_non_negative;
    if (finite_non_negative && probability > 0.0F) {
      local_positive = true;
      local_last_positive = token;
    }
    local_sum += probability;
  }

  const float segment_sum = warp_sum(local_sum);
  const uint32_t segment_last_positive =
      warp_max(local_last_positive);
  const uint32_t invalid_lanes =
      __ballot_sync(0xffffffffU, local_invalid);
  const uint32_t positive_lanes =
      __ballot_sync(0xffffffffU, local_positive);
  if (lane == 0U) {
    warp_sums[warp] = segment_sum;
    if (invalid_lanes != 0U) {
      atomicExch(&invalid_probability, 1U);
    }
    if (positive_lanes != 0U) {
      atomicExch(&has_positive_probability, 1U);
      atomicMax(&last_positive_token, segment_last_positive);
    }
  }
  __syncthreads();

  if (thread == 0U) {
    float prefix = 0.0F;
    for (uint32_t index = 0U; index < kWarps; ++index) {
      warp_prefixes[index] = prefix;
      const double next = prefix + warp_sums[index];
      if (target_warp == kWarps && next > uniform) {
        target_warp = index;
      }
      prefix = next;
    }
    row_sum = prefix;
  }
  __syncthreads();

  if (warp == target_warp) {
    float prefix = warp_prefixes[warp];
    for (uint32_t start = warp_start; start < warp_end;
         start += kWarpSize) {
      const uint32_t token = start + lane;
      const float probability =
          token < warp_end
              ? probabilities[probability_offset + token]
              : 0.0F;
      const float inclusive_probability =
          warp_inclusive_sum(probability);
      const uint32_t crossing_lanes = __ballot_sync(
          0xffffffffU,
          token < warp_end &&
              static_cast<double>(prefix + inclusive_probability) >
                  uniform);
      if (crossing_lanes != 0U) {
        if (lane == 0U) {
          candidate_token =
              start + static_cast<uint32_t>(__ffs(crossing_lanes) - 1);
        }
        break;
      }
      prefix += __shfl_sync(0xffffffffU, inclusive_probability, 31);
    }
  }
  __syncthreads();

  if (thread == 0U) {
    const bool valid_probabilities =
        invalid_probability == 0U && has_positive_probability != 0U &&
        isfinite(row_sum) &&
        fabs(static_cast<double>(row_sum) - 1.0) <=
            kProbabilitySumTolerance;
    if (valid_state != 0U && valid_probabilities) {
      token_ids[row] =
          candidate_token == vocab_size
              ? static_cast<int64_t>(last_positive_token)
              : static_cast<int64_t>(candidate_token);
      rng_state[state_offset + 1U] += 1;
    }
  }
}

}  // namespace

extern "C" int loom_cuda_categorical_sample_f32(
    const float* probabilities, int64_t* rng_state, int64_t* token_ids,
    uint32_t rows, uint32_t vocab_size, void* stream) {
  if (probabilities == nullptr || rng_state == nullptr ||
      token_ids == nullptr || rows == 0U || vocab_size == 0U ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      vocab_size >
          static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      static_cast<size_t>(rows) >
          std::numeric_limits<size_t>::max() /
              static_cast<size_t>(vocab_size)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  categorical_sample_kernel<<<rows, kThreads, 0,
                              static_cast<cudaStream_t>(stream)>>>(
      probabilities, rng_state, token_ids, vocab_size);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}
