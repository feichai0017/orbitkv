#include "loom_cuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr unsigned long long kEmptyEntry = 0x00000000ffffffffULL;
constexpr uint32_t kPromptPresent = 0x80000000U;
constexpr uint32_t kOutputCountMask = 0x7fffffffU;

__device__ __forceinline__ uint32_t hash_token(uint32_t token) {
  token ^= token >> 16;
  token *= 0x7feb352dU;
  token ^= token >> 15;
  token *= 0x846ca68bU;
  return token ^ (token >> 16);
}

__device__ __forceinline__ unsigned long long pack_entry(uint32_t token,
                                                          uint32_t state) {
  return static_cast<unsigned long long>(token) |
         (static_cast<unsigned long long>(state) << 32);
}

__device__ __forceinline__ void insert_token(
    unsigned long long* workspace, uint32_t capacity, uint32_t token,
    bool prompt_token) {
  const uint32_t mask = capacity - 1;
  uint32_t slot = hash_token(token) & mask;
  const uint32_t initial_state = prompt_token ? kPromptPresent : 1U;

  while (true) {
    unsigned long long* entry = workspace + slot;
    unsigned long long observed =
        atomicCAS(entry, kEmptyEntry, pack_entry(token, initial_state));
    if (observed == kEmptyEntry) {
      return;
    }
    if (static_cast<uint32_t>(observed) == token) {
      while (true) {
        const uint32_t state = static_cast<uint32_t>(observed >> 32);
        const uint32_t updated_state =
            prompt_token ? (state | kPromptPresent) : (state + 1U);
        const unsigned long long updated = pack_entry(token, updated_state);
        const unsigned long long previous = atomicCAS(entry, observed, updated);
        if (previous == observed) {
          return;
        }
        observed = previous;
      }
    }
    slot = (slot + 1) & mask;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads) void apply_token_penalties_kernel(
    float* logits, const int64_t* prompt_token_ids,
    const int64_t* output_token_ids, const float* presence_penalties,
    const float* frequency_penalties, const float* repetition_penalties,
    unsigned long long* workspace, uint32_t vocab_size,
    uint32_t prompt_tokens, uint32_t output_tokens,
    uint32_t workspace_capacity, uint64_t logits_row_stride,
    uint64_t prompt_row_stride, uint64_t output_row_stride,
    uint64_t workspace_row_stride) {
  const uint32_t row = blockIdx.x;
  unsigned long long* row_workspace =
      workspace + static_cast<uint64_t>(row) * workspace_row_stride;

  for (uint32_t slot = threadIdx.x; slot < workspace_capacity;
       slot += Threads) {
    row_workspace[slot] = kEmptyEntry;
  }
  __syncthreads();

  const int64_t* row_prompt =
      prompt_token_ids + static_cast<uint64_t>(row) * prompt_row_stride;
  for (uint32_t index = threadIdx.x; index < prompt_tokens;
       index += Threads) {
    const int64_t token = row_prompt[index];
    if (static_cast<uint64_t>(token) < vocab_size) {
      insert_token(row_workspace, workspace_capacity,
                   static_cast<uint32_t>(token), true);
    }
  }

  const int64_t* row_output =
      output_token_ids + static_cast<uint64_t>(row) * output_row_stride;
  for (uint32_t index = threadIdx.x; index < output_tokens;
       index += Threads) {
    const int64_t token = row_output[index];
    if (static_cast<uint64_t>(token) < vocab_size) {
      insert_token(row_workspace, workspace_capacity,
                   static_cast<uint32_t>(token), false);
    }
  }
  __syncthreads();

  float* row_logits =
      logits + static_cast<uint64_t>(row) * logits_row_stride;
  const float presence = presence_penalties[row];
  const float frequency = frequency_penalties[row];
  const float repetition = repetition_penalties[row];
  for (uint32_t slot = threadIdx.x; slot < workspace_capacity;
       slot += Threads) {
    const unsigned long long entry = row_workspace[slot];
    const uint32_t token = static_cast<uint32_t>(entry);
    if (token >= vocab_size) {
      continue;
    }
    const uint32_t output_count =
        static_cast<uint32_t>(entry >> 32) & kOutputCountMask;
    float value = row_logits[token];
    value = value > 0.0F ? value / repetition : value * repetition;
    if (output_count != 0) {
      value -= frequency * static_cast<float>(output_count);
      value -= presence;
    }
    row_logits[token] = value;
  }
}

}  // namespace

extern "C" int loom_cuda_apply_token_penalties_f32(
    float* logits, const int64_t* prompt_token_ids,
    const int64_t* output_token_ids, const float* presence_penalties,
    const float* frequency_penalties, const float* repetition_penalties,
    uint64_t* workspace, uint32_t rows, uint32_t vocab_size,
    uint32_t prompt_tokens, uint32_t output_tokens,
    uint32_t workspace_capacity, uint64_t logits_row_stride,
    uint64_t prompt_row_stride, uint64_t output_row_stride,
    uint64_t workspace_row_stride, void* stream) {
  if (logits == nullptr || prompt_token_ids == nullptr ||
      output_token_ids == nullptr || presence_penalties == nullptr ||
      frequency_penalties == nullptr || repetition_penalties == nullptr ||
      workspace == nullptr || rows == 0 || vocab_size == 0 ||
      prompt_tokens == 0 || output_tokens == 0 ||
      workspace_capacity == 0 ||
      (workspace_capacity & (workspace_capacity - 1)) != 0 ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      vocab_size > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      output_tokens >
          static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      static_cast<uint64_t>(workspace_capacity) <
          2ULL * (static_cast<uint64_t>(prompt_tokens) + output_tokens) ||
      logits_row_stride < vocab_size ||
      prompt_row_stride < prompt_tokens ||
      output_row_stride < output_tokens ||
      workspace_row_stride < workspace_capacity) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  apply_token_penalties_kernel<256><<<rows, 256, 0,
                                      static_cast<cudaStream_t>(stream)>>>(
      logits, prompt_token_ids, output_token_ids, presence_penalties,
      frequency_penalties, repetition_penalties,
      reinterpret_cast<unsigned long long*>(workspace), vocab_size,
      prompt_tokens, output_tokens, workspace_capacity, logits_row_stride,
      prompt_row_stride, output_row_stride, workspace_row_stride);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}
