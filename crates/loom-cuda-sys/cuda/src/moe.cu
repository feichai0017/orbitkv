#include "loom_cuda.h"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxBlocks = 65535;
constexpr size_t kCubAlignment = 256;

struct PermuteWorkspaceLayout {
  uint32_t* keys_in;
  uint32_t* keys_out;
  int32_t* values_in;
  int32_t* values_out;
  void* cub_storage;
  size_t cub_storage_bytes;
  size_t total_bytes;
};

bool checked_add(size_t left, size_t right, size_t* result) {
  if (left > std::numeric_limits<size_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool checked_mul(size_t left, size_t right, size_t* result) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool align_up(size_t value, size_t alignment, size_t* result) {
  size_t biased = 0;
  if (!checked_add(value, alignment - 1U, &biased)) {
    return false;
  }
  *result = biased & ~(alignment - 1U);
  return true;
}

int sort_end_bit(uint32_t num_experts) {
  uint64_t maximum_key = static_cast<uint64_t>(num_experts) * 2U - 1U;
  int bits = 0;
  do {
    ++bits;
    maximum_key >>= 1U;
  } while (maximum_key != 0U);
  return bits;
}

int workspace_layout(uint8_t* workspace, uint32_t assignments,
                     uint32_t num_experts,
                     PermuteWorkspaceLayout* layout) {
  if (assignments == 0 ||
      assignments > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      num_experts == 0 ||
      num_experts >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      layout == nullptr) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  size_t array_bytes = 0;
  size_t metadata_bytes = 0;
  size_t cub_offset = 0;
  if (!checked_mul(static_cast<size_t>(assignments), sizeof(int32_t),
                   &array_bytes) ||
      !checked_mul(array_bytes, 4U, &metadata_bytes) ||
      !align_up(metadata_bytes, kCubAlignment, &cub_offset)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  size_t cub_storage_bytes = 0;
  const cudaError_t query_status = cub::DeviceRadixSort::SortPairs(
      nullptr, cub_storage_bytes, static_cast<uint32_t*>(nullptr),
      static_cast<uint32_t*>(nullptr), static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr), static_cast<int>(assignments), 0,
      sort_end_bit(num_experts));
  if (query_status != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }
  size_t total_bytes = 0;
  if (!checked_add(cub_offset, cub_storage_bytes, &total_bytes)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  layout->keys_in = workspace == nullptr
                        ? nullptr
                        : reinterpret_cast<uint32_t*>(workspace);
  layout->keys_out = workspace == nullptr
                         ? nullptr
                         : reinterpret_cast<uint32_t*>(workspace + array_bytes);
  layout->values_in =
      workspace == nullptr
          ? nullptr
          : reinterpret_cast<int32_t*>(workspace + 2U * array_bytes);
  layout->values_out =
      workspace == nullptr
          ? nullptr
          : reinterpret_cast<int32_t*>(workspace + 3U * array_bytes);
  layout->cub_storage = workspace == nullptr ? nullptr : workspace + cub_offset;
  layout->cub_storage_bytes = cub_storage_bytes;
  layout->total_bytes = total_bytes;
  return LOOM_CUDA_SUCCESS;
}

uint32_t launch_blocks(size_t work_items) {
  const size_t required = (work_items + kThreads - 1U) / kThreads;
  return static_cast<uint32_t>(
      std::min(required, static_cast<size_t>(kMaxBlocks)));
}

__global__ void initialize_assignments_kernel(
    const int32_t* topk_ids, const int32_t* expert_map, uint32_t* keys,
    int32_t* values, uint32_t assignments, uint32_t num_experts,
    uint32_t num_local_experts) {
  for (uint32_t assignment = blockIdx.x * blockDim.x + threadIdx.x;
       assignment < assignments;
       assignment += blockDim.x * gridDim.x) {
    const int32_t global_expert = topk_ids[assignment];
    uint32_t sort_key = num_experts * 2U - 1U;
    if (global_expert >= 0 &&
        static_cast<uint32_t>(global_expert) < num_experts) {
      const int32_t local_expert =
          expert_map == nullptr ? global_expert : expert_map[global_expert];
      if (local_expert < 0 ||
          static_cast<uint32_t>(local_expert) >= num_local_experts) {
        sort_key = num_experts + static_cast<uint32_t>(global_expert);
      } else {
        sort_key = static_cast<uint32_t>(local_expert);
      }
    }
    keys[assignment] = sort_key;
    values[assignment] = static_cast<int32_t>(assignment);
  }
}

__global__ void expert_offsets_kernel(const uint32_t* sorted_experts,
                                      int64_t* expert_offsets,
                                      uint32_t assignments,
                                      uint32_t num_local_experts) {
  for (uint32_t boundary = blockIdx.x * blockDim.x + threadIdx.x;
       boundary <= num_local_experts;
       boundary += blockDim.x * gridDim.x) {
    uint32_t left = 0;
    uint32_t right = assignments;
    while (left < right) {
      const uint32_t middle = left + (right - left) / 2U;
      if (sorted_experts[middle] < boundary) {
        left = middle + 1U;
      } else {
        right = middle;
      }
    }
    expert_offsets[boundary] = static_cast<int64_t>(left);
  }
}

template <typename T>
__global__ void gather_hidden_kernel(
    const T* hidden_states, const uint32_t* sorted_experts,
    const int32_t* sorted_assignments, T* permuted_hidden_states,
    int32_t* inverse_permutation, int32_t* permuted_assignment_ids,
    size_t output_elements, uint32_t hidden_size, uint32_t top_k,
    uint32_t assignments, uint32_t num_local_experts) {
  for (size_t index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < output_elements;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const uint32_t permuted_row =
        static_cast<uint32_t>(index / hidden_size);
    const uint32_t column = static_cast<uint32_t>(index % hidden_size);
    const int32_t assignment = sorted_assignments[permuted_row];
    const bool is_local = sorted_experts[permuted_row] < num_local_experts;
    if (column == 0U) {
      inverse_permutation[assignment] = static_cast<int32_t>(permuted_row);
      permuted_assignment_ids[permuted_row] =
          is_local ? assignment : static_cast<int32_t>(assignments);
    }
    if (is_local) {
      const uint32_t token = static_cast<uint32_t>(assignment) / top_k;
      permuted_hidden_states[index] =
          hidden_states[static_cast<size_t>(token) * hidden_size + column];
    } else {
      permuted_hidden_states[index] = T{};
    }
  }
}

template <typename T>
__global__ void gather_hidden_vector_kernel(
    const T* hidden_states, const uint32_t* sorted_experts,
    const int32_t* sorted_assignments, T* permuted_hidden_states,
    int32_t* inverse_permutation, int32_t* permuted_assignment_ids,
    uint32_t assignments, uint32_t vectors_per_row, uint32_t top_k,
    uint32_t num_local_experts) {
  const auto* input = reinterpret_cast<const uint4*>(hidden_states);
  auto* output = reinterpret_cast<uint4*>(permuted_hidden_states);
  for (uint32_t permuted_row = blockIdx.x; permuted_row < assignments;
       permuted_row += gridDim.x) {
    const int32_t assignment = sorted_assignments[permuted_row];
    const bool is_local = sorted_experts[permuted_row] < num_local_experts;
    if (threadIdx.x == 0U) {
      inverse_permutation[assignment] = static_cast<int32_t>(permuted_row);
      permuted_assignment_ids[permuted_row] =
          is_local ? assignment : static_cast<int32_t>(assignments);
    }
    const uint32_t token = static_cast<uint32_t>(assignment) / top_k;
    for (uint32_t vector_column = threadIdx.x;
         vector_column < vectors_per_row; vector_column += blockDim.x) {
      const size_t output_index =
          static_cast<size_t>(permuted_row) * vectors_per_row + vector_column;
      output[output_index] =
          is_local
              ? input[static_cast<size_t>(token) * vectors_per_row +
                      vector_column]
              : make_uint4(0U, 0U, 0U, 0U);
    }
  }
}

template <typename T>
__device__ float to_float(T value);

template <>
__device__ float to_float<float>(float value) {
  return value;
}

template <>
__device__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ T from_float(float value);

template <>
__device__ float from_float<float>(float value) {
  return value;
}

template <>
__device__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void combine_kernel(
    const T* expert_outputs, const float* routing_weights,
    const int32_t* inverse_permutation, const int64_t* expert_offsets,
    T* output, size_t output_elements, uint32_t hidden_size, uint32_t top_k,
    uint32_t num_local_experts, uint32_t assignments) {
  int64_t valid_value = expert_offsets[num_local_experts];
  if (valid_value < 0) {
    valid_value = 0;
  } else if (valid_value > static_cast<int64_t>(assignments)) {
    valid_value = static_cast<int64_t>(assignments);
  }
  const uint32_t valid_assignments = static_cast<uint32_t>(valid_value);
  for (size_t index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < output_elements;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const uint32_t token = static_cast<uint32_t>(index / hidden_size);
    const uint32_t column = static_cast<uint32_t>(index % hidden_size);
    float accumulator = 0.0F;
    for (uint32_t route = 0; route < top_k; ++route) {
      const uint32_t assignment = token * top_k + route;
      const int32_t permuted_row = inverse_permutation[assignment];
      if (permuted_row >= 0 &&
          static_cast<uint32_t>(permuted_row) < valid_assignments) {
        accumulator = fmaf(
            routing_weights[assignment],
            to_float(expert_outputs[static_cast<size_t>(permuted_row) *
                                        hidden_size +
                                    column]),
            accumulator);
      }
    }
    output[index] = from_float<T>(accumulator);
  }
}

template <typename T>
union PackedVector {
  uint4 words;
  T elements[sizeof(uint4) / sizeof(T)];
};

template <typename T>
__global__ void combine_vector_kernel(
    const T* expert_outputs, const float* routing_weights,
    const int32_t* inverse_permutation, const int64_t* expert_offsets,
    T* output, uint32_t tokens, uint32_t vectors_per_row, uint32_t top_k,
    uint32_t num_local_experts, uint32_t assignments) {
  constexpr uint32_t kElementsPerVector = sizeof(uint4) / sizeof(T);
  const auto* input = reinterpret_cast<const uint4*>(expert_outputs);
  auto* result = reinterpret_cast<uint4*>(output);
  int64_t valid_value = expert_offsets[num_local_experts];
  if (valid_value < 0) {
    valid_value = 0;
  } else if (valid_value > static_cast<int64_t>(assignments)) {
    valid_value = static_cast<int64_t>(assignments);
  }
  const uint32_t valid_assignments = static_cast<uint32_t>(valid_value);
  for (uint32_t token = blockIdx.x; token < tokens; token += gridDim.x) {
    for (uint32_t vector_column = threadIdx.x;
         vector_column < vectors_per_row; vector_column += blockDim.x) {
      float accumulators[kElementsPerVector]{};
      for (uint32_t route = 0; route < top_k; ++route) {
        const uint32_t assignment = token * top_k + route;
        const int32_t permuted_row = inverse_permutation[assignment];
        if (permuted_row >= 0 &&
            static_cast<uint32_t>(permuted_row) < valid_assignments) {
          PackedVector<T> values{};
          values.words =
              input[static_cast<size_t>(permuted_row) * vectors_per_row +
                    vector_column];
          const float weight = routing_weights[assignment];
#pragma unroll
          for (uint32_t element = 0; element < kElementsPerVector; ++element) {
            accumulators[element] =
                fmaf(weight, to_float(values.elements[element]),
                     accumulators[element]);
          }
        }
      }
      PackedVector<T> values{};
#pragma unroll
      for (uint32_t element = 0; element < kElementsPerVector; ++element) {
        values.elements[element] = from_float<T>(accumulators[element]);
      }
      result[static_cast<size_t>(token) * vectors_per_row + vector_column] =
          values.words;
    }
  }
}

template <typename T>
int launch_permute(const T* hidden_states, const int32_t* topk_ids,
                   const int32_t* expert_map, T* permuted_hidden_states,
                   int64_t* expert_offsets, int32_t* inverse_permutation,
                   int32_t* permuted_assignment_ids, uint8_t* workspace,
                   uint64_t workspace_bytes, uint32_t tokens,
                   uint32_t hidden_size, uint32_t top_k,
                   uint32_t num_experts, uint32_t num_local_experts,
                   void* stream) {
  if (hidden_states == nullptr || topk_ids == nullptr ||
      permuted_hidden_states == nullptr || expert_offsets == nullptr ||
      inverse_permutation == nullptr || permuted_assignment_ids == nullptr ||
      workspace == nullptr || tokens == 0 || hidden_size == 0 || top_k == 0 ||
      num_experts == 0 || num_local_experts == 0 || top_k > num_experts ||
      num_local_experts > num_experts ||
      (expert_map == nullptr && num_local_experts != num_experts) ||
      num_experts >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      num_local_experts >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const uint64_t assignments_u64 =
      static_cast<uint64_t>(tokens) * static_cast<uint64_t>(top_k);
  if (assignments_u64 == 0 ||
      assignments_u64 >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const uint32_t assignments = static_cast<uint32_t>(assignments_u64);
  size_t output_elements = 0;
  if (!checked_mul(static_cast<size_t>(assignments),
                   static_cast<size_t>(hidden_size), &output_elements)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  PermuteWorkspaceLayout layout{};
  const int layout_status = workspace_layout(
      workspace, assignments, num_experts, &layout);
  if (layout_status != LOOM_CUDA_SUCCESS ||
      workspace_bytes < layout.total_bytes ||
      reinterpret_cast<uintptr_t>(workspace) % kCubAlignment != 0U) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  initialize_assignments_kernel<<<launch_blocks(assignments), kThreads, 0,
                                  cuda_stream>>>(
      topk_ids, expert_map, layout.keys_in, layout.values_in, assignments,
      num_experts, num_local_experts);
  if (cudaGetLastError() != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }

  const cudaError_t sort_status = cub::DeviceRadixSort::SortPairs(
      layout.cub_storage, layout.cub_storage_bytes, layout.keys_in,
      layout.keys_out, layout.values_in, layout.values_out,
      static_cast<int>(assignments), 0, sort_end_bit(num_experts),
      cuda_stream);
  if (sort_status != cudaSuccess) {
    return LOOM_CUDA_LAUNCH_ERROR;
  }

  expert_offsets_kernel<<<launch_blocks(num_local_experts + 1U), kThreads, 0,
                          cuda_stream>>>(layout.keys_out, expert_offsets,
                                         assignments, num_local_experts);
  const size_t row_bytes = static_cast<size_t>(hidden_size) * sizeof(T);
  const bool vectorized = row_bytes % sizeof(uint4) == 0U &&
                          reinterpret_cast<uintptr_t>(hidden_states) %
                                  alignof(uint4) ==
                              0U &&
                          reinterpret_cast<uintptr_t>(permuted_hidden_states) %
                                  alignof(uint4) ==
                              0U;
  if (vectorized) {
    const uint32_t vectors_per_row =
        static_cast<uint32_t>(row_bytes / sizeof(uint4));
    gather_hidden_vector_kernel<<<std::min(assignments, kMaxBlocks), kThreads,
                                  0, cuda_stream>>>(
        hidden_states, layout.keys_out, layout.values_out,
        permuted_hidden_states, inverse_permutation, permuted_assignment_ids,
        assignments, vectors_per_row, top_k, num_local_experts);
  } else {
    gather_hidden_kernel<<<launch_blocks(output_elements), kThreads, 0,
                           cuda_stream>>>(
        hidden_states, layout.keys_out, layout.values_out,
        permuted_hidden_states, inverse_permutation, permuted_assignment_ids,
        output_elements, hidden_size, top_k, assignments, num_local_experts);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

template <typename T>
int launch_combine(const T* expert_outputs, const float* routing_weights,
                   const int32_t* inverse_permutation,
                   const int64_t* expert_offsets, T* output, uint32_t tokens,
                   uint32_t hidden_size, uint32_t top_k,
                   uint32_t num_local_experts, void* stream) {
  if (expert_outputs == nullptr || routing_weights == nullptr ||
      inverse_permutation == nullptr || expert_offsets == nullptr ||
      output == nullptr || tokens == 0 || hidden_size == 0 || top_k == 0 ||
      num_local_experts == 0) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const uint64_t assignments_u64 =
      static_cast<uint64_t>(tokens) * static_cast<uint64_t>(top_k);
  if (assignments_u64 == 0 ||
      assignments_u64 >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  size_t output_elements = 0;
  if (!checked_mul(static_cast<size_t>(tokens),
                   static_cast<size_t>(hidden_size), &output_elements)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  const size_t row_bytes = static_cast<size_t>(hidden_size) * sizeof(T);
  const bool vectorized = row_bytes % sizeof(uint4) == 0U &&
                          reinterpret_cast<uintptr_t>(expert_outputs) %
                                  alignof(uint4) ==
                              0U &&
                          reinterpret_cast<uintptr_t>(output) % alignof(uint4) ==
                              0U;
  if (vectorized) {
    const uint32_t vectors_per_row =
        static_cast<uint32_t>(row_bytes / sizeof(uint4));
    combine_vector_kernel<<<std::min(tokens, kMaxBlocks), kThreads, 0,
                            cuda_stream>>>(
        expert_outputs, routing_weights, inverse_permutation, expert_offsets,
        output, tokens, vectors_per_row, top_k, num_local_experts,
        static_cast<uint32_t>(assignments_u64));
  } else {
    combine_kernel<<<launch_blocks(output_elements), kThreads, 0,
                     cuda_stream>>>(
        expert_outputs, routing_weights, inverse_permutation, expert_offsets,
        output, output_elements, hidden_size, top_k, num_local_experts,
        static_cast<uint32_t>(assignments_u64));
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_moe_permute_workspace_size(
    uint32_t assignments, uint32_t num_experts,
    uint64_t* workspace_bytes) {
  if (workspace_bytes == nullptr) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  PermuteWorkspaceLayout layout{};
  const int status = workspace_layout(
      nullptr, assignments, num_experts, &layout);
  if (status != LOOM_CUDA_SUCCESS) {
    return status;
  }
  *workspace_bytes = static_cast<uint64_t>(layout.total_bytes);
  return LOOM_CUDA_SUCCESS;
}

extern "C" int loom_cuda_moe_permute_f32(
    const float* hidden_states, const int32_t* topk_ids,
    const int32_t* expert_map, float* permuted_hidden_states,
    int64_t* expert_offsets, int32_t* inverse_permutation,
    int32_t* permuted_assignment_ids, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t tokens, uint32_t hidden_size,
    uint32_t top_k, uint32_t num_experts, uint32_t num_local_experts,
    void* stream) {
  return launch_permute(
      hidden_states, topk_ids, expert_map, permuted_hidden_states,
      expert_offsets, inverse_permutation, permuted_assignment_ids, workspace,
      workspace_bytes, tokens, hidden_size, top_k, num_experts,
      num_local_experts, stream);
}

extern "C" int loom_cuda_moe_permute_f16(
    const uint16_t* hidden_states, const int32_t* topk_ids,
    const int32_t* expert_map, uint16_t* permuted_hidden_states,
    int64_t* expert_offsets, int32_t* inverse_permutation,
    int32_t* permuted_assignment_ids, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t tokens, uint32_t hidden_size,
    uint32_t top_k, uint32_t num_experts, uint32_t num_local_experts,
    void* stream) {
  return launch_permute(
      reinterpret_cast<const __half*>(hidden_states), topk_ids, expert_map,
      reinterpret_cast<__half*>(permuted_hidden_states), expert_offsets,
      inverse_permutation, permuted_assignment_ids, workspace,
      workspace_bytes, tokens, hidden_size, top_k, num_experts,
      num_local_experts, stream);
}

extern "C" int loom_cuda_moe_permute_bf16(
    const uint16_t* hidden_states, const int32_t* topk_ids,
    const int32_t* expert_map, uint16_t* permuted_hidden_states,
    int64_t* expert_offsets, int32_t* inverse_permutation,
    int32_t* permuted_assignment_ids, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t tokens, uint32_t hidden_size,
    uint32_t top_k, uint32_t num_experts, uint32_t num_local_experts,
    void* stream) {
  return launch_permute(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states), topk_ids,
      expert_map, reinterpret_cast<__nv_bfloat16*>(permuted_hidden_states),
      expert_offsets, inverse_permutation, permuted_assignment_ids, workspace,
      workspace_bytes, tokens, hidden_size, top_k, num_experts,
      num_local_experts, stream);
}

extern "C" int loom_cuda_moe_permute_fp8_e4m3fn(
    const uint8_t* hidden_states, const int32_t* topk_ids,
    const int32_t* expert_map, uint8_t* permuted_hidden_states,
    int64_t* expert_offsets, int32_t* inverse_permutation,
    int32_t* permuted_assignment_ids, uint8_t* workspace,
    uint64_t workspace_bytes, uint32_t tokens, uint32_t hidden_size,
    uint32_t top_k, uint32_t num_experts, uint32_t num_local_experts,
    void* stream) {
  return launch_permute(
      hidden_states, topk_ids, expert_map, permuted_hidden_states,
      expert_offsets, inverse_permutation, permuted_assignment_ids, workspace,
      workspace_bytes, tokens, hidden_size, top_k, num_experts,
      num_local_experts, stream);
}

extern "C" int loom_cuda_moe_combine_f32(
    const float* expert_outputs, const float* routing_weights,
    const int32_t* inverse_permutation, const int64_t* expert_offsets,
    float* output, uint32_t tokens, uint32_t hidden_size, uint32_t top_k,
    uint32_t num_local_experts, void* stream) {
  return launch_combine(expert_outputs, routing_weights, inverse_permutation,
                        expert_offsets, output, tokens, hidden_size, top_k,
                        num_local_experts, stream);
}

extern "C" int loom_cuda_moe_combine_f16(
    const uint16_t* expert_outputs, const float* routing_weights,
    const int32_t* inverse_permutation, const int64_t* expert_offsets,
    uint16_t* output, uint32_t tokens, uint32_t hidden_size, uint32_t top_k,
    uint32_t num_local_experts, void* stream) {
  return launch_combine(
      reinterpret_cast<const __half*>(expert_outputs), routing_weights,
      inverse_permutation, expert_offsets, reinterpret_cast<__half*>(output),
      tokens, hidden_size, top_k, num_local_experts, stream);
}

extern "C" int loom_cuda_moe_combine_bf16(
    const uint16_t* expert_outputs, const float* routing_weights,
    const int32_t* inverse_permutation, const int64_t* expert_offsets,
    uint16_t* output, uint32_t tokens, uint32_t hidden_size, uint32_t top_k,
    uint32_t num_local_experts, void* stream) {
  return launch_combine(
      reinterpret_cast<const __nv_bfloat16*>(expert_outputs), routing_weights,
      inverse_permutation, expert_offsets,
      reinterpret_cast<__nv_bfloat16*>(output), tokens, hidden_size, top_k,
      num_local_experts, stream);
}
