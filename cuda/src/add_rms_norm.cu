#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kAddRmsNormThreads = 256;
constexpr int kLowPrecisionThreads = 512;
constexpr int kWideVectorElements = 8;

__global__ __launch_bounds__(kAddRmsNormThreads) void add_rms_norm_f32_kernel(
    float* input, float* residual, const float* weight, uint32_t hidden_size,
    float epsilon) {
  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_size;
  float local_sum = 0.0F;
  for (uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const size_t offset = row_offset + column;
    const float sum = input[offset] + residual[offset];
    residual[offset] = sum;
    local_sum = fmaf(sum, sum, local_sum);
  }

  using BlockReduce = cub::BlockReduce<float, kAddRmsNormThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float inverse_rms;
  const float square_sum = BlockReduce(reduce_storage).Sum(local_sum);
  if (threadIdx.x == 0) {
    inverse_rms =
        rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();

  for (uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const size_t offset = row_offset + column;
    input[offset] = residual[offset] * inverse_rms * weight[column];
  }
}

struct HalfOps {
  using Scalar = __half;
  using Pair = __half2;

  __device__ static float to_float(Scalar value) {
    return __half2float(value);
  }

  __device__ static Scalar from_float(float value) {
    return __float2half_rn(value);
  }

  __device__ static float2 to_float2(Pair value) {
    return __half22float2(value);
  }

  __device__ static Pair from_float2(float2 value) {
    return __float22half2_rn(value);
  }
};

struct Bfloat16Ops {
  using Scalar = __nv_bfloat16;
  using Pair = __nv_bfloat162;

  __device__ static float to_float(Scalar value) {
    return __bfloat162float(value);
  }

  __device__ static Scalar from_float(float value) {
    return __float2bfloat16_rn(value);
  }

  __device__ static float2 to_float2(Pair value) {
    return __bfloat1622float2(value);
  }

  __device__ static Pair from_float2(float2 value) {
    return __float22bfloat162_rn(value);
  }
};

template <typename Scalar, int Width>
struct alignas(sizeof(Scalar) * Width) AlignedPack {
  Scalar values[Width];
};

template <typename Ops, int VectorWidth>
__global__ __launch_bounds__(kLowPrecisionThreads)
    void add_rms_norm_low_precision_kernel(typename Ops::Scalar* input,
                                           typename Ops::Scalar* residual,
                                           const typename Ops::Scalar* weight,
                                           uint32_t hidden_size,
                                           float epsilon) {
  using Scalar = typename Ops::Scalar;
  using Pair = typename Ops::Pair;

  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_size;
  Scalar* row_input = input + row_offset;
  Scalar* row_residual = residual + row_offset;
  float local_sum = 0.0F;

  if constexpr (VectorWidth == kWideVectorElements) {
    using Pack = AlignedPack<Scalar, kWideVectorElements>;
    static_assert(sizeof(Pack) == 16);
    auto* input_packs = reinterpret_cast<Pack*>(row_input);
    auto* residual_packs = reinterpret_cast<Pack*>(row_residual);
    const uint32_t pack_count = hidden_size / kWideVectorElements;
    for (uint32_t pack_column = threadIdx.x; pack_column < pack_count;
         pack_column += blockDim.x) {
      const Pack input_value = input_packs[pack_column];
      const Pack residual_value = residual_packs[pack_column];
      Pack quantized_sum;
#pragma unroll
      for (int element = 0; element < kWideVectorElements; ++element) {
        quantized_sum.values[element] = Ops::from_float(
            Ops::to_float(input_value.values[element]) +
            Ops::to_float(residual_value.values[element]));
        const float sum = Ops::to_float(quantized_sum.values[element]);
        local_sum = fmaf(sum, sum, local_sum);
      }
      residual_packs[pack_column] = quantized_sum;
    }
  } else if constexpr (VectorWidth == 2) {
    auto* input_pairs = reinterpret_cast<Pair*>(row_input);
    auto* residual_pairs = reinterpret_cast<Pair*>(row_residual);
    const uint32_t pair_count = hidden_size / 2U;
    for (uint32_t pair_column = threadIdx.x; pair_column < pair_count;
         pair_column += blockDim.x) {
      const float2 input_value = Ops::to_float2(input_pairs[pair_column]);
      const float2 residual_value =
          Ops::to_float2(residual_pairs[pair_column]);
      const Pair quantized_sum = Ops::from_float2(
          make_float2(input_value.x + residual_value.x,
                      input_value.y + residual_value.y));
      residual_pairs[pair_column] = quantized_sum;
      const float2 sum = Ops::to_float2(quantized_sum);
      local_sum = fmaf(sum.x, sum.x, local_sum);
      local_sum = fmaf(sum.y, sum.y, local_sum);
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const Scalar quantized_sum = Ops::from_float(
          Ops::to_float(row_input[column]) +
          Ops::to_float(row_residual[column]));
      row_residual[column] = quantized_sum;
      const float sum = Ops::to_float(quantized_sum);
      local_sum = fmaf(sum, sum, local_sum);
    }
  }

  using BlockReduce = cub::BlockReduce<float, kLowPrecisionThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float inverse_rms;
  const float square_sum = BlockReduce(reduce_storage).Sum(local_sum);
  if (threadIdx.x == 0) {
    inverse_rms =
        rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();

  if constexpr (VectorWidth == kWideVectorElements) {
    using Pack = AlignedPack<Scalar, kWideVectorElements>;
    auto* input_packs = reinterpret_cast<Pack*>(row_input);
    const auto* residual_packs = reinterpret_cast<const Pack*>(row_residual);
    const auto* weight_packs = reinterpret_cast<const Pack*>(weight);
    const uint32_t pack_count = hidden_size / kWideVectorElements;
    for (uint32_t pack_column = threadIdx.x; pack_column < pack_count;
         pack_column += blockDim.x) {
      const Pack sum = residual_packs[pack_column];
      const Pack scale = weight_packs[pack_column];
      Pack output_value;
#pragma unroll
      for (int element = 0; element < kWideVectorElements; ++element) {
        output_value.values[element] = Ops::from_float(
            Ops::to_float(sum.values[element]) * inverse_rms *
            Ops::to_float(scale.values[element]));
      }
      input_packs[pack_column] = output_value;
    }
  } else if constexpr (VectorWidth == 2) {
    auto* input_pairs = reinterpret_cast<Pair*>(row_input);
    const auto* residual_pairs = reinterpret_cast<const Pair*>(row_residual);
    const auto* weight_pairs = reinterpret_cast<const Pair*>(weight);
    const uint32_t pair_count = hidden_size / 2U;
    for (uint32_t pair_column = threadIdx.x; pair_column < pair_count;
         pair_column += blockDim.x) {
      const float2 sum = Ops::to_float2(residual_pairs[pair_column]);
      const float2 scale = Ops::to_float2(weight_pairs[pair_column]);
      input_pairs[pair_column] = Ops::from_float2(
          make_float2(sum.x * inverse_rms * scale.x,
                      sum.y * inverse_rms * scale.y));
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      row_input[column] =
          Ops::from_float(Ops::to_float(row_residual[column]) * inverse_rms *
                          Ops::to_float(weight[column]));
    }
  }
}

template <typename Ops>
int launch_low_precision_add_rms_norm(uint16_t* input, uint16_t* residual,
                                      const uint16_t* weight, uint32_t rows,
                                      uint32_t hidden_size, float epsilon,
                                      void* stream) {
  if (input == nullptr || residual == nullptr || weight == nullptr ||
      input == residual || input == weight || residual == weight || rows == 0 ||
      hidden_size == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  auto* typed_input = reinterpret_cast<typename Ops::Scalar*>(input);
  auto* typed_residual = reinterpret_cast<typename Ops::Scalar*>(residual);
  const auto* typed_weight =
      reinterpret_cast<const typename Ops::Scalar*>(weight);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const uintptr_t combined_address =
      reinterpret_cast<uintptr_t>(input) |
      reinterpret_cast<uintptr_t>(residual) |
      reinterpret_cast<uintptr_t>(weight);
  if (hidden_size % kWideVectorElements == 0U &&
      (combined_address & 15U) == 0U) {
    add_rms_norm_low_precision_kernel<Ops, kWideVectorElements>
        <<<rows, kLowPrecisionThreads, 0, cuda_stream>>>(
            typed_input, typed_residual, typed_weight, hidden_size, epsilon);
  } else if ((hidden_size & 1U) == 0U &&
             (combined_address & 3U) == 0U) {
    add_rms_norm_low_precision_kernel<Ops, 2>
        <<<rows, kLowPrecisionThreads, 0, cuda_stream>>>(
            typed_input, typed_residual, typed_weight, hidden_size, epsilon);
  } else {
    add_rms_norm_low_precision_kernel<Ops, 1>
        <<<rows, kLowPrecisionThreads, 0, cuda_stream>>>(
            typed_input, typed_residual, typed_weight, hidden_size, epsilon);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_add_rms_norm_f32(
    float* input, float* residual, const float* weight, uint32_t rows,
    uint32_t hidden_size, float epsilon, void* stream) {
  if (input == nullptr || residual == nullptr || weight == nullptr ||
      input == residual || input == weight || residual == weight || rows == 0 ||
      hidden_size == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  add_rms_norm_f32_kernel<<<rows, kAddRmsNormThreads, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
      input, residual, weight, hidden_size, epsilon);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

extern "C" int loom_cuda_add_rms_norm_f16(
    uint16_t* input, uint16_t* residual, const uint16_t* weight, uint32_t rows,
    uint32_t hidden_size, float epsilon, void* stream) {
  return launch_low_precision_add_rms_norm<HalfOps>(
      input, residual, weight, rows, hidden_size, epsilon, stream);
}

extern "C" int loom_cuda_add_rms_norm_bf16(
    uint16_t* input, uint16_t* residual, const uint16_t* weight, uint32_t rows,
    uint32_t hidden_size, float epsilon, void* stream) {
  return launch_low_precision_add_rms_norm<Bfloat16Ops>(
      input, residual, weight, rows, hidden_size, epsilon, stream);
}
