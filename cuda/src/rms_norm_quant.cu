#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kRmsNormQuantThreads = 1024;
constexpr float kFp8E4M3Max = 448.0F;
constexpr float kDynamicFp8MinScale = 1.0F / (kFp8E4M3Max * 512.0F);

struct Maximum {
  __device__ float operator()(float left, float right) const {
    return fmaxf(left, right);
  }
};

struct Addition {
  __device__ float operator()(float left, float right) const {
    return left + right;
  }
};

struct FloatOps {
  using Scalar = float;

  __device__ static float to_float(Scalar value) { return value; }
  __device__ static Scalar from_float(float value) { return value; }
};

struct HalfOps {
  using Scalar = __half;

  __device__ static float to_float(Scalar value) {
    return __half2float(value);
  }

  __device__ static Scalar from_float(float value) {
    return __float2half_rn(value);
  }
};

struct Bfloat16Ops {
  using Scalar = __nv_bfloat16;

  __device__ static float to_float(Scalar value) {
    return __bfloat162float(value);
  }

  __device__ static Scalar from_float(float value) {
    return __float2bfloat16_rn(value);
  }
};

template <typename Scalar, int Width>
struct alignas(sizeof(Scalar) * Width) AlignedPack {
  Scalar values[Width];
};

struct alignas(4) Fp8Pack4 {
  __nv_fp8x4_storage_t bits;
};

template <typename Ops, bool Vectorized>
__global__ __launch_bounds__(kRmsNormQuantThreads)
    void rms_norm_dynamic_fp8_kernel(const typename Ops::Scalar* input,
                                     const typename Ops::Scalar* weight,
                                     uint8_t* output, float* scales,
                                     uint32_t hidden_size, float epsilon) {
  using Scalar = typename Ops::Scalar;
  using BlockReduce = cub::BlockReduce<float, kRmsNormQuantThreads>;

  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_size;
  const Scalar* row_input = input + row_offset;
  uint8_t* row_output = output + row_offset;
  float local_square_sum = 0.0F;
  if constexpr (Vectorized) {
    using Pack = AlignedPack<Scalar, 4>;
    const auto* input_packs = reinterpret_cast<const Pack*>(row_input);
    const uint32_t pack_count = hidden_size / 4U;
    for (uint32_t pack_column = threadIdx.x; pack_column < pack_count;
         pack_column += blockDim.x) {
      const Pack values = input_packs[pack_column];
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        const float value = Ops::to_float(values.values[element]);
        local_square_sum += value * value;
      }
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const float value = Ops::to_float(row_input[column]);
      local_square_sum += value * value;
    }
  }

  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float inverse_rms;
  __shared__ float token_scale;
  const float square_sum = BlockReduce(reduce_storage)
                               .Reduce(local_square_sum, Addition{},
                                       static_cast<int>(blockDim.x));
  if (threadIdx.x == 0) {
    inverse_rms =
        rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();

  float local_absolute_maximum = 0.0F;
  if constexpr (Vectorized) {
    using Pack = AlignedPack<Scalar, 4>;
    const auto* input_packs = reinterpret_cast<const Pack*>(row_input);
    const auto* weight_packs = reinterpret_cast<const Pack*>(weight);
    const uint32_t pack_count = hidden_size / 4U;
    for (uint32_t pack_column = threadIdx.x; pack_column < pack_count;
         pack_column += blockDim.x) {
      const Pack values = input_packs[pack_column];
      const Pack weights = weight_packs[pack_column];
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        // Match the vLLM fused quantization boundary: x * inverse_rms is
        // rounded to the input storage dtype before applying the weight.
        const Scalar normalized = Ops::from_float(
            Ops::to_float(values.values[element]) * inverse_rms);
        const Scalar weighted_storage = Ops::from_float(
            Ops::to_float(normalized) *
            Ops::to_float(weights.values[element]));
        const float weighted = Ops::to_float(weighted_storage);
        local_absolute_maximum =
            fmaxf(local_absolute_maximum, fabsf(weighted));
      }
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const Scalar normalized =
          Ops::from_float(Ops::to_float(row_input[column]) * inverse_rms);
      const Scalar weighted_storage = Ops::from_float(
          Ops::to_float(normalized) * Ops::to_float(weight[column]));
      const float weighted = Ops::to_float(weighted_storage);
      local_absolute_maximum =
          fmaxf(local_absolute_maximum, fabsf(weighted));
    }
  }

  __syncthreads();
  const float absolute_maximum =
      BlockReduce(reduce_storage)
          .Reduce(local_absolute_maximum, Maximum{},
                  static_cast<int>(blockDim.x));
  if (threadIdx.x == 0) {
    token_scale =
        fmaxf(absolute_maximum / kFp8E4M3Max, kDynamicFp8MinScale);
    scales[blockIdx.x] = token_scale;
  }
  __syncthreads();

  if constexpr (Vectorized) {
    using Pack = AlignedPack<Scalar, 4>;
    const auto* input_packs = reinterpret_cast<const Pack*>(row_input);
    const auto* weight_packs = reinterpret_cast<const Pack*>(weight);
    auto* output_packs = reinterpret_cast<Fp8Pack4*>(row_output);
    const uint32_t pack_count = hidden_size / 4U;
    for (uint32_t pack_column = threadIdx.x; pack_column < pack_count;
         pack_column += blockDim.x) {
      const Pack values = input_packs[pack_column];
      const Pack weights = weight_packs[pack_column];
      float quantized_values[4];
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        const Scalar normalized = Ops::from_float(
            Ops::to_float(values.values[element]) * inverse_rms);
        const Scalar weighted = Ops::from_float(
            Ops::to_float(normalized) *
            Ops::to_float(weights.values[element]));
        quantized_values[element] = Ops::to_float(weighted) / token_scale;
      }
      const __nv_fp8x4_e4m3 quantized(make_float4(
          quantized_values[0], quantized_values[1], quantized_values[2],
          quantized_values[3]));
      output_packs[pack_column].bits = quantized.__x;
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const Scalar normalized =
          Ops::from_float(Ops::to_float(row_input[column]) * inverse_rms);
      const Scalar weighted_storage = Ops::from_float(
          Ops::to_float(normalized) * Ops::to_float(weight[column]));
      const float weighted = Ops::to_float(weighted_storage);
      row_output[column] = __nv_cvt_float_to_fp8(
          weighted / token_scale, __NV_SATFINITE, __NV_E4M3);
    }
  }
}

template <typename Ops, typename Input>
int launch_rms_norm_dynamic_fp8(const Input* input, const Input* weight,
                                uint8_t* output, float* scales, uint32_t rows,
                                uint32_t hidden_size, float epsilon,
                                void* stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      scales == nullptr || rows == 0 || hidden_size == 0 ||
      !std::isfinite(epsilon) || epsilon <= 0.0F ||
      reinterpret_cast<const void*>(input) ==
          reinterpret_cast<const void*>(output) ||
      reinterpret_cast<const void*>(weight) ==
          reinterpret_cast<const void*>(output)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  using Scalar = typename Ops::Scalar;
  const uintptr_t combined_input_address =
      reinterpret_cast<uintptr_t>(input) |
      reinterpret_cast<uintptr_t>(weight);
  const bool can_vectorize = hidden_size % 4U == 0U &&
                             combined_input_address % (sizeof(Scalar) * 4U) ==
                                 0U &&
                             reinterpret_cast<uintptr_t>(output) % 4U == 0U;
  const uint32_t threads =
      hidden_size < static_cast<uint32_t>(kRmsNormQuantThreads)
          ? hidden_size
          : static_cast<uint32_t>(kRmsNormQuantThreads);
  if (can_vectorize) {
    rms_norm_dynamic_fp8_kernel<Ops, true>
        <<<rows, threads, 0,
           reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const Scalar*>(input),
            reinterpret_cast<const Scalar*>(weight), output, scales,
            hidden_size, epsilon);
  } else {
    rms_norm_dynamic_fp8_kernel<Ops, false>
        <<<rows, threads, 0,
           reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const Scalar*>(input),
            reinterpret_cast<const Scalar*>(weight), output, scales,
            hidden_size, epsilon);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_rms_norm_dynamic_fp8_f32(
    const float* input, const float* weight, uint8_t* output, float* scales,
    uint32_t rows, uint32_t hidden_size, float epsilon, void* stream) {
  return launch_rms_norm_dynamic_fp8<FloatOps>(
      input, weight, output, scales, rows, hidden_size, epsilon, stream);
}

extern "C" int loom_cuda_rms_norm_dynamic_fp8_f16(
    const uint16_t* input, const uint16_t* weight, uint8_t* output,
    float* scales, uint32_t rows, uint32_t hidden_size, float epsilon,
    void* stream) {
  return launch_rms_norm_dynamic_fp8<HalfOps>(
      input, weight, output, scales, rows, hidden_size, epsilon, stream);
}

extern "C" int loom_cuda_rms_norm_dynamic_fp8_bf16(
    const uint16_t* input, const uint16_t* weight, uint8_t* output,
    float* scales, uint32_t rows, uint32_t hidden_size, float epsilon,
    void* stream) {
  return launch_rms_norm_dynamic_fp8<Bfloat16Ops>(
      input, weight, output, scales, rows, hidden_size, epsilon, stream);
}
