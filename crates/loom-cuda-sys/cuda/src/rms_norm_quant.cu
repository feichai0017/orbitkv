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

struct alignas(4) BytePack4 {
  uint8_t values[4];
};

static inline __device__ int8_t float_to_int8_rn(float value) {
  uint32_t result;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(value));
  return reinterpret_cast<const int8_t&>(result);
}

struct Fp8QuantOps {
  __device__ static float scale(float absolute_maximum) {
    return fmaxf(absolute_maximum / kFp8E4M3Max, kDynamicFp8MinScale);
  }

  __device__ static uint8_t quantize(float value, float scale,
                                     float absolute_maximum) {
    (void)absolute_maximum;
    return __nv_cvt_float_to_fp8(value / scale, __NV_SATFINITE, __NV_E4M3);
  }

  __device__ static void store_pack(uint8_t* output,
                                    const float (&values)[4], float scale,
                                    float absolute_maximum) {
    (void)absolute_maximum;
    const __nv_fp8x4_e4m3 quantized(
        make_float4(values[0] / scale, values[1] / scale, values[2] / scale,
                    values[3] / scale));
    reinterpret_cast<Fp8Pack4*>(output)->bits = quantized.__x;
  }
};

struct Int8QuantOps {
  __device__ static float scale(float absolute_maximum) {
    return absolute_maximum / 127.0F;
  }

  __device__ static uint8_t quantize(float value, float scale,
                                     float absolute_maximum) {
    (void)scale;
    const float scaled = absolute_maximum == 0.0F
                             ? 0.0F
                             : value * (127.0F / absolute_maximum);
    return static_cast<uint8_t>(float_to_int8_rn(scaled));
  }

  __device__ static void store_pack(uint8_t* output,
                                    const float (&values)[4], float scale,
                                    float absolute_maximum) {
    BytePack4 quantized{};
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      quantized.values[element] =
          quantize(values[element], scale, absolute_maximum);
    }
    *reinterpret_cast<BytePack4*>(output) = quantized;
  }
};

template <typename Ops, typename QuantOps, bool Vectorized, bool HasResidual>
__global__ __launch_bounds__(kRmsNormQuantThreads)
    void rms_norm_dynamic_quant_kernel(const typename Ops::Scalar* input,
                                       const typename Ops::Scalar* weight,
                                       typename Ops::Scalar* residual,
                                       uint8_t* output, float* scales,
                                       uint32_t hidden_size, float epsilon) {
  // This is the vLLM native-IR boundary consumed by the compiler fusion:
  // residual addition stays in F32 for RMS, and normalized values are rounded
  // to the weight dtype before applying the weight.
  using Scalar = typename Ops::Scalar;
  using BlockReduce = cub::BlockReduce<float, kRmsNormQuantThreads>;

  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_size;
  const Scalar* row_input = input + row_offset;
  uint8_t* row_output = output + row_offset;
  float local_square_sum = 0.0F;
  if constexpr (Vectorized) {
    using Pack = AlignedPack<Scalar, 4>;
    const auto* input_packs = reinterpret_cast<const Pack*>(row_input);
    const Pack* residual_packs = nullptr;
    if constexpr (HasResidual) {
      residual_packs =
          reinterpret_cast<const Pack*>(residual + row_offset);
    }
    const uint32_t pack_count = hidden_size / 4U;
    for (uint32_t pack_column = threadIdx.x; pack_column < pack_count;
         pack_column += blockDim.x) {
      const Pack values = input_packs[pack_column];
      if constexpr (HasResidual) {
        const Pack residual_values = residual_packs[pack_column];
#pragma unroll
        for (int element = 0; element < 4; ++element) {
          const float sum = Ops::to_float(values.values[element]) +
                            Ops::to_float(residual_values.values[element]);
          local_square_sum += sum * sum;
        }
      } else {
#pragma unroll
        for (int element = 0; element < 4; ++element) {
          const float value = Ops::to_float(values.values[element]);
          local_square_sum += value * value;
        }
      }
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      float value = Ops::to_float(row_input[column]);
      if constexpr (HasResidual) {
        value += Ops::to_float(residual[row_offset + column]);
      }
      local_square_sum += value * value;
    }
  }

  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float inverse_rms;
  __shared__ float quantization_absolute_maximum;
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
    const Pack* residual_packs = nullptr;
    if constexpr (HasResidual) {
      residual_packs =
          reinterpret_cast<const Pack*>(residual + row_offset);
    }
    const uint32_t pack_count = hidden_size / 4U;
    for (uint32_t pack_column = threadIdx.x; pack_column < pack_count;
         pack_column += blockDim.x) {
      const Pack values = input_packs[pack_column];
      const Pack weights = weight_packs[pack_column];
      Pack residual_values{};
      if constexpr (HasResidual) {
        residual_values = residual_packs[pack_column];
      }
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        float value = Ops::to_float(values.values[element]);
        if constexpr (HasResidual) {
          value += Ops::to_float(residual_values.values[element]);
        }
        const float normalized =
            Ops::to_float(Ops::from_float(value * inverse_rms));
        const Scalar weighted_storage = Ops::from_float(
            normalized * Ops::to_float(weights.values[element]));
        const float weighted = Ops::to_float(weighted_storage);
        local_absolute_maximum =
            fmaxf(local_absolute_maximum, fabsf(weighted));
      }
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      float value = Ops::to_float(row_input[column]);
      if constexpr (HasResidual) {
        value += Ops::to_float(residual[row_offset + column]);
      }
      const float normalized =
          Ops::to_float(Ops::from_float(value * inverse_rms));
      const Scalar weighted_storage = Ops::from_float(
          normalized * Ops::to_float(weight[column]));
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
    quantization_absolute_maximum = absolute_maximum;
    token_scale = QuantOps::scale(absolute_maximum);
    scales[blockIdx.x] = token_scale;
  }
  __syncthreads();

  if constexpr (Vectorized) {
    using Pack = AlignedPack<Scalar, 4>;
    const auto* input_packs = reinterpret_cast<const Pack*>(row_input);
    const auto* weight_packs = reinterpret_cast<const Pack*>(weight);
    Pack* residual_packs = nullptr;
    if constexpr (HasResidual) {
      residual_packs = reinterpret_cast<Pack*>(residual + row_offset);
    }
    const uint32_t pack_count = hidden_size / 4U;
    for (uint32_t pack_column = threadIdx.x; pack_column < pack_count;
         pack_column += blockDim.x) {
      const Pack values = input_packs[pack_column];
      const Pack weights = weight_packs[pack_column];
      Pack residual_values{};
      if constexpr (HasResidual) {
        residual_values = residual_packs[pack_column];
      }
      float weighted_values[4];
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        float value = Ops::to_float(values.values[element]);
        if constexpr (HasResidual) {
          value += Ops::to_float(residual_values.values[element]);
          residual_values.values[element] = Ops::from_float(value);
        }
        const float normalized =
            Ops::to_float(Ops::from_float(value * inverse_rms));
        const Scalar weighted = Ops::from_float(
            normalized * Ops::to_float(weights.values[element]));
        weighted_values[element] = Ops::to_float(weighted);
      }
      if constexpr (HasResidual) {
        residual_packs[pack_column] = residual_values;
      }
      QuantOps::store_pack(row_output + pack_column * 4U, weighted_values,
                           token_scale, quantization_absolute_maximum);
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      float value = Ops::to_float(row_input[column]);
      if constexpr (HasResidual) {
        value += Ops::to_float(residual[row_offset + column]);
        residual[row_offset + column] = Ops::from_float(value);
      }
      const float normalized =
          Ops::to_float(Ops::from_float(value * inverse_rms));
      const Scalar weighted_storage = Ops::from_float(
          normalized * Ops::to_float(weight[column]));
      const float weighted = Ops::to_float(weighted_storage);
      row_output[column] = QuantOps::quantize(
          weighted, token_scale, quantization_absolute_maximum);
    }
  }
}

template <typename Ops, typename QuantOps, typename Input>
int launch_rms_norm_dynamic_quant(const Input* input, const Input* weight,
                                  Input* residual, uint8_t* output,
                                  float* scales, uint32_t rows,
                                  uint32_t hidden_size, float epsilon,
                                  void* stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      scales == nullptr || rows == 0 || hidden_size == 0 ||
      !std::isfinite(epsilon) || epsilon <= 0.0F ||
      reinterpret_cast<const void*>(input) ==
          reinterpret_cast<const void*>(output) ||
      reinterpret_cast<const void*>(weight) ==
          reinterpret_cast<const void*>(output) ||
      (residual != nullptr &&
       (reinterpret_cast<const void*>(residual) ==
            reinterpret_cast<const void*>(input) ||
        reinterpret_cast<const void*>(residual) ==
            reinterpret_cast<const void*>(weight) ||
        reinterpret_cast<const void*>(residual) ==
            reinterpret_cast<const void*>(output) ||
        reinterpret_cast<const void*>(residual) ==
            reinterpret_cast<const void*>(scales)))) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  using Scalar = typename Ops::Scalar;
  const uintptr_t combined_input_address =
      reinterpret_cast<uintptr_t>(input) |
      reinterpret_cast<uintptr_t>(weight) |
      reinterpret_cast<uintptr_t>(residual);
  const bool can_vectorize = hidden_size % 4U == 0U &&
                             combined_input_address % (sizeof(Scalar) * 4U) ==
                                 0U &&
                             reinterpret_cast<uintptr_t>(output) % 4U == 0U;
  // Vectorized kernels assign one four-element pack to each work item. Size
  // the block from that pack count instead of the scalar width; otherwise a
  // hidden size such as 896 launches 896 threads for only 224 packs and makes
  // 672 idle threads participate in both reductions.
  const uint32_t work_items =
      can_vectorize ? hidden_size / 4U : hidden_size;
  const uint32_t threads =
      work_items < static_cast<uint32_t>(kRmsNormQuantThreads)
          ? work_items
          : static_cast<uint32_t>(kRmsNormQuantThreads);
  if (can_vectorize) {
    if (residual != nullptr) {
      rms_norm_dynamic_quant_kernel<Ops, QuantOps, true, true>
          <<<rows, threads, 0,
             reinterpret_cast<cudaStream_t>(stream)>>>(
              reinterpret_cast<const Scalar*>(input),
              reinterpret_cast<const Scalar*>(weight),
              reinterpret_cast<Scalar*>(residual), output, scales,
              hidden_size, epsilon);
    } else {
      rms_norm_dynamic_quant_kernel<Ops, QuantOps, true, false>
          <<<rows, threads, 0,
             reinterpret_cast<cudaStream_t>(stream)>>>(
              reinterpret_cast<const Scalar*>(input),
              reinterpret_cast<const Scalar*>(weight), nullptr, output,
              scales, hidden_size, epsilon);
    }
  } else {
    if (residual != nullptr) {
      rms_norm_dynamic_quant_kernel<Ops, QuantOps, false, true>
          <<<rows, threads, 0,
             reinterpret_cast<cudaStream_t>(stream)>>>(
              reinterpret_cast<const Scalar*>(input),
              reinterpret_cast<const Scalar*>(weight),
              reinterpret_cast<Scalar*>(residual), output, scales,
              hidden_size, epsilon);
    } else {
      rms_norm_dynamic_quant_kernel<Ops, QuantOps, false, false>
          <<<rows, threads, 0,
             reinterpret_cast<cudaStream_t>(stream)>>>(
              reinterpret_cast<const Scalar*>(input),
              reinterpret_cast<const Scalar*>(weight), nullptr, output,
              scales, hidden_size, epsilon);
    }
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_rms_norm_dynamic_fp8_f32(
    const float* input, const float* weight, float* residual, uint8_t* output,
    float* scales, uint32_t rows, uint32_t hidden_size, float epsilon,
    void* stream) {
  return launch_rms_norm_dynamic_quant<FloatOps, Fp8QuantOps>(
      input, weight, residual, output, scales, rows, hidden_size, epsilon,
      stream);
}

extern "C" int loom_cuda_rms_norm_dynamic_fp8_f16(
    const uint16_t* input, const uint16_t* weight, uint16_t* residual,
    uint8_t* output, float* scales, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream) {
  return launch_rms_norm_dynamic_quant<HalfOps, Fp8QuantOps>(
      input, weight, residual, output, scales, rows, hidden_size, epsilon,
      stream);
}

extern "C" int loom_cuda_rms_norm_dynamic_fp8_bf16(
    const uint16_t* input, const uint16_t* weight, uint16_t* residual,
    uint8_t* output, float* scales, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream) {
  return launch_rms_norm_dynamic_quant<Bfloat16Ops, Fp8QuantOps>(
      input, weight, residual, output, scales, rows, hidden_size, epsilon,
      stream);
}

extern "C" int loom_cuda_rms_norm_dynamic_int8_f32(
    const float* input, const float* weight, float* residual, int8_t* output,
    float* scales, uint32_t rows, uint32_t hidden_size, float epsilon,
    void* stream) {
  return launch_rms_norm_dynamic_quant<FloatOps, Int8QuantOps>(
      input, weight, residual, reinterpret_cast<uint8_t*>(output), scales, rows,
      hidden_size, epsilon, stream);
}

extern "C" int loom_cuda_rms_norm_dynamic_int8_f16(
    const uint16_t* input, const uint16_t* weight, uint16_t* residual,
    int8_t* output, float* scales, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream) {
  return launch_rms_norm_dynamic_quant<HalfOps, Int8QuantOps>(
      input, weight, residual, reinterpret_cast<uint8_t*>(output), scales, rows,
      hidden_size, epsilon, stream);
}

extern "C" int loom_cuda_rms_norm_dynamic_int8_bf16(
    const uint16_t* input, const uint16_t* weight, uint16_t* residual,
    int8_t* output, float* scales, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream) {
  return launch_rms_norm_dynamic_quant<Bfloat16Ops, Int8QuantOps>(
      input, weight, residual, reinterpret_cast<uint8_t*>(output), scales, rows,
      hidden_size, epsilon, stream);
}
