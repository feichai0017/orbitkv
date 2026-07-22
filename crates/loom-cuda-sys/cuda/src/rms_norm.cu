#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kRmsNormThreads = 256;

__global__ __launch_bounds__(kRmsNormThreads) void rms_norm_f32_kernel(
    const float* input, const float* weight, float* output,
    uint32_t hidden_size, float epsilon) {
  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_size;
  float local_sum = 0.0F;
  for (uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const float value = input[row_offset + column];
    local_sum = fmaf(value, value, local_sum);
  }

  using BlockReduce = cub::BlockReduce<float, kRmsNormThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float inverse_rms;
  const float sum = BlockReduce(reduce_storage).Sum(local_sum);
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();

  for (uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    output[row_offset + column] =
        input[row_offset + column] * inverse_rms * weight[column];
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

template <typename Ops>
__global__ __launch_bounds__(kRmsNormThreads)
    void rms_norm_low_precision_kernel(const typename Ops::Scalar* input,
                                       const typename Ops::Scalar* weight,
                                       typename Ops::Scalar* output,
                                       uint32_t hidden_size, float epsilon) {
  using Scalar = typename Ops::Scalar;
  using Pair = typename Ops::Pair;

  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_size;
  const Scalar* row_input = input + row_offset;
  Scalar* row_output = output + row_offset;
  float local_sum = 0.0F;

  if ((hidden_size & 1U) == 0U) {
    const auto* input_pairs = reinterpret_cast<const Pair*>(row_input);
    const uint32_t pair_count = hidden_size / 2U;
    for (uint32_t pair_column = threadIdx.x; pair_column < pair_count;
         pair_column += blockDim.x) {
      const float2 value = Ops::to_float2(input_pairs[pair_column]);
      local_sum = fmaf(value.x, value.x, local_sum);
      local_sum = fmaf(value.y, value.y, local_sum);
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const float value = Ops::to_float(row_input[column]);
      local_sum = fmaf(value, value, local_sum);
    }
  }

  using BlockReduce = cub::BlockReduce<float, kRmsNormThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float inverse_rms;
  const float sum = BlockReduce(reduce_storage).Sum(local_sum);
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();

  if ((hidden_size & 1U) == 0U) {
    const auto* input_pairs = reinterpret_cast<const Pair*>(row_input);
    const auto* weight_pairs = reinterpret_cast<const Pair*>(weight);
    auto* output_pairs = reinterpret_cast<Pair*>(row_output);
    const uint32_t pair_count = hidden_size / 2U;
    for (uint32_t pair_column = threadIdx.x; pair_column < pair_count;
         pair_column += blockDim.x) {
      const float2 value = Ops::to_float2(input_pairs[pair_column]);
      const float2 scale = Ops::to_float2(weight_pairs[pair_column]);
      output_pairs[pair_column] = Ops::from_float2(
          make_float2(value.x * inverse_rms * scale.x,
                      value.y * inverse_rms * scale.y));
    }
  } else {
    for (uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      row_output[column] = Ops::from_float(Ops::to_float(row_input[column]) *
                                           inverse_rms *
                                           Ops::to_float(weight[column]));
    }
  }
}

template <typename Ops>
int launch_low_precision_rms_norm(const uint16_t* input,
                                  const uint16_t* weight, uint16_t* output,
                                  uint32_t rows, uint32_t hidden_size,
                                  float epsilon, void* stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows == 0 ||
      hidden_size == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  rms_norm_low_precision_kernel<Ops>
      <<<rows, kRmsNormThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
          reinterpret_cast<const typename Ops::Scalar*>(input),
          reinterpret_cast<const typename Ops::Scalar*>(weight),
          reinterpret_cast<typename Ops::Scalar*>(output), hidden_size,
          epsilon);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" const char* loom_cuda_status_string(int status) {
  switch (status) {
    case LOOM_CUDA_SUCCESS:
      return "success";
    case LOOM_CUDA_INVALID_ARGUMENT:
      return "invalid argument";
    case LOOM_CUDA_UNSUPPORTED:
      return "unsupported";
    case LOOM_CUDA_LAUNCH_ERROR:
      return "CUDA launch error";
    case LOOM_CUDA_UNAVAILABLE:
      return "CUDA unavailable";
    default:
      return "unknown Loom CUDA status";
  }
}

extern "C" int loom_cuda_rms_norm_f32(
    const float* input, const float* weight, float* output, uint32_t rows,
    uint32_t hidden_size, float epsilon, void* stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows == 0 ||
      hidden_size == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  rms_norm_f32_kernel<<<rows, kRmsNormThreads, 0,
                        reinterpret_cast<cudaStream_t>(stream)>>>(
      input, weight, output, hidden_size, epsilon);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

extern "C" int loom_cuda_rms_norm_f16(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    uint32_t rows, uint32_t hidden_size, float epsilon, void* stream) {
  return launch_low_precision_rms_norm<HalfOps>(
      input, weight, output, rows, hidden_size, epsilon, stream);
}

extern "C" int loom_cuda_rms_norm_bf16(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    uint32_t rows, uint32_t hidden_size, float epsilon, void* stream) {
  return launch_low_precision_rms_norm<Bfloat16Ops>(
      input, weight, output, rows, hidden_size, epsilon, stream);
}
