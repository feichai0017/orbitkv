#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr float kFp8E4M3Max = 448.0F;
constexpr float kDynamicFp8MinScale = 1.0F / (kFp8E4M3Max * 512.0F);

struct Maximum {
  __device__ float operator()(float left, float right) const {
    return fmaxf(left, right);
  }
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

__device__ int8_t float_to_int8_rn(float value) {
  int32_t result;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(value));
  return static_cast<int8_t>(result);
}

template <typename Ops, int GroupSize>
__global__ __launch_bounds__(GroupSize) void silu_and_mul_dynamic_fp8_kernel(
    const typename Ops::Scalar* input, uint8_t* output, float* scales,
    const float* scale_ub, uint32_t rows, uint32_t width,
    uint32_t group_count, bool scales_transposed) {
  using Scalar = typename Ops::Scalar;
  using BlockReduce = cub::BlockReduce<float, GroupSize>;

  const uint32_t block_index = blockIdx.x;
  const uint32_t row = block_index / group_count;
  const uint32_t group = block_index - row * group_count;
  const uint32_t column = group * GroupSize + threadIdx.x;
  const size_t input_row_offset =
      static_cast<size_t>(row) * static_cast<size_t>(width) * 2U;
  const size_t output_index =
      static_cast<size_t>(row) * static_cast<size_t>(width) + column;

  const Scalar gate = input[input_row_offset + column];
  const Scalar up = input[input_row_offset + width + column];
  const float gate_f32 = Ops::to_float(gate);
  // The fused vLLM contract keeps activation and multiplication in F32 and
  // quantizes directly, without a low-precision intermediate tensor.
  const float sigmoid_gate = 1.0F / (1.0F + expf(-gate_f32));
  const float value = gate_f32 * sigmoid_gate * Ops::to_float(up);

  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float block_scale;
  const float absolute_maximum =
      BlockReduce(reduce_storage).Reduce(fabsf(value), Maximum{});
  if (threadIdx.x == 0) {
    block_scale = absolute_maximum / kFp8E4M3Max;
    if (scale_ub != nullptr) {
      block_scale = fminf(block_scale, *scale_ub);
    }
    block_scale = fmaxf(block_scale, kDynamicFp8MinScale);
    const size_t scale_index = scales_transposed
                                   ? static_cast<size_t>(group) * rows + row
                                   : block_index;
    scales[scale_index] = block_scale;
  }
  __syncthreads();

  output[output_index] = __nv_cvt_float_to_fp8(
      value / block_scale, __NV_SATFINITE, __NV_E4M3);
}

template <typename Ops, int Threads, bool StoreValues>
__global__ __launch_bounds__(Threads)
void silu_and_mul_dynamic_int8_kernel(const typename Ops::Scalar* input,
                                      int8_t* output, float* scales,
                                      uint32_t width) {
  using Scalar = typename Ops::Scalar;
  using BlockReduce = cub::BlockReduce<float, Threads>;

  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  Scalar* values = reinterpret_cast<Scalar*>(dynamic_storage);
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float absolute_maximum;

  const size_t input_row_offset =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(width) * 2U;
  const size_t output_row_offset =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(width);
  const Scalar* gate = input + input_row_offset;
  const Scalar* up = gate + width;

  float local_absolute_maximum = 0.0F;
  for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float gate_value = Ops::to_float(gate[column]);
    const Scalar product = Ops::from_float(
        gate_value / (1.0F + expf(-gate_value)) * Ops::to_float(up[column]));
    const float value = Ops::to_float(product);
    if constexpr (StoreValues) {
      values[column] = product;
    }
    local_absolute_maximum = fmaxf(local_absolute_maximum, fabsf(value));
  }

  const float row_absolute_maximum =
      BlockReduce(reduce_storage).Reduce(local_absolute_maximum, Maximum{});
  if (threadIdx.x == 0) {
    absolute_maximum = row_absolute_maximum;
    scales[blockIdx.x] = row_absolute_maximum / 127.0F;
  }
  __syncthreads();

  const float multiplier =
      absolute_maximum == 0.0F ? 0.0F : 127.0F / absolute_maximum;
  for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
    float value;
    if constexpr (StoreValues) {
      value = Ops::to_float(values[column]);
    } else {
      const float gate_value = Ops::to_float(gate[column]);
      value = Ops::to_float(Ops::from_float(
          gate_value / (1.0F + expf(-gate_value)) *
          Ops::to_float(up[column])));
    }
    output[output_row_offset + column] =
        float_to_int8_rn(value * multiplier);
  }
}

bool ranges_overlap(const void* left, size_t left_bytes, const void* right,
                    size_t right_bytes) {
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
  if (left_begin <= right_begin) {
    return right_begin - left_begin < left_bytes;
  }
  return left_begin - right_begin < right_bytes;
}

template <typename Ops, typename Input>
int launch_silu_and_mul_dynamic_fp8(const Input* input, uint8_t* output,
                                    float* scales, uint32_t rows,
                                    uint32_t width, uint32_t group_size,
                                    const float* scale_ub,
                                    uint32_t scales_transposed, void* stream) {
  if (input == nullptr || output == nullptr || scales == nullptr || rows == 0 ||
      width == 0 || (group_size != 64U && group_size != 128U) ||
      width % group_size != 0U) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  using Scalar = typename Ops::Scalar;
  const uint32_t group_count = width / group_size;
  if (rows > std::numeric_limits<uint32_t>::max() / group_count) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const size_t output_elements =
      static_cast<size_t>(rows) * static_cast<size_t>(width);
  const size_t maximum_size = std::numeric_limits<size_t>::max();
  if (output_elements > maximum_size / (2U * sizeof(Scalar))) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const size_t input_bytes = output_elements * 2U * sizeof(Scalar);
  const size_t output_bytes = output_elements;
  const size_t scale_bytes =
      static_cast<size_t>(rows) * group_count * sizeof(float);
  if (ranges_overlap(input, input_bytes, output, output_bytes) ||
      ranges_overlap(input, input_bytes, scales, scale_bytes) ||
      ranges_overlap(output, output_bytes, scales, scale_bytes)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  const uint32_t block_count = rows * group_count;
  if (group_size == 64U) {
    silu_and_mul_dynamic_fp8_kernel<Ops, 64>
        <<<block_count, 64, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const Scalar*>(input), output, scales, scale_ub,
            rows, width, group_count, scales_transposed != 0U);
  } else {
    silu_and_mul_dynamic_fp8_kernel<Ops, 128>
        <<<block_count, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const Scalar*>(input), output, scales, scale_ub,
            rows, width, group_count, scales_transposed != 0U);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

template <typename Ops, int Threads, bool StoreValues>
void launch_silu_and_mul_dynamic_int8_kernel(
    const typename Ops::Scalar* input, int8_t* output, float* scales,
    uint32_t rows, uint32_t width, cudaStream_t stream) {
  const size_t shared_bytes =
      StoreValues ? static_cast<size_t>(width) * sizeof(typename Ops::Scalar)
                  : 0U;
  silu_and_mul_dynamic_int8_kernel<Ops, Threads, StoreValues>
      <<<rows, Threads, shared_bytes, stream>>>(input, output, scales, width);
}

template <typename Ops, typename Input>
int launch_silu_and_mul_dynamic_int8(const Input* input, int8_t* output,
                                    float* scales, uint32_t rows,
                                    uint32_t width, void* stream) {
  if (input == nullptr || output == nullptr || scales == nullptr || rows == 0 ||
      width == 0) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  using Scalar = typename Ops::Scalar;
  const size_t output_elements =
      static_cast<size_t>(rows) * static_cast<size_t>(width);
  const size_t maximum_size = std::numeric_limits<size_t>::max();
  if (output_elements > maximum_size / (2U * sizeof(Scalar))) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const size_t input_bytes = output_elements * 2U * sizeof(Scalar);
  const size_t output_bytes = output_elements * sizeof(int8_t);
  const size_t scale_bytes = static_cast<size_t>(rows) * sizeof(float);
  if (ranges_overlap(input, input_bytes, output, output_bytes) ||
      ranges_overlap(input, input_bytes, scales, scale_bytes) ||
      ranges_overlap(output, output_bytes, scales, scale_bytes)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  // Leave room for CUB's statically allocated reduction storage on devices
  // whose default per-block shared-memory limit is 48 KiB.
  constexpr uint32_t kMaximumDynamicSharedBytes = 40U * 1024U;
  constexpr uint32_t kMaximumStoredWidth =
      kMaximumDynamicSharedBytes / sizeof(Scalar);
  const bool store_values = width <= kMaximumStoredWidth;
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (rows <= 128U) {
    if (store_values) {
      launch_silu_and_mul_dynamic_int8_kernel<Ops, 1024, true>(
          reinterpret_cast<const Scalar*>(input), output, scales, rows, width,
          cuda_stream);
    } else {
      launch_silu_and_mul_dynamic_int8_kernel<Ops, 1024, false>(
          reinterpret_cast<const Scalar*>(input), output, scales, rows, width,
          cuda_stream);
    }
  } else if (rows <= 256U) {
    if (store_values) {
      launch_silu_and_mul_dynamic_int8_kernel<Ops, 512, true>(
          reinterpret_cast<const Scalar*>(input), output, scales, rows, width,
          cuda_stream);
    } else {
      launch_silu_and_mul_dynamic_int8_kernel<Ops, 512, false>(
          reinterpret_cast<const Scalar*>(input), output, scales, rows, width,
          cuda_stream);
    }
  } else if (store_values) {
    launch_silu_and_mul_dynamic_int8_kernel<Ops, 256, true>(
        reinterpret_cast<const Scalar*>(input), output, scales, rows, width,
        cuda_stream);
  } else {
    launch_silu_and_mul_dynamic_int8_kernel<Ops, 256, false>(
        reinterpret_cast<const Scalar*>(input), output, scales, rows, width,
        cuda_stream);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_silu_and_mul_dynamic_fp8_f16(
    const uint16_t* input, uint8_t* output, float* scales, uint32_t rows,
    uint32_t width, uint32_t group_size, const float* scale_ub,
    uint32_t scales_transposed, void* stream) {
  return launch_silu_and_mul_dynamic_fp8<HalfOps>(
      input, output, scales, rows, width, group_size, scale_ub,
      scales_transposed, stream);
}

extern "C" int loom_cuda_silu_and_mul_dynamic_fp8_bf16(
    const uint16_t* input, uint8_t* output, float* scales, uint32_t rows,
    uint32_t width, uint32_t group_size, const float* scale_ub,
    uint32_t scales_transposed, void* stream) {
  return launch_silu_and_mul_dynamic_fp8<Bfloat16Ops>(
      input, output, scales, rows, width, group_size, scale_ub,
      scales_transposed, stream);
}

extern "C" int loom_cuda_silu_and_mul_dynamic_int8_f16(
    const uint16_t* input, int8_t* output, float* scales, uint32_t rows,
    uint32_t width, void* stream) {
  return launch_silu_and_mul_dynamic_int8<HalfOps>(input, output, scales, rows,
                                                  width, stream);
}

extern "C" int loom_cuda_silu_and_mul_dynamic_int8_bf16(
    const uint16_t* input, int8_t* output, float* scales, uint32_t rows,
    uint32_t width, void* stream) {
  return launch_silu_and_mul_dynamic_int8<Bfloat16Ops>(
      input, output, scales, rows, width, stream);
}
