#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

constexpr uint32_t kMaximumThreads = 1024;
constexpr size_t kVectorBytes = 16;

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

template <typename Ops>
__device__ typename Ops::Scalar silu_and_mul_value(
    typename Ops::Scalar gate, typename Ops::Scalar up) {
  const float gate_f32 = Ops::to_float(gate);
  const typename Ops::Scalar activated =
      Ops::from_float(gate_f32 / (1.0F + expf(-gate_f32)));
  return Ops::from_float(Ops::to_float(activated) * Ops::to_float(up));
}

template <typename Ops, bool Vectorized>
__global__ void silu_and_mul_kernel(const typename Ops::Scalar* input,
                                    typename Ops::Scalar* output,
                                    uint32_t width) {
  using Scalar = typename Ops::Scalar;
  constexpr int kPackWidth = static_cast<int>(kVectorBytes / sizeof(Scalar));
  const size_t input_row_offset =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(width) * 2U;
  const size_t output_row_offset =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(width);
  const Scalar* gate = input + input_row_offset;
  const Scalar* up = gate + width;
  Scalar* output_row = output + output_row_offset;

  if constexpr (Vectorized) {
    using Pack = AlignedPack<Scalar, kPackWidth>;
    const auto* gate_packs = reinterpret_cast<const Pack*>(gate);
    const auto* up_packs = reinterpret_cast<const Pack*>(up);
    auto* output_packs = reinterpret_cast<Pack*>(output_row);
    const uint32_t pack_count = width / static_cast<uint32_t>(kPackWidth);
    for (uint32_t pack_index = threadIdx.x; pack_index < pack_count;
         pack_index += blockDim.x) {
      const Pack gate_values = gate_packs[pack_index];
      const Pack up_values = up_packs[pack_index];
      Pack result;
#pragma unroll
      for (int element = 0; element < kPackWidth; ++element) {
        result.values[element] = silu_and_mul_value<Ops>(
            gate_values.values[element], up_values.values[element]);
      }
      output_packs[pack_index] = result;
    }
  } else {
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      output_row[column] =
          silu_and_mul_value<Ops>(gate[column], up[column]);
    }
  }
}

bool ranges_overlap(const void* left, size_t left_bytes, const void* right,
                    size_t right_bytes) {
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
  const uintptr_t left_end = left_begin + left_bytes;
  const uintptr_t right_end = right_begin + right_bytes;
  return left_begin < right_end && right_begin < left_end;
}

template <typename Ops, typename Input>
int launch_silu_and_mul(const Input* input, Input* output, uint32_t rows,
                        uint32_t width, void* stream) {
  if (input == nullptr || output == nullptr || rows == 0 || width == 0) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  using Scalar = typename Ops::Scalar;
  constexpr uint32_t kPackWidth =
      static_cast<uint32_t>(kVectorBytes / sizeof(Scalar));
  const size_t output_elements =
      static_cast<size_t>(rows) * static_cast<size_t>(width);
  if (ranges_overlap(input, output_elements * 2U * sizeof(Scalar), output,
                     output_elements * sizeof(Scalar))) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  const uintptr_t combined_address = reinterpret_cast<uintptr_t>(input) |
                                     reinterpret_cast<uintptr_t>(output);
  const bool can_vectorize = width % kPackWidth == 0U &&
                             combined_address % kVectorBytes == 0U;
  uint32_t work_items = can_vectorize ? width / kPackWidth : width;
  const uint32_t threads = std::min(work_items, kMaximumThreads);
  if (can_vectorize) {
    silu_and_mul_kernel<Ops, true>
        <<<rows, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const Scalar*>(input),
            reinterpret_cast<Scalar*>(output), width);
  } else {
    silu_and_mul_kernel<Ops, false>
        <<<rows, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const Scalar*>(input),
            reinterpret_cast<Scalar*>(output), width);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_silu_and_mul_f32(const float* input, float* output,
                                            uint32_t rows, uint32_t width,
                                            void* stream) {
  return launch_silu_and_mul<FloatOps>(input, output, rows, width, stream);
}

extern "C" int loom_cuda_silu_and_mul_f16(const uint16_t* input,
                                            uint16_t* output, uint32_t rows,
                                            uint32_t width, void* stream) {
  return launch_silu_and_mul<HalfOps>(input, output, rows, width, stream);
}

extern "C" int loom_cuda_silu_and_mul_bf16(const uint16_t* input,
                                             uint16_t* output, uint32_t rows,
                                             uint32_t width, void* stream) {
  return launch_silu_and_mul<Bfloat16Ops>(input, output, rows, width, stream);
}
