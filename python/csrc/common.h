#pragma once

#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))

#include "loom_cuda_bridge.h"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace loom_kernels::torch_adapter {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

class CudaDeviceGuard final {
 public:
  explicit CudaDeviceGuard(const torch::stable::Device& device)
      : guard_(device.index()) {}

 private:
  torch::stable::accelerator::DeviceGuard guard_;
};

class CurrentCudaStream final {
 public:
  explicit CurrentCudaStream(int32_t device_index) {
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_));
  }

  void* stream() const {
    return stream_;
  }

 private:
  void* stream_ = nullptr;
};

inline CurrentCudaStream current_cuda_stream(int32_t device_index) {
  return CurrentCudaStream(device_index);
}

inline uint64_t tensor_nbytes(const Tensor& tensor) {
  STD_TORCH_CHECK(tensor.numel() >= 0, "negative tensor element count");
  const auto elements = static_cast<uint64_t>(tensor.numel());
  const auto element_size = static_cast<uint64_t>(tensor.element_size());
  STD_TORCH_CHECK(
      element_size == 0 ||
          elements <= std::numeric_limits<uint64_t>::max() / element_size,
      "tensor byte size exceeds uint64");
  return elements * element_size;
}

inline Tensor new_empty(
    const Tensor& reference,
    std::initializer_list<int64_t> sizes,
    ScalarType dtype) {
  return torch::stable::new_empty(reference, sizes, dtype);
}

inline uint32_t bridge_dtype(const Tensor& tensor) {
  if (tensor.scalar_type() == ScalarType::Float) {
    return LOOM_CUDA_BRIDGE_F32;
  }
  if (tensor.scalar_type() == ScalarType::Half) {
    return LOOM_CUDA_BRIDGE_F16;
  }
  if (tensor.scalar_type() == ScalarType::BFloat16) {
    return LOOM_CUDA_BRIDGE_BF16;
  }
  STD_TORCH_CHECK(false, "unsupported Loom bridge dtype");
}

inline uint64_t storage_span_elements(const Tensor& tensor) {
  STD_TORCH_CHECK(tensor.numel() > 0,
              "Loom bridge tensors must contain at least one element");
  uint64_t span = 1;
  for (int64_t dimension = 0; dimension < tensor.dim(); ++dimension) {
    STD_TORCH_CHECK(tensor.size(dimension) > 0 && tensor.stride(dimension) > 0,
                "Loom bridge requires positive tensor sizes and strides");
    const auto extent =
        static_cast<uint64_t>(tensor.size(dimension) - 1);
    const auto stride = static_cast<uint64_t>(tensor.stride(dimension));
    STD_TORCH_CHECK(
        extent == 0 ||
            stride <=
                (std::numeric_limits<uint64_t>::max() - span) / extent,
        "Loom tensor storage span exceeds the bridge ABI");
    span += extent * stride;
  }
  return span;
}

inline void check_bridge_status(int status, const char* operation) {
  STD_TORCH_CHECK(status == LOOM_CUDA_BRIDGE_SUCCESS, "Loom Rust ", operation,
              " bridge failed: ", loom_cuda_bridge_last_error_message(),
              " (status ", status, ")");
}

inline bool byte_ranges_overlap(const Tensor& left, const Tensor& right) {
  const auto left_begin =
      reinterpret_cast<uintptr_t>(left.const_data_ptr());
  const auto right_begin =
      reinterpret_cast<uintptr_t>(right.const_data_ptr());
  const auto left_bytes = tensor_nbytes(left);
  const auto right_bytes = tensor_nbytes(right);
  STD_TORCH_CHECK(
      left_bytes <= std::numeric_limits<uintptr_t>::max() - left_begin &&
          right_bytes <= std::numeric_limits<uintptr_t>::max() - right_begin,
      "tensor byte range exceeds uintptr_t");
  const auto left_end = left_begin + left_bytes;
  const auto right_end = right_begin + right_bytes;
  return left_begin < right_end && right_begin < left_end;
}

inline bool has_dense_nhd_inner_strides(const Tensor& tensor) {
  if (tensor.dim() != 4) {
    return false;
  }
  const int64_t block_elements =
      tensor.size(1) * tensor.size(2) * tensor.size(3);
  return tensor.stride(3) == 1 &&
         tensor.stride(2) == tensor.size(3) &&
         tensor.stride(1) == tensor.size(2) * tensor.size(3) &&
         tensor.stride(0) >= block_elements;
}

}  // namespace loom_kernels::torch_adapter
