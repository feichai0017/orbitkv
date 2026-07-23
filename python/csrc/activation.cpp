#include "common.h"

namespace loom_kernels::torch_adapter {

void check_silu_and_mul_contract(const Tensor& input,
                                 const Tensor& output) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom SiLU-and-Mul input must be CUDA");
  STD_TORCH_CHECK(output.device() == input.device(),
              "Loom SiLU-and-Mul tensors must be on the same CUDA device");
  STD_TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "Loom SiLU-and-Mul input/output dtypes must match");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Float ||
                  input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom SiLU-and-Mul supports F32, FP16, and BF16");
  STD_TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "Loom SiLU-and-Mul tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom SiLU-and-Mul input must be non-empty");
  STD_TORCH_CHECK(input.size(-1) % 2 == 0,
              "Loom SiLU-and-Mul input last dimension must be even");
  STD_TORCH_CHECK(output.dim() == input.dim(),
              "Loom SiLU-and-Mul output rank must match input");
  for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
    STD_TORCH_CHECK(output.size(dimension) == input.size(dimension),
                "Loom SiLU-and-Mul output prefix shape must match input");
  }
  STD_TORCH_CHECK(output.size(-1) == input.size(-1) / 2,
              "Loom SiLU-and-Mul output last dimension must be half input");
  STD_TORCH_CHECK(!byte_ranges_overlap(input, output),
              "Loom SiLU-and-Mul input/output storage must not overlap");
}

void launch_silu_and_mul(const Tensor& input, Tensor output) {
  const int64_t width_i64 = input.size(-1) / 2;
  const int64_t rows_i64 = input.numel() / input.size(-1);
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  width_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom SiLU-and-Mul shape exceeds the CUDA ABI");

  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto width = static_cast<uint32_t>(width_i64);
  const int status = loom_cuda_bridge_silu_and_mul(
      bridge_dtype(input), input.const_data_ptr(),
      static_cast<uint64_t>(input.numel()), output.mutable_data_ptr(),
      static_cast<uint64_t>(output.numel()), rows, width, stream.stream());
  check_bridge_status(status, "SiLU-and-Mul");
}

void silu_and_mul(const Tensor& input, Tensor output) {
  check_silu_and_mul_contract(input, output);
  launch_silu_and_mul(input, output);
}

void check_silu_and_mul_dynamic_fp8_contract(const Tensor& input,
                                              const Tensor& output,
                                              const Tensor& scales,
                                              int64_t group_size,
                                              bool scales_transposed = false) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom SiLU-and-Mul+FP8 input must be CUDA");
  STD_TORCH_CHECK(output.device() == input.device() &&
                  scales.device() == input.device(),
              "Loom SiLU-and-Mul+FP8 tensors must be on the same CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom SiLU-and-Mul+FP8 supports FP16 and BF16 input");
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::Float8_e4m3fn,
              "Loom SiLU-and-Mul+FP8 output must use torch.float8_e4m3fn");
  STD_TORCH_CHECK(scales.scalar_type() == ScalarType::Float,
              "Loom SiLU-and-Mul+FP8 scales must use F32");
  STD_TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "Loom SiLU-and-Mul+FP8 input/output must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom SiLU-and-Mul+FP8 input must be non-empty");
  STD_TORCH_CHECK(input.size(-1) % 2 == 0,
              "Loom SiLU-and-Mul+FP8 input last dimension must be even");
  STD_TORCH_CHECK(group_size == 64 || group_size == 128,
              "Loom SiLU-and-Mul+FP8 group size must be 64 or 128");
  const int64_t width = input.size(-1) / 2;
  STD_TORCH_CHECK(width % group_size == 0,
              "Loom SiLU-and-Mul+FP8 width must be divisible by group size");
  STD_TORCH_CHECK(output.dim() == input.dim(),
              "Loom SiLU-and-Mul+FP8 output rank must match input");
  for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
    STD_TORCH_CHECK(output.size(dimension) == input.size(dimension),
                "Loom SiLU-and-Mul+FP8 output prefix shape must match input");
  }
  STD_TORCH_CHECK(output.size(-1) == width,
              "Loom SiLU-and-Mul+FP8 output last dimension must be half input");
  const int64_t rows = input.numel() / input.size(-1);
  STD_TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == width / group_size,
              "Loom SiLU-and-Mul+FP8 scales must have shape "
              "[rows, width / group_size]");
  if (scales_transposed) {
    STD_TORCH_CHECK(scales.stride(0) == 1 && scales.stride(1) == rows,
                "Loom transposed FP8 scales must use group-major storage");
  } else {
    STD_TORCH_CHECK(scales.is_contiguous(),
                "Loom row-major FP8 scales must be contiguous");
  }
  STD_TORCH_CHECK(!byte_ranges_overlap(input, output) &&
                  !byte_ranges_overlap(input, scales) &&
                  !byte_ranges_overlap(output, scales),
              "Loom SiLU-and-Mul+FP8 mutable tensor storage must not overlap");
}

void launch_silu_and_mul_dynamic_fp8_layout(
    const Tensor& input, Tensor output, Tensor scales,
    int64_t group_size_i64, const std::optional<Tensor>& scale_ub,
    bool scales_transposed) {
  const int64_t width_i64 = input.size(-1) / 2;
  const int64_t rows_i64 = input.numel() / input.size(-1);
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  width_i64 <= std::numeric_limits<uint32_t>::max() &&
                  group_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom SiLU-and-Mul+FP8 shape exceeds the CUDA ABI");

  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto width = static_cast<uint32_t>(width_i64);
  const auto group_size = static_cast<uint32_t>(group_size_i64);
  auto* output_bytes =
      reinterpret_cast<uint8_t*>(output.mutable_data_ptr());
  auto* scale_values = scales.mutable_data_ptr<float>();
  const float* scale_ub_value =
      scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;
  const int status = loom_cuda_bridge_silu_and_mul_dynamic_fp8(
      bridge_dtype(input), input.const_data_ptr(),
      static_cast<uint64_t>(input.numel()), output_bytes,
      static_cast<uint64_t>(output.numel()), scale_values,
      static_cast<uint64_t>(scales.numel()), scale_ub_value,
      scale_ub.has_value() ? static_cast<uint64_t>(scale_ub->numel()) : 0U,
      rows, width, group_size, scales_transposed ? 1U : 0U, stream.stream());
  check_bridge_status(status, "SiLU-and-Mul+FP8");
}

void launch_silu_and_mul_dynamic_fp8(const Tensor& input,
                                      Tensor output, Tensor scales,
                                      int64_t group_size) {
  launch_silu_and_mul_dynamic_fp8_layout(input, output, scales, group_size,
                                         std::nullopt, false);
}

void silu_and_mul_dynamic_fp8(const Tensor& input, Tensor output,
                              Tensor scales, int64_t group_size) {
  check_silu_and_mul_dynamic_fp8_contract(input, output, scales, group_size);
  launch_silu_and_mul_dynamic_fp8(input, output, scales, group_size);
}

void vllm_silu_and_mul_per_block_fp8(
    Tensor output, const Tensor& input, Tensor scales,
    int64_t group_size, const std::optional<Tensor>& scale_ub,
    bool scales_transposed) {
  check_silu_and_mul_dynamic_fp8_contract(input, output, scales, group_size,
                                          scales_transposed);
  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(scale_ub->device() == input.device() &&
                    scale_ub->scalar_type() == ScalarType::Float &&
                    scale_ub->numel() == 1 && scale_ub->is_contiguous(),
                "Loom FP8 scale upper bound must be one same-device F32 value");
  }
  launch_silu_and_mul_dynamic_fp8_layout(input, output, scales, group_size,
                                         scale_ub, scales_transposed);
}


}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "silu_and_mul",
      TORCH_BOX(&loom_kernels::torch_adapter::silu_and_mul));
  library.impl(
      "silu_and_mul_dynamic_fp8",
      TORCH_BOX(&loom_kernels::torch_adapter::silu_and_mul_dynamic_fp8));
  library.impl(
      "silu_and_mul_per_block_fp8",
      TORCH_BOX(&loom_kernels::torch_adapter::vllm_silu_and_mul_per_block_fp8));
}
