#include "common.h"

namespace loom_kernels::torch_adapter {

void check_rms_norm_contract(const Tensor& input,
                             const Tensor& weight,
                             const Tensor& output, double epsilon) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom RMSNorm input must be CUDA");
  STD_TORCH_CHECK(weight.device() == input.device() &&
                  output.device() == input.device(),
              "Loom RMSNorm tensors must be on the same CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == weight.scalar_type() &&
                  output.scalar_type() == input.scalar_type(),
              "Loom RMSNorm tensors must have matching dtypes");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Float ||
                  input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom RMSNorm supports F32, FP16, and BF16");
  STD_TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous(),
              "Loom RMSNorm tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom RMSNorm input must be non-empty");
  STD_TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom RMSNorm weight must match the hidden dimension");
  STD_TORCH_CHECK(output.sizes().equals(input.sizes()),
              "Loom RMSNorm output shape must match input");
  STD_TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom RMSNorm epsilon must be finite and positive");
  STD_TORCH_CHECK(!byte_ranges_overlap(output, input) &&
                  !byte_ranges_overlap(output, weight),
              "Loom RMSNorm output storage must not overlap inputs");
}

void rms_norm(const Tensor& input, const Tensor& weight,
              Tensor output, double epsilon) {
  check_rms_norm_contract(input, weight, output, epsilon);
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom RMSNorm shape exceeds the bridge ABI");
  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const int status = loom_cuda_bridge_rms_norm(
      bridge_dtype(input), input.const_data_ptr(),
      static_cast<uint64_t>(input.numel()), weight.const_data_ptr(),
      static_cast<uint64_t>(weight.numel()), output.mutable_data_ptr(),
      static_cast<uint64_t>(output.numel()),
      static_cast<uint32_t>(rows_i64),
      static_cast<uint32_t>(hidden_size_i64), static_cast<float>(epsilon),
      stream.stream());
  check_bridge_status(status, "RMSNorm");
}

void check_contract(const Tensor& input, const Tensor& residual,
                    const Tensor& weight, double epsilon) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom Add+RMSNorm input must be CUDA");
  STD_TORCH_CHECK(residual.device() == input.device() &&
                  weight.device() == input.device(),
              "Loom Add+RMSNorm tensors must be on the same CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type() &&
                  input.scalar_type() == weight.scalar_type(),
              "Loom Add+RMSNorm tensors must have matching dtypes");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Float ||
                  input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom Add+RMSNorm supports F32, FP16, and BF16");
  STD_TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
                  weight.is_contiguous(),
              "Loom Add+RMSNorm tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom Add+RMSNorm input must be non-empty");
  STD_TORCH_CHECK(input.sizes().equals(residual.sizes()),
              "Loom Add+RMSNorm input/residual shapes must match");
  STD_TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom Add+RMSNorm weight must match the hidden dimension");
  STD_TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom Add+RMSNorm epsilon must be finite and positive");
  STD_TORCH_CHECK(!byte_ranges_overlap(input, residual) &&
                  !byte_ranges_overlap(input, weight) &&
                  !byte_ranges_overlap(residual, weight),
              "Loom Add+RMSNorm tensor storage ranges must not overlap");
}

void launch_add_rms_norm(Tensor input, Tensor residual,
                         const Tensor& weight, double epsilon) {
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom Add+RMSNorm shape exceeds the CUDA ABI");

  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto hidden_size = static_cast<uint32_t>(hidden_size_i64);
  const auto input_elements = static_cast<uint64_t>(input.numel());
  const auto residual_elements = static_cast<uint64_t>(residual.numel());
  const auto weight_elements = static_cast<uint64_t>(weight.numel());
  const auto epsilon_f32 = static_cast<float>(epsilon);
  const int status = loom_cuda_bridge_add_rms_norm(
      bridge_dtype(input), input.mutable_data_ptr(), input_elements,
      residual.mutable_data_ptr(), residual_elements, weight.const_data_ptr(),
      weight_elements, rows, hidden_size, epsilon_f32, stream.stream());
  check_bridge_status(status, "Add+RMSNorm");
}

void add_rms_norm_mut(Tensor input, Tensor residual,
                      const Tensor& weight, double epsilon) {
  check_contract(input, residual, weight, epsilon);
  launch_add_rms_norm(input, residual, weight, epsilon);
}

void check_dynamic_fp8_contract(const Tensor& input,
                                const Tensor& weight,
                                const Tensor& output,
                                const Tensor& scales, double epsilon) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom RMSNorm+FP8 input must be CUDA");
  STD_TORCH_CHECK(weight.device() == input.device() &&
                  output.device() == input.device() &&
                  scales.device() == input.device(),
              "Loom RMSNorm+FP8 tensors must be on the same CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "Loom RMSNorm+FP8 input and weight dtypes must match");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Float ||
                  input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom RMSNorm+FP8 supports F32, FP16, and BF16 inputs");
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::Float8_e4m3fn,
              "Loom RMSNorm+FP8 output must use torch.float8_e4m3fn");
  STD_TORCH_CHECK(scales.scalar_type() == ScalarType::Float,
              "Loom RMSNorm+FP8 scales must use F32");
  STD_TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous() && scales.is_contiguous(),
              "Loom RMSNorm+FP8 tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom RMSNorm+FP8 input must be non-empty");
  STD_TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom RMSNorm+FP8 weight must match the hidden dimension");
  STD_TORCH_CHECK(output.sizes().equals(input.sizes()),
              "Loom RMSNorm+FP8 output shape must match input");
  const int64_t rows = input.numel() / input.size(-1);
  STD_TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == 1,
              "Loom RMSNorm+FP8 scales must have shape [rows, 1]");
  STD_TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom RMSNorm+FP8 epsilon must be finite and positive");
  STD_TORCH_CHECK(!byte_ranges_overlap(output, input) &&
                  !byte_ranges_overlap(output, weight) &&
                  !byte_ranges_overlap(output, scales) &&
                  !byte_ranges_overlap(scales, input) &&
                  !byte_ranges_overlap(scales, weight),
              "Loom RMSNorm+FP8 mutable tensor storage must not overlap");
}

void launch_rms_norm_dynamic_fp8(const Tensor& input,
                                 const Tensor& weight, Tensor output,
                                 Tensor scales, double epsilon) {
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom RMSNorm+FP8 shape exceeds the CUDA ABI");

  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto hidden_size = static_cast<uint32_t>(hidden_size_i64);
  const auto input_elements = static_cast<uint64_t>(input.numel());
  const auto weight_elements = static_cast<uint64_t>(weight.numel());
  const auto output_elements = static_cast<uint64_t>(output.numel());
  const auto scale_elements = static_cast<uint64_t>(scales.numel());
  const auto epsilon_f32 = static_cast<float>(epsilon);
  auto* output_bytes =
      reinterpret_cast<uint8_t*>(output.mutable_data_ptr());
  auto* scale_values = scales.mutable_data_ptr<float>();
  const int status = loom_cuda_bridge_rms_norm_dynamic_fp8(
      bridge_dtype(input), input.const_data_ptr(), input_elements,
      weight.const_data_ptr(), weight_elements, output_bytes, output_elements,
      scale_values, scale_elements, rows, hidden_size, epsilon_f32,
      stream.stream());
  check_bridge_status(status, "RMSNorm+FP8");
}

void rms_norm_dynamic_fp8(const Tensor& input, const Tensor& weight,
                          Tensor output, Tensor scales,
                          double epsilon) {
  check_dynamic_fp8_contract(input, weight, output, scales, epsilon);
  launch_rms_norm_dynamic_fp8(input, weight, output, scales, epsilon);
}

}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "rms_norm",
      TORCH_BOX(&loom_kernels::torch_adapter::rms_norm));
  library.impl(
      "add_rms_norm_mut",
      TORCH_BOX(&loom_kernels::torch_adapter::add_rms_norm_mut));
  library.impl(
      "rms_norm_dynamic_fp8",
      TORCH_BOX(&loom_kernels::torch_adapter::rms_norm_dynamic_fp8));
}
