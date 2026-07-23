#include "common.h"

namespace loom_kernels::torch_adapter {

void check_min_p_filter_shape(const Tensor& logits,
                              const Tensor& min_p) {
  STD_TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0,
              "Loom min-p logits must be non-empty rank-2");
  STD_TORCH_CHECK(logits.size(0) <= std::numeric_limits<uint32_t>::max() &&
                  logits.size(1) <= std::numeric_limits<uint32_t>::max(),
              "Loom min-p shape exceeds the CUDA ABI");
  STD_TORCH_CHECK((min_p.dim() == 1 && min_p.size(0) == logits.size(0)) ||
                  (min_p.dim() == 2 && min_p.size(0) == logits.size(0) &&
                   min_p.size(1) == 1),
              "Loom min-p probabilities must have shape [rows] or [rows, 1]");
}

void check_min_p_filter_contract(const Tensor& logits,
                                 const Tensor& min_p) {
  check_min_p_filter_shape(logits, min_p);
  STD_TORCH_CHECK(logits.is_cuda(), "Loom min-p logits must be CUDA");
  STD_TORCH_CHECK(min_p.device() == logits.device(),
              "Loom min-p probabilities and logits must share a CUDA device");
  STD_TORCH_CHECK(logits.scalar_type() == ScalarType::Float ||
                  logits.scalar_type() == ScalarType::Half ||
                  logits.scalar_type() == ScalarType::BFloat16,
              "Loom min-p supports F32, FP16, and BF16 logits");
  STD_TORCH_CHECK(min_p.scalar_type() == ScalarType::Float,
              "Loom min-p probabilities must use F32");
  STD_TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
              "Loom min-p logits require unit vocabulary stride and "
              "non-overlapping positive row stride");
  STD_TORCH_CHECK(min_p.is_contiguous(),
              "Loom min-p probabilities must be contiguous");
  STD_TORCH_CHECK(!byte_ranges_overlap(logits, min_p),
              "Loom min-p logits and probabilities must not overlap");
}

void launch_min_p_filter(Tensor logits, const Tensor& min_p) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const int status = loom_cuda_bridge_min_p_filter(
      bridge_dtype(logits), logits.mutable_data_ptr(),
      storage_span_elements(logits), min_p.const_data_ptr<float>(),
      static_cast<uint64_t>(min_p.numel()), rows,
      vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "min-p");
}

void min_p_filter(Tensor logits, const Tensor& min_p) {
  check_min_p_filter_contract(logits, min_p);
  launch_min_p_filter(logits, min_p);
}


}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "min_p_filter_",
      TORCH_BOX(&loom_kernels::torch_adapter::min_p_filter));
}
