#include "common.h"

namespace loom_kernels::torch_adapter {

void check_logits_preprocess_contract(
    const Tensor& logits, const Tensor& temperatures,
    const std::optional<Tensor>& blocked_mask,
    const std::optional<Tensor>& bias_row_ids,
    const std::optional<Tensor>& bias_token_ids,
    const std::optional<Tensor>& bias_values,
    const std::optional<Tensor>& suppressed_row_ids,
    const std::optional<Tensor>& suppressed_token_ids) {
  STD_TORCH_CHECK(
      logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0,
      "Loom logits preprocessing requires non-empty rank-2 logits");
  STD_TORCH_CHECK(
      logits.size(0) <= std::numeric_limits<int32_t>::max() &&
          logits.size(1) <= std::numeric_limits<int32_t>::max(),
      "Loom logits-preprocessing shape exceeds the int32 sparse-index ABI");
  STD_TORCH_CHECK(logits.is_cuda(),
                  "Loom logits-preprocessing logits must be CUDA");
  STD_TORCH_CHECK(logits.scalar_type() == ScalarType::Float,
                  "Loom logits-preprocessing logits must be F32");
  STD_TORCH_CHECK(
      logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
      "Loom logits preprocessing requires unit vocabulary stride and "
      "non-overlapping positive row stride");

  STD_TORCH_CHECK(
      temperatures.device() == logits.device() &&
          temperatures.scalar_type() == ScalarType::Float &&
          temperatures.dim() == 1 &&
          temperatures.size(0) == logits.size(0) &&
          temperatures.is_contiguous(),
      "Loom logits-preprocessing temperatures must be same-device contiguous "
      "F32 [rows]");
  STD_TORCH_CHECK(
      !byte_ranges_overlap(logits, temperatures),
      "Loom logits and preprocessing temperatures must not overlap");

  if (blocked_mask.has_value()) {
    STD_TORCH_CHECK(
        blocked_mask->device() == logits.device() &&
            blocked_mask->scalar_type() == ScalarType::Bool &&
            blocked_mask->dim() == 2 &&
            blocked_mask->size(0) == logits.size(0) &&
            blocked_mask->size(1) == logits.size(1) &&
            blocked_mask->is_contiguous(),
        "Loom blocked mask must be same-device contiguous bool [rows, vocab]");
    STD_TORCH_CHECK(
        !byte_ranges_overlap(logits, *blocked_mask),
        "Loom logits and blocked mask must not overlap");
  }

  const bool has_bias = bias_row_ids.has_value();
  STD_TORCH_CHECK(
      has_bias == bias_token_ids.has_value() &&
          has_bias == bias_values.has_value(),
      "Loom sparse bias row IDs, token IDs, and values must be supplied "
      "together");
  if (has_bias) {
    const int64_t count = bias_row_ids->numel();
    STD_TORCH_CHECK(
        bias_row_ids->device() == logits.device() &&
            bias_token_ids->device() == logits.device() &&
            bias_values->device() == logits.device(),
        "Loom sparse bias tensors must share the logits CUDA device");
    STD_TORCH_CHECK(
        bias_row_ids->scalar_type() == ScalarType::Int &&
            bias_token_ids->scalar_type() == ScalarType::Int &&
            bias_values->scalar_type() == ScalarType::Float,
        "Loom sparse bias row/token IDs must be int32 and values must be F32");
    STD_TORCH_CHECK(
        bias_row_ids->dim() == 1 && bias_token_ids->dim() == 1 &&
            bias_values->dim() == 1 && count > 0 &&
            bias_token_ids->numel() == count &&
            bias_values->numel() == count &&
            bias_row_ids->is_contiguous() &&
            bias_token_ids->is_contiguous() &&
            bias_values->is_contiguous(),
        "Loom sparse bias tensors must be non-empty contiguous vectors with "
        "equal lengths");
    STD_TORCH_CHECK(
        count <= std::numeric_limits<uint32_t>::max(),
        "Loom sparse bias count exceeds the CUDA ABI");
    STD_TORCH_CHECK(
        !byte_ranges_overlap(logits, *bias_row_ids) &&
            !byte_ranges_overlap(logits, *bias_token_ids) &&
            !byte_ranges_overlap(logits, *bias_values),
        "Loom logits and sparse bias tensors must not overlap");
  }

  const bool has_suppression = suppressed_row_ids.has_value();
  STD_TORCH_CHECK(
      has_suppression == suppressed_token_ids.has_value(),
      "Loom sparse suppression row and token IDs must be supplied together");
  if (has_suppression) {
    const int64_t count = suppressed_row_ids->numel();
    STD_TORCH_CHECK(
        suppressed_row_ids->device() == logits.device() &&
            suppressed_token_ids->device() == logits.device(),
        "Loom sparse suppression tensors must share the logits CUDA device");
    STD_TORCH_CHECK(
        suppressed_row_ids->scalar_type() == ScalarType::Int &&
            suppressed_token_ids->scalar_type() == ScalarType::Int,
        "Loom sparse suppression row/token IDs must be int32");
    STD_TORCH_CHECK(
        suppressed_row_ids->dim() == 1 &&
            suppressed_token_ids->dim() == 1 && count > 0 &&
            suppressed_token_ids->numel() == count &&
            suppressed_row_ids->is_contiguous() &&
            suppressed_token_ids->is_contiguous(),
        "Loom sparse suppression tensors must be non-empty contiguous vectors "
        "with equal lengths");
    STD_TORCH_CHECK(
        count <= std::numeric_limits<uint32_t>::max(),
        "Loom sparse suppression count exceeds the CUDA ABI");
    STD_TORCH_CHECK(
        !byte_ranges_overlap(logits, *suppressed_row_ids) &&
            !byte_ranges_overlap(logits, *suppressed_token_ids),
        "Loom logits and sparse suppression tensors must not overlap");
  }
}

void logits_preprocess(
    Tensor logits, const Tensor& temperatures,
    const std::optional<Tensor>& blocked_mask,
    const std::optional<Tensor>& bias_row_ids,
    const std::optional<Tensor>& bias_token_ids,
    const std::optional<Tensor>& bias_values,
    const std::optional<Tensor>& suppressed_row_ids,
    const std::optional<Tensor>& suppressed_token_ids) {
  check_logits_preprocess_contract(
      logits, temperatures, blocked_mask, bias_row_ids, bias_token_ids,
      bias_values, suppressed_row_ids, suppressed_token_ids);
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const int status = loom_cuda_bridge_logits_preprocess(
      logits.mutable_data_ptr<float>(), storage_span_elements(logits),
      temperatures.const_data_ptr<float>(),
      static_cast<uint64_t>(temperatures.numel()),
      blocked_mask.has_value()
          ? reinterpret_cast<const uint8_t*>(blocked_mask->const_data_ptr())
          : nullptr,
      blocked_mask.has_value()
          ? static_cast<uint64_t>(blocked_mask->numel())
          : 0U,
      bias_row_ids.has_value() ? bias_row_ids->const_data_ptr<int32_t>()
                               : nullptr,
      bias_row_ids.has_value()
          ? static_cast<uint64_t>(bias_row_ids->numel())
          : 0U,
      bias_token_ids.has_value() ? bias_token_ids->const_data_ptr<int32_t>()
                                 : nullptr,
      bias_token_ids.has_value()
          ? static_cast<uint64_t>(bias_token_ids->numel())
          : 0U,
      bias_values.has_value() ? bias_values->const_data_ptr<float>() : nullptr,
      bias_values.has_value() ? static_cast<uint64_t>(bias_values->numel())
                              : 0U,
      suppressed_row_ids.has_value()
          ? suppressed_row_ids->const_data_ptr<int32_t>()
          : nullptr,
      suppressed_row_ids.has_value()
          ? static_cast<uint64_t>(suppressed_row_ids->numel())
          : 0U,
      suppressed_token_ids.has_value()
          ? suppressed_token_ids->const_data_ptr<int32_t>()
          : nullptr,
      suppressed_token_ids.has_value()
          ? static_cast<uint64_t>(suppressed_token_ids->numel())
          : 0U,
      rows, vocab_size, static_cast<uint64_t>(logits.stride(0)),
      stream.stream());
  check_bridge_status(status, "logits preprocessing");
}

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
      "logits_preprocess_",
      TORCH_BOX(&loom_kernels::torch_adapter::logits_preprocess));
  library.impl(
      "min_p_filter_",
      TORCH_BOX(&loom_kernels::torch_adapter::min_p_filter));
}
