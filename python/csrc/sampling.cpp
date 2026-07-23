#include "common.h"

namespace loom_kernels::torch_adapter {

void check_greedy_sample_logprobs_shape(const Tensor& logits) {
  STD_TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0,
              "Loom greedy sampling logits must be non-empty rank-2");
  STD_TORCH_CHECK(logits.size(0) <= std::numeric_limits<uint32_t>::max() &&
                  logits.size(1) <= std::numeric_limits<int32_t>::max(),
              "Loom greedy sampling shape exceeds the CUDA ABI");
}

void check_greedy_sample_logprobs_contract(const Tensor& logits) {
  check_greedy_sample_logprobs_shape(logits);
  STD_TORCH_CHECK(logits.is_cuda(), "Loom greedy sampling logits must be CUDA");
  STD_TORCH_CHECK(logits.scalar_type() == ScalarType::Float ||
                  logits.scalar_type() == ScalarType::Half ||
                  logits.scalar_type() == ScalarType::BFloat16,
              "Loom greedy sampling supports F32, FP16, and BF16 logits");
  STD_TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
              "Loom greedy sampling logits require unit vocabulary stride "
              "and non-overlapping positive row stride");
}

std::tuple<Tensor, Tensor, Tensor>
launch_greedy_sample_logprobs(const Tensor& logits) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  const auto logits_elements = storage_span_elements(logits);
  const auto output_elements = static_cast<uint64_t>(logits.size(0));
  Tensor token_ids = new_empty(logits, {logits.size(0)}, ScalarType::Int);
  Tensor logprobs = new_empty(logits, {logits.size(0)}, ScalarType::Float);
  Tensor ranks = new_empty(logits, {logits.size(0)}, ScalarType::Long);

  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const int status = loom_cuda_bridge_greedy_sample_logprobs(
      bridge_dtype(logits), logits.const_data_ptr(), logits_elements,
      token_ids.mutable_data_ptr<int32_t>(), output_elements,
      logprobs.mutable_data_ptr<float>(), output_elements,
      ranks.mutable_data_ptr<int64_t>(),
      output_elements, rows, vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "greedy-sampling");
  return {token_ids, logprobs, ranks};
}

std::tuple<Tensor, Tensor, Tensor> greedy_sample_logprobs(
    const Tensor& logits) {
  check_greedy_sample_logprobs_contract(logits);
  return launch_greedy_sample_logprobs(logits);
}

std::tuple<Tensor, Tensor, Tensor> greedy_sample_logprobs_meta(
    const Tensor& logits) {
  check_greedy_sample_logprobs_shape(logits);
  return {
      new_empty(logits, {logits.size(0)}, ScalarType::Int),
      new_empty(logits, {logits.size(0)}, ScalarType::Float),
      new_empty(logits, {logits.size(0)}, ScalarType::Long),
  };
}

void check_selected_token_logprobs_shape(const Tensor& logits,
                                         const Tensor& token_ids) {
  check_greedy_sample_logprobs_shape(logits);
  STD_TORCH_CHECK(token_ids.dim() == 1 && token_ids.size(0) == logits.size(0),
              "Loom selected token IDs must contain one value per logits row");
}

void check_selected_token_logprobs_contract(const Tensor& logits,
                                            const Tensor& token_ids) {
  check_greedy_sample_logprobs_contract(logits);
  check_selected_token_logprobs_shape(logits, token_ids);
  STD_TORCH_CHECK(token_ids.device() == logits.device(),
              "Loom selected token IDs and logits must share a CUDA device");
  STD_TORCH_CHECK(token_ids.scalar_type() == ScalarType::Long,
              "Loom selected token IDs must be int64");
  STD_TORCH_CHECK(token_ids.is_contiguous(),
              "Loom selected token IDs must be contiguous");
}

std::tuple<Tensor, Tensor> launch_selected_token_logprobs(
    const Tensor& logits, const Tensor& token_ids) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  Tensor logprobs = new_empty(logits, {logits.size(0)}, ScalarType::Float);
  Tensor ranks = new_empty(logits, {logits.size(0)}, ScalarType::Long);

  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const auto output_elements = static_cast<uint64_t>(logits.size(0));
  const int status = loom_cuda_bridge_selected_token_logprobs(
      bridge_dtype(logits), logits.const_data_ptr(),
      storage_span_elements(logits), token_ids.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(token_ids.numel()),
      logprobs.mutable_data_ptr<float>(), output_elements,
      ranks.mutable_data_ptr<int64_t>(),
      output_elements, rows, vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "selected-token logprob");
  return {logprobs, ranks};
}

std::tuple<Tensor, Tensor> selected_token_logprobs(
    const Tensor& logits, const Tensor& token_ids) {
  check_selected_token_logprobs_contract(logits, token_ids);
  return launch_selected_token_logprobs(logits, token_ids);
}

std::tuple<Tensor, Tensor> selected_token_logprobs_meta(
    const Tensor& logits, const Tensor& token_ids) {
  check_selected_token_logprobs_shape(logits, token_ids);
  return {
      new_empty(logits, {logits.size(0)}, ScalarType::Float),
      new_empty(logits, {logits.size(0)}, ScalarType::Long),
  };
}


}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "greedy_sample_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::greedy_sample_logprobs));
  library.impl(
      "selected_token_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::selected_token_logprobs));
}

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, Meta, library) {
  library.impl(
      "greedy_sample_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::greedy_sample_logprobs_meta));
  library.impl(
      "selected_token_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::selected_token_logprobs_meta));
}
