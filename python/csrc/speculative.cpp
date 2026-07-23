#include "common.h"

namespace loom_kernels::torch_adapter {

void check_greedy_speculative_verify_shape(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    int64_t max_draft_tokens) {
  STD_TORCH_CHECK(
      draft_token_ids.dim() == 1 && draft_token_ids.numel() > 0,
      "Loom greedy speculative draft token IDs must be non-empty rank-1");
  STD_TORCH_CHECK(
      target_token_ids.dim() == 1 &&
          target_token_ids.numel() == draft_token_ids.numel(),
      "Loom greedy speculative target token IDs must match the flattened "
      "draft shape");
  STD_TORCH_CHECK(
      cumulative_draft_lengths.dim() == 1 &&
          cumulative_draft_lengths.numel() > 0,
      "Loom cumulative draft lengths must be non-empty rank-1");
  const int64_t requests = cumulative_draft_lengths.numel();
  STD_TORCH_CHECK(
      bonus_token_ids.dim() == 2 && bonus_token_ids.size(0) == requests &&
          bonus_token_ids.size(1) == 1,
      "Loom speculative bonus token IDs must have shape [requests, 1]");
  STD_TORCH_CHECK(
      max_draft_tokens > 0 &&
          max_draft_tokens <
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "Loom maximum draft length must fit the CUDA ABI");
  STD_TORCH_CHECK(
      requests <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
          draft_token_ids.numel() <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "Loom greedy speculative batch exceeds the CUDA ABI");
  const auto capacity =
      static_cast<uint64_t>(requests) *
      static_cast<uint64_t>(max_draft_tokens);
  STD_TORCH_CHECK(
      static_cast<uint64_t>(draft_token_ids.numel()) <= capacity,
      "Loom flattened draft token count exceeds the ragged batch capacity");
}

void check_greedy_speculative_verify_output_shape(
    const Tensor& output_token_ids, const Tensor& accepted_lengths,
    const Tensor& emitted_lengths, int64_t requests,
    int64_t max_draft_tokens) {
  STD_TORCH_CHECK(
      output_token_ids.dim() == 2 &&
          output_token_ids.size(0) == requests &&
          output_token_ids.size(1) == max_draft_tokens + 1,
      "Loom speculative output token IDs must have shape "
      "[requests, max_draft_tokens + 1]");
  STD_TORCH_CHECK(
      accepted_lengths.dim() == 1 &&
          accepted_lengths.numel() == requests &&
          emitted_lengths.dim() == 1 &&
          emitted_lengths.numel() == requests,
      "Loom speculative accepted and emitted lengths must have shape "
      "[requests]");
}

void check_greedy_speculative_verify_contract(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    const Tensor& output_token_ids, const Tensor& accepted_lengths,
    const Tensor& emitted_lengths, int64_t max_draft_tokens) {
  check_greedy_speculative_verify_shape(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, max_draft_tokens);
  const int64_t requests = cumulative_draft_lengths.numel();
  check_greedy_speculative_verify_output_shape(
      output_token_ids, accepted_lengths, emitted_lengths, requests,
      max_draft_tokens);
  STD_TORCH_CHECK(draft_token_ids.is_cuda(),
                  "Loom greedy speculative inputs must be CUDA");
  STD_TORCH_CHECK(
      target_token_ids.device() == draft_token_ids.device() &&
          bonus_token_ids.device() == draft_token_ids.device() &&
          cumulative_draft_lengths.device() == draft_token_ids.device() &&
          output_token_ids.device() == draft_token_ids.device() &&
          accepted_lengths.device() == draft_token_ids.device() &&
          emitted_lengths.device() == draft_token_ids.device(),
      "Loom greedy speculative tensors must share one CUDA device");
  STD_TORCH_CHECK(
      draft_token_ids.scalar_type() == ScalarType::Int &&
          bonus_token_ids.scalar_type() == ScalarType::Int &&
          cumulative_draft_lengths.scalar_type() == ScalarType::Int &&
          output_token_ids.scalar_type() == ScalarType::Int &&
          accepted_lengths.scalar_type() == ScalarType::Int &&
          emitted_lengths.scalar_type() == ScalarType::Int,
      "Loom draft, bonus, cumulative, and output tensors must use int32");
  STD_TORCH_CHECK(target_token_ids.scalar_type() == ScalarType::Long,
                  "Loom target token IDs must use int64");
  STD_TORCH_CHECK(
      draft_token_ids.is_contiguous() && target_token_ids.is_contiguous() &&
          bonus_token_ids.is_contiguous() &&
          cumulative_draft_lengths.is_contiguous() &&
          output_token_ids.is_contiguous() &&
          accepted_lengths.is_contiguous() &&
          emitted_lengths.is_contiguous(),
      "Loom greedy speculative tensors must be contiguous");
}

void launch_greedy_speculative_verify(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    Tensor output_token_ids, Tensor accepted_lengths, Tensor emitted_lengths,
    int64_t max_draft_tokens) {
  const int64_t requests = cumulative_draft_lengths.numel();
  const CudaDeviceGuard device_guard(draft_token_ids.device());
  const auto stream =
      current_cuda_stream(draft_token_ids.device().index());
  const int status = loom_cuda_bridge_greedy_speculative_verify(
      draft_token_ids.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(draft_token_ids.numel()),
      target_token_ids.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(target_token_ids.numel()),
      bonus_token_ids.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(bonus_token_ids.numel()),
      cumulative_draft_lengths.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(cumulative_draft_lengths.numel()),
      output_token_ids.mutable_data_ptr<int32_t>(),
      static_cast<uint64_t>(output_token_ids.numel()),
      accepted_lengths.mutable_data_ptr<int32_t>(),
      static_cast<uint64_t>(accepted_lengths.numel()),
      emitted_lengths.mutable_data_ptr<int32_t>(),
      static_cast<uint64_t>(emitted_lengths.numel()),
      static_cast<uint32_t>(requests),
      static_cast<uint32_t>(draft_token_ids.numel()),
      static_cast<uint32_t>(max_draft_tokens), stream.stream());
  check_bridge_status(status, "greedy speculative verification");
}

void greedy_speculative_verify(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    Tensor output_token_ids, Tensor accepted_lengths, Tensor emitted_lengths,
    int64_t max_draft_tokens) {
  check_greedy_speculative_verify_contract(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, output_token_ids, accepted_lengths,
      emitted_lengths, max_draft_tokens);
  launch_greedy_speculative_verify(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, output_token_ids, accepted_lengths,
      emitted_lengths, max_draft_tokens);
}

void greedy_speculative_verify_meta(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    Tensor output_token_ids, Tensor accepted_lengths, Tensor emitted_lengths,
    int64_t max_draft_tokens) {
  check_greedy_speculative_verify_shape(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, max_draft_tokens);
  check_greedy_speculative_verify_output_shape(
      output_token_ids, accepted_lengths, emitted_lengths,
      cumulative_draft_lengths.numel(), max_draft_tokens);
}


}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "greedy_speculative_verify",
      TORCH_BOX(&loom_kernels::torch_adapter::greedy_speculative_verify));
}

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, Meta, library) {
  library.impl(
      "greedy_speculative_verify",
      TORCH_BOX(&loom_kernels::torch_adapter::greedy_speculative_verify_meta));
}
