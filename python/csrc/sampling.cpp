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

void check_topk_sampled_logprobs_shape(
    const Tensor& logits, const Tensor& sampled_token_ids, int64_t top_k) {
  check_selected_token_logprobs_shape(logits, sampled_token_ids);
  const int64_t maximum = std::min<int64_t>(logits.size(1), 32);
  STD_TORCH_CHECK(
      top_k > 0 && top_k <= maximum,
      "Loom top-k sampled logprobs require 1 <= top_k <= ", maximum,
      "; got ", top_k);
}

void check_topk_sampled_logprobs_contract(
    const Tensor& logits, const Tensor& sampled_token_ids, int64_t top_k) {
  check_selected_token_logprobs_contract(logits, sampled_token_ids);
  check_topk_sampled_logprobs_shape(logits, sampled_token_ids, top_k);
}

std::tuple<Tensor, Tensor, Tensor> launch_topk_sampled_logprobs(
    const Tensor& logits, const Tensor& sampled_token_ids, int64_t top_k) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  const int64_t output_width = top_k + 1;
  const uint64_t output_elements =
      static_cast<uint64_t>(logits.size(0)) *
      static_cast<uint64_t>(output_width);
  uint64_t workspace_bytes = 0;
  const int workspace_status =
      loom_cuda_bridge_topk_sampled_logprobs_workspace_size(
          rows, vocab_size, static_cast<uint32_t>(top_k), &workspace_bytes);
  check_bridge_status(workspace_status, "top-k workspace query");
  STD_TORCH_CHECK(
      workspace_bytes <=
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "Loom top-k workspace exceeds the PyTorch shape ABI");
  Tensor output_token_ids =
      new_empty(logits, {logits.size(0), output_width}, ScalarType::Int);
  Tensor output_logprobs =
      new_empty(logits, {logits.size(0), output_width}, ScalarType::Float);
  Tensor sampled_token_ranks =
      new_empty(logits, {logits.size(0)}, ScalarType::Long);
  Tensor workspace = new_empty(
      logits, {static_cast<int64_t>(workspace_bytes)}, ScalarType::Byte);

  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const int status = loom_cuda_bridge_topk_sampled_logprobs(
      bridge_dtype(logits), logits.const_data_ptr(),
      storage_span_elements(logits),
      sampled_token_ids.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(sampled_token_ids.numel()),
      output_token_ids.mutable_data_ptr<int32_t>(), output_elements,
      output_logprobs.mutable_data_ptr<float>(), output_elements,
      sampled_token_ranks.mutable_data_ptr<int64_t>(),
      static_cast<uint64_t>(sampled_token_ranks.numel()),
      workspace.mutable_data_ptr<uint8_t>(),
      static_cast<uint64_t>(workspace.numel()), rows, vocab_size,
      static_cast<uint32_t>(top_k), row_stride, stream.stream());
  check_bridge_status(status, "top-k sampled logprobs");
  return {output_token_ids, output_logprobs, sampled_token_ranks};
}

std::tuple<Tensor, Tensor, Tensor> topk_sampled_logprobs(
    const Tensor& logits, const Tensor& sampled_token_ids, int64_t top_k) {
  check_topk_sampled_logprobs_contract(logits, sampled_token_ids, top_k);
  return launch_topk_sampled_logprobs(logits, sampled_token_ids, top_k);
}

std::tuple<Tensor, Tensor, Tensor> topk_sampled_logprobs_meta(
    const Tensor& logits, const Tensor& sampled_token_ids, int64_t top_k) {
  check_topk_sampled_logprobs_shape(logits, sampled_token_ids, top_k);
  const int64_t output_width = top_k + 1;
  return {
      new_empty(logits, {logits.size(0), output_width}, ScalarType::Int),
      new_empty(logits, {logits.size(0), output_width}, ScalarType::Float),
      new_empty(logits, {logits.size(0)}, ScalarType::Long),
  };
}

uint64_t required_token_penalty_workspace(int64_t prompt_tokens,
                                          int64_t output_tokens) {
  STD_TORCH_CHECK(prompt_tokens > 0 && output_tokens > 0,
              "Loom token-penalty history widths must be positive");
  const uint64_t prompt = static_cast<uint64_t>(prompt_tokens);
  const uint64_t output = static_cast<uint64_t>(output_tokens);
  STD_TORCH_CHECK(
      prompt <= (std::numeric_limits<uint64_t>::max() - output) &&
          prompt + output <=
              std::numeric_limits<uint32_t>::max() / 2ULL,
      "Loom token-penalty workspace exceeds the CUDA ABI");
  uint64_t required = 2 * (prompt + output);
  --required;
  required |= required >> 1;
  required |= required >> 2;
  required |= required >> 4;
  required |= required >> 8;
  required |= required >> 16;
  required |= required >> 32;
  return required + 1;
}

void check_token_penalties_shape(
    const Tensor& logits, const Tensor& prompt_token_ids,
    const Tensor& output_token_ids, const Tensor& presence_penalties,
    const Tensor& frequency_penalties, const Tensor& repetition_penalties,
    const Tensor& workspace) {
  STD_TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0,
              "Loom token-penalty logits must be non-empty rank-2");
  const int64_t rows = logits.size(0);
  STD_TORCH_CHECK(prompt_token_ids.dim() == 2 &&
                  prompt_token_ids.size(0) == rows &&
                  prompt_token_ids.size(1) > 0,
              "Loom prompt token IDs must be [rows, positive width]");
  STD_TORCH_CHECK(output_token_ids.dim() == 2 &&
                  output_token_ids.size(0) == rows &&
                  output_token_ids.size(1) > 0,
              "Loom output token IDs must be [rows, positive width]");
  for (const auto* penalty :
       {&presence_penalties, &frequency_penalties, &repetition_penalties}) {
    STD_TORCH_CHECK(penalty->dim() == 1 && penalty->size(0) == rows,
                "Loom token penalties must contain one value per logits row");
  }
  STD_TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == rows &&
                  workspace.size(1) > 0,
              "Loom token-penalty workspace must be [rows, positive capacity]");
  const uint64_t capacity = static_cast<uint64_t>(workspace.size(1));
  const uint64_t required = required_token_penalty_workspace(
      prompt_token_ids.size(1), output_token_ids.size(1));
  STD_TORCH_CHECK((capacity & (capacity - 1)) == 0 && capacity >= required,
              "Loom token-penalty workspace capacity must be a power of two "
              "and at least twice the combined history width; required ",
              required, ", got ", capacity);
  STD_TORCH_CHECK(
      rows <= std::numeric_limits<int32_t>::max() &&
          logits.size(1) <= std::numeric_limits<int32_t>::max() &&
          prompt_token_ids.size(1) <= std::numeric_limits<uint32_t>::max() &&
          output_token_ids.size(1) <= std::numeric_limits<int32_t>::max() &&
          workspace.size(1) <= std::numeric_limits<uint32_t>::max(),
      "Loom token-penalty shape exceeds the CUDA ABI");
}

void check_token_penalties_contract(
    const Tensor& logits, const Tensor& prompt_token_ids,
    const Tensor& output_token_ids, const Tensor& presence_penalties,
    const Tensor& frequency_penalties, const Tensor& repetition_penalties,
    const Tensor& workspace) {
  check_token_penalties_shape(
      logits, prompt_token_ids, output_token_ids, presence_penalties,
      frequency_penalties, repetition_penalties, workspace);
  STD_TORCH_CHECK(logits.is_cuda() && prompt_token_ids.is_cuda() &&
                  output_token_ids.is_cuda() && presence_penalties.is_cuda() &&
                  frequency_penalties.is_cuda() &&
                  repetition_penalties.is_cuda() && workspace.is_cuda(),
              "Loom token penalties require CUDA tensors");
  const auto device = logits.device();
  STD_TORCH_CHECK(prompt_token_ids.device() == device &&
                  output_token_ids.device() == device &&
                  presence_penalties.device() == device &&
                  frequency_penalties.device() == device &&
                  repetition_penalties.device() == device &&
                  workspace.device() == device,
              "Loom token-penalty tensors must share one CUDA device");
  STD_TORCH_CHECK(logits.scalar_type() == ScalarType::Float,
              "Loom token-penalty logits must be float32");
  STD_TORCH_CHECK(prompt_token_ids.scalar_type() == ScalarType::Long &&
                  output_token_ids.scalar_type() == ScalarType::Long,
              "Loom token IDs must be int64");
  STD_TORCH_CHECK(presence_penalties.scalar_type() == ScalarType::Float &&
                  frequency_penalties.scalar_type() == ScalarType::Float &&
                  repetition_penalties.scalar_type() == ScalarType::Float,
              "Loom token penalties must be float32");
  STD_TORCH_CHECK(workspace.scalar_type() == ScalarType::Long,
              "Loom token-penalty workspace must be int64");
  STD_TORCH_CHECK(logits.stride(1) == 1 &&
                  logits.stride(0) >= logits.size(1) &&
                  prompt_token_ids.stride(1) == 1 &&
                  prompt_token_ids.stride(0) >= prompt_token_ids.size(1) &&
                  output_token_ids.stride(1) == 1 &&
                  output_token_ids.stride(0) >= output_token_ids.size(1) &&
                  workspace.stride(1) == 1 &&
                  workspace.stride(0) >= workspace.size(1),
              "Loom token-penalty matrices require unit inner stride and "
              "non-overlapping positive row strides");
  STD_TORCH_CHECK(presence_penalties.is_contiguous() &&
                  frequency_penalties.is_contiguous() &&
                  repetition_penalties.is_contiguous(),
              "Loom token-penalty vectors must be contiguous");
}

void apply_token_penalties_(
    Tensor logits, const Tensor& prompt_token_ids,
    const Tensor& output_token_ids, const Tensor& presence_penalties,
    const Tensor& frequency_penalties, const Tensor& repetition_penalties,
    Tensor workspace) {
  check_token_penalties_contract(
      logits, prompt_token_ids, output_token_ids, presence_penalties,
      frequency_penalties, repetition_penalties, workspace);
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto prompt_tokens = static_cast<uint32_t>(prompt_token_ids.size(1));
  const auto output_tokens = static_cast<uint32_t>(output_token_ids.size(1));
  const auto workspace_capacity = static_cast<uint32_t>(workspace.size(1));
  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const int status = loom_cuda_bridge_apply_token_penalties(
      logits.mutable_data_ptr<float>(), storage_span_elements(logits),
      prompt_token_ids.const_data_ptr<int64_t>(),
      storage_span_elements(prompt_token_ids),
      output_token_ids.const_data_ptr<int64_t>(),
      storage_span_elements(output_token_ids),
      presence_penalties.const_data_ptr<float>(),
      static_cast<uint64_t>(presence_penalties.numel()),
      frequency_penalties.const_data_ptr<float>(),
      static_cast<uint64_t>(frequency_penalties.numel()),
      repetition_penalties.const_data_ptr<float>(),
      static_cast<uint64_t>(repetition_penalties.numel()),
      reinterpret_cast<uint64_t*>(workspace.mutable_data_ptr<int64_t>()),
      storage_span_elements(workspace), rows, vocab_size, prompt_tokens,
      output_tokens, workspace_capacity,
      static_cast<uint64_t>(logits.stride(0)),
      static_cast<uint64_t>(prompt_token_ids.stride(0)),
      static_cast<uint64_t>(output_token_ids.stride(0)),
      static_cast<uint64_t>(workspace.stride(0)), stream.stream());
  check_bridge_status(status, "token penalties");
}

void apply_token_penalties_meta(
    Tensor logits, const Tensor& prompt_token_ids,
    const Tensor& output_token_ids, const Tensor& presence_penalties,
    const Tensor& frequency_penalties, const Tensor& repetition_penalties,
    Tensor workspace) {
  check_token_penalties_shape(
      logits, prompt_token_ids, output_token_ids, presence_penalties,
      frequency_penalties, repetition_penalties, workspace);
}


}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "greedy_sample_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::greedy_sample_logprobs));
  library.impl(
      "selected_token_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::selected_token_logprobs));
  library.impl(
      "topk_sampled_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::topk_sampled_logprobs));
  library.impl(
      "apply_token_penalties_",
      TORCH_BOX(&loom_kernels::torch_adapter::apply_token_penalties_));
}

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, Meta, library) {
  library.impl(
      "greedy_sample_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::greedy_sample_logprobs_meta));
  library.impl(
      "selected_token_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::selected_token_logprobs_meta));
  library.impl(
      "topk_sampled_logprobs",
      TORCH_BOX(&loom_kernels::torch_adapter::topk_sampled_logprobs_meta));
  library.impl(
      "apply_token_penalties_",
      TORCH_BOX(&loom_kernels::torch_adapter::apply_token_penalties_meta));
}
