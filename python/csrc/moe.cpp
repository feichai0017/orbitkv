#include "common.h"

namespace loom_kernels::torch_adapter {

void check_moe_permute_scalar_shape(const Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(
      tensor.dim() == 2 && tensor.size(0) > 0 && tensor.size(1) > 0,
      "Loom ", name, " must be non-empty rank-2");
  STD_TORCH_CHECK(
      tensor.scalar_type() == ScalarType::Float ||
          tensor.scalar_type() == ScalarType::Half ||
          tensor.scalar_type() == ScalarType::BFloat16 ||
          tensor.scalar_type() == ScalarType::Float8_e4m3fn,
      "Loom ", name, " must use F32, FP16, BF16, or FP8 E4M3FN");
}

void check_moe_combine_scalar_shape(const Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(
      tensor.dim() == 2 && tensor.size(0) > 0 && tensor.size(1) > 0,
      "Loom ", name, " must be non-empty rank-2");
  STD_TORCH_CHECK(
      tensor.scalar_type() == ScalarType::Float ||
          tensor.scalar_type() == ScalarType::Half ||
          tensor.scalar_type() == ScalarType::BFloat16,
      "Loom ", name, " must use F32, FP16, or BF16");
}

void check_moe_permute_shape(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_experts, int64_t num_local_experts,
    const std::optional<Tensor>& expert_map) {
  check_moe_permute_scalar_shape(hidden_states, "MoE hidden states");
  STD_TORCH_CHECK(
      topk_ids.dim() == 2 && topk_ids.size(0) == hidden_states.size(0) &&
          topk_ids.size(1) > 0,
      "Loom MoE top-k expert IDs must have shape [tokens, top_k]");
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == ScalarType::Int,
      "Loom MoE top-k expert IDs must use int32");
  STD_TORCH_CHECK(
      num_experts > 0 &&
          num_experts <=
              static_cast<int64_t>(std::numeric_limits<int32_t>::max()) &&
          num_local_experts > 0 && num_local_experts <= num_experts,
      "Loom MoE expert counts must satisfy 0 < local <= global <= int32");
  STD_TORCH_CHECK(
      topk_ids.size(1) <= num_experts,
      "Loom MoE top_k must not exceed the global expert count");
  STD_TORCH_CHECK(
      hidden_states.size(0) <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
          hidden_states.size(1) <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
          topk_ids.size(1) <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "Loom MoE shape exceeds the CUDA ABI");
  STD_TORCH_CHECK(
      hidden_states.size(0) <=
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()) /
              topk_ids.size(1),
      "Loom MoE assignment count exceeds int32");
  if (expert_map.has_value()) {
    STD_TORCH_CHECK(
        expert_map->dim() == 1 && expert_map->numel() == num_experts,
        "Loom MoE expert map must have shape [num_experts]");
    STD_TORCH_CHECK(
        expert_map->scalar_type() == ScalarType::Int,
        "Loom MoE expert map must use int32");
  } else {
    STD_TORCH_CHECK(
        num_local_experts == num_experts,
        "Loom MoE expert-parallel permutation requires an expert map");
  }
}

void check_moe_permute_contract(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_experts, int64_t num_local_experts,
    const std::optional<Tensor>& expert_map) {
  check_moe_permute_shape(
      hidden_states, topk_ids, num_experts, num_local_experts, expert_map);
  STD_TORCH_CHECK(
      hidden_states.is_cuda(), "Loom MoE hidden states must be CUDA");
  STD_TORCH_CHECK(
      topk_ids.device() == hidden_states.device(),
      "Loom MoE hidden states and top-k IDs must share one CUDA device");
  STD_TORCH_CHECK(
      hidden_states.is_contiguous() && topk_ids.is_contiguous(),
      "Loom MoE hidden states and top-k IDs must be contiguous");
  if (expert_map.has_value()) {
    STD_TORCH_CHECK(
        expert_map->device() == hidden_states.device(),
        "Loom MoE expert map must share the hidden-state CUDA device");
    STD_TORCH_CHECK(
        expert_map->is_contiguous(), "Loom MoE expert map must be contiguous");
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> allocate_moe_permute_outputs(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_local_experts) {
  const int64_t assignments = hidden_states.size(0) * topk_ids.size(1);
  return {
      new_empty(
          hidden_states, {assignments, hidden_states.size(1)},
          hidden_states.scalar_type()),
      new_empty(hidden_states, {num_local_experts + 1}, ScalarType::Long),
      new_empty(
          hidden_states, {hidden_states.size(0), topk_ids.size(1)},
          ScalarType::Int),
      new_empty(hidden_states, {assignments}, ScalarType::Int),
  };
}

void check_moe_permute_output_shapes(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_local_experts, const Tensor& permuted_hidden_states,
    const Tensor& expert_offsets, const Tensor& inverse_permutation,
    const Tensor& permuted_assignment_ids) {
  const int64_t assignments = hidden_states.size(0) * topk_ids.size(1);
  STD_TORCH_CHECK(
      permuted_hidden_states.dim() == 2 &&
          permuted_hidden_states.size(0) == assignments &&
          permuted_hidden_states.size(1) == hidden_states.size(1) &&
          permuted_hidden_states.scalar_type() == hidden_states.scalar_type(),
      "Loom MoE permuted hidden-state output has the wrong shape or dtype");
  STD_TORCH_CHECK(
      expert_offsets.dim() == 1 &&
          expert_offsets.numel() == num_local_experts + 1 &&
          expert_offsets.scalar_type() == ScalarType::Long,
      "Loom MoE expert-offset output must be int64 [local_experts + 1]");
  STD_TORCH_CHECK(
      inverse_permutation.dim() == 2 &&
          inverse_permutation.size(0) == hidden_states.size(0) &&
          inverse_permutation.size(1) == topk_ids.size(1) &&
          inverse_permutation.scalar_type() == ScalarType::Int,
      "Loom MoE inverse-permutation output must be int32 [tokens, top_k]");
  STD_TORCH_CHECK(
      permuted_assignment_ids.dim() == 1 &&
          permuted_assignment_ids.numel() == assignments &&
          permuted_assignment_ids.scalar_type() == ScalarType::Int,
      "Loom MoE assignment-ID output must be int32 [tokens * top_k]");
}

void launch_moe_permute(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_experts, int64_t num_local_experts,
    const std::optional<Tensor>& expert_map, Tensor permuted_hidden_states,
    Tensor expert_offsets, Tensor inverse_permutation,
    Tensor permuted_assignment_ids) {
  check_moe_permute_contract(
      hidden_states, topk_ids, num_experts, num_local_experts, expert_map);
  check_moe_permute_output_shapes(
      hidden_states, topk_ids, num_local_experts, permuted_hidden_states,
      expert_offsets, inverse_permutation, permuted_assignment_ids);
  STD_TORCH_CHECK(
      permuted_hidden_states.device() == hidden_states.device() &&
          expert_offsets.device() == hidden_states.device() &&
          inverse_permutation.device() == hidden_states.device() &&
          permuted_assignment_ids.device() == hidden_states.device(),
      "Loom MoE permutation outputs must share the input CUDA device");
  STD_TORCH_CHECK(
      permuted_hidden_states.is_contiguous() && expert_offsets.is_contiguous() &&
          inverse_permutation.is_contiguous() &&
          permuted_assignment_ids.is_contiguous(),
      "Loom MoE permutation outputs must be contiguous");

  const auto tokens = static_cast<uint32_t>(hidden_states.size(0));
  const auto hidden_size = static_cast<uint32_t>(hidden_states.size(1));
  const auto top_k = static_cast<uint32_t>(topk_ids.size(1));
  const auto global_experts = static_cast<uint32_t>(num_experts);
  const auto local_experts = static_cast<uint32_t>(num_local_experts);
  uint64_t workspace_bytes = 0;
  int status = loom_cuda_bridge_moe_permute_workspace_size(
      bridge_dtype(hidden_states), tokens, hidden_size, top_k, global_experts,
      local_experts, expert_map.has_value() ? 1U : 0U, &workspace_bytes);
  check_bridge_status(status, "MoE permutation workspace query");
  STD_TORCH_CHECK(
      workspace_bytes <=
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "Loom MoE workspace exceeds the PyTorch shape ABI");
  Tensor workspace = new_empty(
      hidden_states, {static_cast<int64_t>(workspace_bytes)},
      ScalarType::Byte);

  const CudaDeviceGuard device_guard(hidden_states.device());
  const auto stream = current_cuda_stream(hidden_states.device().index());
  status = loom_cuda_bridge_moe_permute(
      bridge_dtype(hidden_states), hidden_states.const_data_ptr(),
      static_cast<uint64_t>(hidden_states.numel()),
      topk_ids.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(topk_ids.numel()),
      expert_map.has_value() ? expert_map->const_data_ptr<int32_t>() : nullptr,
      expert_map.has_value()
          ? static_cast<uint64_t>(expert_map->numel())
          : 0U,
      permuted_hidden_states.mutable_data_ptr(),
      static_cast<uint64_t>(permuted_hidden_states.numel()),
      expert_offsets.mutable_data_ptr<int64_t>(),
      static_cast<uint64_t>(expert_offsets.numel()),
      inverse_permutation.mutable_data_ptr<int32_t>(),
      static_cast<uint64_t>(inverse_permutation.numel()),
      permuted_assignment_ids.mutable_data_ptr<int32_t>(),
      static_cast<uint64_t>(permuted_assignment_ids.numel()),
      workspace.mutable_data_ptr<uint8_t>(), workspace_bytes, tokens,
      hidden_size, top_k, global_experts, local_experts, stream.stream());
  check_bridge_status(status, "MoE permutation");
}

std::tuple<Tensor, Tensor, Tensor, Tensor> moe_permute(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_experts, int64_t num_local_experts,
    const std::optional<Tensor>& expert_map) {
  check_moe_permute_shape(
      hidden_states, topk_ids, num_experts, num_local_experts, expert_map);
  auto [permuted_hidden_states, expert_offsets, inverse_permutation,
        permuted_assignment_ids] =
      allocate_moe_permute_outputs(
          hidden_states, topk_ids, num_local_experts);
  launch_moe_permute(
      hidden_states, topk_ids, num_experts, num_local_experts, expert_map,
      permuted_hidden_states, expert_offsets, inverse_permutation,
      permuted_assignment_ids);
  return {
      permuted_hidden_states, expert_offsets, inverse_permutation,
      permuted_assignment_ids};
}

void moe_permute_out(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_experts, int64_t num_local_experts,
    const std::optional<Tensor>& expert_map, Tensor permuted_hidden_states,
    Tensor expert_offsets, Tensor inverse_permutation,
    Tensor permuted_assignment_ids) {
  launch_moe_permute(
      hidden_states, topk_ids, num_experts, num_local_experts, expert_map,
      permuted_hidden_states, expert_offsets, inverse_permutation,
      permuted_assignment_ids);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> moe_permute_meta(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_experts, int64_t num_local_experts,
    const std::optional<Tensor>& expert_map) {
  check_moe_permute_shape(
      hidden_states, topk_ids, num_experts, num_local_experts, expert_map);
  return allocate_moe_permute_outputs(
      hidden_states, topk_ids, num_local_experts);
}

void moe_permute_out_meta(
    const Tensor& hidden_states, const Tensor& topk_ids,
    int64_t num_experts, int64_t num_local_experts,
    const std::optional<Tensor>& expert_map, const Tensor& permuted_hidden_states,
    const Tensor& expert_offsets, const Tensor& inverse_permutation,
    const Tensor& permuted_assignment_ids) {
  check_moe_permute_shape(
      hidden_states, topk_ids, num_experts, num_local_experts, expert_map);
  check_moe_permute_output_shapes(
      hidden_states, topk_ids, num_local_experts, permuted_hidden_states,
      expert_offsets, inverse_permutation, permuted_assignment_ids);
}

void check_moe_combine_shape(
    const Tensor& expert_outputs, const Tensor& routing_weights,
    const Tensor& inverse_permutation, const Tensor& expert_offsets) {
  check_moe_combine_scalar_shape(expert_outputs, "MoE expert outputs");
  STD_TORCH_CHECK(
      routing_weights.dim() == 2 && routing_weights.size(0) > 0 &&
          routing_weights.size(1) > 0,
      "Loom MoE routing weights must be non-empty rank-2");
  STD_TORCH_CHECK(
      routing_weights.scalar_type() == ScalarType::Float,
      "Loom MoE routing weights must use F32");
  STD_TORCH_CHECK(
      inverse_permutation.dim() == 2 &&
          inverse_permutation.size(0) == routing_weights.size(0) &&
          inverse_permutation.size(1) == routing_weights.size(1) &&
          inverse_permutation.scalar_type() == ScalarType::Int,
      "Loom MoE inverse permutation must be int32 with the routing shape");
  STD_TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.numel() >= 2 &&
          expert_offsets.scalar_type() == ScalarType::Long,
      "Loom MoE expert offsets must be an int64 vector with local + 1 values");
  STD_TORCH_CHECK(
      expert_outputs.size(0) == routing_weights.numel(),
      "Loom MoE expert-output rows must equal tokens * top_k");
  STD_TORCH_CHECK(
      routing_weights.size(0) <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
          routing_weights.size(1) <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
          expert_outputs.size(1) <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
          expert_offsets.numel() - 1 <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "Loom MoE combine shape exceeds the CUDA ABI");
  STD_TORCH_CHECK(
      routing_weights.size(0) <=
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()) /
              routing_weights.size(1),
      "Loom MoE combine assignment count exceeds int32");
}

void check_moe_combine_contract(
    const Tensor& expert_outputs, const Tensor& routing_weights,
    const Tensor& inverse_permutation, const Tensor& expert_offsets) {
  check_moe_combine_shape(
      expert_outputs, routing_weights, inverse_permutation, expert_offsets);
  STD_TORCH_CHECK(
      expert_outputs.is_cuda(), "Loom MoE expert outputs must be CUDA");
  STD_TORCH_CHECK(
      routing_weights.device() == expert_outputs.device() &&
          inverse_permutation.device() == expert_outputs.device() &&
          expert_offsets.device() == expert_outputs.device(),
      "Loom MoE combine tensors must share one CUDA device");
  STD_TORCH_CHECK(
      expert_outputs.is_contiguous() && routing_weights.is_contiguous() &&
          inverse_permutation.is_contiguous() && expert_offsets.is_contiguous(),
      "Loom MoE combine tensors must be contiguous");
}

Tensor allocate_moe_combine_output(
    const Tensor& expert_outputs, const Tensor& routing_weights) {
  return new_empty(
      expert_outputs,
      {routing_weights.size(0), expert_outputs.size(1)},
      expert_outputs.scalar_type());
}

void check_moe_combine_output_shape(
    const Tensor& expert_outputs, const Tensor& routing_weights,
    const Tensor& output) {
  STD_TORCH_CHECK(
      output.dim() == 2 && output.size(0) == routing_weights.size(0) &&
          output.size(1) == expert_outputs.size(1) &&
          output.scalar_type() == expert_outputs.scalar_type(),
      "Loom MoE combine output has the wrong shape or dtype");
}

void launch_moe_combine(
    const Tensor& expert_outputs, const Tensor& routing_weights,
    const Tensor& inverse_permutation, const Tensor& expert_offsets,
    Tensor output) {
  check_moe_combine_contract(
      expert_outputs, routing_weights, inverse_permutation, expert_offsets);
  check_moe_combine_output_shape(expert_outputs, routing_weights, output);
  STD_TORCH_CHECK(
      output.device() == expert_outputs.device(),
      "Loom MoE combine output must share the input CUDA device");
  STD_TORCH_CHECK(
      output.is_contiguous(), "Loom MoE combine output must be contiguous");
  const CudaDeviceGuard device_guard(expert_outputs.device());
  const auto stream = current_cuda_stream(expert_outputs.device().index());
  const int status = loom_cuda_bridge_moe_combine(
      bridge_dtype(expert_outputs), expert_outputs.const_data_ptr(),
      static_cast<uint64_t>(expert_outputs.numel()),
      routing_weights.const_data_ptr<float>(),
      static_cast<uint64_t>(routing_weights.numel()),
      inverse_permutation.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(inverse_permutation.numel()),
      expert_offsets.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(expert_offsets.numel()), output.mutable_data_ptr(),
      static_cast<uint64_t>(output.numel()),
      static_cast<uint32_t>(routing_weights.size(0)),
      static_cast<uint32_t>(expert_outputs.size(1)),
      static_cast<uint32_t>(routing_weights.size(1)),
      static_cast<uint32_t>(expert_offsets.numel() - 1), stream.stream());
  check_bridge_status(status, "MoE combine");
}

Tensor moe_combine(
    const Tensor& expert_outputs, const Tensor& routing_weights,
    const Tensor& inverse_permutation, const Tensor& expert_offsets) {
  check_moe_combine_shape(
      expert_outputs, routing_weights, inverse_permutation, expert_offsets);
  Tensor output = allocate_moe_combine_output(expert_outputs, routing_weights);
  launch_moe_combine(
      expert_outputs, routing_weights, inverse_permutation, expert_offsets,
      output);
  return output;
}

void moe_combine_out(
    const Tensor& expert_outputs, const Tensor& routing_weights,
    const Tensor& inverse_permutation, const Tensor& expert_offsets,
    Tensor output) {
  launch_moe_combine(
      expert_outputs, routing_weights, inverse_permutation, expert_offsets,
      output);
}

Tensor moe_combine_meta(
    const Tensor& expert_outputs, const Tensor& routing_weights,
    const Tensor& inverse_permutation, const Tensor& expert_offsets) {
  check_moe_combine_shape(
      expert_outputs, routing_weights, inverse_permutation, expert_offsets);
  return allocate_moe_combine_output(expert_outputs, routing_weights);
}

void moe_combine_out_meta(
    const Tensor& expert_outputs, const Tensor& routing_weights,
    const Tensor& inverse_permutation, const Tensor& expert_offsets,
    const Tensor& output) {
  check_moe_combine_shape(
      expert_outputs, routing_weights, inverse_permutation, expert_offsets);
  check_moe_combine_output_shape(expert_outputs, routing_weights, output);
}

}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "moe_permute",
      TORCH_BOX(&loom_kernels::torch_adapter::moe_permute));
  library.impl(
      "moe_permute.out",
      TORCH_BOX(&loom_kernels::torch_adapter::moe_permute_out));
  library.impl(
      "moe_combine",
      TORCH_BOX(&loom_kernels::torch_adapter::moe_combine));
  library.impl(
      "moe_combine.out",
      TORCH_BOX(&loom_kernels::torch_adapter::moe_combine_out));
}

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, Meta, library) {
  library.impl(
      "moe_permute",
      TORCH_BOX(&loom_kernels::torch_adapter::moe_permute_meta));
  library.impl(
      "moe_permute.out",
      TORCH_BOX(&loom_kernels::torch_adapter::moe_permute_out_meta));
  library.impl(
      "moe_combine",
      TORCH_BOX(&loom_kernels::torch_adapter::moe_combine_meta));
  library.impl(
      "moe_combine.out",
      TORCH_BOX(&loom_kernels::torch_adapter::moe_combine_out_meta));
}
