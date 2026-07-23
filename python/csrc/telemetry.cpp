#include "common.h"

namespace loom_kernels::torch_adapter {

int64_t bridge_abi_version() {
  return static_cast<int64_t>(loom_cuda_bridge_abi_version());
}

int64_t bridge_launch_count(int64_t operation) {
  STD_TORCH_CHECK(operation >= 0 &&
                  operation <= LOOM_CUDA_BRIDGE_GREEDY_SPECULATIVE_VERIFY,
              "Loom bridge operator id is out of range");
  uint64_t count = 0;
  const int status = loom_cuda_bridge_launch_count(
      static_cast<uint32_t>(operation), &count);
  check_bridge_status(status, "telemetry query");
  STD_TORCH_CHECK(
      count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "Loom bridge launch count exceeds int64");
  return static_cast<int64_t>(count);
}

void reset_bridge_launch_count(int64_t operation) {
  STD_TORCH_CHECK(operation >= 0 &&
                  operation <= LOOM_CUDA_BRIDGE_GREEDY_SPECULATIVE_VERIFY,
              "Loom bridge operator id is out of range");
  const int status =
      loom_cuda_bridge_reset_launch_count(static_cast<uint32_t>(operation));
  check_bridge_status(status, "telemetry reset");
}

}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(
    loom_kernels, CompositeExplicitAutograd, library) {
  library.impl(
      "bridge_abi_version",
      TORCH_BOX(&loom_kernels::torch_adapter::bridge_abi_version));
  library.impl(
      "bridge_launch_count",
      TORCH_BOX(&loom_kernels::torch_adapter::bridge_launch_count));
  library.impl(
      "reset_bridge_launch_count",
      TORCH_BOX(&loom_kernels::torch_adapter::reset_bridge_launch_count));
}
