#include "orbitkv.h"

#include <cstdint>
#include <type_traits>

static_assert(ORBITKV_ABI_VERSION == 6u, "breaking ABI version");
static_assert(ORBITKV_STATUS_RETRYABLE_CONFLICT == 2,
              "retryable conflict status");
static_assert(ORBITKV_STATUS_FAIL_STOPPED == -4, "fail-stopped status");
static_assert(std::is_standard_layout_v<OrbitKvRequestView>);
static_assert(std::is_standard_layout_v<OrbitKvPreparedBatchItem>);
static_assert(std::is_standard_layout_v<OrbitKvDetachedBinding>);
static_assert(std::is_standard_layout_v<OrbitKvPrefixLookupHint>);

int main() {
  (void)&orbitkv_manager_create;
  (void)&orbitkv_manager_arena_identities;
  (void)&orbitkv_manager_arena_stats;
  (void)&orbitkv_manager_request_acquire_batch;
  (void)&orbitkv_manager_request_fork_batch;
  (void)&orbitkv_manager_prepare_batch;
  (void)&orbitkv_manager_submit_batch;
  (void)&orbitkv_manager_complete_batch;
  (void)&orbitkv_manager_abort_steps_batch;
  (void)&orbitkv_manager_quarantine_steps_batch;
  (void)&orbitkv_manager_quarantine_submissions_batch;
  (void)&orbitkv_manager_release_batch;
  (void)&orbitkv_manager_acknowledge_reclamations_batch;
  (void)&orbitkv_manager_recycle_requests_batch;
  (void)&orbitkv_manager_prefix_lookup_batch;
  (void)&orbitkv_manager_prefix_attach_batch;
  (void)&orbitkv_manager_prefix_publish_batch;
  (void)&orbitkv_manager_prefix_publish_release_batch;
  (void)&orbitkv_manager_prefix_evict_batch;
  (void)&orbitkv_manager_prefix_recycle_batch;
  (void)&orbitkv_manager_stats;
  if (orbitkv_abi_version() != ORBITKV_ABI_VERSION) {
    return 1;
  }
  if (orbitkv_manager_destroy(nullptr, nullptr, 0) != ORBITKV_STATUS_OK) {
    return 2;
  }
  if (orbitkv_manager_create(nullptr, 0, nullptr, nullptr, 0, nullptr,
                             nullptr, 0) !=
      ORBITKV_STATUS_INVALID_ARGUMENT) {
    return 3;
  }
  return 0;
}
