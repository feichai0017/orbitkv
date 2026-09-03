#include "orbitkv.h"

#include <cstdint>
#include <type_traits>

static_assert(ORBITKV_WIRE_VERSION == 14u, "wire version");
static_assert(ORBITKV_STATUS_RETRYABLE_CONFLICT == 2,
              "retryable conflict status");
static_assert(ORBITKV_STATUS_FAIL_STOPPED == -4, "fail-stopped status");
static_assert(ORBITKV_SESSION_RELEASE_COMPLETED == 1u);
static_assert(ORBITKV_SESSION_RELEASE_RECYCLE_PENDING == 2u);
static_assert(ORBITKV_SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING == 1u);
static_assert(ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED == 2u);
static_assert(ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION == 1u);
static_assert(ORBITKV_SESSION_CONTROL_KIND_PREFIX_EVICTION == 2u);
static_assert(ORBITKV_SESSION_CONTROL_OUTCOME_MATERIALIZED == 1u);
static_assert(ORBITKV_SESSION_CONTROL_OUTCOME_EVICTED == 2u);
static_assert(ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE == 1u);
static_assert(ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX == 2u);
static_assert(sizeof(OrbitKvManagerConfig) == 32);
static_assert(alignof(OrbitKvManagerConfig) == 8);
static_assert(offsetof(OrbitKvManagerConfig, plan_format) == 20);
static_assert(offsetof(OrbitKvManagerConfig, reserved) == 24);
static_assert(std::is_standard_layout_v<OrbitKvSessionCreateConfig>);
static_assert(sizeof(OrbitKvSessionCreateConfig) == 40);
static_assert(alignof(OrbitKvSessionCreateConfig) == 8);
static_assert(offsetof(OrbitKvSessionCreateConfig, manager) == 0);
static_assert(offsetof(OrbitKvSessionCreateConfig, cache_sharing_policy) == 32);
static_assert(offsetof(OrbitKvSessionCreateConfig, reserved) == 36);

#define ORBITKV_SMOKE_SESSION_LAYOUT(type, size)                             \
  static_assert(std::is_standard_layout_v<type>);                            \
  static_assert(sizeof(type) == (size));                                     \
  static_assert(alignof(type) == 8)
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionBatchId, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPublicationId, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionReleaseId, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPrefixId, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionControlId, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRelocationId, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPrefixLookup, 104);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPrefixPublishItem, 80);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPublishedPrefix, 96);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPublishedPrefixRelease, 112);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPrefixAttachItem, 104);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRequestForkItem, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionControlPlanInfo, 40);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionMaterializedRequest, 40);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPendingAttachCancel, 64);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPendingAttachCancelOutcome, 64);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionControlEvidence, 24);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionControlOutcome, 24);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRequestView, 32);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionTokenViewQuery, 24);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionTokenView, 40);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionTokenDispositionBatchItem, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPrepareRelocationItem, 32);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRelocationPlan, 64);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRelocationRequestEvidence, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRelocationCopyEvidence, 112);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRelocationAbortEvidence, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRelocationRequestPublication, 32);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRelocationPublicationEvidence, 24);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionAppendIntent, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPreparedStep, 72);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionStepExecutionEvidence, 32);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionBindEvidence, 48);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionCopyEvidence, 104);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionStepAbortEvidence, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionCompletionEvidence, 24);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionStepPublication, 40);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRetirement, 88);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionRetirementEvidence, 48);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionPublicationEvidence, 24);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionReleasedRequest, 16);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionReleaseEvidence, 24);
ORBITKV_SMOKE_SESSION_LAYOUT(OrbitKvSessionReleaseOutcome, 24);
#undef ORBITKV_SMOKE_SESSION_LAYOUT
static_assert(offsetof(OrbitKvSessionBatchId, sequence) == 8);
static_assert(offsetof(OrbitKvSessionPrefixLookup, candidate) == 72);
static_assert(offsetof(OrbitKvSessionPrefixLookup, reserved1) == 100);
static_assert(
    offsetof(OrbitKvSessionPublishedPrefixRelease, detached_offset) == 100);
static_assert(offsetof(OrbitKvSessionControlPlanInfo, request_count) == 24);
static_assert(offsetof(OrbitKvSessionMaterializedRequest, page_offset) == 28);
static_assert(offsetof(OrbitKvSessionPendingAttachCancel, resident_count) == 56);
static_assert(
    offsetof(OrbitKvSessionPendingAttachCancelOutcome, disposition) == 60);
static_assert(offsetof(OrbitKvSessionPreparedStep, class_offset) == 40);
static_assert(offsetof(OrbitKvSessionRetirement, completion_value) == 80);
static_assert(offsetof(OrbitKvSessionReleaseOutcome, disposition) == 16);
static_assert(offsetof(OrbitKvSessionTokenViewQuery, expected_boundary) == 8);
static_assert(offsetof(OrbitKvSessionRelocationPlan, move_offset) == 40);
static_assert(offsetof(OrbitKvSessionRelocationCopyEvidence, observed) == 104);
static_assert(std::is_standard_layout_v<OrbitKvClassLowering>);
static_assert(sizeof(OrbitKvClassLowering) == 48);
static_assert(alignof(OrbitKvClassLowering) == 8);
constexpr std::uint32_t completion_bound(std::uint32_t step,
                                         std::uint32_t page,
                                         std::uint32_t epoch_blocks) {
  const auto delta = (step + page - 1) / page + 2;
  return delta > epoch_blocks ? delta : epoch_blocks;
}
static_assert(completion_bound(16, 16, 8) == 8,
              "resettable completion bound must cover a complete epoch");
static_assert(std::is_standard_layout_v<OrbitKvDetachedBinding>);
static_assert(std::is_standard_layout_v<OrbitKvTokenPlacement>);
static_assert(std::is_standard_layout_v<OrbitKvStateSlotLease>);
static_assert(std::is_standard_layout_v<OrbitKvStatePublication>);

int main() {
  int32_t (*session_create)(
      const std::uint8_t *, std::size_t, const OrbitKvSessionCreateConfig *,
      const OrbitKvBackendArenaRegistration *, std::uint32_t,
      OrbitKvSessionHandle **, char *, std::size_t) = orbitkv_session_create;
  int32_t (*session_prefix_lookup)(
      OrbitKvSessionHandle *, const OrbitKvPrefixSemanticKey *, std::uint32_t,
      OrbitKvSessionPrefixLookup *, std::uint32_t, std::uint32_t *, char *,
      std::size_t) = orbitkv_session_prefix_lookup_batch;
  int32_t (*session_prefix_publish)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrefixPublishItem *,
      std::uint32_t, OrbitKvSessionPublishedPrefix *, std::uint32_t,
      std::uint32_t *, char *, std::size_t) =
      orbitkv_session_prefix_publish_batch;
  int32_t (*session_prefix_publish_release)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrefixPublishItem *,
      std::uint32_t, OrbitKvSessionReleaseId *,
      OrbitKvSessionPublishedPrefixRelease *, std::uint32_t, std::uint32_t *,
      OrbitKvDetachedBinding *, std::uint32_t, std::uint32_t *, char *,
      std::size_t) = orbitkv_session_prefix_publish_release_batch;
  int32_t (*session_prepare_prefix_attach)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrefixAttachItem *,
      std::uint32_t, OrbitKvSessionControlId *, char *, std::size_t) =
      orbitkv_session_prepare_prefix_attach;
  int32_t (*session_prepare_request_fork)(
      OrbitKvSessionHandle *, const OrbitKvSessionRequestForkItem *,
      std::uint32_t, OrbitKvSessionControlId *, char *, std::size_t) =
      orbitkv_session_prepare_request_fork;
  int32_t (*session_prepare_prefix_evict)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrefixId *, std::uint32_t,
      OrbitKvSessionControlId *, char *, std::size_t) =
      orbitkv_session_prepare_prefix_evict;
  int32_t (*session_abort_control)(OrbitKvSessionHandle *,
                                   OrbitKvSessionControlId, char *,
                                   std::size_t) = orbitkv_session_abort_control;
  int32_t (*session_commit_control)(
      OrbitKvSessionHandle *, OrbitKvSessionControlId,
      OrbitKvSessionControlPlanInfo *, char *, std::size_t) =
      orbitkv_session_commit_control;
  int32_t (*session_read_control_plan)(
      OrbitKvSessionHandle *, OrbitKvSessionControlId,
      OrbitKvSessionMaterializedRequest *, std::uint32_t, std::uint32_t *,
      OrbitKvSnapshotPage *, std::uint32_t, std::uint32_t *,
      OrbitKvSessionPrefixId *, std::uint32_t, std::uint32_t *,
      OrbitKvSessionRetirement *, std::uint32_t, std::uint32_t *, char *,
      std::size_t) = orbitkv_session_read_control_plan;
  int32_t (*session_cancel_pending_attach)(
      OrbitKvSessionHandle *, OrbitKvSessionPendingAttachCancel,
      OrbitKvSessionPendingAttachCancelOutcome *, char *, std::size_t) =
      orbitkv_session_cancel_pending_attach;
  int32_t (*session_finalize_pending_attach_cancel)(
      OrbitKvSessionHandle *, OrbitKvSessionPendingAttachCancel,
      OrbitKvSessionPendingAttachCancelOutcome *, char *, std::size_t) =
      orbitkv_session_finalize_pending_attach_cancel;
  int32_t (*session_confirm_control)(
      OrbitKvSessionHandle *, OrbitKvSessionControlEvidence,
      const OrbitKvSessionRetirementEvidence *, std::uint32_t,
      OrbitKvSessionControlOutcome *, char *, std::size_t) =
      orbitkv_session_confirm_control;
  int32_t (*session_quarantine_control)(OrbitKvSessionHandle *,
                                        OrbitKvSessionControlId, char *,
                                        std::size_t) =
      orbitkv_session_quarantine_control;
  int32_t (*session_token_views)(
      OrbitKvSessionHandle *, const OrbitKvSessionTokenViewQuery *,
      std::uint32_t, OrbitKvSessionTokenView *, std::uint32_t,
      std::uint32_t *, OrbitKvTokenPlacement *, std::uint32_t,
      std::uint32_t *, char *, std::size_t) = orbitkv_session_token_views_batch;
  int32_t (*session_mark_dispositions)(
      OrbitKvSessionHandle *,
      const OrbitKvSessionTokenDispositionBatchItem *, std::uint32_t,
      const OrbitKvClassTokenDispositionUpdate *, std::uint32_t,
      OrbitKvSessionRequestView *, std::uint32_t, std::uint32_t *, char *,
      std::size_t) = orbitkv_session_mark_token_dispositions_batch;
  int32_t (*session_prepare_relocation)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrepareRelocationItem *,
      std::uint32_t, OrbitKvSessionRelocationId *,
      OrbitKvSessionRelocationPlan *, std::uint32_t, std::uint32_t *,
      OrbitKvPageLease *, std::uint32_t, std::uint32_t *, OrbitKvPageLease *,
      std::uint32_t, std::uint32_t *, OrbitKvTokenMove *, std::uint32_t,
      std::uint32_t *, char *, std::size_t) =
      orbitkv_session_prepare_relocation_batch;
  int32_t (*session_abort_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationId,
      const OrbitKvSessionRelocationAbortEvidence *, std::uint32_t, char *,
      std::size_t) = orbitkv_session_abort_prepared_relocation;
  int32_t (*session_quarantine_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationId, char *,
      std::size_t) = orbitkv_session_quarantine_relocation;
  int32_t (*session_submit_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationId,
      const OrbitKvSessionRelocationRequestEvidence *, std::uint32_t,
      const OrbitKvSessionRelocationCopyEvidence *, std::uint32_t, char *,
      std::size_t) = orbitkv_session_submit_relocation;
  int32_t (*session_complete_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationId,
      OrbitKvSessionCompletionEvidence,
      OrbitKvSessionRelocationRequestPublication *, std::uint32_t,
      std::uint32_t *, OrbitKvSessionRetirement *, std::uint32_t,
      std::uint32_t *, char *, std::size_t) =
      orbitkv_session_complete_relocation;
  int32_t (*session_confirm_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationPublicationEvidence,
      const OrbitKvSessionRetirementEvidence *, std::uint32_t, char *,
      std::size_t) = orbitkv_session_confirm_relocation_publication;
  int32_t (*session_confirm_release)(
      OrbitKvSessionHandle *, OrbitKvSessionReleaseEvidence,
      const OrbitKvSessionRetirementEvidence *, std::uint32_t,
      OrbitKvSessionReleaseOutcome *, char *, std::size_t) =
      orbitkv_session_confirm_release;
  (void)session_create;
  (void)&orbitkv_session_arena_identities;
  (void)&orbitkv_session_arena_stats;
  (void)&orbitkv_session_stats;
  (void)&orbitkv_session_acquire_requests;
  (void)session_prefix_lookup;
  (void)session_prefix_publish;
  (void)session_prefix_publish_release;
  (void)session_prepare_prefix_attach;
  (void)session_prepare_request_fork;
  (void)session_prepare_prefix_evict;
  (void)session_abort_control;
  (void)session_commit_control;
  (void)session_read_control_plan;
  (void)session_cancel_pending_attach;
  (void)session_finalize_pending_attach_cancel;
  (void)session_confirm_control;
  (void)session_quarantine_control;
  (void)session_token_views;
  (void)session_mark_dispositions;
  (void)session_prepare_relocation;
  (void)session_abort_relocation;
  (void)session_quarantine_relocation;
  (void)session_submit_relocation;
  (void)session_complete_relocation;
  (void)session_confirm_relocation;
  (void)&orbitkv_session_prepare_append;
  (void)&orbitkv_session_submit_execution;
  (void)&orbitkv_session_abort_prepared;
  (void)&orbitkv_session_quarantine_prepared;
  (void)&orbitkv_session_quarantine_submitted;
  (void)&orbitkv_session_complete_execution;
  (void)&orbitkv_session_confirm_publication;
  (void)&orbitkv_session_prepare_release;
  (void)session_confirm_release;
  (void)&orbitkv_session_destroy;
  (void)&orbitkv_state_pool_create;
  (void)&orbitkv_state_pool_identity;
  (void)&orbitkv_state_pool_stats;
  (void)&orbitkv_state_pool_prepare_batch;
  (void)&orbitkv_state_pool_submit_batch;
  (void)&orbitkv_state_pool_complete_batch;
  (void)&orbitkv_state_pool_abort_batch;
  (void)&orbitkv_state_pool_retire_owners_batch;
  (void)&orbitkv_state_pool_acknowledge_batch;
  (void)&orbitkv_state_pool_current_batch;
  (void)&orbitkv_state_pool_destroy;
  if (orbitkv_wire_version() != ORBITKV_WIRE_VERSION) {
    return 1;
  }
  if (orbitkv_session_destroy(nullptr, nullptr, 0) != ORBITKV_STATUS_OK) {
    return 2;
  }
  return 0;
}
