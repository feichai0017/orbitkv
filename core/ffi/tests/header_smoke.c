#include "orbitkv.h"

_Static_assert(ORBITKV_STATUS_FAIL_STOPPED == -4,
               "fail-stopped status changed");
_Static_assert(ORBITKV_WIRE_VERSION == 14u, "wire version");
_Static_assert(ORBITKV_PLAN_FORMAT_KV_PLAN == 1u, "KV plan format");
_Static_assert(ORBITKV_PLAN_FORMAT_RETENTION_IR == 2u,
               "retention IR plan format");
_Static_assert(ORBITKV_CLASS_LOWERING_RESETTABLE == 2u,
               "resettable lowering flag");
_Static_assert(ORBITKV_CLASS_LOWERING_EPOCH_START == 4u,
               "epoch-start lowering flag");
_Static_assert(ORBITKV_SESSION_RELEASE_COMPLETED == 1u,
               "completed release disposition");
_Static_assert(ORBITKV_SESSION_RELEASE_RECYCLE_PENDING == 2u,
               "pending recycle release disposition");
_Static_assert(ORBITKV_SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING == 1u,
               "pending attach cancel recycle-pending disposition");
_Static_assert(ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED == 2u,
               "pending attach cancel finalized disposition");
_Static_assert(ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION == 1u,
               "materialization control kind");
_Static_assert(ORBITKV_SESSION_CONTROL_KIND_PREFIX_EVICTION == 2u,
               "prefix eviction control kind");
_Static_assert(ORBITKV_SESSION_CONTROL_OUTCOME_MATERIALIZED == 1u,
               "materialized control outcome");
_Static_assert(ORBITKV_SESSION_CONTROL_OUTCOME_EVICTED == 2u,
               "evicted control outcome");
_Static_assert(ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE == 1u,
               "request-private cache policy");
_Static_assert(ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX == 2u,
               "shared-prefix cache policy");
_Static_assert(sizeof(OrbitKvManagerConfig) == 32, "manager config size");
_Static_assert(_Alignof(OrbitKvManagerConfig) == 8,
               "manager config alignment");
_Static_assert(offsetof(OrbitKvManagerConfig, plan_format) == 20,
               "manager config plan-format offset");
_Static_assert(offsetof(OrbitKvManagerConfig, reserved) == 24,
               "manager config reserved offset");
_Static_assert(sizeof(OrbitKvSessionCreateConfig) == 40,
               "session create config size");
_Static_assert(_Alignof(OrbitKvSessionCreateConfig) == 8,
               "session create config alignment");
_Static_assert(offsetof(OrbitKvSessionCreateConfig, manager) == 0,
               "session create manager offset");
_Static_assert(offsetof(OrbitKvSessionCreateConfig, cache_sharing_policy) == 32,
               "session create cache policy offset");
_Static_assert(offsetof(OrbitKvSessionCreateConfig, reserved) == 36,
               "session create reserved offset");

#define ORBITKV_SMOKE_SESSION_LAYOUT(type, size)                             \
  _Static_assert(sizeof(type) == (size), #type " size");                 \
  _Static_assert(_Alignof(type) == 8, #type " alignment")
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
_Static_assert(offsetof(OrbitKvSessionBatchId, sequence) == 8,
               "session batch sequence offset");
_Static_assert(offsetof(OrbitKvSessionPrefixLookup, candidate) == 72,
               "session prefix candidate offset");
_Static_assert(offsetof(OrbitKvSessionPrefixLookup, reserved1) == 100,
               "session prefix second reserved offset");
_Static_assert(
    offsetof(OrbitKvSessionPublishedPrefixRelease, detached_offset) == 100,
    "session prefix publish-release detached offset");
_Static_assert(offsetof(OrbitKvSessionControlPlanInfo, request_count) == 24,
               "session control request count offset");
_Static_assert(offsetof(OrbitKvSessionMaterializedRequest, page_offset) == 28,
               "session materialization page offset");
_Static_assert(offsetof(OrbitKvSessionPendingAttachCancel, resident_count) == 56,
               "session pending attach cancel resident count offset");
_Static_assert(
    offsetof(OrbitKvSessionPendingAttachCancelOutcome, disposition) == 60,
    "session pending attach cancel outcome disposition offset");
_Static_assert(offsetof(OrbitKvSessionPreparedStep, class_offset) == 40,
               "session prepared class offset");
_Static_assert(offsetof(OrbitKvSessionRetirement, completion_value) == 80,
               "session retirement completion offset");
_Static_assert(offsetof(OrbitKvSessionReleaseOutcome, disposition) == 16,
               "session release outcome disposition offset");
_Static_assert(offsetof(OrbitKvSessionTokenViewQuery, expected_boundary) == 8,
               "session token view expected-boundary offset");
_Static_assert(offsetof(OrbitKvSessionRelocationPlan, move_offset) == 40,
               "session relocation move offset");
_Static_assert(
    offsetof(OrbitKvSessionRelocationCopyEvidence, observed) == 104,
    "session relocation observed offset");

/* Mirrors the documented resettable completion bound for B=1, S=P=16, E=8. */
enum {
  ORBITKV_SMOKE_STEP_PAGES = (16u + 16u - 1u) / 16u,
  ORBITKV_SMOKE_RESETTABLE_BOUND =
      ORBITKV_SMOKE_STEP_PAGES + 2u > 8u
          ? ORBITKV_SMOKE_STEP_PAGES + 2u
          : 8u
};
_Static_assert(ORBITKV_SMOKE_RESETTABLE_BOUND == 8u,
               "resettable completion bound must cover a complete epoch");

/* Every wire symbol is type-checked by assignment and retained by use. */
int main(void) {
  uint32_t (*wire_version)(void) = orbitkv_wire_version;
  int32_t (*session_create)(
      const uint8_t *, size_t, const OrbitKvSessionCreateConfig *,
      const OrbitKvBackendArenaRegistration *, uint32_t,
      OrbitKvSessionHandle **, char *, size_t) = orbitkv_session_create;
  int32_t (*session_arena_identities)(
      OrbitKvSessionHandle *, OrbitKvArenaIdentity *, uint32_t, uint32_t *,
      char *, size_t) = orbitkv_session_arena_identities;
  int32_t (*session_arena_stats)(OrbitKvSessionHandle *, OrbitKvArenaStats *,
                                 uint32_t, uint32_t *, char *, size_t) =
      orbitkv_session_arena_stats;
  int32_t (*session_stats)(OrbitKvSessionHandle *, OrbitKvManagerStats *, char *,
                           size_t) = orbitkv_session_stats;
  int32_t (*session_acquire)(OrbitKvSessionHandle *, const uint64_t *, uint32_t,
                             OrbitKvSessionRequestView *, uint32_t, uint32_t *,
                             char *, size_t) =
      orbitkv_session_acquire_requests;
  int32_t (*session_prefix_lookup)(
      OrbitKvSessionHandle *, const OrbitKvPrefixSemanticKey *, uint32_t,
      OrbitKvSessionPrefixLookup *, uint32_t, uint32_t *, char *, size_t) =
      orbitkv_session_prefix_lookup_batch;
  int32_t (*session_prefix_publish)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrefixPublishItem *, uint32_t,
      OrbitKvSessionPublishedPrefix *, uint32_t, uint32_t *, char *, size_t) =
      orbitkv_session_prefix_publish_batch;
  int32_t (*session_prefix_publish_release)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrefixPublishItem *, uint32_t,
      OrbitKvSessionReleaseId *, OrbitKvSessionPublishedPrefixRelease *,
      uint32_t, uint32_t *, OrbitKvDetachedBinding *, uint32_t, uint32_t *,
      char *, size_t) = orbitkv_session_prefix_publish_release_batch;
  int32_t (*session_prepare_prefix_attach)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrefixAttachItem *, uint32_t,
      OrbitKvSessionControlId *, char *, size_t) =
      orbitkv_session_prepare_prefix_attach;
  int32_t (*session_prepare_request_fork)(
      OrbitKvSessionHandle *, const OrbitKvSessionRequestForkItem *, uint32_t,
      OrbitKvSessionControlId *, char *, size_t) =
      orbitkv_session_prepare_request_fork;
  int32_t (*session_prepare_prefix_evict)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrefixId *, uint32_t,
      OrbitKvSessionControlId *, char *, size_t) =
      orbitkv_session_prepare_prefix_evict;
  int32_t (*session_abort_control)(OrbitKvSessionHandle *,
                                   OrbitKvSessionControlId, char *, size_t) =
      orbitkv_session_abort_control;
  int32_t (*session_commit_control)(OrbitKvSessionHandle *,
                                    OrbitKvSessionControlId,
                                    OrbitKvSessionControlPlanInfo *, char *,
                                    size_t) = orbitkv_session_commit_control;
  int32_t (*session_read_control_plan)(
      OrbitKvSessionHandle *, OrbitKvSessionControlId,
      OrbitKvSessionMaterializedRequest *, uint32_t, uint32_t *,
      OrbitKvSnapshotPage *, uint32_t, uint32_t *, OrbitKvSessionPrefixId *,
      uint32_t, uint32_t *, OrbitKvSessionRetirement *, uint32_t, uint32_t *,
      char *, size_t) = orbitkv_session_read_control_plan;
  int32_t (*session_cancel_pending_attach)(
      OrbitKvSessionHandle *, OrbitKvSessionPendingAttachCancel,
      OrbitKvSessionPendingAttachCancelOutcome *, char *, size_t) =
      orbitkv_session_cancel_pending_attach;
  int32_t (*session_finalize_pending_attach_cancel)(
      OrbitKvSessionHandle *, OrbitKvSessionPendingAttachCancel,
      OrbitKvSessionPendingAttachCancelOutcome *, char *, size_t) =
      orbitkv_session_finalize_pending_attach_cancel;
  int32_t (*session_confirm_control)(
      OrbitKvSessionHandle *, OrbitKvSessionControlEvidence,
      const OrbitKvSessionRetirementEvidence *, uint32_t,
      OrbitKvSessionControlOutcome *, char *, size_t) =
      orbitkv_session_confirm_control;
  int32_t (*session_quarantine_control)(OrbitKvSessionHandle *,
                                        OrbitKvSessionControlId, char *,
                                        size_t) =
      orbitkv_session_quarantine_control;
  int32_t (*session_token_views)(
      OrbitKvSessionHandle *, const OrbitKvSessionTokenViewQuery *, uint32_t,
      OrbitKvSessionTokenView *, uint32_t, uint32_t *, OrbitKvTokenPlacement *,
      uint32_t, uint32_t *, char *, size_t) = orbitkv_session_token_views_batch;
  int32_t (*session_mark_dispositions)(
      OrbitKvSessionHandle *,
      const OrbitKvSessionTokenDispositionBatchItem *, uint32_t,
      const OrbitKvClassTokenDispositionUpdate *, uint32_t,
      OrbitKvSessionRequestView *, uint32_t, uint32_t *, char *, size_t) =
      orbitkv_session_mark_token_dispositions_batch;
  int32_t (*session_prepare_relocation)(
      OrbitKvSessionHandle *, const OrbitKvSessionPrepareRelocationItem *,
      uint32_t, OrbitKvSessionRelocationId *, OrbitKvSessionRelocationPlan *,
      uint32_t, uint32_t *, OrbitKvPageLease *, uint32_t, uint32_t *,
      OrbitKvPageLease *, uint32_t, uint32_t *, OrbitKvTokenMove *, uint32_t,
      uint32_t *, char *, size_t) = orbitkv_session_prepare_relocation_batch;
  int32_t (*session_abort_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationId,
      const OrbitKvSessionRelocationAbortEvidence *, uint32_t, char *, size_t) =
      orbitkv_session_abort_prepared_relocation;
  int32_t (*session_quarantine_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationId, char *, size_t) =
      orbitkv_session_quarantine_relocation;
  int32_t (*session_submit_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationId,
      const OrbitKvSessionRelocationRequestEvidence *, uint32_t,
      const OrbitKvSessionRelocationCopyEvidence *, uint32_t, char *, size_t) =
      orbitkv_session_submit_relocation;
  int32_t (*session_complete_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationId,
      OrbitKvSessionCompletionEvidence,
      OrbitKvSessionRelocationRequestPublication *, uint32_t, uint32_t *,
      OrbitKvSessionRetirement *, uint32_t, uint32_t *, char *, size_t) =
      orbitkv_session_complete_relocation;
  int32_t (*session_confirm_relocation)(
      OrbitKvSessionHandle *, OrbitKvSessionRelocationPublicationEvidence,
      const OrbitKvSessionRetirementEvidence *, uint32_t, char *, size_t) =
      orbitkv_session_confirm_relocation_publication;
  int32_t (*session_prepare_append)(
      OrbitKvSessionHandle *, const OrbitKvSessionAppendIntent *, uint32_t,
      OrbitKvSessionBatchId *, OrbitKvSessionPreparedStep *, uint32_t,
      uint32_t *, OrbitKvClassLowering *, uint32_t, uint32_t *,
      OrbitKvTailAction *, uint32_t, uint32_t *, OrbitKvCopyIntent *, uint32_t,
      uint32_t *, OrbitKvWriteIntent *, uint32_t, uint32_t *, char *, size_t) =
      orbitkv_session_prepare_append;
  int32_t (*session_submit)(
      OrbitKvSessionHandle *, OrbitKvSessionBatchId,
      const OrbitKvSessionStepExecutionEvidence *, uint32_t,
      const OrbitKvSessionBindEvidence *, uint32_t,
      const OrbitKvSessionCopyEvidence *, uint32_t, char *, size_t) =
      orbitkv_session_submit_execution;
  int32_t (*session_abort)(OrbitKvSessionHandle *, OrbitKvSessionBatchId,
                           const OrbitKvSessionStepAbortEvidence *, uint32_t,
                           char *, size_t) = orbitkv_session_abort_prepared;
  int32_t (*session_quarantine_prepared)(OrbitKvSessionHandle *,
                                         OrbitKvSessionBatchId, char *, size_t) =
      orbitkv_session_quarantine_prepared;
  int32_t (*session_quarantine_submitted)(
      OrbitKvSessionHandle *, OrbitKvSessionBatchId, char *, size_t) =
      orbitkv_session_quarantine_submitted;
  int32_t (*session_complete)(
      OrbitKvSessionHandle *, OrbitKvSessionBatchId,
      OrbitKvSessionCompletionEvidence, OrbitKvSessionPublicationId *,
      OrbitKvSessionStepPublication *, uint32_t, uint32_t *,
      OrbitKvDetachedBinding *, uint32_t, uint32_t *,
      OrbitKvSessionRetirement *, uint32_t, uint32_t *, char *, size_t) =
      orbitkv_session_complete_execution;
  int32_t (*session_confirm_publication)(
      OrbitKvSessionHandle *, OrbitKvSessionPublicationEvidence,
      const OrbitKvSessionRetirementEvidence *, uint32_t, char *, size_t) =
      orbitkv_session_confirm_publication;
  int32_t (*session_prepare_release)(
      OrbitKvSessionHandle *, const uint64_t *, uint32_t,
      OrbitKvSessionReleaseId *, OrbitKvSessionReleasedRequest *, uint32_t,
      uint32_t *, OrbitKvDetachedBinding *, uint32_t, uint32_t *,
      OrbitKvSessionRetirement *, uint32_t, uint32_t *, char *, size_t) =
      orbitkv_session_prepare_release;
  int32_t (*session_confirm_release)(
      OrbitKvSessionHandle *, OrbitKvSessionReleaseEvidence,
      const OrbitKvSessionRetirementEvidence *, uint32_t,
      OrbitKvSessionReleaseOutcome *, char *, size_t) =
      orbitkv_session_confirm_release;
  int32_t (*session_destroy)(OrbitKvSessionHandle *, char *, size_t) =
      orbitkv_session_destroy;
  int32_t (*state_create)(const OrbitKvStatePoolConfig *,
                          OrbitKvStatePoolHandle **, char *, size_t) =
      orbitkv_state_pool_create;
  int32_t (*state_identity)(OrbitKvStatePoolHandle *,
                            OrbitKvStatePoolIdentity *, char *, size_t) =
      orbitkv_state_pool_identity;
  int32_t (*state_stats)(OrbitKvStatePoolHandle *, OrbitKvStatePoolStats *,
                         char *, size_t) = orbitkv_state_pool_stats;
  int32_t (*state_prepare)(OrbitKvStatePoolHandle *,
                           const OrbitKvStatePrepareItem *, uint32_t,
                           OrbitKvStateCopyIntent *, uint32_t, uint32_t *,
                           char *, size_t) = orbitkv_state_pool_prepare_batch;
  int32_t (*state_submit)(OrbitKvStatePoolHandle *,
                          const OrbitKvStateCopyReceipt *, uint32_t, char *,
                          size_t) = orbitkv_state_pool_submit_batch;
  int32_t (*state_complete)(
      OrbitKvStatePoolHandle *, const OrbitKvStateCompletionReceipt *,
      const OrbitKvStateTransitionLease *, uint32_t, OrbitKvStatePublication *,
      uint32_t, uint32_t *, char *, size_t) =
      orbitkv_state_pool_complete_batch;
  int32_t (*state_abort)(OrbitKvStatePoolHandle *,
                         const OrbitKvStateAbortItem *, uint32_t, char *,
                         size_t) = orbitkv_state_pool_abort_batch;
  int32_t (*state_retire)(
      OrbitKvStatePoolHandle *, const OrbitKvStateCompletionReceipt *,
      const OrbitKvStateRetireOwnerItem *, uint32_t,
      OrbitKvStateRetirementCertificate *, uint32_t, uint32_t *, char *,
      size_t) = orbitkv_state_pool_retire_owners_batch;
  int32_t (*state_ack)(OrbitKvStatePoolHandle *,
                       const OrbitKvStateRetirementCertificate *, uint32_t,
                       char *, size_t) = orbitkv_state_pool_acknowledge_batch;
  int32_t (*state_current)(OrbitKvStatePoolHandle *, const uint64_t *, uint32_t,
                           OrbitKvStateCurrent *, uint32_t, uint32_t *, char *,
                           size_t) = orbitkv_state_pool_current_batch;
  int32_t (*state_destroy)(OrbitKvStatePoolHandle *, char *, size_t) =
      orbitkv_state_pool_destroy;

  (void)session_create;
  (void)session_arena_identities;
  (void)session_arena_stats;
  (void)session_stats;
  (void)session_acquire;
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
  (void)session_prepare_append;
  (void)session_submit;
  (void)session_abort;
  (void)session_quarantine_prepared;
  (void)session_quarantine_submitted;
  (void)session_complete;
  (void)session_confirm_publication;
  (void)session_prepare_release;
  (void)session_confirm_release;
  (void)state_create;
  (void)state_identity;
  (void)state_stats;
  (void)state_prepare;
  (void)state_submit;
  (void)state_complete;
  (void)state_abort;
  (void)state_retire;
  (void)state_ack;
  (void)state_current;
  (void)state_destroy;
  if (wire_version() != ORBITKV_WIRE_VERSION) {
    return 1;
  }
  if (session_destroy(NULL, NULL, 0) != ORBITKV_STATUS_OK) {
    return 2;
  }
  return 0;
}
