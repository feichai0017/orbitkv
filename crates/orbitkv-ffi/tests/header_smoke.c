#include "orbitkv.h"

_Static_assert(ORBITKV_STATUS_FAIL_STOPPED == -4,
               "fail-stopped status changed");

/* Every ABI6 symbol is type-checked by assignment and retained by use. */
int main(void) {
  uint32_t (*abi_version)(void) = orbitkv_abi_version;
  int32_t (*create)(const uint8_t *, size_t, const OrbitKvManagerConfig *,
                    const OrbitKvBackendArenaRegistration *, uint32_t,
                    OrbitKvManagerHandle **, char *, size_t) =
      orbitkv_manager_create;
  int32_t (*arena_identities)(OrbitKvManagerHandle *, OrbitKvArenaIdentity *,
                              uint32_t, uint32_t *, char *, size_t) =
      orbitkv_manager_arena_identities;
  int32_t (*arena_stats)(OrbitKvManagerHandle *, OrbitKvArenaStats *, uint32_t,
                         uint32_t *, char *, size_t) =
      orbitkv_manager_arena_stats;
  int32_t (*acquire)(OrbitKvManagerHandle *, uint32_t, OrbitKvRequestView *,
                     uint32_t, uint32_t *, char *, size_t) =
      orbitkv_manager_request_acquire_batch;
  int32_t (*fork_batch)(OrbitKvManagerHandle *,
                        const OrbitKvRequestForkBatchItem *, uint32_t,
                        OrbitKvForkedBatchItem *, uint32_t, uint32_t *,
                        OrbitKvSnapshotPage *, uint32_t, uint32_t *, char *,
                        size_t) = orbitkv_manager_request_fork_batch;
  int32_t (*prepare)(OrbitKvManagerHandle *, const OrbitKvPrepareBatchItem *,
                     uint32_t, OrbitKvPreparedBatchItem *, uint32_t,
                     uint32_t *, OrbitKvClassLowering *, uint32_t, uint32_t *,
                     OrbitKvTailAction *, uint32_t, uint32_t *,
                     OrbitKvCopyIntent *, uint32_t, uint32_t *,
                     OrbitKvWriteIntent *, uint32_t, uint32_t *, char *,
                     size_t) = orbitkv_manager_prepare_batch;
  int32_t (*submit)(OrbitKvManagerHandle *, const OrbitKvSubmitBatchItem *,
                    uint32_t, const OrbitKvBackendBindReceipt *, uint32_t,
                    const OrbitKvBackendCopyReceipt *, uint32_t,
                    OrbitKvSubmittedBatchItem *, uint32_t, uint32_t *, char *,
                    size_t) = orbitkv_manager_submit_batch;
  int32_t (*complete)(OrbitKvManagerHandle *, OrbitKvBatchCompletionReceipt,
                      const OrbitKvCompleteBatchItem *, uint32_t,
                      OrbitKvCompletedBatchItem *, uint32_t, uint32_t *,
                      OrbitKvDetachedBinding *, uint32_t, uint32_t *,
                      OrbitKvReclamationCertificate *, uint32_t, uint32_t *,
                      char *, size_t) = orbitkv_manager_complete_batch;
  int32_t (*abort_batch)(OrbitKvManagerHandle *,
                         const OrbitKvBackendUnobservedReceipt *, uint32_t,
                         char *, size_t) = orbitkv_manager_abort_steps_batch;
  int32_t (*quarantine_steps)(OrbitKvManagerHandle *, const OrbitKvStepLease *,
                              uint32_t, char *, size_t) =
      orbitkv_manager_quarantine_steps_batch;
  int32_t (*quarantine_submissions)(OrbitKvManagerHandle *,
                                    const OrbitKvSubmissionLease *, uint32_t,
                                    char *, size_t) =
      orbitkv_manager_quarantine_submissions_batch;
  int32_t (*release_batch)(
      OrbitKvManagerHandle *, const OrbitKvReleaseBatchItem *, uint32_t,
      OrbitKvReleasedBatchItem *, uint32_t, uint32_t *,
      OrbitKvDetachedBinding *, uint32_t, uint32_t *,
      OrbitKvReclamationCertificate *, uint32_t, uint32_t *, char *,
      size_t) = orbitkv_manager_release_batch;
  int32_t (*ack_batch)(OrbitKvManagerHandle *,
                       const OrbitKvReclamationReceipt *, uint32_t, char *,
                       size_t) = orbitkv_manager_acknowledge_reclamations_batch;
  int32_t (*recycle_requests)(OrbitKvManagerHandle *,
                              const OrbitKvRequestLease *, uint32_t, char *,
                              size_t) = orbitkv_manager_recycle_requests_batch;
  int32_t (*lookup)(OrbitKvManagerHandle *, const OrbitKvPrefixSemanticKey *,
                    uint32_t, OrbitKvPrefixLookupHint *, uint32_t, uint32_t *,
                    char *, size_t) = orbitkv_manager_prefix_lookup_batch;
  int32_t (*attach)(OrbitKvManagerHandle *,
                    const OrbitKvPrefixAttachBatchItem *, uint32_t,
                    OrbitKvAttachedPrefixBatchItem *, uint32_t, uint32_t *,
                    OrbitKvSnapshotPage *, uint32_t, uint32_t *, char *,
                    size_t) = orbitkv_manager_prefix_attach_batch;
  int32_t (*publish)(OrbitKvManagerHandle *,
                     const OrbitKvPrefixPublishBatchItem *, uint32_t,
                     OrbitKvPublishedPrefix *, uint32_t, uint32_t *, char *,
                     size_t) = orbitkv_manager_prefix_publish_batch;
  int32_t (*publish_release)(
      OrbitKvManagerHandle *, const OrbitKvPrefixPublishBatchItem *, uint32_t,
      OrbitKvPrefixPublishReleaseBatchItem *, uint32_t, uint32_t *,
      OrbitKvDetachedBinding *, uint32_t, uint32_t *,
      OrbitKvReclamationCertificate *, uint32_t, uint32_t *, char *,
      size_t) = orbitkv_manager_prefix_publish_release_batch;
  int32_t (*evict)(OrbitKvManagerHandle *, const OrbitKvPrefixLease *, uint32_t,
                   OrbitKvEvictedPrefix *, uint32_t, uint32_t *,
                   OrbitKvReclamationCertificate *, uint32_t, uint32_t *,
                   char *, size_t) = orbitkv_manager_prefix_evict_batch;
  int32_t (*recycle_prefix)(OrbitKvManagerHandle *, const OrbitKvPrefixLease *,
                            uint32_t, char *, size_t) =
      orbitkv_manager_prefix_recycle_batch;
  int32_t (*stats)(OrbitKvManagerHandle *, OrbitKvManagerStats *, char *,
                   size_t) = orbitkv_manager_stats;
  int32_t (*destroy)(OrbitKvManagerHandle *, char *, size_t) =
      orbitkv_manager_destroy;

  (void)arena_identities;
  (void)arena_stats;
  (void)acquire;
  (void)fork_batch;
  (void)prepare;
  (void)submit;
  (void)complete;
  (void)abort_batch;
  (void)quarantine_steps;
  (void)quarantine_submissions;
  (void)release_batch;
  (void)ack_batch;
  (void)recycle_requests;
  (void)lookup;
  (void)attach;
  (void)publish;
  (void)publish_release;
  (void)evict;
  (void)recycle_prefix;
  (void)stats;
  if (abi_version() != ORBITKV_ABI_VERSION) {
    return 1;
  }
  if (destroy(NULL, NULL, 0) != ORBITKV_STATUS_OK) {
    return 2;
  }
  if (create(NULL, 0, NULL, NULL, 0, NULL, NULL, 0) !=
      ORBITKV_STATUS_INVALID_ARGUMENT) {
    return 3;
  }
  return 0;
}
