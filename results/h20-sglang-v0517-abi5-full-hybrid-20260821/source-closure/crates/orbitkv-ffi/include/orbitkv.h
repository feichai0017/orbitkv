#ifndef ORBITKV_H
#define ORBITKV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORBITKV_ABI_VERSION 5u
#define ORBITKV_CLASS_LOWERING_HAS_PREVIOUS_TAIL (UINT16_C(1) << 0)

#define ORBITKV_STATUS_OK 0
#define ORBITKV_STATUS_BUFFER_TOO_SMALL 1
#define ORBITKV_STATUS_INVALID_ARGUMENT -1
#define ORBITKV_STATUS_MANAGER_ERROR -2
#define ORBITKV_STATUS_PANIC -3

/*
 * ABI5 is batch-only for request lifecycle operations. There are no singular
 * lifecycle aliases. Every reserved field supplied by a caller must be zero.
 * Mutating calls first validate every output capacity and every structural
 * flat-buffer range. A short output buffer returns BUFFER_TOO_SMALL without
 * changing manager state.
 */

typedef struct OrbitKvManagerHandle OrbitKvManagerHandle;

typedef struct OrbitKvRequestLease {
  uint64_t engine_epoch;
  uint32_t slot;
  uint32_t generation;
} OrbitKvRequestLease;

typedef struct OrbitKvStepLease {
  uint64_t engine_epoch;
  uint32_t slot;
  uint32_t generation;
} OrbitKvStepLease;

typedef struct OrbitKvSubmissionLease {
  uint64_t engine_epoch;
  uint32_t slot;
  uint32_t generation;
} OrbitKvSubmissionLease;

typedef struct OrbitKvReclamationLease {
  uint64_t engine_epoch;
  uint32_t slot;
  uint32_t generation;
} OrbitKvReclamationLease;

typedef struct OrbitKvPageLease {
  uint64_t engine_epoch;
  uint64_t pool_epoch;
  uint64_t generation;
  uint32_t page_id;
  uint32_t pool_id;
} OrbitKvPageLease;

typedef struct OrbitKvBackendArenaRegistration {
  uint32_t pool_id;
  uint16_t class_id;
  uint16_t backend_domain;
  uint32_t page_count;
  uint32_t reserved;
  uint64_t backend_base_index;
} OrbitKvBackendArenaRegistration;

typedef struct OrbitKvManagerConfig {
  uint32_t maximum_requests;
  uint32_t maximum_operations;
  uint32_t maximum_reclamations;
  uint32_t maximum_step_tokens;
} OrbitKvManagerConfig;

typedef struct OrbitKvArenaIdentity {
  uint64_t engine_epoch;
  uint64_t pool_epoch;
  uint64_t backend_base_index;
  uint32_t pool_id;
  uint32_t page_count;
  uint32_t page_tokens;
  uint16_t class_id;
  uint16_t backend_domain;
  uint32_t first_page_id;
  uint32_t reserved;
} OrbitKvArenaIdentity;

typedef struct OrbitKvArenaStats {
  uint64_t engine_epoch;
  uint64_t pool_epoch;
  uint32_t pool_id;
  uint32_t page_count;
  uint16_t class_id;
  uint16_t backend_domain;
  uint32_t first_page_id;
  uint32_t reserved;
  uint32_t reserved_padding;
  uint64_t free_pages;
  uint64_t reserved_pages;
  uint64_t writing_pages;
  uint64_t active_pages;
  uint64_t retiring_pages;
  uint64_t quarantined_pages;
  uint64_t exhausted_pages;
} OrbitKvArenaStats;

typedef struct OrbitKvPrepareBatchItem {
  OrbitKvRequestLease request;
  uint64_t target_boundary;
  uint64_t reserved;
} OrbitKvPrepareBatchItem;

typedef struct OrbitKvPreparedBatchItem {
  OrbitKvStepLease step;
  OrbitKvRequestLease request;
  uint64_t base_view_version;
  uint64_t target_view_version;
  uint64_t previous_boundary;
  uint64_t target_boundary;
  uint32_t class_offset;
  uint32_t class_count;
  uint32_t write_offset;
  uint32_t write_count;
} OrbitKvPreparedBatchItem;

typedef struct OrbitKvClassLowering {
  uint16_t class_id;
  uint16_t flags;
  uint32_t write_offset;
  uint32_t write_count;
  uint32_t previous_tail_page_id;
  uint64_t previous_tail_generation;
} OrbitKvClassLowering;

typedef struct OrbitKvWriteIntent {
  uint64_t page_generation;
  uint32_t page_id;
  uint32_t reserved;
} OrbitKvWriteIntent;

/*
 * Page id/generation echo a write intent. Pool/domain/backend fields are the
 * exact affine reconstruction from that intent's registered class arena.
 */
typedef struct OrbitKvBackendBindReceipt {
  OrbitKvStepLease step;
  OrbitKvPageLease page;
  uint16_t backend_domain;
  uint8_t mapped;
  uint8_t writable;
  uint32_t reserved;
  uint64_t backend_index;
} OrbitKvBackendBindReceipt;

/* StepLease is authoritative; the manager derives and returns the request. */
typedef struct OrbitKvSubmitBatchItem {
  OrbitKvStepLease step;
  uint32_t receipt_offset;
  uint32_t receipt_count;
  uint64_t reserved;
} OrbitKvSubmitBatchItem;

typedef struct OrbitKvSubmittedBatchItem {
  OrbitKvSubmissionLease submission;
  OrbitKvRequestLease request;
} OrbitKvSubmittedBatchItem;

/* One completion point shared by every item in complete_batch. */
typedef struct OrbitKvBatchCompletionReceipt {
  uint64_t engine_epoch;
  uint64_t completion_domain;
  uint64_t completion_value;
  uint32_t confirmed;
  uint32_t reserved;
} OrbitKvBatchCompletionReceipt;

/* SubmissionLease is authoritative; the manager derives the request. */
typedef struct OrbitKvCompleteBatchItem {
  OrbitKvSubmissionLease submission;
} OrbitKvCompleteBatchItem;

typedef struct OrbitKvReclamationCertificate {
  OrbitKvReclamationLease reclamation;
  OrbitKvRequestLease request;
  OrbitKvPageLease page;
  uint16_t class_id;
  uint16_t backend_domain;
  uint32_t reserved32;
  uint64_t logical_ordinal;
  uint64_t backend_index;
  uint64_t token_begin;
  uint64_t token_end_exclusive;
  uint64_t completion_domain;
  uint64_t completion_value;
} OrbitKvReclamationCertificate;

typedef struct OrbitKvCompletedBatchItem {
  OrbitKvSubmissionLease submission;
  OrbitKvRequestLease request;
  uint64_t published_view_version;
  uint64_t published_boundary;
  uint32_t resident_count;
  uint32_t retirement_offset;
  uint32_t retirement_count;
  uint32_t reserved;
} OrbitKvCompletedBatchItem;

typedef struct OrbitKvReleaseBatchItem {
  OrbitKvRequestLease request;
  uint64_t reserved;
} OrbitKvReleaseBatchItem;

typedef struct OrbitKvReleasedBatchItem {
  OrbitKvRequestLease request;
  uint32_t retirement_offset;
  uint32_t retirement_count;
  uint64_t reserved;
} OrbitKvReleasedBatchItem;

typedef struct OrbitKvBackendUnobservedReceipt {
  OrbitKvStepLease step;
  uint32_t backend_unobserved;
  uint32_t reserved;
} OrbitKvBackendUnobservedReceipt;

typedef struct OrbitKvReclamationReceipt {
  OrbitKvReclamationLease reclamation;
  OrbitKvPageLease page;
  uint16_t backend_domain;
  uint8_t acknowledged;
  uint8_t reserved8;
  uint32_t reserved32;
  uint64_t backend_index;
} OrbitKvReclamationReceipt;

typedef struct OrbitKvManagerStats {
  uint64_t active_requests;
  uint64_t prepared_steps;
  uint64_t submitted_steps;
  uint64_t free_pages;
  uint64_t reserved_pages;
  uint64_t writing_pages;
  uint64_t active_pages;
  uint64_t retiring_pages;
  uint64_t quarantined_pages;
  uint64_t exhausted_pages;
  uint64_t pending_reclamations;
} OrbitKvManagerStats;

uint32_t orbitkv_abi_version(void);

int32_t orbitkv_manager_create(
    const uint8_t *plan_json, size_t plan_json_len,
    const OrbitKvManagerConfig *config,
    const OrbitKvBackendArenaRegistration *backends, uint32_t backend_count,
    OrbitKvManagerHandle **out_manager, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_manager_arena_identities(
    OrbitKvManagerHandle *manager, OrbitKvArenaIdentity *identities,
    uint32_t identity_capacity, uint32_t *out_identity_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_arena_stats(
    OrbitKvManagerHandle *manager, OrbitKvArenaStats *stats,
    uint32_t stats_capacity, uint32_t *out_stats_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_manager_request_acquire_batch(
    OrbitKvManagerHandle *manager, uint32_t request_count,
    OrbitKvRequestLease *requests, uint32_t request_capacity,
    uint32_t *out_request_count, char *error_buffer, size_t error_buffer_len);

/*
 * Class spans are gap-free in item order. Each class write span indexes the
 * global write-intent buffer and class spans are gap-free inside their item.
 * A short hot buffer reports a bound proportional to this batch and maximum
 * configured step, capped by total physical pages; no canonical root crosses
 * this ABI.
 */
int32_t orbitkv_manager_prepare_batch(
    OrbitKvManagerHandle *manager, const OrbitKvPrepareBatchItem *items,
    uint32_t item_count, OrbitKvPreparedBatchItem *prepared,
    uint32_t prepared_capacity, uint32_t *out_prepared_count,
    OrbitKvClassLowering *class_lowerings, uint32_t class_capacity,
    uint32_t *out_class_count, OrbitKvWriteIntent *write_intents,
    uint32_t write_capacity, uint32_t *out_write_count, char *error_buffer,
    size_t error_buffer_len);

/*
 * Structural item/span errors are zero-mutation and retryable. After a
 * structurally complete receipt batch is presented, any page, generation,
 * backend, or mapping semantic mismatch quarantines the entire candidate
 * batch; it cannot be downgraded to abort.
 */
int32_t orbitkv_manager_submit_batch(
    OrbitKvManagerHandle *manager, const OrbitKvSubmitBatchItem *items,
    uint32_t item_count, const OrbitKvBackendBindReceipt *receipts,
    uint32_t receipt_count, OrbitKvSubmittedBatchItem *submitted,
    uint32_t submitted_capacity, uint32_t *out_submitted_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_complete_batch(
    OrbitKvManagerHandle *manager, OrbitKvBatchCompletionReceipt receipt,
    const OrbitKvCompleteBatchItem *items, uint32_t item_count,
    OrbitKvCompletedBatchItem *completed, uint32_t completed_capacity,
    uint32_t *out_completed_count, OrbitKvReclamationCertificate *retirements,
    uint32_t retirement_capacity, uint32_t *out_retirement_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_abort_steps(
    OrbitKvManagerHandle *manager,
    const OrbitKvBackendUnobservedReceipt *receipts, uint32_t receipt_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_quarantine_steps(
    OrbitKvManagerHandle *manager, const OrbitKvStepLease *steps,
    uint32_t step_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_quarantine_submissions(
    OrbitKvManagerHandle *manager,
    const OrbitKvSubmissionLease *submissions, uint32_t submission_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_release_batch(
    OrbitKvManagerHandle *manager, const OrbitKvReleaseBatchItem *items,
    uint32_t item_count, OrbitKvReleasedBatchItem *released,
    uint32_t released_capacity, uint32_t *out_released_count,
    OrbitKvReclamationCertificate *retirements,
    uint32_t retirement_capacity, uint32_t *out_retirement_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_acknowledge_reclamations(
    OrbitKvManagerHandle *manager,
    const OrbitKvReclamationReceipt *receipts, uint32_t receipt_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_recycle_requests(
    OrbitKvManagerHandle *manager, const OrbitKvRequestLease *requests,
    uint32_t request_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_stats(
    OrbitKvManagerHandle *manager, OrbitKvManagerStats *out_stats,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_destroy(
    OrbitKvManagerHandle *manager, char *error_buffer,
    size_t error_buffer_len);

#ifdef __cplusplus
}
#endif

#endif
