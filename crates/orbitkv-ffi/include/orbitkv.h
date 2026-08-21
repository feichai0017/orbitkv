#ifndef ORBITKV_H
#define ORBITKV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORBITKV_ABI_VERSION 6u

#define ORBITKV_STATUS_OK 0
#define ORBITKV_STATUS_BUFFER_TOO_SMALL 1
#define ORBITKV_STATUS_RETRYABLE_CONFLICT 2
#define ORBITKV_STATUS_INVALID_ARGUMENT -1
#define ORBITKV_STATUS_MANAGER_ERROR -2
#define ORBITKV_STATUS_PANIC -3
#define ORBITKV_STATUS_FAIL_STOPPED -4

/*
 * Status contract for mutating calls:
 *
 * - BUFFER_TOO_SMALL, RETRYABLE_CONFLICT, INVALID_ARGUMENT, and MANAGER_ERROR
 *   are returned only before commit and leave manager state unchanged.
 * - FAIL_STOPPED means the call performed a known fail-closed quarantine
 *   mutation after semantic backend-receipt validation failed. The caller
 *   must permanently stop manager lifecycle/allocation work and must not
 *   retry or reuse the handle. Only stats and destroy are allowed.
 * - PANIC means the call outcome is unknown. The caller must permanently
 *   fail-stop, must not retry the operation, and must not reuse the manager
 *   handle for lifecycle or allocation work. Destruction is the only allowed
 *   follow-up.
 * - RETRYABLE_CONFLICT is the only normal status that permits a fresh
 *   lookup/replan and retry.
 */

#define ORBITKV_TAIL_NONE 0u
#define ORBITKV_TAIL_IN_PLACE 1u
#define ORBITKV_TAIL_COPY_ON_WRITE 2u
#define ORBITKV_TAIL_FRESH 3u

#define ORBITKV_DETACHED_CLEAR 1u
#define ORBITKV_DETACHED_REPLACE 2u
#define ORBITKV_DETACHED_RETENTION 1u
#define ORBITKV_DETACHED_COPY_ON_WRITE 2u
#define ORBITKV_DETACHED_REQUEST_RELEASE 3u
#define ORBITKV_DETACHED_PREFIX_TRANSFER 4u

/*
 * ABI6 is breaking and batch-only. Every caller-supplied reserved field must
 * be zero. Mutating calls validate count envelopes, pointers, reserved fields,
 * canonical spans, and all output capacities before core mutation. A short
 * buffer reports required counts and leaves manager state unchanged.
 */

typedef struct OrbitKvManagerHandle OrbitKvManagerHandle;

#define ORBITKV_LEASE(name)                                                   \
  typedef struct name {                                                       \
    uint64_t engine_epoch;                                                    \
    uint32_t slot;                                                            \
    uint32_t generation;                                                      \
  } name

ORBITKV_LEASE(OrbitKvRequestLease);
ORBITKV_LEASE(OrbitKvSnapshotLease);
ORBITKV_LEASE(OrbitKvStepLease);
ORBITKV_LEASE(OrbitKvSubmissionLease);
ORBITKV_LEASE(OrbitKvReclamationLease);
ORBITKV_LEASE(OrbitKvPrefixLease);

#undef ORBITKV_LEASE

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
  uint32_t maximum_prefixes;
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
  uint16_t class_id;
  uint16_t backend_domain;
  uint32_t pool_id;
  uint32_t page_count;
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
  uint64_t request_page_refs;
  uint64_t prefix_page_refs;
  uint64_t reader_pins;
} OrbitKvArenaStats;

typedef struct OrbitKvRequestView {
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease snapshot;
  uint64_t view_version;
  uint64_t boundary;
  uint32_t resident_count;
  uint32_t reserved;
} OrbitKvRequestView;

typedef struct OrbitKvSnapshotPage {
  OrbitKvPageLease page;
  uint64_t logical_ordinal;
  uint64_t temporal_cell_index;
  uint64_t temporal_cycle;
  uint64_t backend_index;
  uint16_t class_id;
  uint16_t backend_domain;
  uint32_t valid_token_count;
  uint32_t visible_token_offset;
  uint32_t visible_token_count;
  uint32_t reserved;
} OrbitKvSnapshotPage;

typedef struct OrbitKvRequestForkBatchItem {
  OrbitKvRequestLease source_request;
  OrbitKvSnapshotLease expected_source_head;
  OrbitKvRequestLease target_empty_request;
  OrbitKvSnapshotLease expected_target_head;
} OrbitKvRequestForkBatchItem;

typedef struct OrbitKvForkedBatchItem {
  OrbitKvRequestLease source;
  OrbitKvRequestView target;
  uint32_t page_offset;
  uint32_t page_count;
} OrbitKvForkedBatchItem;

typedef struct OrbitKvPrepareBatchItem {
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease expected_head;
  uint64_t target_boundary;
  uint64_t reserved;
} OrbitKvPrepareBatchItem;

typedef struct OrbitKvPreparedBatchItem {
  OrbitKvStepLease step;
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease base_snapshot;
  OrbitKvSnapshotLease target_snapshot;
  uint64_t base_view_version;
  uint64_t target_view_version;
  uint64_t previous_boundary;
  uint64_t target_boundary;
  uint32_t class_offset;
  uint32_t class_count;
  uint32_t tail_offset;
  uint32_t tail_count;
  uint32_t copy_offset;
  uint32_t copy_count;
  uint32_t write_offset;
  uint32_t write_count;
} OrbitKvPreparedBatchItem;

typedef struct OrbitKvClassLowering {
  uint16_t class_id;
  uint16_t flags;
  uint32_t tail_offset;
  uint32_t tail_count;
  uint32_t copy_offset;
  uint32_t copy_count;
  uint32_t write_offset;
  uint32_t write_count;
  uint32_t reserved;
} OrbitKvClassLowering;

typedef struct OrbitKvTailAction {
  uint16_t class_id;
  uint16_t kind;
  uint32_t valid_token_count;
  uint64_t logical_ordinal;
  OrbitKvPageLease source;
  OrbitKvPageLease destination;
  uint64_t reserved;
} OrbitKvTailAction;

typedef struct OrbitKvCopyIntent {
  uint16_t class_id;
  uint16_t backend_domain;
  uint32_t token_count;
  uint32_t source_token_offset;
  uint32_t destination_token_offset;
  uint32_t reserved;
  OrbitKvPageLease source;
  OrbitKvPageLease destination;
  uint64_t source_backend_index;
  uint64_t destination_backend_index;
} OrbitKvCopyIntent;

typedef struct OrbitKvWriteIntent {
  uint64_t page_generation;
  uint32_t page_id;
  uint32_t reserved;
} OrbitKvWriteIntent;

typedef struct OrbitKvBackendBindReceipt {
  OrbitKvStepLease step;
  OrbitKvPageLease page;
  uint16_t backend_domain;
  uint8_t mapped;
  uint8_t writable;
  uint32_t reserved;
  uint64_t backend_index;
} OrbitKvBackendBindReceipt;

typedef struct OrbitKvBackendCopyReceipt {
  OrbitKvStepLease step;
  uint16_t class_id;
  uint16_t backend_domain;
  uint32_t token_count;
  uint32_t source_token_offset;
  uint32_t destination_token_offset;
  uint8_t observed;
  uint8_t copied;
  uint8_t ordered_before_writes;
  uint8_t reserved8;
  uint32_t reserved32;
  OrbitKvPageLease source;
  OrbitKvPageLease destination;
  uint64_t source_backend_index;
  uint64_t destination_backend_index;
} OrbitKvBackendCopyReceipt;

typedef struct OrbitKvSubmitBatchItem {
  OrbitKvStepLease step;
  uint32_t receipt_offset;
  uint32_t receipt_count;
  uint32_t copy_receipt_offset;
  uint32_t copy_receipt_count;
} OrbitKvSubmitBatchItem;

typedef struct OrbitKvSubmittedBatchItem {
  OrbitKvSubmissionLease submission;
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease target_snapshot;
} OrbitKvSubmittedBatchItem;

typedef struct OrbitKvBatchCompletionReceipt {
  uint64_t engine_epoch;
  uint64_t completion_domain;
  uint64_t completion_value;
  uint32_t confirmed;
  uint32_t reserved;
} OrbitKvBatchCompletionReceipt;

typedef struct OrbitKvCompleteBatchItem {
  OrbitKvSubmissionLease submission;
} OrbitKvCompleteBatchItem;

/*
 * A detached binding is a non-owning instruction for updating checked backend
 * mirrors such as ReqToToken and Full-to-SWA LUTs. CLEAR removes the old
 * mapping; REPLACE verifies/installs the replacement mapping. A detach never
 * authorizes physical page reuse.
 */
typedef struct OrbitKvDetachedBinding {
  OrbitKvPageLease old;
  OrbitKvPageLease replacement;
  uint64_t logical_ordinal;
  uint64_t old_backend_index;
  uint64_t replacement_backend_index;
  uint64_t token_begin;
  uint64_t token_end_exclusive;
  uint16_t class_id;
  uint16_t backend_domain;
  uint16_t action;
  uint16_t reason;
  uint64_t reserved;
} OrbitKvDetachedBinding;

/*
 * Reclamation certificates are batch-global and page-owned. A successful
 * mutator returns at most one certificate for a physical page generation;
 * only its exact OrbitKvReclamationLease may authorize reuse through
 * orbitkv_manager_acknowledge_reclamations_batch.
 *
 * Callers must consume each successful output batch collectively: validate
 * every item/detach/certificate, apply every mirror CLEAR/REPLACE, synchronize
 * those mirror updates, and only then ACK the complete certificate array.
 * Partial consumption or ACK-before-mirror-sync is invalid.
 */
typedef struct OrbitKvReclamationCertificate {
  OrbitKvReclamationLease reclamation;
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
  OrbitKvSnapshotLease detached_snapshot;
  OrbitKvSnapshotLease published_snapshot;
  uint64_t published_view_version;
  uint64_t published_boundary;
  uint32_t resident_count;
  uint32_t detached_offset;
  uint32_t detached_count;
  uint32_t reserved;
} OrbitKvCompletedBatchItem;

typedef struct OrbitKvBackendUnobservedReceipt {
  OrbitKvStepLease step;
  uint32_t backend_unobserved;
  uint32_t reserved;
} OrbitKvBackendUnobservedReceipt;

typedef struct OrbitKvReleaseBatchItem {
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease expected_head;
} OrbitKvReleaseBatchItem;

typedef struct OrbitKvReleasedBatchItem {
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease detached_snapshot;
  uint32_t detached_offset;
  uint32_t detached_count;
  uint64_t reserved;
} OrbitKvReleasedBatchItem;

typedef struct OrbitKvReclamationReceipt {
  OrbitKvReclamationLease reclamation;
  OrbitKvPageLease page;
  uint16_t backend_domain;
  uint8_t acknowledged;
  uint8_t reserved8;
  uint32_t reserved32;
  uint64_t backend_index;
} OrbitKvReclamationReceipt;

typedef struct OrbitKvPrefixSemanticKey {
  uint8_t namespace_bytes[32];
  uint8_t digest[32];
  uint64_t boundary;
} OrbitKvPrefixSemanticKey;

typedef struct OrbitKvPrefixLookupHint {
  OrbitKvPrefixSemanticKey key;
  OrbitKvPrefixLease candidate;
  uint32_t resident_count;
  uint32_t candidate_present;
  uint32_t reserved;
  uint32_t reserved_padding;
} OrbitKvPrefixLookupHint;

typedef struct OrbitKvPrefixAttachBatchItem {
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease expected_empty_head;
  OrbitKvPrefixLookupHint hint;
} OrbitKvPrefixAttachBatchItem;

typedef struct OrbitKvAttachedPrefixBatchItem {
  OrbitKvPrefixLease prefix;
  OrbitKvRequestView target;
  uint32_t page_offset;
  uint32_t page_count;
} OrbitKvAttachedPrefixBatchItem;

typedef struct OrbitKvPrefixPublishBatchItem {
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease expected_head;
  OrbitKvPrefixSemanticKey key;
} OrbitKvPrefixPublishBatchItem;

typedef struct OrbitKvPublishedPrefix {
  OrbitKvPrefixLease prefix;
  OrbitKvPrefixSemanticKey key;
  uint32_t resident_count;
  uint32_t reserved;
} OrbitKvPublishedPrefix;

typedef struct OrbitKvPrefixPublishReleaseBatchItem {
  OrbitKvPublishedPrefix publication;
  OrbitKvRequestLease request;
  OrbitKvSnapshotLease detached_snapshot;
  uint32_t detached_offset;
  uint32_t detached_count;
  uint64_t reserved;
} OrbitKvPrefixPublishReleaseBatchItem;

typedef struct OrbitKvEvictedPrefix {
  OrbitKvPrefixLease prefix;
  OrbitKvPrefixSemanticKey key;
} OrbitKvEvictedPrefix;

typedef struct OrbitKvManagerStats {
  uint64_t active_requests;
  uint64_t active_snapshots;
  uint64_t active_prefixes;
  uint64_t evicted_prefixes;
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
  uint64_t total_request_page_refs;
  uint64_t total_prefix_page_refs;
  uint64_t total_reader_pins;
} OrbitKvManagerStats;

#if defined(__cplusplus)
#define ORBITKV_STATIC_ASSERT static_assert
#define ORBITKV_ALIGNOF alignof
#else
#define ORBITKV_STATIC_ASSERT _Static_assert
#define ORBITKV_ALIGNOF _Alignof
#endif
#define ORBITKV_LAYOUT(type, size, alignment)                                 \
  ORBITKV_STATIC_ASSERT(sizeof(type) == (size), #type " size");              \
  ORBITKV_STATIC_ASSERT(ORBITKV_ALIGNOF(type) == (alignment),                 \
                        #type " alignment")
#define ORBITKV_OFFSET(type, field, offset)                                  \
  ORBITKV_STATIC_ASSERT(offsetof(type, field) == (offset),                    \
                        #type "." #field " offset")

ORBITKV_LAYOUT(OrbitKvRequestLease, 16, 8);
ORBITKV_OFFSET(OrbitKvRequestLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvRequestLease, slot, 8);
ORBITKV_OFFSET(OrbitKvRequestLease, generation, 12);
ORBITKV_LAYOUT(OrbitKvSnapshotLease, 16, 8);
ORBITKV_OFFSET(OrbitKvSnapshotLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvSnapshotLease, slot, 8);
ORBITKV_OFFSET(OrbitKvSnapshotLease, generation, 12);
ORBITKV_LAYOUT(OrbitKvStepLease, 16, 8);
ORBITKV_OFFSET(OrbitKvStepLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvStepLease, slot, 8);
ORBITKV_OFFSET(OrbitKvStepLease, generation, 12);
ORBITKV_LAYOUT(OrbitKvSubmissionLease, 16, 8);
ORBITKV_OFFSET(OrbitKvSubmissionLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvSubmissionLease, slot, 8);
ORBITKV_OFFSET(OrbitKvSubmissionLease, generation, 12);
ORBITKV_LAYOUT(OrbitKvReclamationLease, 16, 8);
ORBITKV_OFFSET(OrbitKvReclamationLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvReclamationLease, slot, 8);
ORBITKV_OFFSET(OrbitKvReclamationLease, generation, 12);
ORBITKV_LAYOUT(OrbitKvPrefixLease, 16, 8);
ORBITKV_OFFSET(OrbitKvPrefixLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvPrefixLease, slot, 8);
ORBITKV_OFFSET(OrbitKvPrefixLease, generation, 12);
ORBITKV_LAYOUT(OrbitKvPageLease, 32, 8);
ORBITKV_OFFSET(OrbitKvPageLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvPageLease, pool_epoch, 8);
ORBITKV_OFFSET(OrbitKvPageLease, generation, 16);
ORBITKV_OFFSET(OrbitKvPageLease, page_id, 24);
ORBITKV_OFFSET(OrbitKvPageLease, pool_id, 28);
ORBITKV_LAYOUT(OrbitKvBackendArenaRegistration, 24, 8);
ORBITKV_OFFSET(OrbitKvBackendArenaRegistration, pool_id, 0);
ORBITKV_OFFSET(OrbitKvBackendArenaRegistration, class_id, 4);
ORBITKV_OFFSET(OrbitKvBackendArenaRegistration, backend_domain, 6);
ORBITKV_OFFSET(OrbitKvBackendArenaRegistration, page_count, 8);
ORBITKV_OFFSET(OrbitKvBackendArenaRegistration, reserved, 12);
ORBITKV_OFFSET(OrbitKvBackendArenaRegistration, backend_base_index, 16);
ORBITKV_LAYOUT(OrbitKvManagerConfig, 20, 4);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_requests, 0);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_operations, 4);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_prefixes, 8);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_reclamations, 12);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_step_tokens, 16);
ORBITKV_LAYOUT(OrbitKvArenaIdentity, 48, 8);
ORBITKV_OFFSET(OrbitKvArenaIdentity, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvArenaIdentity, pool_epoch, 8);
ORBITKV_OFFSET(OrbitKvArenaIdentity, backend_base_index, 16);
ORBITKV_OFFSET(OrbitKvArenaIdentity, pool_id, 24);
ORBITKV_OFFSET(OrbitKvArenaIdentity, page_count, 28);
ORBITKV_OFFSET(OrbitKvArenaIdentity, page_tokens, 32);
ORBITKV_OFFSET(OrbitKvArenaIdentity, class_id, 36);
ORBITKV_OFFSET(OrbitKvArenaIdentity, backend_domain, 38);
ORBITKV_OFFSET(OrbitKvArenaIdentity, first_page_id, 40);
ORBITKV_OFFSET(OrbitKvArenaIdentity, reserved, 44);
ORBITKV_LAYOUT(OrbitKvArenaStats, 120, 8);
ORBITKV_OFFSET(OrbitKvArenaStats, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvArenaStats, pool_epoch, 8);
ORBITKV_OFFSET(OrbitKvArenaStats, class_id, 16);
ORBITKV_OFFSET(OrbitKvArenaStats, backend_domain, 18);
ORBITKV_OFFSET(OrbitKvArenaStats, pool_id, 20);
ORBITKV_OFFSET(OrbitKvArenaStats, page_count, 24);
ORBITKV_OFFSET(OrbitKvArenaStats, first_page_id, 28);
ORBITKV_OFFSET(OrbitKvArenaStats, reserved, 32);
ORBITKV_OFFSET(OrbitKvArenaStats, reserved_padding, 36);
ORBITKV_OFFSET(OrbitKvArenaStats, free_pages, 40);
ORBITKV_OFFSET(OrbitKvArenaStats, reserved_pages, 48);
ORBITKV_OFFSET(OrbitKvArenaStats, writing_pages, 56);
ORBITKV_OFFSET(OrbitKvArenaStats, active_pages, 64);
ORBITKV_OFFSET(OrbitKvArenaStats, retiring_pages, 72);
ORBITKV_OFFSET(OrbitKvArenaStats, quarantined_pages, 80);
ORBITKV_OFFSET(OrbitKvArenaStats, exhausted_pages, 88);
ORBITKV_OFFSET(OrbitKvArenaStats, request_page_refs, 96);
ORBITKV_OFFSET(OrbitKvArenaStats, prefix_page_refs, 104);
ORBITKV_OFFSET(OrbitKvArenaStats, reader_pins, 112);
ORBITKV_LAYOUT(OrbitKvRequestView, 56, 8);
ORBITKV_OFFSET(OrbitKvRequestView, request, 0);
ORBITKV_OFFSET(OrbitKvRequestView, snapshot, 16);
ORBITKV_OFFSET(OrbitKvRequestView, view_version, 32);
ORBITKV_OFFSET(OrbitKvRequestView, boundary, 40);
ORBITKV_OFFSET(OrbitKvRequestView, resident_count, 48);
ORBITKV_OFFSET(OrbitKvRequestView, reserved, 52);
ORBITKV_LAYOUT(OrbitKvSnapshotPage, 88, 8);
ORBITKV_OFFSET(OrbitKvSnapshotPage, page, 0);
ORBITKV_OFFSET(OrbitKvSnapshotPage, logical_ordinal, 32);
ORBITKV_OFFSET(OrbitKvSnapshotPage, temporal_cell_index, 40);
ORBITKV_OFFSET(OrbitKvSnapshotPage, temporal_cycle, 48);
ORBITKV_OFFSET(OrbitKvSnapshotPage, backend_index, 56);
ORBITKV_OFFSET(OrbitKvSnapshotPage, class_id, 64);
ORBITKV_OFFSET(OrbitKvSnapshotPage, backend_domain, 66);
ORBITKV_OFFSET(OrbitKvSnapshotPage, valid_token_count, 68);
ORBITKV_OFFSET(OrbitKvSnapshotPage, visible_token_offset, 72);
ORBITKV_OFFSET(OrbitKvSnapshotPage, visible_token_count, 76);
ORBITKV_OFFSET(OrbitKvSnapshotPage, reserved, 80);
ORBITKV_LAYOUT(OrbitKvRequestForkBatchItem, 64, 8);
ORBITKV_OFFSET(OrbitKvRequestForkBatchItem, source_request, 0);
ORBITKV_OFFSET(OrbitKvRequestForkBatchItem, expected_source_head, 16);
ORBITKV_OFFSET(OrbitKvRequestForkBatchItem, target_empty_request, 32);
ORBITKV_OFFSET(OrbitKvRequestForkBatchItem, expected_target_head, 48);
ORBITKV_LAYOUT(OrbitKvForkedBatchItem, 80, 8);
ORBITKV_OFFSET(OrbitKvForkedBatchItem, source, 0);
ORBITKV_OFFSET(OrbitKvForkedBatchItem, target, 16);
ORBITKV_OFFSET(OrbitKvForkedBatchItem, page_offset, 72);
ORBITKV_OFFSET(OrbitKvForkedBatchItem, page_count, 76);
ORBITKV_LAYOUT(OrbitKvPrepareBatchItem, 48, 8);
ORBITKV_OFFSET(OrbitKvPrepareBatchItem, request, 0);
ORBITKV_OFFSET(OrbitKvPrepareBatchItem, expected_head, 16);
ORBITKV_OFFSET(OrbitKvPrepareBatchItem, target_boundary, 32);
ORBITKV_OFFSET(OrbitKvPrepareBatchItem, reserved, 40);
ORBITKV_LAYOUT(OrbitKvPreparedBatchItem, 128, 8);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, step, 0);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, request, 16);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, base_snapshot, 32);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, target_snapshot, 48);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, base_view_version, 64);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, target_view_version, 72);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, previous_boundary, 80);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, target_boundary, 88);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, class_offset, 96);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, class_count, 100);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, tail_offset, 104);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, tail_count, 108);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, copy_offset, 112);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, copy_count, 116);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, write_offset, 120);
ORBITKV_OFFSET(OrbitKvPreparedBatchItem, write_count, 124);
ORBITKV_LAYOUT(OrbitKvClassLowering, 32, 4);
ORBITKV_OFFSET(OrbitKvClassLowering, class_id, 0);
ORBITKV_OFFSET(OrbitKvClassLowering, flags, 2);
ORBITKV_OFFSET(OrbitKvClassLowering, tail_offset, 4);
ORBITKV_OFFSET(OrbitKvClassLowering, tail_count, 8);
ORBITKV_OFFSET(OrbitKvClassLowering, copy_offset, 12);
ORBITKV_OFFSET(OrbitKvClassLowering, copy_count, 16);
ORBITKV_OFFSET(OrbitKvClassLowering, write_offset, 20);
ORBITKV_OFFSET(OrbitKvClassLowering, write_count, 24);
ORBITKV_OFFSET(OrbitKvClassLowering, reserved, 28);
ORBITKV_LAYOUT(OrbitKvTailAction, 88, 8);
ORBITKV_OFFSET(OrbitKvTailAction, class_id, 0);
ORBITKV_OFFSET(OrbitKvTailAction, kind, 2);
ORBITKV_OFFSET(OrbitKvTailAction, valid_token_count, 4);
ORBITKV_OFFSET(OrbitKvTailAction, logical_ordinal, 8);
ORBITKV_OFFSET(OrbitKvTailAction, source, 16);
ORBITKV_OFFSET(OrbitKvTailAction, destination, 48);
ORBITKV_OFFSET(OrbitKvTailAction, reserved, 80);
ORBITKV_LAYOUT(OrbitKvCopyIntent, 104, 8);
ORBITKV_OFFSET(OrbitKvCopyIntent, class_id, 0);
ORBITKV_OFFSET(OrbitKvCopyIntent, backend_domain, 2);
ORBITKV_OFFSET(OrbitKvCopyIntent, token_count, 4);
ORBITKV_OFFSET(OrbitKvCopyIntent, source_token_offset, 8);
ORBITKV_OFFSET(OrbitKvCopyIntent, destination_token_offset, 12);
ORBITKV_OFFSET(OrbitKvCopyIntent, reserved, 16);
ORBITKV_OFFSET(OrbitKvCopyIntent, source, 24);
ORBITKV_OFFSET(OrbitKvCopyIntent, destination, 56);
ORBITKV_OFFSET(OrbitKvCopyIntent, source_backend_index, 88);
ORBITKV_OFFSET(OrbitKvCopyIntent, destination_backend_index, 96);
ORBITKV_LAYOUT(OrbitKvWriteIntent, 16, 8);
ORBITKV_OFFSET(OrbitKvWriteIntent, page_generation, 0);
ORBITKV_OFFSET(OrbitKvWriteIntent, page_id, 8);
ORBITKV_OFFSET(OrbitKvWriteIntent, reserved, 12);
ORBITKV_LAYOUT(OrbitKvBackendBindReceipt, 64, 8);
ORBITKV_OFFSET(OrbitKvBackendBindReceipt, step, 0);
ORBITKV_OFFSET(OrbitKvBackendBindReceipt, page, 16);
ORBITKV_OFFSET(OrbitKvBackendBindReceipt, backend_domain, 48);
ORBITKV_OFFSET(OrbitKvBackendBindReceipt, mapped, 50);
ORBITKV_OFFSET(OrbitKvBackendBindReceipt, writable, 51);
ORBITKV_OFFSET(OrbitKvBackendBindReceipt, reserved, 52);
ORBITKV_OFFSET(OrbitKvBackendBindReceipt, backend_index, 56);
ORBITKV_LAYOUT(OrbitKvBackendCopyReceipt, 120, 8);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, step, 0);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, class_id, 16);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, backend_domain, 18);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, token_count, 20);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, source_token_offset, 24);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, destination_token_offset, 28);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, observed, 32);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, copied, 33);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, ordered_before_writes, 34);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, reserved8, 35);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, reserved32, 36);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, source, 40);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, destination, 72);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, source_backend_index, 104);
ORBITKV_OFFSET(OrbitKvBackendCopyReceipt, destination_backend_index, 112);
ORBITKV_LAYOUT(OrbitKvSubmitBatchItem, 32, 8);
ORBITKV_OFFSET(OrbitKvSubmitBatchItem, step, 0);
ORBITKV_OFFSET(OrbitKvSubmitBatchItem, receipt_offset, 16);
ORBITKV_OFFSET(OrbitKvSubmitBatchItem, receipt_count, 20);
ORBITKV_OFFSET(OrbitKvSubmitBatchItem, copy_receipt_offset, 24);
ORBITKV_OFFSET(OrbitKvSubmitBatchItem, copy_receipt_count, 28);
ORBITKV_LAYOUT(OrbitKvSubmittedBatchItem, 48, 8);
ORBITKV_OFFSET(OrbitKvSubmittedBatchItem, submission, 0);
ORBITKV_OFFSET(OrbitKvSubmittedBatchItem, request, 16);
ORBITKV_OFFSET(OrbitKvSubmittedBatchItem, target_snapshot, 32);
ORBITKV_LAYOUT(OrbitKvBatchCompletionReceipt, 32, 8);
ORBITKV_OFFSET(OrbitKvBatchCompletionReceipt, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvBatchCompletionReceipt, completion_domain, 8);
ORBITKV_OFFSET(OrbitKvBatchCompletionReceipt, completion_value, 16);
ORBITKV_OFFSET(OrbitKvBatchCompletionReceipt, confirmed, 24);
ORBITKV_OFFSET(OrbitKvBatchCompletionReceipt, reserved, 28);
ORBITKV_LAYOUT(OrbitKvCompleteBatchItem, 16, 8);
ORBITKV_OFFSET(OrbitKvCompleteBatchItem, submission, 0);
ORBITKV_LAYOUT(OrbitKvDetachedBinding, 120, 8);
ORBITKV_OFFSET(OrbitKvDetachedBinding, old, 0);
ORBITKV_OFFSET(OrbitKvDetachedBinding, replacement, 32);
ORBITKV_OFFSET(OrbitKvDetachedBinding, logical_ordinal, 64);
ORBITKV_OFFSET(OrbitKvDetachedBinding, old_backend_index, 72);
ORBITKV_OFFSET(OrbitKvDetachedBinding, replacement_backend_index, 80);
ORBITKV_OFFSET(OrbitKvDetachedBinding, token_begin, 88);
ORBITKV_OFFSET(OrbitKvDetachedBinding, token_end_exclusive, 96);
ORBITKV_OFFSET(OrbitKvDetachedBinding, class_id, 104);
ORBITKV_OFFSET(OrbitKvDetachedBinding, backend_domain, 106);
ORBITKV_OFFSET(OrbitKvDetachedBinding, action, 108);
ORBITKV_OFFSET(OrbitKvDetachedBinding, reason, 110);
ORBITKV_OFFSET(OrbitKvDetachedBinding, reserved, 112);
ORBITKV_LAYOUT(OrbitKvReclamationCertificate, 104, 8);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, reclamation, 0);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, page, 16);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, class_id, 48);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, backend_domain, 50);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, reserved32, 52);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, logical_ordinal, 56);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, backend_index, 64);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, token_begin, 72);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, token_end_exclusive, 80);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, completion_domain, 88);
ORBITKV_OFFSET(OrbitKvReclamationCertificate, completion_value, 96);
ORBITKV_LAYOUT(OrbitKvCompletedBatchItem, 96, 8);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, submission, 0);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, request, 16);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, detached_snapshot, 32);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, published_snapshot, 48);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, published_view_version, 64);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, published_boundary, 72);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, resident_count, 80);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, detached_offset, 84);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, detached_count, 88);
ORBITKV_OFFSET(OrbitKvCompletedBatchItem, reserved, 92);
ORBITKV_LAYOUT(OrbitKvBackendUnobservedReceipt, 24, 8);
ORBITKV_OFFSET(OrbitKvBackendUnobservedReceipt, step, 0);
ORBITKV_OFFSET(OrbitKvBackendUnobservedReceipt, backend_unobserved, 16);
ORBITKV_OFFSET(OrbitKvBackendUnobservedReceipt, reserved, 20);
ORBITKV_LAYOUT(OrbitKvReleaseBatchItem, 32, 8);
ORBITKV_OFFSET(OrbitKvReleaseBatchItem, request, 0);
ORBITKV_OFFSET(OrbitKvReleaseBatchItem, expected_head, 16);
ORBITKV_LAYOUT(OrbitKvReleasedBatchItem, 48, 8);
ORBITKV_OFFSET(OrbitKvReleasedBatchItem, request, 0);
ORBITKV_OFFSET(OrbitKvReleasedBatchItem, detached_snapshot, 16);
ORBITKV_OFFSET(OrbitKvReleasedBatchItem, detached_offset, 32);
ORBITKV_OFFSET(OrbitKvReleasedBatchItem, detached_count, 36);
ORBITKV_OFFSET(OrbitKvReleasedBatchItem, reserved, 40);
ORBITKV_LAYOUT(OrbitKvReclamationReceipt, 64, 8);
ORBITKV_OFFSET(OrbitKvReclamationReceipt, reclamation, 0);
ORBITKV_OFFSET(OrbitKvReclamationReceipt, page, 16);
ORBITKV_OFFSET(OrbitKvReclamationReceipt, backend_domain, 48);
ORBITKV_OFFSET(OrbitKvReclamationReceipt, acknowledged, 50);
ORBITKV_OFFSET(OrbitKvReclamationReceipt, reserved8, 51);
ORBITKV_OFFSET(OrbitKvReclamationReceipt, reserved32, 52);
ORBITKV_OFFSET(OrbitKvReclamationReceipt, backend_index, 56);
ORBITKV_LAYOUT(OrbitKvPrefixSemanticKey, 72, 8);
ORBITKV_OFFSET(OrbitKvPrefixSemanticKey, namespace_bytes, 0);
ORBITKV_OFFSET(OrbitKvPrefixSemanticKey, digest, 32);
ORBITKV_OFFSET(OrbitKvPrefixSemanticKey, boundary, 64);
ORBITKV_LAYOUT(OrbitKvPrefixLookupHint, 104, 8);
ORBITKV_OFFSET(OrbitKvPrefixLookupHint, key, 0);
ORBITKV_OFFSET(OrbitKvPrefixLookupHint, candidate, 72);
ORBITKV_OFFSET(OrbitKvPrefixLookupHint, resident_count, 88);
ORBITKV_OFFSET(OrbitKvPrefixLookupHint, candidate_present, 92);
ORBITKV_OFFSET(OrbitKvPrefixLookupHint, reserved, 96);
ORBITKV_OFFSET(OrbitKvPrefixLookupHint, reserved_padding, 100);
ORBITKV_LAYOUT(OrbitKvPrefixAttachBatchItem, 136, 8);
ORBITKV_OFFSET(OrbitKvPrefixAttachBatchItem, request, 0);
ORBITKV_OFFSET(OrbitKvPrefixAttachBatchItem, expected_empty_head, 16);
ORBITKV_OFFSET(OrbitKvPrefixAttachBatchItem, hint, 32);
ORBITKV_LAYOUT(OrbitKvAttachedPrefixBatchItem, 80, 8);
ORBITKV_OFFSET(OrbitKvAttachedPrefixBatchItem, prefix, 0);
ORBITKV_OFFSET(OrbitKvAttachedPrefixBatchItem, target, 16);
ORBITKV_OFFSET(OrbitKvAttachedPrefixBatchItem, page_offset, 72);
ORBITKV_OFFSET(OrbitKvAttachedPrefixBatchItem, page_count, 76);
ORBITKV_LAYOUT(OrbitKvPrefixPublishBatchItem, 104, 8);
ORBITKV_OFFSET(OrbitKvPrefixPublishBatchItem, request, 0);
ORBITKV_OFFSET(OrbitKvPrefixPublishBatchItem, expected_head, 16);
ORBITKV_OFFSET(OrbitKvPrefixPublishBatchItem, key, 32);
ORBITKV_LAYOUT(OrbitKvPublishedPrefix, 96, 8);
ORBITKV_OFFSET(OrbitKvPublishedPrefix, prefix, 0);
ORBITKV_OFFSET(OrbitKvPublishedPrefix, key, 16);
ORBITKV_OFFSET(OrbitKvPublishedPrefix, resident_count, 88);
ORBITKV_OFFSET(OrbitKvPublishedPrefix, reserved, 92);
ORBITKV_LAYOUT(OrbitKvPrefixPublishReleaseBatchItem, 144, 8);
ORBITKV_OFFSET(OrbitKvPrefixPublishReleaseBatchItem, publication, 0);
ORBITKV_OFFSET(OrbitKvPrefixPublishReleaseBatchItem, request, 96);
ORBITKV_OFFSET(OrbitKvPrefixPublishReleaseBatchItem, detached_snapshot, 112);
ORBITKV_OFFSET(OrbitKvPrefixPublishReleaseBatchItem, detached_offset, 128);
ORBITKV_OFFSET(OrbitKvPrefixPublishReleaseBatchItem, detached_count, 132);
ORBITKV_OFFSET(OrbitKvPrefixPublishReleaseBatchItem, reserved, 136);
ORBITKV_LAYOUT(OrbitKvEvictedPrefix, 88, 8);
ORBITKV_OFFSET(OrbitKvEvictedPrefix, prefix, 0);
ORBITKV_OFFSET(OrbitKvEvictedPrefix, key, 16);
ORBITKV_LAYOUT(OrbitKvManagerStats, 136, 8);
ORBITKV_OFFSET(OrbitKvManagerStats, active_requests, 0);
ORBITKV_OFFSET(OrbitKvManagerStats, active_snapshots, 8);
ORBITKV_OFFSET(OrbitKvManagerStats, active_prefixes, 16);
ORBITKV_OFFSET(OrbitKvManagerStats, evicted_prefixes, 24);
ORBITKV_OFFSET(OrbitKvManagerStats, prepared_steps, 32);
ORBITKV_OFFSET(OrbitKvManagerStats, submitted_steps, 40);
ORBITKV_OFFSET(OrbitKvManagerStats, free_pages, 48);
ORBITKV_OFFSET(OrbitKvManagerStats, reserved_pages, 56);
ORBITKV_OFFSET(OrbitKvManagerStats, writing_pages, 64);
ORBITKV_OFFSET(OrbitKvManagerStats, active_pages, 72);
ORBITKV_OFFSET(OrbitKvManagerStats, retiring_pages, 80);
ORBITKV_OFFSET(OrbitKvManagerStats, quarantined_pages, 88);
ORBITKV_OFFSET(OrbitKvManagerStats, exhausted_pages, 96);
ORBITKV_OFFSET(OrbitKvManagerStats, pending_reclamations, 104);
ORBITKV_OFFSET(OrbitKvManagerStats, total_request_page_refs, 112);
ORBITKV_OFFSET(OrbitKvManagerStats, total_prefix_page_refs, 120);
ORBITKV_OFFSET(OrbitKvManagerStats, total_reader_pins, 128);

#undef ORBITKV_OFFSET
#undef ORBITKV_LAYOUT
#undef ORBITKV_ALIGNOF
#undef ORBITKV_STATIC_ASSERT

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
    OrbitKvRequestView *requests, uint32_t request_capacity,
    uint32_t *out_request_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_request_fork_batch(
    OrbitKvManagerHandle *manager, const OrbitKvRequestForkBatchItem *items,
    uint32_t item_count, OrbitKvForkedBatchItem *forked,
    uint32_t forked_capacity, uint32_t *out_forked_count,
    OrbitKvSnapshotPage *pages, uint32_t page_capacity,
    uint32_t *out_page_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_prepare_batch(
    OrbitKvManagerHandle *manager, const OrbitKvPrepareBatchItem *items,
    uint32_t item_count, OrbitKvPreparedBatchItem *prepared,
    uint32_t prepared_capacity, uint32_t *out_prepared_count,
    OrbitKvClassLowering *class_lowerings, uint32_t class_capacity,
    uint32_t *out_class_count, OrbitKvTailAction *tail_actions,
    uint32_t tail_capacity, uint32_t *out_tail_count,
    OrbitKvCopyIntent *copy_intents, uint32_t copy_capacity,
    uint32_t *out_copy_count, OrbitKvWriteIntent *write_intents,
    uint32_t write_capacity, uint32_t *out_write_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_manager_submit_batch(
    OrbitKvManagerHandle *manager, const OrbitKvSubmitBatchItem *items,
    uint32_t item_count, const OrbitKvBackendBindReceipt *receipts,
    uint32_t receipt_count, const OrbitKvBackendCopyReceipt *copy_receipts,
    uint32_t copy_receipt_count, OrbitKvSubmittedBatchItem *submitted,
    uint32_t submitted_capacity, uint32_t *out_submitted_count,
    char *error_buffer, size_t error_buffer_len);

/*
 * Hot completion output capacity is delta-bounded. For B items, C classes,
 * maximum step S, and page size P, detached required count is no greater than
 * B*C*(ceil(S/P)+2). Detached bindings are per request reference, so shared
 * pages can occur more than once and this count is not capped by physical page
 * capacity. The batch-global, page-owned certificate required count is no
 * greater than min(total_physical_pages, B*C*(ceil(S/P)+2)). Neither count
 * scales with a request's resident context length.
 */
int32_t orbitkv_manager_complete_batch(
    OrbitKvManagerHandle *manager, OrbitKvBatchCompletionReceipt receipt,
    const OrbitKvCompleteBatchItem *items, uint32_t item_count,
    OrbitKvCompletedBatchItem *completed, uint32_t completed_capacity,
    uint32_t *out_completed_count, OrbitKvDetachedBinding *detached,
    uint32_t detached_capacity, uint32_t *out_detached_count,
    OrbitKvReclamationCertificate *retirements, uint32_t retirement_capacity,
    uint32_t *out_retirement_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_manager_abort_steps_batch(
    OrbitKvManagerHandle *manager,
    const OrbitKvBackendUnobservedReceipt *receipts, uint32_t receipt_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_quarantine_steps_batch(
    OrbitKvManagerHandle *manager, const OrbitKvStepLease *steps,
    uint32_t step_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_quarantine_submissions_batch(
    OrbitKvManagerHandle *manager,
    const OrbitKvSubmissionLease *submissions, uint32_t submission_count,
    char *error_buffer, size_t error_buffer_len);

/*
 * Release is a cold path. While holding the manager mutex it revalidates every
 * expected head, sums the exact current resident_count values, preflights the
 * detached buffer with that exact sum, and commits under the same lock. The
 * certificate required count is bounded by that resident sum.
 */
int32_t orbitkv_manager_release_batch(
    OrbitKvManagerHandle *manager, const OrbitKvReleaseBatchItem *items,
    uint32_t item_count, OrbitKvReleasedBatchItem *released,
    uint32_t released_capacity, uint32_t *out_released_count,
    OrbitKvDetachedBinding *detached, uint32_t detached_capacity,
    uint32_t *out_detached_count,
    OrbitKvReclamationCertificate *retirements, uint32_t retirement_capacity,
    uint32_t *out_retirement_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_manager_acknowledge_reclamations_batch(
    OrbitKvManagerHandle *manager,
    const OrbitKvReclamationReceipt *receipts, uint32_t receipt_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_recycle_requests_batch(
    OrbitKvManagerHandle *manager, const OrbitKvRequestLease *requests,
    uint32_t request_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_prefix_lookup_batch(
    OrbitKvManagerHandle *manager, const OrbitKvPrefixSemanticKey *keys,
    uint32_t key_count, OrbitKvPrefixLookupHint *hints,
    uint32_t hint_capacity, uint32_t *out_hint_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_manager_prefix_attach_batch(
    OrbitKvManagerHandle *manager, const OrbitKvPrefixAttachBatchItem *items,
    uint32_t item_count, OrbitKvAttachedPrefixBatchItem *attached,
    uint32_t attached_capacity, uint32_t *out_attached_count,
    OrbitKvSnapshotPage *pages, uint32_t page_capacity,
    uint32_t *out_page_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_prefix_publish_batch(
    OrbitKvManagerHandle *manager, const OrbitKvPrefixPublishBatchItem *items,
    uint32_t item_count, OrbitKvPublishedPrefix *published,
    uint32_t published_capacity, uint32_t *out_published_count,
    char *error_buffer, size_t error_buffer_len);

/*
 * Prefix publish-release performs the same locked expected-head census as
 * release, so its detached required count is the exact sum of source resident
 * counts. Its batch-global certificate count is zero because page ownership
 * transfers from request references to prefix references.
 */
int32_t orbitkv_manager_prefix_publish_release_batch(
    OrbitKvManagerHandle *manager, const OrbitKvPrefixPublishBatchItem *items,
    uint32_t item_count, OrbitKvPrefixPublishReleaseBatchItem *outputs,
    uint32_t output_capacity, uint32_t *out_output_count,
    OrbitKvDetachedBinding *detached, uint32_t detached_capacity,
    uint32_t *out_detached_count,
    OrbitKvReclamationCertificate *retirements, uint32_t retirement_capacity,
    uint32_t *out_retirement_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_manager_prefix_evict_batch(
    OrbitKvManagerHandle *manager, const OrbitKvPrefixLease *prefixes,
    uint32_t prefix_count, OrbitKvEvictedPrefix *evicted,
    uint32_t evicted_capacity, uint32_t *out_evicted_count,
    OrbitKvReclamationCertificate *retirements, uint32_t retirement_capacity,
    uint32_t *out_retirement_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_manager_prefix_recycle_batch(
    OrbitKvManagerHandle *manager, const OrbitKvPrefixLease *prefixes,
    uint32_t prefix_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_manager_stats(
    OrbitKvManagerHandle *manager, OrbitKvManagerStats *out_stats,
    char *error_buffer, size_t error_buffer_len);

/*
 * Exclusively destroys the handle and discards any remaining host authority.
 * Healthy callers must first verify quiescence with orbitkv_manager_stats.
 * Fail-stopped callers use this operation to release an otherwise
 * non-quiescent or outcome-unknown handle; no lifecycle work is implied.
 */
int32_t orbitkv_manager_destroy(
    OrbitKvManagerHandle *manager, char *error_buffer,
    size_t error_buffer_len);

#ifdef __cplusplus
}
#endif

#endif
