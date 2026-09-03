#ifndef ORBITKV_H
#define ORBITKV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORBITKV_WIRE_VERSION 14u

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
 *   are returned only before commit and leave handle state unchanged.
 * - FAIL_STOPPED means the handle detected an internal invariant failure or
 *   performed a fail-closed quarantine whose scope requires the whole handle
 *   to stop. The caller must not retry lifecycle/allocation work or reuse the
 *   handle. Only stats and destroy are allowed. A successful control-specific
 *   quarantine is local containment and returns OK.
 * - PANIC means the call outcome is unknown. The caller must permanently
 *   fail-stop, must not retry the operation, and must not reuse the handle for
 *   lifecycle or allocation work. Destruction is the only allowed follow-up.
 * - RETRYABLE_CONFLICT is the only normal status that permits a fresh
 *   lookup/replan and retry.
 */

#define ORBITKV_TAIL_NONE 0u
#define ORBITKV_TAIL_IN_PLACE 1u
#define ORBITKV_TAIL_COPY_ON_WRITE 2u
#define ORBITKV_TAIL_FRESH 3u

#define ORBITKV_CLASS_LOWERING_PACKED 1u
#define ORBITKV_CLASS_LOWERING_RESETTABLE 2u
#define ORBITKV_CLASS_LOWERING_EPOCH_START 4u

#define ORBITKV_PLAN_FORMAT_KV_PLAN 1u
#define ORBITKV_PLAN_FORMAT_RETENTION_IR 2u

#define ORBITKV_DETACHED_CLEAR 1u
#define ORBITKV_DETACHED_REPLACE 2u
#define ORBITKV_DETACHED_RETENTION 1u
#define ORBITKV_DETACHED_COPY_ON_WRITE 2u
#define ORBITKV_DETACHED_REQUEST_RELEASE 3u
#define ORBITKV_DETACHED_PREFIX_TRANSFER 4u

#define ORBITKV_TOKEN_RETAINED 0u
#define ORBITKV_TOKEN_SEMANTICALLY_DEAD 1u
#define ORBITKV_TOKEN_POLICY_EVICTED 2u

#define ORBITKV_SESSION_RELEASE_COMPLETED 1u
#define ORBITKV_SESSION_RELEASE_RECYCLE_PENDING 2u
#define ORBITKV_SESSION_PENDING_ATTACH_CANCEL_RECYCLE_PENDING 1u
#define ORBITKV_SESSION_PENDING_ATTACH_CANCEL_FINALIZED 2u
#define ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION 1u
#define ORBITKV_SESSION_CONTROL_KIND_PREFIX_EVICTION 2u
#define ORBITKV_SESSION_CONTROL_OUTCOME_MATERIALIZED 1u
#define ORBITKV_SESSION_CONTROL_OUTCOME_EVICTED 2u
#define ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE 1u
#define ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX 2u

/*
 * This revision is a breaking wire-contract change. RuntimeSession is the
 * public KV lifecycle surface, and its operations are batch-oriented; the
 * separate fixed-state pool also exposes identity, stats, and destroy
 * operations. Every caller-supplied reserved field must be zero. Mutating calls
 * validate count envelopes, pointers, reserved fields, canonical spans, and all
 * output capacities before core mutation. A short buffer reports required
 * counts and leaves handle state unchanged.
 *
 * Pointer contract: every non-null typed pointer must be naturally aligned for
 * its declared type. Every pointer/count or pointer/capacity pair must describe
 * a valid range wholly contained in one allocation; readable typed ranges must
 * contain initialized objects. The byte extent and address calculation of each
 * range must be representable by ptrdiff_t. A zero-count input may be NULL. The
 * active opaque handle must remain live for a call and its storage must not
 * overlap any other argument range. All caller-writable ranges that a call can
 * access (typed outputs, count words, output-handle slots, and the error buffer)
 * must
 * be pairwise non-overlapping. Except where stated otherwise, readable input
 * ranges must not overlap caller-writable ranges. Callers must also prevent
 * concurrent mutation of readable ranges and concurrent access to writable
 * ranges for the duration of a call.
 */

typedef struct OrbitKvSessionHandle OrbitKvSessionHandle;
typedef struct OrbitKvStatePoolHandle OrbitKvStatePoolHandle;

#define ORBITKV_LEASE(name)                                                   \
  typedef struct name {                                                       \
    uint64_t engine_epoch;                                                    \
    uint32_t slot;                                                            \
    uint32_t generation;                                                      \
  } name

ORBITKV_LEASE(OrbitKvStateTransitionLease);
ORBITKV_LEASE(OrbitKvStateRetirementLease);

#undef ORBITKV_LEASE

typedef struct OrbitKvPageLease {
  uint64_t engine_epoch;
  uint64_t pool_epoch;
  uint64_t generation;
  uint32_t page_id;
  uint32_t pool_id;
} OrbitKvPageLease;

typedef struct OrbitKvStatePoolConfig {
  uint64_t engine_epoch;
  uint64_t pool_epoch;
  uint64_t byte_count;
  uint32_t pool_id;
  uint32_t slot_count;
} OrbitKvStatePoolConfig;

typedef struct OrbitKvStateSlotLease {
  uint64_t engine_epoch;
  uint64_t pool_epoch;
  uint64_t generation;
  uint32_t slot_id;
  uint32_t pool_id;
} OrbitKvStateSlotLease;

typedef struct OrbitKvStatePoolIdentity {
  uint64_t engine_epoch;
  uint64_t pool_epoch;
  uint64_t byte_count;
  uint32_t pool_id;
  uint32_t slot_count;
} OrbitKvStatePoolIdentity;

typedef struct OrbitKvStatePoolStats {
  OrbitKvStatePoolIdentity identity;
  uint64_t free_slots;
  uint64_t reserved_slots;
  uint64_t relocating_slots;
  uint64_t live_slots;
  uint64_t retiring_slots;
  uint64_t quarantined_slots;
  uint64_t active_owners;
  uint64_t pending_transitions;
  uint64_t pending_retirements;
} OrbitKvStatePoolStats;

typedef struct OrbitKvStatePrepareItem {
  uint64_t owner_id;
  OrbitKvStateSlotLease expected;
  uint32_t expected_present;
  uint32_t reserved;
} OrbitKvStatePrepareItem;

typedef struct OrbitKvStateCopyIntent {
  OrbitKvStateTransitionLease transition;
  uint64_t owner_id;
  OrbitKvStateSlotLease source;
  OrbitKvStateSlotLease destination;
  uint64_t byte_count;
  uint32_t source_present;
  uint32_t reserved;
} OrbitKvStateCopyIntent;

typedef struct OrbitKvStateCopyReceipt {
  OrbitKvStateTransitionLease transition;
  OrbitKvStateSlotLease source;
  OrbitKvStateSlotLease destination;
  uint64_t byte_count;
  uint8_t source_present;
  uint8_t observed;
  uint8_t written;
  uint8_t reserved8;
  uint32_t reserved32;
} OrbitKvStateCopyReceipt;

typedef struct OrbitKvStateCompletionReceipt {
  uint64_t engine_epoch;
  uint64_t completion_domain;
  uint64_t completion_value;
  uint32_t confirmed;
  uint32_t reserved;
} OrbitKvStateCompletionReceipt;

typedef struct OrbitKvStateRetirementCertificate {
  OrbitKvStateRetirementLease retirement;
  OrbitKvStateSlotLease slot;
  uint64_t byte_count;
  uint64_t completion_domain;
  uint64_t completion_value;
} OrbitKvStateRetirementCertificate;

typedef struct OrbitKvStatePublication {
  uint64_t owner_id;
  OrbitKvStateSlotLease slot;
  OrbitKvStateRetirementCertificate retirement;
  uint32_t retirement_present;
  uint32_t reserved;
} OrbitKvStatePublication;

typedef struct OrbitKvStateAbortItem {
  OrbitKvStateTransitionLease transition;
  uint32_t backend_unobserved;
  uint32_t reserved;
} OrbitKvStateAbortItem;

typedef struct OrbitKvStateRetireOwnerItem {
  uint64_t owner_id;
  OrbitKvStateSlotLease expected;
} OrbitKvStateRetireOwnerItem;

typedef struct OrbitKvStateCurrent {
  uint64_t owner_id;
  OrbitKvStateSlotLease slot;
  uint32_t present;
  uint32_t reserved;
} OrbitKvStateCurrent;

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
  uint32_t plan_format;
  uint64_t reserved;
} OrbitKvManagerConfig;

typedef struct OrbitKvSessionCreateConfig {
  OrbitKvManagerConfig manager;
  uint32_t cache_sharing_policy;
  uint32_t reserved;
} OrbitKvSessionCreateConfig;

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
  uint64_t previous_layout_boundary;
  uint64_t target_layout_boundary;
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

typedef struct OrbitKvPrefixSemanticKey {
  uint8_t namespace_bytes[32];
  uint8_t digest[32];
  uint64_t boundary;
} OrbitKvPrefixSemanticKey;

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

/*
 * Wire token views separate logical token identity and disposition from its
 * current physical placement. location_present is exactly 0 or 1.
 */
typedef struct OrbitKvTokenDisposition {
  uint64_t policy_or_proof_id;
  uint64_t version;
  uint64_t quality_contract;
  uint16_t kind;
  uint16_t reserved16;
  uint32_t reserved32;
} OrbitKvTokenDisposition;

typedef struct OrbitKvTokenLocation {
  OrbitKvPageLease page;
  uint64_t backend_index;
  uint32_t offset;
  uint32_t reserved;
} OrbitKvTokenLocation;

typedef struct OrbitKvTokenPlacement {
  uint64_t token_id;
  OrbitKvTokenDisposition disposition;
  OrbitKvTokenLocation location;
  uint32_t location_present;
  uint32_t reserved;
} OrbitKvTokenPlacement;

typedef struct OrbitKvClassTokenDispositionUpdate {
  uint64_t token_id;
  OrbitKvTokenDisposition disposition;
  uint16_t class_id;
  uint16_t reserved16;
  uint32_t reserved32;
} OrbitKvClassTokenDispositionUpdate;

/*
 * Wire relocation is a prepare/copy/submit/complete transaction. Sources are
 * not reusable until completion returns reclamation certificates and their
 * mirror removals have been synchronized and acknowledged.
 */
typedef struct OrbitKvRelocationPolicy {
  uint32_t maximum_source_pages;
  uint32_t evacuation_headroom_pages;
  uint16_t fragmentation_threshold_milli;
  uint8_t full_evacuation;
  uint8_t reserved8;
  uint32_t reserved32;
} OrbitKvRelocationPolicy;

typedef struct OrbitKvTokenMove {
  uint64_t token_id;
  OrbitKvTokenLocation source;
  OrbitKvTokenLocation destination;
} OrbitKvTokenMove;

/*
 * RuntimeSession wire DTOs expose engine identities and physical backend
 * evidence, never canonical request, snapshot, step, or submission leases.
 */
#define ORBITKV_SESSION_ID(name)                                             \
  typedef struct name {                                                      \
    uint64_t session_epoch;                                                  \
    uint64_t sequence;                                                       \
  } name

ORBITKV_SESSION_ID(OrbitKvSessionBatchId);
ORBITKV_SESSION_ID(OrbitKvSessionPublicationId);
ORBITKV_SESSION_ID(OrbitKvSessionReleaseId);
ORBITKV_SESSION_ID(OrbitKvSessionPrefixId);
ORBITKV_SESSION_ID(OrbitKvSessionControlId);
ORBITKV_SESSION_ID(OrbitKvSessionRelocationId);
#undef ORBITKV_SESSION_ID

typedef struct OrbitKvSessionPrefixLookup {
  OrbitKvPrefixSemanticKey key;
  OrbitKvSessionPrefixId candidate;
  uint32_t resident_count;
  uint32_t candidate_present;
  uint32_t reserved0;
  uint32_t reserved1;
} OrbitKvSessionPrefixLookup;

typedef struct OrbitKvSessionPrefixPublishItem {
  uint64_t request_id;
  OrbitKvPrefixSemanticKey key;
} OrbitKvSessionPrefixPublishItem;

typedef struct OrbitKvSessionPublishedPrefix {
  OrbitKvSessionPrefixId prefix_id;
  OrbitKvPrefixSemanticKey key;
  uint32_t resident_count;
  uint32_t reserved;
} OrbitKvSessionPublishedPrefix;

typedef struct OrbitKvSessionPublishedPrefixRelease {
  uint64_t request_id;
  OrbitKvSessionPrefixId prefix_id;
  OrbitKvPrefixSemanticKey key;
  uint32_t resident_count;
  uint32_t detached_offset;
  uint32_t detached_count;
  uint32_t reserved;
} OrbitKvSessionPublishedPrefixRelease;

typedef struct OrbitKvSessionPrefixAttachItem {
  uint64_t target_request_id;
  OrbitKvSessionPrefixId prefix_id;
  OrbitKvPrefixSemanticKey key;
  uint32_t resident_count;
  uint32_t reserved;
} OrbitKvSessionPrefixAttachItem;

typedef struct OrbitKvSessionRequestForkItem {
  uint64_t source_request_id;
  uint64_t target_request_id;
} OrbitKvSessionRequestForkItem;

typedef struct OrbitKvSessionControlPlanInfo {
  OrbitKvSessionControlId id;
  uint32_t kind;
  uint32_t reserved;
  uint32_t request_count;
  uint32_t page_count;
  uint32_t prefix_count;
  uint32_t retirement_count;
} OrbitKvSessionControlPlanInfo;

typedef struct OrbitKvSessionMaterializedRequest {
  uint64_t request_id;
  uint64_t view_version;
  uint64_t boundary;
  uint32_t resident_count;
  uint32_t page_offset;
  uint32_t page_count;
  uint32_t reserved;
} OrbitKvSessionMaterializedRequest;

typedef struct OrbitKvSessionPendingAttachCancel {
  OrbitKvSessionControlId control_id;
  uint64_t request_id;
  OrbitKvSessionPrefixId prefix_id;
  uint64_t view_version;
  uint64_t boundary;
  uint32_t resident_count;
} OrbitKvSessionPendingAttachCancel;

typedef struct OrbitKvSessionPendingAttachCancelOutcome {
  OrbitKvSessionControlId control_id;
  uint64_t request_id;
  OrbitKvSessionPrefixId prefix_id;
  uint64_t view_version;
  uint64_t boundary;
  uint32_t resident_count;
  uint32_t disposition;
} OrbitKvSessionPendingAttachCancelOutcome;

typedef struct OrbitKvSessionControlEvidence {
  OrbitKvSessionControlId id;
  uint32_t mirror_updates_confirmed;
  uint32_t reserved;
} OrbitKvSessionControlEvidence;

typedef struct OrbitKvSessionControlOutcome {
  OrbitKvSessionControlId id;
  uint32_t disposition;
  uint32_t reserved;
} OrbitKvSessionControlOutcome;

typedef struct OrbitKvSessionRequestView {
  uint64_t request_id;
  uint64_t view_version;
  uint64_t boundary;
  uint32_t resident_count;
  uint32_t reserved;
} OrbitKvSessionRequestView;

typedef struct OrbitKvSessionTokenViewQuery {
  uint64_t request_id;
  uint64_t expected_boundary;
  uint16_t class_id;
  uint16_t reserved16;
  uint32_t reserved32;
} OrbitKvSessionTokenViewQuery;

typedef struct OrbitKvSessionTokenView {
  uint64_t request_id;
  uint64_t view_version;
  uint32_t placement_offset;
  uint32_t placement_count;
  uint32_t page_tokens;
  uint16_t class_id;
  uint16_t reserved16;
  uint32_t reserved32;
} OrbitKvSessionTokenView;

typedef struct OrbitKvSessionTokenDispositionBatchItem {
  uint64_t request_id;
  uint32_t update_offset;
  uint32_t update_count;
} OrbitKvSessionTokenDispositionBatchItem;

typedef struct OrbitKvSessionPrepareRelocationItem {
  uint64_t request_id;
  OrbitKvRelocationPolicy policy;
  uint16_t class_id;
  uint16_t reserved16;
  uint32_t reserved32;
} OrbitKvSessionPrepareRelocationItem;

typedef struct OrbitKvSessionRelocationPlan {
  uint64_t request_id;
  uint64_t base_view_version;
  uint64_t target_view_version;
  uint32_t source_offset;
  uint32_t source_count;
  uint32_t destination_offset;
  uint32_t destination_count;
  uint32_t move_offset;
  uint32_t move_count;
  uint32_t projected_reclaimed_pages;
  uint16_t fragmentation_milli;
  uint16_t class_id;
  uint32_t reserved32;
} OrbitKvSessionRelocationPlan;

typedef struct OrbitKvSessionRelocationRequestEvidence {
  uint64_t request_id;
  uint32_t copy_offset;
  uint32_t copy_count;
} OrbitKvSessionRelocationRequestEvidence;

typedef struct OrbitKvSessionRelocationCopyEvidence {
  uint64_t token_id;
  OrbitKvTokenLocation source;
  OrbitKvTokenLocation destination;
  uint8_t observed;
  uint8_t copied;
  uint16_t reserved16;
  uint32_t reserved32;
} OrbitKvSessionRelocationCopyEvidence;

typedef struct OrbitKvSessionRelocationAbortEvidence {
  uint64_t request_id;
  uint32_t backend_unobserved;
  uint32_t reserved;
} OrbitKvSessionRelocationAbortEvidence;

typedef struct OrbitKvSessionRelocationRequestPublication {
  uint64_t request_id;
  uint64_t view_version;
  uint64_t boundary;
  uint32_t resident_count;
  uint32_t reserved;
} OrbitKvSessionRelocationRequestPublication;

typedef struct OrbitKvSessionRelocationPublicationEvidence {
  OrbitKvSessionRelocationId relocation_id;
  uint32_t mirror_cleanup_confirmed;
  uint32_t reserved;
} OrbitKvSessionRelocationPublicationEvidence;

typedef struct OrbitKvSessionAppendIntent {
  uint64_t request_id;
  uint64_t target_boundary;
} OrbitKvSessionAppendIntent;

typedef struct OrbitKvSessionPreparedStep {
  uint64_t request_id;
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
} OrbitKvSessionPreparedStep;

typedef struct OrbitKvSessionStepExecutionEvidence {
  uint64_t request_id;
  uint32_t bind_offset;
  uint32_t bind_count;
  uint32_t copy_offset;
  uint32_t copy_count;
  uint64_t reserved;
} OrbitKvSessionStepExecutionEvidence;

typedef struct OrbitKvSessionBindEvidence {
  OrbitKvPageLease page;
  uint16_t backend_domain;
  uint8_t mapped;
  uint8_t writable;
  uint32_t reserved;
  uint64_t backend_index;
} OrbitKvSessionBindEvidence;

typedef struct OrbitKvSessionCopyEvidence {
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
} OrbitKvSessionCopyEvidence;

typedef struct OrbitKvSessionStepAbortEvidence {
  uint64_t request_id;
  uint32_t backend_unobserved;
  uint32_t reserved;
} OrbitKvSessionStepAbortEvidence;

typedef struct OrbitKvSessionCompletionEvidence {
  uint64_t completion_domain;
  uint64_t completion_value;
  uint32_t confirmed;
  uint32_t reserved;
} OrbitKvSessionCompletionEvidence;

typedef struct OrbitKvSessionStepPublication {
  uint64_t request_id;
  uint64_t view_version;
  uint64_t boundary;
  uint32_t resident_count;
  uint32_t detached_offset;
  uint32_t detached_count;
  uint32_t reserved;
} OrbitKvSessionStepPublication;

typedef struct OrbitKvSessionRetirement {
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
} OrbitKvSessionRetirement;

typedef struct OrbitKvSessionRetirementEvidence {
  OrbitKvPageLease page;
  uint16_t backend_domain;
  uint8_t acknowledged;
  uint8_t reserved8;
  uint32_t reserved32;
  uint64_t backend_index;
} OrbitKvSessionRetirementEvidence;

typedef struct OrbitKvSessionPublicationEvidence {
  OrbitKvSessionPublicationId publication_id;
  uint32_t mirror_cleanup_confirmed;
  uint32_t reserved;
} OrbitKvSessionPublicationEvidence;

typedef struct OrbitKvSessionReleasedRequest {
  uint64_t request_id;
  uint32_t detached_offset;
  uint32_t detached_count;
} OrbitKvSessionReleasedRequest;

typedef struct OrbitKvSessionReleaseEvidence {
  OrbitKvSessionReleaseId release_id;
  uint32_t mirror_cleanup_confirmed;
  uint32_t reserved;
} OrbitKvSessionReleaseEvidence;

typedef struct OrbitKvSessionReleaseOutcome {
  /* The confirmed release identity, or zeroes when the call returns an error. */
  OrbitKvSessionReleaseId release_id;
  /* One of ORBITKV_SESSION_RELEASE_COMPLETED or
   * ORBITKV_SESSION_RELEASE_RECYCLE_PENDING; zero on error. */
  uint32_t disposition;
  uint32_t reserved;
} OrbitKvSessionReleaseOutcome;

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

ORBITKV_LAYOUT(OrbitKvStateTransitionLease, 16, 8);
ORBITKV_OFFSET(OrbitKvStateTransitionLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvStateTransitionLease, slot, 8);
ORBITKV_OFFSET(OrbitKvStateTransitionLease, generation, 12);
ORBITKV_LAYOUT(OrbitKvStateRetirementLease, 16, 8);
ORBITKV_OFFSET(OrbitKvStateRetirementLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvStateRetirementLease, slot, 8);
ORBITKV_OFFSET(OrbitKvStateRetirementLease, generation, 12);
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
ORBITKV_LAYOUT(OrbitKvManagerConfig, 32, 8);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_requests, 0);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_operations, 4);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_prefixes, 8);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_reclamations, 12);
ORBITKV_OFFSET(OrbitKvManagerConfig, maximum_step_tokens, 16);
ORBITKV_OFFSET(OrbitKvManagerConfig, plan_format, 20);
ORBITKV_OFFSET(OrbitKvManagerConfig, reserved, 24);
ORBITKV_LAYOUT(OrbitKvSessionCreateConfig, 40, 8);
ORBITKV_OFFSET(OrbitKvSessionCreateConfig, manager, 0);
ORBITKV_OFFSET(OrbitKvSessionCreateConfig, cache_sharing_policy, 32);
ORBITKV_OFFSET(OrbitKvSessionCreateConfig, reserved, 36);
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
ORBITKV_LAYOUT(OrbitKvClassLowering, 48, 8);
ORBITKV_OFFSET(OrbitKvClassLowering, class_id, 0);
ORBITKV_OFFSET(OrbitKvClassLowering, flags, 2);
ORBITKV_OFFSET(OrbitKvClassLowering, tail_offset, 4);
ORBITKV_OFFSET(OrbitKvClassLowering, tail_count, 8);
ORBITKV_OFFSET(OrbitKvClassLowering, copy_offset, 12);
ORBITKV_OFFSET(OrbitKvClassLowering, copy_count, 16);
ORBITKV_OFFSET(OrbitKvClassLowering, write_offset, 20);
ORBITKV_OFFSET(OrbitKvClassLowering, write_count, 24);
ORBITKV_OFFSET(OrbitKvClassLowering, reserved, 28);
ORBITKV_OFFSET(OrbitKvClassLowering, previous_layout_boundary, 32);
ORBITKV_OFFSET(OrbitKvClassLowering, target_layout_boundary, 40);
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
ORBITKV_LAYOUT(OrbitKvPrefixSemanticKey, 72, 8);
ORBITKV_OFFSET(OrbitKvPrefixSemanticKey, namespace_bytes, 0);
ORBITKV_OFFSET(OrbitKvPrefixSemanticKey, digest, 32);
ORBITKV_OFFSET(OrbitKvPrefixSemanticKey, boundary, 64);
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
ORBITKV_LAYOUT(OrbitKvTokenDisposition, 32, 8);
ORBITKV_OFFSET(OrbitKvTokenDisposition, policy_or_proof_id, 0);
ORBITKV_OFFSET(OrbitKvTokenDisposition, version, 8);
ORBITKV_OFFSET(OrbitKvTokenDisposition, quality_contract, 16);
ORBITKV_OFFSET(OrbitKvTokenDisposition, kind, 24);
ORBITKV_OFFSET(OrbitKvTokenDisposition, reserved16, 26);
ORBITKV_OFFSET(OrbitKvTokenDisposition, reserved32, 28);
ORBITKV_LAYOUT(OrbitKvTokenLocation, 48, 8);
ORBITKV_OFFSET(OrbitKvTokenLocation, page, 0);
ORBITKV_OFFSET(OrbitKvTokenLocation, backend_index, 32);
ORBITKV_OFFSET(OrbitKvTokenLocation, offset, 40);
ORBITKV_OFFSET(OrbitKvTokenLocation, reserved, 44);
ORBITKV_LAYOUT(OrbitKvTokenPlacement, 96, 8);
ORBITKV_OFFSET(OrbitKvTokenPlacement, token_id, 0);
ORBITKV_OFFSET(OrbitKvTokenPlacement, disposition, 8);
ORBITKV_OFFSET(OrbitKvTokenPlacement, location, 40);
ORBITKV_OFFSET(OrbitKvTokenPlacement, location_present, 88);
ORBITKV_OFFSET(OrbitKvTokenPlacement, reserved, 92);
ORBITKV_LAYOUT(OrbitKvClassTokenDispositionUpdate, 48, 8);
ORBITKV_OFFSET(OrbitKvClassTokenDispositionUpdate, token_id, 0);
ORBITKV_OFFSET(OrbitKvClassTokenDispositionUpdate, disposition, 8);
ORBITKV_OFFSET(OrbitKvClassTokenDispositionUpdate, class_id, 40);
ORBITKV_OFFSET(OrbitKvClassTokenDispositionUpdate, reserved16, 42);
ORBITKV_OFFSET(OrbitKvClassTokenDispositionUpdate, reserved32, 44);
ORBITKV_LAYOUT(OrbitKvRelocationPolicy, 16, 4);
ORBITKV_OFFSET(OrbitKvRelocationPolicy, maximum_source_pages, 0);
ORBITKV_OFFSET(OrbitKvRelocationPolicy, evacuation_headroom_pages, 4);
ORBITKV_OFFSET(OrbitKvRelocationPolicy, fragmentation_threshold_milli, 8);
ORBITKV_OFFSET(OrbitKvRelocationPolicy, full_evacuation, 10);
ORBITKV_OFFSET(OrbitKvRelocationPolicy, reserved8, 11);
ORBITKV_OFFSET(OrbitKvRelocationPolicy, reserved32, 12);
ORBITKV_LAYOUT(OrbitKvTokenMove, 104, 8);
ORBITKV_OFFSET(OrbitKvTokenMove, token_id, 0);
ORBITKV_OFFSET(OrbitKvTokenMove, source, 8);
ORBITKV_OFFSET(OrbitKvTokenMove, destination, 56);
ORBITKV_LAYOUT(OrbitKvStatePoolConfig, 32, 8);
ORBITKV_OFFSET(OrbitKvStatePoolConfig, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvStatePoolConfig, pool_epoch, 8);
ORBITKV_OFFSET(OrbitKvStatePoolConfig, byte_count, 16);
ORBITKV_OFFSET(OrbitKvStatePoolConfig, pool_id, 24);
ORBITKV_OFFSET(OrbitKvStatePoolConfig, slot_count, 28);
ORBITKV_LAYOUT(OrbitKvStateSlotLease, 32, 8);
ORBITKV_OFFSET(OrbitKvStateSlotLease, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvStateSlotLease, pool_epoch, 8);
ORBITKV_OFFSET(OrbitKvStateSlotLease, generation, 16);
ORBITKV_OFFSET(OrbitKvStateSlotLease, slot_id, 24);
ORBITKV_OFFSET(OrbitKvStateSlotLease, pool_id, 28);
ORBITKV_LAYOUT(OrbitKvStatePoolIdentity, 32, 8);
ORBITKV_OFFSET(OrbitKvStatePoolIdentity, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvStatePoolIdentity, pool_epoch, 8);
ORBITKV_OFFSET(OrbitKvStatePoolIdentity, byte_count, 16);
ORBITKV_OFFSET(OrbitKvStatePoolIdentity, pool_id, 24);
ORBITKV_OFFSET(OrbitKvStatePoolIdentity, slot_count, 28);
ORBITKV_LAYOUT(OrbitKvStatePoolStats, 104, 8);
ORBITKV_OFFSET(OrbitKvStatePoolStats, identity, 0);
ORBITKV_OFFSET(OrbitKvStatePoolStats, free_slots, 32);
ORBITKV_OFFSET(OrbitKvStatePoolStats, reserved_slots, 40);
ORBITKV_OFFSET(OrbitKvStatePoolStats, relocating_slots, 48);
ORBITKV_OFFSET(OrbitKvStatePoolStats, live_slots, 56);
ORBITKV_OFFSET(OrbitKvStatePoolStats, retiring_slots, 64);
ORBITKV_OFFSET(OrbitKvStatePoolStats, quarantined_slots, 72);
ORBITKV_OFFSET(OrbitKvStatePoolStats, active_owners, 80);
ORBITKV_OFFSET(OrbitKvStatePoolStats, pending_transitions, 88);
ORBITKV_OFFSET(OrbitKvStatePoolStats, pending_retirements, 96);
ORBITKV_LAYOUT(OrbitKvStatePrepareItem, 48, 8);
ORBITKV_OFFSET(OrbitKvStatePrepareItem, owner_id, 0);
ORBITKV_OFFSET(OrbitKvStatePrepareItem, expected, 8);
ORBITKV_OFFSET(OrbitKvStatePrepareItem, expected_present, 40);
ORBITKV_OFFSET(OrbitKvStatePrepareItem, reserved, 44);
ORBITKV_LAYOUT(OrbitKvStateCopyIntent, 104, 8);
ORBITKV_OFFSET(OrbitKvStateCopyIntent, transition, 0);
ORBITKV_OFFSET(OrbitKvStateCopyIntent, owner_id, 16);
ORBITKV_OFFSET(OrbitKvStateCopyIntent, source, 24);
ORBITKV_OFFSET(OrbitKvStateCopyIntent, destination, 56);
ORBITKV_OFFSET(OrbitKvStateCopyIntent, byte_count, 88);
ORBITKV_OFFSET(OrbitKvStateCopyIntent, source_present, 96);
ORBITKV_OFFSET(OrbitKvStateCopyIntent, reserved, 100);
ORBITKV_LAYOUT(OrbitKvStateCopyReceipt, 96, 8);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, transition, 0);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, source, 16);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, destination, 48);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, byte_count, 80);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, source_present, 88);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, observed, 89);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, written, 90);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, reserved8, 91);
ORBITKV_OFFSET(OrbitKvStateCopyReceipt, reserved32, 92);
ORBITKV_LAYOUT(OrbitKvStateCompletionReceipt, 32, 8);
ORBITKV_OFFSET(OrbitKvStateCompletionReceipt, engine_epoch, 0);
ORBITKV_OFFSET(OrbitKvStateCompletionReceipt, completion_domain, 8);
ORBITKV_OFFSET(OrbitKvStateCompletionReceipt, completion_value, 16);
ORBITKV_OFFSET(OrbitKvStateCompletionReceipt, confirmed, 24);
ORBITKV_OFFSET(OrbitKvStateCompletionReceipt, reserved, 28);
ORBITKV_LAYOUT(OrbitKvStateRetirementCertificate, 72, 8);
ORBITKV_OFFSET(OrbitKvStateRetirementCertificate, retirement, 0);
ORBITKV_OFFSET(OrbitKvStateRetirementCertificate, slot, 16);
ORBITKV_OFFSET(OrbitKvStateRetirementCertificate, byte_count, 48);
ORBITKV_OFFSET(OrbitKvStateRetirementCertificate, completion_domain, 56);
ORBITKV_OFFSET(OrbitKvStateRetirementCertificate, completion_value, 64);
ORBITKV_LAYOUT(OrbitKvStatePublication, 120, 8);
ORBITKV_OFFSET(OrbitKvStatePublication, owner_id, 0);
ORBITKV_OFFSET(OrbitKvStatePublication, slot, 8);
ORBITKV_OFFSET(OrbitKvStatePublication, retirement, 40);
ORBITKV_OFFSET(OrbitKvStatePublication, retirement_present, 112);
ORBITKV_OFFSET(OrbitKvStatePublication, reserved, 116);
ORBITKV_LAYOUT(OrbitKvStateAbortItem, 24, 8);
ORBITKV_OFFSET(OrbitKvStateAbortItem, transition, 0);
ORBITKV_OFFSET(OrbitKvStateAbortItem, backend_unobserved, 16);
ORBITKV_OFFSET(OrbitKvStateAbortItem, reserved, 20);
ORBITKV_LAYOUT(OrbitKvStateRetireOwnerItem, 40, 8);
ORBITKV_OFFSET(OrbitKvStateRetireOwnerItem, owner_id, 0);
ORBITKV_OFFSET(OrbitKvStateRetireOwnerItem, expected, 8);
ORBITKV_LAYOUT(OrbitKvStateCurrent, 48, 8);
ORBITKV_OFFSET(OrbitKvStateCurrent, owner_id, 0);
ORBITKV_OFFSET(OrbitKvStateCurrent, slot, 8);
ORBITKV_OFFSET(OrbitKvStateCurrent, present, 40);
ORBITKV_OFFSET(OrbitKvStateCurrent, reserved, 44);
ORBITKV_LAYOUT(OrbitKvSessionBatchId, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionBatchId, session_epoch, 0);
ORBITKV_OFFSET(OrbitKvSessionBatchId, sequence, 8);
ORBITKV_LAYOUT(OrbitKvSessionPublicationId, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionPublicationId, session_epoch, 0);
ORBITKV_OFFSET(OrbitKvSessionPublicationId, sequence, 8);
ORBITKV_LAYOUT(OrbitKvSessionReleaseId, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionReleaseId, session_epoch, 0);
ORBITKV_OFFSET(OrbitKvSessionReleaseId, sequence, 8);
ORBITKV_LAYOUT(OrbitKvSessionPrefixId, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionPrefixId, session_epoch, 0);
ORBITKV_OFFSET(OrbitKvSessionPrefixId, sequence, 8);
ORBITKV_LAYOUT(OrbitKvSessionControlId, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionControlId, session_epoch, 0);
ORBITKV_OFFSET(OrbitKvSessionControlId, sequence, 8);
ORBITKV_LAYOUT(OrbitKvSessionRelocationId, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationId, session_epoch, 0);
ORBITKV_OFFSET(OrbitKvSessionRelocationId, sequence, 8);
ORBITKV_LAYOUT(OrbitKvSessionPrefixLookup, 104, 8);
ORBITKV_OFFSET(OrbitKvSessionPrefixLookup, key, 0);
ORBITKV_OFFSET(OrbitKvSessionPrefixLookup, candidate, 72);
ORBITKV_OFFSET(OrbitKvSessionPrefixLookup, resident_count, 88);
ORBITKV_OFFSET(OrbitKvSessionPrefixLookup, candidate_present, 92);
ORBITKV_OFFSET(OrbitKvSessionPrefixLookup, reserved0, 96);
ORBITKV_OFFSET(OrbitKvSessionPrefixLookup, reserved1, 100);
ORBITKV_LAYOUT(OrbitKvSessionPrefixPublishItem, 80, 8);
ORBITKV_OFFSET(OrbitKvSessionPrefixPublishItem, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPrefixPublishItem, key, 8);
ORBITKV_LAYOUT(OrbitKvSessionPublishedPrefix, 96, 8);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefix, prefix_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefix, key, 16);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefix, resident_count, 88);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefix, reserved, 92);
ORBITKV_LAYOUT(OrbitKvSessionPublishedPrefixRelease, 112, 8);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefixRelease, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefixRelease, prefix_id, 8);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefixRelease, key, 24);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefixRelease, resident_count, 96);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefixRelease, detached_offset, 100);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefixRelease, detached_count, 104);
ORBITKV_OFFSET(OrbitKvSessionPublishedPrefixRelease, reserved, 108);
ORBITKV_LAYOUT(OrbitKvSessionPrefixAttachItem, 104, 8);
ORBITKV_OFFSET(OrbitKvSessionPrefixAttachItem, target_request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPrefixAttachItem, prefix_id, 8);
ORBITKV_OFFSET(OrbitKvSessionPrefixAttachItem, key, 24);
ORBITKV_OFFSET(OrbitKvSessionPrefixAttachItem, resident_count, 96);
ORBITKV_OFFSET(OrbitKvSessionPrefixAttachItem, reserved, 100);
ORBITKV_LAYOUT(OrbitKvSessionRequestForkItem, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionRequestForkItem, source_request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionRequestForkItem, target_request_id, 8);
ORBITKV_LAYOUT(OrbitKvSessionControlPlanInfo, 40, 8);
ORBITKV_OFFSET(OrbitKvSessionControlPlanInfo, id, 0);
ORBITKV_OFFSET(OrbitKvSessionControlPlanInfo, kind, 16);
ORBITKV_OFFSET(OrbitKvSessionControlPlanInfo, reserved, 20);
ORBITKV_OFFSET(OrbitKvSessionControlPlanInfo, request_count, 24);
ORBITKV_OFFSET(OrbitKvSessionControlPlanInfo, page_count, 28);
ORBITKV_OFFSET(OrbitKvSessionControlPlanInfo, prefix_count, 32);
ORBITKV_OFFSET(OrbitKvSessionControlPlanInfo, retirement_count, 36);
ORBITKV_LAYOUT(OrbitKvSessionMaterializedRequest, 40, 8);
ORBITKV_OFFSET(OrbitKvSessionMaterializedRequest, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionMaterializedRequest, view_version, 8);
ORBITKV_OFFSET(OrbitKvSessionMaterializedRequest, boundary, 16);
ORBITKV_OFFSET(OrbitKvSessionMaterializedRequest, resident_count, 24);
ORBITKV_OFFSET(OrbitKvSessionMaterializedRequest, page_offset, 28);
ORBITKV_OFFSET(OrbitKvSessionMaterializedRequest, page_count, 32);
ORBITKV_OFFSET(OrbitKvSessionMaterializedRequest, reserved, 36);
ORBITKV_LAYOUT(OrbitKvSessionPendingAttachCancel, 64, 8);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancel, control_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancel, request_id, 16);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancel, prefix_id, 24);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancel, view_version, 40);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancel, boundary, 48);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancel, resident_count, 56);
ORBITKV_LAYOUT(OrbitKvSessionPendingAttachCancelOutcome, 64, 8);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancelOutcome, control_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancelOutcome, request_id, 16);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancelOutcome, prefix_id, 24);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancelOutcome, view_version, 40);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancelOutcome, boundary, 48);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancelOutcome, resident_count, 56);
ORBITKV_OFFSET(OrbitKvSessionPendingAttachCancelOutcome, disposition, 60);
ORBITKV_LAYOUT(OrbitKvSessionControlEvidence, 24, 8);
ORBITKV_OFFSET(OrbitKvSessionControlEvidence, id, 0);
ORBITKV_OFFSET(OrbitKvSessionControlEvidence, mirror_updates_confirmed, 16);
ORBITKV_OFFSET(OrbitKvSessionControlEvidence, reserved, 20);
ORBITKV_LAYOUT(OrbitKvSessionControlOutcome, 24, 8);
ORBITKV_OFFSET(OrbitKvSessionControlOutcome, id, 0);
ORBITKV_OFFSET(OrbitKvSessionControlOutcome, disposition, 16);
ORBITKV_OFFSET(OrbitKvSessionControlOutcome, reserved, 20);
ORBITKV_LAYOUT(OrbitKvSessionRequestView, 32, 8);
ORBITKV_OFFSET(OrbitKvSessionRequestView, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionRequestView, view_version, 8);
ORBITKV_OFFSET(OrbitKvSessionRequestView, boundary, 16);
ORBITKV_OFFSET(OrbitKvSessionRequestView, resident_count, 24);
ORBITKV_OFFSET(OrbitKvSessionRequestView, reserved, 28);
ORBITKV_LAYOUT(OrbitKvSessionTokenViewQuery, 24, 8);
ORBITKV_OFFSET(OrbitKvSessionTokenViewQuery, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionTokenViewQuery, expected_boundary, 8);
ORBITKV_OFFSET(OrbitKvSessionTokenViewQuery, class_id, 16);
ORBITKV_OFFSET(OrbitKvSessionTokenViewQuery, reserved16, 18);
ORBITKV_OFFSET(OrbitKvSessionTokenViewQuery, reserved32, 20);
ORBITKV_LAYOUT(OrbitKvSessionTokenView, 40, 8);
ORBITKV_OFFSET(OrbitKvSessionTokenView, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionTokenView, view_version, 8);
ORBITKV_OFFSET(OrbitKvSessionTokenView, placement_offset, 16);
ORBITKV_OFFSET(OrbitKvSessionTokenView, placement_count, 20);
ORBITKV_OFFSET(OrbitKvSessionTokenView, page_tokens, 24);
ORBITKV_OFFSET(OrbitKvSessionTokenView, class_id, 28);
ORBITKV_OFFSET(OrbitKvSessionTokenView, reserved16, 30);
ORBITKV_OFFSET(OrbitKvSessionTokenView, reserved32, 32);
ORBITKV_LAYOUT(OrbitKvSessionTokenDispositionBatchItem, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionTokenDispositionBatchItem, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionTokenDispositionBatchItem, update_offset, 8);
ORBITKV_OFFSET(OrbitKvSessionTokenDispositionBatchItem, update_count, 12);
ORBITKV_LAYOUT(OrbitKvSessionPrepareRelocationItem, 32, 8);
ORBITKV_OFFSET(OrbitKvSessionPrepareRelocationItem, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPrepareRelocationItem, policy, 8);
ORBITKV_OFFSET(OrbitKvSessionPrepareRelocationItem, class_id, 24);
ORBITKV_OFFSET(OrbitKvSessionPrepareRelocationItem, reserved16, 26);
ORBITKV_OFFSET(OrbitKvSessionPrepareRelocationItem, reserved32, 28);
ORBITKV_LAYOUT(OrbitKvSessionRelocationPlan, 64, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, base_view_version, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, target_view_version, 16);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, source_offset, 24);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, source_count, 28);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, destination_offset, 32);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, destination_count, 36);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, move_offset, 40);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, move_count, 44);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, projected_reclaimed_pages, 48);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, fragmentation_milli, 52);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, class_id, 54);
ORBITKV_OFFSET(OrbitKvSessionRelocationPlan, reserved32, 56);
ORBITKV_LAYOUT(OrbitKvSessionRelocationRequestEvidence, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationRequestEvidence, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionRelocationRequestEvidence, copy_offset, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationRequestEvidence, copy_count, 12);
ORBITKV_LAYOUT(OrbitKvSessionRelocationCopyEvidence, 112, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationCopyEvidence, token_id, 0);
ORBITKV_OFFSET(OrbitKvSessionRelocationCopyEvidence, source, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationCopyEvidence, destination, 56);
ORBITKV_OFFSET(OrbitKvSessionRelocationCopyEvidence, observed, 104);
ORBITKV_OFFSET(OrbitKvSessionRelocationCopyEvidence, copied, 105);
ORBITKV_OFFSET(OrbitKvSessionRelocationCopyEvidence, reserved16, 106);
ORBITKV_OFFSET(OrbitKvSessionRelocationCopyEvidence, reserved32, 108);
ORBITKV_LAYOUT(OrbitKvSessionRelocationAbortEvidence, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationAbortEvidence, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionRelocationAbortEvidence, backend_unobserved, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationAbortEvidence, reserved, 12);
ORBITKV_LAYOUT(OrbitKvSessionRelocationRequestPublication, 32, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationRequestPublication, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionRelocationRequestPublication, view_version, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationRequestPublication, boundary, 16);
ORBITKV_OFFSET(OrbitKvSessionRelocationRequestPublication, resident_count, 24);
ORBITKV_OFFSET(OrbitKvSessionRelocationRequestPublication, reserved, 28);
ORBITKV_LAYOUT(OrbitKvSessionRelocationPublicationEvidence, 24, 8);
ORBITKV_OFFSET(OrbitKvSessionRelocationPublicationEvidence, relocation_id, 0);
ORBITKV_OFFSET(OrbitKvSessionRelocationPublicationEvidence, mirror_cleanup_confirmed, 16);
ORBITKV_OFFSET(OrbitKvSessionRelocationPublicationEvidence, reserved, 20);
ORBITKV_LAYOUT(OrbitKvSessionAppendIntent, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionAppendIntent, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionAppendIntent, target_boundary, 8);
ORBITKV_LAYOUT(OrbitKvSessionPreparedStep, 72, 8);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, base_view_version, 8);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, target_view_version, 16);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, previous_boundary, 24);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, target_boundary, 32);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, class_offset, 40);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, class_count, 44);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, tail_offset, 48);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, tail_count, 52);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, copy_offset, 56);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, copy_count, 60);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, write_offset, 64);
ORBITKV_OFFSET(OrbitKvSessionPreparedStep, write_count, 68);
ORBITKV_LAYOUT(OrbitKvSessionStepExecutionEvidence, 32, 8);
ORBITKV_OFFSET(OrbitKvSessionStepExecutionEvidence, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionStepExecutionEvidence, bind_offset, 8);
ORBITKV_OFFSET(OrbitKvSessionStepExecutionEvidence, bind_count, 12);
ORBITKV_OFFSET(OrbitKvSessionStepExecutionEvidence, copy_offset, 16);
ORBITKV_OFFSET(OrbitKvSessionStepExecutionEvidence, copy_count, 20);
ORBITKV_OFFSET(OrbitKvSessionStepExecutionEvidence, reserved, 24);
ORBITKV_LAYOUT(OrbitKvSessionBindEvidence, 48, 8);
ORBITKV_OFFSET(OrbitKvSessionBindEvidence, page, 0);
ORBITKV_OFFSET(OrbitKvSessionBindEvidence, backend_domain, 32);
ORBITKV_OFFSET(OrbitKvSessionBindEvidence, mapped, 34);
ORBITKV_OFFSET(OrbitKvSessionBindEvidence, writable, 35);
ORBITKV_OFFSET(OrbitKvSessionBindEvidence, reserved, 36);
ORBITKV_OFFSET(OrbitKvSessionBindEvidence, backend_index, 40);
ORBITKV_LAYOUT(OrbitKvSessionCopyEvidence, 104, 8);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, class_id, 0);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, backend_domain, 2);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, token_count, 4);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, source_token_offset, 8);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, destination_token_offset, 12);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, observed, 16);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, copied, 17);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, ordered_before_writes, 18);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, reserved8, 19);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, reserved32, 20);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, source, 24);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, destination, 56);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, source_backend_index, 88);
ORBITKV_OFFSET(OrbitKvSessionCopyEvidence, destination_backend_index, 96);
ORBITKV_LAYOUT(OrbitKvSessionStepAbortEvidence, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionStepAbortEvidence, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionStepAbortEvidence, backend_unobserved, 8);
ORBITKV_OFFSET(OrbitKvSessionStepAbortEvidence, reserved, 12);
ORBITKV_LAYOUT(OrbitKvSessionCompletionEvidence, 24, 8);
ORBITKV_OFFSET(OrbitKvSessionCompletionEvidence, completion_domain, 0);
ORBITKV_OFFSET(OrbitKvSessionCompletionEvidence, completion_value, 8);
ORBITKV_OFFSET(OrbitKvSessionCompletionEvidence, confirmed, 16);
ORBITKV_OFFSET(OrbitKvSessionCompletionEvidence, reserved, 20);
ORBITKV_LAYOUT(OrbitKvSessionStepPublication, 40, 8);
ORBITKV_OFFSET(OrbitKvSessionStepPublication, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionStepPublication, view_version, 8);
ORBITKV_OFFSET(OrbitKvSessionStepPublication, boundary, 16);
ORBITKV_OFFSET(OrbitKvSessionStepPublication, resident_count, 24);
ORBITKV_OFFSET(OrbitKvSessionStepPublication, detached_offset, 28);
ORBITKV_OFFSET(OrbitKvSessionStepPublication, detached_count, 32);
ORBITKV_OFFSET(OrbitKvSessionStepPublication, reserved, 36);
ORBITKV_LAYOUT(OrbitKvSessionRetirement, 88, 8);
ORBITKV_OFFSET(OrbitKvSessionRetirement, page, 0);
ORBITKV_OFFSET(OrbitKvSessionRetirement, class_id, 32);
ORBITKV_OFFSET(OrbitKvSessionRetirement, backend_domain, 34);
ORBITKV_OFFSET(OrbitKvSessionRetirement, reserved32, 36);
ORBITKV_OFFSET(OrbitKvSessionRetirement, logical_ordinal, 40);
ORBITKV_OFFSET(OrbitKvSessionRetirement, backend_index, 48);
ORBITKV_OFFSET(OrbitKvSessionRetirement, token_begin, 56);
ORBITKV_OFFSET(OrbitKvSessionRetirement, token_end_exclusive, 64);
ORBITKV_OFFSET(OrbitKvSessionRetirement, completion_domain, 72);
ORBITKV_OFFSET(OrbitKvSessionRetirement, completion_value, 80);
ORBITKV_LAYOUT(OrbitKvSessionRetirementEvidence, 48, 8);
ORBITKV_OFFSET(OrbitKvSessionRetirementEvidence, page, 0);
ORBITKV_OFFSET(OrbitKvSessionRetirementEvidence, backend_domain, 32);
ORBITKV_OFFSET(OrbitKvSessionRetirementEvidence, acknowledged, 34);
ORBITKV_OFFSET(OrbitKvSessionRetirementEvidence, reserved8, 35);
ORBITKV_OFFSET(OrbitKvSessionRetirementEvidence, reserved32, 36);
ORBITKV_OFFSET(OrbitKvSessionRetirementEvidence, backend_index, 40);
ORBITKV_LAYOUT(OrbitKvSessionPublicationEvidence, 24, 8);
ORBITKV_OFFSET(OrbitKvSessionPublicationEvidence, publication_id, 0);
ORBITKV_OFFSET(OrbitKvSessionPublicationEvidence, mirror_cleanup_confirmed, 16);
ORBITKV_OFFSET(OrbitKvSessionPublicationEvidence, reserved, 20);
ORBITKV_LAYOUT(OrbitKvSessionReleasedRequest, 16, 8);
ORBITKV_OFFSET(OrbitKvSessionReleasedRequest, request_id, 0);
ORBITKV_OFFSET(OrbitKvSessionReleasedRequest, detached_offset, 8);
ORBITKV_OFFSET(OrbitKvSessionReleasedRequest, detached_count, 12);
ORBITKV_LAYOUT(OrbitKvSessionReleaseEvidence, 24, 8);
ORBITKV_OFFSET(OrbitKvSessionReleaseEvidence, release_id, 0);
ORBITKV_OFFSET(OrbitKvSessionReleaseEvidence, mirror_cleanup_confirmed, 16);
ORBITKV_OFFSET(OrbitKvSessionReleaseEvidence, reserved, 20);
ORBITKV_LAYOUT(OrbitKvSessionReleaseOutcome, 24, 8);
ORBITKV_OFFSET(OrbitKvSessionReleaseOutcome, release_id, 0);
ORBITKV_OFFSET(OrbitKvSessionReleaseOutcome, disposition, 16);
ORBITKV_OFFSET(OrbitKvSessionReleaseOutcome, reserved, 20);

#undef ORBITKV_OFFSET
#undef ORBITKV_LAYOUT
#undef ORBITKV_ALIGNOF
#undef ORBITKV_STATIC_ASSERT

uint32_t orbitkv_wire_version(void);

/*
 * RuntimeSession is the engine-facing lifecycle owner. Every returned batch,
 * publication, and release id is bound to the session that minted it. Flat
 * spans are canonical and gap-free. Mutation output capacities are checked
 * against configured bounds before core state changes; BUFFER_TOO_SMALL is
 * non-mutating. The retirement DTOs omit private reclamation leases.
 * config->manager.plan_format selects the exact JSON input type: KV_PLAN
 * accepts a KvPlanInput document and RETENTION_IR accepts a
 * RetentionProgramInput document. The implementation never infers the format
 * from JSON fields.
 */
int32_t orbitkv_session_create(
    const uint8_t *plan_json, size_t plan_json_len,
    const OrbitKvSessionCreateConfig *config,
    const OrbitKvBackendArenaRegistration *backends, uint32_t backend_count,
    OrbitKvSessionHandle **out_session, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_arena_identities(
    OrbitKvSessionHandle *session, OrbitKvArenaIdentity *identities,
    uint32_t identity_capacity, uint32_t *out_identity_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_arena_stats(
    OrbitKvSessionHandle *session, OrbitKvArenaStats *stats,
    uint32_t stats_capacity, uint32_t *out_stats_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_stats(
    OrbitKvSessionHandle *session, OrbitKvManagerStats *out_stats,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_acquire_requests(
    OrbitKvSessionHandle *session, const uint64_t *request_ids,
    uint32_t request_count, OrbitKvSessionRequestView *views,
    uint32_t view_capacity, uint32_t *out_view_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_prefix_lookup_batch(
    OrbitKvSessionHandle *session, const OrbitKvPrefixSemanticKey *keys,
    uint32_t key_count, OrbitKvSessionPrefixLookup *lookups,
    uint32_t lookup_capacity, uint32_t *out_lookup_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_prefix_publish_batch(
    OrbitKvSessionHandle *session,
    const OrbitKvSessionPrefixPublishItem *items, uint32_t item_count,
    OrbitKvSessionPublishedPrefix *published, uint32_t published_capacity,
    uint32_t *out_published_count, char *error_buffer,
    size_t error_buffer_len);

/*
 * Atomically transfers each request's page references to a new prefix and
 * begins one release. The operation never returns retirement certificates;
 * complete it with orbitkv_session_confirm_release and an empty retirement
 * evidence array. Detached spans are canonical and gap-free.
 */
int32_t orbitkv_session_prefix_publish_release_batch(
    OrbitKvSessionHandle *session,
    const OrbitKvSessionPrefixPublishItem *items, uint32_t item_count,
    OrbitKvSessionReleaseId *out_release_id,
    OrbitKvSessionPublishedPrefixRelease *outputs, uint32_t output_capacity,
    uint32_t *out_output_count, OrbitKvDetachedBinding *detached,
    uint32_t detached_capacity, uint32_t *out_detached_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_prepare_prefix_attach(
    OrbitKvSessionHandle *session,
    const OrbitKvSessionPrefixAttachItem *items, uint32_t item_count,
    OrbitKvSessionControlId *out_control_id, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_prepare_request_fork(
    OrbitKvSessionHandle *session,
    const OrbitKvSessionRequestForkItem *items, uint32_t item_count,
    OrbitKvSessionControlId *out_control_id, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_prepare_prefix_evict(
    OrbitKvSessionHandle *session, const OrbitKvSessionPrefixId *prefixes,
    uint32_t prefix_count, OrbitKvSessionControlId *out_control_id,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_abort_control(
    OrbitKvSessionHandle *session, OrbitKvSessionControlId control_id,
    char *error_buffer, size_t error_buffer_len);

/* Commits the prepared control and returns its fixed plan summary. */
int32_t orbitkv_session_commit_control(
    OrbitKvSessionHandle *session, OrbitKvSessionControlId control_id,
    OrbitKvSessionControlPlanInfo *out_plan, char *error_buffer,
    size_t error_buffer_len);

/*
 * Pure read of a committed plan. Passing NULL/0 for all four buffers queries
 * exact counts. Required counts are written during preflight; all four
 * capacities are checked before any element-buffer write.
 */
int32_t orbitkv_session_read_control_plan(
    OrbitKvSessionHandle *session, OrbitKvSessionControlId control_id,
    OrbitKvSessionMaterializedRequest *requests, uint32_t request_capacity,
    uint32_t *out_request_count, OrbitKvSnapshotPage *pages,
    uint32_t page_capacity, uint32_t *out_page_count,
    OrbitKvSessionPrefixId *prefixes, uint32_t prefix_capacity,
    uint32_t *out_prefix_count, OrbitKvSessionRetirement *retirements,
    uint32_t retirement_capacity, uint32_t *out_retirement_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_cancel_pending_attach(
    OrbitKvSessionHandle *session, OrbitKvSessionPendingAttachCancel expected,
    OrbitKvSessionPendingAttachCancelOutcome *out_outcome, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_finalize_pending_attach_cancel(
    OrbitKvSessionHandle *session, OrbitKvSessionPendingAttachCancel expected,
    OrbitKvSessionPendingAttachCancelOutcome *out_outcome, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_confirm_control(
    OrbitKvSessionHandle *session, OrbitKvSessionControlEvidence evidence,
    const OrbitKvSessionRetirementEvidence *retirements,
    uint32_t retirement_count, OrbitKvSessionControlOutcome *out_outcome,
    char *error_buffer, size_t error_buffer_len);

/* Local containment: successful control quarantine returns OK. */
int32_t orbitkv_session_quarantine_control(
    OrbitKvSessionHandle *session, OrbitKvSessionControlId control_id,
    char *error_buffer, size_t error_buffer_len);

/*
 * Cold token-view read. expected_boundary is checked against the current
 * session-owned request head before materialization. Passing NULL/0 for both
 * output buffers queries exact counts without mutating session state.
 */
int32_t orbitkv_session_token_views_batch(
    OrbitKvSessionHandle *session,
    const OrbitKvSessionTokenViewQuery *queries, uint32_t query_count,
    OrbitKvSessionTokenView *views, uint32_t view_capacity,
    uint32_t *out_view_count, OrbitKvTokenPlacement *placements,
    uint32_t placement_capacity, uint32_t *out_placement_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_mark_token_dispositions_batch(
    OrbitKvSessionHandle *session,
    const OrbitKvSessionTokenDispositionBatchItem *items, uint32_t item_count,
    const OrbitKvClassTokenDispositionUpdate *updates, uint32_t update_count,
    OrbitKvSessionRequestView *outputs, uint32_t output_capacity,
    uint32_t *out_output_count, char *error_buffer, size_t error_buffer_len);

/*
 * Prepare is mutating and therefore uses conservative preflight bounds rather
 * than an exact two-call sizing pass. A short buffer returns BUFFER_TOO_SMALL,
 * leaves the session unchanged, and writes a zero relocation id. The returned
 * opaque id replaces private canonical request, snapshot, and relocation
 * authority.
 */
int32_t orbitkv_session_prepare_relocation_batch(
    OrbitKvSessionHandle *session,
    const OrbitKvSessionPrepareRelocationItem *items, uint32_t item_count,
    OrbitKvSessionRelocationId *out_relocation_id,
    OrbitKvSessionRelocationPlan *plans, uint32_t plan_capacity,
    uint32_t *out_plan_count, OrbitKvPageLease *sources,
    uint32_t source_capacity, uint32_t *out_source_count,
    OrbitKvPageLease *destinations, uint32_t destination_capacity,
    uint32_t *out_destination_count, OrbitKvTokenMove *moves,
    uint32_t move_capacity, uint32_t *out_move_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_abort_prepared_relocation(
    OrbitKvSessionHandle *session, OrbitKvSessionRelocationId relocation_id,
    const OrbitKvSessionRelocationAbortEvidence *evidence,
    uint32_t evidence_count, char *error_buffer, size_t error_buffer_len);

/* Successful explicit quarantine returns FAIL_STOPPED, never OK. */
int32_t orbitkv_session_quarantine_relocation(
    OrbitKvSessionHandle *session, OrbitKvSessionRelocationId relocation_id,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_submit_relocation(
    OrbitKvSessionHandle *session, OrbitKvSessionRelocationId relocation_id,
    const OrbitKvSessionRelocationRequestEvidence *requests,
    uint32_t request_count,
    const OrbitKvSessionRelocationCopyEvidence *copies, uint32_t copy_count,
    char *error_buffer, size_t error_buffer_len);

/* completion must carry a confirmed, positive, advancing GPU frontier. */
int32_t orbitkv_session_complete_relocation(
    OrbitKvSessionHandle *session, OrbitKvSessionRelocationId relocation_id,
    OrbitKvSessionCompletionEvidence completion,
    OrbitKvSessionRelocationRequestPublication *publications,
    uint32_t publication_capacity, uint32_t *out_publication_count,
    OrbitKvSessionRetirement *retirements, uint32_t retirement_capacity,
    uint32_t *out_retirement_count, char *error_buffer,
    size_t error_buffer_len);

/* Exact ordered retirement evidence is required before page reuse. */
int32_t orbitkv_session_confirm_relocation_publication(
    OrbitKvSessionHandle *session,
    OrbitKvSessionRelocationPublicationEvidence evidence,
    const OrbitKvSessionRetirementEvidence *retirements,
    uint32_t retirement_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_prepare_append(
    OrbitKvSessionHandle *session, const OrbitKvSessionAppendIntent *intents,
    uint32_t intent_count, OrbitKvSessionBatchId *out_batch_id,
    OrbitKvSessionPreparedStep *steps, uint32_t step_capacity,
    uint32_t *out_step_count, OrbitKvClassLowering *class_lowerings,
    uint32_t class_capacity, uint32_t *out_class_count,
    OrbitKvTailAction *tail_actions, uint32_t tail_capacity,
    uint32_t *out_tail_count, OrbitKvCopyIntent *copy_intents,
    uint32_t copy_capacity, uint32_t *out_copy_count,
    OrbitKvWriteIntent *write_intents, uint32_t write_capacity,
    uint32_t *out_write_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_submit_execution(
    OrbitKvSessionHandle *session, OrbitKvSessionBatchId batch_id,
    const OrbitKvSessionStepExecutionEvidence *steps, uint32_t step_count,
    const OrbitKvSessionBindEvidence *binds, uint32_t bind_count,
    const OrbitKvSessionCopyEvidence *copies, uint32_t copy_count,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_abort_prepared(
    OrbitKvSessionHandle *session, OrbitKvSessionBatchId batch_id,
    const OrbitKvSessionStepAbortEvidence *evidence, uint32_t evidence_count,
    char *error_buffer, size_t error_buffer_len);

/* Successful explicit quarantine returns FAIL_STOPPED, never OK. */
int32_t orbitkv_session_quarantine_prepared(
    OrbitKvSessionHandle *session, OrbitKvSessionBatchId batch_id,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_quarantine_submitted(
    OrbitKvSessionHandle *session, OrbitKvSessionBatchId batch_id,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_complete_execution(
    OrbitKvSessionHandle *session, OrbitKvSessionBatchId batch_id,
    OrbitKvSessionCompletionEvidence evidence,
    OrbitKvSessionPublicationId *out_publication_id,
    OrbitKvSessionStepPublication *steps, uint32_t step_capacity,
    uint32_t *out_step_count, OrbitKvDetachedBinding *detached,
    uint32_t detached_capacity, uint32_t *out_detached_count,
    OrbitKvSessionRetirement *retirements, uint32_t retirement_capacity,
    uint32_t *out_retirement_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_confirm_publication(
    OrbitKvSessionHandle *session, OrbitKvSessionPublicationEvidence evidence,
    const OrbitKvSessionRetirementEvidence *retirements,
    uint32_t retirement_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_session_prepare_release(
    OrbitKvSessionHandle *session, const uint64_t *request_ids,
    uint32_t request_count, OrbitKvSessionReleaseId *out_release_id,
    OrbitKvSessionReleasedRequest *releases, uint32_t release_capacity,
    uint32_t *out_release_count, OrbitKvDetachedBinding *detached,
    uint32_t detached_capacity, uint32_t *out_detached_count,
    OrbitKvSessionRetirement *retirements, uint32_t retirement_capacity,
    uint32_t *out_retirement_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_session_confirm_release(
    OrbitKvSessionHandle *session, OrbitKvSessionReleaseEvidence evidence,
    const OrbitKvSessionRetirementEvidence *retirements,
    uint32_t retirement_count, OrbitKvSessionReleaseOutcome *out_outcome,
    char *error_buffer, size_t error_buffer_len);

/*
 * Exclusively destroys the handle and discards pending batch, publication,
 * release, and control authority. No abort, quarantine, ACK, or recycle work
 * is implied. This remains available for fail-stopped sessions.
 */
int32_t orbitkv_session_destroy(
    OrbitKvSessionHandle *session, char *error_buffer,
    size_t error_buffer_len);

/*
 * The wire fixed-state pool uses request-level recurrent/convolution slots
 * and deliberately exposes no token ids or TokenMove surface.
 */
int32_t orbitkv_state_pool_create(
    const OrbitKvStatePoolConfig *config, OrbitKvStatePoolHandle **out_handle,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_state_pool_identity(
    OrbitKvStatePoolHandle *handle, OrbitKvStatePoolIdentity *output,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_state_pool_stats(
    OrbitKvStatePoolHandle *handle, OrbitKvStatePoolStats *output,
    char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_state_pool_prepare_batch(
    OrbitKvStatePoolHandle *handle, const OrbitKvStatePrepareItem *items,
    uint32_t item_count, OrbitKvStateCopyIntent *outputs,
    uint32_t output_capacity, uint32_t *out_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_state_pool_submit_batch(
    OrbitKvStatePoolHandle *handle, const OrbitKvStateCopyReceipt *receipts,
    uint32_t receipt_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_state_pool_complete_batch(
    OrbitKvStatePoolHandle *handle,
    const OrbitKvStateCompletionReceipt *completion,
    const OrbitKvStateTransitionLease *transitions, uint32_t transition_count,
    OrbitKvStatePublication *outputs, uint32_t output_capacity,
    uint32_t *out_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_state_pool_abort_batch(
    OrbitKvStatePoolHandle *handle, const OrbitKvStateAbortItem *items,
    uint32_t item_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_state_pool_retire_owners_batch(
    OrbitKvStatePoolHandle *handle,
    const OrbitKvStateCompletionReceipt *completion,
    const OrbitKvStateRetireOwnerItem *items, uint32_t item_count,
    OrbitKvStateRetirementCertificate *outputs, uint32_t output_capacity,
    uint32_t *out_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_state_pool_acknowledge_batch(
    OrbitKvStatePoolHandle *handle,
    const OrbitKvStateRetirementCertificate *certificates,
    uint32_t certificate_count, char *error_buffer, size_t error_buffer_len);

int32_t orbitkv_state_pool_current_batch(
    OrbitKvStatePoolHandle *handle, const uint64_t *owner_ids,
    uint32_t owner_count, OrbitKvStateCurrent *outputs,
    uint32_t output_capacity, uint32_t *out_count, char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_state_pool_destroy(
    OrbitKvStatePoolHandle *handle, char *error_buffer,
    size_t error_buffer_len);

#ifdef __cplusplus
}
#endif

#endif
