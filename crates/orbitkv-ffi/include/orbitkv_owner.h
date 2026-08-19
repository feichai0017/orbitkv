#ifndef ORBITKV_OWNER_H
#define ORBITKV_OWNER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORBITKV_OWNER_ABI_VERSION 1u
#define ORBITKV_DENSE_ABI_VERSION 1u
#define ORBITKV_STATUS_OK 0
#define ORBITKV_STATUS_NO_CERTIFICATE 1
#define ORBITKV_DENSE_STATUS_BUFFER_TOO_SMALL 2
#define ORBITKV_STATUS_INVALID_ARGUMENT -1
#define ORBITKV_STATUS_OWNER_ERROR -2
#define ORBITKV_STATUS_PANIC -3

typedef struct OrbitKvOwnerHandle OrbitKvOwnerHandle;
typedef struct OrbitKvDenseHandle OrbitKvDenseHandle;

typedef struct OrbitKvDenseRequestLeaseV1 {
  uint32_t slot;
  uint32_t generation;
} OrbitKvDenseRequestLeaseV1;

typedef struct OrbitKvDenseCertificateV1 {
  uint32_t abi_version;
  uint16_t class_id;
  uint16_t reserved;
  uint64_t certificate_id;
  uint64_t ordinal;
  uint64_t physical_slot;
  uint64_t physical_generation;
  uint64_t backend_index;
  uint64_t token_start;
  uint64_t token_end_exclusive;
} OrbitKvDenseCertificateV1;

typedef struct OrbitKvCertificateV1 {
  uint32_t abi_version;
  uint32_t reserved;
  uint64_t certificate_id;
  uint64_t page_tokens;
  uint64_t token_start;
  uint64_t token_end_exclusive;
  uint64_t semantic_frontier;
  uint64_t window_tokens;
  uint64_t maximum_reclaimable_end;
  uint64_t execution_epoch;
  uint8_t plan_fingerprint[32];
} OrbitKvCertificateV1;

typedef struct OrbitKvOwnerStatsV1 {
  uint32_t abi_version;
  uint32_t reserved;
  uint64_t tracked_requests;
  uint64_t pending_certificates;
  uint64_t committed_reclamations;
  uint64_t committed_tokens;
  uint8_t plan_fingerprint[32];
} OrbitKvOwnerStatsV1;

uint32_t orbitkv_owner_abi_version(void);

int32_t orbitkv_owner_create(const uint8_t *plan_json,
                             size_t plan_json_len,
                             OrbitKvOwnerHandle **out_owner,
                             char *error_buffer,
                             size_t error_buffer_len);

int32_t orbitkv_owner_plan_chunk_reclamation(
    OrbitKvOwnerHandle *owner,
    const uint8_t *request_id,
    size_t request_id_len,
    uint64_t observed_evicted_seqlen,
    uint64_t semantic_frontier,
    uint64_t execution_epoch,
    OrbitKvCertificateV1 *out_certificate,
    char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_owner_commit_reclamations(
    OrbitKvOwnerHandle *owner,
    const uint64_t *certificate_ids,
    size_t certificate_count,
    char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_owner_release_request(OrbitKvOwnerHandle *owner,
                                      const uint8_t *request_id,
                                      size_t request_id_len,
                                      char *error_buffer,
                                      size_t error_buffer_len);

int32_t orbitkv_owner_stats(OrbitKvOwnerHandle *owner,
                            OrbitKvOwnerStatsV1 *out_stats,
                            char *error_buffer,
                            size_t error_buffer_len);

void orbitkv_owner_destroy(OrbitKvOwnerHandle *owner);

uint32_t orbitkv_dense_abi_version(void);
size_t orbitkv_dense_response_capacity(void);

int32_t orbitkv_dense_create(const uint8_t *plan_json,
                             size_t plan_json_len,
                             OrbitKvDenseHandle **out_dense,
                             char *error_buffer,
                             size_t error_buffer_len);

int32_t orbitkv_dense_execute_json(OrbitKvDenseHandle *dense,
                                   const uint8_t *command_json,
                                   size_t command_json_len,
                                   uint8_t *response_buffer,
                                   size_t response_buffer_len,
                                   size_t *out_response_len,
                                   char *error_buffer,
                                   size_t error_buffer_len);

size_t orbitkv_dense_certificate_capacity(OrbitKvDenseHandle *dense);

int32_t orbitkv_dense_submit_view(OrbitKvDenseHandle *dense,
                                  OrbitKvDenseRequestLeaseV1 request,
                                  uint64_t *out_submission_id,
                                  size_t *out_live_blocks,
                                  char *error_buffer,
                                  size_t error_buffer_len);

int32_t orbitkv_dense_complete_step(
    OrbitKvDenseHandle *dense,
    OrbitKvDenseRequestLeaseV1 request,
    uint64_t submission_id,
    uint64_t boundary,
    const uint8_t *binding_json,
    size_t binding_json_len,
    OrbitKvDenseCertificateV1 *certificates,
    size_t certificate_capacity,
    size_t *out_certificate_count,
    char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_dense_release_request(
    OrbitKvDenseHandle *dense,
    OrbitKvDenseRequestLeaseV1 request,
    OrbitKvDenseCertificateV1 *certificates,
    size_t certificate_capacity,
    size_t *out_certificate_count,
    char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_dense_commit_reclamations_and_recycle(
    OrbitKvDenseHandle *dense,
    OrbitKvDenseRequestLeaseV1 request,
    const OrbitKvDenseCertificateV1 *certificates,
    size_t certificate_count,
    char *error_buffer,
    size_t error_buffer_len);

int32_t orbitkv_dense_commit_reclamations(
    OrbitKvDenseHandle *dense,
    const OrbitKvDenseCertificateV1 *certificates,
    size_t certificate_count,
    char *error_buffer,
    size_t error_buffer_len);

void orbitkv_dense_destroy(OrbitKvDenseHandle *dense);

#ifdef __cplusplus
}
#endif

#endif
