#ifndef ORBITKV_OWNER_H
#define ORBITKV_OWNER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORBITKV_OWNER_ABI_VERSION 1u
#define ORBITKV_STATUS_OK 0
#define ORBITKV_STATUS_NO_CERTIFICATE 1
#define ORBITKV_STATUS_INVALID_ARGUMENT -1
#define ORBITKV_STATUS_OWNER_ERROR -2
#define ORBITKV_STATUS_PANIC -3

typedef struct OrbitKvOwnerHandle OrbitKvOwnerHandle;

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

#ifdef __cplusplus
}
#endif

#endif
