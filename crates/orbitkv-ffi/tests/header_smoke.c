#include "orbitkv_owner.h"

#include <stddef.h>

_Static_assert(sizeof(OrbitKvCertificateV1) == 104,
               "certificate ABI size changed");
_Static_assert(offsetof(OrbitKvCertificateV1, plan_fingerprint) == 72,
               "certificate fingerprint offset changed");
_Static_assert(sizeof(OrbitKvOwnerStatsV1) == 72, "stats ABI size changed");
_Static_assert(offsetof(OrbitKvOwnerStatsV1, plan_fingerprint) == 40,
               "stats fingerprint offset changed");
_Static_assert(sizeof(OrbitKvDenseRequestLeaseV1) == 8,
               "Dense request lease ABI size changed");
_Static_assert(sizeof(OrbitKvDenseCertificateV1) == 64,
               "Dense certificate ABI size changed");
_Static_assert(offsetof(OrbitKvDenseCertificateV1, backend_index) == 40,
               "Dense backend index offset changed");

int main(void) {
  if (orbitkv_owner_abi_version() != ORBITKV_OWNER_ABI_VERSION) {
    return 1;
  }
  if (orbitkv_dense_abi_version() != ORBITKV_DENSE_ABI_VERSION) {
    return 2;
  }
  return orbitkv_dense_response_capacity() > 0 ? 0 : 3;
}
