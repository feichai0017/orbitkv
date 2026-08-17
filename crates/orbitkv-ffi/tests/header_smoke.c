#include "orbitkv_owner.h"

#include <stddef.h>

_Static_assert(sizeof(OrbitKvCertificateV1) == 104,
               "certificate ABI size changed");
_Static_assert(offsetof(OrbitKvCertificateV1, plan_fingerprint) == 72,
               "certificate fingerprint offset changed");
_Static_assert(sizeof(OrbitKvOwnerStatsV1) == 72, "stats ABI size changed");
_Static_assert(offsetof(OrbitKvOwnerStatsV1, plan_fingerprint) == 40,
               "stats fingerprint offset changed");

int main(void) {
  return orbitkv_owner_abi_version() == ORBITKV_OWNER_ABI_VERSION ? 0 : 1;
}
