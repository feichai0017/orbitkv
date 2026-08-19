# Dense Runtime Validation

OrbitKV compiled the same Full+SWA StatePlan into a fixed-capacity ownership
runtime with dense Class IDs, generation-checked request leases, direct request
cell stripes, and bounded binding/submission/certificate arenas.

The release benchmark executes the complete control-plane lifecycle on both the
Reference Manager and Dense Runtime: materialize, semantic advance, immutable
view submission, completion, certificate generation, physical commit, request
release, and request-slot recycling.

Across the recorded 1x128, 8x512, and 32x256 geometries, Dense median time was
`0.081x`, `0.063x`, and `0.065x` of Reference respectively. All final resident
counts were zero. A deterministic 1,000-event differential state machine and
1,000 request-lifecycle arena-reuse test passed.

This is a host control-plane result. It does not qualify a SGLang/vLLM adapter,
GPU kernel time, or end-to-end serving latency for the Dense Runtime.
