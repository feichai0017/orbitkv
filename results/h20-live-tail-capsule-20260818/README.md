# H20 Live-Tail Capsule

OrbitKV compiled a 16K logical pure-SWA prefix into a 1K live KV tail. The
12,585,267-byte Capsule was hydrated into SGLang's paged-periodic allocator;
all six paired real-checkpoint runs matched cold output digests.

Median continuation E2E changed by -6.67% at 4K and -37.65% at 16K. This is a
single-request, one-decode-token pure-SWA result, not a Hybrid Full+SWA claim.
