# H20 Component-Aware Prefix

OrbitKV ran GPT-OSS 20B on pinned SGLang `UnifiedRadixCache`. A seed request
published authenticated Full and SWA Capsule components; after SGLang inserted
them, the Rust owner marked both components resident. The continuation request
hit 1,024 cached tokens and acquired/released a shared Full+SWA Prefix lease.

All six runs matched the same output digest. Across three runs per mode, the
continuation median was 139.7 ms for OrbitKV and 152.4 ms for stock SGLang. This
small smoke does not establish a speedup; it shows no visible shared-hit
regression. Durable Capsule publication increased seed median from 1.45 s to
4.78 s and remains the dominant cost to optimize.

The qualified path is single-request, exact-prefix, non-overlap Full+SWA on H20.
Pressure eviction, concurrent sharing, speculative decoding, and vLLM are not
qualified here.
