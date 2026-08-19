# H20 CUDA-Event Overlap Frontier

OrbitKV ran GPT-OSS 20B with pinned SGLang overlap scheduling enabled. The thin
adapter recorded real CUDA events on SGLang's forward stream; the Rust owner
tracked request readers and emitted SWA retirement certificates only after the
required execution sequence ranges completed.

Across three single-request runs, all output digests matched stock SGLang. The
median E2E time was 3.352 s for OrbitKV and 3.389 s for stock (`0.989x`). This
small smoke demonstrates no visible overlap regression, not a speedup claim.
Every OrbitKV run registered and completed 34 execution events and committed one
completion-frontier-backed reclamation.

A two-request run registered 38 events and completed all 38. Decode submissions
shared one execution ticket across both requests; both retirement paths carried
request-specific completion witnesses and produced 64 total completion tokens.

The qualified boundary is Full+SWA ChunkCache, eager execution, overlap on,
sidecar owner, no speculation, no Radix+overlap, no disaggregation, and no CUDA
Graph claim.
