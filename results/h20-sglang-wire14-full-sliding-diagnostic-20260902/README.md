# Current-wire Full+Sliding H20 diagnostic

Status: real-device, released-checkpoint, three-epoch diagnostic. Paired
correctness and native CUDA stream/event plus Sliding lifecycle checks passed.
The throughput gate failed. This archive is unsealed and makes no capacity,
memory-saving, speedup, production, or general-replacement claim.

## Exact scope

- GPU: one NVIDIA H20, UUID `GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3`.
- Engine: complete pinned SGLang `v0.5.17`, revision
  `29481685462732237d80d86076d6563e1f658102`.
- Model: local released GPT-OSS 20B checkpoint, 24 alternating Sliding/Full
  layers, Sliding window 128, 13,761,316,904 observed weight bytes.
- OrbitKV: `WIRE_VERSION = 14`, runtime target contract 4, page16 BF16 NHD,
  eager/non-overlap, one request, one device, FA3, deterministic PyTorch
  sampling, Triton MoE, no A2A, EP=1.
- Workload: 128 prompt tokens, 513 decode tokens, 640 materialized KV tokens,
  three independent process pairs with three samples per process.
- Capacity: Full class 1024 tokens; Sliding class 512 tokens; compiled Sliding
  resident/staging floor 400 tokens.

## Verified result

The original raw records and pair/summary artifacts are under `roomy/`. At the
time of capture all six records and three pairs passed the then-current
independent verifier. Across every manager epoch the native runtime directly
reported:

- 72 Sliding retirement certificates;
- 72 reclaimed Sliding pages;
- 12 periodic wrap events;
- 88 page-generation reuse events;
- current-forward-stream CUDA completion high-water `{domain: 1, value: 1539}`;
- zero final requests, Prefixes, active/retiring/quarantined pages, pending
  reclamations, or fail-stops.

All three stock/manager pairs matched output tokens exactly. Their paired
correctness and stream/event gates passed.

The frozen three-epoch throughput policy did **not** pass:

| Metric | Observed | Maximum allowed | Result |
| --- | ---: | ---: | --- |
| Paired median latency regression | `3.9194782230290826%` | `3%` | NO-GO |
| 95% bootstrap upper regression | `5.2764004667100184%` | `3%` | NO-GO |
| Paired samples | `9` | `>=9` | sufficient sample count |

The aggregate process-local time ratio was approximately `1.03111`, also in
the slower direction. These figures establish no speedup.

## Why this remains diagnostic

The run exposed and led to two harness corrections:

1. final manager census must be read after `flush_cache()`, not from the
   pre-flush server snapshot; and
2. a Full-class exact floor must reserve the next decode write slot:
   `page_align(materialized_kv_tokens + 1)`, not merely
   `page_align(materialized_kv_tokens)`.

The roomy runs crossed neither corrected boundary and therefore remain useful
runtime diagnostics, but their stored capacity-floor metadata was produced
before correction 2. The current verifier correctly refuses to promote this
archive to a current-HEAD sealed qualification. Exact-floor attempts at the
old 656-token Full floor returned 511 of 529 requested decode tokens for both
stock and manager with `finish_reason={type: length, length: 511}`; this is not
evidence of an OrbitKV capacity advantage. Before the corrected 672-token
floor could be rerun, GPU devices were removed from the container.

Pure Sliding, Chunked, latent KV, relocation, overlap, CUDA Graphs, distributed
execution, and production readiness are outside this archive.
