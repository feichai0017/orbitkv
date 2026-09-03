# Current-source Full engine E2E after host-path optimization

Status: real-device correctness diagnostic with descriptive timing; unsealed,
not a performance, capacity, memory-saving, or production qualification.

This append-only record exercises the whole-domain Full/shared-Prefix native
`RuntimeSession` path after three host-path changes: the product runtime became
session-only, ordinary location checks stopped reading device values back to
the host by default, and scheduler-turn polling/capacity census plus empty
publication cleanup gained bounded fast paths. Exact physical-location value
checking remains available through the explicit diagnostic environment switch.

The manager and pristine stock processes use the same pinned SGLang revision,
local Qwen2.5-0.5B-Instruct checkpoint, one NVIDIA H20, deterministic inputs,
sampling settings, engine geometry, Python environment, and cache policy. All
12 process records completed. Both independent three-pair verifications passed
and every output token matched stock. Every manager record reports real
current-forward-stream CUDA events, zero fail-stops, no pending events, and a
complete final drain.

Two workloads are preserved:

| Workload | Repetitions | Median latency ratio | Median throughput ratio | Direction |
| --- | ---: | ---: | ---: | --- |
| 32 prompt / 4 decode | 3 pairs; 1 warmup + 3 measured per process | `1.110648717926606` | `0.9011844583047831` | mixed across pairs; high process-level jitter |
| 128 prompt / 32 decode | 3 pairs; 2 warmups + 5 measured per process | `1.0466919427065124` across all iterations; `1.0543203475398686` across pair medians | `0.9597361661930917` | all three pairs have higher manager latency |

The longer workload is the more stable descriptive result: OrbitKV retains an
approximately four-percent throughput cost on this Full profile. Full has no
bounded-retention memory advantage for the manager to exploit, so neither
workload establishes a product benefit. Both verifier records set
`speedup_qualified=false`.

The repository was undergoing an intentional breaking layout migration. The
source inventory binds the exact manager/runtime/adapter/product files and the
complete SGLang source tree used here, but the worktree was dirty and this is
not a sealed qualification. These results do not transfer to Full+Sliding,
pure Sliding, Chunked, MLA, fixed state, relocation, continuous batching,
overlap, CUDA Graphs, distributed execution, capacity, memory use, or general
KV-manager replacement.

Contents:

- `short/` and `long/`: original stock/manager records plus independent
  verifier output for each workload.
- `environment/runtime-manifest.json`: exact canonical runtime manifest.
- `environment/requirements.lock.txt`: isolated Python environment lock.
- `environment/source-inventory.json`: final assembled source inventory.
- `summary.json`: narrow claims and aggregate values.
- `SHA256SUMS`: hashes for every artifact except the checksum file itself.
