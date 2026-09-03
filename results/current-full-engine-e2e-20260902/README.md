# Current-source Full engine E2E diagnostic

Status: real-device correctness diagnostic; unsealed, not a performance or
capacity qualification.

This record exercises the current whole-domain Full, shared-Prefix
`RuntimeSession` route through the complete pinned SGLang source assembly.  It
uses the local Qwen2.5-0.5B-Instruct checkpoint and one NVIDIA H20.  The
manager and pristine stock processes use the same SGLang revision, checkpoint,
engine geometry, deterministic prompts, sampling parameters, Python
environment, cache policy, and accelerator.

Three order-balanced pairs were run serially.  Every process used one warmup
and three measured requests with 32 prompt tokens and four generated tokens.
The independent pair verifier passed all three pairs and every generated token
matched stock.  Each manager process recorded the native-session lifecycle,
16 current-forward-stream CUDA events, zero fail-stops, and a final drain with
all 32 pages free and no request, Prefix, reader, retirement, quarantine, or
pending-event residue.

The aggregate descriptive result is a manager/stock median latency ratio of
`1.083477833233289` and a manager/stock median throughput ratio of
`0.9228393784288481`.  All three pairs have higher manager latency.  This
short Full workload has no semantic-retention advantage to exploit, and the
result demonstrates overhead rather than a speedup.  The verifier records
`speedup_qualified=false`.

The repository worktree was under an intentional breaking layout migration,
so this archive is current-source diagnostic evidence rather than a clean,
sealed qualification.  It does not qualify performance, capacity, memory
savings, Prefix-hit benefit, continuous batching, overlap scheduling, CUDA
Graphs, distributed execution, production readiness, general KV-manager
replacement, or any other native-session profile.  In particular, ordered
Full+Sliding, pure Sliding, and exact Chunked remain without current real-device
and released-checkpoint engine qualification.

Contents:

- `raw/stock-e{1,2,3}.json` and `raw/manager-e{1,2,3}.json`: the six original
  machine-readable runner records.
- `verification.json`: output of the independent three-pair verifier.
- `summary.json`: narrow claims, aggregate values, and source identities.
- `environment/runtime-manifest.json`: the exact canonical runtime manifest.
- `environment/requirements.lock.txt`: the Python dependency lock used to
  create the isolated environment.
- `environment/source-inventory.json`: content inventory emitted by
  `engine/assemble.py` for the source tree used by the run.
- `SHA256SUMS`: hashes for every archived artifact other than the checksum
  file itself.

The source inventory binds the manager, runtime, adapter, product profile, and
complete SGLang tree by file inventory.  Absolute paths in the raw records are
provenance only and are not portable execution instructions.
