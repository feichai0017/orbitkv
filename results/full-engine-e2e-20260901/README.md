# Full engine E2E evidence

Status: current-source, real-device correctness evidence with descriptive
timing only. This directory is append-only evidence; it is unsealed and is not
a performance, capacity, memory-saving, production-readiness, or general
KV-manager-replacement qualification.

This record exercises the current wire-13, target-contract-4 whole-domain Full
token-KV path with shared, page-aligned Prefix caching through the native
`RuntimeSession` lifecycle. The stock and manager processes used the same
pinned SGLang revision, local Qwen2.5-0.5B-Instruct checkpoint, one NVIDIA H20,
BF16 model and KV cache, FA3 attention, deterministic inputs and sampling,
engine geometry, Python environment, cache policy, and accelerator. Provenance
names such as the device, checkpoint, wire version, and target contract are
kept in content rather than artifact filenames.

Three order-balanced stock/manager pairs ran serially. Each process used two
warmups and five measured requests with 128 prompt tokens and 32 generated
tokens. Re-running the independent verifier against the files in `records/`
passes all three pairs, and every warmup and measured output token matches
stock.

The verifier reports these descriptive manager/stock ratios:

- all-iteration latency-ratio median: `1.0700109456807014`
- median of pair latency medians: `1.069134713635376`
- pair-median latency-ratio p95: `1.094222544022402`
- all-iteration latency-ratio p95: `1.0998971824357717`
- output-throughput ratio median: `0.9365623890654379`

All three pair medians show higher manager latency, so this evidence does not
show a speedup. Full retention has no bounded-retention memory advantage to
exercise here. `speedup_qualified`, `capacity_qualified`,
`memory_saving_qualified`, `production_qualified`, and
`general_replacement_qualified` are all `false`.

Every manager record has exactly one final snapshot. Each final snapshot has
all 128 pages free, zero fail-stops, zero pending CUDA events, zero quarantined
or retiring pages, and zero active requests, snapshots, prefixes, prepared or
submitted steps, pending reclamations, request references, Prefix references,
or reader pins. Each process recorded 224 current-forward-stream CUDA events
and 224 completion values. Shared Prefix residency observed immediately after
the workload is expected and is completely drained at final shutdown.

The repository was in an intentional breaking layout migration.
`environment/source-inventory.json` was freshly emitted by `engine/assemble.py`
from the live production source and its complete temporary assembly passed
`engine/assemble.py verify` before the inventory was copied here. It binds all
7,956 production files; its internal aggregate inventory digest is
`643f7dc2cd824ca82dab6c98b6f7d46e3a721f87ebe3bdf5b821e16096ec5d45`.
The live release FFI used by the records is identified in `summary.json` with
SHA-256 `615760dbe5fdbca529856f7b6d2cace16974c7d71a3d8c83ea232107da48e37f`.

Scope is deliberately narrow: current-source Full/shared-Prefix correctness
and descriptive timing on the recorded device and checkpoint. It does not
qualify Full+Sliding, pure Sliding, Chunked, MLA, fixed state, relocation,
continuous batching, overlap scheduling, CUDA Graphs, distributed execution,
performance benefit, capacity, memory savings, production readiness, or
general replacement. Absolute paths inside raw records are provenance, not
portable execution instructions.

Contents:

- `records/stock-e{1,2,3}.json` and `records/manager-e{1,2,3}.json`: six
  original wire-13 records, renamed generically without modifying their bytes.
- `verification.json`: verifier output regenerated from the packaged record
  paths, so its embedded paths resolve within this directory.
- `environment/runtime-manifest.json`: exact canonical runtime manifest used
  by the manager records.
- `environment/requirements.lock.txt`: exact 212-line Python package lock
  captured for the run environment. It requires Python 3.11; the lock's
  historical editable-install URL is preserved verbatim as run provenance.
- `environment/source-inventory.json`: freshly generated live production
  assembly manifest.
- `summary.json`: narrow claims, identities, lifecycle invariants, and
  descriptive aggregates.
- `SHA256SUMS`: SHA-256 for every packaged artifact except the checksum file.

Verification from the repository root:

```sh
python tools/verify_engine_e2e.py \
  --pair results/full-engine-e2e-20260901/records/stock-e1.json results/full-engine-e2e-20260901/records/manager-e1.json \
  --pair results/full-engine-e2e-20260901/records/stock-e2.json results/full-engine-e2e-20260901/records/manager-e2.json \
  --pair results/full-engine-e2e-20260901/records/stock-e3.json results/full-engine-e2e-20260901/records/manager-e3.json

cd results/full-engine-e2e-20260901
sha256sum -c SHA256SUMS
```
