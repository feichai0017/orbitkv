# Full engine E2E alternating-order evidence

Status: current-source, real-device Full/shared-Prefix correctness evidence
with descriptive timing only. This directory is append-only and unsealed. It
is not a speedup, performance, capacity, memory-saving, production-readiness,
or general KV-manager-replacement qualification.

This record exercises the current wire-13, target-contract-4 whole-domain Full
token-KV path with shared, page-aligned Prefix caching through the native
`RuntimeSession` lifecycle. Stock and manager processes used the same pinned
SGLang revision, local Qwen2.5-0.5B-Instruct checkpoint, one NVIDIA H20, BF16
model and KV cache, FA3 attention, deterministic inputs and sampling, engine
geometry, Python environment, cache policy, and accelerator. Device, model,
wire, and target-contract names are provenance content rather than artifact
filenames.

## Alternating launch order

The three pairs were launched serially in this order:

| Pair | First process | Second process |
| ---: | --- | --- |
| 1 | stock | manager |
| 2 | manager | stock |
| 3 | stock | manager |

This is an alternating, counterbalanced order, but it is not perfectly
balanced: an odd count of three pairs necessarily leaves two stock-first pairs
and one manager-first pair. It reduces a fixed-order bias without eliminating
all temporal or process-level variation.

Each process used two warmups and five measured requests with 128 prompt
tokens and 32 generated tokens. Re-running the independent verifier against
the packaged files passes all three pairs. Every warmup and measured output
token matches its stock counterpart.

## Descriptive result

The independent verifier reports these manager/stock ratios:

- all-iteration latency-ratio median: `1.075540757036277`
- all-iteration latency-ratio p95: `1.177579167070933`
- median of pair latency medians: `1.075540757036277`
- pair-median latency-ratio p95: `1.0954815891944474`
- output-throughput ratio median: `0.9309377122993483`

All three pair medians show higher manager latency. This evidence therefore
does not establish a speedup or performance benefit. Full retention also has
no bounded-retention memory advantage to exercise. `speedup_qualified`,
`performance_qualified`, `capacity_qualified`, `memory_saving_qualified`,
`production_qualified`, and `general_replacement_qualified` are all `false`.

Every manager record has exactly one final snapshot. Each final snapshot has
all 128 pages free, zero fail-stops, zero pending CUDA events, zero quarantined
or retiring pages, and zero active requests, snapshots, prefixes, prepared or
submitted steps, pending reclamations, request references, Prefix references,
or reader pins. Every manager process records 224 current-forward-stream CUDA
events and 224 completion values. Shared Prefix residency immediately after
the workload is expected and is completely drained at final shutdown.

## Supersession and limits

This directory supersedes `results/full-engine-e2e-20260901/` as the preferred
record for performance description because that earlier immutable bundle uses
a fixed stock-then-manager launch order for all three pairs. The earlier
bundle has not been edited or deleted, and this order correction does not
invalidate its independently verified token-correctness result.

The repository was in an intentional breaking layout migration.
`environment/source-inventory.json` was freshly emitted by `engine/assemble.py`
from the live production source after every destination artifact path existed,
and the complete temporary assembly passed `engine/assemble.py verify` before
the inventory was copied here. Its component file inventories and digests bind
the production source. The manifest's `worktree_status_sha256` is only the
dirty-worktree status captured immediately before publication; adding this
append-only evidence changes repository-wide Git status afterward, so that
field is not claimed as a live post-publication equality check. Source
inventory identities and counts are recorded in `summary.json`. The live
release FFI used by the records has
SHA-256 `615760dbe5fdbca529856f7b6d2cace16974c7d71a3d8c83ea232107da48e37f`.

Scope is deliberately narrow: current-source, real-device Full/shared-Prefix
correctness and descriptive timing on the recorded device and checkpoint. It
does not qualify Full+Sliding, pure Sliding, Chunked, MLA, fixed state,
relocation, continuous batching, overlap scheduling, CUDA Graphs, distributed
execution, performance benefit, capacity, memory savings, production
readiness, or general replacement. Absolute paths in raw records are
provenance, not portable execution instructions.

## Contents

- `records/stock-e{1,2,3}.json` and `records/manager-e{1,2,3}.json`: the six
  original records, renamed generically without changing their bytes.
- `verification.json`: independent verifier output regenerated from the
  packaged record paths. Its facts match the source verification; only the
  embedded artifact paths are rebased into this directory.
- `environment/runtime-manifest.json`: exact canonical runtime manifest used
  by all manager records.
- `environment/requirements.lock.txt`: exact 212-line Python environment lock
  used for qualification. The run used the Python 3.11 environment; the
  historical editable-install URL is retained verbatim as provenance.
- `environment/source-inventory.json`: freshly generated live production
  assembly manifest.
- `summary.json`: scoped claims, launch order, identities, lifecycle
  invariants, and descriptive aggregates.
- `SHA256SUMS`: SHA-256 for every packaged artifact except the checksum file.

## Verification

From the repository root:

```sh
python tools/verify_engine_e2e.py \
  --pair results/full-engine-e2e-alternating-20260901/records/stock-e1.json results/full-engine-e2e-alternating-20260901/records/manager-e1.json \
  --pair results/full-engine-e2e-alternating-20260901/records/stock-e2.json results/full-engine-e2e-alternating-20260901/records/manager-e2.json \
  --pair results/full-engine-e2e-alternating-20260901/records/stock-e3.json results/full-engine-e2e-alternating-20260901/records/manager-e3.json

cd results/full-engine-e2e-alternating-20260901
sha256sum -c SHA256SUMS
```
