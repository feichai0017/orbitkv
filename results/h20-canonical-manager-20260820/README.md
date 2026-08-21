# Canonical manager Phase-1 H20 validation

This directory is the exact-source qualification record for OrbitKV's
breaking-only ABI3 manager and pinned SGLang adapter. It qualifies one profile:
Mistral-7B-v0.1, all 32 layers using a 4,096-token sliding window, page16 BF16
NHD KV, eager FlashInfer, ChunkCache, one H20, and TP/PP/DCP equal to one.

It does not qualify a general replacement for Full or Hybrid caches,
Prefix/Radix/COW, CUDA Graph, overlap, speculation, multi-GPU, remote memory,
or other engines.

## Correctness and lifecycle

| Workload | OrbitKV digest | Native pure-SWA digest | Final manager state |
| --- | --- | --- | --- |
| B1: 12,001 prompt + 33 decode | `2a17477503c7…` | `2a17477503c7…` | 385/385 pages free; all live/pending/quarantine counters zero |
| B4: 4 × (12,001 prompt + 33 decode) | `3c8b982e76e0…` | `3c8b982e76e0…` | 1,159/1,159 pages free; all live/pending/quarantine counters zero |

The semantic reference is a separate pinned SGLang worktree with only the
reviewed `095ec6c-mistral-pure-swa-reference.patch`. It keeps SGLang's native
page allocator, uses its required page size 1, and corrects Mistral's metadata
and per-layer attention window to native pure SWA. The product adapter is not
used in that reference.

Primary records:

- `b1-manager-current-final.{log,json}`
- `b1-native-swa-reference-final.{log,json}`
- `b4-manager-current-final.{log,json}`
- `b4-native-swa-reference-final.{log,json}`

## Memory accounting

The model uses 131,072 KV bytes per token across all 32 layers.

- OrbitKV's B4 manager pool has 1,159 pages = 18,544 advertised tokens. SGLang
  adds one page of tensor padding, so the physical arena is 18,560 slots or
  2.265625 GiB.
- Stock page16 at the same advertised capacity also has 18,560 physical slots.
  The intrinsic same-capacity KV saving is therefore **0%**.
- Native page1 uses 18,545 physical slots; its 15-slot difference is only the
  different page padding granularity.
- A 50,000-token configuration uses 50,016 physical slots or 6.10546875 GiB.
  Moving to the compiled 18,560-slot arena saves 3.83984375 GiB (62.89%), but
  this is an admission-budget difference, not KV compression.

## Performance boundary

The current B4 manager smoke took 10.0858 seconds; the native page1 SWA
reference took 9.1686 seconds. OrbitKV was about 10.00% slower in one sample.
This is a negative smoke and not a formal performance result or a performance
GO. A future performance gate needs paired fresh processes, repeated samples,
confidence intervals, and the wider serving matrix.

## Why pristine stock differs

At pinned revision `095ec6c997bfdd25d3864cb0ce77a6562a934b96`, pristine
SGLang treats Mistral as full attention rather than pure SWA. It matches the
manager on the 4,000-token pre-window diagnostic and diverges after crossing
the 4,096-token window. Stock records are retained to document that behavior
and capacity, but they are not semantic or performance oracles for OrbitKV.

## Provenance and excluded records

`summary.json` gives the machine-readable decision boundary. `manifest.json`
hashes the current source, records, checkpoint-facing plan, reviewed SGLang
patches, adapter, ABI, and test surfaces.

Setup, missing-dependency, admission-liveness, and completion-progress failures
are retained append-only. The earlier `b1-manager-final` and
`b4-manager-same-cap-final` records used an older harness hash. They are not
part of the primary success proof; the `*-manager-current-final` records are.
