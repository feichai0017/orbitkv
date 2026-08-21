# OrbitKV

OrbitKV compiles attention-retention semantics into a generation-checked KV
page manager. It is designed to own page choice, immutable request snapshots,
Prefix references, GPU completion pins, and reclamation while an inference
engine continues to own tensor allocation, scheduling, kernels, and model
execution.

OrbitKV is still developed with breaking interfaces. There is one live core,
one typed C wire, and no compatibility loader for superseded lifecycle ABIs.

## Current boundary

The live tree is **ABI6**:

- the modular Rust host core is L2 GO for immutable snapshots, shared-page
  references, request fork, page-aligned Prefix lookup/publish/attach/evict,
  Full+SWA joint copy-on-write, and page-owned reclamation;
- the typed, batch-only C wire is L2 GO with exactly 23 exported
  `orbitkv_*` symbols, C/C++ layout checks, short-buffer zero-mutation checks,
  and no ABI5 scalar-named lifecycle aliases; and
- the split ABI6 Python FFI/runtime and SGLang `OrbitKVPrefixCache` are L2 GO
  on the host against the release library and pinned official `v0.5.17`
  source contract. No ABI6 H20 Prefix result exists yet.

The latest engine evidence is an immutable **historical ABI5-v5** snapshot,
not evidence for ABI6. Its exact `9233c06d…` source closure has scoped L4
correctness on one H20 against official SGLang `v0.5.17`, peeled commit
`29481685462732237d80d86076d6563e1f658102`.

The normative current/historical distinction is in the
[Capability Matrix](docs/capability-matrix.md).

## Architecture

```text
compiled retention plan
          |
          v
CanonicalKvManager                         sole ownership authority
  identity + arena                         generations and physical pages
  persistent snapshot                     immutable request roots
  append transaction                      prepare / submit / complete / COW
  Prefix                                  lookup / publish / attach / evict
  reclamation                             detach / certificate / ACK / recycle
          |
          | compact leases, intents, copies, detached bindings, certificates
          v
ABI6 C wire                                exact 23-symbol batch surface
          |
          v
Python runtime + SGLang adapter            host-qualified Prefix/COW path
          |
          v
ReqToToken / class LUT mirrors             checked mirrors, never authorities
          |
          v
FlashInfer / FA3 / engine KV tensor arenas
```

Requests hold generation-checked `SnapshotLease` heads. Snapshot class roots
are immutable persistent trees, so append work is proportional to changed
pages rather than total resident pages. Physical pages carry request refs,
Prefix refs, reader pins, writer state, and generation. A page is reusable
only after every reference is gone and an exact reclamation receipt is
acknowledged.

If a shared or pinned partial tail must be extended, the manager emits an
exact copy intent and publishes the new root only after the backend proves
that the copy was observed, completed, and ordered before new writes. For a
Hybrid request, partial Full and SWA tails enter the same joint-COW decision.

See [Standalone KV Manager Architecture](docs/standalone-kv-manager-architecture.md)
for the invariants and module boundaries.

## What is proven

| Surface | Status | Boundary |
| --- | --- | --- |
| ABI6 Rust core | L2 GO | Host unit, property, fault, stale-lease, Prefix, fork, COW, and reclamation tests |
| ABI6 C wire | L2 GO | Exact 23 symbols, C/C++ layouts, batch atomicity, short-buffer and malformed-receipt gates |
| ABI6 Python/SGLang | L2 GO | Exact ctypes layouts, incremental journals, pinned cache seam, warm Prefix, joint COW, mirror cleanup, fail-stop, and teardown host gates |
| ABI6 H20 Prefix | Pending | No engine run may inherit ABI5 evidence |
| Frozen ABI5-v5 | Historical scoped L4 | Qwen Full and GPT-OSS Full+SWA B1/B4 correctness on one H20 |

In the frozen ABI5-v5 H20 record, all eight manager/stock JSON records pass
independent verification, all request traces match, and every Full/SWA arena
drains. Grouped B4 release reduces 20 request-level release/recycle calls to
five batch transactions.

The same-capacity intrinsic memory reduction is **0%** because the compared
SGLang processes reserve identical KV tensor arenas. The one H20 epoch reports
B4 steady manager overhead of +4.1932% for Qwen and -5.2048% for GPT-OSS, while
Qwen B1 is +5.0009%. There are no repeated-epoch statistics, so
`performance_go=false` and no general speedup is claimed.

[Frozen ABI5-v5 H20 record](results/h20-sglang-v0517-abi5-v5-grouped-release-20260821/README.md)

## Build and verify

```bash
cargo fmt --all -- --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings

cargo test --locked --manifest-path crates/orbitkv-ffi/Cargo.toml --all-targets
python tools/verify_active_source.py
python tools/verify_capability_matrix.py
python tools/verify_manifests.py
```

The active-source gate limits production Rust/Python modules to 1,500 lines,
test/benchmark modules to 2,000 lines, verifies ABI6 markers, and rejects the
removed ABI5 lifecycle aliases. It deliberately ignores append-only evidence
under `results/`.

## Next gates

The ordered work is:

1. run exact-source SGLang `OrbitKVPrefixCache` correctness on H20;
2. implement token-exact relocation/compaction against immutable snapshots;
3. qualify overlap and CUDA Graph completion domains; and
4. add speculation, multi-GPU placement, and disaggregation.

Compaction means byte-exact K/V relocation and physical defragmentation. It is
not quantization, numerical compression, or evidence of a same-capacity memory
win. See the [Token Virtualization and Attention Roadmap](docs/token-virtualization-and-attention-roadmap.md).

Historical records and their source hashes are indexed in
[results/README.md](results/README.md). They are append-only and never qualify a
later ABI automatically.
