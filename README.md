# OrbitKV

OrbitKV is an attention-state compiler and transactional ownership runtime.
It compiles attention-retention semantics into generation-checked physical
plans and is designed to own page choice, immutable request snapshots, Prefix
references, GPU completion pins, and reclamation while an inference engine
continues to own tensor allocation, scheduling, kernels, and model execution.
It is neither a full SGLang replacement nor a mature L5 production system.

OrbitKV is still developed with breaking interfaces. There is one live core,
one typed C wire, and no compatibility loader for superseded lifecycle ABIs.
The compiler also distinguishes token KV, MLA latent KV, recurrent state, and
convolution state instead of forcing every Hybrid layer into a KV-block shape.

## Current boundary

The live tree is **ABI8**:

- the modular Rust host core is L2 GO for immutable snapshots, shared-page
  references, request fork, page-aligned Prefix lookup/publish/attach/evict,
  Full+SWA joint copy-on-write, page-owned reclamation, canonical token views,
  policy/proof dispositions, and batch relocation subtransactions with
  pre-commit validation and fail-stop quarantine;
- the typed C wire is L2 GO with exactly 40 exported `orbitkv_*` symbols:
  29 batch-only canonical-manager symbols plus 11 independent fixed-state-pool
  symbols, with C/C++ layout checks, short-buffer zero-mutation checks, and no
  ABI5 scalar-named lifecycle aliases; and
- the split ABI8 Python FFI/runtime, independent `CtypesStatePool`, and SGLang
  `OrbitKVPrefixCache` are L2 GO
  on the host against the release library and pinned official `v0.5.17`
  source contract. The explicit eager token-relocation path now has host L2
  seams for Full and common-victim-set Full+SWA, real copy/event orchestration,
  checked Full-to-SWA LUTs, and split absolute/active lengths. Without changing
  ABI8, one admitted multi-request scheduler batch performs one native
  disposition mark, prepare, submit, and complete call; the runtime then makes
  one aggregate page-registry commit followed by one aggregate request-head
  replacement. The SGLang plugin flattens all
  request moves into one backend move and one completion event, validates every
  mirror plan before the first mirror write, and sends one batch ACK. The scalar
  relocation API remains a singleton compatibility wrapper over this batch
  path. Within the narrower single-Full, request-private, full-evacuation
  boundary, host tests also cover repeated append -> mark -> relocate ->
  exact-ACK cycles, including append after a packed publication, and the
  periodic trigger across two reclamation boundaries. Once a mark has
  succeeded, every later failure or uncertain return is fail-stop: the workflow
  does not roll back the disposition mark or restore an older head. A
  producer-to-copy event orders relocation, but current completion handling is
  eager and host-blocking; asynchronous consumer-stream overlap is not
  implemented. Packed Prefix publication, fork, and shared
  partial-tail COW remain unsupported and fail closed; the dense
  Prefix/fork/COW claims above do not extend through a packed root. The restricted
  production fixed-state seam connects request allocation, initial
  `MambaPool.clear_slots`, forward completion-event registration, and
  release-time wait/retire/clear/exact-ACK. The strict normalized official
  `Qwen/Qwen3.5-0.8B` profile is host-qualified on this seam: six Full layers
  use token KV, while 18 linear-attention layers use request-private FP32 GDN
  recurrent state plus BF16 convolution history. Its fail-closed production
  contract requires Full FA3, Triton general/prefill/decode linear attention,
  FP32 temporal state, `mamba_radix_cache_strategy=no_buffer`, and
  `disable_radix_cache=true`; the pair-verification workload uses fresh prompts
  and requires zero cached tokens and zero fixed-state copies. Same-owner replacement
  through `MambaPool.copy_from` still has no production trigger, and KDA,
  ShortConv, and other linear-attention family bindings remain pending. This
  exact Qwen3.5 profile now also has scoped pair-verification evidence from
  recorded H20 runtime snapshots; it is independently unattested and does not
  promote the profile to L4 qualification.

The formally qualified engine evidence remains the **sealed ABI8** record at
`results/h20-sglang-v0517-abi8-full-hybrid-20260823`. It binds exact source
commit `6f62a23b9abaa9bf12e9b060389259fa9185e70f` (short id `6f62a23`) to
Scoped L4 correctness for the Prefix path on one H20 against official SGLang
`v0.5.17`, peeled commit `29481685462732237d80d86076d6563e1f658102`.
The separate Qwen3.5 archive records six stock/manager pairs from H20 runtime
snapshots but is explicitly `qualified=false` and
`hardware_attested=false`, so the device remains independently unattested. A
newer Qwen3.8-27B-FP8 run records eight passing diagnostic pairs and is
`diagnostic_only`, `sealed=false`, dirty-source, `preflight_bound=false`,
`hardware_attested=false`, `qualified=false`, and `performance_go=false`.
A separate Qwen2.5-0.5B token-relocation diagnostic records 8/8 exact-token,
census-clean Naive/Relocate pairs from H20 runtime observations. It has the
same strict claim boundary: unsealed, dirty-source, independently unattested,
`diagnostic_only`, `qualified=false`, and `performance_go=false`.
The earlier sealed ABI5-v5 record remains historical evidence for its exact
`9233c06d…` source closure only.

The normative current/historical distinction is in the
[Capability Matrix](docs/capability-matrix.md).
The restricted fixed-state seam additionally requires `ORBITKV_STATE_PLAN` to
name the canonical heterogeneous plan whose token projection matches
`ORBITKV_PLAN`; missing or drifting geometry fails startup.
The token manager and fixed-state pool are separate ABI8 handles. A failure
between their commits has fail-stop containment only: there is no cross-handle
atomic commit or rollback guarantee.
The Qwen `qwen3_5` dense-config-family frontend intentionally accepts only the
explicit nested manifest shape used by the official Qwen3.5/Qwen3.8 dense
checkpoints. The checked non-FP8 Qwen3.8-27B fixture proves compilation of its
structural geometry. The separate H20 diagnostic records the declared
`Qwen/Qwen3.8-27B-FP8` repository revision and binds the downloaded config,
index, and every shard by hash and byte count; the repository provenance was
not independently online-attested by the archive verifier. Runtime admission
is structural for the currently implemented exact GDN+convolution contract,
rather than a model marketing name. Evidence remains model- and
workload-specific.

## Architecture

```text
HF config / retention IR / heterogeneous state schema
          |
          v
checked attention-state compiler
          | token-addressable projection       | fixed-width projection
          v                                    v
CanonicalKvManager                         StateCheckpointPool
  identity + arenas                         generation leases
  immutable snapshots                      clear/copy receipts
  append / Prefix / COW                    event / retire / exact ACK
  relocation / reclamation
          |                                    |
          +----------------+-------------------+
                           v
ABI8 C wire                                exact 40-symbol typed surface
                           |
                           v
Python runtime + pinned SGLang hooks       checked mirrors, never authorities
                           |
                           v
SGLang tensor arenas / kernels / scheduler / model execution
```

Requests hold generation-checked `SnapshotLease` heads. Snapshot class roots
are immutable persistent trees, so append work is proportional to changed
pages rather than total resident pages. Physical pages carry request refs,
Prefix refs, reader pins, writer state, and generation. A page is reusable
only after both frontiers pass: the Semantic Frontier proves that no live
snapshot needs it, and the Execution Frontier proves that prior GPU work has
completed. An exact reclamation receipt must then be acknowledged before
reuse. SGLang continues to allocate tensors and execute kernels and scheduling;
OrbitKV owns the checked plan, identities, page/state lifecycle, transactions,
and reclamation decisions for the admitted profile.

If a shared or pinned partial tail must be extended, the manager emits an
exact copy intent and publishes the new root only after the backend proves
that the copy was observed, completed, and ordered before new writes. For a
Hybrid request, partial Full and SWA tails enter the same joint-COW decision.

See [Standalone KV Manager Architecture](docs/standalone-kv-manager-architecture.md)
for the invariants and module boundaries.

## What is proven

| Surface | Status | Boundary |
| --- | --- | --- |
| ABI8 Rust core | L2 GO | Host unit, property, fault, stale-lease, Prefix, fork, COW, token relocation, reclamation, and fixed-state checkpoint tests |
| Heterogeneous state compiler | L1 GO | Token KV, MLA latent+RoPE, recurrent, and convolution contracts compile to distinct backends; official Qwen3.5-0.8B and Qwen3.8-27B configs produce exact Full/GDN/convolution geometry |
| Recurrent/convolution checkpoint pool | L2 host | Generation-checked core/wire initial/replace/retire/ACK, abort, and quarantine; the request-private GDN+convolution capability is structurally gated, with model-specific Qwen3.5 pair evidence from recorded H20 runtime snapshots and Qwen3.8 recorded-device diagnostic evidence; neither is independently hardware-attested or L4-qualified |
| Pure MLA SGLang seam | L2 host / L4 pending | Explicit latent+RoPE geometry checked against the real SGLang pool; combined-row relocation host-tested; H20 and model correctness pending |
| ABI8 C wire | L2 GO | Exact 40 symbols, C/C++ layouts, per-handle manager-batch and state-pool-batch atomicity, short-buffer and malformed-receipt gates; no cross-handle atomicity |
| ABI8 Python/Prefix/state wire | L2 GO plus scoped Prefix L4 | 73 frozen ctypes layouts and broad host gates; the sealed Full/Full+SWA Prefix subset has exact-source H20 evidence |
| ABI8 multi-request private Full relocation | L2 host plus recorded-device H20 diagnostic / L4 pending | ABI8-preserving scheduler batches invoke mark/prepare/submit/complete once each, perform one aggregate registry commit followed by one aggregate head replacement, flatten plugin moves behind one event, validate all mirror plans before writes, and use one batch ACK; the scalar API is a singleton compatibility wrapper. Mirror writes are not rollback-atomic: post-mark failure is fail-stop without rollback. A producer-to-copy event exists, but completion is eager and host-blocking with no asynchronous overlap. Repeated append/relocate cycles and the periodic trigger are host-tested; the Qwen2.5-0.5B diagnostic exercises the path on a recorded H20; packed Prefix, fork, and shared COW fail closed |
| Engine-neutral CUDA opaque-byte relocation harness | H20 component conformance: 7 passed | B1/B4/B32 each execute two cycles with an independent live-token/payload oracle, 257-byte coordinate-bearing records, real non-default append/copy/consumer streams, exact 3-page-to-2-page evacuation with 24 moves per request/cycle, event-ordered byte readback, ACK-gated same-page/higher-generation reuse, and final drain. This is component conformance, not sealed L3/L4, capacity, or performance qualification |
| Qwen2.5-0.5B SGLang relocation diagnostic | Diagnostic only / L4 pending | Official SGLang v0.5.17, BF16 NHD page16 eager single-H20 observation, FlashInfer same-policy oracle, four alternating-order epochs, B1/B4, and five iterations per process. All 8/8 pairs are token-exact with expected relocation/reclamation counters, complete drain, and zero failures; dirty and unsealed, `hardware_attested=false`, `qualified=false`, and `performance_go=false` |
| ABI8 SGLang fixed state | Scoped L2 host plus recorded-device evidence / L4 pending | Request-private GDN+convolution allocation, clear, forward event, retire, clear, and ACK are wired and fail closed on geometry/backend drift; six Qwen3.5 verification pairs and eight Qwen3.8 diagnostic pairs pass their scoped checks, but both records have `hardware_attested=false` and `qualified=false` and neither qualifies performance or a full SGLang replacement |
| Qwen3.8-27B-FP8 diagnostic | Diagnostic only / L4 pending | Four epochs for each of B1 and B4 yield eight stock/manager pairs total; all pass verifier and exact-token checks on recorded-device data; `diagnostic_only`, `sealed=false`, dirty source, `preflight_bound=false`, `hardware_attested=false`, `qualified=false`, and `performance_go=false` |
| ABI8 H20 Prefix path | Scoped L4 correctness | Exact `6f62a23`; Qwen Full and GPT-OSS Full+SWA B1/B4 only; performance and excluded features remain unqualified |
| Frozen ABI5-v5 | Historical scoped L4 | Qwen Full and GPT-OSS Full+SWA B1/B4 correctness on one H20 |

The sealed ABI8 H20 record uses official SGLang `v0.5.17` on one H20
with page16 BF16 NHD storage, eager FA3 execution, and TP/PP/DP/DCP = 1.
Across Qwen2.5-7B Full and GPT-OSS-20B Full+SWA at B1 and B4, all 12
manager/stock pairs pass over three epochs. Manager and stock each contain 126
measured request traces and 4,158 output tokens. The manager records exercise
Prefix publication, warm hits and attach, eviction, and final drain; every
Hybrid record also has positive SWA retirement-certificate and reclaimed-page
counters.

Mean manager-over-stock time is +7.3678% for Full B1, +11.8684% for Full B4,
+3.9865% for Full+SWA B1, and +4.4476% for Full+SWA B4. These are diagnostic
measurements, `performance_go=false`, and they do not establish a speedup. The
manager and stock reserve equal configured KV tensor arenas, so the observed
configured arena reservation difference is **0%**; this is not a qualified end-to-end
memory-saving result. The record does not qualify a complete
SGLang replacement or production readiness. It covers only Qwen2.5-7B Full and
GPT-OSS-20B Full+SWA; Qwen3.5, token relocation, MLA, fixed state, overlap,
CUDA Graphs, speculation, distributed execution, and performance
qualification are explicitly excluded.

The Qwen3.5 pair-verification archive records official
`Qwen/Qwen3.5-0.8B`, official SGLang `v0.5.17` at
`29481685462732237d80d86076d6563e1f658102`, and recorded H20 runtime
snapshots (UUID prefix `GPU-3a35…`) that remain independently unattested. Both
stock and manager run page16 BF16 NHD eager
Full FA3 plus Triton linear-attention backends, with Radix disabled, fresh
prompts, and `cached_tokens=0`. The cases are B1 with one request, state
capacity two, and one iteration, and B4 with four requests, state capacity
four, and five iterations. Across three epochs, all six stock/manager pairs
have equal output-token totals. Exact token and fixed-state lifecycle
counters, CUDA stream/event completion, and final drain all pass. Per epoch,
fixed-state prepare/clear/retire/ACK counts are one for B1 and 20 for B4;
copies remain zero.

| Qwen3.5 case | Manager aggregate | Stock aggregate | Manager over stock |
| --- | ---: | ---: | ---: |
| B1 | 1.5898847853 s | 1.5818591726 s | +0.507353% |
| B4 | 0.9498731474 s | 0.9156339097 s | +3.739403% |

These are diagnostic pair-verification timings, not a performance win. The
derived output throughput is 20.7562 versus 20.8615 tokens/s at B1 and
138.9659 versus 144.1624 tokens/s at B4 (OrbitKV versus stock), respectively
-0.5048% and -3.6046%. Excluding each B4 process's first iteration gives
172.2316 versus 184.3654 tokens/s, or -6.5814%; this steady-state diagnostic is
not part of the pair contract. The
archive states `performance_go=false`, `qualified=false`, and
`hardware_attested=false`: it is based on recorded H20 runtime snapshots, but
the hardware is independently unattested. The observed configured arena reservation
difference is **0%**, not a qualified end-to-end memory-saving result. It
therefore does not establish a speedup, L4 qualification, production readiness,
or a complete SGLang replacement.

[Qwen3.5 H20 fixed-state pair-verification record](results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md)

The current Qwen3.8 diagnostic records declared `Qwen/Qwen3.8-27B-FP8`
repository revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Its raw
records bind the downloaded config, index, and all 66 weight shards by hash and
byte count, totaling 30,866,866,928 bytes and 1,606 indexed tensors; repository
provenance is declared rather than independently online-attested. It runs
official SGLang `v0.5.17` at revision
`29481685462732237d80d86076d6563e1f658102` on the recorded NVIDIA H20
`GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3`, with explicit
`fp8_gemm_runner_backend=triton`, single-GPU eager execution, and page16 BF16
NHD KV storage. Four epochs were order-balanced: epochs 1 and 3 ran
manager then stock, while epochs 2 and 4 ran stock then manager. Each epoch
contains one B1 pair and one B4 pair, yielding eight pairs total; every process
runs five iterations. All eight stock/manager pairs pass the
verifier and match tokens exactly; every manager final census has zero live,
pending, reserved, retiring, or quarantined ownership, and failure/fail-stop
counters are zero.

Hot statistics exclude iteration 0 of every process, leaving 16 samples per
mode and batch size:

| Qwen3.8 case | Stock mean / median / p95 | Manager mean / median / p95 | Latency / throughput delta | Per-epoch latency delta |
| --- | --- | --- | --- | --- |
| B1 | 2.6296723178 / 2.5374011379 / 3.0721712420 s | 2.8031814888 / 2.6451683380 / 3.3369778013 s | +6.5981% / -6.1897% | +2.7822%, +0.7129%, +26.3407%, -1.9835% |
| B4 | 2.9576119229 / 2.9602836296 / 3.0001941137 s | 3.0395950049 / 3.0345056280 / 3.0800264925 s | +2.7719% / -2.6972% | +1.8441%, +2.8850%, +5.0070%, +1.3885% |

B1 has conspicuous inter-epoch jitter, including the +26.3407% outlier and a
negative fourth-epoch delta. The result supports no positive performance
claim; B4 is also slower in the pooled diagnostic. Manager and stock use the
same configured tensor-arena capacity and report the same KV-cache reservation,
so the observed configured arena reservation difference is **0%**. This is not a qualified
end-to-end memory-saving result. This is diagnostic evidence only: the source
closure is dirty and the record is `diagnostic_only`, `sealed=false`,
`preflight_bound=false`, `hardware_attested=false`, `qualified=false`, and
`performance_go=false`. A separate default-auto DeepGEMM attempt loaded all
66/66 shards, but its E2E run did not complete because lengthy precompilation
was externally terminated; it contributes no pair or performance result.

[Qwen3.8-27B-FP8 H20 diagnostic archive](results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md)

The token-relocation diagnostic pins official SGLang `v0.5.17`,
Qwen2.5-0.5B, BF16 NHD page16 storage, eager single-GPU execution, and
FlashInfer as the token-indexed same-policy oracle. Four alternating-order
epochs at B1 and B4 produce eight Naive/Relocate pairs; every process runs
five iterations, and iteration 0 is excluded from the 16 hot samples per mode
and batch size. All 8/8 pairs match output tokens exactly, exercise two
reclamation rounds per iteration, drain fully, and report zero failure,
quarantine, and fail-stop counters.

| Relocation case | Throughput delta | Mean latency delta | p95 latency delta |
| --- | ---: | ---: | ---: |
| B1 | +7.629% | -7.088% | -22.844% |
| B4 | -1.232% | +1.247% | +2.880% |

B1 varies substantially across epochs, and B4 is slightly slower. These are
descriptive observations only. The archive is `diagnostic_only`, unsealed,
dirty-source, based on recorded H20 observations rather than independent
hardware attestation, `qualified=false`, and `performance_go=false`. It makes
no capacity, end-to-end memory-saving, general speedup, L3/L4, production, or
complete-SGLang-replacement claim. The paired matrix uses FlashInfer because
FA3 page tables cannot represent the sparse Naive oracle and that combination
now fails closed; a separate relocate-only FA3 smoke passed but is not a valid
same-policy comparison.

[Qwen2.5-0.5B token-relocation H20 diagnostic](results/h20-sglang-v0517-token-relocation-diagnostic-20260825/README.md)

[Sealed ABI8 H20 record](results/h20-sglang-v0517-abi8-full-hybrid-20260823/README.md)

In the frozen ABI5-v5 H20 record, all eight manager/stock JSON records pass
independent verification, all request traces match, and every Full/SWA arena
drains. Grouped B4 release reduces 20 request-level release/recycle calls to
five batch transactions.

The compared SGLang processes reserve equal configured KV tensor arenas, so
the observed configured arena reservation difference is **0%**; this is not a qualified
end-to-end memory-saving result. The one H20 epoch reports
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
test/benchmark modules to 2,000 lines, verifies ABI8 markers, and rejects the
removed ABI5 lifecycle aliases. It deliberately ignores append-only evidence
under `results/`.

## Next gates

The ordered work is:

1. rerun Qwen3.8-27B-FP8 from a clean sealed source closure after the full
   qualification preflight, then independently attest hardware and qualify
   correctness before considering any performance gate;
2. independently attest the hardware reported by the recorded Qwen3.5 H20
   runtime snapshots and close the L4
   release-qualification gates; any performance claim needs a separate
   qualified benchmark because both current manager aggregates are slower;
3. add the production trigger for the host-tested same-owner
   `MambaPool.copy_from` replacement path;
4. rerun the demonstrated SGLang relocation diagnostic from a clean, sealed,
   preflight-bound source closure with independent hardware attestation and
   complete L3/L4 qualification gates; separately implement
   and qualify KDA, ShortConv, and other linear-attention family bindings and
   extend exact-source H20 qualification to MLA;
5. qualify overlap and CUDA Graph completion domains; and
6. add speculation, multi-GPU placement, and disaggregation.

Compaction means byte-exact K/V relocation and physical defragmentation. It is
not quantization, numerical compression, or evidence of a same-capacity memory
win. See the [Token Virtualization and Attention Roadmap](docs/token-virtualization-and-attention-roadmap.md).

Historical records and their source hashes are indexed in
[results/README.md](results/README.md). They are append-only and never qualify a
later ABI automatically.

Compile the heterogeneous-state example with:

```bash
cargo run -- compile-state-plan examples/hybrid-attention-state-plan.json
cargo run -- compile-state-manager-plan examples/hybrid-attention-state-plan.json
cargo run -- compile-state-manager-plan examples/deepseek-v2-lite-mla-state-plan.json
cargo run -- compile-plan examples/deepseek-v2-lite-mla.json
```

For this generic example, `compile-state-plan` preserves every backend-specific
contract and component geometry. `compile-state-manager-plan` projects only
token-addressable TokenKV/MLA classes into canonical manager input; recurrent
and convolution states remain on their generation-checked checkpoint path.

Compile the strict normalized official Qwen3.5-0.8B fixture with:

```bash
cargo run -- compile-hf-state-input fixtures/qwen3.5-0.8b/config.json --page-tokens 16 --kv-dtype-bytes 2
cargo run -- compile-hf-state-plan fixtures/qwen3.5-0.8b/config.json --page-tokens 16 --kv-dtype-bytes 2
cargo run -- compile-hf-token-manager-plan fixtures/qwen3.5-0.8b/config.json --page-tokens 16 --kv-dtype-bytes 2
```

For the HF fixture, `compile-hf-state-input` emits the consumable
`ORBITKV_STATE_PLAN` input, `compile-hf-state-plan` shows its compiled backend
contracts, and `compile-hf-token-manager-plan` emits only the Full KV
projection.
The corresponding checked-in runtime inputs are
`examples/qwen3.5-0.8b-attention-state-input-page16-bf16.json` and
`examples/qwen3.5-0.8b-token-manager-page16-bf16.json`. The H20 runner uses a
separate `preflight --scope qwen35`, so the historical sealed Full/Full+SWA
preflight does not acquire a Qwen3.5 checkpoint dependency.
