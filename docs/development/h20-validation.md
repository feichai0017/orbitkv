# H20 validation

NVIDIA H20 is Loom Infer's first device target. Product providers live in
`crates/loom-infer-cuda`. Hardware gates live in
`crates/loom-infer-validation`.

## Evidence boundaries

Validate each boundary separately.

| Boundary | Required proof |
| --- | --- |
| Host contract | Rust validation, CPU or independent reference, and error behavior |
| Device correctness | H20 output, numerical limit, sentinels, and exact admitted cases |
| Lifecycle | Stream order, retained resources, completion, and failure settlement |
| Sanitizer | Declared Compute Sanitizer tools over the permanent runner |
| Graph | Capture, poisoned replay, fixed-binding policy, and completion behavior |
| Performance | Matched contract, independent oracle, raw samples, and both provider orders |
| Engine | Real call site, no-copy proof, provider hit count, and model output |

A lower boundary never proves a higher one. A result applies only to its
recorded source tree, contract, toolchain, artifact, and device.

## Canonical commands

Use one checkout for source, build, and execution.

```bash
ssh <h20-host>
cd <loom-infer-checkout>
git status --short
git rev-parse HEAD

make cuda-doctor
make cuda-check
make cuda-test
make h20
```

The Make targets are the canonical entry points. Evidence records include the
expanded commands.

Run one gate during development:

| Target | Permanent runner |
| --- | --- |
| `make h20-rms-norm` | `rms_norm_h20` |
| `make h20-gemm` | `bf16_gemm_h20` |
| `make h20-attention` | `single_decode_h20` |
| `make h20-paged-attention` | `paged_batch_decode_h20` |
| `make h20-ragged-prefill` | `ragged_prefill_h20` |
| `make h20-paged-prefill` | `paged_prefill_h20` |
| `make h20-rope` | `rope_h20` |
| `make h20-engine-interop` | `engine_interop_h20` |

`make cuda-test` writes `loom_infer_cuda.ptx` at the workspace root. Record
its hash and assemble it for `sm_90` with the same CUDA toolkit used by the
run.

Before a run, record:

- source commit, tree state, and `Cargo.lock` hash.
- Rust nightly and cuda-oxide revision.
- CUDA toolkit, driver, LLVM, and Clang versions.
- GPU model, compute capability, clocks, and power policy.
- PTX or cubin target and content hash.

Do not qualify a remote-only patch or a copied artifact from another source
tree.

## Common correctness rules

Attention and RoPE use BF16 storage with F32 reference arithmetic. Attention
output maximum absolute error is `0.015625`. Log2-LSE maximum absolute error is
`0.01`. Initialize every output with a NaN sentinel.

RMSNorm limits are:

| Dtype | Maximum absolute error | Additional limit |
| --- | --- | --- |
| F32 | `5e-5` | All outputs finite |
| FP16 | `4e-3` | At most two storage ULPs |
| BF16 | `4e-2` | At most two storage ULPs |

Reject short fixed spans, wrong contexts, and duplicate writable bindings
before CUDA submission.

Paged decode, paged prefill, and fused append validate device metadata on the
CUDA stream. Each validator fully overwrites its status packet. Semantic
rejection returns a typed completion error and preserves checked bindings.

## RMSNorm gate

The F32 runner covers:

```text
(rows, hidden) = (1,1), (3,127), (8,4096), (16,8192)
```

The FP16 and BF16 runner covers:

```text
scalar: (1,1), (3,127), (3,4097)
packed: (1,2), (32,256), (8,4096), (16,8192), (1,11008)
```

The gate also checks short buffers, signed zero, scalar and packed dispatch,
typed bindings, a non-default stream, queue reuse, chained commands, and one
partial-scope rejection.

## BF16 GEMM gate

The provider fixes this cuBLASLt contract before enqueue:

```text
D[M,N] = A[M,K] * W[N,K]^T
A, W, D: contiguous row-major BF16
accumulation: F32
```

The runner covers `(M,N,K) = (1,4096,4096)` and the transpose-sensitive
`(2,3,4)` case. It checks exact spans, command capacity, reusable scopes, and
the selected algorithm's actual workspace requirement.

The same runner checks one RMSNorm-to-GEMM command chain and one fixed-address
Graph:

```text
BF16 RMSNorm (1,4096)
  -> BF16 cuBLASLt GEMM (1,4096,4096)
```

## Attention gates

All admitted attention contracts fix NHD layout and head dimension 128. Scores,
online-softmax state, split states, and merge arithmetic use F32.

| Gate | Storage and mask | Exact current cases |
| --- | --- | --- |
| Single decode direct | `Q/O [qh,128]`, `K/V [kv_len,kvh,128]`, full attention | `(kv_len,qh,kvh)`: `(1,8,8)`, `(33,8,1)`, `(127,16,4)`, `(4096,32,4)` |
| Single decode split-K | Same tensors plus caller F32 workspace `[qh,partitions,130]` | `(kv_len,qh,kvh,p)`: `(7,8,1,3)`, `(33,8,1,12)`, `(127,16,4,16)`, `(4096,32,4,64)` |
| Paged batch decode | `Q/O [batch,qh,128]`, pages `[max_pages,16,kvh,128]`, full attention | MHA `(1,2,8,8)`, MQA `(3,7,8,1)`, GQA `(4,8,16,4)` as `(batch,max_pages,qh,kvh)` |
| Ragged causal prefill | Contiguous Q/K/V with separate I32 `qo_indptr` and `kv_indptr` | `(batch,nnz_qo,nnz_kv,qh,kvh)`: `(1,4,4,8,8)`, `(3,6,13,8,1)`, `(2,6,11,16,4)`, `(3,21,896,8,1)`, `(2,96,1280,16,4)` |
| Paged causal prefill | Ragged Q plus page-size-16 KV and bottom-right causal mask | Short: `(batch,nnz_qo,max_pages,qh,kvh)` is `(1,4,2,8,8)`, `(3,6,7,8,1)`, or `(2,6,6,16,4)` |
| Paged causal prefill, long | Same contract with block-local token partitioning | MQA `(3,21,64,8,1)` uses sixteen warps. GQA4 `(2,96,96,16,4)` uses eight warps |

Paged metadata uses I32 `page_indptr`, `page_indices`, and `last_page_len`.
Cover mixed lengths, partial tails, physical-page order, and read-only page
reuse. Invalid page indices must preserve the affected request's sentinels.

Ragged requests satisfy `1 <= qo_len <= kv_len`. The causal rule is:

```text
kv_index <= kv_len - qo_len + query_index
```

The current ragged runner selects direct for the three short cases, sixteen
warps for long MQA, and tiled split-eight for long GQA4. Earlier records cover
the eight-warp stage, but the current runner needs a dedicated eight-warp case
before that path gains current-source qualification.

The tiled GQA4 workspace is F32
`[nnz_qo, query_heads, 8, 130]`. One completion covers the partial and merge
kernels. A missing workspace must fail before CUDA submission.

Paged prefill requires an explicit algorithm at plan creation. The gate uses
direct for short cases, sixteen warps for long MQA, and eight warps for long
GQA4. These choices are fixtures, not an automatic runtime policy.

The token-parallel plans keep partial F32 online-softmax states in block-local
shared memory and need no caller workspace.

Paged decode and paged prefill each submit three commands: one validator
kernel, one attention kernel, and one device-to-host status copy. Invalid
metadata must preserve sentinels, return bindings, and leave the queue or
fixed-address Graph reusable.

## RoPE and paged append gate

Standalone RoPE fixes BF16 NHD D128, NeoX split-half rotation, scale one,
theta 10,000, and explicit I32 positions. The current fixture uses five tokens,
16 query heads, four KV heads, and positions `0, 1, 127, 4096, 32767`.

Fused append uses page size 16 and adds one I32 reference count for every
physical page. Every write target must have reference count one. Shared pages
remain legal for read-only prefixes.

The current runner covers:

- one token per request with batch four, 16 query heads, and four KV heads.
- six shuffled explicit tokens with batch three and private target pages.
- the 64-token contract limit with one query and KV head.
- short metadata, duplicate slots, invalid pages, invalid token mappings, and
  shared target rejection.
- one fixed-address six-token Graph case.

One mapped append uses three commands: one validator kernel, one mapped append
kernel, and one device-to-host status copy. The fixed-address Graph case
captures this three-command sequence.

Each map owns one workspace for the full scope. A second map from the same
workspace must fail before submission. The map also binds the exact writable
K/V pages. Using another cache must fail before submission.

The one-token gate also reuses one map for two mapped appends in the same
scope and the same cache binding.

That scope contains four commands: one validator, two mapped appends, and one
status copy.

A semantic rejection must return the exact `ContractError`, preserve every
output sentinel, return the checked bindings, and leave the queue or Graph
reusable.

The engine or KV pager makes a shared tail private before append. The operator
does not allocate, copy, or remap pages. Reference counts and the page table
must describe one stable snapshot through completion.

The fused-append records dated 2026-08-06 predate this exclusive-page contract.
They remain historical and do not qualify current correctness, Graph, or
performance.

The [paged-prefill token-parallel correctness record](../results/h20-bf16-paged-prefill-token-parallel-correctness-20260807.json)
and the [matched long-context record](../results/h20-flashinfer-v0.6.16.post1-paged-prefill-long-eager-performance-20260807.json)
describe clean source commit `8478ee9`. They cover sixteen-warp long MQA and
eight-warp long GQA4.

They do not qualify the merged DeviceRegion submission path.

The DeviceRegion refactor changed submission for every CUDA provider. Existing
RMSNorm, GEMM, decode, prefill, RoPE, and Graph records predate the current
source. Run the matching gates and publish new records before restoring their
device-qualified status.

## Compute Sanitizer

Run all four tools against every admitted runner. Use leak checking with
memcheck.

```bash
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  target/release/<runner>
compute-sanitizer --tool racecheck --error-exitcode 99 \
  target/release/<runner>
compute-sanitizer --tool synccheck --error-exitcode 99 \
  target/release/<runner>
compute-sanitizer --tool initcheck --error-exitcode 99 \
  target/release/<runner>
```

For Graph gates, include capture, replay, completion settlement, and graph
destruction in the sanitizer process. A clean sanitizer run does not replace
the numerical oracle.

## Fixed-address Graph gate

The current Graph contract fixes device addresses and uses a private
non-default capture stream. It rejects rebinding, graph updates, cross-stream
launch, concurrent replay, and default-stream capture.

Current Graph cases are:

| Operator | Captured commands | Fixture |
| --- | --- | --- |
| RMSNorm to GEMM | Two | `(1,4096)` RMSNorm into `(1,4096,4096)` GEMM |
| Ragged prefill | Two | Long tiled GQA4 `(2,96,1280,16,4)` |
| Paged decode rejection | Three | Invalid page index with two reusable replays |
| Paged prefill | Three | Direct GQA4 `(2,6,6,16,4)` |
| Fused append | Three | Validator, six-token mapped append, and status copy under the exclusive-page contract |

A current Graph qualification must:

1. Compare the standalone result with the independent reference.
2. Poison every graph-written output span or append target before replay.
3. Replay and prove that no poison remains in a valid result.
4. Drop external plan and read owners before replay.
5. Exercise explicit `wait()` and completion-drop settlement.
6. Record capture count, command count, replay count, and binding policy.
7. Run all four sanitizer tools.

Captured command counts come from Loom's command plan. Benchmark
`graph_nodes` fields are handwritten metadata, not CUDA driver enumeration.
Do not use them as verified node counts.

Earlier attention Graph records did not use the current poisoned-replay and
strict-summary policy. Treat their correctness and performance claims as
historical until a new record passes this gate.

## External device regions

The `engine_interop_h20` runner creates external allocations and a non-default stream in a simulated engine module.
It checks five unchanged pointers, five external-region leases, provider identity, and zero adapter device-to-device copies.
It executes the pre/post event bridge but is not a negative-control proof of post-wait causality.
The handoff returns opaque stream-ordered authority before host completion and keeps the Loom bindings private.
The gate enqueues engine readback, settles Loom, and drops the engine storage guards.
It then waits on the readback's own source lease.

The runner reports `boundary=simulated_engine`.
It does not qualify a model runner.
HND paged decode and the authority handoff now exist in source.
A real mistral.rs invocation, provider hit, pointer trace, and model-output comparison remain required.

## Matched performance requalification

Existing matched attention and attention-Graph records predate the current
evidence tooling. Keep them immutable as historical records.

Do not reuse their ratios as current claims. This rule includes the 2026-08-07
paged-prefill long-context record.

Before a new matched run:

```bash
make check-tools
```

The new scripts require:

- equivalent F32 attention formulas and matching reference digests across the
  Rust CPU reference and the separate PyTorch reference.
- identical fixture bits and digests across providers and orders.
- identical dtype, layout, shape, mask, caller-visible contract, and timed
  boundary. Record every provider-private workspace and include its required
  device work in the timing.
- preallocated buffers and no compile, tune, allocation, or result transfer in
  the timed interval.
- one distinct status workspace for each logical append call in a batched
  eager timing scope. The final status readbacks must be inside the timed
  interval.
- raw samples from both provider orders.
- a verified provider version and source identity, or an explicit unverified
  source field.
- poisoned Graph outputs before the correctness replay.
- strict summaries that reject incompatible contract, run, or execution
  metadata.

This cross-provider oracle check is not yet published. Do not call the current
references one common oracle until a gate compares their fixture digests.

Run each provider in a separate process. A paged-decode example is:

```bash
LOOM_BENCH_RUN_LABEL=loom_first \
  make bench-paged-loom > loom-first.jsonl
LOOM_BENCH_RUN_LABEL=flashinfer_second \
  LOOM_BENCH_OPERATORS=paged_batch_decode \
  python3 tools/flashinfer/matched_bench.py > flashinfer-second.jsonl

LOOM_BENCH_RUN_LABEL=flashinfer_first \
  LOOM_BENCH_OPERATORS=paged_batch_decode \
  python3 tools/flashinfer/matched_bench.py > flashinfer-first.jsonl
LOOM_BENCH_RUN_LABEL=loom_second \
  make bench-paged-loom > loom-second.jsonl

python3 tools/flashinfer/summarize.py \
  loom-first.jsonl flashinfer-second.jsonl \
  flashinfer-first.jsonl loom-second.jsonl > summary.json
```

Use these Loom targets for the other admitted measurements:

| Measurement | Loom target | FlashInfer script |
| --- | --- | --- |
| All eager cases | `make bench-loom` | `tools/flashinfer/matched_bench.py` |
| Ragged prefill eager | `make bench-ragged-loom` | `tools/flashinfer/matched_bench.py` |
| Paged prefill eager | `make bench-paged-prefill-loom` | `tools/flashinfer/matched_bench.py` |
| Ragged prefill Graph | `make bench-ragged-graph-loom` | `tools/flashinfer/ragged_graph_bench.py` |
| Paged prefill Graph | `make bench-paged-prefill-graph-loom` | `tools/flashinfer/paged_prefill_graph_bench.py` |
| Standard RoPE eager | `make bench-rope-loom` | `tools/flashinfer/matched_bench.py` |
| Fused append eager | `make bench-rope-append-tokens-loom` | `tools/flashinfer/matched_bench.py` |
| Fused append Graph | `make bench-rope-append-tokens-graph-loom` | `tools/flashinfer/rope_append_graph_bench.py` |

Keep eager and Graph samples in separate summaries. Keep different algorithms
and run labels separate. Exclude a ranking when either provider's order median
changes by more than five percent.

## Evidence records

Store reviewed JSON records in [the evidence directory](../results/README.md).
Each record includes source identity, environment, contract, commands, artifact
hashes, accepted claims, excluded claims, and raw samples when applicable.

Use `h20-<operator>-<gate>-YYYYMMDD.json`. Do not edit a reviewed file. A
changed source, contract, provider, toolchain, oracle, poison policy, or summary
schema requires a new record.
