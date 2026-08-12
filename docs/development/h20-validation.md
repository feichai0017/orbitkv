# H20 validation

NVIDIA H20 is Oxide Infer's first device target. Product providers live in
`crates/oxide-infer-cuda`. Hardware gates live in `crates/oxide-infer-lab`.

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
cd <oxide-infer-checkout>
git status --short
git rev-parse HEAD

make cuda-doctor
make cuda-check
make cuda-test
make h20
```

The Make targets are the canonical entry points. Evidence records include the
expanded commands. `cuda-test` accepts `CUDA_ARCH` for generic device tests.
Every H20 correctness and benchmark target fixes `H20_ARCH` to `sm_90a`.

Run one gate during development:

| Target | Permanent runner |
| --- | --- |
| `make h20-rms-norm` | `rms_norm_h20` |
| `make h20-gemm` | `bf16_gemm_h20` |
| `make h20-oxide-gemm` | `oxide_sm90_simt_gemv_h20` |
| `make h20-attention` | `single_decode_h20` |
| `make h20-paged-attention` | `paged_batch_decode_h20` |
| `make h20-ragged-prefill` | `ragged_prefill_h20` |
| `make h20-paged-prefill` | `paged_prefill_h20` |
| `make h20-rope` | `rope_h20` |
| `make h20-engine-interop` | `engine_interop_h20` |

`make cuda-test` writes `oxide_infer_cuda.ptx` at the workspace root. Record
its hash and assemble it for the selected `CUDA_ARCH` with the same CUDA
toolkit used by the run. This generic artifact does not qualify an H20 target.

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
| Paged causal prefill, long | Same contract with partitioned F32 state | MQA `(3,21,64,8,1)` uses sixteen warps. GQA4 `(2,96,96,16,4)` uses tiled split-four |

Paged metadata uses I32 `page_indptr`, `page_indices`, and `last_page_len`.
Cover mixed lengths, partial tails, physical-page order, and read-only page
reuse. Invalid page indices must preserve the affected request's sentinels.

Ragged requests satisfy `1 <= qo_len <= kv_len`. The causal rule is:

```text
kv_index <= kv_len - qo_len + query_index
```

The current ragged runner selects direct for the three short cases, sixteen
warps for long MQA, and tiled split-four for long GQA4. Earlier records cover
the eight-warp stage, but the current runner needs a dedicated eight-warp case
before that path gains current-source qualification.

The tiled GQA4 workspace is F32
`[nnz_qo, query_heads, 4, 130]`. One completion covers the partial and merge
kernels. A missing workspace must fail before CUDA submission.

Paged prefill requires an explicit algorithm at plan creation. The gate uses
direct for short cases, sixteen warps for long MQA, and tiled split-four for
long GQA4. The tiled plan requires an explicit F32 workspace and rejects a
missing binding before submission.

The token-parallel plans keep partial F32 online-softmax states in block-local
shared memory and need no caller workspace.

Paged decode and direct or token-parallel paged prefill submit three commands:
one validator, one attention kernel, and one status copy. Tiled paged prefill
submits four commands because partial and merge are separate attention kernels.
Invalid metadata must preserve sentinels, return bindings, and leave the queue
or fixed-address Graph reusable.

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

Each mapped-append stage uses three commands: one validator kernel, one mapped
append kernel, and one device-to-host status copy. The valid-output Graph
captures three stages, for nine Oxide commands in total.

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

Build each admitted runner once, then run all four tools against that exact
binary. For example:

```bash
set -euo pipefail
make h20-build-runner H20_RUNNER=rms_norm_h20 2>&1 | tee h20-build.log
runner_sha="$(sed -n 's/^runner_binary=.* sha256=\([0-9a-f]\{64\}\) arch=.*/\1/p' h20-build.log)"
test "${#runner_sha}" -eq 64
make h20-sanitize-runner H20_RUNNER=rms_norm_h20 \
  H20_RUNNER_SHA256="$runner_sha"
```

The build target uses `sm_90a` with device line information and prints the
runner's SHA-256 hash. Pass that exact hash to the sanitizer target. The
sanitizer target has no build dependency.

It rejects a different binary and invokes `compute-sanitizer` directly for
memcheck, racecheck, synccheck, and initcheck. It verifies the hash after each
tool. Memcheck includes full leak checking. Each tool uses error exit code 99.

Do not use `cargo oxide sanitize` for qualification. That shortcut can rebuild
the runner, so it cannot prove that all four reports cover one recorded
artifact.

For Graph gates, include capture, replay, completion settlement, and graph
destruction in the sanitizer process. A clean sanitizer run does not replace
the numerical oracle.

## Fixed-address Graph gate

The current Graph contract fixes device addresses and uses a private
non-default capture stream. It rejects rebinding, graph updates, cross-stream
launch, concurrent replay, and default-stream capture.

Current Graph cases are:

Every valid-output case captures three stages: an independent poison observer,
a target-poison stage, and the real stage. The last two stages write the same
checked addresses in order during every replay.

| Operator | Commands per stage | Captured total | Fixture |
| --- | ---: | ---: | --- |
| RMSNorm to GEMM | 2 | 6 | `(1,4096)` RMSNorm into `(1,4096,4096)` GEMM |
| Ragged prefill | 2 | 6 | Long tiled GQA4 `(2,96,1280,16,4)` |
| Paged decode rejection | Not applicable | 3 | Invalid page index with two reusable replays |
| Paged prefill | 4 | 12 | Long tiled GQA4 `(2,96,96,16,4)` |
| Fused append | 3 | 9 | Validator, six-token mapped append, and status copy under the exclusive-page contract |

The paged-decode row is a sentinel-preserving rejection Graph. It does not
qualify a valid-output Graph path.

A current Graph qualification must:

1. Compare the standalone result with the independent reference.
2. Capture a target-poison stage before the real stage in every valid replay.
3. Replay and prove that the real stage replaces the poison.
4. Drop external plan and read owners before replay.
5. Exercise explicit `wait()` and completion-drop settlement.
6. Record capture count, command count, replay count, and binding policy.
7. Run all four sanitizer tools.

Captured command counts come from the Oxide command plan. Benchmark
`graph_nodes` fields are handwritten metadata, not CUDA driver enumeration.
Do not use them as verified node counts.

Earlier attention Graph records did not use the current poisoned-replay and
strict-summary policy. Treat their correctness and performance claims as
historical until a new record passes this gate.

## External device regions

The `engine_interop_h20` runner creates external allocations and a non-default stream in a simulated engine module.
It checks five single-decode pointers and nine HND paged-decode pointers, provider identity, and zero adapter device-to-device copies.
It keeps two completions in flight and verifies backpressure and retry.
It also settles one completion on another host thread and checks a typed paged-metadata rejection with queue reuse.
It executes the pre/post event bridge but is not a negative-control proof of post-wait causality.

The handoff returns stream-ordered engine authority before host completion and keeps the Oxide bindings private.
The gate enqueues engine readback, settles Oxide work, and drops the engine storage guards.
It then waits on the readback's own source lease.

The runner reports `boundary=simulated_engine`.
It does not qualify a model runner.
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
- for valid Graph cases, a captured target-poison stage before the real stage
  inside each correctness replay.
- strict summaries that reject incompatible contract, run, or execution
  metadata.

This cross-provider oracle check is not yet published. Do not call the current
references one common oracle until a gate compares their fixture digests.

Run each provider in a separate process. A paged-decode example is:

```bash
OXIDE_BENCH_RUN_LABEL=oxide_first \
  make bench-paged-oxide > oxide-first.jsonl
OXIDE_BENCH_RUN_LABEL=flashinfer_second \
  OXIDE_BENCH_OPERATORS=paged_batch_decode \
  python3 tools/flashinfer/matched_bench.py > flashinfer-second.jsonl

OXIDE_BENCH_RUN_LABEL=flashinfer_first \
  OXIDE_BENCH_OPERATORS=paged_batch_decode \
  python3 tools/flashinfer/matched_bench.py > flashinfer-first.jsonl
OXIDE_BENCH_RUN_LABEL=oxide_second \
  make bench-paged-oxide > oxide-second.jsonl

python3 tools/flashinfer/summarize.py \
  oxide-first.jsonl flashinfer-second.jsonl \
  flashinfer-first.jsonl oxide-second.jsonl > summary.json
```

Use these Oxide targets for the other admitted measurements:

| Measurement | Oxide target | FlashInfer script |
| --- | --- | --- |
| All eager cases | `make bench-oxide` | `tools/flashinfer/matched_bench.py` |
| Ragged prefill eager | `make bench-ragged-oxide` | `tools/flashinfer/matched_bench.py` |
| Paged prefill eager | `make bench-paged-prefill-oxide` | `tools/flashinfer/matched_bench.py` |
| Ragged prefill Graph | `make bench-ragged-graph-oxide` | `tools/flashinfer/ragged_graph_bench.py` |
| Paged prefill Graph | `make bench-paged-prefill-graph-oxide` | `tools/flashinfer/paged_prefill_graph_bench.py` |
| Standard RoPE eager | `make bench-rope-oxide` | `tools/flashinfer/matched_bench.py` |
| Fused append eager | `make bench-rope-append-tokens-oxide` | `tools/flashinfer/matched_bench.py` |
| Fused append Graph | `make bench-rope-append-tokens-graph-oxide` | `tools/flashinfer/rope_append_graph_bench.py` |

Keep eager and Graph samples in separate summaries. Keep different algorithms
and run labels separate. Exclude a ranking when either provider's order median
changes by more than five percent.

The native M=1 GEMV decision uses two Oxide Infer benchmark targets rather
than a FlashInfer script: `make bench-sm90-gemv-oxide` and
`make bench-sm90-gemv-cublaslt`. Run them as Oxide/cuBLASLt and
cuBLASLt/Oxide process pairs with distinct run labels. The separate matched
Mistral.rs custom-GEMV pair is complete. Only one of five shapes passed both
10% margins, so the exact native candidate is performance-stopped and does not
advance to an engine gate.

## Evidence records

Store reviewed JSON records in [the evidence directory](../results/README.md).
Each record includes source identity, environment, contract, commands, artifact
hashes, accepted claims, excluded claims, and raw samples when applicable.

Use `h20-<operator>-<gate>-YYYYMMDD.json`. Do not edit a reviewed file. A
changed source, contract, provider, toolchain, oracle, poison policy, or summary
schema requires a new record.
