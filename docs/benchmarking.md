# Matched serving benchmarks

OrbitKV uses `vllm bench serve` as a common OpenAI-compatible client for both
the OrbitKV candidate and the reference server. Sharing the client removes one
source of measurement drift; it does not by itself make the systems comparable.

## Required experiment layers

1. Correctness preflight: deterministic greedy outputs and lifecycle final drain
   must pass before timings are interpreted.
2. Compiler ablation: run conservative retention and compiled retention through
   the same OrbitKV/Luminal executor. This isolates the compiler contribution.
3. Product comparison: run OrbitKV/Luminal and tuned stock SGLang with the same
   model, weights, dtype, kernels where possible, request trace, batching limits,
   device budget, and sampling semantics.
4. Tier comparison: separately compare cold prefill, local retention, and
   external restore. Transport results must include copy cost and overlap.

## Standard workload families

| Profile | Purpose | Required observations |
| --- | --- | --- |
| decode-heavy | Expose steady token latency and graph dispatch | TPOT, ITL, output throughput, graph hit/recapture |
| prefill-heavy | Exercise TTFT and restore-vs-recompute | TTFT, prefill time, restored bytes, cache hit |
| hybrid-pressure | Cross Sliding retirement boundaries under concurrency | RA, resident bytes, admission failures, p95/p99 latency, reclaimed pages |
| capacity-sweep | Hold the device budget fixed and increase concurrency/context | maximum admitted requests, OOM/failure point, throughput |

For the compiler ablation, both arms must use identical Luminal graphs and
kernels. Only the retention/layout policy may differ:

```text
conservative: retain token state until request release
compiled:     use the manifest-derived retirement and placement program
```

The implementation exposes these as `PhysicalResidencePolicy::RequestLifetime`
and `PhysicalResidencePolicy::Compiled`. The default constructor always selects
`Compiled`; the conservative variant is available only through
`CanonicalKvManager::new_with_residence`. Both arms preserve the same compiled
attention visibility. The executor must compare CSR geometry and output values,
not physical page IDs, because correct physical plans may bind different pages.

A first H20 mechanism check now compiles one Luminal graph and executes both
arms at an 80-token boundary plus one decode under an in-memory interleaved
Full/Sliding policy. Token IDs and logits matched exactly in both phases; after
decode, compiled residence used 5 active Sliding pages / 491,520 bytes and
request-lifetime residence used 6 / 589,824 bytes. This is qualification of the
ablation seam, not a statistically matched performance result. Raw repeated
workload measurements still belong under `.qualification/` until the promotion
rule below passes.
The byte counts are live manager payload within equal preallocated arenas; they
represent reusable admission headroom, not an immediate CUDA allocator release.

## Harness

`tools/run_matched_serving.py` starts candidate and baseline sequentially in an
alternating order and invokes the same vLLM benchmark client for every run. It
accepts both the Python `vllm bench serve` CLI and the Rust `vllm-bench` binary.
It writes unreviewed data under `.qualification/` by default. The command fails
if the client is missing, a server never becomes ready, a benchmark exits
non-zero, or the expected JSON is absent.

Example, after a runnable OrbitKV engine binary is available:

```bash
python tools/run_matched_serving.py \
  --candidate-command 'target/release/orbitkv-server --model /models/model --port 8000' \
  --baseline-command 'python -m sglang.launch_server --model-path /models/model --port 8000' \
  --candidate-url http://127.0.0.1:8000 \
  --baseline-url http://127.0.0.1:8000 \
  --model served-model-name \
  --tokenizer /models/model \
  --profile hybrid-pressure \
  --epochs 4 \
  --vllm-command vllm \
  --client-style python
```

Use an even epoch count. Odd epochs run baseline first and even epochs run the
candidate first. Do not run both servers concurrently on the same accelerator.
The harness defaults to deterministic greedy generation and `--ignore-eos`.
It fixes the random dataset seed and explicitly requests p95/p99 for TTFT,
TPOT, ITL, and end-to-end latency. It also fixes the random length range to zero
instead of relying on a client-version default.
Every run must report all requested completions, zero failures, and numeric
TTFT/TPOT/ITL/throughput metrics. The generated paired summary reports
candidate-over-baseline ratios. When the client emits detailed generated text,
it also checks an output digest for each epoch; otherwise output equivalence is
explicitly marked unevaluated. Raw metrics are never automatically promoted to
a performance claim. Ratios above one favor the candidate for throughput;
ratios below one favor it for latency. The run manifest records the benchmark
client version when the client exposes one.

## Promotion rule

Only reviewed runs move from `.qualification/` to `results/`. A promoted result
contains environment identity, raw client JSON, a summary, and checksums—never a
source checkout, build directory, model weights, or dependency cache. A positive
claim requires all predeclared correctness, sample-count, confidence, and
regression gates to pass. Failed experiments remain valid compact evidence when
they inform an architectural decision.
