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

A release-mode H20 compiler ablation now executes a released native
Full+Sliding checkpoint for ten paired alternating epochs. Each arm uses the
same searched graph for a 512-token prefill plus 255 decode steps. All 256
generated tokens match between arms. The first 13 stable tokens also match the
independent reference; the next reference choice has a BF16 top-two margin of
only 0.125 and is not used as a cross-search-winner discrete-token gate.
Compiled residence uses 32 Sliding pages versus 48, reducing total live payload
from 14,155,776 to 10,223,616 bytes. Under a fixed 35-Full/33-Sliding-page budget,
it advances to boundary 560 while request-lifetime residence stops at 528. At
the measured boundary, semantic-live payload is 10,205,184 bytes, so Retention
Amplification is 1.002 for compiled residence and 1.387 for the baseline.
Median total test-path time is 1.075590 s versus 1.083926 s; paired mean
improvement is 11.633 ms with a 95% confidence interval of 5.697-17.570 ms.

This qualifies a narrow same-executor lifecycle benefit. The byte counts are
live manager payload within equal preallocated arenas; they represent reusable
admission headroom, not an immediate CUDA allocator release. The timing includes
model execution plus host lifecycle work for one batch-one request and is not a
continuous-batching TTFT/TPOT or throughput result.

## Harness

`tools/run_matched_serving.py` starts candidate and baseline sequentially in an
alternating order and invokes the same vLLM benchmark client for every run. It
accepts both the Python `vllm bench serve` CLI and the Rust `vllm-bench` binary.
It writes unreviewed data under `.qualification/` by default. The command fails
if the client is missing, a server never becomes ready, a benchmark exits
non-zero, or the expected JSON is absent.

Example with the single-process OrbitKV server:

```bash
python tools/run_matched_serving.py \
  --candidate-command 'target/release/orbitkv-serve --model /models/model --page-counts 128,66 --max-model-tokens 1024 --max-prefill-tokens 512 --max-batch-tokens 1024 --max-active-requests 2 --port 8000' \
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
