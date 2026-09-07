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
TTFT/TPOT/ITL/throughput metrics. Detailed output lengths must equal the
requested generation length, their sum must equal `total_output_tokens`, and
every per-request error must be empty. These gates are mandatory because a
streaming client can otherwise count an initial empty SSE frame followed by an
error frame as a completed request. The generated paired summary reports
candidate-over-baseline ratios. When the client emits detailed generated text,
it also checks an output digest for each epoch; otherwise output equivalence is
explicitly marked unevaluated. Raw metrics are never automatically promoted to
a performance claim. Ratios above one favor the candidate for throughput;
ratios below one favor it for latency. The run manifest records the benchmark
client version when the client exposes one.

## Current single-process load closure

A same-instance release-mode run of the released Full+Sliding checkpoint held
the request trace fixed at 16 requests, 127 observed input tokens, and 256 output
tokens per request while sweeping C1/C2/C4/C8. Every arm passed the strengthened
completion gate. Output throughput was 184.24, 350.59, 448.43, and 518.88
token/s. Median TTFT was 163.06, 296.07, 574.80, and 1127.81 ms; median TPOT was
4.80, 4.51, 6.70, and 11.06 ms. This establishes executable capacity through
C8 and a clear throughput/latency frontier; it does not locate the failure point
or establish a win over another engine.

Cross-batch-size generated text is not a correctness gate by itself for BF16
greedy decoding near tied logits. The direct executor qualification instead
teacher-forces the same inputs: all eight B=8 rows are bit-identical, B=1 versus
B=8 has maximum absolute logit difference 0.4296875 over 16 positions, and no tested
argmax differs. The C2/C4/C8 text digests match; C1 differs and is reported, not
hidden.

## Current stock-SGLang product comparison

The released Full+Sliding checkpoint was compared with clean stock SGLang
v0.5.17 for four alternating epochs. Both arms used BF16 weights/KV, page size
16, 1024-token context, an eight-request/8192-token logical capacity, greedy
sampling, and the same 16-request 127-to-256-token trace at C2. Radix prefix
reuse was disabled because the trace contains no intentional shared prefix.
Each arm passed the full-output and per-request error gates.

OrbitKV loaded one strict selected-schedule artifact in every measured epoch;
candidate logs contain no search and all candidate digests match across starts.
Median output throughput was 592.32 token/s versus 1112.60 for SGLang
(0.534x). Median TPOT was 2.997 versus 1.699 ms (1.76x), and median TTFT was
98.08 versus 15.14 ms (6.50x). The candidate is therefore not serving-speed
competitive on this trace.

The persistent-state result points in the other direction. SGLang reports that
hybrid SWA memory is disabled for this Gemma3 path, so its resolved 8192-token
pool carries 144.0 MiB of BF16 K/V tensor payload across all 18 layers. OrbitKV
uses separate Full and Sliding arenas totaling 85.875 MiB, 40.4% less. These are
geometry-derived K/V payload bytes, not allocator peak.

Both engines match the existing independent eight-token reference probe, but
their full random-trace text digests differ. Consequently this is retained as a
product diagnostic and negative performance result, not a matched-output
benefit claim.

## Compiler-constrained schedule follow-up

R4.1 moved the persistent K/V address contract into Luminal candidate
selection. All selected buckets must resolve every K/V output directly to its
registered input arena; candidates and stored artifacts that require copy-back
are rejected before deployment. A 16-candidate search produced two buckets
with 36/36 K/V tensors in place and zero copy-back bytes.

Against the prior two-candidate OrbitKV artifact, four alternating C2 epochs
improved median throughput from 593.57 to 679.52 token/s (+14.4%), TTFT from
98.14 to 66.68 ms (-32.0%), TPOT from 2.984 to 2.682 ms (-10.1%), and E2E from
858.93 to 750.56 ms (-12.6%). Both arms were deterministic across their own
epochs, but their random-trace text digests differ. The new artifact passes the
existing independent B2 eight-token reference probe.

The corresponding four-epoch stock-SGLang comparison remains negative:
OrbitKV reached 678.64 versus 1135.11 token/s (0.598x), with 2.685 versus
1.692 ms TPOT (1.59x), 66.92 versus 14.28 ms TTFT (4.69x), and 751.75 versus
444.74 ms E2E (1.69x). This improves the previous executor baseline but does
not qualify an OrbitKV-over-SGLang serving advantage.

## Promotion rule

Only reviewed runs move from `.qualification/` to `results/`. A promoted result
contains environment identity, raw client JSON, a summary, and checksums—never a
source checkout, build directory, model weights, or dependency cache. A positive
claim requires all predeclared correctness, sample-count, confidence, and
regression gates to pass. Failed experiments remain valid compact evidence when
they inform an architectural decision.
