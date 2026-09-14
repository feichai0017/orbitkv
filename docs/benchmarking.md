# Model benchmarks

Use **`vllm bench serve`** as the common HTTP benchmark client for OrbitKV,
vLLM and SGLang. Publish model inference measurements under `results/`;
keep raw experiments, compiler traces and correctness diagnostics in ignored
`.qualification/`.

## Shared workload

Every comparison fixes the checkpoint revision and full file hashes, tokenizer,
weight/activation/KV precision, device count, sequence/admission budgets,
sampling parameters, client version and request trace. Preserve the full server
commands: different engines need different configuration flags and may reserve
different amounts of memory. Record those differences rather than treating
similarly named flags as proof of identical behavior.

[model-serving.json](../benchmarks/model-serving.json) defines the current bounded
matrix: input lengths 4/32, output lengths 64/128, C1/C8 and 16 requests per run.
The [frozen traces](../benchmarks/traces/) use the official `timed_trace` dataset:
one integer hash expands to one vocabulary token, with Python's hash seed fixed
to the common client seed. The client sends token IDs directly and uses
`--no-self-timed` to apply the declared concurrency/request rate. Record the
expanded token IDs and client/tokenizer versions with each published report.
Temperature 0 and ignored EOS are common to all engines. Disable prefix/radix reuse for this
fresh-prompt comparison and record each engine's CUDA Graph/provider settings.

Random text is insufficient to fix model inputs across tokenizer implementations.
The first three-engine matrix reported 24–32 tokens in OrbitKV versus 32 in the
baselines; it failed the input-length gate. Retain such attempts as diagnostics,
and never relabel them as a matched trace. Older single-engine random-text
measurements preserve their original input lengths and remain separate.

With the Python client, pass both:

```sh
--bench-arg=--num-warmups --bench-arg=0 \
--bench-arg=--ready-check-timeout-sec --bench-arg=0
```

This avoids unmeasured generation probes. Engine-owned preparation still occurs
before readiness. Retain the first measured request and all latency outliers.
Disk caches are retained unless an experiment explicitly uses isolated caches;
a new process does not mean a cold compiler cache.

## Three-engine comparison

`tools/run_serving_comparison.py` takes a JSON server manifest. Commands are
argument arrays, executed without a shell. Each server has a unique name and
base URL; an OrbitKV server declares `"lifecycle": "orbitkv"` to enable startup
and final state-drain checks. For example:

```json
{
  "servers": [
    {"name": "orbitkv", "command": ["/absolute/path/orbitkv-serve", "--model", "/models/checkpoint", "--port", "8010"], "base_url": "http://127.0.0.1:8010", "lifecycle": "orbitkv"},
    {"name": "vllm", "command": ["/absolute/path/vllm", "serve", "/models/checkpoint", "--port", "8030"], "base_url": "http://127.0.0.1:8030"},
    {"name": "sglang", "command": ["/absolute/path/python", "-m", "sglang.launch_server", "--model-path", "/models/checkpoint", "--port", "8020"], "base_url": "http://127.0.0.1:8020"}
  ]
}
```

This illustrates the manifest format. Add the shared served-model name, precision,
capacity and prepared artifact settings before running. The published model
report includes the complete measured commands.

```sh
python tools/run_serving_comparison.py \
  --servers-file /absolute/path/servers.json \
  --model served-model-name --tokenizer /models/checkpoint \
  --profiles-file benchmarks/model-serving.json \
  --trace-dir benchmarks/traces \
  --profile short-c1 --profile short-c8 \
  --profile context-c1 --profile context-c8 \
  --epochs 3 --vllm-command /absolute/path/vllm --client-style python \
  --memory-device 0 --startup-timeout-seconds 1200 \
  --identity-file /absolute/path/orbitkv-serve \
  --identity-file /absolute/path/decoder.json \
  --bench-arg=--num-warmups --bench-arg=0 \
  --bench-arg=--ready-check-timeout-sec --bench-arg=0
```

Only one server runs on the GPU at a time. Engine order rotates each epoch;
the epoch count must be a multiple of the engine count so each occupies every
position equally. Each engine starts a fresh process per epoch and runs the
profiles in declared order. Later profiles share that process's warmed state.
Compiler/provider disk caches persist between processes. Record readiness
separately; it is excluded from the HTTP benchmark duration.

The manifest, workload file, traces and explicit `--identity-file` inputs are hashed
before and after measurement. Pin binaries, artifacts and relevant configuration;
archive package versions and source provenance alongside them. `--dry-run`
prints the plan. Completed sessions are saved incrementally; a failed session
cannot produce an overall completed comparison.

For a single-engine regression use [run_model_serving.py](../tools/run_model_serving.py).
For a two-arm ablation use [run_matched_serving.py](../tools/run_matched_serving.py),
which alternates candidate/baseline order over an even epoch count. All three
entry points share client construction and complete-request validation.

## Gates and interpretation

- Require every request to complete with no per-request error, exactly the
  requested output length and a matching total token count. An initial empty
  SSE frame followed by an error must not count as a valid completion.
- Verify input lengths and ordering across runs. Preserve generated texts and
  their digests; report repeatability and cross-engine agreement separately.
  Text disagreement needs [teacher-forced logits](logit-diagnosis.md), not a
  relaxed tolerance or a claim that every difference is a near tie.
- Inspect shutdown status. Reject forced SIGKILL. OrbitKV additionally requires
  admitted/completed counts to match and no queued/active requests or live
  token/fixed-state owners. Other engines do not expose this OrbitKV census.
- Report output tokens/s, TTFT, TPOT, ITL and end-to-end latency, including P95/P99.
  Summaries use medians of per-run metrics and retain ranges. A median of run
  P95 values is not a pooled P95 or a population-tail estimate.
- Keep process RSS, process GPU memory and device-wide GPU usage distinct.
  Missing process measurements remain null; periodic samples are not allocator
  peaks. Device-wide usage alone does not establish a KV-memory advantage.

Numerical preflight runs separately with frozen artifact/reference identities.
A short probe qualifies its recorded inputs and shapes, not arbitrary generated
text. Disable stage and per-node profiling for serving measurements; see
[stage tracing](stage-tracing.md) and [search coverage](search-coverage.md).

## Publication

Publish one compact model report with reviewed aggregates, individual client
results, environment/checkpoint identity, commands and checksums. The website
imports `results/*/performance.json`, keeping displayed numbers tied to data.
Historical model measurements remain associated with their original builds.
Superseded development journals are recoverable from Git history.

The bounded matrix establishes diagnostic timing observations. An overall
performance-win claim additionally needs representative long-context/pressure
workloads, independent correctness, adequate repeated samples and confidence
bounds exceeding both tuned baselines without P95 TTFT/TPOT regressions.
State-policy and external-tier ablations must hold compute semantics fixed and
include lifecycle, transfer and overlap costs respectively.
