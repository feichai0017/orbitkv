# Single-node offload: final results

H20, BF16 Qwen3-8B, TP=1, vLLM 0.29.0 and SGLang 0.5.20.
Model revision: `b968826d9c46dd6066d109eabc6255188de91218`.

`summary.csv` retains aggregate results from the final large-working-set
comparison. Each row identifies the implementation, prefill batch limit,
latency, throughput, transfer bytes, sampled budget peaks and resource drain.
Raw manifests, responses, traces and logs stay in ignored
`benches/results/runs/20260923-offload-*` directories. See the
[analysis and measurement limits](../../../docs/ssd-performance.md).

The baseline is source `4712f780c900120719f178b2ea36c9e0ac7c135f` with the
sustained context-length and prefill-batch harness fixes backported. Its
prebuilt Manager was built at `0877c662`; the local offload sources are
identical to the baseline checkout. The queue/fence control is `cb74770c`,
including independent total-budget metrics. The final SSD save/restore
allocation policy is `db61df96`. Candidate summaries
check those metrics; baseline phase-summed peaks are labelled separately.
The same source-built Python extension is used throughout; the client ABI
does not change.

Each window admits requests for 60 seconds and drains all admitted requests.
The request cap is 10,000 and is never reached. There are 32 alternating 4K/8K
prefixes, a 75% chance of selecting a prepared prefix, concurrency eight, and
16 generated tokens. Their nominal KV footprint is 27 GiB, exceeding 9 GiB
HBM plus 4 GiB Manager DRAM. SSD capacity is 64 GiB; query admission is 3 GiB.
Preparation of the prefix references is outside timing; cold requests create
new state throughout the window. Reuse selection is not proof of a cache hit.

Build the release Manager before starting any runtime in the checkout:

```bash
cargo build --release -p orbitkv-server --bin orbitkv-cache-manager \
  --no-default-features --features cuda-13,mooncake
```

Then run the matching source checkout and engine environment:

```bash
ORBITKV_CACHE_MANAGER_BINARY="$PWD/target/release/orbitkv-cache-manager" \
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload sustained --lengths 4096 8192 --concurrencies 8 \
  --gpu-tokens 65536 --prefill-tokens 8192 --host-gib 4 --ssd-gib 64 \
  --query-budget-gib 3 --working-set 32 --duration-seconds 60 --max-requests 10000 \
  --read-batch-mib 0 --queue-warmup off --prepare-requests off \
  --trace-transfers --seed 20260923 \
  --output benches/results/runs/offload-vllm
```

Use the SGLang environment and `--engine sglang` for its comparison. Run engines
sequentially. Repeat with `--prefill-tokens 4096` in a new output directory to
measure the latency/throughput tradeoff. Use separate, already-built source
checkouts and explicit Manager paths for before/after runs; do not build or
restage native libraries while a Manager has them mapped. Final code comparison
uses the same 8K prefill limit. A different compute batch is a configuration
experiment, not evidence of a code-only speedup.

`python -m benches.report RUN... --output EMPTY_DIRECTORY` rebuilds ordinary
per-run reports. The retained CSV selects the final comparison rows and adds
the process-local timeline summaries. Keep repeated requests and output
differences; do not select only fast prefixes or successful hits. The
[previous ordinary-recovery report](https://github.com/feichai0017/orbitkv/tree/4712f780c900120719f178b2ea36c9e0ac7c135f/benches/results/20260922-recovery-baseline)
remains available as an immutable snapshot.
