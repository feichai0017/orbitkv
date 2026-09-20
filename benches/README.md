# OrbitKV benchmarks

Run performance experiments from the repository root. Benchmark code, harness
tests, curated measurements, and local raw runs all live here. Runtime Python
code belongs in `python/orbitkv/`; correctness gates belong in `python/tests/`.

## Layout

| Path | Responsibility |
| --- | --- |
| `single_node.py` | Fixed-capacity cold, HBM-hit, and post-pressure experiment |
| `launch.py` | Engine/backend commands and matched memory budgets |
| `runtime.py` | Owned process groups, readiness, teardown, and launch manifest |
| `workload.py` | Token-exact requests, streaming timings, and pressure traffic |
| `metrics.py` | Cache-source evidence and statistical summaries |
| `report.py` | Offline CSV/JSON reports from complete raw runs |
| `serving.sh` | vLLM serving measurements against an already running endpoint |
| `sharegpt.py` | Multi-turn workload using the pinned vLLM benchmark scripts |
| `tests/` | CPU-only checks for measurement and report correctness |
| `results/` | Reviewed CSV/JSON measurements committed to Git |
| `results/runs/` | Ignored raw runs: manifests, responses, counters, logs, and failures |

## Single-node comparison

Use separate environments for the validated vLLM 0.29.0 and SGLang 0.5.20
releases. Follow the [deployment guide](../docs/single-node.md) to build/install
the release wheel. The harness needs `requests`; the selected engine provides
its tokenizer and GPU runtime. LMCache and FlexKV are comparison dependencies,
not OrbitKV package dependencies.

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b

.venv/sglang-release/bin/python -m benches.single_node \
  --engine sglang --backend orbitkv --model /workspace/models/qwen3-8b
```

By default, each run gets a new timestamped directory below `results/runs/`.
Use `--output benches/results/runs/<name>` for an explicit empty directory. The
harness owns the engine and the OrbitKV/LMCache process; do not start other GPU
workloads during measurement. An OrbitKV source run uses the staged Cache Manager
binary in `python/orbitkv/` and the Python adapters from this checkout.

Use `--backend native`, `cpu`, `orbitkv`, `lmcache`, or `flexkv`. Compare within
one engine, using identical model, GPU token capacity, host capacity, request
sequence, and dependencies. Supported model scope is dense Qwen3 in BF16, TP=1.
FlexKV compatibility failures observed on these releases are documented in the
[measurement report](../docs/single-node-performance.md); accepting a backend
option does not mean its current integration can start successfully.

All outputs and cache-source evidence are retained, including mixed hits,
misses, and generated-text differences. A startup/request failure writes
`failure.json` and is not a latency result. The serial workload measures TTFT;
it does not establish concurrent goodput or production tail latency.

## Report existing runs

```bash
python -m benches.report \
  benches/results/runs/<cpu-run> benches/results/runs/<orbitkv-run> \
  --output benches/results/runs/<report-name>
```

This needs only `requests`, not torch, either inference engine, or the native
extension. It writes `summary.csv` and `summary.json` without mixing samples
across runs. Incomplete workloads, duplicate samples, and failed runs are
rejected. Copy reviewed exports into `results/` when publishing a measurement;
keep raw logs and dataset downloads in the ignored `results/runs/` directory.

## Additional workloads and harness checks

```bash
BASE_URL=http://127.0.0.1:8000 MODEL=/path/to/model LABEL=orbitkv \
  bash benches/serving.sh

git submodule update --init third-party/vllm
.venv/vllm-release/bin/python -m benches.sharegpt \
  --model /path/to/model --dataset-path /path/to/sharegpt.json

uv run --isolated --no-project --with pytest --with requests pytest benches/tests
```

The ShareGPT workload requires the dependencies listed by the pinned vLLM
`benchmarks/multi_turn/requirements.txt`. These endpoint workloads have their
own workload definitions; do not combine their numbers with the fixed-capacity
single-node comparison.
