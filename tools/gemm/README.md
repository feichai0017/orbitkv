# GEMM shape census tools

`shape_census.py` validates and summarizes dense-linear call counts from a
pinned model run. It does not measure latency or select a GEMM provider.

The producer records resolved linear shapes before it selects Mistral.rs GEMV
or Candle matmul. One JSON Lines record represents one complete model run. The
record includes source, model, hardware, workload, and capture provenance.

Validate a record:

```bash
python3 tools/gemm/shape_census.py validate run.jsonl
```

Create a deterministic summary:

```bash
python3 tools/gemm/shape_census.py summarize run.jsonl > summary.json
```

## Output

The summary reports:

- call and FLOP shares
- path and site breakdowns
- logical Oxide Infer BF16 compatibility
- each input record SHA-256

Runtime pointer alignment and buffer capacity remain binding-time checks. The
summary makes no provider performance claim.

Raw model-runner records stay in the engine repository. A later Oxide selection
record can reference a validated raw blob by SHA-256.

## Matched M=1 benchmark

The H20 benchmark runner reuses the five validated Qwen2.5-1.5B census shapes
and one bit-exact dyadic fixture. Run the native provider and cuBLASLt in
separate processes so their order can be reversed:

```bash
OXIDE_BENCH_RUN_LABEL=oxide_first make bench-sm90-gemv-oxide
OXIDE_BENCH_RUN_LABEL=cublaslt_second make bench-sm90-gemv-cublaslt

OXIDE_BENCH_RUN_LABEL=cublaslt_first make bench-sm90-gemv-cublaslt
OXIDE_BENCH_RUN_LABEL=oxide_second make bench-sm90-gemv-oxide
```

Each JSON Lines record includes the selected provider and algorithm, provider
version, workspace, fixture digests, CPU-reference correctness, and raw CUDA
event samples. These operator records do not establish an engine-performance
claim. The paired Mistral.rs baseline and the resulting no-promotion decision
are summarized in the
[reviewed stop record](../../docs/results/h20-sm90a-m1-gemv-stop-ac2bd5a-20260812.json).
