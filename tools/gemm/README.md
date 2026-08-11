# GEMM shape census tools

These tools validate and summarize dense-linear call counts from a pinned model
run. They do not measure latency and do not select a GEMM provider.

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
