# TileLang paged-prefill spike

This directory evaluates TileLang as an experimental kernel provider for
Oxide Infer. It is benchmark tooling, not a product dependency or a supported
Python API.

The first fixed contract is the existing long paged-prefill GQA4 case:

```text
BF16, batch 2, query lengths [32, 64], KV lengths [256, 1024]
query heads 16, KV heads 4, head dimension 128, page size 16
NHD pages, reordered page table, bottom-right causal mask
```

The TileLang kernel consumes the page table directly. Compilation, tuning,
planning, allocation, page materialization, and correctness reads are outside
the timed region. TileLang and FlashInfer run in the same process on the same
tensors and CUDA stream with caller-owned outputs.

## Environment

Use an external virtual environment with a CUDA-enabled PyTorch installation.
Do not install these packages as Oxide Infer product dependencies:

```bash
python -m venv --system-site-packages <venv>
<venv>/bin/python -m pip install -r tools/tilelang/requirements.txt
```

The source reference used to inspect TileLang is tag `v0.1.13`, commit
`8001cc4ccf6149382d2019654a19f59c1d4d0482`. This reference does not prove the
provenance of a release wheel; the result record therefore distinguishes the
installed version from the inspected source reference.

## Run

```bash
<venv>/bin/python tools/tilelang/paged_prefill_spike.py \
  --provider-order tilelang-first \
  --output /tmp/tilelang-paged-prefill-first.json

<venv>/bin/python tools/tilelang/paged_prefill_spike.py \
  --provider-order flashinfer-first \
  --output /tmp/tilelang-paged-prefill-second.json
```

The default formal protocol uses 100 warmups, 100 launches per CUDA-event
sample, 50 samples per provider block, and complementary balanced six-block
provider schedules. The runner fails closed if output or LSE exceeds its
declared error limit and records failed tuning candidates rather than silently
dropping them.

## Current admission result

The 2026-08-14 fixed-shape result admitted TileLang only as an experimental
optional provider. Its pooled median was 31.83 microseconds, compared with
47.01 microseconds for the current Oxide path and 25.90 microseconds for
FlashInfer. This is a 32.3% latency reduction from Oxide, but a 22.9% overhead
over FlashInfer. See the source-bound record in `docs/results` for raw-record
hashes, correctness limits, provider-order results, and claim boundaries.

This single operator shape is an admission spike. It does not establish model
throughput, serving latency, CUDA Graph behavior, broad-shape coverage, or
production readiness.
