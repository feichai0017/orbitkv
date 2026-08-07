# FlashInfer matched baseline

This directory contains an external Python baseline for comparing Loom Infer
with the latest pinned stable FlashInfer release. It is measurement tooling,
not Loom product code or a supported Python API.

The current baseline is:

```text
FlashInfer: v0.6.16.post1
commit:     5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57
```

The harness measures the admitted BF16 contracts with preallocated tensors and
CUDA events. Single decode calls the precompiled low-level module directly so
temporary and output allocation stay outside the timed region. Paged decode
and ragged/paged prefill plan their wrappers before measurement, fix the FA2
CUDA-core backend, and pass caller-owned output and LSE tensors on every run.
GEMM fixes the cuBLASLt backend and tactic zero after preparation.

## Environment

Use an external virtual environment. Do not install these packages as product
dependencies:

```bash
python -m venv --system-site-packages <venv>
<venv>/bin/python -m pip install flashinfer-python==0.6.16.post1
<venv>/bin/python -m pip install \
  apache-tvm-ffi==0.1.13.post0 \
  cuda-python==13.1.1 click einops packaging requests tabulate tqdm ninja \
  nvidia-ml-py
```

Install any additional release dependencies required by the host package index.
The benchmark records the FlashInfer package version and source commit.

## Run

Use the same parameters for both providers:

```bash
export LOOM_BENCH_WARMUP=200
export LOOM_BENCH_LAUNCHES=100
export LOOM_BENCH_SAMPLES=50
export LOOM_BENCH_OPERATORS=ragged_prefill
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

LOOM_BENCH_RUN_LABEL=loom_first \
taskset -c <gpu-local-physical-cpu> \
make bench-ragged-loom > /tmp/loom-first.jsonl

FLASHINFER_WORKSPACE_BASE=/path/to/jit-cache \
LOOM_BENCH_RUN_LABEL=flashinfer_second \
taskset -c <gpu-local-physical-cpu> \
<venv>/bin/python tools/flashinfer/matched_bench.py \
  > /tmp/flashinfer-second.jsonl
```

`make bench-loom` records the current full Git commit automatically. Set
`LOOM_SOURCE_COMMIT` only when deliberately identifying an equivalent external
source projection, and record that mapping separately.

Use `nvidia-smi topo -m` to choose one otherwise idle physical CPU on the
GPU-local NUMA node, and pin both providers to that same CPU. Short eager paths
include host submission gaps inside their CUDA-event interval; unrestricted
CPU migration can therefore invalidate provider-order stability even when GPU
clocks are fixed. Record the chosen CPU, NUMA node, and thread limits in the
evidence. Do not mix pinned and unpinned samples.

On hosts where PyTorch links NVSHMEM outside the default loader path, add its
installed library directory to `LD_LIBRARY_PATH` for the external Python
baseline. This does not affect Loom product dependencies.

Repeat in the reverse provider order. The measurement
`eager_stream_batch_cuda_event` records one CUDA-event interval around several
sequential calls and divides by the call count. Planning, JIT, tensor
allocation, copies, and final host synchronization are outside that interval.
If a provider cannot submit work quickly enough to keep the stream occupied,
the resulting GPU idle gaps remain inside the event interval. This is an eager
provider-path metric, not an isolated kernel-duration claim. A CUDA Graph
comparison is a separate gate.

Use `LOOM_BENCH_OPERATORS=paged_prefill` and
`make bench-paged-prefill-loom` for the ragged-query, page-size-16 NHD
paged-prefill surface. The external baseline uses
`BatchPrefillWithPagedKVCacheWrapper` with the same Q/K/V bits, query
`indptr`, page table, output, LSE, and FA2 backend. The harness includes short
direct cases plus long MQA `[128,256,512]` and GQA4 `[256,1024]` cases. Retain
both provider orders and report the long-context results separately from the
short eager and fixed-address Graph records.

The ragged Graph gate uses one replay per CUDA-event sample:

```bash
export LOOM_BENCH_WARMUP=200
export LOOM_BENCH_LAUNCHES=1
export LOOM_BENCH_SAMPLES=100

LOOM_BENCH_RUN_LABEL=loom_graph_first \
make bench-ragged-graph-loom > /tmp/loom-graph-first.jsonl

FLASHINFER_WORKSPACE_BASE=/path/to/jit-cache \
LOOM_BENCH_RUN_LABEL=flashinfer_graph_second \
<venv>/bin/python tools/flashinfer/ragged_graph_bench.py \
  > /tmp/flashinfer-graph-second.jsonl
```

Repeat in reverse provider order. Both paths record a start event, replay one
fixed-address graph, record one completion event, and record an end event.
Capture, instantiation, planning, JIT, allocation, fixture copies, correctness
reads, and final synchronization are outside the timed interval. Do not combine
these samples with `eager_stream_batch_cuda_event` records.

The paged-prefill Graph gate uses the same protocol and the admitted GQA4
page-reorder/reuse fixture:

```bash
LOOM_BENCH_RUN_LABEL=loom_graph_first \
make bench-paged-prefill-graph-loom > /tmp/loom-paged-graph-first.jsonl

FLASHINFER_WORKSPACE_BASE=/path/to/jit-cache \
LOOM_BENCH_RUN_LABEL=flashinfer_graph_second \
<venv>/bin/python tools/flashinfer/paged_prefill_graph_bench.py \
  > /tmp/flashinfer-paged-graph-second.jsonl
```

Repeat in reverse provider order. Both paths include one completion event after
one fixed-address replay; planning, capture, instantiation, allocation, and
correctness reads remain outside the interval.

RMSNorm is omitted when the unmodified FlashInfer release cannot compile or
load its provider on the declared host. Do not patch baseline source and report
the result as an official release comparison.
