# Decode CUDA Graph correctness diagnostic

Status: real-device, released-checkpoint correctness diagnostic. This record
does not qualify latency, throughput, memory, capacity, or production behavior.

The generic Luminal regression captured an already-prepared elementwise graph,
updated its stable input allocation, replayed the graph, and observed the
updated output. The test exposed and fixed two capture-boundary defects before
passing: runtime-owned inputs were consumed after warmup, and the direct-launch
path performed first-use CUDA work inside capture. The final implementation
retains inputs during external execution and primes direct launches before
beginning stream capture.

The model-level test used one released dense decoder checkpoint. It performed
one real two-bucket search, prefill, decode warmup plus outer graph capture, one
replay with diagnostic logits, one token-only replay, and four successful
OrbitKV publications. Device greedy token IDs matched host argmax where logits
were read. The selected plan used materialized K/V updates with a graph-visible
copy back to the same stable arena.

Observed wall times were approximately 220.6 seconds for one-time search and
compile, 13 ms for prefill, 68 ms for decode warmup plus graph construction,
7 ms for replay with full-logit readback, and 5 ms for token-only replay. These
are single-run diagnostic timings with different readback work and are not a
matched benchmark.

A second run added a 20-iteration alternating comparison on the exact same
prepared decode step. Both paths included the same stable-input uploads and one
token-ID readback. Median eager bucket dispatch was 4437.9 us and median
flattened outer-graph replay was 5544.1 us, a replay/eager ratio of 1.249. This
is a failed performance result: the first outer-graph implementation was 24.9%
slower than Luminal's already-materialized inner graph path. It establishes no
speedup and motivates composing the selected inner graphs as child nodes rather
than flattening them into raw launches. The second one-time search took 260.1
seconds and is excluded from the comparison.

The capture signature fixes query-token count, batch size, context-page count,
`query_indptr`, and `page_indptr`. Tokens, positions, write slots, physical page
indices, and last-page lengths may change within that signature. A signature
mismatch fails closed and requires recapture.

## Environment

- accelerator: NVIDIA H20
- compute capability: 9.0
- reported memory: 97871 MiB
- driver: 535.161.08
- checkpoint: `/workspace/models/qwen2.5-0.5b-instruct`
- dtype: BF16
- page tokens: 16
- physical pages: 64
- decode bucket: `s=1`
- prefill bucket: `s=2..8`, representative `s=4`
- maximum batch: 1
- maximum context pages: 64
- search candidates per bucket: 2
- search seed: 7
- parent test source: `c0643ce`
- Luminal fork: `281d1062ff482e998580d5313b6436f88797c123`

## Reproduction

```bash
env \
  PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  CARGO_HOME=/root/.cargo \
  RUSTUP_HOME=/root/.rustup \
  HOME=/tmp/orbitkv-device-cache \
  CUDA_CACHE_PATH=/tmp/orbitkv-device-cache/.nv/ComputeCache \
  LUMINAL_FLASHINFER_DIR=/tmp/orbitkv-device-cache/.cache/luminal/flashinfer/flashinfer-src/f1e6fdcb8f65 \
  FLASHINFER_CUDA_ARCH=sm_90 \
  SEARCH_LOG=0 \
  ORBITKV_MODEL_DIR=/workspace/models/qwen2.5-0.5b-instruct \
  ORBITKV_SEARCH_GRAPHS=2 \
  cargo test --locked -p orbitkv-executor --features cuda \
    --test model_execution \
    released_decoder_reuses_one_compiled_runtime_and_kv_arena \
    -- --ignored --nocapture
```

The generic Luminal regression command was:

```bash
env \
  PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  CARGO_HOME=/root/.cargo \
  RUSTUP_HOME=/root/.rustup \
  HOME=/tmp/orbitkv-device-cache \
  CUDA_CACHE_PATH=/tmp/orbitkv-device-cache/.nv/ComputeCache \
  cargo test -p luminal_cuda_lite \
    captured_execution_reads_updated_stable_inputs \
    -- --ignored --nocapture
```
