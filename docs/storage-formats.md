# KV precision and SSD compression

Choose KV precision in the inference engine and SSD compression in the Cache
Manager. They solve different problems and can be used together. OrbitKV copies
the engine's registered bytes without requantizing them; optional SSD LZ4 is
lossless. Neither changes the model's weights.

## Configure the Manager

```bash
orbitkv-cache-manager --pool-size 8gb \
  --ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 100gb \
  --ssd-compression lz4 --ssd-codec-budget 64mb
```

Compression defaults to `none`. With `lz4`, Rust encodes sealed host state on
bounded blocking workers, outside the directory lock and I/O executor. It keeps
the compressed form only when the aligned SSD extent saves at least **12.5%**.
Incompressible objects, exhausted scratch budgets and oversized objects retain
their raw form. Encoding does not delay demand waiting for scratch memory.

`--ssd-codec-budget` limits temporary host buffers across encoding, submitted I/O
and decoding. Its default is 64 MB; valid values are 4 KiB through 4 GiB minus
one. This memory is separate from `--pool-size`. Restored pages still consume
their full logical pinned-memory and query budgets. This version compresses
SSD capacity and I/O, **not DRAM or HBM capacity**.

## Configure engine KV precision

Add `--kv-cache-dtype fp8_e4m3` to either engine's existing
[OrbitKV launch command](single-node.md). Keep the engine's required attention
backend and calibrated scales with the deployment. Engines perform quantization
and dequantization during inference; the Manager stores and restores those
bytes. FP8 model weights alone do not select FP8 KV.

Cache identity includes engine release, model artifacts, KV dtype, layout and
execution geometry. Static scales loaded from model weights belong to the model
identity. SGLang's external `--quantization-param-path` file is additionally
fingerprinted by content, including when an operator provides
`ORBITKV_MODEL_FINGERPRINT`. Change the deployment fingerprint after changing
any covered model artifact. Runtime weight/scale mutation requires a new cache
identity; it is not a supported way to update a live registration.

Qualification is per engine, model and format. FP8 recovery does not establish
model-quality parity with BF16. Per-token scales, NVFP4/MXFP4, packed MLA variants
and recurrent-state quantization need their own registered-state and serving
gates; do not infer coverage from an engine accepting a dtype flag.

## How compression and GDS interact

| Stored representation | Write | Demand restore |
| --- | --- | --- |
| Compression disabled, complete GPU state group | cuFile when available; also retain hot DRAM | cuFile through registered GPU staging |
| LZ4 enabled, compressible object | D2H → Rust LZ4 → io_uring | io_uring → bounded Rust decode → pinned DRAM → H2D |
| LZ4 enabled, raw fallback | D2H → io_uring | cuFile when the selected SSD prefix is entirely raw; otherwise host read |

Enabling CPU compression routes publications through the host writer. It does
not silently run a GPU codec or retain GPU writeback for compressible pages.
The Manager keeps automatic cuFile capability selection for raw reads. A mixed
raw/compressed prefix uses host reads so it does not lose a recoverable suffix
at the representation boundary. [GDS qualification](gds.md) remains independent
of successful compression or CUDA IPC.

Each compressed SSD state object records a versioned encoding, original segment sizes,
compressed segment lengths and CRC32 checksums in the in-memory index. Each
segment decodes directly into its bounded destination. Exact decoded lengths
and checksums must pass before cache admission; a damaged decode becomes a miss.
The failed index generation is invalidated so recomputation can repair it; a
late failed reader cannot invalidate a newer replacement.
CRC32 detects accidental corruption, not malicious tampering. Alignment padding
is initialized before encoding. SSD files and their indexes are ephemeral and
recreated on Manager restart.

Canceled queries cannot free submitted I/O buffers. The worker completes the
read and validation before releasing scratch; ordinary query generation and
lease checks still decide whether a result can be adopted. Encoding decisions
do not change semantic state keys or broaden `required_ranges`. A selected
state object remains the physical decode granularity.

## Measure before enabling by default

- `orbitkv_ssd_codec_bytes_total{representation="logical"|"stored"}` measures
  successful compressed writes before and after alignment. It excludes raw
  fallbacks; use total SSD write bytes to assess the whole workload.
- `orbitkv_ssd_codec_skips_total{reason=...}` identifies ratio, budget, oversized,
  allocation and encoding fallbacks.
- `orbitkv_ssd_codec_scratch_bytes` tracks live scratch through I/O completion.
- `orbitkv_ssd_codec_duration_seconds` records encode/decode time;
  `orbitkv_ssd_codec_decode_failures_total` counts rejected decodes.

Compare `none` and `lz4` with identical engine KV dtype, capacities, request
sequence and concurrency. Measure TTFT, throughput, CPU time, physical SSD
bytes and scratch/pinned-memory peaks. FP8 data may compress differently from
BF16. The 12.5% size threshold does not prove a latency benefit.

Correctness gates accept both format options, for example from `python/`:

```bash
../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_vllm_e2e_correctness.py --model /path/to/Qwen3-8B \
  --max-model-len 4096 --orbitkv-pool-size 1gb --vllm-cache-tier ssd \
  --kv-cache-dtype fp8_e4m3 --ssd-compression lz4

../.venv/sglang-release/bin/python -m pytest -m e2e \
  tests/e2e/test_sglang_direct_e2e.py -k ssd --model /path/to/Qwen3-8B \
  --kv-cache-dtype fp8_e4m3 --ssd-compression lz4
```

Use a prebuilt Manager through `ORBITKV_CACHE_MANAGER_BINARY`; run Cargo builds
and GPU gates sequentially. Both gates force SSD recovery after DRAM eviction
and engine restart, and use the same engine KV dtype for their output controls.

## Recorded qualification

On 2026-09-24, Qwen3-8B ran on one H20 with BF16 weights and explicit
`fp8_e4m3` KV, using the engines' default static scales. The output controls used
the same precision. These are recovery checks, not BF16-versus-FP8 quality tests
or matched performance benchmarks.

| Configuration | vLLM 0.29.0 | SGLang 0.5.20 |
| --- | --- | --- |
| Raw SSD, cuFile CPU compatibility | 6 passed; 1 recurrent-only check skipped | Restart, concurrent restore and output control passed |
| `lz4`, io_uring | 6 passed; 1 recurrent-only check skipped | Restart, concurrent restore and output control passed |

The LZ4 option did not establish a space benefit for these FP8 pages: live
samples recorded 48 vLLM and 8 SGLang ratio skips, with no encoded writes at
those samples. They verify safe raw fallback. Actual encoding/decoding is
separately proven by padded split/page-first GPU fixtures whose logical objects
cannot fit in the configured SSD capacity without compression, and by mixed
raw/encoded process fault tests. Keep compression optional; evaluate an entropy
codec suited to quantized tensors before claiming additional FP8 savings.

The GPU-storage checks forced cuFile CPU compatibility on an overlay mount.
Native GDS bandwidth and latency still require the bare-metal gate in
[GPU storage](gds.md#bare-metal-acceptance-script).

## Upstream references and next steps

LMCache's [MP serialization](https://docs.lmcache.ai/mp/serde.html) separates
L2 representation from the engine cache and includes FP8/TurboQuant options.
[FlexKV](https://github.com/taco-project/FlexKV/tree/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/flexkv/transfer/compression)
uses an optional nvCOMP/ANS path with explicit build and layout gates. These
are references for format boundaries and capability checks, not claims that
OrbitKV implements their codecs.

The current implementation is Rust CPU LZ4. GPU lossless codecs, compressed
remote transfers and optional lossy storage quantization are subsequent work.
Lossy formats need scale/layout/version metadata, error and quality evaluations,
and end-to-end transfer savings after decode costs. Recurrent checkpoints must
remain exact until separately qualified. Mooncake TE transports bytes and does
not provide implicit cache quantization.
