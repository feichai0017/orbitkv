# KV precision and storage quantization

OrbitKV separates engine KV precision from the representation stored on SSD.
`--ssd-codec fp8` opts into **experimental lossy FP8 E4M3FN storage** for registered BF16/FP16
attention regions. Rust reconstructs the engine's original dtype on restore.
Checkpoints, opaque layouts and engine-native FP8 remain exact.

## Configure the Manager

```bash
orbitkv-cache-manager --pool-size 8gb \
  --ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 100gb \
  --ssd-codec fp8 --ssd-codec-budget 64mb
```

The default is `--ssd-codec none`: preserve every byte. `fp8` follows the
storage-transform boundary of [LMCache's FP8 serde](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/serde/fp8.py).
It stores one E4M3FN byte per eligible 16-bit element and casts back on load.
This is separate from engine-native `--kv-cache-dtype fp8_e4m3`, which changes
HBM representation and attention execution. FP8 model weights do not imply FP8 KV.

Encoding/decoding runs in Rust on bounded blocking workers. Precomputed scalar
conversion tables avoid Python, Torch operations and per-element allocation in
the hot path. Rounding is nearest, ties to even; signed zero and subnormal rules
match E4M3FN. No additional scaling or clipping is applied. Nonfinite inputs or
values outside [-448, 448] retain the entire original object, including NaN
payloads. This fallback avoids introducing saturation or nonfinite values.

Physical extents use the actual encoded size, rounded to the backend alignment.
An encoded object is kept only when it saves at least 12.5% after alignment.
The target payload reduction for eligible BF16/FP16 regions is 2:1; raw regions,
alignment, fallback and indexing reduce the total saving. DRAM and HBM still
hold the original-width representation.

`--ssd-codec-budget` limits temporary encoded host buffers across encoding,
submitted I/O and decoding. The default is 64 MB, with a valid range of 4 KiB
through 4 GiB minus one, separate from `--pool-size`. Encoding never waits for
scratch held by demand reads. Fixed conversion tables use approximately
257 KiB when both source dtypes have been used. Restored pages continue to
consume full logical pinned-memory and query budgets.

## Registration and identity

The adapters declare attention regions as `bf16` or `fp16`; recurrent/convolution
state, packed or unknown dtypes are `exact`. The Manager checks the declared
scalar representation against the imported CUDA tensor. It does not infer dtype
or quantization eligibility from page size, group number or a layer's name.

The selected per-region storage policy is part of the sealed storage namespace,
so exact and quantization-enabled registrations cannot reuse one another's
lossy results. TP replicas must agree on that policy. A page-first object is
quantized only when every constituent layer has the same eligible scalar type;
a mixed page retains its original representation.

Ordinary identity still includes engine release, model artifacts, KV dtype,
layout and execution geometry. SGLang's external `--quantization-param-path`
file is fingerprinted by content, even with `ORBITKV_MODEL_FINGERPRINT` set.
Changing weights or scales requires a new identity; live mutation is unsupported.

## Transfer and integrity

| Representation | Write | Restore |
| --- | --- | --- |
| Codec disabled | cuFile when eligible; retain hot DRAM | cuFile or io_uring |
| FP8 storage enabled, eligible state | D2H → Rust quantize → io_uring | io_uring → Rust reconstruct → pinned DRAM → H2D |
| Raw fallback under FP8 policy | D2H → io_uring | cuFile for an entirely raw selected prefix; otherwise host read |

The current implementation uses CPU conversion after D2H. It reduces SSD bytes,
not GPU↔CPU transfer volume, and is not a GPU compressor or a direct compressed
GDS path. A mixed raw/encoded prefix uses host reads without truncating the
recoverable suffix. [Native GDS qualification](gds.md) remains a separate gate.

The in-memory SSD index records a versioned format, original segment geometry,
per-segment restoration type and CRC32 over stored bytes. Checksums and exact
lengths are checked before admission. Corruption becomes a miss and invalidates
only the failed generation, allowing recomputation to repair the same key.
CRC checks detect storage corruption; they do not assess quantization error.

Canceled queries retain submitted I/O buffers until completion. Normal query
revisions and leases decide whether completed results may be adopted. Encoding
does not broaden `required_ranges`. SSD files and indexes remain ephemeral and
are recreated on Manager restart.

## Qualification and measurement

Compare raw and FP8 storage while keeping the **engine** KV dtype, model,
capacities, requests and concurrency fixed. A useful BF16 qualification command
from `python/` is:

```bash
../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_vllm_e2e_correctness.py --model /path/to/Qwen3-8B \
  --max-model-len 4096 --orbitkv-pool-size 1gb --vllm-cache-tier ssd \
  --kv-cache-dtype auto --ssd-codec fp8 --ssd-backend uring

../.venv/sglang-release/bin/python -m pytest -m e2e \
  tests/e2e/test_sglang_direct_e2e.py -k ssd --model /path/to/Qwen3-8B \
  --kv-cache-dtype auto --ssd-codec fp8 --ssd-backend uring
```

Use a prebuilt Manager through `ORBITKV_CACHE_MANAGER_BINARY`; run Cargo builds
and GPU gates sequentially. GPU layout gates compare reconstruction against
Torch's E4M3FN cast, including page-first layouts. Process faults cover raw/FP8
mixed prefixes, cancellation, corrupted data and repair. Serving gates force
SSD recovery after eviction/restart and retain unquantized output controls.
The output-equality assertions remain strict: a failed comparison is a measured
precision change, not a passing exact-recovery gate. Run with `--ssd-codec none`
for exact regression qualification.
Passing these prompts does not establish general model-quality equivalence.

Measure physical SSD bytes, encode/decode time, TTFT, throughput and model
quality. Inspect `orbitkv_ssd_codec_bytes_total{representation="logical"|"stored"}`
for successful encoded objects; use total SSD write bytes to include fallbacks.
`orbitkv_ssd_codec_skips_total{reason=...}` distinguishes format, alignment ratio,
range/nonfinite and resource limits. Scratch, duration and decode-failure
metrics are documented in [metrics](metrics.md).

On 2026-09-24, the H20/Qwen3-8B vLLM 0.29.0 run with BF16 engine KV wrote
168.75 MiB of eligible logical state as 84.375 MiB of encoded payloads. Two of
the 12 execution-plan responses (`long_warm`, `rollback_short`) changed against
the unquantized control; the strict output test failed while the five cache
behavior checks passed and one recurrent-only check was skipped. This is a
capacity result with observed output drift, not a model-quality or latency win.
The SGLang 0.5.20 run wrote 72 MiB as 36 MiB; its eight-token recovery probe
passed output equality and the existing log-probability tolerance after restart
and concurrent prefix recovery. This single prompt does not cancel the vLLM
precision finding or establish quality on other models.
The 27 Manager fault checks and two SGLang GPU layout checks passed, including
BF16/FP16 reconstruction against Torch for every finite source bit pattern
within the FP8 range and isolation from exact registrations.

## Next formats

The generic LZ4 experiment has been removed. Its sampled Qwen3-8B FP8 pages did
not meet the storage-saving threshold, so it did not justify a separate public
codec. Engine-native FP8 recovery was qualified in both engines independently
of extra storage compression.

[FlexKV's nvCOMP ANS path](https://github.com/taco-project/FlexKV/tree/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/flexkv/transfer/compression)
is the GPU lossless reference: evaluate it on real tensors, including native FP8,
and account for GPU workspace, contention and the complete transfer path.
[LMCache's TurboQuant serde](https://docs.lmcache.ai/mp/serde.html) is the low-bit
reference. It needs explicit head, K/V and layer geometry plus quality gates;
those properties must not be guessed from opaque bytes. Neither ANS nor
TurboQuant is implemented in OrbitKV yet. Recurrent checkpoints remain exact.
