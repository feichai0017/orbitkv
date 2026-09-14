# CUDA weight loading

The executor validates checkpoint names, shapes and model semantics before
building its graph. OrbitKV compiler owns storage decoding and upload for the graph's
named inputs. This boundary depends on source and destination dtypes; checkpoint
names, layer counts and model families do not select an upload implementation.

## API and ownership

`CudaRuntimeImpl::load_safetensors(&graph, path)` returns
`anyhow::Result<WeightLoadReport>`. The report contains bound tensor count,
converted tensor count, checkpoint bytes consumed and device bytes written.
The executor propagates failures as `DecoderError::WeightLoading`; malformed
files and unsupported encodings no longer panic inside the loader.

The private `runtime/weights.rs` module owns mapping, validation, upload and
binding. Its `weights/convert.rs` child owns the explicit encoding table and
pure float conversion. Runtime input ownership and execution stay in
`runtime.rs`. Private tests mirror these modules under `tests/unit/runtime/weights/`.

Each immutable shard is mapped once. All matching named inputs are checked
for supported source/destination encodings before any binding changes. Inputs
absent from a shard and extra checkpoint tensors are ignored, allowing multiple
shards and partial model graphs. The executor's earlier checkpoint contract
checks that the complete model has every required weight.

Storage-compatible inputs upload a borrowed view of the mapping directly into
their owned device allocation. Conversions allocate one `Vec<f32>`, `Vec<f16>`
or `Vec<bf16>` and expose a borrowed byte view; the allocation retains its
original type and deallocation alignment. Source decoding accepts unaligned
little-endian bytes. Equal byte widths alone do not authorize reinterpretation:
FP8 formats remain distinct, with an explicit U8 encoding for E8M0 scales.

There is no intermediate `Vec<u8>` copy and no persistent host mirror for
weights. CUDA's driver can still stage pageable memory internally. The loader
drains the upload stream once per shard before releasing its mapping, including
the device-error path. This is initialization work. A device failure may leave
earlier tensors bound; callers must abandon a failed initialization. The API
does not promise transactional live weight replacement.

## Measurement contract

Stage tracing separates `cuda.weights.map`, `metadata`, `validate`, `convert`,
`allocate`, `upload`, and `complete`. A `tensor` span retains the label, source
and target dtype, source byte count, and whether conversion occurred. Each
`loaded` record reports actual counts and bytes for its shard.

`map` measures opening and creating the mapping. Page faults and disk reads may
occur later when bytes are first touched, inside conversion or upload. `upload`
is the CPU wall time of the CUDA copy API, not an isolated DMA or PCIe benchmark.
`complete` records the loader's own completion wait; enabling tracing introduces
no additional synchronization. Nested inclusive durations must not be summed
as independent process costs.

The [fixed-artifact H20 result](validation/weight-loading-20260913/README.md)
compares frozen old and new loaders against the same selected programs,
checkpoint and eight-step oracle. It qualifies weight loading and replay
startup; warm execution uses the same selected kernels.
