# M1 Qwen3.8 bring-up status

Status: in progress. This document records what the Rust compiler currently
derives and what still comes from the imported handwritten artifact.

## Evidence boundaries

Two independent oracles are required.

The imported `examples/qwen3.8-27b.json` is a **BF16 executable oracle**. It is
verified by local `kern-manifest` and is authoritative for:

- the 64-layer 3-GDN/1-attention topology;
- physical KV and GDN state packing;
- prefill, single-sequence decode, and batched-decode call order;
- kernel ABI, buffer wiring, manifest protocol, and serving behavior.

It is not evidence for official FP8 weight execution: all 659 of its bound
weight buffers are BF16.

The local official `Qwen3.8-27B-FP8` checkpoint is the **FP8 weight oracle**.
Its config, index, and safetensors headers establish:

- dynamic E4M3 quantization with 128 by 128 inverse-scale blocks;
- 1,606 indexed tensors in 66 shards;
- 1,251 target text-tower tensors;
- 400 FP8 matrices paired with 400 BF16 `weight_scale_inv` tensors;
- 451 other BF16 text-tower tensors;
- 355 vision/MTP tensors outside the target-only M1 gate.

The local snapshot does not carry trustworthy revision metadata. The project
records the expected revision separately; file-content qualification is based on
the config, index, and all referenced safetensors headers.

## Rust-derived surface

The compiler now derives the following without per-layer JSON templates:

| Surface | Derived result | Gate |
| --- | --- | --- |
| Model topology | 48 GDN and 16 Full Attention layers | Exact contract test |
| Persistent state | KV plus logical recurrent/conv state, packed into the oracle GDN slot | Typed state and packing checks |
| Variables/states | `tokens`, `seqs`, `kv`, `gdn` | Typed equality with BF16 oracle |
| Prefill skeleton | 1,127 label/op calls | Exact per-call equality |
| Decode skeleton | 646 label/op calls | Exact per-call equality |
| Batched decode skeleton | 534 label/op calls | Exact per-call equality |
| Raw FP8 bindings | 915 physical buffers consuming all 1,251 text tensors once | Generated contract validation |
| Fused FP8 matrices | 256 execution matrices | Shape and scale-pair validation |
| Load transforms | 465 derived F32 buffers | 256 scale casts, 48 A-log casts, 161 unit-offset norms |

The physical FP8 weight plan keeps checkpoint BF16 inverse scales as raw bound
buffers and emits separate F32 execution-scale buffers. This preserves the
checkpoint contract while matching the expected low-precision provider ABI.

## H20 FP8 projection contract

The compiler now derives provider-neutral projection families from the weight
plan. The 256 execution matrices collapse to five `[N,K]` families: 64 uses of
`[34816,5120]`, 64 of `[5120,17408]`, 48 of `[16384,5120]`, 16 of
`[14336,5120]`, and 64 of `[5120,6144]`. Provider selection receives these
shapes and row limits, never a model name.

The first capability is pinned to H20/SM90a, 78 SMs, DeepGEMM revision
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`, and numerical ABI
`fp8-e4m3-row128-f32-kmajor-align4-rne-reciprocal-v2`. It validates the exact
BF16 input, block-scaled E4M3 weight, F32 weight scale, packed activation scale,
BF16 output, scratch, tile, cluster, thread, and shared-memory layouts for low
latency row buckets 1, 2, 4, and 8. Legal candidates are not executable claims:
only a content-addressed cubin with numerical evidence may become qualified.

The initial GPU qualification slice covers the fused attention QKV projection
at `M=8, N=14336, K=5120`. On the local H20 with CUDA 13.1 it passed patterned
activation quantization and analytical block-scale output checks exactly, then
measured about 0.050 ms median across 100 warm samples for combined BF16-to-FP8
activation quantization plus GEMM. This is a kernel-level gate, not a
decode-token or serving-performance claim.

## Commands

Inspect or validate the BF16 executable oracle:

```sh
cargo run --locked --bin orbitkv -- oracle validate \
  examples/qwen3.8-27b.json
```

Inspect the official FP8 checkpoint headers:

```sh
cargo run --locked --bin orbitkv -- oracle checkpoint \
  /workspace/models/qwen3.8-27b-fp8
```

Validate both the checkpoint and generated physical weight plan:

```sh
cargo run --locked --bin orbitkv -- oracle weights \
  /workspace/models/qwen3.8-27b-fp8
```

Inspect a compiler-selected H20 provider contract or render its AOT source:

```sh
cargo run --locked --bin orbitkv -- \
  provider deepgemm-h20 plan 8 14336 5120
cargo run --locked --bin orbitkv -- \
  provider deepgemm-h20 source 8 14336 5120
```

Rebuild and qualify that exact contract against a pinned DeepGEMM checkout:

```sh
python tools/qualify_deepgemm_h20.py \
  --provider-dir /path/to/deepgemm-559d79f \
  --output-dir /tmp/orbitkv-deepgemm-qualification \
  --nvcc /usr/local/cuda-13.1/bin/nvcc
```

The qualification directory is disposable output. Its JSON records source,
host-library, and deployable cubin SHA-256 digests; generated binaries are not
checked into the repository.

The qualified contract now lowers to a schema-v6 two-launch op. Launch one
quantizes BF16 activations into private FP8 and F32-scale scratch. Launch two
passes four 128-byte TMA descriptors directly to the qualified DeepGEMM entry.
Schema v6 adds TMA-over-private-scratch while retaining v5 compatibility, so
these temporary planes do not leak into the model graph. The resulting probe
has been loaded and executed by `kern-runtime` on the H20, with its output
checked against an independent closed-form block-scale reference. The probe is
also captured and replayed as one CUDA Graph, matching the intended
`decode_batch` execution mode.

Re-admit a saved qualification record through the compiler contract validator.
This also re-hashes the source and cubin named by the record:

```sh
cargo run --locked --bin orbitkv -- oracle provider-contract \
  /tmp/orbitkv-deepgemm-qualification/qualification.json
```

Compare any candidate manifest structurally with the BF16 executable oracle:

```sh
cargo run --locked --bin orbitkv -- oracle diff candidate.json \
  examples/qwen3.8-27b.json
```

## Remaining M1 lowering

The next closed slice is the decode-batch program, not the complete manifest at
once:

1. derive activation, carry, input, output, and workspace buffers for
   `decode_batch`;
2. reuse the now-verified FP8 projection manifest op across the five shape
   families and four low-latency row buckets;
3. lower buffer/state/var arguments for every skeleton call;
4. emit the load program for FP8 scale casts, A-log casts, norm transforms,
   RoPE, and fixed tables;
5. construct a complete verified target-only manifest;
6. compare it against the BF16 topology oracle and the FP8 checkpoint oracle;
7. execute the qualified cubin through local `kern-runtime`, then expose it
   through `kern-serve`.

Only after this provider-composed FP8 decoder is correct do GDN fusion and
persistent-island optimization begin.
