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
2. define typed provider/kernel capabilities and exact op interfaces;
3. lower buffer/state/var arguments for every skeleton call;
4. emit the load program for FP8 scale casts, A-log casts, norm transforms,
   RoPE, and fixed tables;
5. construct a complete verified target-only manifest;
6. compare it against the BF16 topology oracle and the FP8 checkpoint oracle;
7. execute it through local `kern-runtime`, then expose it through
   `kern-serve`.

Only after this provider-composed FP8 decoder is correct do GDN fusion and
persistent-island optimization begin.
