# H20 Real-Checkpoint SGLang Validation — 2026-08-17

## Scope

This record compares unmodified Stock SGLang with the OrbitKV owning adapter on
the public `openai/gpt-oss-20b` checkpoint. Unlike the earlier system fixture,
these runs use:

```text
load_format = auto
3 indexed MXFP4 safetensors shards
13,761,316,904 checkpoint bytes
14.986 GiB measured GPU weight residency
```

The benchmark verifies every shard referenced by
`model.safetensors.index.json` before running. Stock mode removes
`SGLANG_PLUGINS` and every OrbitKV environment variable. OrbitKV mode loads the
same checkpoint and kernels, then replaces only the SWA reclamation decision
with the Rust owner and its two-phase retirement certificate.

The final four-way ablation does not consume a hand-written model plan. It
invokes:

```text
HF config.json
    -> Rust compile-hf-config
    -> generated Retention IR
    -> LayoutProgram / SGLang policy
```

OrbitKV does not replace SGLang's FA3 attention kernel, Triton-kernel MoE
runner, scheduler, KV tensors, or physical paged pool in this experiment.

## Model and compiled plan

The checkpoint declares:

```text
24 layers
12 sliding-attention layers
12 full-attention layers
sliding window = 128 tokens
8 KV heads
head dimension = 64
BF16 KV
```

`examples/gpt_oss_20b_retention.json` declares the corresponding future-read
relations. The benchmark checks the checkpoint config against the compiled
layout before launching:

```text
Full layer set matches the checkpoint
SWA layer set matches the checkpoint
KV bytes/token/layer = 2 * 8 * 64 * 2 = 2,048
page size = 16
minimum SWA cells = 1 + ceil((128 - 1) / 16) = 9
```

The compiled plan fingerprint is:

```text
sha256:dfbeded59980b8784b79097e96ae51addbee8cd8ffb19977b6e5aa1df2c1c756
```

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA H20, SM90, 97,871 MiB |
| Driver | 535.161.08 |
| SGLang | `095ec6c997bfdd25d3864cb0ce77a6562a934b96` |
| SGLang package | `0.0.0.dev1+g095ec6c99` |
| PyTorch | 2.13.0 + CUDA 13.0 |
| Transformers | 5.12.1 |
| FlashInfer | 0.6.17 |
| Attention backend | FA3 |
| MoE runner | `triton_kernel` |
| Page size | 16 |
| Radix cache | disabled |
| Overlap scheduling | disabled |
| Speculative decoding | disabled |
| CUDA Graph | disabled |

The disabled features isolate the first qualified OrbitKV owning path. They are
not general SGLang requirements.

## Fixed-budget capacity

With `mem_fraction_static=0.18`, `max_running_requests=128`, and the same
reported 1.979 GiB KV budget:

| Metric | Stock SGLang | OrbitKV | Difference |
| --- | ---: | ---: | ---: |
| Full token capacity | 47,616 | 59,904 | +12,288 / +25.81% |
| SWA token capacity | 38,800 | 26,512 | -12,288 / -31.67% |
| Reported KV pool | 1.979 GiB | 1.979 GiB | equal |

The difference comes from the per-request SWA overshoot reserve. Stock uses its
default 128-token reclamation interval. OrbitKV compiles the 128-token lifetime
and selects a 32-token physical reclamation interval, reducing reserved SWA
headroom and redirecting the same byte budget to the binding Full pool.

This is not a compression result and does not drop semantically live KV.

## Balanced four-way attribution

The initial Stock/Owner experiment showed that the capacity difference improves
admission, but it did not isolate whether Stock SGLang could reproduce the same
result by manually changing its reclamation interval.

The final ablation therefore runs four modes:

```text
Stock128:
    unmodified Stock SGLang, default interval 128

Stock32:
    Stock SGLang with interval 32, no OrbitKV plugin

Policy32:
    Rust-generated HF plan lowers to interval 32
    SGLang still owns reclamation decisions

Owner32:
    same generated plan
    Rust issues and commits retirement certificates
```

Four fresh-process rounds use balanced execution order, so each mode appears
once in each process position. All modes use the same complete checkpoint,
resolved runtime, workload, and output digest.

| Contribution | Median ratio | Interpretation |
| --- | ---: | --- |
| Stock32 / Stock128 | `0.7828x` | physical-policy benefit, 21.72% lower makespan |
| Policy32 / Stock32 | `0.9969x` | automatic plan injection is noise-level |
| Owner32 / Policy32 | `1.0218x` | proof-carrying ownership costs 2.18% here |
| Owner32 / Stock128 | `0.7970x` | end-to-end OrbitKV path is 20.30% lower |

Mode median makespans:

```text
Stock128  7.574 s
Stock32   5.929 s
Policy32  5.908 s
Owner32   6.030 s
```

The strict conclusion is:

> The interval-32 capacity result is not unique to OrbitKV; a human can
> configure Stock SGLang to reproduce it. OrbitKV automates deriving the model
> state classes and physical policy from the checkpoint, then adds auditable
> ownership without changing outputs.

No speedup is attributed to Policy32 versus Stock32. The measured difference is
within process variation. Owner32's 2.18% median cost is reported rather than
hidden.

## Physical policy compiler

The next compiler stage consumes the generated lifetime plan together with an
explicit per-rank KV byte budget, attention DP size, chunk size, request
distribution, admission target, and reclamation-call budget.

For each page-aligned interval, OrbitKV reproduces SGLang's non-overlap
`SWAChunkCapPoolConfigurator` formula and emits:

```text
physical SWA slots and bytes
remaining Full token capacity
admitted requests and waves
estimated prefill/decode reclamation calls
retention amplification
rejection reasons
selected SGLang policy
runtime compatibility contract
```

The compiler selected interval 32:

| Interval | Full slots | SWA slots | Waves | Calls/request | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 61,952 | 24,464 | 1 | 5 | reject: call budget is 4 |
| 32 | 59,904 | 26,512 | 1 | 4 | selected |
| 64 | 55,808 | 30,608 | 1 | 4 | feasible |
| 128 | 47,616 | 38,800 | 2 | 4 | reject: only 7 of 8 requests admitted |

Four fresh SGLang processes were launched with intervals 16, 32, 64, and 128.
For every candidate, the predicted Full and SWA capacities exactly matched
SGLang's actual pools. A separate Owner run consumed the complete physical-plan
artifact and passed runtime checks for page size, cache type, disabled features,
and both pool capacities.

This is a constraint optimizer, not a claim that a static formula predicts
kernel time. Future profile feedback can refine reclamation CPU/GPU cost while
preserving these hard safety and capacity constraints.

## Admission workload

The pressure workload uses:

```text
8 requests
6,000 prompt tokens per request
32 decode tokens per request
48,000 total prompt tokens
same 1.979 GiB KV budget
```

The workload is intentionally just above Stock's 47,616 Full-token capacity and
below OrbitKV's 59,904 capacity. Three fresh-process pairs alternate execution
order:

| Pair | Order | Stock | OrbitKV | Makespan reduction |
| ---: | --- | ---: | ---: | ---: |
| 0 | Stock → OrbitKV | 14.136 s | 7.314 s | 48.26% |
| 1 | OrbitKV → Stock | 7.547 s | 5.932 s | 21.40% |
| 2 | Stock → OrbitKV | 7.514 s | 6.138 s | 18.30% |

This earlier three-pair Stock/Owner screen measured:

```text
Stock   7.547 s
OrbitKV 6.138 s
reduction 21.40%
```

The balanced four-way result above supersedes 21.40% as the primary attribution
result. The earlier screen is retained as an audit trail.

In the two stable Stock runs, seven requests finish around 5.56–5.57 seconds
and the eighth finishes around 7.51–7.55 seconds. OrbitKV completes all eight
requests in one cohort. Pair 0 contains additional first-run process/JIT
variation, which is why the report uses all pairs and the median rather than
the best pair.

Every run produced the same 256 output-token digest:

```text
5c9dddcf946496dd4e77d6ad5aec1026f2e947acdf5d7249a75d813ac12401dd
```

No request reported a retraction.

## Fixed-capacity overhead

To separate capacity benefit from owner control-plane cost, a second experiment
fixes Full capacity to 47,616 for both modes and uses:

```text
4 requests
4,096 prompt tokens per request
64 decode tokens per request
```

Across three alternating fresh-process pairs:

```text
median OrbitKV / Stock ratio = 0.9992x
median reported difference = -0.08%
```

At the same 47,616 Full-token capacity, Stock allocated 38,800 SWA token
slots while OrbitKV allocated 26,512. With 12 SWA layers and 2,048 bytes per
token per layer, this is exactly:

```text
12,288 * 12 * 2,048 = 301,989,888 bytes = 288 MiB
```

less reported KV allocation.

This is noise-level equivalence, not an OrbitKV speedup claim. It supports the
interpretation that the admission result comes from the larger usable Full
capacity rather than a faster attention kernel.

## Proof-carrying reclamation

A separate real-checkpoint owner trace used two requests with 512 prompt and 16
decode tokens. OrbitKV emitted two SWA retirement certificates:

```text
semantic frontier = 513
window = 128
maximum reclaimable end = 385
page-aligned retired range = [0, 384)
execution proof = non-overlap scheduler barrier, epoch 1
```

Both certificate IDs were committed only after SGLang's physical `free_swa`
group succeeded. The trace also confirms `SWAChunkCache hybrid_swa=True`.

## Reproduction

After downloading `openai/gpt-oss-20b` to a local directory:

```bash
cargo build --release --bin orbitkv
cargo build --release --manifest-path crates/orbitkv-ffi/Cargo.toml

.venv-sglang-h20/bin/python \
  integrations/sglang/bench_real_model_ab.py \
  --model /path/to/gpt-oss-20b \
  --plan examples/gpt_oss_20b_retention.json \
  --pairs 3 \
  --requests 8 \
  --max-running-requests 128 \
  --prompt-tokens 6000 \
  --decode-tokens 32 \
  --context-length 8192 \
  --mem-fraction-static 0.18 \
  --eviction-interval 32
```

The runner validates checkpoint completeness, model/plan geometry, output
digests, completion counts, and fresh-process execution order.

The balanced four-way attribution can be reproduced with:

```bash
.venv-sglang-h20/bin/python \
  integrations/sglang/bench_real_model_ablation.py \
  --model /path/to/gpt-oss-20b \
  --rounds 4 \
  --requests 8 \
  --max-running-requests 128 \
  --prompt-tokens 6000 \
  --decode-tokens 32 \
  --context-length 8192 \
  --mem-fraction-static 0.18 \
  --page-tokens 16 \
  --kv-dtype-bytes 2
```

This command invokes the Rust HF frontend and uses its temporary Retention IR
artifact for Policy32 and Owner32. It does not read the hand-written example
plan.

## Qualified conclusion

This evidence supports:

- a released Hybrid Full+SWA checkpoint can execute through OrbitKV's Rust
  owning path;
- the compiled model plan matches the checkpoint's actual layer and KV
  geometry;
- the model plan can be generated directly from the checkpoint config rather
  than manually authored;
- tighter safe SWA reclamation can redirect a fixed KV budget to Full state;
- manual Stock32 reproduces the same capacity, demonstrating that the physical
  policy rather than the plugin produces the capacity gain;
- on the balanced four-way workload, Owner32 reduces median makespan by 20.30%
  versus Stock128 while adding 2.18% median cost versus Policy32;
- fixed-capacity owner overhead is within the measured process noise;
- Stock and OrbitKV output token IDs are identical for every tested request.

It does not support:

- a claim that interval 32 or its capacity benefit is unique to OrbitKV;
- a universal 20.30% speedup across budgets or workloads;
- released-checkpoint quality or benchmark-accuracy improvement;
- radix/prefix-cache, overlap, speculative decoding, or CUDA Graph support;
- a claim that OrbitKV replaces SGLang's physical tensor storage or attention
  kernels;
- results on GPUs other than the recorded H20.
