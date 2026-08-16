# H20 SGLang Validation — 2026-08-17

## Scope

This record evaluates OrbitKV's first executable SGLang physical policy:

```text
Retention plan
    -> block/page lifetime bound
    -> calibrated SWA reclamation interval
    -> SGLang allocator policy
```

OrbitKV does not replace SGLang's attention kernels, scheduler, physical KV
pool implementation, or request protocol in this experiment. The policy changes
only the page-aligned SWA reclamation cadence. Stock and OrbitKV use the same
SGLang revision, model implementation, Triton kernels, random seed, page size,
and request traces.

The model is a deliberately small dummy-weight `GptOssForCausalLM` system
fixture. It preserves SGLang's real alternating Full/SWA attention semantics
and hybrid KV manager, but it is not a model-quality or accuracy benchmark.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA H20, SM90 |
| OrbitKV | `a5d0f73288b141c97cc98a2a777295b9944470e4` |
| SGLang | `095ec6c997bfdd25d3864cb0ce77a6562a934b96` |
| Python | 3.11.10 |
| PyTorch | 2.13.0+cu130 |
| FlashInfer | 0.6.17 |
| Transformers | 5.12.1 |
| Page size | 16 tokens |
| SWA window | 1,024 tokens |
| Stock eviction interval | 128 tokens |
| Selected OrbitKV interval | 32 tokens |

The fixture and plans are:

- `fixtures/gpt-oss-hybrid-tiny/config.json`
- `fixtures/gpt-oss-hybrid-62l/config.json`
- `examples/gpt_oss_hybrid_tiny.json`
- `examples/gpt_oss_hybrid_62l.json`

## Why interval 32

OrbitKV's semantic lower bound requires 65 blocks, or 1,040 token slots, per
request for `W=1024, P=16`. Executing reclamation every page minimizes resident
slots, but SGLang's current `free_swa` path has non-zero control-plane cost.

The candidate scan measured intervals 16, 32, 64, and the stock interval 128.
Intervals 16/32/64 produced the same measured peak for this workload. Interval
32 retained more static memory savings than 64 while avoiding the more
aggressive interval-16 tail behavior. It is therefore the first calibrated
physical policy, not a universal constant.

## Fixed token-capacity result

The 62-layer fixture contains 10 Full layers and 52 SWA layers. With Full token
capacity fixed at 32,768:

| Metric | Stock | OrbitKV-32 | Difference |
| --- | ---: | ---: | ---: |
| SWA token capacity | 11,536 | 10,768 | -6.66% |
| Reported KV pool | 1.771 GiB | 1.695 GiB | -77.8 MiB |
| Output digest | `9f2e...bf63a` | `9f2e...bf63a` | equal |

This verifies that the slot reduction becomes a real physical-pool reduction
when per-token KV geometry is large enough. The `nvidia-smi` after-load snapshot
is allocator-noisy at this scale, so the authoritative allocation result is
SGLang's pool byte accounting.

## Fixed HBM-budget result

At the same reported 4.608 GiB KV budget and `max_running_requests=32`:

| Metric | Stock | OrbitKV-32 | Difference |
| --- | ---: | ---: | ---: |
| Full token capacity | 33,904 | 49,888 | +47.14% |
| SWA token capacity | 39,920 | 36,848 | -7.70% |
| Output digest | `9275...32ed` | `9275...32ed` | equal |

The reclaimed SWA budget is redirected to the Full pool, which is the binding
resource for long hybrid requests.

## Admission workload

The admission workload contains eight requests, each with 6,000 prompt tokens
and 32 decode tokens, under the same 4.608 GiB KV budget.

Stock SGLang can keep only five requests resident at once. Three requests wait
for a second wave. OrbitKV admits all eight into one wave.

Across three fresh-process pairs:

| Pair | Stock makespan | OrbitKV makespan | Reduction |
| ---: | ---: | ---: | ---: |
| 0 | 9.480 s | 6.381 s | 32.69% |
| 1 | 8.750 s | 6.409 s | 26.76% |
| 2 | 8.749 s | 6.277 s | 28.25% |

Median makespan reduction: **28.25%**.

All runs produced the same output-token digest:

```text
76cf04ef40736dcfe5761952139ce6c1e73c99b41b1a0a7cf05ccead8be3333d
```

No request reported a retraction. The gain comes from capacity-based admission,
not approximate KV eviction or changed model semantics.

## Same-workload overhead

For the smaller 8-layer fixture with a fixed token capacity, nine fresh-process
Stock/OrbitKV pairs produced a median steady-state ratio of `1.0009x`. The
process-level bootstrap interval was wide because this host showed occasional
cross-process outliers:

```text
95% bootstrap ratio: [0.9269, 1.1157]
```

This does not prove a performance speedup or a tight zero-overhead result. It
does show that the selected interval-32 policy's normal-path overhead is small
relative to the capacity benefit, while the more aggressive interval-16 policy
showed worse tail behavior.

## Qualified conclusion

The evidence supports:

- generated block lifetime plans can safely drive an existing inference
  engine's reclamation policy;
- a cost-aware physical policy is better than mechanically using the semantic
  minimum;
- the selected policy reduces allocated KV bytes;
- under a fixed KV budget, the reduction increases admitted Full-context
  capacity and improves long-request makespan;
- Stock and OrbitKV outputs are identical for the tested deterministic dummy
  workload.

The evidence does not yet support:

- model-quality claims on released checkpoints;
- arbitrary Attention-mask compilation;
- radix/prefix-cache, speculative, overlap, or CUDA Graph qualification;
- a production-ready replacement for every SGLang KV allocator;
- the same percentage gain on models with different layer ratios or KV
  geometry.

## Next gate

The next implementation should replace environment-based policy injection with
a typed Rust-to-SGLang plan artifact and extend the optimizer cost model with:

```text
resident bytes
+ reclamation calls
+ measured CPU time
+ descriptor/span cost
```

Then repeat the experiment on a released hybrid model and enable radix cache,
overlap scheduling, and CUDA Graph one at a time.
