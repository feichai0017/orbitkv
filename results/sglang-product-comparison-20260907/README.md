# OrbitKV versus stock SGLang product comparison

Status: completed as a negative performance result with a positive KV-payload
capacity result. This record does not qualify an OrbitKV serving-performance
advantage.

Four alternating release-mode epochs compared the single-process OrbitKV engine
against clean stock SGLang v0.5.17 on one H20. Both arms used the same released
Gemma3 text checkpoint, tokenizer, BF16 weights and KV dtype, page size 16,
1024-token context, eight-request admission limit, 8192 logical KV tokens,
greedy sampling, and one Rust `vllm-bench` request trace. Each arm completed 16
requests of 127 observed input tokens and 256 output tokens at concurrency two,
with zero request failures and no per-request error.

OrbitKV loaded one strict 2.41 MiB Luminal schedule artifact in every measured
epoch. Candidate logs contain no graph search, and all four candidate output
digests are identical. This removes the cross-start search noise seen in an
earlier unqualified run. The artifact is bound to the manifest, decoder and
weight-family geometry, arena shape, and compile buckets; mismatches fail closed.

Median across the four arms:

| Metric | OrbitKV | stock SGLang | OrbitKV / SGLang |
| --- | ---: | ---: | ---: |
| Output throughput | 592.32 token/s | 1112.60 token/s | 0.534 |
| TTFT | 98.08 ms | 15.14 ms | 6.50 |
| TPOT | 2.997 ms | 1.699 ms | 1.76 |
| ITL | 2.858 ms | 1.692 ms | 1.69 |
| End-to-end latency | 862.12 ms | 448.32 ms | 1.92 |
| Ready time | 10.51 s | 30.32 s | 0.347 |

The result is unambiguous: OrbitKV/Luminal is currently substantially slower
than tuned SGLang on this trace. Persisting the selected schedule improved
candidate repeatability and startup time, but did not close the steady-state
execution gap.

There is a separate persistent-state advantage. SGLang v0.5.17 reports that
hybrid Sliding-Window memory is disabled for `Gemma3ForCausalLM`, so its 8192
logical tokens allocate Full-attention K/V payload for all 18 layers. At the
resolved BF16 geometry this is 144.0 MiB. OrbitKV's class arenas hold 24.0 MiB
for three Full layers and 61.875 MiB for fifteen Sliding layers, 85.875 MiB
total: 40.4% less configured K/V tensor payload. These are geometry-derived
payload bytes, not an `nvidia-smi` allocator-peak measurement.

Both engines independently reproduce the existing 512-token eight-token
reference probe. However, the random serving trace produces stable but different
text digests across the engines. BF16 greedy paths can diverge at narrow logit
margins, so the strict cross-engine output-equivalence gate is false. The
performance numbers are retained as a transparent product diagnostic, not a
matched-output benefit claim.

The next performance work is executor-side: integrate fixed-signature decode
CUDA Graph replay into the continuous-batching hot loop, reduce graph-visible
KV materialization/copy cost, and profile the selected schedule against
SGLang's FA3/CUDA-Graph path. KV lifetime heuristics are not the primary blocker
shown by this trace.
