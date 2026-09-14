# Decode child-graph benefit diagnostic

Status: real-device, released-checkpoint correctness plus narrow matched decode
dispatch evidence. This record does not qualify serving throughput, larger
batches, long-context behavior, capacity, memory reduction, or production use.

The previous correctness-first implementation flattened Luminal's selected
executables into raw launches and was 24.9% slower than eager dispatch. The
current implementation instead constructs one parent CUDA Graph from the
already-searched materialized executables as ordered child nodes. When search
selects materialized K/V updates, their device-to-device copies back into the
stable OrbitKV arena are appended as ordered graph nodes.

Two independent runs used the same released dense decoder checkpoint and the
same batch-one fixed decode step. Eager and parent-graph executions alternated
order, and both paths included identical stable-input uploads and one token-ID
readback.

| Run | Iterations per path | Eager median | Child-graph median | Replay/eager | Wall-time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| confirmation A | 20 | 2228.1 us | 2043.2 us | 0.917 | 8.3% |
| confirmation B | 100 | 4441.8 us | 4182.6 us | 0.942 | 5.8% |

Absolute latency differed across the two process runs, so the result is reported
per run rather than pooled. Both independent alternating comparisons favored
the child-graph path. This supports only a narrow fixed-signature dispatch
benefit; it is not a claim about end-to-end token throughput.

Both runs also completed prefill, decode capture, replay with diagnostic logits,
token-only replay, device/host greedy parity, persistent K/V updates, and four
OrbitKV publications. `cache_updates_in_place=false`, so the test exercises the
parent graph's ordered K/V D2D epilogue.

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
- parent source: `b14a50f`
- Luminal fork: `06ea1928a673b53de5aad132fbc896c534a22621`

## Reproduction

Use the command in the preceding decode CUDA Graph correctness record and set
`ORBITKV_DECODE_BENCH_ITERATIONS` to `20` or `100`. The selected test is
`released_decoder_reuses_one_compiled_runtime_and_kv_arena`.
