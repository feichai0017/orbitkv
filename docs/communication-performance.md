# Local communication measurements

For artifact locations and verification limits, see [benchmark evidence](benchmark-evidence.md).
The [pre-migration report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md) retains full tables and historical run details.

## Repeated serving comparison after layer readiness

The September 29 candidate `de451189` combines raw per-layer CUDA dependencies,
synchronous vLLM external-hit scheduling and SGLang consumer-ordered copies.
Thirty accepted cohorts compare it with the pre-change `f99f14ab` implementation,
native HBM, native CPU offload and LMCache 0.5.5 in each pinned engine. All
**1,536 non-native requests** match their same-round native outputs, with no
missing pairs. Each cohort also matches its own prepared-prefix outputs.

This is the complete implementation increment, not an isolated attribution to
copy/compute overlap or 2D DMA. The baseline uses the repaired wheel identified
below; the candidate uses a source release build. They share the same engine
virtual environment, model, hardware and benchmark inputs. The preserved artifact
hashes identify those builds; the comparison does not eliminate build-provenance
variation. Native overlap itself has a separate
[CUDA event and graph-replay gate](engine-local-restore.md#layer-readiness-qualification-2026-09-29).

The workload uses one H20, Qwen3-8B, TP=1, CPU set `8–23`, 16,384 GPU KV tokens,
8,192 prefill tokens and 16 GiB host capacity. It has concurrency 4, twelve
alternating 1,024/4,096-token prefixes, 49 reuse and 15 cold requests, 16 output
tokens and seed `20260920`. Its 30,720-token working set exceeds HBM capacity;
HBM-only results include recomputation after eviction. This does not compare
external restoration with a resident HBM hit. Both cost observations and tracing
are disabled. vLLM uses batch-invariant inference and SGLang deterministic
inference, so compare backends within each engine.

Cells are **medians of three per-run measurements**, not pooled percentiles.

[Full historical measurement table](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md)

vLLM throughput improves **18.67%** over the old implementation, **13.93%** over
native CPU offload and **25.36%** over LMCache in this workload. TTFT p50 improves
28.12% versus the old implementation. All three candidate throughput values
(5.1765–5.2056 requests/s) exceed all three baseline values (4.3691–4.3922).
The change also removes an asynchronous load scheduling interval; these numbers
must not be presented as the isolated effect of overlapping GPU copies.

SGLang does **not** demonstrate a throughput improvement from this increment:
7.2402 versus 7.2556 requests/s, **-0.21%**, with overlapping ranges. Its TTFT p50
improves 3.33%, while p95 increases 0.45%. Candidate throughput remains **1.67%**
below native CPU offload and **6.27%** above LMCache. Its p95 remains 2.92% above
native CPU and 0.28% above LMCache. Keep the native CPU gap open; native event
correctness and observed overlap do not imply a serving speedup in every engine.

Every cohort passes its post-window cache/drain assertions. The 100 ms process
audit observes no Cargo/rustc activity, and all Manager, extension, adapter and
benchmark source hashes remain unchanged during each run. Small standalone C++
checks on CPU set `0–3` overlapped part of the matrix; large compilation and other
GPU qualification started only after it finished. This is not proof of exclusive
machine use or a hard affinity budget for every native worker thread.

### Artifacts and reproduction

Raw logs, commands, per-cohort audits, output comparisons and
`accepted-summary.json` are retained under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/layered-restore-20260929/deterministic-c4/`.
The frozen baseline wheel SHA256 is
`eef36f8406ae854a9ea1567da5ac09adba5a3d6f17077cca99d7be6fdd3ff163`.

| Build | Manager SHA256 | Extension SHA256 |
| --- | --- | --- |
| Before | `85a38e9b5600854524cf3f3505ae79b9b1dc1e67594a4c3f2b4d2bef17db4ed5` | `65437c4b7e7985c4265c449b23129a96325e74124dcf14b7f7ab01497d58140b` |
| After | `93e962a4e2be44bd42a7ec255ab4fa1cd541ab3268cffc69ba7190c74cdc2be2` | `e4b4d552e2c98c8feddaa0142b8d60b3334b507256df82db8fef603a89ab5b70` |

Use the command in the strided-DMA section below with `--settle-seconds 1.2`.
Run both engines in three backend orders:
`native,cpu,before,after,lmcache`, `lmcache,after,before,cpu,native`, then
`cpu,native,before,lmcache,after`. Before/after both select `--backend orbitkv
--orbitkv-transfer-backend direct`, with their own matching Manager, extension
and Python package. Give every cohort a new output directory. Run
`python -m benches.report ... --reference-run <same-round-native-run>` for
matched output checks; retain the audit alongside each result. Preparation,
cache drain and engine startup are outside throughput; all admitted request
completions remain inside the measured window.

## Repeated serving comparison after strided DMA

Native source `d94a3b26` adds registration-aware 2D CUDA copies. A matched
Qwen3-8B gate compares each engine's HBM-only cache, native CPU offload,
OrbitKV raw DRAM Restore and LMCache 0.5.5. It uses one H20, TP=1, CPU set
`8–23`, 16,384 GPU KV tokens, 8,192 prefill tokens and 16 GiB host capacity.
The deterministic workload has concurrency 4, 12 prefixes, 1,024/4,096 input
tokens, 16 output tokens, seed `20260920` and 64 measured requests per cohort.
Its 30,720-token working set exceeds HBM capacity. Preparation and cache drain
are outside measured throughput; the complete admitted request window is inside.

Cells are **median of three per-run measurements**, not pooled percentiles.

| Backend | vLLM requests/s | vLLM P95 TTFT, ms | SGLang requests/s | SGLang P95 TTFT, ms |
| --- | ---: | ---: | ---: | ---: |
| Native HBM only | 3.1388 | 1172.28 | 3.5583 | 1249.66 |
| Native CPU offload | 4.5645 | 654.16 | 7.3567 | 661.56 |
| OrbitKV | 4.3784 | 636.60 | 7.1914 | 706.42 |
| LMCache | 4.1629 | 820.45 | 6.8314 | 677.26 |

OrbitKV throughput is 5.18%/5.27% above LMCache for vLLM/SGLang, but
4.08%/2.25% below native CPU offload. SGLang P95 TTFT is 6.78% above its
native CPU cache and 4.31% above LMCache. The native HBM comparison includes
recomputation after eviction; it is not a resident-HBM latency comparison.
These results do not establish that 2D DMA itself caused a serving improvement:
a repeated before/after implementation control is still required. The
native-CPU throughput gap and SGLang tail remain open acceptance items.

All 24 accepted cohorts completed. Across them, 1,152 non-native requests
matched the text of their corresponding native controls, with no missing pairs.
This qualifies the synthetic deterministic output controls, not model quality.
Both cost collection and tracing were disabled. A 100 ms process audit found no
Cargo/rustc activity in accepted cohorts; Manager and extension SHA-256 hashes
remained unchanged. The initial third round detected a background Cargo check
and was discarded; the complete round was rerun. This is a process audit, not
proof of exclusive machine use.

Reproduce from the repository root with the pinned engine environment. Use
`--engine vllm|sglang` and `--backend native|cpu|orbitkv|lmcache` for each
cohort; only OrbitKV receives `--orbitkv-transfer-backend direct`:

```bash
ORBITKV_COST_OBSERVATIONS=0 ORBITKV_TRACE_TRANSFERS=0 \
  taskset -c 8-23 .venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --orbitkv-transfer-backend direct \
  --model /workspace/models/Qwen3-8B --output /path/to/fresh-run \
  --workload sustained --lengths 1024 4096 --concurrencies 4 \
  --duration-seconds 60 --max-requests 64 --working-set 12 --reuse-ratio 0.75 \
  --gpu-tokens 16384 --prefill-tokens 8192 --host-gib 16 --output-tokens 16 \
  --seed 20260920 --deterministic-inference
```

Run three rounds in backend orders `native,cpu,orbitkv,lmcache`,
`lmcache,orbitkv,cpu,native`, then `cpu,native,lmcache,orbitkv`. Use each round's
native cohort as the explicit `benches.report --reference-run`. Raw logs,
commands, artifact hashes, per-cohort audits and the rejected round are retained
under `/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/strided-dma-20260928/deterministic-c4/`.

## Completion evidence and copy-path diagnosis

The [archived experiment report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md#completion-evidence-and-copy-path-diagnosis)
records the measured revision, controls, unsuccessful runs and reproduction commands.
These earlier measurements do not qualify the current implementation.

## Compacted raw plans and idle-stream readiness

The [archived experiment report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md#compacted-raw-plans-and-idle-stream-readiness)
records the measured revision, controls, unsuccessful runs and reproduction commands.
These earlier measurements do not qualify the current implementation.

## Engine-local raw Restore: functional cutover, measured latency regression

The [archived experiment report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md#engine-local-raw-restore-functional-cutover-measured-latency-regression)
records the measured revision, controls, unsuccessful runs and reproduction commands.
These earlier measurements do not qualify the current implementation.

## Client-reserved Restore identity (preceding increment)

The [archived experiment report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md#client-reserved-restore-identity-preceding-increment)
records the measured revision, controls, unsuccessful runs and reproduction commands.
These earlier measurements do not qualify the current implementation.

## Raw descriptors and split K/V coalescing

The [archived experiment report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md#raw-descriptors-and-split-kv-coalescing)
records the measured revision, controls, unsuccessful runs and reproduction commands.
These earlier measurements do not qualify the current implementation.

## Earlier restore preparation increment

The [archived experiment report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md#earlier-restore-preparation-increment)
records the measured revision, controls, unsuccessful runs and reproduction commands.
These earlier measurements do not qualify the current implementation.

## Earlier request-notification increment

The [archived experiment report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/communication-performance.md#earlier-request-notification-increment)
records the measured revision, controls, unsuccessful runs and reproduction commands.
These earlier measurements do not qualify the current implementation.
