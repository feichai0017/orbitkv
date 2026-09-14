# CUDA weight loading without intermediate byte copies

The CUDA loader now lives in `runtime/weights.rs`, with pure encoding conversion
in `weights/convert.rs` and private tests under `tests/unit/runtime/weights/`.
Its single fallible API returns actual tensor/byte counts. Compatible checkpoint
storage uploads directly from an immutable mapping; conversion retains one
typed allocation and exposes its byte view. Unsupported encodings and malformed
files fail before bindings change. CUDA failures propagate to initialization;
this is not transactional live weight replacement.

Decoder artifacts now require schema 6 with generated module images. The legacy
schema-5 decoder mode, image-omission API and qualification flag are removed.
Older artifacts must be regenerated. The historic module-image experiment
retains its original source and record.

Two timing samples per arm, executed in ABBA arm order on H20 with the local
Qwen3.8 27B block-FP8 checkpoint, use the **same immutable selected artifact**.
Each timing process has a separate process with CUDA Graph profiling; all eight
processes use buffered CPU stage tracing. There is implementation/build/check
work between baseline/a and final/a. Existing provider and OS caches are reused.

| Measurement | Frozen old loader | Final loader | Median change |
| --- | --- | --- | --- |
| Weight loading | 22.33 / 23.90 s | 5.68 / 7.53 s | 23.12 → 6.60 s; **71.4% shorter** |
| Decoder compile-or-load during strict replay | 31.58 / 33.12 s | 15.11 / 16.75 s | 32.35 → 15.93 s; **50.7% shorter** |
| Complete diagnostic process | 32.57 / 34.16 s | 16.29 / 17.79 s | 33.36 → 17.04 s; **48.9% shorter** |
| Warm diagnostic decode, median per timing process | 24.50 / 24.58 ms | 24.55 / 24.60 ms | No warm-token benefit established |

The instrumented old loader spends **17.36 / 18.52 s** making intermediate host
byte copies. Its conversion takes only **2.13 / 2.60 ms**. Removing those copies
addresses the dominant measured cost. Final upload API calls take 4.72 / 6.22 s;
they now first-touch mapped checkpoint bytes and can include page faults and
driver staging. This is not isolated DMA bandwidth. Mapping timers measure
mapping creation, not all physical disk I/O. The new per-shard completion waits
total about 0.6 ms per timing process and run even without tracing.

All eight processes bind the same **1,251 tensors** from **66 shards**, with
**705 conversions**, **29,468,003,328 source bytes** and **29,472,331,776 device
bytes**. Binding labels, source/target dtypes and source sizes match across arms;
the audit checks device upload totals against the final loader's reports.
These are logical binding bytes, not a peak-memory measurement. CUDA may still
stage pageable host memory internally.

All processes hit all **428** module images with **zero NVRTC compilations**,
pass eight teacher-forced reference steps and complete token/fixed-state drain.
Per-step measured errors are identical; maximum absolute logit error is **0.5**
under the unchanged **1.0** gate. The oracle uses independent Transformers
5.12.1 model code with local DeepGEMM 2.6.1, sharing the underlying math library.
Highest-index argmax tie behavior and the opt-in shared-FP8 tuning are unchanged;
shared FP8 remains off by default in production.

`identity-audit.json` verifies **1,334** file references, frozen baseline/final
source manifests and binaries, artifact images, per-process inputs, stage traces
and numerical/drain records. The 428 captured source/input files also match the
current workspace. Raw build inputs, binaries, artifact copies, traces and
scripts remain under `.qualification/weight-loading-20260913/`.
Source snapshots have no Git identity of their own: the frozen runner's parent
Git observations are superseded by the explicit build receipts. The current
runner now rejects parent-checkout inheritance and has a regression for it.

`checks.json` records four CPU conversion tests, two explicit CUDA loading
regressions, three artifact format tests, 25 engine tests, 20 relevant Python
tests, all-target workspace checks, root/CUDA Clippy, formatting and source
layout validation. The CPU conversion suite exhaustively checks all FP16/BF16
encodings against an independent value formula and tests rounding and
unaligned input. The CUDA tests verify exact uploaded bytes after mapping
release, stale-mirror removal and failed-shard behavior. Five prior result
packages retain their original checksums.

This qualifies bounded replay startup with existing caches and diagnostic
logits. It does not establish server TTFT, serving throughput, cold-disk loading
or faster compiler search. The final replay still spends about 4.29 s building
the graph and 4.97 s loading its schedule. Runtime bucket resource ownership
and measured operator regions are the next serving-performance work; see the
[loader contract](../../weight-loading.md) and [roadmap](../../roadmap.md).
