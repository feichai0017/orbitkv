# Selected CUDA module image replay

Decoder schema 6 stores the selected schedule and generated CUDA module images.
The shared Luminal runtime API validates target/NVRTC/options, source identity
and image integrity, and keeps strict lookup active through execution. Reused
provider helper modules participate in capture and completeness checks. Legacy
decoder schema 5 remains readable and recompiles generated kernels. Tests live
under their owning crate's `tests/` directories.

On the local Qwen3.8 27B block-FP8 checkpoint and H20, two fixed-artifact ABBA
timing pairs produce the following results. Each strict timing process has a
separate process with per-node device profiling. All phases also use buffered
CPU stage tracing; process time includes diagnostic logits and is not serving
TPOT or time to the first HTTP response.

| Measurement | Images omitted in memory | Cached images | Observed change |
| --- | --- | --- | --- |
| Strict process time | 38.92 / 38.47 s | 33.21 / 33.36 s | Median 38.69 → 33.28 s, 14.0% shorter |
| Schedule loading | 11.37 / 11.42 s | 4.96 / 4.95 s | Median 11.40 → 4.96 s, 56.5% shorter |
| NVRTC calls per replay process | 428 | 0 | All 428 captured sources hit |
| Warm diagnostic decode median per strict process | 24.50 / 24.51 ms | 24.54 / 24.51 ms | No warm-token benefit established |

The control reads the same immutable artifact, then explicitly omits images
in the test process. Model, tuning, binary, bucket selections, provider caches
and oracle files match. The harness confirms the ablation rather than merely
accepting an environment flag. The complete artifact is 15,236,929 bytes;
its 428 unique CUBINs total 1,886,024 bytes before base64 encoding.

All **nine** processes (fresh search, four strict replays and four profiles)
pass eight teacher-forced reference steps and complete token/fixed-state drain.
Per-step error measurements are identical across processes: maximum absolute
logit error **0.5**, within the unchanged **1.0** gate. The oracle uses
Transformers 5.12.1 and local DeepGEMM 2.6.1; it is an independent model
implementation but shares an underlying math library. Shared FP8 preparation
is opt-in in this bounded tuning manifest; production defaults and highest-index
argmax tie behavior are unchanged.

Fresh creation spends **7.43 s** rebuilding the selected programs to capture
their images, avoiding retention of every rejected search binary. The full fresh
process takes 258.10 s. Its selected programs differ from the preceding stage
record, so this is not evidence of faster search. The replay improvement is
qualified only by the identical-artifact comparison. Weight-loading medians
differ between arms (22.88 s cached, 20.97 s omitted); the schedule span isolates
the eliminated compilation more directly than complete process time. Weights,
graph normalization/extraction, resource validation and CUDA Graph
materialization still run. This result does not improve warm serving throughput.

`identity-audit.json` verifies 877 file references and correlates both selected
programs through measurement, validation and the stored artifact. The additional
module audit checks every image checksum, all source hits, zero NVRTC on cached
replay, fixed artifact bytes, and all nine reference/drain results. The final
audit checks current runtime/test source against the 424-input frozen build.
`checks.json` records seven module tests including two CUDA regressions, three
decoder format tests, 25 engine unit tests, 33 Python tests, Clippy, formatting
and layout checks. Raw source, binaries, artifact images, traces and audit
scripts stay in `.qualification/module-image-artifact-20260913`.

The compatibility and extension contract is in
[generated module artifacts](../../docs/module-artifacts.md). Next work should
split weight conversion/upload costs and address runtime bucket transitions
and generated regions before repeating complete serving qualification.
