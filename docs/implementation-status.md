# Implementation status

Support is reported separately for compiler representation, manager lifecycle,
executor lowering, device execution, complete-model execution, and measured
benefit. A model is supported end to end only when every required state and
operator passes all applicable layers.

## Current implementation

| State or attention family | Compiler and manager | Executor | Complete model | Benefit |
| --- | --- | --- | --- | --- |
| Full MHA/GQA token KV | Implemented and host-tested | Paged attention, Prefix/COW, writes, relocation | Narrow released-checkpoint H20 closure | No lifetime-management L5 result |
| Sliding Window token KV | Periodic placement, retirement, ACK, and generation reuse host-tested; request-lifetime residence provides a same-semantics baseline | CSR/window lowering implemented; compiled/baseline CSR geometry is host-matched | Native Sliding layers cross a 512-token window in the released hybrid H20 closure | Released-hybrid matched run: Sliding residency 48 to 32 pages; no serving-throughput claim |
| Full + Sliding interleaving | Independent class lifetimes and joint transactions host-tested | Manifest-driven per-layer graph construction, independent arenas, write slots, CSR metadata, and capture signatures pass host tests | Released 18-layer 3-Full/15-Sliding checkpoint passes independent token parity, retirement/reuse, cancellation, and final drain on H20 | Narrow same-executor L5: 27.8% less resident payload, 6.1% longer fixed-budget boundary, 0.77% lower median test-path time |
| Exact Chunked attention | Resettable epoch arena host-tested; one whole-domain class only | Metadata lowering implemented | Not independently device-qualified | Unproven |
| MLA/latent KV | Component-aware latent/RoPE lifecycle compiles | Matching Luminal attention kernel contract missing | Unsupported | Unproven |
| Mamba/GDN/KDA/linear attention | Recurrent checkpoint geometry compiles and checkpoint pool is host-tested | Recurrent operators are not integrated into one transaction | Unsupported | Unproven |
| Convolution state | Generation-checked checkpoint lifecycle host-tested | Convolution operator/state transaction missing | Unsupported | Unproven |
| Sparse, tree, speculative, cross-attention | No complete general contract | Missing | Unsupported | Unproven |

The native dense decoder accepts multiple token-KV classes with exact,
non-overlapping layer coverage. Each class owns independent dynamic page
metadata and arena geometry. Its configuration-driven block vocabulary now
covers both pre-norm/SwiGLU and sandwich-norm/GeGLU dense decoders, direct or
unit-offset RMSNorm weights, optional QKV bias and QK norm, non-hidden query
widths, and per-layer local/global RoPE. A safetensors-header contract verifies
every required tensor, shape, dtype, and optional family before graph search.
Unknown or incomplete semantics fail closed instead of silently becoming Full
attention.

The released hybrid qualification uses the checkpoint's unmodified native
attention schedule. Four short probes and the complete 34-token greedy sequence
match a separate Transformers run. The 512-token prefill plus 33 decodes cross
the Sliding window, observe retirement and generation reuse, then execute and
cancel a second request using recycled storage; both releases drain all manager
state and arenas. This is an L4 correctness/lifecycle result, not a performance
or model-family-wide claim.

The manager exposes `PhysicalResidencePolicy::RequestLifetime` only through an
explicit constructor. It preserves compiler-authored Sliding token
dispositions and execution visibility while retaining physical pages through
request release. Host tests prove identical attention geometry and token
semantics, different physical residency, generation reuse in compiled mode,
and a fixed-capacity admission difference. Chunked reset, Prefix publication,
external export, and relocation reject this diagnostic baseline.

## External tiers

Backend-neutral export, restore, replica admission, deletion acknowledgement,
completion ordering, and ambiguity quarantine are implemented. A host-memory
reference adapter moves real bytes and verifies a partial-tail round trip across
sessions. Mooncake/NIXL, remote lease renewal, eviction races, shared Prefix
restore, and distributed recovery remain open.

## Serving

The async local `Engine` contract and optional vLLM Rust frontend adapter are
implemented and host-tested. The `orbitkv-engine` composition root has bounded
admission and output queues and combines independently submitted fresh prompts
into decode-first token-budgeted batches over one `RuntimeSession` and compiled
Luminal decoder. On H20, two concurrent 512-token hybrid requests execute with
B=2 prefill/decode and match the existing reference prefix; a late prefill also
joins an active decode request. Length, stop-token, cancellation, backpressure,
and complete drain are covered. The same released checkpoint also passes a
direct B=1/B=8 teacher-forced parity gate: all B=8 rows are bit-identical,
maximum absolute cross-batch logit difference is 0.4296875, and 16 tested
argmax decisions match. Sampling remains
greedy. Chunked prefill, fairness, and production soak remain open.

The `orbitkv-serve` binary provides the complete single-process HTTP product
path. A released hybrid checkpoint passes real H20 OpenAI completion tests with
pre-tokenized prompts, matching tokenizer assets, non-streaming output, ordered
SSE token IDs plus `[DONE]`, concurrent requests, client-disconnect auto-abort,
graceful shutdown, and manager final drain. This is a correctness closure, not a
general performance result. A fixed 16-request trace completes at C1/C2/C4/C8
with full 256-token outputs and no errors. Throughput rises from 184.24 to
518.88 output token/s, but median TTFT rises from 163.06 to 1127.81 ms and TPOT
from 4.80 to 11.06 ms. This qualifies a narrow internal scaling frontier, not a
win over SGLang.

A strict schedule artifact now removes cross-process search variation. It stores
no model weights, device pointers, or KV bytes; loading rebinds current custom
ops and verifies LLIR fingerprints plus the manifest/model/arena/bucket identity.
Four artifact-loaded H20 restarts produced identical candidate output digests
and 0.74% output-throughput coefficient of variation.

Persistent K/V state is now a compiler constraint rather than a post-search
observation. Luminal rejects candidates and stored artifacts unless every K/V
output aliases its registered input arena in every retained bucket. A deeper
16-candidate search produced 36/36 in-place tensors and zero copy-back bytes.
Four alternating C2 epochs improved the same engine's throughput by 14.4%, TTFT
by 32.0%, TPOT by 10.1%, and E2E by 12.6% versus the previous artifact. The new
artifact passes the existing independent B2 reference-token probe.

## Evidence interpretation

The child CUDA Graph result demonstrates a narrow dispatch optimization. The
released-hybrid residence experiment is the first narrow L5 compiler result:
ten paired release-mode epochs show a 27.8% live-payload reduction, a fixed-budget
boundary increase from 528 to 560, and a positive paired total-time interval.
It is not a comparative serving benefit. TTFT/TPOT/tails and internal concurrency
scaling are measured through C8, while fairness, soak, and the capacity limit
remain open. The current tuned comparison remains negative: at C2 OrbitKV
reaches 0.598x stock SGLang throughput with 1.59x TPOT and 4.69x TTFT. Its configured
persistent K/V tensor payload is 40.4% smaller on the same Gemma3 capacity
because SGLang disables hybrid SWA memory. Executor performance, not KV lifetime
correctness, is the next blocker.
