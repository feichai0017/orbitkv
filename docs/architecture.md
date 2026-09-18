# AletheiaRT architecture

## System boundary

```text
PyTorch model / exported graph              production request stream
              |                                         |
              v                                         v
 TensorRT-LLM AutoDeploy transforms          source-pinned SGLang
  + bounded candidate generation        scheduler / KV / ModelRunner
              |                                         |
              v                                         v
      qualification harness  ----->  qualified plan registry
  correctness / memory / perf               | selection by shape + SLO
              ^                             v
              |                       qualified executor
              |                      prepare / run / fallback
              |                             |
              +------ trace + replay <------+
                                            |
                         +------------------+------------------+
                         v                  v                  v
                     FlashInfer          vendor             generated
                       provider          provider             provider
```

The trusted control plane is independent of the production serving frontend and
the GPU providers. It deals only in immutable schemas and content digests.

## Workspace

| Path | Responsibility | Forbidden dependencies |
| --- | --- | --- |
| `aletheia-contracts` | Portable plan, evidence, workload, SLO, trace schemas | CUDA, Python, serving frameworks |
| `aletheia-control` | Qualification checks, registry, deterministic selection | Provider SDKs, request serving |
| `aletheia-executor` | Provider ABI, plan preparation, execution and fallback | Model-specific branches |
| `aletheia-cli` | Inspection and local orchestration | Hidden policy |
| `integrations/sglang` | In-process batch/execution trace hook | Core schema ownership |
| `integrations/autodeploy` | AutoDeploy graph inventory and future candidate transforms | Serving lifecycle |
| `integrations/providers` | Source provenance and provider qualification helpers | Scheduler policy |
| `third-party/*` | Exact upstream sources and their native build systems | Local product policy |

Dependencies point inward: integrations and providers may depend on the core;
the core never depends on them.

## Physical plan

A plan binds a semantic model digest and a workload domain to an ordered set of
provider steps. Each step identifies its provider, operation, immutable artifact
digest, input/output names, and provider configuration. A plan also declares
peak memory, workspace, address-stability requirements, capture mode, and an
ordered fallback list.

The plan is not trusted because it parses. `aletheia-contracts` validates structural
invariants; the registry additionally requires a certificate whose plan digest
matches the canonical serialized plan.

## Qualification certificate

A certificate contains evidence rather than a boolean:

- oracle identity, case count, numerical bounds, and output agreement;
- measured workload digest, raw sample count, latency distribution and goodput;
- observed and certified memory bounds;
- soak duration and named fault cases;
- exact hardware and software fingerprints;
- the workload domain over which the evidence is valid.

The control plane admits only `qualified` certificates. Revocation is explicit.
Future signing can wrap the same canonical certificate without changing plan
selection.

M1 certificates cover one exact shape bucket. A candidate is publishable only
when a successful SGLang trace contains that exact phase, batch, rows-per-seq,
and context point; at least twelve raw device timing samples exist; numerical
and resource evidence is present; and every referenced artifact's bytes match
its declared digest. Broader ranges require independent evidence and a future
explicit schema rather than a min/max extrapolation.

## Selection

A request to select a plan contains model semantics, exact hardware, one
workload point, and an SLO. The registry removes plans that are incompatible,
out of domain, over memory, numerically insufficient, or slower than the SLO.
Remaining plans are ordered deterministically by operator preference, goodput,
p99 latency, memory, then plan ID. Fallbacks receive the same eligibility check.

Runtime adaptation means selecting among this finite qualified set. Candidate
generation and benchmarking remain outside the request path.

## Execution

The executor resolves every step's provider and prepares all artifacts before a
plan becomes live. It executes the primary plan, records a typed failure, and
tries its prequalified fallbacks in order. A provider cannot mutate the registry
or silently choose an unqualified artifact.

Bundles are published through `tools/stage_bundle.py`: it calls the Rust
validator for the plan, builds the certificate from raw evidence, verifies and
copies content-addressed artifacts, validates the certificate, atomically
renames the bundle, then validates the entire registry fallback graph. Any
failure removes the candidate bundle.

The initial reference provider is deliberately trivial. Its purpose is to test
the lifecycle without CUDA. GPU providers will implement the same interface and
add asynchronous completion and persistent-state contracts before serving real
models.

## Source and process topology

SGLang is the initial serving owner. Our `sglang.srt.plugins` hook observes the
actual scheduler-to-`TpModelWorker` path; kernel choices flow through SGLang's
existing attention, GEMM, MoE, and compilation backends. We do not proxy to a
second inference server.

TensorRT-LLM AutoDeploy runs offline in a separate environment. Its registered
transforms inspect or transform exported graphs and produce candidate metadata.
It is both a reusable compiler and a strong independent baseline; it is not
linked into the SGLang request process.

FlashInfer is installed editable from its submodule. DeepGEMM and
`sglang-kernel` are built as local wheels from their pinned submodules because
they contain native extensions. Their JIT/AOT products become content-addressed
artifacts referenced by a physical plan. Provider qualification imports must
pass a provenance check showing that the module came from the pinned checkout
rather than an unrelated wheel.
