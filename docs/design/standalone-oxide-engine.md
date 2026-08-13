# Standalone Oxide Infer architecture

**Decision:** accepted target on 2026-08-14. This decision supersedes the
earlier proposal to embed a branded upstream serving shell.

Oxide Infer will become a standalone Rust LLM inference engine. Product code,
crates, features, configuration, telemetry, and release artifacts use Oxide
names. Its server, request lifecycle, scheduler, tokenizer, streaming, and
other non-compute capabilities will start from a pinned Mistral.rs source
baseline and retain that architecture where it fits the Oxide boundary. No
external engine remains a product dependency, component, or runtime owner.

TileLang is the only source language for product custom compute kernels. It
runs in the offline build and qualification environment. The production Rust
process loads immutable cubin or PTX artifacts through the checked Oxide CUDA
runtime.

This is a target architecture, not a current implementation claim. The current
tree remains a checked operator runtime with cuda-oxide and cuBLASLt paths
until the migration phases remove them.

## Product identity

The product is one engine:

```text
oxide-infer serve --model <model> --dtype bf16
```

Its intended position is:

- Rust-native API, scheduling, model execution, and GPU resource ownership;
- an OpenAI-compatible server with no Python production runtime;
- a narrow, fail-closed model surface optimized deeply before it expands;
- ahead-of-time TileLang kernels with versioned ABI and artifact hashes;
- evidence at kernel, model, scheduler, and complete-serving levels.

External engine names are permitted only in license notices, provenance,
historical evidence, and benchmark-baseline records. They do not appear in
product crate names, features, configuration keys, provider identities, log
fields, or user-facing architecture diagrams.

## System architecture

```text
OpenAI-compatible clients
          |
          v
oxide-infer-server
  HTTP/SSE, request validation, tokenizer, chat templates,
  tool and grammar request state, response assembly, telemetry
          |
          | EngineRequest / EngineEvent
          v
oxide-infer-engine
  admission, sequence state, continuous batching,
  weight loader, model IR, prefill/decode plans,
  KV pager, logits processing, sampling, cancellation
          |
          | checked plans and operands
          v
oxide-infer-cuda
  memory, streams, events, command scopes, completions,
  CUDA Graphs, artifact registry, checked launches
          |
          | manifest-verified artifact ABI
          v
AOT TileLang cubin/PTX -> CUDA Driver -> NVIDIA GPU
```

`oxide-infer-server` never owns a GPU tensor or KV page.
`oxide-infer-engine` never owns HTTP types. `oxide-infer-cuda` never owns a
model, sequence, tokenizer, or scheduling decision. TileLang source never
enters the serving process.

## Ownership

### Server and external capabilities

`oxide-infer-server` owns:

- OpenAI-compatible routes and streaming responses;
- tokenizer and chat-template loading;
- request parsing, validation, authentication hooks, and limits;
- tool, grammar, and response-format request state;
- cancellation propagation and response assembly;
- user-facing configuration, metrics export, and process lifecycle.

The initial implementation imports the reusable control-plane code from the
pinned source baseline rather than rewriting it for novelty. Imported code is
renamed to Oxide concepts and narrowed to a stable `EngineRequest` and
`EngineEvent` boundary. It does not bring the upstream engine's tensor,
model-forward, KV-cache storage, provider, or pipeline abstractions into the
product dependency graph.

### Engine

`oxide-infer-engine` owns:

- model configuration and direct safetensors loading;
- Oxide-managed weights and device placement;
- a narrow model IR and immutable prefill and decode plans;
- request admission, sequence state, and continuous batching;
- paged KV allocation, sharing, copy-on-write, eviction, and block tables;
- logits transforms, RNG state, sampling, logprobs, and finish reasons;
- cancellation, overload, out-of-memory, and teardown behavior;
- artifact coverage and engine-step telemetry.

The model IR is not a general tensor framework. It represents only admitted
model profiles. If a complete plan cannot be constructed with registered
artifacts, model loading fails. The engine cannot interpret the missing
operation through a hidden framework path.

### CUDA runtime

`oxide-infer-cuda` owns:

- typed owned and external device regions;
- streams, events, workspace, status, and completion lifetimes;
- CUDA module and function loading;
- artifact manifest, ABI, hash, target, and shape validation;
- eager command submission and fixed-address CUDA Graph execution;
- provider and artifact hit accounting.

CUDA Driver, CUDA Graph, and NCCL APIs are infrastructure interfaces, not
custom compute-kernel providers. Product GPU computation violates the target
rule if it executes through FlashInfer, cuBLASLt, CUTLASS, Triton, handwritten
CUDA, cuda-oxide, or a framework CUDA kernel.

## Stable Rust boundaries

The public server-to-engine boundary is asynchronous request state, not a
framework tensor adapter:

```rust
pub trait InferenceEngine {
    fn load_model(&self, spec: ModelSpec) -> Result<ModelHandle>;
    fn submit(&self, request: EngineRequest) -> Result<RequestHandle>;
    fn cancel(&self, request: RequestId) -> Result<()>;
    fn poll(&self) -> Result<Vec<EngineEvent>>;
}
```

Inside the engine, the scheduler constructs batches owned by Oxide:

```rust
pub struct EngineBatch {
    pub request_ids: Vec<RequestId>,
    pub input_tokens: Vec<u32>,
    pub sequence_offsets: Vec<u32>,
    pub positions: Vec<u32>,
    pub kv_handles: Vec<KvHandle>,
    pub mode: BatchMode,
}
```

These sketches fix ownership, not the final public API. A concrete type is
added only with the vertical slice that uses it.

## Target repository layout

```text
oxide-infer/
|-- apps/
|   `-- oxide-infer/              CLI and server binary
|-- crates/
|   |-- oxide-infer/              operator contracts and CPU references
|   |-- oxide-infer-cuda/         checked CUDA runtime and artifact launcher
|   |-- oxide-infer-engine/       model IR, scheduler, KV pager, execution
|   |-- oxide-infer-server/       API, tokenizer, streaming, process control
|   `-- oxide-infer-lab/          correctness and performance evidence
|-- kernels/
|   `-- tilelang/
|       |-- src/                  kernel definitions and fixed schedules
|       |-- build/                compiler and manifest generation
|       |-- profiles/             admitted model, shape, and target matrices
|       `-- tests/                independent correctness fixtures
|-- artifacts/
|   `-- manifests/                release manifests; binaries stay external
|-- benchmarks/
|   |-- kernels/                  matched FlashInfer comparisons
|   `-- engines/                  neutral serving driver and baselines
|-- docs/
|   |-- design/
|   |-- development/
|   |-- provenance/               adapted-source and license mapping
|   `-- results/                  immutable evidence
|-- UPSTREAM.md
|-- THIRD_PARTY_NOTICES.md
|-- website/
`-- .github/
```

This is a target tree. Directories and crates land with their first executable
vertical slice, not as empty placeholders.

The existing `oxide-infer` contract crate and `oxide-infer-cuda` runtime keep
their names during migration so historical evidence and downstream paths stay
understandable. `oxide-infer-engine` is justified by model and KV state.
`oxide-infer-server` is justified by its HTTP, tokenizer, and process
dependencies. The binary under `apps/oxide-infer` composes both without
putting server dependencies into the engine.

## Dependency direction

```text
apps/oxide-infer
  -> oxide-infer-server
       -> oxide-infer-engine
            -> oxide-infer-cuda
            -> oxide-infer

oxide-infer-lab
  -> all product crates needed by one evidence gate

offline TileLang build
  -> versioned artifacts and manifests
  -> oxide-infer-cuda loads them at runtime
```

Dependencies never point from contracts or runtime back to the engine or
server. Benchmark baselines never become product dependencies.

## TileLang artifact supply chain

```text
TileLang Python source
        |
        v
pinned compiler + deterministic build profile
        |
        v
candidate correctness and offline tuning
        |
        v
fixed schedule -> cubin/PTX + manifest
        |
        v
release packaging -> checked Rust artifact registry
```

Production servers do not import Python, compile kernels, auto-tune, access
the network, or mutate the artifact registry. Every artifact records:

- contract version and stable algorithm identity;
- TileLang source, compiler, CUDA toolkit, and target identities;
- dtype, layout, shape domain, numerical, and determinism limits;
- parameter ABI, alignment, alias, launch, and workspace requirements;
- cubin or PTX SHA-256 and source-tree identity;
- correctness, sanitizer, Graph, and benchmark record identities.

The runtime rejects incompatible devices, hashes, ABIs, shapes, or layouts
before launch. Enqueue cannot tune, switch algorithms, or fall back.

## Source-derived control plane

Oxide uses a source transplant, not an embedded shell and not a runtime
adapter. One pinned upstream commit seeds the non-compute modules. The import
is performed as reviewable commits:

1. record the source URL, commit, license, and original file hashes;
2. import the selected source with copyright headers intact;
3. mechanically rename crates, modules, types, features, and configuration to
   Oxide identities;
4. replace the upstream pipeline/tensor boundary with `EngineRequest`,
   `EngineEvent`, and Oxide-owned sequence handles;
5. remove code that can reach Candle GPU execution, upstream paged attention,
   quantized kernels, or another provider;
6. add parity tests before behavior changes or performance tuning.

"Complete reuse" means retaining all useful behavior and implementation from
the admitted non-compute modules. It does not mean copying every source file
whose filename is not `kernel`: model layers, quantized modules, cache objects,
device mapping, speculative execution, and some sampling paths mix control
state with framework tensors and must be split or replaced.

The initial source map is defined in
[Control-plane source map](control-plane-source-map.md).

## External-source and provenance policy

Oxide reuses external capabilities in three explicit forms:

| Form | Treatment |
| --- | --- |
| Behavioral reference | Reimplement from a documented contract and test it against the reference |
| Source-derived module | Import the complete admitted module, rename its product boundary, and record source commit and modifications |
| Unmodified dependency | Pin the package and keep its public identity; use only for generic libraries such as tokenizers |

Copied or adapted code keeps its original copyright and MIT notice as
required. `THIRD_PARTY_NOTICES.md` names the source project and license.
`UPSTREAM.md` records the pinned baseline and update procedure.
`docs/provenance` maps each derived module to an exact source path and
revision. Git history does not replace those release-visible notices.

The source engine also remains an independently built benchmark and behavioral
oracle outside the Oxide product dependency graph. This separation makes
output and performance comparisons meaningful and prevents reference paths
from becoming silent production fallbacks.

## Naming policy

The target product uses these identities:

| Concern | Identity |
| --- | --- |
| Binary and product | `oxide-infer` |
| HTTP and process layer | `oxide-infer-server` |
| Scheduling and model execution | `oxide-infer-engine` |
| GPU runtime | `oxide-infer-cuda` |
| Operator contracts | `oxide-infer` |
| Custom compute provider | `OxideTile` |
| Kernel artifact source | `TileLang` |

External engine names may remain in historical file names and immutable
records. A rename does not rewrite evidence provenance.

## Request execution

```text
HTTP request
  -> Oxide tokenizer and request validation
  -> EngineRequest
  -> admission and continuous-batch scheduler
  -> EngineBatch
  -> KV pager and immutable PrefillPlan or DecodePlan
  -> registered TileLang artifacts
  -> logits and Oxide sampling
  -> EngineEvent
  -> streaming response
```

Weights and KV pages are Oxide allocations from the start. No server boundary
passes a Candle or other framework GPU tensor. Common decode batch classes may
use fixed-address Graph plans; a Graph miss selects a registered eager
TileLang plan, never another provider.

## First implementation slice

The first complete profile is BF16 Qwen2.5-1.5B on one NVIDIA GPU:

1. safetensors loading and embeddings;
2. RMSNorm, RoPE, dense GEMM, SwiGLU, and residual operations;
3. paged prefill, paged decode, and KV append;
4. final norm, logits projection, and greedy token selection;
5. request admission and continuous batching;
6. one OpenAI-compatible streaming endpoint.

Completion requires identical greedy token IDs against the frozen reference,
complete artifact-hit accounting, no unregistered GPU computation, and the
kernel and engine evidence gates.

## Benefits and costs

The standalone architecture gives Oxide one product identity, one GPU owner,
one kernel supply chain, and complete control over fusion, KV layout, Graphs,
and serving evidence. Source transplantation preserves mature external
capabilities without preserving a permanent runtime shell boundary.

The cost is a substantial derived codebase. New upstream API, scheduler,
grammar, adapter, quantization, multimodal, and distributed changes do not
arrive automatically. Oxide must review and port each update through the
source map. This is more maintenance than an untouched dependency but avoids
rewriting proven control-plane behavior and leaves the resulting runtime
unambiguously owned and executed by Oxide.

## Non-goals for the first release

- matching every feature or model supported by any reference engine;
- exposing TileLang or Python as a production runtime API;
- accepting arbitrary eager graphs;
- claiming every TileLang kernel beats every FlashInfer kernel;
- claiming engine leadership from one operator or one concurrency;
- multi-GPU, quantized, MoE, speculative, diffusion, vision, or audio serving.
