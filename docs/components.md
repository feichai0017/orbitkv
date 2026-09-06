# Components and external projects

OrbitKV is one Rust inference product assembled from three owned layers and a
small number of explicit external boundaries. An imported component never gains
KV lifecycle authority.

## Owned components

| Component | Responsibility | Does not own |
| --- | --- | --- |
| `orbitkv` core | Compile attention-state semantics; own request/snapshot identity, page generations, Prefix/COW, token disposition, retirement, publication, and reuse | Kernels, HTTP, network byte movement |
| `orbitkv-executor` | Lower manager plans, bind persistent tensor arenas, execute Luminal graphs, move local or external bytes, and produce completion evidence | Page allocation, semantic liveness, final publication |
| `orbitkv-server` | Tokenization/protocol adaptation, admission, batching, cancellation, backpressure, and output streaming | Physical page names, tensor addresses, retirement decisions |
| Engine coordinator | Join one server batch to one `RuntimeSession` transaction, one Luminal execution, and optional tier operations | A second cache index or allocator |

The engine coordinator is the largest missing production component. The public
`Engine` trait exists, but the scheduler, continuous batching loop, cancellation
drain, and model-backed HTTP path are not yet one released executable.

## External projects

| Project | Relationship | Reused or planned surface | Excluded surface |
| --- | --- | --- | --- |
| Luminal fork | Embedded compiler/executor | Graph IR, search, CUDA kernels, persistent inputs, bucket dispatch, child CUDA graphs | Luminal page allocation as KV authority |
| vLLM | Optional frontend and benchmark client | Rust OpenAI/tokenizer/chat/SSE crates; `vllm bench serve` as the common load generator | vLLM scheduler or KV block manager in the OrbitKV process |
| PegaInfer | Design and measurement reference only | Small Rust server boundary, hybrid full/linear-attention operator structure, matched vLLM-client workflow | Source copying, model-name dispatch, contiguous model-owned KV cache |
| Dynamo | Distributed-system reference and optional outer control plane | KV-aware routing ideas, event schemas, telemetry, service discovery where justified | `kvbm-logical`, `KvBlockManager`, lifecycle pins, or any second page manager |
| Mooncake | Planned external storage transport | Registered memory, Store objects, placement/lease observations, RDMA/TCP transfer completion | Local page allocation, generation, retirement, or publication |
| NIXL | Planned external data plane | Registered regions, asynchronous transfer descriptors, topology-aware movement | Dynamo KVBM ownership semantics |

The PegaInfer source reviewed at commit `14338f65` does not depend on Dynamo. Its
Rust crate directly uses Axum, Tokio, Hugging Face tokenizers, safetensors,
cudarc, and its own CUDA kernels. Its vLLM relationship is primarily the use of
`vllm bench serve` as a common client. OrbitKV should therefore learn hybrid
execution and benchmark discipline from PegaInfer, while studying Dynamo
directly for distributed routing and NIXL integration.

The Dynamo source reviewed at commit `7ee72a3a` exposes `dynamo-runtime`,
`dynamo-llm`, `kvbm-logical`, and physical/NIXL layers. `dynamo-llm` enables its
block manager by default, while `kvbm-logical` defines its own manager IDs,
lifecycle pins, pools, registry, eviction, and event pipeline. Those middle
layers conflict with OrbitKV authority and are intentionally not dependencies.

## Dependency rule

Every future integration must fit one of two directions around the sole
authority:

```text
external routing/events/telemetry
                |
                v
        OrbitKV RuntimeSession
                |
       +--------+---------+
       |                  |
       v                  v
Luminal execution   ExternalKvTransport
                          |
                    Mooncake / NIXL
```

No adapter may mint a `PageLease`, advance semantic death, infer completion
from enqueue, or recycle a generation.
