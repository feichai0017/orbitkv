# Decoder graph residency

A decoder keeps its selected programs and compiled functions independently of
which CUDA Graph executables are currently materialized. The executor exposes
`CompiledDecoder::set_graph_cache_capacity(NonZeroUsize)`; the engine requires
`ModelEngineConfig.graph_cache_capacity`, and `orbitkv-serve` accepts
`--graph-cache-capacity`. The default is one bucket. A deployment can retain
both decode and prefill with `--graph-cache-capacity 2` when its memory budget
allows it. This policy does not change the selected artifact's identity.

Before materializing a bucket, Luminal evicts least recently used buckets until
there is room. It synchronizes before retirement and releases executables before
the captured resources. Reducing the capacity takes effect at the next bucket
materialization. Changing it also invalidates the executor's outer fixed-signature
decode capture. All buckets share the runtime's stable high-water intermediate
arena; growing that arena invalidates captures before its address can change.

## Resource ownership

| Resource | Owner and lifetime | Device-memory charge |
| --- | --- | --- |
| Selected functions and module images | Compiled program | Driver-managed; outside provider payload estimates |
| Intermediate buffers | Runtime arena, stream-ordered across buckets | Maximum planned arena |
| KV and recurrent/convolution state | OrbitKV arenas and request lifecycle | Persistent state allocation |
| FlashInfer integer plan metadata | Each prepared plan, retained by its graph | 8 MiB per distinct prepared allocation |
| FlashInfer float scratch | Shared by dependency-ordered attention execution | 128 MiB once |
| FlashInfer pinned host staging | Serialized planner access, drained before reuse | Host allocation, outside device-memory total |
| Other prepared provider allocations | Owning captured plan or child graph | Provider resource contract |

The FlashInfer capacities are provider workspace budgets, independent of model
identity. Workloads exceeding them return a native planning error. Device
preflight charges each retained bucket/variant's private allocations, deduplicates
the shared float scratch, and reserves another generation of FlashInfer plans
while replacement plans coexist with the old graph islands. Retained-bucket
planning is conservative: it budgets all compiled buckets even if a smaller
residency capacity will evict some. CUDA driver-internal graph allocations are
not fully represented by these payload estimates; the finite bucket limit also
bounds accumulation of driver graph objects.

Previously, all prepared FlashInfer plans pointed into one mutable integer
workspace. Its offsets were retained, but preparing another shape overwrote the
device request/tile/indptr metadata those offsets referenced. Plans now own that
metadata. Host planning also serializes access to pinned staging and waits for
its asynchronous upload before releasing the staging lock, including failures.
The float workspace remains execution scratch: the current graph topology orders
its users, and the decoder dispatches on its owning stream.

## Startup preparation

`CompiledDecoder::prepare_execution()` uses the validated representative maps
stored in the selected artifact. Existing materializations take priority;
unused slots are filled in artifact order up to the configured residency
capacity. Preparation must not evict an existing graph merely to replace it
with a speculative representative. This is an explicit bounded policy, not
automatic workload or memory optimization. With fewer retained slots
than compiled buckets, an arriving request can still evict a prepared bucket.

`model/representative.rs` owns the legal private-page metadata shared by search
and preparation. Startup writes only those descriptors into existing reserved
input allocations, verifies their addresses, and calls
`CudaRuntime::prepare_cuda_graphs()`. It does not execute the model or update KV,
recurrent or convolution state. Real request metadata replaces the descriptors
before execution. Metadata-dependent provider plans may still need refresh;
preparation does not promise immutable captures or zero per-request CPU work.

The runtime's `runtime/residency.rs` owns its residency and preparation interface.
Optional warmup preserves the configured capacity, and preparation applies the
same eviction boundary as normal execution. Strict finite-shape freezing remains
a separate explicit API. Ordinary preparation preserves dynamic execution.

The engine performs this work in `model_engine/startup.rs` before publishing
readiness. `--prepare-execution false` disables it without changing the artifact
or graph capacity. Startup reports include wall time, prepared bucket dimensions,
new full graph builds and retained graph counts. Reported preparation time is
host planning plus synchronization, not kernel execution time. First-request
experiments must record both readiness and request latency and must disable any
benchmark client's generation-based readiness check or warmup requests.

## Validation and remaining work

The provider regression captures single-request decode, ragged decode and causal
sliding-window prefill with distinct page tables. It verifies each against an
independent uniform-attention CPU oracle, alternates replay, retires one graph,
and replays the survivors. An assertion that retained plan metadata is unchanged
fails against the old shared-integer-workspace implementation and passes after
the ownership fix. A separate graph-resource regression charges both current
and replacement plan generations.

The model transition fixture reuses one decoder and state arenas across four
request lifecycles. Each request checks eight teacher-forced full-vocabulary
logit rows and drains KV and fixed-state ownership. It retains buckets for the
first three requests, then reduces capacity to one to exercise eviction. The
[H20 qualification](../results/bucket-resources-20260913/README.md) records the
fixed artifact, source/binary fingerprints, alternating capacity timings and
graph-build counters.

`graph_cache_stats()` counts full graph builds and currently materialized graphs;
a bucket may contain several graphs. Attention with explicit CSR metadata still
replans and updates its captured islands when needed. The work above qualifies
single-device execution on one owning stream. Cross-stream scratch ownership,
long-context and ragged multi-request model transitions, automatic residency
selection from a joint memory budget remain separate qualification work. The
[serving comparison](../results/bucket-serving-20260913/README.md) adds two C1
HTTP workloads with exact paired output and checked shutdown. It retains a
short-output tail regression and leaves the default capacity unchanged.

The [startup preparation comparison](../results/startup-preparation-20260913/README.md)
holds capacity fixed within each off/on pair and covers capacities two and one.
It retains the first generation request in each process and reports startup
separately. All 24 timing processes complete and drain. At capacity two,
short-output first stream interval and P99 TPOT are 138.9→29.9 ms and
37.7→24.5 ms; steady ITL remains about 23.6 ms. At capacity one, preparation
keeps the already loaded graph without an extra complete build; phase switches
still require ordinary eviction. Its P99 TPOT is 42.4→43.9 ms and
throughput 18.34→18.12 token/s, so no default-capacity performance benefit is
established. The earlier artifact-prefix policy's unnecessary
eviction is preserved as a corrected regression under its original source
identity. These are narrow paired observations, not population-tail estimates.
The context-growth trace reaches 128 generated tokens; it does not qualify
long prefill. A rejected over-limit prefill attempt remains recorded, including
the output-length gate that caught the client's empty-response classification.
