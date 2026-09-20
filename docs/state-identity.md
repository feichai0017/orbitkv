# Model-aware state identity and recovery

Correct KV reuse requires more than a matching prefix hash. A hit must name
the same model computation, identify compatible bytes, cover the requested
token range, and contain every component required to resume at that boundary.
OrbitKV must reject an ambiguous candidate and let the engine recompute it.

## Current implementation

The hot storage key is `BlockKey { namespace, hash }`. vLLM derives an
eight-hex-character namespace digest from the model path and selected dtype,
cache, layout, and parallelism settings. SGLang derives a SHA-256 namespace
from configured model/revision/weight version, quantization, page and tensor
layout, and rank settings. Each adapter passes its engine's native block or
radix hash. These namespaces isolate many ordinary layouts; they are **not** a
verified digest of the model weights, tokenizer/processor, implementation, or
all request-specific state. In-place weight replacement at the same identity
cannot be assumed safe. Use immutable model deployments and a new identity for
weight changes until the stronger contract below is implemented.

`orbitkv-state` already defines `StateKey` (content, token span, component,
format), `StateFormat`, `StateBundle`, and `LocalPageRef`. The adapters do not
yet put these types into Query/Publish or the manager's storage and directory
keys. `StateBundle::has_required_components` only checks that component kinds
are present; it does not prove contiguous token coverage, compatible formats,
or a valid recovery boundary. Publish still carries raw engine block IDs, so
registered GPU page generations are not validated at the transfer boundary.
The current SGLang linker therefore rejects state it cannot restore completely.

## Target contract

1. **Name the computation.** Each adapter supplies a versioned model identity
   from immutable weight revision/content, tokenizer and multimodal processor,
   active adapters/LoRA or other request-specific parameters, and the cache
   implementation variant. A deployment operator may supply a stable digest
   when hashing large weight files at startup is impractical. The adapter must
   fail closed if a required identity cannot be established.
2. **Name the bytes.** Serialize a versioned, domain-separated `StateKey`
   containing the computation identity, framework prefix/content digest,
   `[start, end)` token range, component and cache group, and the full
   `StateFormat`: dtype/quantization, block size, layout and strides, head
   shape, parallel rank/shape, and implementation compatibility. Use the same
   key in Query, Publish, DRAM/SSD indexes, and remote directory records.
   Re-keying the cache before 1.0 can invalidate old entries rather than
   carrying a compatibility path. Equal model names alone never authorize
   cross-framework byte reuse.
3. **Prove recovery.** A validator checks that a `StateBundle` covers every
   requested token without gaps or overlap, all components agree on model and
   format, and the model's recovery rule requires exactly those components at
   that boundary. Dense attention KV, MLA, sliding-window attention, recurrent
   checkpoints, DSA, draft, and auxiliary state need explicit capability
   definitions. A partial hit can only shorten prefill to a boundary the
   validator proves restorable; otherwise the engine recomputes.
4. **Fence page reuse.** The adapter sends a registration/session epoch and a
   generation-qualified page reference for every GPU source and destination.
   The manager validates these before copy and holds the relevant leases until
   CUDA completion. A stale page ID must never address a newly reused GPU slot.
5. **Expose capability, not guesses.** vLLM maps cache groups and HMA boundary
   state to the shared bundle contract. SGLang maps RadixAttention page and
   lifecycle events to that contract without maintaining a second radix tree.
   Unsupported components remain rejected rather than treated as attention KV.

The first integration should use the existing vLLM and SGLang native hash
algorithms within a versioned domain, then test whether a common token/content
digest can safely support cross-framework sharing. A matching text prompt is
insufficient evidence: byte-level compatibility needs an explicit format and
implementation check or a conversion path.

## Order and acceptance gates

| Order | Work | Gate |
| --- | --- | --- |
| 1 | Record cold/warm/partial/restart correctness and TTFT, TPOT, throughput, P50/P95 query/save/restore latency, CPU/GPU use, pinned-memory footprint, and save amplification for both release targets | External loads are proved by counters after engine restart, and a cold run supplies output parity; measurements include native-engine and no-cache baselines |
| 2 | Add a versioned model fingerprint and canonical key at both adapters, then carry it through Query, Publish, storage, and remote lookup | Weight revision, tokenizer/processor, adapter, dtype, layout, rank, and token-span changes never produce a false hit; identical immutable deployments can still hit |
| 3 | Validate bundle coverage and model-specific recovery rules; move vLLM hybrid reconciliation to shared logic | Full-attention, MLA, supported hybrid boundaries, partial prefixes, and unsupported components each have explicit correctness tests |
| 4 | Add generation-qualified GPU references and cancellation/restart fences | Recycled page IDs, delayed transfers, manager restart, and cancellation cannot corrupt or falsely restore state |
| 5 | Tune admission, batching, prefetch, and copy backend from measured traces | Tail latency and throughput improve or stay within an agreed budget versus the baseline, with bounded memory and save-worker stalls |
| 6 | Make the distributed directory recoverable, then add KV-aware routing | Remote candidates are revalidated by their owner; router decisions use complete, versioned state evidence |

The design is a work plan, not a current guarantee. Today, single-node GPU
recovery works for the validated adapters and layouts, but the semantic key,
complete bundle proof, page-generation enforcement, and revised performance
qualification remain open. See [single-node deployment](single-node.md),
[architecture](architecture.md), and [the implementation checklist](../TODO.md).
