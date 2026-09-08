# State Lifetime and Reclamation

OrbitKV treats KV memory as compiled temporal state rather than an append-only
tensor. The compiler derives where state may live and when it may die from the
attention visibility relation.

## Two-frontier rule

A physical generation is reusable only when both conditions hold:

1. The Semantic Frontier has passed every logical token stored in it.
2. The Execution Frontier has passed every submitted device operation that may
   access it.

Detaching a request mapping proves neither condition by itself. Device metadata
cleanup, completion, exact retirement receipts, and acknowledgement are part of
the reuse protocol.

## Compiled lifetimes

### Full

Full attention keeps all reachable tokens. Prefix sharing and immutable partial
tails can reduce duplicated prefill state, while COW preserves isolation. The
active manager does not compact or relocate live tokens inside pages.

### Sliding

Sliding attention compiles to periodic physical slots plus a retirement rule.
When the window advances, old logical blocks become semantically dead. Their
physical generations still wait for executor completion and acknowledgement.

### Full + Sliding

Mixed attention uses separate classes and frontiers. Sliding pages may retire
while corresponding Full pages remain live. Shared Prefix and COW cover both
classes consistently, but reclamation evidence remains class-specific.

### Chunked

Exact chunk-local visibility compiles to a resettable arena. Logical token
positions remain absolute while physical cells are reclaimed at proved epoch
boundaries after completion.

### Latent and fixed state

Latent KV carries component geometry for latent and positional payloads.
Recurrent and convolution state use generation-checked checkpoint slots. These
states cannot inherit ordinary K/V copy or reclamation assumptions; their
executor transactions remain a separate qualification task.

## Measuring a benefit

The memory numerator must include resident bytes, padding, shared state, and
temporary copy headroom. The semantic denominator is state reachable by future
queries under the admitted attention policy. Their ratio is retention
amplification.

Reclaimed pages do not necessarily reduce a statically reserved tensor, and a
smaller live set does not necessarily improve throughput. Claims therefore need
matched output correctness, resident and reserved memory, admission capacity,
copy overhead, latency, throughput, and long-running pressure measurements.
