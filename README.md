# OrbitKV

OrbitKV is an attention-state compiler and transactional ownership runtime. It
compiles retention semantics into checked physical plans, then owns logical
state identity, immutable request snapshots, page generations, sharing, and
reclamation. Inference engines continue to own scheduling, kernels, tensor
allocation, and model execution.

The live interface is **ABI8** and is still evolving. Unsupported semantics
and adapter contracts fail closed.

## From retention semantics to ownership

```text
attention / state retention semantics
                |
                v
        checked retention IR
                |
                v
 lifetime classes + address programs
                |
                v
 immutable snapshots + physical intents
                |
                v
 engine-neutral adapter effects
                |
                v
 completion-gated publication and reuse
```

Ring layouts, append-only regions, pinned regions, and relocation are physical
lowerings. The semantic lifetime and ownership program remains the source of
truth. Token KV, latent KV, recurrent state, and convolution state retain
distinct contracts instead of being forced into one block shape.

## Ownership boundary

| Layer | Owns |
| --- | --- |
| OrbitKV | plans, logical identities, generations, snapshots, references, transactions, and reclamation decisions |
| Adapter | checked tensor effects and exact completion evidence |
| Inference engine | allocation, scheduling, kernels, model execution, and request protocol |

Requests point to generation-checked immutable snapshot heads. Physical pages
carry request and Prefix references, reader pins, writer state, and generation.
Mirrors in an adapter are checked effects, never ownership authorities.

## Token-level reclamation

OrbitKV separates two frontiers:

- the **Semantic Frontier** proves which logical tokens are no longer reachable;
- the **Execution Frontier** proves that prior device work has completed.

Relocation can pack retained tokens out of partially dead pages. A source page
is reusable only after both frontiers pass and the backend acknowledges the
exact reclamation receipt. This is byte-exact state movement and ownership
reclamation, not numerical compression.

## Sharing and packed copy-on-write

Prefix sharing and request forks add references to immutable roots. Extending a
shared or pinned partial tail produces an exact copy-on-write intent; the new
root is published only after the adapter proves the copy completed before new
writes. Packed request fork and shared partial-tail COW are host-tested across
the Rust core, ABI8 wire, and Python FFI. Packed Prefix operations remain
unsupported and fail closed.

## Engine-neutral adapter

`orbitkv-runtime` defines a typed data-plane SPI for append, copy, clear,
completion, and mirror-cleanup effects. `orbitkv-reference` provides an
external CPU/CUDA tensor-arena contract oracle. Engine-specific integrations
remain adapters to the same ownership protocol; capability does not transfer
automatically between adapters.
With the `structured-data-plane` package extra and
`ORBITKV_STRUCTURED_DATA_PLANE=1`, the scoped SGLang adapter uses the optional
two-phase external-write path for its eager BF16/NHD `token_kv` Full, pure-SWA,
and Full+SWA subset with relocation disabled. This is an opt-in lifecycle
integration, not a performance or full-engine replacement claim.

See [Engine Adapter SPI](docs/engine-adapter-spi.md) and
[Standalone KV Manager Architecture](docs/standalone-kv-manager-architecture.md).

## Executable runtime manifest

The preferred SGLang input is `orbitkv.runtime-manifest` v1: one versioned
artifact containing the checked attention-state contract, an optional
token-manager input plus compiled layout, exact capability requirements, and a
stable content fingerprint. Generate it from a heterogeneous state plan or a
supported Hugging Face config:

```bash
cargo run --locked --bin orbitkv -- \
  compile-runtime-manifest \
  examples/hybrid-attention-state-plan.json \
  > runtime-manifest.json

cargo run --locked --bin orbitkv -- \
  compile-hf-runtime-manifest config.json \
  --page-tokens 16 --kv-dtype-bytes 2 \
  > runtime-manifest.json

export ORBITKV_RUNTIME_MANIFEST=/absolute/path/runtime-manifest.json
export ORBITKV_LIBRARY=/absolute/path/liborbitkv_ffi.so
```

`ORBITKV_RUNTIME_MANIFEST` cannot be combined with legacy `ORBITKV_PLAN` or
`ORBITKV_STATE_PLAN`. The legacy plan path remains supported unchanged.
`ORBITKV_TOKEN_RECLAMATION` is an orthogonal runtime policy and remains valid
with either input path; it is not included in the manifest fingerprint. A
fixed-state-only manifest is valid compiler output, but the current SGLang
adapter rejects it because that adapter still requires a token-manager plan.

This is the P1 compiled contract and admission artifact. It is not the P3
token/fixed-state transaction graph: the two native handles still have no
cross-handle atomic commit or rollback. It establishes no performance,
capacity, memory-saving, or broader engine-replacement claim.

## Evidence boundary

- The ABI8 compiler, ownership core, typed wire, and Python runtime have broad
  host correctness, lifecycle, stale-identity, fault, and packaging gates.
- Separate sealed engine records cover scoped Prefix correctness and scoped
  request-private token-relocation correctness/lifecycle. Their scopes do not
  transfer, and performance remains unqualified.
- Packed COW is host-qualified but is outside those sealed engine scopes.
- Fixed-state, asynchronous overlap, broader attention families, graphs, and
  distributed execution retain narrower or pending qualification boundaries.
- Historical records remain append-only evidence for their original source and
  ABI only; they never qualify the live ABI8 tree automatically.

The [Capability Matrix](docs/capability-matrix.md) is normative. Detailed
manifests, methods, exclusions, and historical records live in the
[Results Index](results/README.md); public summaries intentionally do not copy
their model, device, source, or timing tables. No performance, capacity, memory
saving, production-readiness, or general engine-replacement claim is made.

## Qualification and evidence tools

Use the capability-oriented entry points for new qualification and evidence
workflows:

```bash
python integrations/sglang/qualify_token_relocation.py --help
python tools/verify_token_relocation_evidence.py <evidence-root>
python tools/verify_token_relocation_seal.py <archive>
python tools/verify_fixed_state_pair_evidence.py <archive>
```

The qualifier and verifiers are thin facades over the currently selected,
manifest-bound implementations. Their generic names improve discovery but do
not add profiles, reinterpret evidence, or widen any correctness, lifecycle,
hardware, performance, or production claim. Legacy hardware- or model-named
paths remain compatibility and evidence entry points; immutable copies under
`results/` preserve existing source closures and archive reproducibility.

## Build and verify

```bash
cargo fmt --all -- --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings

cargo test --locked --manifest-path crates/orbitkv-ffi/Cargo.toml --all-targets
python tools/verify_active_source.py
python tools/verify_capability_matrix.py
python tools/verify_manifests.py
```

The active-source gate verifies ABI8 markers and excludes append-only evidence
under `results/`.
