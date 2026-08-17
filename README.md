# OrbitKV

**Compile attention semantics into lifetime-safe KV blocks.**

OrbitKV is an independent Rust attention-state compiler and KV block manager.
It derives block lifetimes from retention semantics, synthesizes finite address
programs, and authorizes physical reuse only after both semantic death and GPU
execution completion.

OrbitKV is not an inference engine. SGLang is the first external end-to-end
validator and performance baseline.

## Current scope

The first executable slice supports:

- full-attention KV classes;
- sliding-window KV classes;
- block-atomic lifetime lifting;
- the optimal equal-size slot count for a single sliding class;
- machine-readable temporal address and retirement programs;
- continuation block sets at a pre-query boundary;
- a multi-request owning block manager with generation-checked handles;
- immutable KV views, semantic/execution frontiers, and two-phase retirement
  certificates;
- an owning SGLang SWA chunk-cache adapter backed by a typed in-process Rust C ABI;
- a cost-gated NVIDIA CUDA VMM physical-slot backend.

The compiler, lifecycle runtime, plan verifier, trace analyzer, and CLI are
implemented in Rust. Python is restricted to the thin SGLang plugin under
`integrations/sglang/`.

The current repository does **not** yet claim production ownership of every
SGLang allocation path, prefix-cache replacement, arbitrary Python-mask
compilation, or VMM-backed SGLang tensor storage. Those are explicit
implementation gates, not implied features.

## Quick start

```bash
cargo test --all-targets
cargo run -- check-sglang /path/to/sglang
cargo run -- compile examples/full_swa.json --boundary 32768
cargo run -- emit-layout examples/full_swa.json
cargo run -- emit-sglang-policy examples/gpt_oss_hybrid_tiny.json \
  --eviction-interval 32
cargo run -- serve-sglang-owner examples/full_swa.json
cargo build --manifest-path crates/orbitkv-ffi/Cargo.toml
cargo run --manifest-path crates/orbitkv-cuda/Cargo.toml \
  --bin orbitkv-cuda -- vmm-smoke 1048576
```

## SGLang validation

OrbitKV uses SGLang's general plugin interface. Shadow mode leaves allocator
results unchanged; policy mode lowers a Rust-validated retention plan into
SGLang's page-granular SWA reclamation path.

Validated source target:

```text
sglang revision 095ec6c997bfdd25d3864cb0ce77a6562a934b96
```

Install `integrations/sglang` into the same Python environment as SGLang, then
launch a hybrid Full+SWA model with:

```bash
python3 -m pip install ./integrations/sglang
export SGLANG_PLUGINS=orbitkv_shadow
export ORBITKV_TRACE_PATH=/tmp/orbitkv-sglang.jsonl
export ORBITKV_SGLANG_REVISION=095ec6c997bfdd25d3864cb0ce77a6562a934b96
```

To enable the owning FFI path:

```bash
cargo build --release --bin orbitkv
cargo build --release --manifest-path crates/orbitkv-ffi/Cargo.toml

export ORBITKV_BIN="$PWD/target/release/orbitkv"
export ORBITKV_SGLANG_POLICY="$PWD/examples/gpt_oss_hybrid_62l.json"
export ORBITKV_SGLANG_OWNING=1
export ORBITKV_OWNER_TRANSPORT=ffi
export ORBITKV_OWNER_LIB="$PWD/crates/orbitkv-ffi/target/release/liborbitkv_ffi.so"
```

`ORBITKV_OWNER_TRANSPORT=sidecar` remains available as a protocol regression
baseline, not the default production transport.

For the first experiment, disable radix cache and speculative decoding. This
isolates per-request Full growth and bounded SWA residency before prefix locks,
copy-on-write, and HiCache are added.

After the workload:

```bash
cargo run -- analyze-sglang \
  examples/full_swa.json \
  /tmp/orbitkv-sglang.jsonl \
  --max-active-requests 8
```

See `docs/sglang-e2e.md` for the staged experiment and replacement gates.

## Verified H20 result

The first cost-aware SGLang policy was validated on an NVIDIA H20 using a
dummy-weight hybrid system fixture with real alternating Full/SWA execution.
Under a fixed KV budget, OrbitKV increased Full token capacity by 47.14% and
reduced the median makespan of an eight-request long-context workload by
28.25%, with identical output-token digests.

These are systems results, not model-quality results. See
`docs/h20-sglang-validation-20260817.md` and `results/README.md` for the
workload, constraints, raw matrices, and claim boundaries.

## Owning manager

OrbitKV's Rust control plane owns:

- generation-checked logical-to-physical block handles;
- immutable KV views for submitted GPU work;
- semantic and execution frontiers;
- proof-carrying reclamation certificates;
- physical-slot reuse authorization only after the backend commits reclamation.

The first SGLang owning adapter is intentionally strict: it supports the
validated SWA chunk-cache, non-overlap, non-speculative path. Rust emits an
exact page-aligned retirement certificate; SGLang frees the physical pages;
Rust advances its committed frontier only after the complete free group
succeeds. Injected physical-free failures do not commit the certificate.

Three fresh-process H20 Policy/Owner pairs on the same 62-layer admission
workload produced a median Owner/Policy ratio of `1.0062x`, or `+0.62%`.
Capacity and output digests were identical. This is not evidence for radix
cache, overlap scheduling, or speculative decoding.

The current adapter calls Rust through `orbitkv-ffi` instead of a JSONL
sidecar. A release-build transport benchmark reduced median `plan + commit`
latency from `31.03 µs` to `7.29 µs`, a `4.26x` control-plane speedup. Six
fresh-process, alternating-order H20 pairs measured a median FFI/sidecar
end-to-end ratio of `1.0030x` and mean ratio of `1.0008x`, with identical
capacity and output digests. The microbenchmark speedup is not a serving
speedup claim.

## NVIDIA physical backend

`orbitkv-cuda` isolates CUDA Driver API `unsafe` code from the safe Rust
manager. Its VMM slot reserves a stable virtual range, creates and maps fresh
physical backing, grants device access, and explicitly unmaps/releases backing
before freeing the address reservation.

On the recorded H20:

- CUDA VMM is supported;
- minimum and recommended allocation granularity are both 2 MiB;
- 64 fresh physical backings were remapped at one unchanged virtual address;
- every data pattern was verified;
- GPU memory returned to the pre-test value.

The pinned SGLang validator already implements post-capture VMM backing that
reserves stable addresses and monotonically commits the final KV span.
OrbitKV's additional contract is generation-aware reclamation: a compiled
logical cell/cycle is bound to a physical generation, real CUDA Event
completion gates retirement, VMM unmap returns a generation-matched receipt,
and the core manager commits before the stable address can host the next
generation.

A 64-cycle H20 closed-loop test passed 64 CUDA Event completions, 64 data
pattern checks, 63 stale-generation rejections, and 65 receipt commits, with
zero final manager residency and no GPU-memory delta.

The optimizer therefore rejects VMM for small regions whose 2 MiB rounding
amplification is too expensive. VMM is selected only when stable addresses are
required and the rounded physical cost fits the configured budget. The next
gate is to use these slots as real SGLang KV tensor storage and qualify CUDA
Graph replay.
