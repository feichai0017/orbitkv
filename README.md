# OrbitKV

OrbitKV is an independent Rust KV block manager project. It compiles attention
retention semantics into block-capacity, address, view, and reclamation plans.

OrbitKV is not an inference engine. SGLang is the first external end-to-end
validator and performance baseline. The former Oxide Infer workspace is not a
runtime dependency and is not the implementation substrate for this repository.

This repository replaces the former Oxide Infer project at the existing GitHub
URL. The final pre-OrbitKV main branch is preserved as
`archive/oxide-infer-20260816`.

## Current scope

The first executable slice supports:

- full-attention KV classes;
- sliding-window KV classes;
- block-atomic lifetime lifting;
- the optimal equal-size slot count for a single sliding class;
- continuation block sets at a pre-query boundary;
- a reference block lifecycle simulator with asynchronous GPU pins;
- a non-owning SGLang shadow plugin that records allocator transitions.

The compiler, lifecycle runtime, plan verifier, trace analyzer, and CLI are
implemented in Rust. Python is restricted to the thin SGLang plugin under
`integrations/sglang/`.

The current repository does **not** yet claim SGLang GPU performance,
production ownership of SGLang blocks, prefix-cache replacement, or arbitrary
mask compilation.

## Quick start

```bash
cargo test --all-targets
cargo run -- check-sglang /path/to/sglang
cargo run -- compile examples/full_swa.json --boundary 32768
cargo run -- emit-sglang-policy examples/gpt_oss_hybrid_tiny.json \
  --eviction-interval 32
```

## SGLang shadow validation

OrbitKV uses SGLang's general plugin interface and leaves SGLang's allocator
results unchanged.

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

## H20 result

The first cost-aware SGLang policy was validated on an NVIDIA H20 using a
dummy-weight hybrid system fixture with real alternating Full/SWA execution.
Under a fixed KV budget, OrbitKV increased Full token capacity by 47.14% and
reduced the median makespan of an eight-request long-context workload by
28.25%, with identical output-token digests.

These are systems results, not model-quality results. See
`docs/h20-sglang-validation-20260817.md` and `results/README.md` for the
workload, constraints, raw matrices, and claim boundaries.
