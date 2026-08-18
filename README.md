# OrbitKV

**Compile attention retention semantics into KV block plans.**

OrbitKV is a Rust attention-state compiler and KV block manager. It generates
address programs, memory policies, immutable views, and proof-carrying
reclamation. SGLang is the first external validator.

## Implemented

- Retention IR for Full, Sliding, Sink+Sliding, and Same-Chunk attention.
- `AppendOnly`, `Periodic`, `Pinned`, and `ResettableArena` address programs.
- Per-head lifetime stripes and Retention Amplification analysis.
- HF applicability report and executable StatePlan.
- SGLang physical-plan optimizer and runtime contract.
- Generation-checked ownership and two-phase reclamation.
- Continuation Capsule schema and Holt 0.9.2 persistent prefix catalog.
- CUDA VMM physical-slot primitive.

Unsupported or unprovable semantics fail closed.

Capsule metadata is conditionally published into one Holt tree after its
immutable, content-addressed KV payload is durable. Payloads are files rather
than Holt values, so they are not constrained by Holt's metadata-size limit.

## Key results

| Result | Value | Boundary |
| --- | ---: | --- |
| GPT-OSS Full capacity | +25.81% | same 1.979 GiB KV budget |
| GPT-OSS Owner vs Stock128 | -20.30% | 8×6K prompt workload |
| Mistral KV slots | -61.696% | page16, 4×12K, same output digest |
| Mistral median runtime | 0.9855× | decode Graph, same output digest |
| Physical-plan predictions | 4/4 | intervals 16/32/64/128 |
| Multi-scale head KV | -42.105% | exact geometry |

All GPT-OSS output-token digests matched. The capacity gain is reproducible by
manually setting SGLang interval 32; OrbitKV contributes automatic plan
synthesis and auditable ownership.

## Quick start

```bash
cargo test --all-targets

cargo run -- analyze-retention \
  examples/chunked_local_retention.json

cargo run -- analyze-lifetime-normalization \
  examples/multi_scale_head_windows.json

cargo run -- compile-hf-config /path/to/config.json \
  --page-tokens 16 --kv-dtype-bytes 2

cargo run -- compile-hf-state-plan /path/to/config.json \
  --page-tokens 1 --kv-dtype-bytes 2 --boundary 32768 \
  --max-running-requests 4 --chunked-prefill-tokens 2048 \
  --eviction-interval 128 --decode-headroom-tokens 32 \
  --cuda-graph-mode disabled
```

Physical-plan compilation:

```bash
cargo run -- compile-hf-physical-plan /path/to/config.json \
  --page-tokens 16 --kv-dtype-bytes 2 \
  --available-kv-bytes 2123759616 \
  --max-running-requests 128 --attention-dp-size 1 \
  --chunked-prefill-tokens 2048 \
  --workload-requests 8 --prompt-tokens 6000 --decode-tokens 32 \
  --candidate-intervals 16,32,64,128 \
  --max-reclamation-calls 4 --min-admitted-requests 8 \
  --objective capacity
```

## Boundaries

The qualified SGLang paths disable radix cache, overlap, speculation, and
disaggregation. Page16 Paged Periodic also qualifies decode CUDA Graph replay
with eager prefill, up to four requests. Per-head, Sink+Sliding, and Same-Chunk layouts remain
compiler/reference-manager results. VMM does not yet back SGLang KV tensors.
Capsule persistence is engine-neutral; SGLang export and hydration are not yet
qualified.

See:

- `docs/h20-gpt-oss-20b-real-validation-20260817.md`
- `docs/sglang-e2e.md`
- `results/README.md`
