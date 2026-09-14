# Checkpoint import and semantic operations

The checkpoint frontend is `orbitkv-executor/src/model/import.rs` with shared
field validation in `model/import/input.rs`. It resolves architecture conventions
once and returns `DecoderConfig`. The normalized configuration contains explicit
block layout, normalization weight convention, embedding scale, activation,
position geometry, state geometry and tensor namespace; it contains no model
name. Graph construction, state binding and kernel selection consume that result.

## Admitted serialization contracts

| `model_type` | Import convention |
| --- | --- |
| `qwen2` | Pre-norm, direct RMSNorm weights, SiLU, `model.*` tensors |
| `mistral` | Pre-norm, direct RMSNorm weights, SiLU, `model.*` tensors |
| `gemma3_text` | Sandwich norm, unit-offset RMSNorm, BF16-rounded embedding scale, tanh GELU and explicit local/full attention layers |
| `qwen3_5_text` | Pre-norm, unit-offset RMSNorm, SiLU, gated full attention and explicitly declared gated-delta layers |
| `qwen3_5` | Validated `qwen3_5_text` envelope; text tensors under `model.language_model.*` |

`model_type` is required. If `architectures` is present it must agree with the
supported causal-LM class (or the Qwen3.5 conditional-generation envelope). Missing,
unknown, contradictory or incorrectly nested metadata is rejected. Checkpoint
paths and release names do not affect the normalized graph. `tie_word_embeddings`
uses the architecture default when omitted, with the envelope supplying the
nested default when explicitly present.

These are frontend contracts, not blanket model support claims. Weight inventory,
state topology, numerical semantics and backend capabilities must still pass
admission. The Qwen3.5 envelope admits text inference only. Unsupported MoE,
position-scaling and softcapping semantics remain rejected. Provider-supported
head sizes do not belong in the checkpoint parser: an importable shape may still
have no executable lowering. See the [capability matrix](capability-matrix.md)
for qualified model paths.

## Operation ownership

`orbitkv_ops::ops::attention` owns logical scaled dot-product attention and explicit
KV-view contracts. Computation and storage facts are separate. Its scale is explicit, positive and finite; the
frontend resolves default scaling rather than passing a zero sentinel. `orbitkv_ops::ops::linear` owns block-scaled FP8 linear
arithmetic, including 128-element activation tiles, the scale floor, rounding,
weight scaling and accumulation/output dtypes. The 128 tile is part of this
numerical contract, not a device tuning preference. A different quantization
policy requires explicit semantics and matching rewrite guards.

OrbitKV compiler core represents an unlowered custom operation distinctly from executable
dialects. Extraction rejects it before profiling if no implementation exists.
CUDA providers supply equivalent implementations through egglog; `orbitkv-ops` has no CUDA
or provider dependency. Capability rules admit FlashInfer algorithms and optional FlashAttention-3
for the supported paged combinations. DeepGEMM variants and optional shared quantization
remain alternatives for FP8 linear. The portable CUDA FP8 oracle is only built
for tests, with no silent deployment fallback.

`AttentionSpec` no longer contains page geometry or state identity; those belong
to `KvView::Paged`. The semantic contract can express unequal QK/V dimensions and
unmasked attention, but current CUDA adapters admit only causal/sliding NHD views
with equal dimensions. Contiguous/ragged or latent views, FlashMLA, Triton/TileLang
adapters, joint layout selection and megakernel generation remain further work.
See [attention providers](attention-providers.md) for exact candidate admission.

Adding a checkpoint format requires an explicit importer and tests for normalized
semantics; a new kernel requires a semantic contract, guarded egglog equivalence,
resource/launch ownership and an independent numerical gate. No Rust LLIR
pattern-rewriting pass is used.

Decoder artifact schema 11 retains explicit attention algorithms and request-count
expressions through provider lowering. Schema 9 and earlier artifacts require fresh search. CUDA module integrity, target and
compiler checks still apply; historical qualification records retain the schema
and source identities they actually measured.
