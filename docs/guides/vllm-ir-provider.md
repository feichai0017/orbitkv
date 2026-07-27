# vLLM IR Provider

Loom Kernels can replace fused residual Add+RMSNorm implementations in vLLM
0.24 and 0.25 through the vLLM IR provider registry. The integration is
inference-only, mutates both tensors in place, launches on PyTorch's current
CUDA stream, and survives vLLM compilation and CUDA Graph capture.

The same package also provides an opt-in out-of-tree replacement for vLLM's
standard `SiluAndMul` layer. It is not enabled merely by installing the plugin:
the current H20 result establishes exact compatibility and graph parity, not a
performance win.

A second opt-in replaces vLLM's fused SiLU-and-Mul plus dynamic symmetric
per-block FP8 implementations for group sizes 64 and 128. This boundary is
bitwise compatible with vLLM's fused operator and has an operator-level H20
advantage. It has also completed a pinned Qwen2.5 online-FP8 engine gate with
direct compiler-match and launch evidence; that small-model end-to-end result
is at parity rather than a demonstrated speedup.

A third opt-in uses the existing RoPE+KV compiler fusion pass in vLLM 0.24 and
0.25 with Loom's CUDA implementation for FlashAttention and FlashInfer native
or static FP8 E4M3 caches. It preserves packed-QKV token/head strides, NHD or
HND cache strides, negative slots, the shorter slot mapping used with padded
engine inputs, and vLLM's `[1]` or `[num_kv_heads]` K/V scales. E5M2, dynamic
per-token-head scales, INT8, NVFP4, and model-specific cache formats are
deliberately declined.

A fourth explicit registration replaces only the pure-greedy `logprobs=0`
sampler tail in vLLM 0.24 and 0.25. It fuses argmax, sampled-token raw logprob,
and tie-aware rank without materializing a full-vocabulary F32 logprob tensor.
Unlike the parity-only integrations above, pinned Qwen2.5-0.5B H20 runs show
an order-stable end-to-end latency and TPOT improvement for this narrow
request contract.

A fifth registration extends the same idea to general sampling without taking
over policy: vLLM still applies masks, penalties, temperature, top-k/top-p,
and RNG, while Loom computes only the chosen token's raw logprob and rank from
the preserved BF16/FP16 logits. Pinned top-k/top-p H20 runs show exact tokens
and ranks plus an order-stable end-to-end improvement.

A sixth opt-in replaces only a measured short-context slice of vLLM's
FlashAttention decode method. Loom reads vLLM's interleaved native KV cache
directly and routes every unsupported shape or semantic feature to the
original FA3 method.

A seventh explicit registration replaces vLLM's deterministic all-greedy
speculative rejection kernel. It consumes the engine's flattened ragged draft
metadata, verifies target argmax IDs, and compacts the accepted prefix plus
mismatch or bonus token. Stochastic rejection and every model, attention,
GEMM, RNG, scheduler, and KV-cache policy remain engine-owned.

An eighth registration replaces vLLM's vocabulary-sized repetition,
frequency, and presence temporaries with a sparse history hash.

A ninth registration handles raw top-k logprob lists without changing sampling
or observable tie order. vLLM keeps `torch.topk`, processors, top-k/top-p, and
RNG; Loom supplies the shared raw normalization and selected-token rank.

A tenth explicit registration replaces top-k-only filtering for one through
seven rows, where vLLM otherwise sorts the full vocabulary. Loom preserves
vLLM's threshold ties, keeps every `top_k` value on device, and leaves top-p,
softmax, random sampling, per-request generators, processed-logit return
semantics, and the eight-row Qrita Triton boundary unchanged.

An eleventh explicit registration fuses top-p-only filtering with F32
renormalization for measured large-vocabulary decode shapes. vLLM retains its
per-request generators, random selection, processed-logit modes, joint
top-k/top-p path, and every unqualified shape.

A twelfth explicit registration fuses the preprocessing before selection for
mixed greedy/random batches. It applies the engine's allowed-token mask,
unique sparse logit bias, min-token or active bad-word suppression, and
per-row temperature in one F32 pass. Penalties, thinking-budget state, custom
processors, overlapping min-token/bad-word policy, and all-greedy or all-random
batches conservatively stay on vLLM.

The registered contract is:

```text
residual = input + residual
input = RMSNorm(residual, weight, epsilon)
```

## Compatibility

The supported package interval is `vllm>=0.24,<0.26`. The bridge-ABI-7 native
wheel passes 286 H20 tests with each official vLLM minor and is the current
qualified artifact. It includes fused logits preprocessing, exact top-k
filtering, and fused top-p renormalization, and is not published.
Existing model-level performance artifacts were captured on 0.24.0 and are
not automatically performance claims for 0.25.1.
See the
[compatibility matrix](../compatibility.md) and
[native-wheel gate](../results/h20-native-wheel-clean-install-abi7-20260727.json).

## Build and install

Build the matrix artifact from a clean Linux x86_64 checkout with a
CUDA-enabled PyTorch:

```bash
python3 -m venv .venv-wheel
.venv-wheel/bin/pip install \
  'setuptools>=80,<82' 'wheel>=0.45' build 'torch>=2.10,<2.12'

CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  .venv-wheel/bin/python python/build_wheel.py \
  --cuda-home /usr/local/cuda-13.1 \
  --archs 90 \
  --wheel-dir dist

python3 -m venv .venv-vllm
.venv-vllm/bin/pip install \
  'dist/loom_kernels-1.0.0a1-7cu131torch210sm90-py3-none-linux_x86_64.whl[vllm,test]' \
  'vllm>=0.24,<0.26'
```

The wheel contains the single native backend, `libloom_cuda_bridge.so`, and
the boxed LibTorch Stable ABI dispatcher, `libloom_kernels_torch.so`, targeting
PyTorch 2.10. Installed packages validate the matrix manifest and both hashes,
then load only that package-local pair. Every admitted operator passes physical
buffer spans, strides, and PyTorch's current stream through the Rust bridge
into safe borrowed dispatch. There is no Python/ctypes fallback, ATen
dispatcher twin, unchecked twin, direct C++-to-CUDA route, or external
dispatcher override.

This command builds the current ABI7 artifact. Its clean-install qualification
is recorded for the exact source revision and wheel hash; it is not published
to a package index. Editable
source development remains documented in the
[Python README](../../python/README.md#source-development), but it cannot
produce a source-only wheel.

## Direct PyTorch Use

```python
from loom_kernels.torch_ops import add_rms_norm_

output, updated_residual = add_rms_norm_(
    input_tensor,
    residual,
    weight,
    1.0e-5,
)

from loom_kernels import (
    greedy_sample_logprobs,
    greedy_speculative_verify,
    paged_decode_attention_out,
    rope_paged_kv_write_,
    silu_and_mul,
    silu_and_mul_dynamic_fp8,
    silu_and_mul_dynamic_fp8_out,
    silu_and_mul_out,
    selected_token_logprobs,
    top_k_filter_,
    top_p_renorm_,
    topk_sampled_logprobs,
)

output = silu_and_mul(gate_and_up)
silu_and_mul_out(gate_and_up, reusable_output)

fp8_output, block_scales = silu_and_mul_dynamic_fp8(
    gate_and_up_bf16,
    group_size=128,
)

token_ids, sampled_logprobs, sampled_ranks = greedy_sample_logprobs(logits)
sampled_logprobs, sampled_ranks = selected_token_logprobs(logits, token_ids_i64)
topk_ids, topk_logprobs, sampled_ranks = topk_sampled_logprobs(
    logits, token_ids_i64, 20
)
top_k_filter_(sampling_logits_f32, per_row_top_k_i32)
sampling_probabilities_f32 = top_p_renorm_(
    sampling_logits_f32, per_row_top_p_f32
)
verified_ids, accepted_lengths, emitted_lengths = greedy_speculative_verify(
    flattened_draft_ids_i32,
    flattened_target_argmax_ids_i64,
    bonus_ids_i32,
    inclusive_cumulative_draft_lengths_i32,
    max_draft_tokens,
)
silu_and_mul_dynamic_fp8_out(
    gate_and_up_bf16,
    reusable_fp8_output,
    reusable_block_scales,
    group_size=128,
)

rope_paged_kv_write_(
    query,
    key,
    value,
    positions,
    cos_sin_cache,
    packed_kv_cache,
    key_scales_f32,
    value_scales_f32,
    slot_mapping,
    is_neox=True,
)

paged_decode_attention_out(
    decode_query,
    interleaved_kv_cache[:, 0],
    interleaved_kv_cache[:, 1],
    block_table_i32,
    sequence_lengths_i32,
    reusable_attention_output,
    max_sequence_length=32,
)
```

Add+RMSNorm and standalone SiLU-and-Mul tensors must be contiguous CUDA tensors
using their documented matching F32, FP16, or BF16 dtype. The dynamic-block-FP8
path accepts FP16/BF16 input, group size 64 or 128, and a width divisible by the
group. `weight` must be one-dimensional and match the final normalization
dimension. The RoPE+KV path accepts native caches or uint8 FP8 E4M3 caches,
with contiguous CUDA F32 K/V scales shaped `[1]` or `[num_kv_heads]`. Checked
public operators reject gradients and aliasing.

## vLLM Use

Installing the Python package exposes a `vllm.general_plugins` entry point.
Select Loom for only the supported IR operation:

```python
from vllm import LLM

engine = LLM(
    model="/path/to/model",
    ir_op_priority={"fused_add_rms_norm": ["loom_cuda"]},
)
```

vLLM appends its native fallback to the priority list. Loom declines tensors
outside its contiguous same-dtype contract, weighted RMSNorm calls without a
normal variance size, and unsupported devices.

To replace vLLM's standard SwiGLU layer as well, opt in before the engine
process starts:

```bash
LOOM_KERNELS_ENABLE_SILU_AND_MUL=1 python your_vllm_service.py
```

Python embedding code can instead call
`loom_kernels.vllm.register_vllm_silu_and_mul()` explicitly before constructing
the model. The replacement supports contiguous CUDA F32/FP16/BF16 input with
an even final dimension and preserves vLLM's output dtype and rounding.

To replace the activation-quant fusion table entries for dynamic symmetric FP8
groups 64 and 128, enable the separate opt-in before vLLM imports its model:

```bash
LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8=1 python your_vllm_service.py
```

Embedding code can call
`loom_kernels.vllm.register_vllm_silu_and_mul_dynamic_fp8()` explicitly. The
replacement uses vLLM's mutable custom-op schema, including an optional F32
scale upper bound and row-major or transposed scale storage. Registration is
intentionally version-specific to the vLLM 0.24/0.25 activation-quant compiler
pass;
unsupported versions should leave the opt-in unset.

To enable fused RoPE+paged-KV on vLLM 0.24/0.25 CUDA, configure the compilation
object before constructing the engine:

```python
from vllm import LLM
from loom_kernels.vllm import configure_vllm_rope_paged_kv

engine = LLM(
    model="/path/to/model",
    compilation_config=configure_vllm_rope_paged_kv(max_token_num=256),
)
```

The helper explicitly enables `+rotary_embedding` and `+quant_fp8`, keeps the
cache update in the compiled graph, registers Loom on the
FlashAttention/FlashInfer backend classes, and enables fusion only through 256
tokens by default. Keeping static FP8 query quant opaque is required for the
FlashAttention FP8 graph to match the official fusion pass. The threshold is
intentional: the H20 advantage is largest for decode-sized batches and narrows
as long prefill becomes compute-bound. The adapter targets vLLM's
version-specific compiler contract and native or static FP8 E4M3 cache dtype.

To enable the measured paged-decode route, opt in before vLLM constructs the
engine:

```bash
LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION=1 python your_vllm_service.py
```

Embedding code can instead call
`loom_kernels.vllm.register_vllm_paged_decode_attention()` explicitly. The
fast path requires FP16/BF16 native KV, Hq/Hkv `32/8`, head size 128, block
size 16 or 32, one causal decoder token per sequence, batch 1-128, and maximum
context 1-32. Sliding windows, ALiBi, soft caps, sinks, cascade/common prefix,
DCP, KV sharing, quantized cache, and multimodal prefix masks all execute the
original `FlashAttentionImpl.forward`. FA3 AOT scheduler metadata is allowed
because it affects only FA3's kernel scheduling, not attention semantics.

To enable the pure-greedy sampled-logprob fast path, register it before engine
construction:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_greedy_sample_logprobs

assert register_vllm_greedy_sample_logprobs() == "greedy_sample_logprobs"
engine = LLM(model="/path/to/model")
```

The adapter only intercepts requests whose sampler contract is all-greedy,
uses raw logprobs, and asks for `logprobs=0`. It also requires no penalties,
allowed-token mask, bad words, per-request logprob token IDs, thinking-budget
state, or active argmax-changing logits processor. F32/FP16/BF16 logits may
have padded rows but require unit vocabulary stride. Every unsupported case
runs the original vLLM sampler; speculative bonus-token sampling is also
declined. Registration is version-gated to vLLM 0.24/0.25. Both contiguous and
padded logits enter the same checked Rust bridge with an explicit row stride.

To replace vLLM's full-vocabulary repetition/frequency/presence temporaries,
register the sparse penalty path before engine construction:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_token_penalties

assert register_vllm_token_penalties() == "token_penalties"
engine = LLM(model="/path/to/model")
```

vLLM still owns penalty parameters and output-history collection. Loom accepts
the existing F32 sampling logits, padded int64 prompt/output matrices, and
three F32 penalty vectors. One packed int64 open-addressing workspace records
prompt presence and output counts, then one CUDA kernel applies repetition
once to the prompt/output union followed by frequency and presence updates.
Negative IDs and IDs at or beyond the vocabulary are padding. The adapter
caches workspace per CUDA stream and uses Loom only while its power-of-two
history capacity is no larger than the vocabulary; other contracts execute
vLLM unchanged. The current H20
[operator gate](../results/h20-token-penalties-20260725.json) is exact and
measures `5.82–34.30x` ratios across rows 1–128. The process-isolated
Qwen2.5-0.5B [baseline-first](../results/h20-vllm-qwen25-token-penalties-baseline-first-20260725.json)
and [Loom-first](../results/h20-vllm-qwen25-token-penalties-loom-first-20260725.json)
vLLM 0.24 gates preserve every token and record `1440/0` Loom submissions in
each order. Their batch-latency ratios are `1.056–1.123x` and TPOT ratios are
`1.068–1.126x`; serving concurrency and goodput remain separate claims.

To fuse the mixed-sampling preprocessing pass, register it before constructing
sampler or engine instances:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_logits_preprocess

assert register_vllm_logits_preprocess() == "logits_preprocess"
engine = LLM(model="/path/to/model")
```

The equivalent process-wide opt-in is
`LOOM_KERNELS_ENABLE_LOGITS_PREPROCESS=1`. The route admits only mixed
greedy/random batches with contiguous F32 sampler logits and a temperature
tensor. It recognizes vLLM's allowed-token mask, one logit-bias processor, one
min-tokens processor, or active bad-word targets. Min-tokens and active
bad-word suppression cannot be combined in the same admitted call because
their sparse metadata may overlap. Active penalties, thinking-budget state,
custom non-argmax-invariant processors, unsupported metadata, and all-greedy
or all-random batches execute the original vLLM path unchanged.

The [direct H20 gate](../results/h20-logits-preprocess-20260727.json) measures
the complete PyTorch operator against the composed mask, sparse bias,
suppression, temperature-guard, and divide sequence. Outputs are exact and
Loom is `3.26–7.30x` faster for 1–32 rows at a 151,936-token vocabulary with
zero measured temporary bytes. The process-isolated
[baseline-first](../results/h20-vllm-logits-preprocess-baseline-first-20260727.json)
and [Loom-first](../results/h20-vllm-logits-preprocess-loom-first-20260727.json)
Qwen2.5-0.5B gates preserve every token, record `720/0` Loom submissions per
order, and exercise mask, bias, suppression, min-tokens, and mixed
temperature. TPOT ratios are order-stable at `1.010–1.084x`; batch latency
crosses parity at batch 32, so this is not a stable model-level batch-latency
claim.

To replace vLLM's top-k-only full sort for small decode batches, register the
exact filter before engine construction:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_top_k_filter

assert register_vllm_top_k_filter() == "top_k_filter"
engine = LLM(model="/path/to/model")
```

The adapter admits F32 sampling logits with one contiguous same-device int32
`top_k` per row, no top-p, and at most seven rows. Loom mutates the logits in
place, preserving every value tied at the kth threshold, then vLLM performs
its original softmax and RNG work. Every valid `top_k` remains on one
device-only algorithm. Eight or more rows use vLLM's original Qrita Triton
path because it exposes exact-position rather than threshold-tie semantics.
The [H20 gate](../results/h20-top-k-filter-20260727.json) measures the complete
PyTorch operator, including its internal workspace allocation, at
`1.42–2.15x` over the corresponding vLLM full sort.

To fuse vLLM's top-p-only full sort and softmax, register the override before
constructing sampler or engine instances:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_top_p_renorm

assert register_vllm_top_p_renorm() == "top_p_renorm"
engine = LLM(model="/path/to/model")
```

The adapter admits contiguous F32 top-p-only sampling logits with rows 2–7,
vocabulary at least 32,768, and one contiguous same-device F32 `top_p` per
row. Loom filters logits in place and returns the renormalized F32
probabilities; vLLM then invokes its unchanged `random_sample` with the
original generators. Joint top-k/top-p, row one, smaller vocabularies,
non-F32 logits, and eight or more rows call the original native path.

The H20 operator reports at
[151,936 tokens](../results/h20-top-p-renorm-20260727.json) and the
[32,768-token crossover](../results/h20-top-p-renorm-vocab32768-20260727.json)
include internal allocations and hard-fail on a latency regression. They
measure `1.72–1.77x` and `1.15–1.34x` ratios respectively. Long F32 cutoff
accumulation is not bitwise associative: versus vLLM the qualified boundary
permits at most one cutoff token per row and probability L1 below `1e-4`.
Deterministic ties within Loom use descending token ID. This is an operator and
adapter gate, not yet an end-to-end model speedup claim.

To replace vLLM's deterministic speculative verifier, register it before
constructing the engine:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_greedy_speculative_verify

assert (
    register_vllm_greedy_speculative_verify()
    == "greedy_speculative_verify"
)
engine = LLM(model="/path/to/model")
```

The hook intercepts only `sampling_metadata.all_greedy` with standard,
non-synthetic rejection semantics. vLLM computes target argmax and owns draft
generation, bonus-token selection, attention, GEMM, scheduler state, and every
stochastic path. Loom consumes contiguous flattened int32 draft IDs, matching
int64 target IDs, int32 bonus IDs shaped `[requests, 1]`, and inclusive int32
cumulative draft lengths. Unsupported contracts call the original vLLM
function. Registration is explicit because the current gates prove exact
operator behavior, lower verifier latency, and real draft/target engine
invocation, but not end-to-end speculative decode acceleration.

To preserve vLLM's full sampling policy but avoid its full-vocabulary raw
log-softmax output, use the general registration instead:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_selected_token_logprobs

assert register_vllm_selected_token_logprobs() == "selected_token_logprobs"
engine = LLM(model="/path/to/model")
```

This registration includes the narrower greedy registration, so pure-greedy
batches keep the fused argmax path. Non-greedy and mixed batches qualify when
vLLM 0.24/0.25 requests raw `logprobs=0` from BF16/FP16 logits and does not
request specific-token or top-k logprob lists. vLLM executes its original F32
processors and sampler first; Loom then scans the preserved raw logits for the
selected int64 IDs. F32 logits and processed-logprob modes conservatively fall
back because vLLM may mutate their storage in place.

For raw top-k logprob lists, register the exact adapter before engine
construction:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_topk_sampled_logprobs

assert register_vllm_topk_sampled_logprobs() == "topk_sampled_logprobs"
engine = LLM(model="/path/to/model")
```

This path accepts 1–32 requested raw logprobs, at most 32 rows, and no
specific-token list. It retains vLLM's F32 `torch.topk` result because equal
low-precision logits make that order externally observable through returned
ranks. After vLLM applies its processors and samples normally, Loom scans the
preserved raw logits for the sampled IDs. The adapter derives the shared
normalizer from the sampled raw logit/logprob pair and applies it only to the
small top-k values. The standalone `topk_sampled_logprobs` operator instead
declares deterministic descending-logit, ascending-token-ID ties; the two
interfaces do not silently substitute one tie contract for the other.

The provider can only replace a graph-visible activation-quant boundary. On
the tested H20 stack, vLLM's automatic `fp8_per_block` selection uses a
FlashInfer/DeepGEMM linear kernel that accepts BF16 and performs activation
quantization inside GEMM. That path contains no separate node for Loom to
replace. The engine A/B therefore fixes `linear_backend="cutlass"`, enables
the `quant_fp8` custom op, and enables `fuse_act_quant` for both providers. The
GEMM is identical on both sides; only the fused activation-quant operator
changes.

To verify selection without starting an engine:

```bash
.venv-vllm/bin/python - <<'PY'
from loom_kernels.vllm import provider_metadata, register_vllm_ir

register_vllm_ir()
print(provider_metadata())
PY
```

## Validation

```bash
.venv-vllm/bin/pytest -q python/tests

.venv-vllm/bin/python benchmarks/vllm_greedy_speculative_verify.py \
  --batches 1,8,32,128,256 --draft-lengths 1,4,8 \
  --warmup 30 --iterations 300 --samples 9 \
  --output /tmp/greedy-speculative-verify.json

.venv-vllm/bin/python benchmarks/vllm_engine_speculative_decode.py \
  --tested-revision "$(git rev-parse HEAD)" \
  --target-model /path/to/Qwen2.5-1.5B-Instruct \
  --target-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --draft-model /path/to/Qwen2.5-0.5B-Instruct \
  --draft-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --spec-tokens 4 --prompt-mode natural \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --warmup 2 --repeats 7 --boundary-profile-repeats 3 \
  --gpu-memory-utilization 0.6 --provider-order native-first \
  --result-json /tmp/speculative-native-first.json

# Repeat with --provider-order loom-first and a distinct result path.

.venv-vllm/bin/python benchmarks/vllm_ir_add_rms_norm.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 100 --iterations 2000 --samples 15

.venv-vllm/bin/python benchmarks/vllm_engine_add_rms_norm.py \
  --model build/synthetic-qwen2-h4096-l4 \
  --provider loom_cuda --batch-size 8 \
  --input-len 128 --output-len 128

.venv-vllm/bin/python benchmarks/vllm_silu_and_mul.py \
  --dtype bf16 --rows 8 --width 11008 \
  --warmup 100 --iterations 2000 --samples 15

.venv-vllm/bin/python benchmarks/vllm_silu_and_mul_dynamic_fp8.py \
  --dtype bf16 --rows 8 --width 11008 --group-size 128 \
  --warmup 100 --iterations 2000 --samples 15 \
  --provider-order forward

.venv-vllm/bin/python benchmarks/vllm_engine_fp8_ab.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --warmup 2 --repeats 7 --provider-order baseline-first \
  --result-json /tmp/qwen25-fp8-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_fp8_ab.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --warmup 2 --repeats 7 --provider-order loom-first \
  --result-json /tmp/qwen25-fp8-loom-first.json

.venv-vllm/bin/python benchmarks/vllm_rope_paged_kv.py \
  --dtype bf16 --layout NHD --tokens 1,8,32,128,256,512 \
  --warmup 100 --iterations 2000 --repeats 5

.venv-vllm/bin/python benchmarks/vllm_engine_rope_paged_kv.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --warmup 2 --repeats 5 \
  --provider-order baseline-first \
  --result-json /tmp/qwen25-rope-kv-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_rope_paged_kv.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --warmup 2 --repeats 5 \
  --provider-order loom-first \
  --result-json /tmp/qwen25-rope-kv-loom-first.json

.venv-vllm/bin/python benchmarks/vllm_paged_decode_shape_sweep.py \
  --batches 1,8,32 --contexts 16,32,64,128 \
  --cache-storage vllm-interleaved \
  --output /tmp/paged-decode-shape-sweep.json

.venv-vllm/bin/python benchmarks/vllm_paged_decode_backend.py \
  --batches 1,8,32 --contexts 16,32,64 \
  --dtypes bf16,f16 --block-sizes 16,32 \
  --output /tmp/paged-decode-backend.json

.venv-vllm/bin/python benchmarks/create_synthetic_qwen2.py \
  --output build/synthetic-qwen2-h4096-l1-stable --layers 1 \
  --hidden-size 4096 --intermediate-size 4096 \
  --attention-heads 32 --kv-heads 8 --max-position-embeddings 64 \
  --stable-token-zero

.venv-vllm/bin/python benchmarks/vllm_engine_paged_decode.py \
  --model build/synthetic-qwen2-h4096-l1-stable \
  --case 1x16x16 --case 8x16x16 --case 32x16x16 \
  --provider-order baseline-first \
  --result-json /tmp/paged-decode-engine.json

.venv-vllm/bin/python benchmarks/vllm_rope_paged_kv.py \
  --dtype bf16 --cache-dtype fp8 --scale-mode per-tensor \
  --layouts NHD,HND --tokens 1,2,4,8,16,32,64,128 \
  --output /tmp/rope-paged-kv-fp8.json

.venv-vllm/bin/python benchmarks/vllm_engine_rope_paged_kv.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --kv-cache-dtype fp8 --provider-order baseline-first \
  --result-json /tmp/qwen25-rope-paged-kv-fp8.json

.venv-calibration/bin/python benchmarks/calibrate_fp8_kv.py \
  --model /path/to/Qwen2.5-7B-Instruct \
  --model-revision <pinned-upstream-revision> \
  --dataset-parquet /path/to/ultrachat-200k-train-sft.parquet \
  --output /path/to/Qwen2.5-7B-Instruct-kvattn-fp8-attn-head \
  --device cuda:0 \
  --attention-target Qwen2Attention \
  --strategy attn_head \
  --observer minmax \
  --samples 512 \
  --max-seq-len 2048

.venv-calibration/bin/python benchmarks/prepare_quality_jsonl.py \
  --model /path/to/Qwen2.5-7B-Instruct \
  --dataset-parquet /path/to/ultrachat-200k-train-sft.parquet \
  --messages-column messages \
  --calibration-manifest \
    /path/to/Qwen2.5-7B-Instruct-kvattn-fp8-attn-head/loom-calibration.json \
  --output /path/to/ultrachat-qwen2.5-heldout-64x512.jsonl \
  --sequences 64 \
  --min-tokens 256 \
  --max-tokens 512 \
  --seed 43

.venv-vllm/bin/python benchmarks/vllm_fp8_kv_system.py \
  --model /path/to/Qwen2.5-7B-Instruct-kvattn-fp8-attn-head \
  --model-revision <pinned-revision-or-checkpoint-digest> \
  --quality-jsonl /path/to/ultrachat-qwen2.5-heldout-64x512.jsonl \
  --quality-max-tokens 512 \
  --attention-backend FLASH_ATTN \
  --variant-order native-first \
  --result-json /tmp/fp8-kv-system-native-first.json

# Repeat with --variant-order fp8-first and a distinct result path.

.venv-vllm/bin/python benchmarks/vllm_greedy_sample_logprobs.py \
  --rows 1,2,4,8,16,32,64,128 --vocab-size 151936 --dtype bf16 \
  --warmup 100 --iterations 1000 --repeats 7

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 5 --provider-order baseline-first \
  --result-json /tmp/qwen25-greedy-logprobs-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 5 --provider-order loom-first \
  --result-json /tmp/qwen25-greedy-logprobs-loom-first.json

.venv-vllm/bin/python benchmarks/vllm_selected_token_logprobs.py \
  --rows 1,2,4,8,16,32,64,128 --vocab-size 151936 --dtype bf16 \
  --warmup 100 --iterations 1000 --repeats 7

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct --sampling-mode top-k-top-p \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 7 --provider-order baseline-first \
  --result-json /tmp/qwen25-selected-logprobs-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct --sampling-mode top-k-top-p \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 7 --provider-order loom-first \
  --result-json /tmp/qwen25-selected-logprobs-loom-first.json

.venv-vllm/bin/python benchmarks/vllm_topk_sampled_logprobs.py \
  --rows 1,8,32,128 --vocab-size 151936 --top-k 20 --dtype bf16 \
  --warmup 100 --iterations 1000 --repeats 7

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct --sampling-mode top-k-top-p \
  --num-logprobs 20 \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 7 --provider-order baseline-first \
  --result-json /tmp/qwen25-topk-logprobs-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct --sampling-mode top-k-top-p \
  --num-logprobs 20 \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 7 --provider-order loom-first \
  --result-json /tmp/qwen25-topk-logprobs-loom-first.json
```

The calibration helper requires `llmcompressor`, `compressed-tensors`, and
`datasets`; keep those optional tools outside the runtime environment. It
refuses to overwrite a checkpoint and records source weights, corpus,
model config, tokenizer, dependency versions, output weights, scale shapes, and
SHA-256 provenance, including the helper itself, in `loom-calibration.json`.
The attention/KV-only recipe calibrates the post-RoPE query and cache K/V
scales required by vLLM without changing model weights. The observer is
mandatory because it is a workload decision: `static_minmax` keeps the global
range, while `minmax` uses an exponential moving average to smooth infrequent
outliers. Stateless memoryless observers are deliberately not exposed by this
tool. The corpus helper verifies the same source data and tokenizer, performs
a deterministic tokenizer-qualified selection, excludes the exact calibration
rows, and writes its own tool/source/output digests to a sidecar manifest. A
small sample count is useful only as a pipeline smoke test; no observer or
checkpoint is accepted until the pinned held-out system gate passes. A
separately pinned WikiText JSONL remains useful as a cross-domain robustness
diagnostic, but it is not a substitute for the representative held-out serving
distribution.

> [!WARNING]
> These commands reproduce the qualification procedure, not an accepted
> Qwen2.5 recipe. On H20, the pinned Qwen2.5-7B per-head minmax candidate
> passed operational capacity and native-vLLM/Loom FP8 equivalence, but both
> FP8 providers regressed BF16 perplexity by about `3.07x` on an early-stop
> slice of 8 held-out sequences and 1,016 scored tokens. The candidate was
> rejected before dual-order TTFT/TPOT measurement; see the
> [rejected system result](../results/h20-fp8-kv-system-rejected-20260727.json).

The microbenchmark compares `loom_cuda` and `vllm_c` through the same vLLM IR
eager dispatcher and CUDA Graph replay. It warms the GPU before each provider
to avoid clock-state order bias. The engine benchmark uses the normal Qwen2
model runner, compilation, scheduler, KV cache, and graph capture, but the
repository's generated checkpoint contains random weights and is not a
pretrained model.

On H20 with vLLM 0.24.0, Loom and `vllm_c` were bitwise identical for the
tested BF16 shapes. Both differ slightly from vLLM IR's FP32-add formal
reference because the CUDA path materializes the BF16 residual before its RMS
reduction. See the
[H20 integration report](../results/h20-vllm-ir-add-rms-norm-20260721.json).

For SiLU-and-Mul, F32/FP16/BF16 and odd-width fallback tests were bitwise equal
to vLLM. Order-reversed CUDA Graph medians were within 0.1%, while eager
dispatch was sensitive to run order. The synthetic Qwen2 engine completed
compilation, graph capture, and generation with the opt-in replacement. See
the [H20 SiLU-and-Mul report](../results/h20-silu-and-mul-20260721.json).

For SiLU-and-Mul+block-FP8, Loom was bitwise identical to vLLM's fused
operator for both supported input dtypes and group sizes. On BF16 `8x11008`,
order-reversed runs showed `1.216-1.231x` eager speedup ratios
(`17.7-18.8%` lower latency) and `1.037-1.082x` CUDA Graph ratios
(`3.6-7.5%` lower latency). The composed vLLM SiLU-then-quantize path is slower
but rounds an intermediate BF16 tensor, so it is not the exact correctness
baseline. See the
[H20 fused activation-quant report](../results/h20-silu-and-mul-dynamic-fp8-20260721.json).

The real-model gate pins Qwen2.5-0.5B-Instruct, online-quantizes it with
vLLM's `fp8_per_block` mode, and runs each provider in a fresh process with an
isolated compile cache. Both provider orders matched every generated token,
each compiler graph recorded two activation-quant matches, and the launch
probe recorded 1584 Loom submissions only in the Loom process. Across the
three cases, batch-latency ratios ranged from `0.9991x` to `1.0043x`, so this
is integration and correctness evidence rather than a model-level performance
claim. See the
[H20 Qwen2.5 engine report](../results/h20-vllm-qwen25-05b-fp8-engine-20260722.json).

For RoPE+paged-KV, FP16/BF16 results were bitwise equal to vLLM's separate
rotary and cache-write operators across packed-QKV, padding, partial rotary,
NHD/HND, and both pairing styles; F32 remained within the qualified tolerance.
BF16 Qwen2.5-style dispatcher ratios were roughly `2.30-2.40x` for 1-512
tokens, then narrowed to `1.088x` at 8192 tokens. Two provider orders on the
real Qwen2.5-0.5B engine matched every generated token and recorded 552 Loom
host submissions only in Loom processes. End-to-end batch-latency ratios
ranged from `0.9957x` to `1.0180x`, so the correct conclusion is engine
integration plus operator-level benefit, not model-level acceleration. See the
[operator report](../results/h20-rope-paged-kv-20260722.json),
[large-token sweep](../results/h20-rope-paged-kv-large-20260722.json), and
[engine report](../results/h20-vllm-qwen25-rope-paged-kv-engine-20260722.json).
Those artifacts cover native caches only. The
[static FP8 E4M3 result](../results/h20-fp8-kv-cache-write-20260724.json)
separately qualifies exact cache bytes, the original ABI2 clean wheel, a
`1.317-1.378x` named-operator range, and exact tokens plus Loom path hits in
both engine orders. Its latency ratios are order-sensitive, and the
current ABI7 wheel matrix requalifies the same FP8 operator tests. The
[first Qwen2.5-7B system candidate](../results/h20-fp8-kv-system-rejected-20260727.json)
then proves operational capacity and provider equivalence but fails the
native-versus-FP8 quality precondition, so no TTFT/TPOT or system-value claim
is made and the family gate remains open for another candidate.
See the [FP8 KV-cache contract](../design/fp8-kv-cache.md).

For paged decode, the native-interleaved
[156-case shape sweep](../results/h20-paged-decode-interleaved-shape-sweep-20260722.json)
has 82 FA3 wins and 74 losses. The focused
[132-case batch sweep](../results/h20-paged-decode-qwen-batch-sweep-20260722.json)
qualifies both low-precision dtypes and block sizes across batches 1-128:
every context-16/32 case wins. The
[backend report](../results/h20-vllm-paged-decode-backend-20260722.json)
confirms all 24 routed cases at `1.154-2.374x` CUDA Graph speedup and graph-parity
fallback for 12 context-64 cases. Order-reversed stable-output synthetic-Qwen
[baseline-first](../results/h20-vllm-paged-decode-engine-baseline-first-20260722.json)
and [Loom-first](../results/h20-vllm-paged-decode-engine-loom-first-20260722.json)
runs match tokens and record zero/18 Loom submissions. Their latency ratios
are process-order sensitive. The stable fixture preserves nonzero Q/K/V work
but zeros the downstream projection and forces a robust token-zero winner, so
the result proves integration rather than pretrained-model numerics or speedup.
The later [odd-GQA sweep](../results/h20-paged-decode-odd-gqa-20260722.json)
passes 72 Qwen2.5-style `14/2`, D64 cases, but the
[pretrained-model experiment](../results/h20-vllm-qwen25-paged-decode-rejected-20260722.json)
matched every generated token in only two of five cases and was 3-5% slower.
That profile is intentionally absent from the adapter; the
[non-regression gate](../results/h20-vllm-paged-decode-tail-gqa-backend-20260722.json)
keeps the existing `32/8`, D128 route at 24/24 wins.

For greedy sampled logprobs, Loom matched vLLM's token IDs and tie-aware ranks
exactly over a 151,936-token BF16 vocabulary; maximum sampled-logprob error was
`9.54e-7`. The fused operator measured `3.16-4.35x` faster for 1-128 rows. Two
isolated provider orders on Qwen2.5-0.5B matched every generated token and
rank, recorded 1120 Loom submissions only in each Loom process, and measured
`1.129-1.250x` batch-latency plus `1.147-1.257x` TPOT ratios. See the
[operator report](../results/h20-greedy-sample-logprobs-20260722.json),
[baseline-first engine report](../results/h20-vllm-greedy-logprobs-baseline-first-20260722.json),
and [Loom-first engine report](../results/h20-vllm-greedy-logprobs-loom-first-20260722.json).

For general selected-token logprobs, caller-selected IDs covered ranks from
288 through 151,842 over the same 151,936-token BF16 vocabulary. Ranks were
exact, maximum logprob error was `9.54e-7`, and the operator measured
`2.77-3.78x` faster for 1-128 rows. Baseline-first and Loom-first Qwen2.5
top-k/top-p runs preserved every vLLM-selected token and rank, recorded 1440
Loom submissions only in each Loom process, and measured `1.044-1.125x`
batch-latency plus `1.054-1.130x` TPOT ratios. See the
[operator report](../results/h20-selected-token-logprobs-20260722.json),
[baseline-first engine report](../results/h20-vllm-selected-logprobs-baseline-first-20260722.json),
and [Loom-first engine report](../results/h20-vllm-selected-logprobs-loom-first-20260722.json).

For sampled-token plus top-k logprobs, the direct deterministic BF16 operator
matches sampled ranks exactly and top-k values within `9.54e-7`. Against
vLLM's full F32 composition it measures `3.25x`, `2.60x`, `1.19x`, and
`0.998x` at 1/8/32/128 rows while reducing peak temporary bytes by roughly two
orders of magnitude. Its tie IDs may differ in position from `torch.topk`
because Loom declares ascending token IDs and PyTorch does not declare an equal
value order. The exact vLLM adapter therefore retains engine `torch.topk`:
both provider orders preserve every token, returned ID/rank, and logprob within
`1.91e-6`, with 1440 Loom selected-token submissions only in the candidate.
Baseline-first latency ratios are `1.037-1.053x`; Loom-first ratios are
`0.969-0.982x`, so the engine result proves invocation and temporary reduction,
not stable acceleration. See the
[operator report](../results/h20-topk-sampled-logprobs-20260725.json),
[baseline-first engine report](../results/h20-vllm-qwen25-topk-logprobs-baseline-first-20260725.json),
and [Loom-first engine report](../results/h20-vllm-qwen25-topk-logprobs-loom-first-20260725.json).

For deterministic greedy speculative verification, Loom matches vLLM's
flattened ragged rejection output bit-for-bit, including zero-draft requests,
first mismatches, full acceptance, and bonus-token emission. All 15 H20 cases
across batches 1-256 and draft lengths 1/4/8 measured `1.101-1.128x` against
vLLM 0.24's exact Triton verifier through equivalent allocating Python calls.
Both vLLM minors pass the expanded 202-test source suite. See the
[H20 verifier report](../results/h20-greedy-speculative-verify-20260723.json).

The process-isolated real-engine gate uses vLLM 0.24 with a pinned
Qwen2.5-1.5B target and Qwen2.5-0.5B draft. Native and Loom speculative paths
match every generated token and acceptance counter in both provider orders;
each Loom run records `714/714` measured verifier launches. Post-timing CUDA
events show a `1.026-1.133x` verifier-boundary ratio, but the verifier accounts
for only `0.048-0.200%` of batch latency. Native/Loom end-to-end ratios cross
parity under order reversal, while speculative decode is `3.18-4.97x` slower
than target-only for the measured cases. See the
[native-first](../results/h20-vllm-qwen25-speculative-native-first-20260723.json)
and [Loom-first](../results/h20-vllm-qwen25-speculative-loom-first-20260723.json)
reports.

The target-only baseline uses different target execution shapes. At batch 32,
two target-only trajectories diverge from both mutually exact speculative
providers after token 51 or 53; the raw reports retain those differences.
Provider correctness is therefore exact native-vLLM versus Loom speculative
equivalence. vLLM's dummy sampler warm-up uses a non-greedy metadata fixture,
so lifetime fallback telemetry is informational; measured Loom launches must
equal measured rejection calls.

## Opt-In Min-P Filtering

```bash
LOOM_KERNELS_ENABLE_MIN_P=1 python your_service.py
```

vLLM 0.24 promotes sampling logits to F32 before its processors. Loom replaces
the allocating `softmax + amax + compare + masked_fill` sequence with the
equivalent in-place threshold `logit < row_max + log(min_p)`. The adapter uses
Loom only for at least 32 rows and a vocabulary of at least 65,536 tokens. It
calls the original vLLM processor for smaller shapes because H20 evidence shows
that the current one-block-per-row kernel is slower there.

The [Qwen-vocabulary report](../results/h20-min-p-filter-20260722.json) records
exact masks, retained logits, temporary memory, all raw samples, and the row
crossover. The
[65,536-token boundary report](../results/h20-min-p-filter-vocab65536-20260722.json)
independently validates the lower vocabulary gate. The override is opt-in and
has not yet completed a real-model end-to-end gate.

The compatible arithmetic and schema follow vLLM 0.24's
[fused CUDA implementation](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu)
and its documented
[fusion mechanism](https://docs.vllm.ai/en/v0.23.0/design/fusions/).

The provider API follows vLLM's
[IR design](https://docs.vllm.ai/en/v0.22.1/design/vllm_ir/) and the mutable
dispatcher follows PyTorch's
[LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
and [custom-operator contract](https://docs.pytorch.org/docs/stable/library.html).

## Current Limits

- Linux and CUDA only;
- the first native artifact is qualified only for Linux x86_64, CUDA 13.1,
  SM90, and H20 Python 3.11; it is not published;
- inference-only mutation, with no autograd implementation;
- one selectable IR provider (`fused_add_rms_norm`), one opt-in out-of-tree
  layer replacement (`SiluAndMul`), and one vLLM-version-specific
  activation-quant fusion-table replacement, plus a vLLM 0.24/0.25-specific
  RoPE+native/static-FP8-KV compiler-pass adapter, greedy/general selected-token
  and top-k sampled-logprob sampler overrides, shape-gated top-k and fused
  top-p/renormalization overrides, a shape-gated Min-P override, and a
  measured-shape FlashAttention paged-decode override;
- the activation-quant provider requires a graph-visible quantization boundary;
  it does not intercept vLLM's fused BF16-input FlashInfer/DeepGEMM path;
- the isolated operator is faster on H20 and real-model invocation is proven,
  but no model-level speedup has been established for either FP8 activation
  fusion or RoPE+paged-KV;
- vLLM-owned penalties, masks, and stochastic sampling can feed the
  selected-token path; Loom now accelerates measured top-k-only and top-p-only
  filtering shapes but leaves joint top-k/top-p and RNG native. Min-P remains
  separately shape-gated, raw top-k logprob lists have an exact
  rows-at-most-32 adapter, and non-raw modes still fall back;
- paged decode is limited to the exact H20-qualified 32/8-head, D128,
  context-at-most-32 envelope; pretrained-model and serving-scale evidence plus
  competitive 128-1,024-token kernels remain open.
