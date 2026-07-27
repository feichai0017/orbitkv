export const repositoryUrl = "https://github.com/feichai0017/loom-kernels";

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Operators", href: "/docs/operators/" },
  { label: "Integration", href: "/docs/integration/" },
  { label: "Evidence", href: "/benchmarks/" },
];

export const supportedOperators = [
  {
    name: "RMSNorm",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Standalone normalization",
    status: "supported",
  },
  {
    name: "Add + RMSNorm",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Residual update + normalization",
    status: "supported",
  },
  {
    name: "RMSNorm + dynamic FP8",
    dtypes: "F32 · FP16 · BF16 → E4M3FN",
    boundary: "Normalization + GEMM input quantization",
    status: "supported",
  },
  {
    name: "SiLU-and-Mul",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Split-half SwiGLU activation",
    status: "supported",
  },
  {
    name: "SiLU-and-Mul + block FP8",
    dtypes: "FP16 · BF16 → E4M3FN",
    boundary: "Activation + group-64/128 quantization",
    status: "supported",
  },
  {
    name: "RoPE + paged-KV write",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Packed Q/K rotation + native cache write",
    status: "supported",
  },
  {
    name: "Greedy + sampled logprob",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Argmax + logsumexp + gather + tie rank",
    status: "supported",
  },
  {
    name: "Selected-token logprob + rank",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Engine-owned sampling + one-pass normalization",
    status: "supported",
  },
  {
    name: "Sampled-token + top-k logprobs",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Deterministic direct reduction · exact engine-order adapter",
    status: "supported",
  },
  {
    name: "Min-P filtering",
    dtypes: "F32 · FP16 · BF16",
    boundary: "In-place row-max threshold; shape-gated in vLLM",
    status: "supported",
  },
  {
    name: "Sparse token penalties",
    dtypes: "F32 logits · int64 history",
    boundary: "Repetition + frequency + presence through an O(history) hash",
    status: "supported",
  },
  {
    name: "Fused logits preprocessing",
    dtypes: "F32 sampler logits",
    boundary: "Mask + sparse bias/suppression + mixed-row temperature",
    status: "supported",
  },
  {
    name: "Paged MQA/GQA decode",
    dtypes: "F32 · FP16 · BF16",
    boundary: "GQA packing + local split-K/LSE; short shapes route into vLLM",
    status: "supported",
  },
  {
    name: "Greedy speculative verify",
    dtypes: "int32 drafts · int64 target IDs",
    boundary: "Ragged acceptance + mismatch/bonus-token compaction",
    status: "supported",
  },
];

export const nextOperators = [
  {
    milestone: "K4 · ABI8-A",
    name: "Counter-based sampling",
    reason: "Admission passed; implement Rust Philox/inverse-CDF, one CUDA execution pattern, checked ABI8, and persistent request-slot state.",
  },
  {
    milestone: "K2.5 · P1",
    name: "Quantization plumbing",
    reason: "Remove scale, pack/unpack, dequant/requant, and layout passes around an unchanged vendor GEMM.",
  },
  {
    milestone: "K5 · P1",
    name: "MoE routing + movement",
    reason: "Own routing, histogram/prefix sum, permutation, and combine while grouped GEMM stays vendor-owned.",
  },
  {
    milestone: "K3 · Gated",
    name: "Physical KV movement",
    reason: "Default vLLM prefix/preemption was rejected; revisit only for a measured offload, beam, or compaction path.",
  },
  {
    milestone: "K3 · Evidence",
    name: "FP8 KV-cache qualification",
    reason: "The first Qwen2.5 candidate was rejected on quality; retry only with a distinct pinned model, backend, or cache representation.",
  },
  {
    milestone: "K4.5 · Gated",
    name: "Speculative extensions",
    reason: "Add tree, stochastic, or KV metadata only when a named profile exposes material non-GEMM cost.",
  },
  {
    milestone: "K8 · Proof",
    name: "Rust decode step",
    reason: "Prove zero-copy engine-neutral orchestration over borrowed tensors and streams without building an inference engine.",
  },
];

export const evidence = [
  {
    operator: "Add + RMSNorm",
    shape: "BF16 · 8 × 4096",
    result: "2.914 µs",
    detail: "Raw H20 kernel median",
  },
  {
    operator: "RMSNorm + FP8",
    shape: "BF16 · 8 × 4096",
    result: "1.057–1.076×",
    detail: "CUDA Graph ratio vs vLLM",
  },
  {
    operator: "SiLU + Mul + FP8",
    shape: "BF16 · 8 × 11008 · G128",
    result: "1.037–1.082×",
    detail: "CUDA Graph ratio vs vLLM fused",
  },
  {
    operator: "Qwen2.5 FP8 engine",
    shape: "0.5B · batches 1 / 8 / 32",
    result: "0.999–1.004×",
    detail: "Exact-token path hit; end-to-end parity",
  },
  {
    operator: "RoPE + paged-KV write",
    shape: "BF16 · Qwen2.5-style · 1–512 tokens",
    result: "2.30–2.40×",
    detail: "Dispatcher ratio vs separate vLLM ops",
  },
  {
    operator: "Static FP8 KV system candidate",
    shape: "Qwen2.5-7B · 8 held-out sequences · 1,016 scored tokens",
    result: "Rejected · 3.07× PPL",
    detail: "1.99879× cache capacity and 1.00064 provider ratio; quality fails before TTFT/TPOT",
  },
  {
    operator: "Default vLLM KV movement candidate",
    shape: "Qwen2.5-0.5B · 1,024-token prefix · 96-request pressure",
    result: "Rejected · 0 copy calls",
    detail: "Prefix hit and three preemptions observed; reuse is logical and preemption recomputes",
  },
  {
    operator: "Seeded categorical-sampling admission",
    shape: "F32 · 151,936 vocab · 8 / 32 all-seeded rows",
    result: "Admitted · 10 / 34 kernels",
    detail: "3.00× at 8 rows; isolated 32-row path is 4.82× with 19.45 MB temporary storage",
  },
  {
    operator: "Greedy + sampled logprob",
    shape: "Qwen2.5-0.5B · batches 1 / 8 / 32",
    result: "1.129–1.250×",
    detail: "Order-stable real-engine batch-latency ratio",
  },
  {
    operator: "Selected-token logprob + rank",
    shape: "Qwen2.5 top-k/top-p · batches 1 / 8 / 32",
    result: "1.044–1.125×",
    detail: "vLLM-owned sampling; order-stable engine ratio",
  },
  {
    operator: "Sampled-token + top-k logprobs",
    shape: "BF16 · 151,936 vocab · 1 / 8 / 32 rows",
    result: "3.25 / 2.60 / 1.19×",
    detail: "Direct operator; exact engine adapter crosses parity after order reversal",
  },
  {
    operator: "Min-P filtering",
    shape: "F32 · 151,936 vocab · 128 rows",
    result: "1.885×",
    detail: "0 tensor-sized temp; smaller batches route back to vLLM",
  },
  {
    operator: "Sparse token penalties",
    shape: "Qwen2.5-0.5B · batches 1 / 8 / 32",
    result: "1.056–1.123×",
    detail: "Order-stable engine ratio; operator ratio 5.82–34.30×",
  },
  {
    operator: "Fused logits preprocessing",
    shape: "F32 · 151,936 vocab · 1–32 rows",
    result: "3.26–7.30×",
    detail: "Exact operator; order-stable 1.010–1.084× Qwen TPOT ratio",
  },
  {
    operator: "Paged MQA/GQA decode",
    shape: "FP16/BF16 · Hq/Hkv 32/8 · context ≤ 32",
    result: "1.154–2.374×",
    detail: "24/24 routed vLLM backend cases win; other shapes fall back to FA3",
  },
  {
    operator: "Paged decode split-K/LSE",
    shape: "BF16 · batch 1–8 · context 128–1,024",
    result: "1.14–6.22×",
    detail: "CUDA Graph ratio vs legacy Loom; FA3 remains the engine fallback",
  },
  {
    operator: "Greedy speculative verify",
    shape: "H20 · batch 1–256 · draft length 1 / 4 / 8",
    result: "1.101–1.128×",
    detail: "Bit-exact verifier-level ratio vs vLLM 0.24",
  },
  {
    operator: "Real-model speculative decode",
    shape: "Qwen2.5 1.5B target + 0.5B draft · batch 1 / 8 / 32",
    result: "0.048–0.200%",
    detail: "Verifier share of batch latency; exact native/Loom path, no end-to-end win",
  },
];
