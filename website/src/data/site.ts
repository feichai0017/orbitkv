export const repositoryUrl = "https://github.com/feichai0017/loom-infer"

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Operators", href: "/docs/operators/" },
  { label: "Evidence", href: "/evidence/" },
]

export const operatorFamilies = [
  {
    name: "Normalization",
    boundary: "RMSNorm, residual fusion, and quantized output.",
    state: "Device correct",
  },
  {
    name: "Attention",
    boundary: "Single and paged decode, split-K, state merge, and ragged causal prefill.",
    state: "Partial device correct",
  },
  {
    name: "Decode tail",
    boundary: "Sampling, filters, penalties, logprobs, and verification.",
    state: "Planned",
  },
  {
    name: "KV cache",
    boundary: "RoPE append, gather, scatter, compaction, and quantization.",
    state: "Planned",
  },
  {
    name: "MoE",
    boundary: "Routing support, permutation, grouped-GEMM input, and combine.",
    state: "Planned",
  },
  {
    name: "Matrix work",
    boundary: "Fixed Rust plans for qualified vendor GEMM providers.",
    state: "Device correct",
  },
]

export const milestones = [
  {
    milestone: "01",
    name: "Permanent RMSNorm",
    reason: "Owned bindings pass H20 correctness; the pinned FlashInfer RMSNorm baseline did not compile on CUDA 13.1.",
  },
  {
    milestone: "02",
    name: "Vendor GEMM",
    reason: "The fixed M=1 cuBLASLt eager path is 1.33x lower-latency than the matched FlashInfer path.",
  },
  {
    milestone: "03",
    name: "Attention core",
    reason: "Ragged causal prefill is H20-correct; matched performance and Graph replay remain open.",
  },
]
