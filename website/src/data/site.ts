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
    boundary: "Single decode baseline, then paged decode, split-K, and state merge.",
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
    reason: "Owned bindings pass H20 correctness and sanitizer gates. Performance remains open.",
  },
  {
    milestone: "02",
    name: "Vendor GEMM",
    reason: "The fixed plan and Graph chain pass H20 gates. Baseline and engine gates remain open.",
  },
  {
    milestone: "03",
    name: "Attention core",
    reason: "Single decode passes H20 correctness. Matched performance and paged decode remain open.",
  },
]
