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
    state: "Device-correct",
  },
  {
    name: "Attention",
    boundary: "Ragged prefill, paged decode, split-K, and state merge.",
    state: "Planned",
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
    state: "Device-correct",
  },
]

export const milestones = [
  {
    milestone: "01",
    name: "Permanent RMSNorm",
    reason: "F32, FP16, and BF16 correctness passes. Fixed argument packs and later gates remain open.",
  },
  {
    milestone: "02",
    name: "Vendor GEMM",
    reason: "The fixed BF16 cuBLASLt plan passes correctness. Graph, baseline, and engine gates remain open.",
  },
  {
    milestone: "03",
    name: "Attention core",
    reason: "Implement ragged prefill and paged decode against matched providers.",
  },
]
