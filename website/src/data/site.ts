export const repositoryUrl = "https://github.com/feichai0017/loom-infer"

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Operators", href: "/docs/operators/" },
  { label: "Evidence", href: "/evidence/" },
]

export const operatorFamilies = [
  {
    name: "RMSNorm",
    boundary: "F32, FP16, and BF16 scalar and packed paths.",
    state: "Qualified",
  },
  {
    name: "Decode",
    boundary: "Single-request direct and split-K plus paged batch decode.",
    state: "Qualified",
  },
  {
    name: "Prefill",
    boundary: "Ragged bottom-right causal MHA, MQA, and GQA.",
    state: "Qualified",
  },
  {
    name: "RoPE + KV",
    boundary: "Standard RoPE and fused 1-through-64-token paged append.",
    state: "Qualified",
  },
  {
    name: "GEMM",
    boundary: "Fixed-algorithm contiguous BF16 cuBLASLt plans.",
    state: "Qualified",
  },
]

export const evidenceHighlights = [
  {
    value: "2.942×",
    label: "lower eager latency",
    detail: "Fused RoPE + one-token paged append",
  },
  {
    value: "1.656×",
    label: "lower Graph replay latency",
    detail: "Explicit six-token fused append",
  },
  {
    value: "4.41×",
    label: "lower eager latency",
    detail: "Paged batch-1 MHA decode",
  },
]

export const milestones = [
  {
    milestone: "01",
    name: "Broader attention",
    reason: "Expand ragged query tiling and add common window and mask contracts.",
  },
  {
    milestone: "02",
    name: "Graph coverage",
    reason: "Qualify remaining fixed-address paths before adding mutable Graph contracts.",
  },
  {
    milestone: "03",
    name: "Engine integration",
    reason: "Invoke Loom from a real Rust model engine without tensor copies.",
  },
]
