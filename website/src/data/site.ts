export const repositoryUrl = "https://github.com/feichai0017/loom-infer"

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Operators", href: "/docs/operators/" },
  { label: "Evidence", href: "/evidence/" },
]

export const executionStages = [
  { key: "01", name: "Contract", detail: "Shape, dtype, layout, and numerical rules" },

  { key: "02", name: "Plan", detail: "One provider, algorithm, workspace, and Graph policy" },

  { key: "03", name: "Bind", detail: "Typed read leases and exclusive write authority" },

  { key: "04", name: "Enqueue", detail: "Rust kernel or qualified vendor call on one stream" },

  { key: "05", name: "Complete", detail: "One fence reports rejection or returns write authority" },
]

export const operatorFamilies = [
  {
    name: "RMSNorm",
    boundary: "F32, FP16, and BF16 scalar and packed paths.",
    provider: "cuda-oxide",
    state: "Requalification",
    tone: "review",
  },
  {
    name: "GEMM",
    boundary: "Fixed contiguous BF16 D = A × Wᵀ plans.",
    provider: "cuBLASLt",
    state: "Requalification",
    tone: "review",
  },
  {
    name: "Decode",
    boundary: "Single direct and split-K, plus typed-status page-size-16 batch decode.",
    provider: "cuda-oxide",
    state: "Requalification",
    tone: "review",
  },
  {
    name: "Prefill",
    boundary: "Ragged and explicit-plan paged causal MHA, MQA, and GQA.",
    provider: "cuda-oxide",
    state: "Requalification",
    tone: "review",
  },
  {
    name: "RoPE",
    boundary: "BF16 D128 NeoX with explicit I32 positions.",
    provider: "cuda-oxide",
    state: "Requalification",
    tone: "review",
  },
  {
    name: "KV append",
    boundary: "Cache-bound append map, fused RoPE, and exclusive page writes.",
    provider: "cuda-oxide",
    state: "Requalification",
    tone: "review",
  },
]

export const evidenceHighlights = [
  {
    value: "Source",
    label: "Paged status",
    detail: "Typed metadata errors for decode, prefill, and append",
    sample: "records pending",
  },
  {
    value: "H20",
    label: "Simulated interop",
    detail: "Single and HND paged decode, two in flight, no adapter D2D copy",
    sample: "not a real engine",
  },
  {
    value: "Pending",
    label: "Current records",
    detail: "Device, sanitizer, Graph, and matched evidence",
    sample: "old records are historical",
  },
  {
    value: "Open",
    label: "mistral.rs proof",
    detail: "Real adapter, provider trace, and model-output parity",
    sample: "no model claim",
  },
]

export const evidenceLevels = [
  { level: "01", name: "Contract", status: "Required", detail: "CPU oracle and edge cases" },

  { level: "02", name: "Device", status: "Required", detail: "H20 correctness and sanitizer" },

  { level: "03", name: "Performance", status: "Per shape", detail: "Matched provider timings" },

  { level: "04", name: "Graph", status: "Per path", detail: "Capture, replay, and lifetime" },

  { level: "05", name: "Engine", status: "Open", detail: "Zero-copy model invocation" },

  { level: "06", name: "Serving", status: "Open", detail: "TTFT, TPOT, throughput, and memory" },
]

export const milestones = [
  {
    milestone: "01",
    name: "Current-source records",
    reason: "Publish H20 correctness, sanitizer, and Graph evidence for the merged paths.",
  },
  {
    milestone: "02",
    name: "mistral.rs boundary",
    reason: "Use HND decode and linear authority handoff in one real model call.",
  },
  {
    milestone: "03",
    name: "Matched evidence",
    reason: "Prove reference parity and query Graph nodes before a provider ranking.",
  },
]
