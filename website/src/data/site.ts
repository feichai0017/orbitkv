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
    boundary: "Single direct and split-K, plus page-size-16 batch decode.",
    provider: "cuda-oxide",
    state: "Requalification",
    tone: "review",
  },
  {
    name: "Prefill",
    boundary: "Ragged and paged bottom-right causal MHA, MQA, and GQA.",
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
    value: "Current",
    label: "Host contracts",
    detail: "Shared-tail isolation and generated paged-decode cases",
    sample: "local gates",
  },
  {
    value: "Pending",
    label: "H20 records",
    detail: "Current-source device, sanitizer, and Graph evidence",
    sample: "old records are historical",
  },
  {
    value: "Re-run",
    label: "Matched attention",
    detail: "Common F32 oracle and strict contract grouping",
    sample: "old timings are historical",
  },
  {
    value: "Open",
    label: "Engine proof",
    detail: "Zero-copy invocation from one real model path",
    sample: "no serving claim",
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
    name: "Append requalification",
    reason: "Publish H20 records for exclusive-page writes and typed completion status.",
  },
  {
    milestone: "02",
    name: "Engine memory",
    reason: "Qualify external CUDA regions without copying or transferring allocations.",
  },
  {
    milestone: "03",
    name: "Paged status",
    reason: "Return typed metadata errors from decode and prefill completion.",
  },
]
