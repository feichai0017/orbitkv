export const repositoryUrl = "https://github.com/feichai0017/oxide-infer"
export const homeUrl = "https://feichai0017.github.io/oxide-infer/"

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Docs", href: "/docs/" },
  { label: "Operators", href: "/docs/operators/" },
  { label: "Evidence", href: "/evidence/" },
]

export const executionStages = [
  { key: "01", name: "Spec", detail: "Fix tensor semantics, shape, dtype, layout, and numerical limits" },
  { key: "02", name: "Planner", detail: "Reject unsupported contracts and select one provider and algorithm" },
  { key: "03", name: "Plan", detail: "Freeze launch geometry, workspace, artifacts, and Graph policy" },

  { key: "04", name: "Operands", detail: "Bind typed read regions and transfer exclusive write authority" },
  { key: "05", name: "CommandScope", detail: "Resolve aliases, retain resources, and enqueue on one checked stream" },
  { key: "06", name: "Completion", detail: "Return status and memory authority only after the fence settles" },
]

export const operatorGroups = [
  {
    state: "Current source",
    tone: "current",
    note: "Implemented contracts. Most CUDA paths need current-source H20 requalification.",
    operators: [
      { name: "RMSNorm", boundary: "F32, FP16, BF16 · scalar and packed", provider: "Native" },
      { name: "Dense GEMM", boundary: "BF16 D=A×Wᵀ · F32 accumulation", provider: "Vendor" },
      { name: "Decode", boundary: "single, split-K, paged · BF16 D128", provider: "Native" },
      { name: "Prefill", boundary: "ragged and paged causal · BF16 D128", provider: "Native" },
      { name: "RoPE", boundary: "NeoX split-half · BF16 D128", provider: "Native" },
      { name: "Paged KV append", boundary: "fused RoPE · exclusive target pages", provider: "Native" },
    ],
  },
  {
    state: "Experimental",
    tone: "experimental",
    note: "Usable only inside the declared contract. No performance or engine claim.",
    operators: [
      { name: "M=1 GEMV", boundary: "BF16 · H20 sm_90a · fixed census", provider: "Native" },
      { name: "Engine stream bridge", boundary: "external regions · event ordering", provider: "Runtime" },
    ],
  },
  {
    state: "Planned",
    tone: "planned",
    note: "Roadmap items. These rows do not describe implemented providers.",
    operators: [
      { name: "Fused activations", boundary: "contract and provider not admitted", provider: "Open" },
      { name: "Sampling", boundary: "contract and provider not admitted", provider: "Open" },
      { name: "Engine adapters", boundary: "real model call and no-copy proof", provider: "Open" },
    ],
  },
]

export const providerLanes = [
  {
    id: "NATIVE / 01",
    name: "Native provider",
    detail: "Rust host contracts and cuda-oxide device code. Used where ownership, launch policy, or fusion must remain explicit.",
    examples: ["attention", "RMSNorm", "RoPE", "paged KV append", "experimental GEMV"],
  },
  {
    id: "VENDOR / 02",
    name: "Vendor provider",
    detail: "Qualified libraries behind the same planner and plan contract. The current dense BF16 GEMM path uses cuBLASLt.",
    examples: ["cuBLASLt", "explicit algorithm", "caller workspace", "same completion"],
  },
]

export const evidenceHighlights = [
  { value: "Phase 1", label: "Current H20 source", detail: "GEMM, ragged and paged prefill, RoPE, and fused append", sample: "correctness and named Graph cases" },
  { value: "Pending", label: "Remaining R1", detail: "five runners, Compute Sanitizer, and performance", sample: "not qualified by phase 1" },
  { value: "Simulated", label: "Engine interop", detail: "external pointers and event bridge", sample: "not a model run" },
  { value: "Open", label: "Serving", detail: "TTFT, TPOT, throughput, and memory", sample: "no serving evidence" },
]

export const evidenceLevels = [
  { level: "01", name: "Contract", status: "Required", detail: "CPU oracle, edge cases, and typed rejection" },
  { level: "02", name: "Device", status: "Per source", detail: "GPU correctness, lifecycle, and sanitizer" },
  { level: "03", name: "Performance", status: "Per shape", detail: "Matched inputs, streams, and raw samples" },

  { level: "04", name: "Graph", status: "Per plan", detail: "Capture, replay, addresses, and retention" },
  { level: "05", name: "Engine", status: "Open", detail: "Real call site, provider hit, and model parity" },
  { level: "06", name: "Serving", status: "Open", detail: "Workload, TTFT, TPOT, throughput, and memory" },
]

export const milestones = [
  { milestone: "NOW", name: "Requalify merged paths", reason: "Publish current H20 correctness, sanitizer, Graph, and matched records." },
  { milestone: "NEXT", name: "Close one engine adapter", reason: "Run a real model call with provider trace, output parity, and no-copy evidence." },
  { milestone: "THEN", name: "Measure serving impact", reason: "Report workload, TTFT, TPOT, throughput, and memory before broader claims." },
]

export const engineAdapters = [
  { name: "mistral.rs", state: "First target", detail: "Rust-native model call and stream ownership boundary." },
  { name: "vLLM", state: "Planned", detail: "Out-of-tree custom operator or attention backend seam." },
  { name: "SGLang", state: "Planned", detail: "Operator backend boundary after one adapter passes qualification." },
  { name: "Candle", state: "Planned", detail: "Direct Rust integration after contracts stabilize." },
]
