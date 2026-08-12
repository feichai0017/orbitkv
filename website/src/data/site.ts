export const repositoryUrl = "https://github.com/feichai0017/oxide-infer"
export const homeUrl = "https://feichai0017.github.io/oxide-infer/"
export const performanceEvidenceUrl = `${repositoryUrl}/blob/main/docs/results/h20-flashinfer-v0.6.17-attention-eager-performance-7f3d08e-20260812.json`
export const pagedPrefillOptimizationEvidenceUrl = `${repositoryUrl}/blob/main/docs/results/h20-flashinfer-v0.6.17-paged-prefill-current-gqa4-eager-performance-02faf27-20260812.json`
export const raggedPrefillOptimizationEvidenceUrl = `${repositoryUrl}/blob/main/docs/results/h20-flashinfer-v0.6.17-ragged-prefill-dual-tile-gqa4-eager-performance-f9b95b0-20260812.json`

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
    note: "Implemented contracts. Declared permanent-runner paths passed current-source R1 device qualification.",
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
      { name: "M=1 GEMV", boundary: "BF16 · SM90a · fixed census", provider: "Native" },
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
  { value: "9 / 9", label: "Qualification runners", detail: "89 passing case lines and 12 named Graph cases", sample: "recorded H20 test host · exact source and binary hashes" },
  { value: "36 / 36", label: "Compute Sanitizer", detail: "memcheck, racecheck, synccheck, and initcheck", sample: "no filters or suppressions" },
  { value: "14 / 14", label: "Stable benchmark shapes", detail: "8 lower-latency Oxide paths and 6 FlashInfer paths", sample: "matched eager-provider timing · not serving" },
  { value: "Open", label: "Engine serving", detail: "TTFT, TPOT, throughput, and memory", sample: "no model-level speed claim" },
]

export const evidenceLevels = [
  { level: "01", name: "Contract", status: "Required", detail: "CPU oracle, edge cases, and typed rejection" },
  { level: "02", name: "Device", status: "R1 qualified", detail: "Current-source correctness, lifecycle, Graph, and sanitizer on the recorded target" },
  { level: "03", name: "Performance", status: "14 shapes", detail: "Matched inputs, streams, provider orders, and raw samples" },

  { level: "04", name: "Graph", status: "Per plan", detail: "Capture, replay, addresses, and retention" },
  { level: "05", name: "Engine", status: "Open", detail: "Real call site, provider hit, and model parity" },
  { level: "06", name: "Serving", status: "Open", detail: "Workload, TTFT, TPOT, throughput, and memory" },
]

export const milestones = [
  { milestone: "NOW", name: "Optimize measured gaps", reason: "Long-context GQA prefill is the clearest current bottleneck." },
  { milestone: "NEXT", name: "Close one engine adapter", reason: "Measure provider hits, output parity, TTFT, TPOT, throughput, and memory." },
  { milestone: "THEN", name: "Expand measured contracts", reason: "Add operators only after a workload census and explicit evidence gate." },
]

export const performanceSummary = [
  { value: "14 / 14", label: "stable shapes", detail: "both provider-order deltas ≤ 5%" },
  { value: "8", label: "Oxide lower", detail: "combined median eager latency" },
  { value: "6", label: "FlashInfer lower", detail: "combined median eager latency" },
]

export const performanceRows = [
  {
    name: "Paged decode · MHA",
    shape: "B1 · KV 1 · NHD · D128",
    oxideUs: "9.54",
    flashinferUs: "13.77",
    result: "Oxide 1.44× lower",
    winner: "oxide",
  },
  {
    name: "Ragged prefill · MHA",
    shape: "Q 16 · KV 16 · D128",
    oxideUs: "8.25",
    flashinferUs: "13.99",
    result: "Oxide 1.69× lower",
    winner: "oxide",
  },
  {
    name: "Ragged prefill · GQA4",
    shape: "Q 32+64 · KV 256+1024 · D128",
    oxideUs: "36.94",
    flashinferUs: "21.93",
    result: "FlashInfer 1.68× lower",
    winner: "flashinfer",
  },
  {
    name: "Paged prefill · GQA4",
    shape: "Q 32+64 · KV 256+1024 · D128",
    oxideUs: "46.60",
    flashinferUs: "23.21",
    result: "FlashInfer 2.01× lower",
    winner: "flashinfer",
  },
]

export const engineAdapters = [
  { name: "mistral.rs", state: "First target", detail: "Rust-native model call and stream ownership boundary." },
  { name: "vLLM", state: "Planned", detail: "Out-of-tree custom operator or attention backend seam." },
  { name: "SGLang", state: "Planned", detail: "Operator backend boundary after one adapter passes qualification." },
  { name: "Candle", state: "Planned", detail: "Direct Rust integration after contracts stabilize." },
]
