export const repositoryUrl = "https://github.com/feichai0017/orbitkv";
export const author = {
  name: "feichai",
  url: "https://github.com/feichai0017",
};

// The build binds documentation links to the same source revision as the site.
export const sourceUrl = (path: string) =>
  `${repositoryUrl}/blob/${import.meta.env.PUBLIC_SOURCE_REF}/${path}`;
export const localUrl = (path: string) =>
  `${import.meta.env.BASE_URL.replace(/\/$/, "")}${path}`;

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Architecture", href: "/docs/" },
  { label: "Evidence", href: "/evidence/" },
];

export const layers = [
  {
    name: "orbitkv",
    role: "Compile & own state",
    detail:
      "Attention lifetimes, page layouts, prefix sharing, copy-on-write, and safe reclamation. Usable as a backend-independent Rust crate.",
    path: "crates/orbitkv/README.md",
  },
  {
    name: "orbitkv-executor",
    role: "Lower & execute graphs",
    detail:
      "Bind manager-owned arenas to Luminal, profile legal kernel candidates on CUDA, and persist the selected programs.",
    path: "crates/orbitkv-executor/README.md",
  },
  {
    name: "orbitkv-engine",
    role: "Schedule & serve",
    detail:
      "Batch requests, coordinate execution, stream tokens, and handle cancellation through an optional OpenAI-compatible frontend.",
    path: "crates/orbitkv-engine/README.md",
  },
];

export const compilerStages = [
  {
    name: "Describe",
    detail: "Model math, attention visibility, and state retention.",
  },
  {
    name: "Compile",
    detail: "Physical layouts, ownership rules, and legal graph alternatives.",
  },
  {
    name: "Measure",
    detail: "CUDA execution selects candidates for each shape bucket.",
  },
  {
    name: "Replay",
    detail: "Load the selected program and reuse state after completion.",
  },
];

export const providers = [
  {
    name: "cuBLASLt",
    role: "Dense & batched matrix products",
    scope: "Supported scaling, bias, and activation epilogues.",
  },
  {
    name: "DeepGEMM",
    role: "Block-scaled FP8 linear",
    scope: "SM90 tile candidates and optional shared activation preparation.",
  },
  {
    name: "FlashInfer",
    role: "Paged attention",
    scope: "CUDA-core decode and tensor-core decode / prefill.",
  },
  {
    name: "FlashAttention-3",
    role: "Paged attention",
    scope: "Optional SM90 F16 / BF16 decode and packed prefill.",
  },
];

export const evidenceHighlights = [
  {
    value: "27B",
    label: "Hybrid model execution",
    detail:
      "Block-FP8 text decoder with Full attention, Gated DeltaNet, and convolution state.",
  },
  {
    value: "96",
    label: "Logit comparisons",
    detail:
      "Full-vocabulary checks across fresh search and two strict replay configurations.",
  },
  {
    value: "0",
    label: "NVRTC calls on replay",
    detail:
      "423 generated module images loaded in the recorded HTTP replay run; provider libraries cached.",
  },
];

export const evidenceRows = [
  {
    surface: "Attention-state compiler",
    status: "Host verified",
    detail:
      "Full, Sliding, Full + Sliding, and exact Chunked lifetimes compile into deterministic plans.",
    boundary: "Exact Chunked has no released-model device qualification.",
  },
  {
    surface: "KV & fixed-state lifecycle",
    status: "Host + H20",
    detail:
      "Generation ownership, prefix / COW, event-gated completion, cancellation, and final drain.",
    boundary:
      "Bounded workloads; production soak and capacity limits remain open.",
  },
  {
    surface: "Model execution",
    status: "Bounded H20",
    detail:
      "Dense Full, interleaved Full + Sliding, and the primary 27B block-FP8 hybrid text checkpoint.",
    boundary:
      "No end-to-end MLA, MoE, multimodal, or multi-device qualification.",
  },
  {
    surface: "Attention selection",
    status: "Measured on CUDA",
    detail:
      "FlashInfer CUDA-core / tensor-core algorithms and optional FlashAttention-3 enter the same search space.",
    boundary:
      "Current execution uses causal / sliding attention over paged NHD K/V. No global-optimum guarantee.",
  },
  {
    surface: "Artifacts & serving",
    status: "Bounded H20",
    detail:
      "Strict schedule and module-image replay, continuous batching, HTTP / SSE, shutdown, and state drain.",
    boundary:
      "Greedy text generation. Warm cached replay is distinct from a cold installation.",
  },
  {
    surface: "External KV tiers",
    status: "Host verified",
    detail:
      "An async transport contract executes export, restore, deletion, and failure handling against real host bytes.",
    boundary:
      "Mooncake / NIXL adapters, remote leases, and network benefit remain open.",
  },
  {
    surface: "Serving performance",
    status: "Open",
    detail:
      "Narrow same-executor results demonstrate memory savings and compiler-selection improvements.",
    boundary:
      "Recorded vLLM / SGLang comparisons remain slower. No competitive serving advantage is established.",
  },
];

export const roadmap = [
  {
    state: "01 / NEXT",
    name: "Make optimization accountable",
    detail:
      "Attribute 27B execution and search time to regions, then optimize the measured bottlenecks against independent numerical references.",
  },
  {
    state: "02 / NEXT",
    name: "Expand joint planning",
    detail:
      "Let state layouts, kernel implementations, workspace, and graph residency compete under an explicit workload and memory budget.",
  },
  {
    state: "03 / EXPLORE",
    name: "Grow the compiler vocabulary",
    detail:
      "Add KV representations and attention families as their contracts become ready. Evaluate larger fused regions and megakernels where measurements justify them.",
  },
];

export const docs = [
  {
    name: "Architecture",
    detail: "Ownership, execution, and the three-crate boundary.",
    path: "docs/architecture.md",
  },
  {
    name: "Joint compilation",
    detail: "How state plans and kernel search fit together.",
    path: "docs/joint-compilation.md",
  },
  {
    name: "Luminal design",
    detail: "Semantic operations, e-graphs, and measured CUDA search.",
    path: "docs/luminal-design.md",
  },
  {
    name: "Kernel providers",
    detail: "Capabilities, native adapters, and source setup.",
    path: "docs/attention-providers.md",
  },
  {
    name: "Checkpoint import",
    detail: "Normalize model configuration without model-name dispatch.",
    path: "docs/checkpoint-import.md",
  },
  {
    name: "RuntimeSession",
    detail: "Transactions, prefix sharing, completion, and reuse.",
    path: "docs/runtime-session.md",
  },
  {
    name: "Execution artifacts",
    detail: "Persist generated CUDA images and replay strictly.",
    path: "docs/module-artifacts.md",
  },
  {
    name: "External KV tiers",
    detail: "Move bytes while preserving one state owner.",
    path: "docs/external-kv.md",
  },
  {
    name: "Code & test layout",
    detail: "Module boundaries and tests in their owning crates.",
    path: "docs/code-layout.md",
  },
  {
    name: "Capability matrix",
    detail: "Implemented, verified, and planned surfaces.",
    path: "docs/capability-matrix.md",
  },
  {
    name: "Benchmarking",
    detail: "Matched workloads, attribution, and qualification.",
    path: "docs/benchmarking.md",
  },
  {
    name: "Roadmap",
    detail: "Current baseline and the next acceptance gates.",
    path: "docs/roadmap.md",
  },
];

export const records = [
  {
    date: "2026.09.14",
    name: "Mature attention providers",
    detail:
      "FA3 and FlashInfer selection, independent logit checks, graph lifetime, strict replay, and HTTP drain on H20.",
    path: "results/provider-kernels-20260914/README.md",
    tag: "LATEST",
  },
  {
    date: "2026.09.14",
    name: "Semantic compiler boundaries",
    detail:
      "Checkpoint normalization, portable operation contracts, and the reduced inference-only Luminal workspace.",
    path: "results/semantic-boundaries-20260914/README.md",
    tag: "ARCHITECTURE",
  },
  {
    date: "2026.09.13",
    name: "Startup preparation",
    detail:
      "Preparing selected decoder buckets before readiness, with bounded graph residency and final state drain.",
    path: "results/startup-preparation-20260913/README.md",
    tag: "RUNTIME",
  },
];
