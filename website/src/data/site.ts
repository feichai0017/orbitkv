export const repositoryUrl = "https://github.com/feichai0017/orbitkv";

// Keep deployed documentation on the same source revision as the website.
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
    role: "Own the state.",
    detail:
      "Compile attention lifetimes. Manage pages, prefix sharing, and safe reuse. Use the core independently.",
    path: "crates/orbitkv/README.md",
  },
  {
    name: "orbitkv-compiler",
    role: "Explore equivalent programs.",
    detail: "Build symbolic tensor graphs and legal implementation search spaces.",
    path: "crates/orbitkv-compiler/README.md",
  },
  {
    name: "orbitkv-ops",
    role: "Describe the computation.",
    detail: "Express inference semantics independently of CUDA providers.",
    path: "crates/orbitkv-ops/README.md",
  },
  {
    name: "orbitkv-cuda",
    role: "Measure and execute.",
    detail: "Compile kernels, compare legal candidates on the GPU, and execute selected programs.",
    path: "crates/orbitkv-cuda/README.md",
  },
  {
    name: "orbitkv-tracing",
    role: "Make costs visible.",
    detail: "Record compiler stages, search decisions, and execution measurements.",
    path: "crates/orbitkv-tracing/README.md",
  },
  {
    name: "orbitkv-executor",
    role: "Compile the work.",
    detail:
      "Import model graphs, bind state arenas, profile CUDA candidates, and save execution artifacts.",
    path: "crates/orbitkv-executor/README.md",
  },
  {
    name: "orbitkv-engine",
    role: "Serve the model.",
    detail:
      "Schedule batches, stream tokens, and coordinate cancellation through an optional OpenAI-compatible frontend.",
    path: "crates/orbitkv-engine/README.md",
  },
];

export const compilerStages = [
  { name: "Describe", detail: "Model math and state contracts." },
  { name: "Compile", detail: "Admit compatible graph implementations." },
  { name: "Measure", detail: "Profile candidates on the target GPU." },
  { name: "Replay", detail: "Load the selected schedule and CUDA images." },
];

export const providers = [
  { name: "cuBLASLt", role: "Dense and batched matrix products." },
  { name: "DeepGEMM", role: "SM90 block-scaled FP8 linear." },
  { name: "FlashInfer", role: "Paged decode and packed prefill." },
  { name: "FlashAttention-3", role: "Optional SM90 F16/BF16 paged attention." },
];

export const evidenceHighlights = [
  {
    value: "27B",
    label: "Hybrid text decoder",
    detail: "Qwen3.8 block-FP8 on NVIDIA H20.",
  },
  {
    value: "144",
    label: "Final-binary comparisons",
    detail:
      "Full vocabulary at B1 and B8 after workspace integration.",
  },
  {
    value: "7",
    label: "Compiled workload buckets",
    detail: "Explicit request geometry through search and retained replay.",
  },
];

export const docs = [
  { name: "System architecture", path: "docs/architecture.md" },
  { name: "Model compiler", path: "docs/compiler.md" },
  { name: "CUDA backend and providers", path: "docs/cuda-backend.md" },
  { name: "State lifecycle", path: "docs/runtime-session.md" },
  { name: "Execution artifacts", path: "docs/module-artifacts.md" },
  { name: "Joint compilation", path: "docs/joint-compilation.md" },
  { name: "Models on one H20", path: "docs/model-targets.md" },
  { name: "Code and test layout", path: "docs/code-layout.md" },
];

export const records = [
  {
    date: "2026-09-14",
    name: "CUDA backend and provider builds",
    detail:
      "One provider lock, device-aware compilation, and 152 passing B1/B8 reference comparisons on H20.",
    path: "results/cuda-backend-refactor-20260914/README.md",
  },
  {
    date: "2026-09-14",
    name: "Integrated compiler workspace",
    detail: "Seven owned crates. The final binary passes 144 B1/B8 reference comparisons and state drains.",
    path: "results/workspace-integration-20260914/README.md",
  },
  {
    date: "2026-09-14",
    name: "State checks before CUDA compilation",
    detail:
      "296 logit comparisons pass. Invalid candidates are cheaper to reject; warmed decode remains close.",
    path: "results/state-preflight-20260914/README.md",
  },
  {
    date: "2026-09-14",
    name: "Search coverage and limits",
    detail:
      "Stable snapshot sampling and 56 measured graphs. Correctness passes; performance remains mixed.",
    path: "results/search-coverage-20260914/README.md",
  },
  {
    date: "2026-09-14",
    name: "Compiler and runtime attribution",
    detail:
      "B1/B8 replay, request geometry, and measured search and execution costs.",
    path: "results/workload-attribution-20260914/README.md",
  },
  {
    date: "2026-09-14",
    name: "Attention provider qualification",
    detail:
      "CUDA selection, logit parity, artifact replay, and HTTP state drain.",
    path: "results/provider-kernels-20260914/README.md",
  },
  {
    date: "2026-09-14",
    name: "Semantic compiler boundaries",
    detail:
      "Model normalization, portable operations, and the inference-only fork.",
    path: "results/semantic-boundaries-20260914/README.md",
  },
  {
    date: "2026-09-13",
    name: "Startup preparation",
    detail:
      "Bucket preparation, bounded graph residency, and final state drain.",
    path: "results/startup-preparation-20260913/README.md",
  },
];
