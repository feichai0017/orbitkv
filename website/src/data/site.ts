export const repositoryUrl = "https://github.com/feichai0017/orbitkv";
export const homeUrl = "https://feichai0017.github.io/orbitkv/";

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Architecture", href: "/docs/" },
  { label: "Evidence", href: "/evidence/" },
];

export const compilerStages = [
  { key: "01", name: "Retention IR", detail: "Declare future reads for each persistent state class." },
  { key: "02", name: "Lifetime", detail: "Derive block birth, last read, and exact semantic death." },
  { key: "03", name: "Normalize", detail: "Partition state by retirement predicate, not token birth alone." },
  { key: "04", name: "Synthesize", detail: "Emit append-only or periodic address programs and minimum slots." },
  { key: "05", name: "Submit", detail: "Freeze immutable, generation-checked KV views for GPU readers." },
  { key: "06", name: "Reclaim", detail: "Reuse only after semantic death, execution completion, and commit." },
];

export const architectureLanes = [
  {
    id: "SAFE CORE / 01",
    name: "Attention-state compiler",
    detail: "Pure Rust analysis and block ownership. Unsafe code is forbidden in the core crate.",
    examples: ["LayoutProgram", "SHA-256 fingerprint", "interval slots", "continuation cuts"],
  },
  {
    id: "RUNTIME / 02",
    name: "Bitemporal block manager",
    detail: "Request identity, physical generations, immutable views, dual frontiers, and proof-carrying reclamation.",
    examples: ["ACTIVE", "RETIRING", "CERTIFIED", "FREE"],
  },
  {
    id: "NVIDIA / 03",
    name: "Cost-gated physical backends",
    detail: "Use mature paged pools by default and CUDA VMM only when stable addresses justify its granularity cost.",
    examples: ["paged", "CUDA VMM", "stable VA", "future CUDA Graph"],
  },
];

export const implemented = [
  { name: "Full retention", boundary: "append-only lifetime · no semantic retirement", state: "Implemented" },
  { name: "Sliding retention", boundary: "optimal equal-size interval coloring", state: "Implemented" },
  { name: "Owning manager", boundary: "multi-request identity · generations · immutable views", state: "Implemented" },
  { name: "Reclamation proof", boundary: "semantic proof · execution proof · backend commit", state: "Implemented" },
  { name: "SGLang adapter", boundary: "SWA chunk cache · no radix/spec/overlap", state: "H20 validated" },
  { name: "CUDA VMM slot", boundary: "stable VA · fresh physical backing · 2 MiB granularity", state: "H20 qualified" },
];

export const metrics = [
  { value: "−77.8 MiB", label: "KV pool", detail: "reported physical reduction at fixed token capacity" },
  { value: "+47.14%", label: "Full capacity", detail: "under the same 4.608 GiB KV budget" },
  { value: "−28.25%", label: "Makespan", detail: "median for eight 6K-token requests" },
  { value: "4.26×", label: "Owner transport", detail: "release plan+commit median speedup with in-process FFI" },
  { value: "64×", label: "VMM remaps", detail: "fresh backing at one stable virtual address" },
  { value: "2 MiB", label: "VMM granularity", detail: "measured H20 minimum and recommended allocation unit" },
];

export const evidenceRows = [
  {
    result: "Admission",
    value: "+47.14%",
    contract: "10 Full + 52 SWA layers, fixed 4.608 GiB KV budget",
    boundary: "dummy weights; no radix/spec/overlap/Graph",
  },
  {
    result: "Long-context makespan",
    value: "−28.25%",
    contract: "8 requests × 6000 prompt + 32 decode",
    boundary: "three fresh-process Stock/OrbitKV pairs",
  },
  {
    result: "Owning control",
    value: "1.0030x",
    contract: "in-process FFI versus JSONL sidecar with identical capacity",
    boundary: "six fresh-process alternating-order H20 pairs",
  },
  {
    result: "Owner transport",
    value: "4.26x",
    contract: "release plan + commit control-path median",
    boundary: "five host trials × 5,000 cycles; not a serving speedup",
  },
  {
    result: "CUDA VMM",
    value: "64 / 64",
    contract: "fresh backing remap and data verification at stable VA",
    boundary: "isolated physical primitive; not SGLang tensor storage",
  },
];

export const roadmap = [
  { state: "DONE", name: "In-process Rust ABI", detail: "Versioned fixed-layout certificates without JSON serialization." },
  { state: "NEXT", name: "Graph-stable KV storage", detail: "Back real SGLang KV tensors with cost-approved VMM regions." },
  { state: "THEN", name: "Richer Retention IR", detail: "Compile sink, chunk, periodic-global, and per-head lifetime classes." },
];

export const docs = [
  {
    key: "01 / DESIGN",
    name: "End-to-end plan",
    detail: "Validation stages, ownership boundary, value gates, and remaining work.",
    href: `${repositoryUrl}/blob/main/docs/sglang-e2e.md`,
  },
  {
    key: "02 / POLICY",
    name: "First H20 result",
    detail: "Memory, admission, makespan, and exact claim boundaries.",
    href: `${repositoryUrl}/blob/main/docs/h20-sglang-validation-20260817.md`,
  },
  {
    key: "03 / OWNERSHIP",
    name: "Owning and VMM evidence",
    detail: "Two-phase certificate protocol and CUDA VMM qualification.",
    href: `${repositoryUrl}/blob/main/docs/h20-owning-vmm-validation-20260817.md`,
  },
  {
    key: "04 / RECORDS",
    name: "Evidence index",
    detail: "Raw matrices, summaries, manifests, hashes, and exclusions.",
    href: `${repositoryUrl}/tree/main/results`,
  },
];
