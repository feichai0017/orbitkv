export const repositoryUrl = "https://github.com/feichai0017/orbitkv";
export const homeUrl = "https://feichai0017.github.io/orbitkv/";

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Architecture", href: "/docs/" },
  { label: "Evidence", href: "/evidence/" },
];

export const compilerStages = [
  { key: "01", name: "Retention IR", detail: "Declare affine may_read(query, key) relations for persistent state." },
  { key: "02", name: "Lifetime", detail: "Prove unbounded or fixed-window death from query-key distance constraints." },
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
  { name: "HF config frontend", boundary: "explicit Full/SWA layer_types · KV geometry derivation · fail-closed fallback", state: "Implemented" },
  { name: "Retention IR", boundary: "affine q/k AST · difference bounds · sink/local partition · safe fallback", state: "Implemented" },
  { name: "Full retention", boundary: "append-only lifetime · no semantic retirement", state: "Implemented" },
  { name: "Sliding retention", boundary: "optimal equal-size interval coloring", state: "Implemented" },
  { name: "Sink + sliding", boundary: "one may_read OR relation · pinned + periodic regions · host exhaustive proof", state: "Implemented" },
  { name: "Physical optimizer", boundary: "KV budget · admission waves · reclaim budget · engine contract", state: "H20 validated" },
  { name: "Owning manager", boundary: "multi-request identity · generations · immutable views", state: "Implemented" },
  { name: "Reclamation proof", boundary: "semantic proof · execution proof · backend commit", state: "Implemented" },
  { name: "SGLang adapter", boundary: "SWA chunk cache · no radix/spec/overlap", state: "H20 validated" },
  { name: "CUDA VMM slot", boundary: "stable VA · fresh physical backing · 2 MiB granularity", state: "H20 qualified" },
];

export const metrics = [
  { value: "q-k < W", label: "Retention IR", detail: "declarative relation lowered to window, death, and periodic cells" },
  { value: "+25.81%", label: "Full capacity", detail: "real gpt-oss-20b, same 1.979 GiB KV budget" },
  { value: "−20.30%", label: "Owner vs Stock", detail: "balanced four-way real-checkpoint ablation" },
  { value: "+2.18%", label: "Owner cost", detail: "Owner32 versus Policy32 on the pressure workload" },
  { value: "4 / 4", label: "Plan predictions", detail: "16/32/64/128 Full and SWA capacities matched SGLang" },
  { value: "−288 MiB", label: "Fixed-capacity KV", detail: "same 47,616 Full tokens; median Owner/Stock 0.9992x" },
  { value: "4.26×", label: "Owner transport", detail: "release plan+commit median speedup with in-process FFI" },
  { value: "64×", label: "VMM remaps", detail: "fresh backing at one stable virtual address" },
  { value: "63 / 63", label: "Stale generations", detail: "old physical handles rejected after VMM slot reuse" },
  { value: "2 MiB", label: "VMM granularity", detail: "measured H20 minimum and recommended allocation unit" },
];

export const evidenceRows = [
  {
    result: "Real checkpoint capacity",
    value: "+25.81%",
    contract: "openai/gpt-oss-20b, Full capacity 47,616 to 59,904",
    boundary: "same 1.979 GiB KV budget; no radix/spec/overlap/Graph",
  },
  {
    result: "Real checkpoint makespan",
    value: "−20.30%",
    contract: "Owner32 vs Stock128, four balanced execution orders",
    boundary: "8 x 6000 prompt + 32 decode; identical output-token digests",
  },
  {
    result: "Physical policy attribution",
    value: "−21.72%",
    contract: "manual Stock32 versus unmodified Stock128",
    boundary: "capacity gain is reproducible without the OrbitKV plugin",
  },
  {
    result: "Proof-carrying owner cost",
    value: "+2.18%",
    contract: "Owner32 versus Policy32 with the same compiled plan and capacity",
    boundary: "pressure workload only; outputs identical",
  },
  {
    result: "Physical-plan synthesis",
    value: "32 tokens",
    contract: "selected from 16/32/64/128 under admission and reclaim-call constraints",
    boundary: "non-overlap SWA ChunkCache contract",
  },
  {
    result: "Capacity prediction",
    value: "4 / 4",
    contract: "predicted Full/SWA pools matched fresh SGLang processes exactly",
    boundary: "gpt-oss-20b, one H20, recorded 1.979 GiB KV budget",
  },
  {
    result: "Real fixed-capacity control",
    value: "−288 MiB",
    contract: "4 requests x 4096 prompt + 64 decode, same Full capacity",
    boundary: "median Owner/Stock 0.9992x; not a speedup claim",
  },
  {
    result: "Sink + Sliding synthesis",
    value: "2 regions",
    contract: "one may_read OR relation to pinned sink plus periodic local cells",
    boundary: "host exhaustive proof; SGLang partition lowering not yet enabled",
  },
  {
    result: "Fixture admission",
    value: "+47.14%",
    contract: "10 Full + 52 SWA layers, fixed 4.608 GiB KV budget",
    boundary: "dummy weights; no radix/spec/overlap/Graph",
  },
  {
    result: "Fixture makespan",
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
  {
    result: "Generation lifecycle",
    value: "64 / 64",
    contract: "temporal cell cycle, CUDA Event, VMM receipt, manager commit",
    boundary: "closed-loop core/backend test; not SGLang tensor storage",
  },
];

export const roadmap = [
  { state: "DONE", name: "In-process Rust ABI", detail: "Versioned fixed-layout certificates without JSON serialization." },
  { state: "DONE", name: "HF model frontend", detail: "Compile explicit Full/SWA layer types and KV geometry into Retention IR." },
  { state: "DONE", name: "Physical-plan optimizer", detail: "Select a constrained SGLang policy from budget and workload candidates." },
  { state: "DONE", name: "Lifetime partitioning", detail: "Compile sink plus local semantics into pinned and periodic block regions." },
  { state: "NEXT", name: "Graph-stable KV storage", detail: "Back real SGLang KV tensors with cost-approved VMM regions." },
  { state: "THEN", name: "Richer Retention IR", detail: "Compile chunk, periodic-global, and per-head lifetime classes." },
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
  {
    key: "05 / NORMALIZE",
    name: "Sink + Sliding proof",
    detail: "Exhaustive retention equivalence, periodic-cell safety, and adapter boundary.",
    href: `${repositoryUrl}/blob/main/results/sink-sliding-20260817/summary.json`,
  },
  {
    key: "06 / REAL MODEL",
    name: "gpt-oss-20b validation",
    detail: "Released checkpoint capacity, admission, overhead, and reclamation certificates.",
    href: `${repositoryUrl}/blob/main/docs/h20-gpt-oss-20b-real-validation-20260817.md`,
  },
];
