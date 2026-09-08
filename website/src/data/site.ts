export const repositoryUrl = "https://github.com/feichai0017/orbitkv";
export const homeUrl = "https://feichai0017.github.io/orbitkv/";

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Architecture", href: "/docs/" },
  { label: "Evidence", href: "/evidence/" },
];

export const compilerStages = [
  {
    key: "01",
    name: "Compile semantics",
    detail: "Turn attention visibility and retention rules into a fingerprinted RuntimeManifest.",
  },
  {
    key: "02",
    name: "Prepare state",
    detail: "RuntimeSession selects pages, generations, Prefix/COW actions, and retirement rules.",
  },
  {
    key: "03",
    name: "Lower execution",
    detail: "ExecutorPlan converts manager-owned views into Luminal attention metadata and physical writes.",
  },
  {
    key: "04",
    name: "Execute locally",
    detail: "Luminal runs the model graph and records completion on the owning device stream.",
  },
  {
    key: "05",
    name: "Reuse with proof",
    detail: "OrbitKV publishes new heads and recycles generations only after semantic death, completion, and ACK.",
  },
];

export const metrics = [
  {
    value: "3",
    label: "Native layers",
    detail: "Rust core, forked Luminal executor, and Rust server form the only active product architecture.",
  },
  {
    value: "1",
    label: "KV authority",
    detail: "RuntimeSession alone selects, retires, acknowledges, and reuses physical page generations.",
  },
  {
    value: "4",
    label: "Compiled lifetime profiles",
    detail: "Full, Sliding, Full+Sliding, and exact Chunked compile into lifecycle and executor plans on the host.",
  },
  {
    value: "L4 narrow",
    label: "Model qualification",
    detail: "Qwen2.5 Full and Gemma 3 Full+Sliding checkpoints have narrow H20 correctness closures.",
  },
];

export const evidenceRows = [
  {
    result: "Attention-state compiler",
    value: "L1 + L2",
    contract: "Typed attention state and Retention IR compile into deterministic manifests and physical lifetime plans.",
    boundary: "Compiler output alone is not device or performance evidence.",
  },
  {
    result: "RuntimeSession",
    value: "L2 host",
    contract: "Page generations, Prefix/COW, frontiers, retirement, ACK, external transfer, and failures are host-tested.",
    boundary: "Raw completion fields are not yet authenticated against the integrated device executor.",
  },
  {
    result: "Executor lowering",
    value: "L2 host",
    contract: "Full, Sliding, mixed, and Chunked manifests lower to manager-owned page metadata.",
    boundary: "Latent and fixed-state execution are rejected until their kernels and transactions are integrated.",
  },
  {
    result: "Luminal fork",
    value: "L3 + narrow L4",
    contract: "One bucketed decoder graph consumes OrbitKV page metadata; Full and Full+Sliding checkpoints run on H20.",
    boundary: "Paged attention is currently a FlashInfer custom op, not a multi-backend compiler choice.",
  },
  {
    result: "Residence ablation",
    value: "narrow L5",
    contract: "Released-hybrid paired runs preserve outputs while reducing live payload and increasing fixed-budget reach.",
    boundary: "Batch-one same-executor result; no SGLang-relative serving win.",
  },
  {
    result: "External KV transport",
    value: "L2 host",
    contract: "Async export, restore, deletion, checksums, partial tails, and ambiguous-failure quarantine move real host bytes.",
    boundary: "Mooncake, NIXL, remote leases, and network benefit remain open.",
  },
  {
    result: "Rust server boundary",
    value: "narrow L4",
    contract: "The single-process engine serves OpenAI completions, batches requests, cancels streams, and drains state.",
    boundary: "Greedy text-only; fairness, soak, capacity limit, and competitive performance remain open.",
  },
  {
    result: "Current device evidence",
    value: "11 indexed records",
    contract: "Released-checkpoint correctness, lifecycle, HTTP serving, compiler attribution, and matched product comparisons retain their exact environments.",
    boundary: "Historical measurements qualify only their recorded source closure; the current SGLang comparison remains negative.",
  },
];

export const roadmap = [
  {
    state: "NEXT",
    name: "Run the primary hybrid model",
    detail: "Integrate recurrent and convolution state, GDN execution, partial RoPE, then block-FP8 loading for the 27B target.",
  },
  {
    state: "NEXT",
    name: "Make attention searchable",
    detail: "Offer FlashInfer and a Luminal-native CUDA candidate behind one semantic paged-attention op.",
  },
  {
    state: "THEN",
    name: "Beat both reference engines",
    detail: "Use one matched client and require higher throughput without worse p95 TTFT or TPOT than vLLM and SGLang.",
  },
];

export const docs = [
  {
    key: "00 / ARCHITECTURE",
    name: "Three-layer engine",
    detail: "The ownership split between core, executor, and server.",
    href: `${repositoryUrl}/blob/main/docs/architecture.md`,
  },
  {
    key: "01 / CAPABILITIES",
    name: "Capability matrix",
    detail: "Implemented, host-tested, device, engine, benefit, and production boundaries.",
    href: `${repositoryUrl}/blob/main/docs/capability-matrix.md`,
  },
  {
    key: "02 / RUNTIME",
    name: "RuntimeSession",
    detail: "Transactional KV ownership, Prefix/COW, completion, and safe reuse.",
    href: `${repositoryUrl}/blob/main/docs/runtime-session.md`,
  },
  {
    key: "03 / LIFETIMES",
    name: "State lifetime",
    detail: "Semantic and execution frontiers across heterogeneous attention.",
    href: `${repositoryUrl}/blob/main/docs/state-lifecycle.md`,
  },
  {
    key: "04 / ROADMAP",
    name: "Qualification roadmap",
    detail: "The shortest path to a complete measured native engine.",
    href: `${repositoryUrl}/blob/main/docs/roadmap.md`,
  },
  {
    key: "05 / RECORDS",
    name: "Evidence index",
    detail: "Compact evidence that directly qualifies the current architecture.",
    href: `${repositoryUrl}/blob/main/results/README.md`,
  },
  {
    key: "06 / COMPONENTS",
    name: "Components and dependencies",
    detail: "Owned boundaries and the precise roles of Luminal, vLLM, PegaInfer, Dynamo, Mooncake, and NIXL.",
    href: `${repositoryUrl}/blob/main/docs/components.md`,
  },
  {
    key: "07 / STATUS",
    name: "Implementation status",
    detail: "Compiler, manager, executor, model, and benefit support without overclaiming.",
    href: `${repositoryUrl}/blob/main/docs/implementation-status.md`,
  },
  {
    key: "08 / BENCHMARKS",
    name: "Matched serving benchmarks",
    detail: "Common-client compiler ablation and OrbitKV-versus-SGLang methodology.",
    href: `${repositoryUrl}/blob/main/docs/benchmarking.md`,
  },
];
