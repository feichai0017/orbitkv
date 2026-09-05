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
    detail: "One Full checkpoint closes prefill and decode; hybrid models and lifecycle benefit remain open.",
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
    contract: "Page generations, Prefix/COW, frontiers, retirement, ACK, relocation, and failures are host-tested.",
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
    contract: "Paged attention consumes OrbitKV page metadata; one released Full checkpoint and child-graph decode run on H20.",
    boundary: "No hybrid model, continuous-batch, or serving-throughput qualification.",
  },
  {
    result: "External KV transport",
    value: "L2 host",
    contract: "Async export, restore, deletion, checksums, partial tails, and ambiguous-failure quarantine move real host bytes.",
    boundary: "Mooncake, NIXL, remote leases, and network benefit remain open.",
  },
  {
    result: "Rust server boundary",
    value: "L2 contract",
    contract: "Async local Engine accepts semantic batch intent and returns ordered events.",
    boundary: "HTTP endpoints, tokenizer, scheduler, sampling, and tool orchestration remain open.",
  },
  {
    result: "Current device evidence",
    value: "5 compact records",
    contract: "Correctness, multi-class plumbing, failed flattened capture, and narrow child-graph benefit retain their exact environments.",
    boundary: "No compiler-lifecycle L5 or complete product comparison yet.",
  },
];

export const roadmap = [
  {
    state: "NEXT",
    name: "Close a multi-class hybrid graph",
    detail: "Build every decoder layer from its manifest class and run Full+Sliding through one Luminal runtime.",
  },
  {
    state: "NEXT",
    name: "Build the Rust serving loop",
    detail: "Connect tokenization, continuous batching, cancellation, RuntimeSession, Luminal, and OpenAI-compatible streaming.",
  },
  {
    state: "THEN",
    name: "Prove compiler and product benefit",
    detail: "Run conservative-vs-compiled ablation, then use vllm bench serve against tuned SGLang.",
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
