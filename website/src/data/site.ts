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
    label: "Token lifetime classes",
    detail: "Full, Sliding, Full+Sliding, and exact Chunked compile into executor plans on the host.",
  },
  {
    value: "OPEN",
    label: "Current device qualification",
    detail: "The new native engine still needs same-source model correctness, stream/event, capacity, and performance runs.",
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
    value: "Source integrated",
    contract: "Paged attention accepts external page geometry and CSR metadata without owning allocation.",
    boundary: "Host compilation does not establish accelerator correctness or throughput.",
  },
  {
    result: "Rust server boundary",
    value: "L2 contract",
    contract: "Async local Engine accepts semantic batch intent and returns ordered events.",
    boundary: "HTTP endpoints, tokenizer, scheduler, sampling, and tool orchestration remain open.",
  },
  {
    result: "Historical measurements",
    value: "Provenance only",
    contract: "Archived records preserve their exact source, model, hardware, method, and outcome.",
    boundary: "They do not qualify the current core + Luminal + server architecture.",
  },
];

export const roadmap = [
  {
    state: "NEXT",
    name: "Close one native model transaction",
    detail: "Wire RuntimeSession prepare through Luminal forward, sampling, completion, publication, and reuse.",
  },
  {
    state: "NEXT",
    name: "Build the Rust serving loop",
    detail: "Add tokenization, continuous batching, cancellation, backpressure, and OpenAI-compatible streaming above the local Engine.",
  },
  {
    state: "THEN",
    name: "Prove correctness and benefit",
    detail: "Run matched real-device model tests for each lifetime class before making capacity or performance claims.",
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
    detail: "Append-only provenance for historical experiments.",
    href: `${repositoryUrl}/blob/main/results/README.md`,
  },
];
