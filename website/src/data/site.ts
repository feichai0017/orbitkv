export const repositoryUrl = "https://github.com/feichai0017/orbitkv";

export const sourceUrl = (path: string) =>
  `${repositoryUrl}/blob/${import.meta.env.PUBLIC_SOURCE_REF}/${path}`;
export const localUrl = (path: string) =>
  `${import.meta.env.BASE_URL.replace(/\/$/, "")}${path}`;

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Architecture", href: "/docs/" },
  { label: "Integration", href: "/integration/" },
];

export const layers = [
  {
    name: "orbitkv-state",
    role: "Name the state.",
    detail:
      "State identity, page, and recovery types; full validation is not yet in the cache hot path.",
    path: "crates/orbitkv-state/src/lib.rs",
  },
  {
    name: "orbitkv-channel",
    role: "Control the Cache Manager.",
    detail: "Versioned iceoryx2 requests plus UDS bootstrap and lifecycle for both adapters.",
    path: "crates/orbitkv-channel/src/lib.rs",
  },
  {
    name: "orbitkv-core",
    role: "Own the blocks.",
    detail:
      "Content-addressed KV blocks, leases, admission, eviction, and tier coordination.",
    path: "crates/orbitkv-core/src/lib.rs",
  },
  {
    name: "orbitkv-transfer",
    role: "Move the bytes.",
    detail: "Pinned Mooncake Transfer Engine for experimental remote cache fetch and vLLM P/D.",
    path: "crates/orbitkv-transfer/README.md",
  },
  {
    name: "Cache Manager",
    role: "Share the cache.",
    detail: "Node-local cache operations, pinned DRAM/SSD, health, and peer transfer control.",
    path: "crates/orbitkv-server/README.md",
  },
  {
    name: "orbitkv.sglang",
    role: "Link SGLang GPU pages.",
    detail:
      "A direct linker for full-attention MHA/MLA models; hybrid state is not yet supported.",
    path: "python/orbitkv/sglang/linker.py",
  },
  {
    name: "orbitkv.vllm",
    role: "Connect vLLM.",
    detail: "External KV cache connector plus a separate experimental Mooncake P/D adapter.",
    path: "python/orbitkv/vllm/connector.py",
  },
];

export const plannerStages = [
  { name: "Describe", detail: "Attention visibility and state contracts." },
  { name: "Prove", detail: "Derive semantic death and safe reuse conditions." },
  { name: "Place", detail: "Choose HBM, DRAM, SSD, or a remote replica." },
  { name: "Adapt", detail: "Re-plan from measured cost and next-touch evidence." },
];

export const providers = [
  { name: "HBM", role: "Active pages allocated and scheduled by the inference engine." },
  { name: "Pinned DRAM", role: "NUMA-aware warm storage and staging." },
  { name: "SSD", role: "Optional backing for colder prefixes." },
  { name: "Remote", role: "Experimental Mooncake fetch over RDMA or TCP." },
];

export const docs = [
  { name: "Single-node vLLM and SGLang", path: "docs/single-node.md" },
  { name: "Model-aware state identity", path: "docs/state-identity.md" },
  { name: "System architecture", path: "docs/architecture.md" },
  { name: "Local and remote transport", path: "docs/transport.md" },
  { name: "Roadmap and validation gates", path: "docs/roadmap.md" },
  { name: "Server configuration", path: "docs/server.md" },
  { name: "Cross-node sharing", path: "docs/p2p.md" },
  { name: "P/D disaggregation", path: "docs/pd.md" },
  { name: "Deployment examples", path: "docs/deployment.md" },
  { name: "Metrics", path: "docs/metrics.md" },
  { name: "Implementation TODO", path: "TODO.md" },
];
