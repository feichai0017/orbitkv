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
    name: "orbitkv-contract",
    role: "Name the state.",
    detail:
      "Framework-neutral identity, byte compatibility, page generations, and recovery bundles.",
    path: "crates/orbitkv-contract/src/lib.rs",
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
    detail: "Topology-aware CUDA, pinned-memory, and RDMA transfer paths.",
    path: "crates/orbitkv-transfer/README.md",
  },
  {
    name: "orbitkv-server",
    role: "Share the cache.",
    detail: "Versioned gRPC, process lifecycle, P/D routing, health, and metrics.",
    path: "crates/orbitkv-server/README.md",
  },
  {
    name: "orbitkv-sglang",
    role: "Integrate the runtime.",
    detail:
      "A native HiCache path first, followed by an OrbitKV-authored page and lifetime boundary.",
    path: "docs/roadmap.md",
  },
];

export const compilerStages = [
  { name: "Describe", detail: "Attention visibility and state contracts." },
  { name: "Prove", detail: "Derive semantic death and safe reuse conditions." },
  { name: "Place", detail: "Choose HBM, DRAM, SSD, or a remote replica." },
  { name: "Adapt", detail: "Re-plan from measured cost and next-touch evidence." },
];

export const providers = [
  { name: "HBM", role: "Active pages on the execution path." },
  { name: "Pinned DRAM", role: "NUMA-aware warm storage and staging." },
  { name: "SSD", role: "Durable capacity for colder prefixes." },
  { name: "RDMA", role: "Replica-aware cross-node reuse." },
];

export const docs = [
  { name: "System architecture", path: "docs/architecture.md" },
  { name: "Roadmap and validation gates", path: "docs/roadmap.md" },
  { name: "Server configuration", path: "docs/server.md" },
  { name: "Cross-node sharing", path: "docs/p2p.md" },
  { name: "P/D disaggregation", path: "docs/pd.md" },
  { name: "Metrics", path: "docs/metrics.md" },
  { name: "Implementation TODO", path: "TODO.md" },
];
