export const repositoryUrl = "https://github.com/feichai0017/orbitkv";

export const sourceUrl = (path: string) =>
  `${repositoryUrl}/blob/${import.meta.env.PUBLIC_SOURCE_REF}/${path}`;
export const localUrl = (path: string) =>
  `${import.meta.env.BASE_URL.replace(/\/$/, "")}${path}`;

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Docs", href: "/docs/" },
  { label: "Architecture", href: "/architecture/" },
  { label: "Integration", href: "/integration/" },
];

export const layers = [
  {
    name: "orbitkv-state",
    role: "Name the state.",
    detail:
      "Versioned model/storage keys and compiled rules for demanded page ranges and leased-evidence validation, shared by SGLang and vLLM hybrid adapters.",
    path: "crates/orbitkv-state/src/lib.rs",
  },
  {
    name: "orbitkv-channel",
    role: "Control the Cache Manager.",
    detail:
      "Rust ownership of query revisions, warming and transfer waits; versioned iceoryx2 requests plus UDS lifecycle for both adapters.",
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
    name: "orbitkv-catalog",
    role: "Locate replicas.",
    detail:
      "Embedded directory shards, residency journals, candidate indexes and etcd membership; one copy per shard today.",
    path: "crates/orbitkv-catalog/src/lib.rs",
  },
  {
    name: "orbitkv-transfer",
    role: "Move the bytes.",
    detail:
      "Pinned Mooncake Transfer Engine for experimental remote cache fetch and vLLM P/D.",
    path: "crates/orbitkv-transfer/README.md",
  },
  {
    name: "Cache Manager",
    role: "Share the cache.",
    detail:
      "Node-local cache operations, pinned DRAM/SSD, health, and peer transfer control.",
    path: "crates/orbitkv-server/README.md",
  },
  {
    name: "orbitkv.sglang",
    role: "Link SGLang GPU pages.",
    detail:
      "Direct GPU recovery for full attention, sliding windows and recurrent/conv checkpoints, validated at a common token boundary.",
    path: "python/orbitkv/sglang/linker.py",
  },
  {
    name: "orbitkv.vllm",
    role: "Connect vLLM.",
    detail:
      "External KV cache connector plus a separate experimental Mooncake P/D adapter.",
    path: "python/orbitkv/vllm/connector.py",
  },
];
