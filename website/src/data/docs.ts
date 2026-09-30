export const docGroups = [
  {
    title: "Getting started",
    description: "Install, connect an engine, and verify a cache hit.",
    items: [
      { slug: "goals", title: "Overview & supported features" },
      { slug: "single-node", title: "Installation & quickstart" },
      { slug: "adapters", title: "vLLM & SGLang integration" },
      { slug: "models", title: "Model qualification" },
      { slug: "deployment", title: "Deployment patterns" },
      { slug: "server", title: "Manager configuration" },
    ],
  },
  {
    title: "Architecture",
    description: "Understand page ownership, cache identity, and transfers.",
    items: [
      { slug: "architecture", title: "System architecture" },
      { slug: "transport", title: "Process & network transport" },
      { slug: "engine-local-restore", title: "Engine-local restore ownership" },
      { slug: "state-identity", title: "State identity & recovery" },
      { slug: "hybrid-recovery", title: "Compiled hybrid recovery" },
      { slug: "state-planning", title: "Demand & transfer planning" },
      { slug: "queued-warming", title: "Queued-request warming" },
      { slug: "request-preparation", title: "Consumer-owned preparation" },
      { slug: "cache-policies", title: "Retention & SSD write admission" },
      { slug: "gds", title: "GPU storage" },
      { slug: "storage-formats", title: "GPU quantization & compression" },
      { slug: "vllm-request-state-machine", title: "vLLM request lifecycle" },
    ],
  },
  {
    title: "Distributed cache",
    description: "Explore independent replicas and prefill/decode handoff.",
    items: [
      { slug: "p2p", title: "Cross-node deployment" },
      { slug: "peer-control", title: "Peer control boundary" },
      { slug: "shared-cache-qualification", title: "Shared-cache qualification" },
      { slug: "distributed-cache", title: "Distributed global index" },
      {
        slug: "distributed-comparison",
        title: "LMCache, FlexKV & Mooncake comparison",
      },
      { slug: "pd", title: "P/D, cache reuse & NIXL" },
      { slug: "pd-mooncake-push", title: "Retired P/D protocol" },
    ],
  },
  {
    title: "Benchmarks & operations",
    description:
      "Inspect metrics, reproduce measurements, and read their limits.",
    items: [
      { slug: "metrics", title: "Metrics & observability" },
      { slug: "fault-qualification", title: "Single-node fault gates" },
      { slug: "client-performance", title: "Client control overhead" },
      { slug: "benchmark-evidence", title: "Benchmark evidence & archives" },
      { slug: "communication-performance", title: "Communication measurements" },
      { slug: "single-node-performance", title: "Single-node comparisons" },
      { slug: "ssd-performance", title: "SSD recovery & capacity pressure" },
      { slug: "concurrent-performance", title: "Concurrent query budgets" },
      { slug: "sustained-performance", title: "Sustained serving" },
      { slug: "recovery-performance", title: "Recovery stage profile" },
    ],
  },
  {
    title: "Development",
    description: "Follow the implementation gates and contribute changes.",
    items: [
      { slug: "completion-plan", title: "Completion stages & acceptance" },
      { slug: "engine-release-audit", title: "Released engine interface audit" },
      { slug: "rust-quality", title: "Rust quality gates" },
      { slug: "releases", title: "Python releases" },
    ],
  },
];

export const docPages = docGroups.flatMap((group) =>
  group.items.map((item) => ({ ...item, group: group.title })),
);
