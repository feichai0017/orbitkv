export const docGroups = [
  {
    title: "Get started",
    description: "Install, connect an engine, and verify a cache hit.",
    items: [
      { slug: "single-node", title: "Installation & quickstart" },
      { slug: "deployment", title: "Deployment patterns" },
      { slug: "server", title: "Manager configuration" },
      { slug: "goals", title: "Capabilities & boundaries" },
    ],
  },
  {
    title: "Architecture",
    description: "Understand page ownership, cache identity, and transfers.",
    items: [
      { slug: "architecture", title: "System architecture" },
      { slug: "transport", title: "Process & network transport" },
      { slug: "state-identity", title: "State identity & recovery" },
      { slug: "state-planning", title: "Demand & transfer planning" },
      { slug: "vllm-request-state-machine", title: "vLLM request lifecycle" },
    ],
  },
  {
    title: "Distributed cache",
    description: "Explore independent replicas and prefill/decode handoff.",
    items: [
      { slug: "p2p", title: "Cross-node deployment" },
      { slug: "distributed-cache", title: "Embedded catalog design" },
      {
        slug: "distributed-comparison",
        title: "LMCache & Mooncake comparison",
      },
      { slug: "pd", title: "P/D, cache reuse & NIXL" },
      { slug: "pd-mooncake-push", title: "Experimental Mooncake P/D" },
    ],
  },
  {
    title: "Measure & operate",
    description:
      "Inspect metrics, reproduce measurements, and read their limits.",
    items: [
      { slug: "metrics", title: "Metrics & observability" },
      { slug: "single-node-performance", title: "Single-node comparisons" },
      { slug: "ssd-performance", title: "SSD recovery" },
      { slug: "concurrent-performance", title: "Concurrent query budgets" },
      { slug: "sustained-performance", title: "Sustained serving" },
    ],
  },
  {
    title: "Development",
    description: "Follow the implementation gates and contribute changes.",
    items: [
      { slug: "roadmap", title: "Roadmap & validation gates" },
      { slug: "rust-quality", title: "Rust quality gates" },
    ],
  },
];

export const docPages = docGroups.flatMap((group) =>
  group.items.map((item) => ({ ...item, group: group.title })),
);
