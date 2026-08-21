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
    detail: "Turn Full and retained attention relations into checked classes and address programs.",
  },
  {
    key: "02",
    name: "Publish snapshots",
    detail: "Keep generation-checked request heads over immutable persistent class roots.",
  },
  {
    key: "03",
    name: "Share safely",
    detail: "Aggregate request and Prefix references; copy shared partial tails before append.",
  },
  {
    key: "04",
    name: "Reclaim with proof",
    detail: "Reuse a page only after refs, readers, writers, mirrors, and backend ACKs are discharged.",
  },
];

export const metrics = [
  {
    value: "ABI6 / L2",
    label: "Host core + C wire",
    detail: "Snapshots, Prefix, fork, joint COW, and reclamation pass host gates.",
  },
  {
    value: "23 exact",
    label: "C symbols",
    detail: "Batch-only exported surface with C/C++ layout and dynamic symbol checks.",
  },
  {
    value: "20 → 5",
    label: "Historical B4 release",
    detail: "Frozen ABI5-v5 grouped 20 request releases into five transactions.",
  },
  {
    value: "0%",
    label: "Same-cap memory",
    detail: "Historical stock/manager pairs reserve identical KV tensor capacity.",
  },
  {
    value: "NO GO",
    label: "General performance",
    detail: "One ABI5 epoch is insufficient for a general latency or throughput claim.",
  },
];

export const evidenceRows = [
  {
    result: "ABI6 Rust core",
    value: "L2 GO",
    contract: "Immutable snapshots, request fork, Prefix refs, joint Full+SWA COW, page-owned reclamation",
    boundary: "host only; no engine or GPU inheritance",
  },
  {
    result: "ABI6 C wire",
    value: "L2 GO / 23",
    contract: "Exact symbol set, C/C++ layouts, batch atomicity, short-buffer and malformed-receipt gates",
    boundary: "host wire only; no H20 Prefix inheritance",
  },
  {
    result: "ABI6 Python/SGLang",
    value: "L2 GO",
    contract: "Split FFI/runtime/plugin, official OrbitKVPrefixCache seam, joint COW and collective cleanup",
    boundary: "host-qualified only; no H20 Prefix record",
  },
  {
    result: "Frozen ABI5-v5 H20",
    value: "historical L4",
    contract: "Qwen Full and GPT-OSS Full+SWA B1/B4 token-exact correctness and all-free drain",
    boundary: "exact 9233c06d source; official v0.5.17; one H20; excluded features disabled",
  },
  {
    result: "ABI5-v5 grouped release",
    value: "20 → 5",
    contract: "Twenty B4 request releases through five release/recycle transactions",
    boundary: "historical control-plane result; not ABI6 Prefix performance",
  },
  {
    result: "ABI5-v5 B4 steady",
    value: "+4.19 / −5.20%",
    contract: "Qwen Full / GPT-OSS Hybrid manager latency relative to stock",
    boundary: "one epoch; no repeated statistics; performance_go=false",
  },
  {
    result: "Same-capacity memory",
    value: "0%",
    contract: "Equal page16 SGLang KV tensor capacity in manager and stock processes",
    boundary: "no compression or intrinsic same-capacity memory-win claim",
  },
  {
    result: "Relocation and Graph",
    value: "pending",
    contract: "Token-exact placement moves and multiple completion domains",
    boundary: "future milestones; no projected vToken or Graph benefit",
  },
];

export const roadmap = [
  {
    state: "NEXT",
    name: "Run ABI6 H20 Prefix",
    detail: "Qualify warm-hit sharing, grouped release, pressure eviction and COW against the exact frozen source.",
  },
  {
    state: "THEN",
    name: "Token-exact relocation",
    detail: "Move retained K/V bytes against one immutable snapshot; this is not numerical compression.",
  },
  {
    state: "LATER",
    name: "Graph and distributed",
    detail: "Qualify completion domains, speculation, multi-GPU placement, and disaggregated transfer.",
  },
];

export const docs = [
  {
    key: "00 / CAPABILITIES",
    name: "Capability Matrix",
    detail: "Normative live ABI6, historical ABI5, and exclusion boundary.",
    href: `${repositoryUrl}/blob/main/docs/capability-matrix.md`,
  },
  {
    key: "01 / ARCHITECTURE",
    name: "Standalone manager",
    detail: "Module ownership, immutable snapshots, Prefix/COW, and reclamation invariants.",
    href: `${repositoryUrl}/blob/main/docs/standalone-kv-manager-architecture.md`,
  },
  {
    key: "02 / MIGRATION",
    name: "ABI5 to ABI6",
    detail: "Historical SGLang evidence and the breaking adapter migration boundary.",
    href: `${repositoryUrl}/blob/main/docs/abi5-sglang-batch-adapter.md`,
  },
  {
    key: "03 / ROADMAP",
    name: "Token virtualization",
    detail: "H20 Prefix qualification, exact relocation, Graph, speculation, and distribution.",
    href: `${repositoryUrl}/blob/main/docs/token-virtualization-and-attention-roadmap.md`,
  },
  {
    key: "04 / HISTORICAL EVIDENCE",
    name: "ABI5-v5 Full/Hybrid H20",
    detail: "Frozen 9233c06d scoped L4 correctness; same-cap 0%; performance not GO.",
    href: `${repositoryUrl}/tree/main/results/h20-sglang-v0517-abi5-v5-grouped-release-20260821`,
  },
  {
    key: "05 / RECORDS",
    name: "Evidence index",
    detail: "Append-only snapshots with explicit source and ABI boundaries.",
    href: `${repositoryUrl}/blob/main/results/README.md`,
  },
];
