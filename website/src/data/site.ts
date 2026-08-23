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
    value: "12 / 3",
    label: "ABI8 H20 pairs / epochs",
    detail: "Every matched Qwen Full and GPT-OSS Full+SWA B1/B4 Prefix pair passed.",
  },
  {
    value: "6f62a23",
    label: "Sealed ABI8 source",
    detail: "Official SGLang v0.5.17 on one H20 with page16 BF16 NHD eager FA3.",
  },
  {
    value: "40 exact",
    label: "C symbols",
    detail: "Batch-only exported surface with C/C++ layout and dynamic symbol checks.",
  },
  {
    value: "+3.99–11.87%",
    label: "Observed mean overhead",
    detail: "Full B1/B4: +7.37/+11.87%; Hybrid B1/B4: +3.99/+4.45%.",
  },
  {
    value: "NO GO",
    label: "Performance qualification",
    detail: "performance_go=false; no speedup, memory-saving, or general replacement claim.",
  },
];

export const evidenceRows = [
  {
    result: "Sealed ABI8 H20 Prefix",
    value: "12 / 12 PASS",
    contract: "Qwen Full and GPT-OSS Full+SWA B1/B4 over 3 epochs; stock and manager request traces and output digests match exactly",
    boundary: "exact 6f62a23 source; official SGLang v0.5.17; one H20; page16 BF16 NHD eager FA3",
  },
  {
    result: "ABI8 Prefix / drain / SWA",
    value: "PASS / PASS / PASS",
    contract: "512-token Prefix publish and measured hits; final all-free census; every Hybrid pair advances SWA retirement, reclamation, and wrap counters",
    boundary: "Prefix path only; relocation, MLA, fixed-state, overlap, Graph, speculation, and distributed execution excluded",
  },
  {
    result: "ABI8 Full mean overhead",
    value: "+7.37 / +11.87%",
    contract: "B1 / B4 manager over matched stock across 3 epochs",
    boundary: "descriptive measurements only; performance_go=false",
  },
  {
    result: "ABI8 Hybrid mean overhead",
    value: "+3.99 / +4.45%",
    contract: "B1 / B4 manager over matched stock across 3 epochs",
    boundary: "descriptive measurements only; performance_go=false",
  },
  {
    result: "ABI8 Rust core",
    value: "L2 GO",
    contract: "Immutable snapshots, Prefix/COW, token views, Full evacuation, packed publication, reclamation",
    boundary: "host only; no engine or GPU inheritance",
  },
  {
    result: "ABI8 C wire",
    value: "L2 GO / 40",
    contract: "29 manager plus 11 state-pool symbols, C/C++ layouts, per-handle transactions, and receipt gates",
    boundary: "host wire only; manager and state-pool handles are separate, with no cross-handle atomicity or H20 inheritance",
  },
  {
    result: "ABI8 fixed-state seam",
    value: "scoped host evidence",
    contract: "Initial clear, forward event, and retire/clear/ACK on the production request-owned seam",
    boundary: "excluded from the sealed H20 record; replacement trigger, family bindings, CUDA/model/H20/performance remain pending",
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
    boundary: "historical control-plane result; not ABI8 Prefix performance",
  },
  {
    result: "Historical ABI5-v5 B4 steady",
    value: "+4.19 / −5.20%",
    contract: "Qwen Full / GPT-OSS Hybrid manager latency relative to stock",
    boundary: "one epoch; no repeated statistics; performance_go=false",
  },
  {
    result: "Historical ABI5 same-capacity memory",
    value: "0%",
    contract: "Equal page16 SGLang KV tensor capacity in manager and stock processes",
    boundary: "no compression or intrinsic same-capacity memory-win claim",
  },
  {
    result: "Unqualified ABI8 paths",
    value: "pending",
    contract: "Relocation, MLA, fixed-state, overlap, Graph, speculation, and distributed execution",
    boundary: "not covered by the sealed Prefix record; no general SGLang replacement claim",
  },
];

export const roadmap = [
  {
    state: "NEXT",
    name: "Complete fixed-state integration",
    detail: "Add the production replacement trigger and GDN, KDA, ShortConv, and linear-attention bindings, then qualify real CUDA/model/H20 behavior and performance.",
  },
  {
    state: "THEN",
    name: "Expand ABI8 H20 coverage",
    detail: "Extend the sealed Prefix-only Full and Full+SWA record to relocation, MLA, and each bound fixed-state family.",
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
    detail: "Normative live ABI8, historical ABI5, and exclusion boundary.",
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
    key: "04 / SEALED EVIDENCE",
    name: "ABI8 Prefix Full/Hybrid H20",
    detail: "Sealed exact 6f62a23; 12 matched pairs over 3 epochs; scoped correctness only.",
    href: `${repositoryUrl}/tree/main/results/h20-sglang-v0517-abi8-full-hybrid-20260823`,
  },
  {
    key: "05 / HISTORICAL EVIDENCE",
    name: "ABI5-v5 Full/Hybrid H20",
    detail: "Frozen 9233c06d scoped L4 correctness; same-cap 0%; performance not GO.",
    href: `${repositoryUrl}/tree/main/results/h20-sglang-v0517-abi5-v5-grouped-release-20260821`,
  },
  {
    key: "06 / RECORDS",
    name: "Evidence index",
    detail: "Append-only snapshots with explicit source and ABI boundaries.",
    href: `${repositoryUrl}/blob/main/results/README.md`,
  },
];
