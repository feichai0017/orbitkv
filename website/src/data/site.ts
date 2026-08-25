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
    detail: "Lower attention and state-retention rules into checked lifetime classes and address programs.",
  },
  {
    key: "02",
    name: "Establish ownership",
    detail: "Publish generation-checked request heads over immutable persistent roots.",
  },
  {
    key: "03",
    name: "Copy shared tails",
    detail: "Emit exact packed COW intents before extending shared or pinned partial tails.",
  },
  {
    key: "04",
    name: "Reclaim live-token state",
    detail: "Pack retained tokens and reuse pages only after semantic death, execution completion, and backend ACK.",
  },
];

export const metrics = [
  {
    value: "ABI8",
    label: "Live ownership protocol",
    detail: "Compiler, host core, typed wire, and Python runtime are actively developed and fail closed.",
  },
  {
    value: "L2 host",
    label: "Core lifecycle gates",
    detail: "Snapshots, sharing, packed COW, relocation, reclamation, stale identities, and fault paths are host-tested.",
  },
  {
    value: "SCOPED",
    label: "Engine evidence",
    detail: "Separate sealed records cover narrow Prefix and token-relocation correctness/lifecycle boundaries.",
  },
  {
    value: "PENDING",
    label: "Performance qualification",
    detail: "Published records establish no general performance, capacity, or memory-saving result.",
  },
];

export const evidenceRows = [
  {
    result: "Attention-state compiler",
    value: "L1 / L2",
    contract: "Distinct token, latent, recurrent, and convolution state contracts lower into checked ownership plans.",
    boundary: "Host/compiler evidence; unsupported semantics fail closed.",
  },
  {
    result: "Transactional ownership core",
    value: "L2 host",
    contract: "Immutable snapshots, generations, sharing, exact receipts, completion gates, and reclamation are tested together.",
    boundary: "Host and ABI status does not imply engine or production qualification.",
  },
  {
    result: "Packed fork and COW",
    value: "L2 host",
    contract: "Packed request fork and shared partial-tail copy-on-write cross the core, ABI8 wire, and Python FFI.",
    boundary: "Packed Prefix and device/model qualification remain pending.",
  },
  {
    result: "Token-level relocation",
    value: "Scoped",
    contract: "Retained logical tokens can be moved from partially dead pages under exact receipt and ACK-gated reuse.",
    boundary: "Sealed correctness/lifecycle scope is narrow; performance and broader execution modes remain unqualified.",
  },
  {
    result: "Engine-neutral adapter SPI",
    value: "L2 package",
    contract: "Typed append, copy, clear, completion, and mirror-cleanup effects with a reference external arena.",
    boundary: "The reference adapter is a contract oracle, not a serving engine.",
  },
  {
    result: "Historical evidence",
    value: "Append-only",
    contract: "Archived records retain their original source, workload, and ABI boundaries.",
    boundary: "Historical results never qualify the live ABI8 surface automatically.",
  },
];

export const roadmap = [
  {
    state: "NEXT",
    name: "Expand state ownership",
    detail: "Complete remaining fixed-state bindings and close their qualification gates.",
  },
  {
    state: "THEN",
    name: "Broaden packed execution",
    detail: "Qualify packed shared-tail COW, add packed Prefix, and validate asynchronous reclamation pressure.",
  },
  {
    state: "LATER",
    name: "Extend completion domains",
    detail: "Qualify graphs, speculation, multi-device placement, and disaggregated transfer separately.",
  },
];

export const docs = [
  {
    key: "00 / CAPABILITIES",
    name: "Capability Matrix",
    detail: "Normative live ABI8, historical ABI, and exclusion boundaries.",
    href: `${repositoryUrl}/blob/main/docs/capability-matrix.md`,
  },
  {
    key: "01 / ARCHITECTURE",
    name: "Ownership architecture",
    detail: "Compiler lowering, immutable snapshots, page ownership, COW, and reclamation invariants.",
    href: `${repositoryUrl}/blob/main/docs/standalone-kv-manager-architecture.md`,
  },
  {
    key: "02 / ADAPTER",
    name: "Engine-neutral data plane",
    detail: "Typed adapter effects and the reference external-arena contract oracle.",
    href: `${repositoryUrl}/blob/main/docs/engine-adapter-spi.md`,
  },
  {
    key: "03 / RECLAMATION",
    name: "Token virtualization",
    detail: "Token-level liveness, relocation, completion evidence, and open qualification gates.",
    href: `${repositoryUrl}/blob/main/docs/token-virtualization-and-attention-roadmap.md`,
  },
  {
    key: "04 / RECORDS",
    name: "Evidence index",
    detail: "Detailed scoped manifests and append-only historical records.",
    href: `${repositoryUrl}/blob/main/results/README.md`,
  },
];
