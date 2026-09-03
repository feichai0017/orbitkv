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
    name: "Build RuntimeManifest",
    detail: "Compile declarative attention and retention semantics into canonical plans, requirements, and a content fingerprint.",
  },
  {
    key: "02",
    name: "Load RuntimeTarget",
    detail: "Read the pinned SGLang execution contract, admitted topologies, and required wire version.",
  },
  {
    key: "03",
    name: "Derive RuntimeBinding",
    detail: "Bind the exact manifest and target fingerprints to a complete structural execution signature.",
  },
  {
    key: "04",
    name: "Check the live boundary",
    detail: "Require wire 14, all 48 exports, all 78 layouts, and dynamic engine checks to agree before mutation.",
  },
  {
    key: "05",
    name: "Execute in RuntimeSession",
    detail: "Publish immutable request heads, copy shared tails, and reclaim only after semantic death, execution completion, and ACK.",
  },
];

export const metrics = [
  {
    value: "WIRE 14",
    label: "Live RuntimeSession contract",
    detail: "The pinned SGLang bridge, native library, runtime target, and binding must agree on the current wire or fail closed.",
  },
  {
    value: "48",
    label: "Exact live exports",
    detail: "The typed RuntimeSession boundary exports exactly 48 symbols; the former raw manager ABI is absent.",
  },
  {
    value: "78",
    label: "Checked wire layouts",
    detail: "Rust, C, and Python agree on 78 exact layouts before the native session can be used.",
  },
  {
    value: "NO-GO",
    label: "Live real-device qualification",
    detail: "Current-wire Full+Sliding passed correctness and stream/reuse diagnostics but failed the frozen throughput gate; capacity and sealed qualification remain open.",
  },
];

export const evidenceRows = [
  {
    result: "Single complete product",
    value: "Pinned SGLang + OrbitKV",
    contract: "The complete pinned SGLang source and the OrbitKV native RuntimeSession are maintained, assembled, and tested as one product.",
    boundary: "SGLang remains the execution engine; RuntimeSession is the sole KV lifecycle and physical-page-selection authority on admitted routes.",
  },
  {
    result: "Native RuntimeSession route",
    value: "Sole live route",
    contract: "Manifest admission, request identity, Prefix/COW decisions, completion evidence, retirement, ACK, and reuse flow through one opaque native session.",
    boundary: "The raw manager ABI, optional structured path, reference product, and second-engine path have been deleted.",
  },
  {
    result: "Typed wire boundary",
    value: "14 · 48 · 78",
    contract: "Wire version 14 exposes exactly 48 symbols and checks 78 Rust/C/Python layouts as one closed contract.",
    boundary: "Version, symbol-set, layout, target, or binding disagreement fails before allocation or execution.",
  },
  {
    result: "Compiler artifacts",
    value: "Manifest · Target · Binding",
    contract: "A canonical manifest is admitted against the pinned SGLang target and bound by exact fingerprints.",
    boundary: "Static admission does not replace live engine checks or real-device qualification.",
  },
  {
    result: "Transactional ownership core",
    value: "L2 host",
    contract: "Immutable snapshots, generations, sharing, exact receipts, completion gates, and reclamation are tested together.",
    boundary: "Host qualification does not imply accelerator, performance, capacity, or production qualification.",
  },
  {
    result: "Native profile coverage",
    value: "Five host-tested profiles",
    contract: "Full, ordered Full+Sliding, pure Sliding, exact Chunked, and request-private Full latent state run through RuntimeSession under their admitted constraints.",
    boundary: "Each profile keeps its own sharing, lifecycle, and qualification boundary; support does not transfer between profiles.",
  },
  {
    result: "Historical Full accelerator evidence",
    value: "Earlier wire · correctness",
    contract: "Recorded Full runs passed their scoped correctness and lifecycle checks on the source and workload captured in their provenance.",
    boundary: "They reported no speedup and do not qualify the live wire-14 product.",
  },
  {
    result: "Full+Sliding real-device evidence",
    value: "Pending",
    contract: "The ordered Full+Sliding profile has native RuntimeSession host coverage.",
    boundary: "Live wire-14 real-device correctness and performance qualification remain pending.",
  },
  {
    result: "Pure Sliding real-device evidence",
    value: "Pending",
    contract: "The request-private pure Sliding profile has native RuntimeSession host coverage.",
    boundary: "Live wire-14 real-device correctness and performance qualification remain pending.",
  },
  {
    result: "Archived scoped evidence",
    value: "Append-only provenance",
    contract: "Historical records retain their exact source, workload, environment, and interface boundaries.",
    boundary: "Provenance-specific device or model names describe only those records and never name the general architecture.",
  },
];

export const roadmap = [
  {
    state: "NEXT",
    name: "Qualify Full+Sliding on the live product",
    detail: "Close wire-14 real-device correctness, lifecycle, and independent evidence gates for the ordered mixed-retention profile.",
  },
  {
    state: "NEXT",
    name: "Qualify pure Sliding on the live product",
    detail: "Close the same live real-device gates for request-private bounded retention without borrowing Full evidence.",
  },
  {
    state: "THEN",
    name: "Broaden qualification independently",
    detail: "Qualify additional state families, completion domains, scheduling modes, capacity, and performance as separate claims.",
  },
];

export const docs = [
  {
    key: "00 / CAPABILITIES",
    name: "Capability matrix",
    detail: "Normative product, wire, host, real-device, historical, and exclusion boundaries.",
    href: `${repositoryUrl}/blob/main/docs/capability-matrix.md`,
  },
  {
    key: "01 / PRODUCT",
    name: "Complete product boundary",
    detail: "Complete pinned SGLang source, OrbitKV native RuntimeSession, source assembly, and ownership split.",
    href: `${repositoryUrl}/blob/main/compat/README.md`,
  },
  {
    key: "02 / RUNTIME",
    name: "RuntimeSession architecture",
    detail: "Compiler lowering, immutable snapshots, physical-page ownership, COW, and reclamation invariants.",
    href: `${repositoryUrl}/blob/main/docs/runtime-session.md`,
  },
  {
    key: "03 / ENGINE BOUNDARY",
    name: "SGLang execution boundary",
    detail: "The checked effects and completion evidence exchanged inside the single product.",
    href: `${repositoryUrl}/blob/main/docs/sglang-compatibility.md`,
  },
  {
    key: "04 / RECLAMATION",
    name: "Token virtualization",
    detail: "Token-level liveness, relocation, completion evidence, and open qualification gates.",
    href: `${repositoryUrl}/blob/main/docs/roadmap.md`,
  },
  {
    key: "05 / RECORDS",
    name: "Evidence index",
    detail: "Exact provenance for earlier-wire Full correctness, no-speedup findings, and other scoped records.",
    href: `${repositoryUrl}/blob/main/results/README.md`,
  },
  {
    key: "06 / QUALIFICATION",
    name: "Qualification workflow",
    detail: "Source closure, live runtime readback, independent verification, and claim gates.",
    href: `${repositoryUrl}/blob/main/README.md#qualification-and-evidence-tools`,
  },
];
