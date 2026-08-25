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
    value: "2 cycles",
    label: "Repeated Full relocation",
    detail: "Single-Full, request-private, full-evacuation core/Python path; the SGLang periodic trigger is also host-tested.",
  },
  {
    value: "1 × batch",
    label: "Multi-request relocation",
    detail: "Host-tested ABI8 mark, prepare, submit, and complete, followed by one aggregate registry commit, one aggregate head replacement, checked non-atomic mirror writes, and one ACK.",
  },
  {
    value: "7 / 7 H20",
    label: "Relocation conformance",
    detail: "Engine-neutral B1/B4/B32 opaque-byte cases pass with real append, copy, and consumer streams.",
  },
  {
    value: "16 / 8",
    label: "Sealed relocation records / pairs",
    detail: "Exact clean 7e029310 source; Qwen2.5-0.5B Full B1/B4 across four balanced epochs; every pair is token-exact and fully drained.",
  },
  {
    value: "L2 host",
    label: "Pressure telemetry",
    detail: "Request-private consumed/resident/reachable/semantic-live metrics; real async-GPU pressure has not run.",
  },
  {
    value: "2 wheels",
    label: "Engine-neutral SPI",
    detail: "orbitkv-runtime and orbitkv-reference build and clean-install; SGLang has not migrated to the SPI.",
  },
  {
    value: "8 / 8",
    label: "Qwen3.8 diagnostic pairs",
    detail: "Four epochs each at B1 and B4; every pair is token-exact and census-clean under explicit Triton FP8.",
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
    boundary: "exact 6f62a23 source; official SGLang v0.5.17 base with record-bound manager loader patch; one H20; page16 BF16 NHD eager FA3",
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
    contract: "Immutable snapshots, Prefix/COW, token views, Full evacuation, packed publication and generation-safe packed fork, reclamation",
    boundary: "host only; no engine or GPU inheritance",
  },
  {
    result: "ABI8 C wire",
    value: "L2 GO / 40",
    contract: "29 manager plus 11 state-pool symbols, C/C++ layouts, per-handle transactions, and receipt gates",
    boundary: "host wire only; manager and state-pool handles are separate, with no cross-handle atomicity or H20 inheritance",
  },
  {
    result: "Repeated private Full relocation",
    value: "L2 host / 2 cycles",
    contract: "Rust core and Python runtime repeat append, disposition mark, full evacuation, publication, and exact ACK; the SGLang periodic trigger is host-tested across two reclamation boundaries",
    boundary: "single-Full request-private profile only; packed fork is host+FFI tested, while packed Prefix and packed shared-tail COW fail closed",
  },
  {
    result: "ABI8 multi-request relocation",
    value: "L2 host / one batch transaction",
    contract: "One mark, prepare, submit, and complete call per scheduler batch; flattened copy plus one event, one aggregate registry commit, one aggregate head replacement, every mirror plan validated before writes, and one aggregate ACK",
    boundary: "mirror writes are not rollback-atomic; post-mark failures fail-stop without rollback; a producer-to-copy event exists, but completion is eagerly host-blocking with no asynchronous overlap",
  },
  {
    result: "CUDA opaque-byte harness",
    value: "7 / 7 H20 PASS",
    contract: "Engine-neutral B1/B4/B32 conformance with two cycles, a 257-byte payload oracle, real append/copy/consumer streams, event-ordered readback, ACK-gated reuse, and final drain",
    boundary: "component conformance only; no sealed L3/L4, performance, capacity, or complete-engine qualification",
  },
  {
    result: "Qwen2.5 token relocation on H20",
    value: "16 records / 8 pairs PASS",
    contract: "Four order-balanced epochs at B1/B4; exact Naive/Relocate output tokens, complete drain, zero failure/quarantine/fail-stop counters, and 7/7 bundled H20 component cases",
    boundary: "exact clean 7e029310 source; official SGLang v0.5.17 base plus manifest-bound canonical loader patch, request-private Full, page16 BF16 NHD eager FlashInfer on one observed H20; source_clean/preflight_bound/sealed/qualified=true; hardware_attested=false; performance_go=false",
  },
  {
    result: "Relocation sealed timing",
    value: "−1.3467% / +2.8096%",
    contract: "B1 / B4 hot throughput; mean latency +1.3651% / −2.7328%, median +2.0033% / +0.9810%, p95 +0.5141% / −19.8635%",
    boundary: "descriptive mixed results only; performance_go=false; no general speedup, capacity, or memory-saving claim",
  },
  {
    result: "Prior relocation diagnostic",
    value: "historical / superseded",
    contract: "The earlier dirty-source eight-pair record remains append-only for auditability",
    boundary: "unsealed and unqualified; its timing values do not describe the clean-source sealed qualification",
  },
  {
    result: "Request-private pressure telemetry",
    value: "L2 host",
    contract: "Opt-in event samples for consumed, resident, request-reachable, semantic-live, free-space, high-water, and retention-amplification fields",
    boundary: "no real async-GPU pressure run; fixed-state bytes excluded; shared Prefix/request-fork RA fail closed",
  },
  {
    result: "Engine-neutral adapter SPI",
    value: "2 clean-installable wheels",
    contract: "orbitkv-runtime typed SPI plus orbitkv-reference external CPU/CUDA tensor-arena implementation and contract oracle",
    boundary: "reference is not a complete engine; SGLang has not migrated to this SPI; no engine, hardware, or performance inheritance",
  },
  {
    result: "ABI8 fixed-state seam",
    value: "host + scoped H20 pairs",
    contract: "Integrated, structurally admitted exact GDN+convolution runtime: request allocation, initial clear, completion event, wait, retire, clear, and exact ACK",
    boundary: "Qwen3.5 has scoped pair verification and Qwen3.8 is diagnostic; both are model-specific and below L4; replacement trigger and other families remain pending",
  },
  {
    result: "Qwen3.8-27B-FP8 H20",
    value: "8 / 8 diagnostic PASS",
    contract: "Four epochs each at B1 and B4 (8 pairs total); exact output tokens, clean final census, zero failure/fail-stop counters, and explicit Triton FP8",
    boundary: "dirty and unsealed; no preflight; hardware_attested=false; qualified=false; performance_go=false; hot throughput -6.19% B1 (noisy) and -2.70% B4; equal configured arena reservation observed difference 0%, not a qualified memory result",
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
    boundary: "observed reservation difference only; not qualified end-to-end memory saving",
  },
  {
    result: "Unqualified ABI8 paths",
    value: "pending",
    contract: "Relocation hardware attestation/performance, packed Prefix/shared-tail COW, MLA, fixed-state L4, real async pressure/overlap, Graph, speculation, and distributed execution",
    boundary: "not covered by either scoped sealed record; no general SGLang replacement claim",
  },
];

export const roadmap = [
  {
    state: "NEXT",
    name: "Complete fixed-state integration",
    detail: "Add the production replacement trigger and KDA, ShortConv, and other family bindings; then run clean, preflighted fixed-state qualification.",
  },
  {
    state: "THEN",
    name: "Expand ABI8 H20 coverage",
    detail: "Independently attest the sealed Full-relocation hardware, qualify performance and real async pressure/overlap, and add packed Prefix/shared-tail COW only with separate evidence.",
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
    detail: "Sealed scoped H20 relocation correctness/lifecycle, pending performance and hardware attestation, Graph, speculation, and distribution.",
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
    key: "06 / DIAGNOSTIC EVIDENCE",
    name: "Qwen3.8-27B-FP8 H20",
    detail: "Four B1 plus four B4 token-exact, census-clean pairs under explicit Triton FP8; dirty, unsealed, and unqualified.",
    href: `${repositoryUrl}/tree/main/results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824`,
  },
  {
    key: "07 / SEALED EVIDENCE",
    name: "Qwen2.5 Full relocation on H20",
    detail: "Exact clean 7e029310; 16 records and eight exact, drained B1/B4 pairs; scoped correctness/lifecycle only, unattested and performance not GO.",
    href: `${repositoryUrl}/tree/main/results/h20-sglang-v0517-abi8-token-relocation-20260825`,
  },
  {
    key: "08 / ADAPTER SPI",
    name: "Engine-neutral data plane",
    detail: "Separate runtime/reference wheels and an external-arena contract oracle; not a complete engine and not yet used by SGLang.",
    href: `${repositoryUrl}/blob/main/docs/engine-adapter-spi.md`,
  },
  {
    key: "09 / RECORDS",
    name: "Evidence index",
    detail: "Append-only snapshots with explicit source and ABI boundaries.",
    href: `${repositoryUrl}/blob/main/results/README.md`,
  },
];
