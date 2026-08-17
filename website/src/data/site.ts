export const repositoryUrl = "https://github.com/feichai0017/orbitkv";
export const homeUrl = "https://feichai0017.github.io/orbitkv/";

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Architecture", href: "/docs/" },
  { label: "Evidence", href: "/evidence/" },
];

export const compilerStages = [
  { key: "01", name: "Retention IR", detail: "Declare affine may_read(query, key) relations for persistent state." },
  { key: "02", name: "Lifetime", detail: "Prove unbounded or fixed-window death from query-key distance constraints." },
  { key: "03", name: "Normalize", detail: "Partition state by retirement predicate, not token birth alone." },
  { key: "04", name: "Synthesize", detail: "Emit append-only or periodic address programs and minimum slots." },
];

export const metrics = [
  { value: "−61.696%", label: "Mistral KV", detail: "page16, 4 x 12K prompts with 19,152 physical slots" },
  { value: "0.9855×", label: "Median runtime", detail: "decode Graph replay with identical output digests" },
  { value: "+25.81%", label: "Full capacity", detail: "real gpt-oss-20b, same 1.979 GiB KV budget" },
  { value: "−20.30%", label: "Owner vs Stock", detail: "balanced four-way real-checkpoint ablation" },
  { value: "−42.105%", label: "Head-stripe KV", detail: "exact multi-scale-window reduction versus max-window allocation" },
];

export const evidenceRows = [
  {
    result: "Uniform SWA execution",
    value: "−61.696%",
    contract: "Mistral-7B, page16, 4 x 12K prompts, 19,152 slots versus 50,000",
    boundary: "one H20; decode Graph replay; three balanced pairs; identical digests",
  },
  {
    result: "Real checkpoint capacity",
    value: "+25.81%",
    contract: "openai/gpt-oss-20b, Full capacity 47,616 to 59,904",
    boundary: "same 1.979 GiB KV budget; no radix/spec/overlap/Graph",
  },
  {
    result: "Real checkpoint makespan",
    value: "−20.30%",
    contract: "Owner32 vs Stock128, four balanced execution orders",
    boundary: "8 x 6000 prompt + 32 decode; identical output-token digests",
  },
  {
    result: "Capacity prediction",
    value: "4 / 4",
    contract: "predicted Full/SWA pools matched fresh SGLang processes exactly",
    boundary: "gpt-oss-20b, one H20, recorded 1.979 GiB KV budget",
  },
  {
    result: "Chunked Local synthesis",
    value: "Resettable",
    contract: "floor(q/C) == floor(k/C) to epoch arena and chunk-end death",
    boundary: "host exhaustive proof; SGLang lowering not yet enabled",
  },
  {
    result: "Lifetime Normalization",
    value: "−42.105%",
    contract: "32 KV heads with 512/2048/8192 token windows",
    boundary: "exact host geometry and Manager proof; no GPU claim",
  },
];

export const roadmap = [
  { state: "NEXT", name: "Graph-stable KV storage", detail: "Back real SGLang KV tensors with cost-approved VMM regions." },
  { state: "THEN", name: "Recurrent and prefix state", detail: "Compile Mamba state and continuation capsules." },
];

export const docs = [
  {
    key: "01 / DESIGN",
    name: "End-to-end plan",
    detail: "Validation stages, ownership boundary, value gates, and remaining work.",
    href: `${repositoryUrl}/blob/main/docs/sglang-e2e.md`,
  },
  {
    key: "02 / REAL MODEL",
    name: "gpt-oss-20b validation",
    detail: "Released checkpoint capacity, admission, overhead, and reclamation certificates.",
    href: `${repositoryUrl}/blob/main/docs/h20-gpt-oss-20b-real-validation-20260817.md`,
  },
  {
    key: "03 / APPLICABILITY",
    name: "Three-model applicability",
    detail: "Qwen fallback, Mistral bounded execution, and GPT-OSS Hybrid plans.",
    href: `${repositoryUrl}/blob/main/results/applicability-h20-20260817/applicability.json`,
  },
  {
    key: "04 / NORMAL FORM",
    name: "Lifetime Normalization",
    detail: "Per-head multi-scale windows, max-window baseline, and exact retention amplification.",
    href: `${repositoryUrl}/blob/main/results/lifetime-normalization-20260817/summary.json`,
  },
  {
    key: "05 / RECORDS",
    name: "Evidence index",
    detail: "Summaries, manifests, and historical runs.",
    href: `${repositoryUrl}/tree/main/results`,
  },
];
