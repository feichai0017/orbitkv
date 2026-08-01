export const repositoryUrl = "https://github.com/feichai0017/loom-kernels";

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Operators", href: "/docs/operators/" },
  { label: "Integration", href: "/docs/integration/" },
  { label: "Evidence", href: "/benchmarks/" },
];

export const operatorFamilies = [
  {
    name: "Normalization",
    boundary: "RMSNorm, residual fusion, and dynamic FP8 or INT8 output.",
  },
  {
    name: "MLP activation",
    boundary: "SiLU-and-Mul with optional block FP8 or dynamic INT8 output.",
  },
  {
    name: "KV update",
    boundary: "RoPE with paged-KV write to native or static FP8 cache storage.",
  },
  {
    name: "Decode tail",
    boundary: "Logits processing, penalties, filtering, logprobs, and sampling.",
  },
  {
    name: "MoE movement",
    boundary: "Stable permutation and weighted combine around vendor grouped GEMM.",
  },
  {
    name: "Paged decode",
    boundary: "Short-context MQA/GQA and local split-K/LSE with explicit fallback.",
  },
];

export const supportedOperators = [
  {
    name: "RMSNorm",
    dtypes: "F32, FP16, BF16",
    boundary: "Standalone normalization",
    status: "supported",
  },
  {
    name: "Add + RMSNorm",
    dtypes: "F32, FP16, BF16",
    boundary: "Residual update and normalization",
    status: "supported",
  },
  {
    name: "RMSNorm + dynamic FP8",
    dtypes: "F32, FP16, BF16 to E4M3FN",
    boundary: "Normalization and GEMM input quantization",
    status: "supported",
  },
  {
    name: "RMSNorm + dynamic INT8",
    dtypes: "F32, FP16, BF16 to INT8",
    boundary: "Explicit W8A8 path before vendor GEMM",
    status: "in progress",
  },
  {
    name: "SiLU-and-Mul",
    dtypes: "F32, FP16, BF16",
    boundary: "Split-half SwiGLU activation",
    status: "supported",
  },
  {
    name: "SiLU-and-Mul + block FP8",
    dtypes: "FP16, BF16 to E4M3FN",
    boundary: "Activation and group-64 or group-128 quantization",
    status: "supported",
  },
  {
    name: "SiLU-and-Mul + dynamic INT8",
    dtypes: "FP16, BF16 to INT8",
    boundary: "Activation and per-token quantization before vendor GEMM",
    status: "profile-gated",
  },
  {
    name: "RoPE + paged-KV write",
    dtypes: "F32, FP16, BF16, static FP8 cache",
    boundary: "Packed Q/K rotation and engine-owned cache write",
    status: "supported",
  },
  {
    name: "Logits preprocessing",
    dtypes: "F32 logits",
    boundary: "Mask, sparse bias, suppression, and temperature",
    status: "supported",
  },
  {
    name: "Token penalties",
    dtypes: "F32 logits, int64 history",
    boundary: "Sparse repetition, frequency, and presence update",
    status: "supported",
  },
  {
    name: "Top-k, top-p, and Min-P",
    dtypes: "F32, FP16, BF16",
    boundary: "In-place filtering with measured vLLM shape gates",
    status: "supported",
  },
  {
    name: "Sample logprobs",
    dtypes: "F32, FP16, BF16",
    boundary: "Selection, normalization, rank, and top-k return",
    status: "supported",
  },
  {
    name: "Categorical sampling",
    dtypes: "F32 probabilities, int64 state",
    boundary: "Philox sampling with caller-owned seed and counter",
    status: "supported",
  },
  {
    name: "Greedy speculative verify",
    dtypes: "int32 drafts, int64 target IDs",
    boundary: "Ragged acceptance and token compaction",
    status: "supported",
  },
  {
    name: "MoE permutation + combine",
    dtypes: "F32, FP16, BF16, FP8 permutation",
    boundary: "Caller-owned movement around vendor grouped GEMM",
    status: "supported",
  },
  {
    name: "Paged MQA/GQA decode",
    dtypes: "F32, FP16, BF16",
    boundary: "Short-context vLLM route with FA3 fallback",
    status: "supported",
  },
  {
    name: "Local split-K/LSE merge",
    dtypes: "FP16, BF16 with F32 workspace",
    boundary: "Longer-context internal decode path",
    status: "supported",
  },
];

export const nextOperators = [
  {
    milestone: "K5",
    name: "Production MoE gate",
    reason: "Run the pinned pretrained workload before adding routing.",
  },
  {
    milestone: "K2.5",
    name: "Quantization plumbing",
    reason: "Add scale, pack, or layout work only for a measured vendor-kernel consumer.",
  },
  {
    milestone: "K8",
    name: "Rust decode proof",
    reason: "Chain one zero-copy decode step over borrowed tensors and streams.",
  },
];

export const evidence = [
  {
    operator: "ABI12 native wheel",
    shape: "H20, CUDA 13.1, Python 3.11",
    result: "359 / 359 / 245 tests",
    detail: "One two-library artifact across PyTorch 2.10/2.11 and vLLM 0.24/0.25.",
    file: "h20-native-wheel-clean-install-abi12-20260801.json",
  },
  {
    operator: "RMSNorm to FP8",
    shape: "Qwen2.5-0.5B prefill",
    result: "1.0066-1.0506x",
    detail: "Exact output and an order-stable batch-latency ratio.",
    file: "h20-rms-norm-dynamic-fp8-residual-20260727.json",
  },
  {
    operator: "Sparse token penalties",
    shape: "F32, 1-128 rows",
    result: "5.82-34.30x",
    detail: "Exact output with O(history) caller workspace.",
    file: "h20-token-penalties-20260725.json",
  },
  {
    operator: "Categorical sampling",
    shape: "F32, 4-32 rows",
    result: "1.15-5.40x",
    detail: "One kernel with caller-owned Philox state.",
    file: "h20-categorical-sample-20260727.json",
  },
  {
    operator: "Short paged decode",
    shape: "FP16/BF16, context at most 32",
    result: "1.154-2.374x",
    detail: "All 24 admitted vLLM cases win. Other shapes use FA3.",
    file: "h20-vllm-paged-decode-backend-20260722.json",
  },
  {
    operator: "MoE engine admission",
    shape: "Synthetic Qwen2-MoE, Cutlass FP8",
    result: "Exact, 48 / 48 hits",
    detail: "Grouped GEMM stays unchanged. Production value remains open.",
    file: "h20-vllm-engine-moe-movement-20260801.json",
  },
  {
    operator: "FP8 KV candidate",
    shape: "Qwen2.5-7B held-out slice",
    result: "Rejected, about 3.07x PPL",
    detail: "Cache capacity doubled, but the representation failed quality.",
    file: "h20-fp8-kv-system-rejected-20260727.json",
  },
];
