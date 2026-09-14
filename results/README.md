# Model inference performance

Current model: **Qwen3.8-27B-FP8 on one NVIDIA H20**.
[Support and checkpoint identity](../docs/capability-matrix.md) define the scope.

- [Three-engine comparison](qwen3.8-27b-fp8-h20-20260915/README.md):
  OrbitKV, vLLM and SGLang measured with one `vllm bench serve` client.
- [Initial serving baseline](qwen3.8-27b-fp8-h20-20260914/README.md):
  four workloads, three independent processes each; includes first-run latency
  and output-repeatability limitations.

Each report records checkpoint, precision, hardware, source/binary identity,
workload, startup/cache policy, TTFT, TPOT, throughput and memory observations.
Per-run metrics and checksums accompany the report. The website imports reviewed
performance data directly from these records.

Use the [benchmark method](../docs/benchmarking.md) for new comparisons.
Raw experiments and compiler/provider diagnostics belong in `.qualification/`.
Superseded development journals and older model experiments remain in Git history.
