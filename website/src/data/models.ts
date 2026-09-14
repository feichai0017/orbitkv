export const models = [
  {
    name: "Qwen3.8 27B",
    precision: "Block FP8",
    attention: "Full + Gated DeltaNet",
    status: "Bounded text inference",
    scope:
      "Verified official checkpoint · one NVIDIA H20 · text inference through C8. Vision and MTP are outside this scope.",
    path: "docs/capability-matrix.md",
  },
];

export interface PerformanceReport {
  schema: "orbitkv.model-performance.v1";
  model: { name: string; precision: string };
  measured_at: string;
  hardware: { name: string; count: number };
  source: { commit: string; binary_sha256: string };
  measurement: string;
  notes: string[];
  comparison?: {
    client: string;
    engines: { name: string; version: string }[];
  };
  workloads: {
    name: string;
    engine?: string;
    input_tokens: number[];
    output_tokens: number;
    concurrency: number;
    requests_per_run: number;
    runs: number;
    metrics: {
      output_throughput: number;
      median_ttft_ms: number;
      p95_ttft_ms: number;
      median_tpot_ms: number;
      p95_tpot_ms: number;
    };
    metric_ranges: {
      output_throughput: { minimum: number; maximum: number };
    };
    sampled_max_device_gpu_bytes: number | null;
    output_repeatable: boolean;
  }[];
}

// Only reviewed model-performance records enter the public tables. Numerical
// data stays in results; page copy never duplicates benchmark measurements.
const reports = import.meta.glob<PerformanceReport>(
  "../../../results/*/performance.json",
  { eager: true, import: "default" },
);

export const performance = Object.entries(reports)
  .map(([path, report]) => {
    if (report.schema !== "orbitkv.model-performance.v1") {
      throw new Error(`Unsupported performance schema: ${path}`);
    }
    return {
      ...report,
      path: path
        .replace("../../../", "")
        .replace("performance.json", "README.md"),
    };
  })
  .sort((a, b) => b.measured_at.localeCompare(a.measured_at));
