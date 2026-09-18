# AletheiaRT qualification and plan publication

The optimizer may create candidates freely. Production sees only immutable
bundles that pass this publication sequence:

```text
candidate plan
  + provider artifacts
  + numerical evidence
  + raw device timing samples
  + measured resource bounds
  + SGLang workload trace
             |
             v
tools/stage_bundle.py
  1. Rust plan validation before using the plan ID as a path
  2. exact trace-bucket match
  3. evidence completeness and >=12 timing samples
  4. artifact SHA-256 verification
  5. certificate construction
  6. Rust certificate validation
  7. atomic bundle publication
  8. whole-registry fallback-DAG validation
```

The published layout is self-contained:

```text
registry/<plan-id>/
  plan.json
  certificate.json
  evidence.json
  trace.jsonl
  artifacts/<sha256>
```

`evidence.json` inside the bundle rewrites artifact paths to the bundled
content-addressed files, so moving the registry does not invalidate provenance.
Raw samples are retained in evidence even though the certificate carries only
summary statistics.

## Fixture

`make stage-reference` publishes a CPU reference bundle to `.aletheia/registry`.
It proves the control-plane mechanics only and is labeled as a fixture. It is
not GPU performance evidence.

## GPU runner contract

A FlashInfer or DeepGEMM qualification runner must emit evidence schema v1 with:

- exact hardware and software fingerprints;
- PyTorch/reference numerical cases and tolerances;
- at least twelve raw CUDA-event timing samples;
- measured peak device memory and workspace;
- soak and fault evidence when applicable;
- one path and SHA-256 for every artifact named by the plan.

Provider-local autotuner caches are inputs to qualification, not deployable
proof by themselves. FlashInfer's chosen runner/tactic and DeepGEMM's generated
source/cubin identity must be carried into the plan configuration and software
fingerprint.

The first real runner is `aletheia-qualify-rmsnorm`, exposed by
`integrations/providers`. `make qualify-rmsnorm` runs the source-pinned
FlashInfer BF16 RMSNorm on SM90, compares multiple seeded cases against a
PyTorch FP32-accumulation oracle, retains raw CUDA-event samples, and copies the
JIT `norm.so` into a candidate directory. The candidate remains unpublished
until a matching real SGLang trace is supplied to `stage_bundle.py`.
