# Multi-class attention plumbing

Status: real-device plumbing qualification on the exact clean source commit
recorded in `environment.json`. This is not a released hybrid-model correctness
or performance result.

The test loaded one released dense Full-attention checkpoint and applied a
synthetic alternating Full/Sliding lifetime plan in memory. The Sliding window
was 64 tokens while the test covered four prefill tokens and one decode token,
so both attention policies were semantically equivalent over the exercised
range. No model files or config files were changed.

The purpose was to verify the new general executor path:

- one manifest assigns every decoder layer to exactly one attention class;
- 12 layers use the Full class and 12 use the Sliding class;
- each class owns an independent OrbitKV arena, write-slot input, CSR metadata,
  context dimension, and stable K/V address range;
- the Luminal graph includes both Full and sliding-window FlashInfer kernels;
- prefill, execution evidence, OrbitKV publication, decode warmup, and CUDA Graph
  capture complete successfully;
- unsafe in-place candidates are rejected during search.

Observed wall times are included only to identify the run. The one-time graph
search took 455.8 seconds, the four-token prefill dispatch took 17 ms, and the
single-token decode warmup plus capture took 45 ms. They are not repeated or
matched measurements.

## Reproduction

```bash
env \
  PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  CARGO_HOME=/root/.cargo \
  RUSTUP_HOME=/root/.rustup \
  HOME=/tmp/orbitkv-device-cache \
  CUDA_CACHE_PATH=/tmp/orbitkv-device-cache/.nv/ComputeCache \
  LUMINAL_FLASHINFER_DIR=/tmp/orbitkv-device-cache/.cache/luminal/flashinfer/flashinfer-src/f1e6fdcb8f65 \
  FLASHINFER_CUDA_ARCH=sm_90 \
  SEARCH_LOG=0 \
  ORBITKV_MODEL_DIR=/workspace/models/qwen2.5-0.5b-instruct \
  ORBITKV_SEARCH_GRAPHS=2 \
  cargo test --locked -p orbitkv-executor --features cuda \
    --test model_execution \
    released_dense_checkpoint_executes_multi_class_policy_plumbing \
    -- --ignored --nocapture
```

See `summary.json` for the machine-readable claim boundary.
