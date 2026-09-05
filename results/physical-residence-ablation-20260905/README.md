# Physical residence ablation

Status: real-device mechanism qualification on the dirty working tree recorded
in `environment.json`. This is not a released hybrid-model or repeated
performance result.

One Luminal graph was compiled once, then executed against two independent
OrbitKV sessions with the same manifest, model weights, input tokens, attention
visibility, kernel selection, arena geometry, and executor metadata shape:

- `Compiled` used the compiler-derived periodic Sliding address and retirement
  program.
- `RequestLifetime` used append-only physical addresses and retained
  semantically dead pages until request release.

The test used a released dense checkpoint with an in-memory alternating
Full/Sliding policy solely to exercise the generic multi-class path. It ran an
80-token prefill, crossing the 64-token Sliding boundary, followed by one decode
token after publication and retirement acknowledgement. Token IDs and full
logits were byte-identical between arms in both phases.

After decode, the compiled Sliding arena held 5 pages / 491,520 bytes; the
request-lifetime baseline held 6 pages / 589,824 bytes. The one-page difference
is 98,304 bytes, or 16.7% of the baseline Sliding allocation at this boundary.
These numbers prove the ablation seam can isolate physical lifetime management.
They measure manager-owned live payload inside an identically preallocated HBM
arena; they do not mean CUDA returned 98,304 bytes to the device allocator. The
benefit is reusable in-arena admission headroom.
The observed timings are diagnostics only: there was one sample per arm, no
alternating epochs, and no serving workload.

## Reproduction

```bash
env \
  PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  CARGO_HOME=/root/.cargo \
  RUSTUP_HOME=/root/.rustup \
  CUDA_CACHE_PATH=/tmp/orbitkv-device-cache/.nv/ComputeCache \
  LUMINAL_FLASHINFER_DIR=/tmp/orbitkv-device-cache/.cache/luminal/flashinfer/flashinfer-src/f1e6fdcb8f65 \
  FLASHINFER_CUDA_ARCH=sm_90 \
  SEARCH_LOG=0 \
  ORBITKV_MODEL_DIR=/workspace/models/qwen2.5-0.5b-instruct \
  ORBITKV_SEARCH_GRAPHS=2 \
  cargo test --locked -p orbitkv-executor --features cuda \
    --test model_execution \
    released_checkpoint_compares_compiled_and_request_lifetime_residence \
    -- --ignored --nocapture
```

See `summary.json` for the machine-readable claim boundary.
