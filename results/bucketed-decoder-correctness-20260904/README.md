# Bucketed decoder correctness diagnostic

Status: real-device, released-checkpoint correctness diagnostic. This record is
not a latency, throughput, memory, capacity, or production qualification.

The test ran the committed `CompiledDecoder` path on one NVIDIA H20. One
symbolic dense-decoder graph was searched once into two query-token buckets:
`s=1` decode and `s=2..8` prefill. Batch size and context-page count each used
one bounded capacity bucket. Tokens, positions, write slots, and all CSR inputs
were allocated to their configured capacities before search, and every dispatch
checked that their device addresses remained unchanged.

The released checkpoint completed a four-token prefill followed by two greedy
decode steps through the same graph, CUDA runtime, stream, and persistent K/V
arena. The test observed the prefill bucket and then the decode bucket, produced
finite logits, and confirmed three OrbitKV publications. No second model graph,
second runtime, or cross-runtime cache transfer was used.

The selected bucket set did not use in-place K/V updates for every layer.
Luminal materialized some updates and executed its registered device-to-device
epilogue back into the same stable K/V arena. The search rejected five invalid
aliasing candidates. This is evidence for address stability and correct repeated
execution, not for zero-copy K/V writes.

The same source also passed manager-authored token relocation against persistent
K/V graph inputs, CUDA-event completion, publication, and a following packed
decode.

Observed diagnostic timings from the single correctness run were approximately:

- one-time two-bucket search and compile: 215.8 seconds;
- four-token prefill dispatch: 15 milliseconds;
- first decode dispatch: 36 milliseconds;
- second warm decode dispatch: 4 milliseconds.

These values include an extremely small prompt and only two searched candidates.
They have no matched reference, no repetitions, and no statistical treatment,
so they must not be used as a performance claim.

## Reproduction

The model path and cache directories are environment inputs rather than product
source. The exact command shape was:

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
    released_decoder_reuses_one_compiled_runtime_and_kv_arena \
    -- --ignored --nocapture
```

The relocation closure used the same CUDA and FlashInfer environment and ran
`manager_authored_token_moves_execute_and_publish` from `cuda_relocation`.
