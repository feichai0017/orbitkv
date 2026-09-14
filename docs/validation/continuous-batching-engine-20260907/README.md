# Continuous batching engine qualification

Status: passed correctness and lifecycle qualification on a real H20 with a
released, unmodified Full+Sliding checkpoint. This result does not qualify
serving throughput, latency, fairness, capacity, or the HTTP frontend.

The tested engine uses bounded admission and per-request output queues. One
dedicated worker owns the OrbitKV `RuntimeSession`, stable per-class arenas,
and the compiled Luminal decoder. It selects ready requests under a token budget,
prioritizes decode work, and submits one atomic manager transaction and one
Luminal dispatch for each selected active set.

Two device scenarios passed:

- Two concurrent 512-token prompts were admitted together. Their prefill and
  subsequent decode steps reached a maximum observed batch size of two and
  produced the same eight-token sequence as the previously qualified independent
  Transformers reference for each request.
- A 16-token prefill was admitted after a 512-token request had entered a
  32-token decode. Scheduler counters proved at least one mixed prefill/decode
  dispatch and at least one multi-request dispatch. Both requests terminated and
  all manager state drained.

The scheduler also passed host contracts for bounded admission, duplicate-ID
preservation, queue-disconnect cleanup, token-budget ordering, slow-consumer
backpressure, per-CSR-row sampled-token selection, and per-class concurrent page
budgets.

## Reproduction

```bash
ORBITKV_MODEL_DIR=/workspace/models/gemma-3-270m-it \
ORBITKV_SEARCH_GRAPHS=2 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/local/cuda/lib64 \
cargo test --release --locked -p orbitkv-engine --features cuda \
  --test model_engine \
  released_hybrid_engine_batches_concurrent_prefill_and_decode \
  -- --ignored --nocapture

ORBITKV_MODEL_DIR=/workspace/models/gemma-3-270m-it \
ORBITKV_SEARCH_GRAPHS=2 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/local/cuda/lib64 \
cargo test --release --locked -p orbitkv-engine --features cuda \
  --test model_engine \
  released_hybrid_engine_batches_decode_with_late_prefill \
  -- --ignored --nocapture
```

The final same-source, single-threaded invocation ran all three model-engine
device tests in 91.25 seconds with warm compiler/FlashInfer caches. The total includes graph
search and is diagnostic only.
