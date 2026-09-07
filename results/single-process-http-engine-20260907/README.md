# Single-process HTTP engine qualification

Status: passed correctness and lifecycle qualification on a real H20 with a
released, unmodified Full+Sliding checkpoint and its matching tokenizer assets.
This result does not qualify serving latency, throughput, fairness, capacity, or
production readiness.

The tested `orbitkv-serve` composition keeps the vLLM Rust OpenAI frontend and
the real `ModelEngine` in one process. The frontend owns OpenAI request schemas,
tokenizer/detokenizer behavior, SSE framing, request identity, and dropped-stream
auto-abort. OrbitKV remains the sole KV authority; Luminal executes only
manager-authored page metadata.

One automated end-to-end test passed all of the following against the real
checkpoint:

- non-streaming `/v1/completions` returned token IDs
  `[106, 107, 106, 106]` with `finish_reason=length`;
- streaming completions returned the same ordered token IDs and `[DONE]`;
- two concurrent HTTP completions returned equal eight-token sequences and
  exercised a multi-request model dispatch;
- dropping a live long-running HTTP stream reached the engine as exactly one
  cancellation instead of running to its full 1,000-token budget;
- graceful frontend shutdown completed; queued, active, reserved, writing,
  retiring, quarantined, and referenced KV state all drained.

The final same-source test completed in 33.99 seconds with warm compiler and
FlashInfer caches. This includes graph search and is diagnostic only.

## Reproduction

`ORBITKV_FRONTEND_MODEL_DIR` may equal `ORBITKV_MODEL_DIR` when tokenizer assets
live beside the weights. It was separate in this qualification because the
local weight directory contained only `config.json` and safetensors.

```bash
ORBITKV_MODEL_DIR=/workspace/models/gemma-3-270m-it +ORBITKV_FRONTEND_MODEL_DIR=/tmp/orbitkv-http-model +LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/local/cuda/lib64 +cargo test --release --locked -p orbitkv-engine --features server +  --test http_server +  openai_http_executes_streams_batches_cancels_and_drains +  -- --ignored --nocapture
```
