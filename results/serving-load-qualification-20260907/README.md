# Single-process serving load qualification

Status: passed a narrow real-device load and concurrency qualification for one
released Full+Sliding checkpoint. This is not a comparison with SGLang and does
not establish a general performance advantage.

One `orbitkv-serve` process used OrbitKV as the sole KV authority, one compiled
Luminal decoder, and the embedded vLLM Rust OpenAI frontend. The same warm server
instance served four fixed-workload runs at concurrency 1, 2, 4, and 8. Every
run completed 16 requests, every request produced the requested 256 tokens, and
the client reported no request error.

Output throughput rose from 184.24 token/s at C1 to 518.88 token/s at C8. This
is a concurrency scaling observation, not a free speedup: median TTFT rose from
163.06 ms to 1127.81 ms and median TPOT rose from 4.80 ms to 11.06 ms. C2 was
the measured latency-efficient point for this trace: 350.59 output token/s with
4.51 ms median TPOT, while median TTFT was 296.07 ms.

The discrete generated-text digest is stable across C2/C4/C8 but differs at C1.
A separate teacher-forced executor diagnostic therefore compares logits rather
than requiring cross-batch-size text identity. For the same 32-token prompt and
16 generation positions, every row within B=8 was bit-identical; B=1 versus B=8
had maximum absolute logit difference 0.4296875 and zero argmax mismatches. This
qualifies request isolation and bounded numerical parity for the tested trace.
It does not promise identical long autoregressive text across batch sizes when
BF16 top candidates are nearly tied.

An invalid preflight also exposed a benchmark-client false-positive mode: a
stream can emit an initial empty SSE frame followed by an error frame, while the
client reports the request as completed. The repository harness now rejects a
run unless every detailed output length, total output-token count, and per-request
error field match the requested workload.

Only compact reviewed measurements are retained here. Model weights, binaries,
raw logs, and generated texts are excluded.

## Reproduction

Start one release server with the configuration in `environment.json`, then run
the pinned Rust benchmark client independently for each concurrency value:

```bash
vllm-bench \
  --backend openai \
  --base-url http://127.0.0.1:18080 \
  --endpoint /v1/completions \
  --model orbitkv-hybrid \
  --tokenizer /tmp/orbitkv-http-model \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 256 \
  --random-range-ratio 0 \
  --num-prompts 16 \
  --request-rate inf \
  --max-concurrency 8 \
  --max-model-len 1024 \
  --ignore-eos \
  --temperature 0 \
  --seed 7 \
  --save-result --save-detailed
```

Run the direct B=1/B=8 parity gate with:

```bash
ORBITKV_MODEL_DIR=/workspace/models/gemma-3-270m-it \
ORBITKV_SEARCH_GRAPHS=2 \
cargo test --release --locked -p orbitkv-executor --features cuda \
  --test model_execution \
  released_checkpoint_bounds_single_and_multi_request_logits \
  -- --ignored --nocapture
```
