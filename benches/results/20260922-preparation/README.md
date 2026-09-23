# Request preparation: final results

H20, BF16 Qwen3-8B, TP=1, Python 3.11, vLLM 0.29.0 and SGLang 0.5.20.
Source: `ffada83e`. Model: `b968826d9c46dd6066d109eabc6255188de91218`.

- `summary.csv`: three matched off/on pairs per engine, unbounded demand,
  a 100 ms deadline and a one-batch stop. The middle pair reverses run order.
- `dram.csv`: ordinary DRAM-only recovery with cold controls, concurrency 1/4
  and three repetitions.
- `reproduce.sh`: all 20 runs, from the repository root with matching binaries
  and `.venv/{vllm,sglang}-release` environments.

Sustained controls use seed 20260922, C4, 64 requests, 75% reuse selection,
12 prefixes, 1,024/4,096-token inputs and 16-token outputs. GPU capacity is
16,384 tokens; Manager DRAM/SSD/query budgets are 4/16/3 GiB. The DRAM-only
supplement uses an 8 GiB host pool and no SSD, so it is not a matched tier-speed
comparison. Tracing is on and unowned warming is off throughout.

Only final aggregate rows are committed. Raw manifests, outputs, timelines and
logs remain in ignored `benches/results/runs/20260922-preparation-*` directories.
The [report](../../../docs/request-preparation.md#measured-results) explains
metric boundaries, output differences, cleanup and why the policy stays off.
