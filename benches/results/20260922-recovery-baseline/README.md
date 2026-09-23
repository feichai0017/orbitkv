# Ordinary recovery: final results

H20, BF16 Qwen3-8B, TP=1, vLLM 0.29.0 and SGLang 0.5.20.
Source: `90bb88255597513be9c92ef32ed3f96ebee2c0e0`.
Model: `b968826d9c46dd6066d109eabc6255188de91218`.

`summary.csv` retains only final aggregate rows. Raw manifests, samples and
logs remain in ignored `benches/results/runs/20260922-recovery-*` directories.
This source predates consumer preparation. See the
[stage analysis and limits](../../../docs/recovery-performance.md).

Reproduce from the matching checkout and engine environment, replacing `vllm`
with `sglang` for the second engine:

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload concurrent --lengths 1024 4096 --concurrencies 1 4 8 --repeats 3 \
  --gpu-tokens 16384 --host-gib 4 --ssd-gib 16 --query-budget-gib 2 \
  --queue-warmup off --trace-transfers --seed 20260922
```

For sustained traffic, replace `--workload concurrent` with
`--workload sustained --duration-seconds 20 --max-requests 128 --working-set 12`.
The default reuse ratio is 0.75 and output length is 16 tokens.
