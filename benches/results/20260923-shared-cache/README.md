# Shared-cache serving: final results

Qwen3-8B BF16, one H20 (97,871 MiB), TP=1 independent replicas, same-host
Mooncake TCP. vLLM 0.29.0 and SGLang 0.5.20 ran separately under Python 3.11.
Model revision: `b968826d9c46dd6066d109eabc6255188de91218`.
Source: the commit adding this report, based on `b9a6c625`.

Each engine passed two fresh-prefix restores (513/1025 input tokens), a restore
after restarting the sole catalog host, and recomputation after the source
restarted without payload. Each generated eight tokens. All outputs matched
their controls. Each engine restored 288 MiB through Mooncake and H2D across
three requests; the source-restart miss restored zero bytes. Query, inflight,
source-transfer and I/O gauges drained after every case.

Each replica had its own 2 GiB Manager pool, 1 GiB query budget and default
1 GiB source-transfer budget. Read batches were limited to 32 MiB. SSD,
warming and request preparation were disabled. Both engine caches were limited
to 4096 GPU tokens. vLLM used eager execution and `VLLM_BATCH_INVARIANT=1`;
SGLang used `--enable-deterministic-inference`. etcd 3.6.5 ran locally with
12-second Manager membership leases.

`summary.csv` keeps the final rows and native stage totals. These short serial
controls establish correctness, not throughput, tail-latency or competitor
superiority. The initial 513-token restore was slower than cold source computation
on both engines; startup/kernel effects are not isolated. This is not evidence of
two-host, RDMA, cross-engine, multi-rank or P/D support. Permanent requester loss
still needs proven transport revocation before orphaned source pins can be freed.

Build the Manager before running, then from the matching checkout:

```bash
cd python
for engine in vllm sglang; do
  ETCD_BIN=/path/to/etcd \
  ORBITKV_CACHE_MANAGER_BINARY=/workspace/orbitkv/target/release/orbitkv-cache-manager \
  PYTHONPATH=. "../.venv/${engine}-release/bin/python" -m pytest -m e2e \
    tests/e2e/test_shared_cache.py -k "$engine" \
    --model /workspace/models/qwen3-8b \
    --basetemp="../benches/results/runs/shared-cache-${engine}"
done
```

Raw process logs and summaries remain in ignored `benches/results/runs/`.
The [qualification guide](../../../docs/shared-cache-qualification.md) also
provides a driver for already running replicas on separate hosts.
