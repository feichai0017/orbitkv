# Shared-cache serving: final results

Qwen3-8B BF16, one H20 (97,871 MiB), TP=1 independent replicas, same-host
Mooncake TCP. vLLM 0.29.0 and SGLang 0.5.20 ran separately under Python 3.11.
Model revision: `b968826d9c46dd6066d109eabc6255188de91218`.
Runtime and gate: `0877c662141c7848c74bb9a90246dcc7d0d521f8`. This report keeps the
final rerun after adding reusable transfer windows and slot generations for lost
authorization replies. The prior completion/batching result remains in its
[immutable snapshot](https://github.com/feichai0017/orbitkv/blob/e24541850ef3f66e154ce83c6e107b83f81e06b3/benches/results/20260923-shared-cache/summary.csv).

Each engine passed two fresh-prefix restores (513/1025 input tokens), a restore
after restarting the sole catalog host, and recomputation after the source
restarted without payload. Each generated eight tokens. All outputs matched
their controls. Each engine restored 288 MiB through Mooncake and H2D across
three requests; the source-restart miss restored zero bytes. Query, inflight,
source-transfer, requester-completion and I/O gauges drained after every case.
Each engine received 12 source-release acknowledgements across its three restores.

Each replica had its own 2 GiB Manager pool, 1 GiB query budget and default
1 GiB source-transfer budget. Read batches were limited to 32 MiB. SSD,
warming and request preparation were disabled. Both engine caches were limited
to 4096 GPU tokens. vLLM used eager execution and `VLLM_BATCH_INVARIANT=1`;
SGLang used `--enable-deterministic-inference`. etcd 3.6.5 ran locally with
12-second Manager membership leases.

Compared with the [preceding gate](https://github.com/feichai0017/orbitkv/blob/a67610f1d76adc5d86c8da8e9c4ff6f72e712270/benches/results/20260923-shared-cache/summary.csv),
per-request discovery RPC counts changed as follows:

| Engine | 513 tokens | 1025 tokens |
| --- | --- | --- |
| vLLM | 8 → 3 | 14 → 6 |
| SGLang | 6 → 3 | 16 → 6 |

Shards on one catalog host now share a lookup; separate bounded read batches
still issue separate discovery requests. Source authorization and inventory
synchronization RPCs are separate and are not included in these counts.

`summary.csv` keeps the final rows and native stage totals, including release
acknowledgement counts and latency. Authorization includes window setup on first
use; totals per restored request were 1.055–1.345 ms for 513 tokens and
1.143–1.363 ms for 1025 tokens. These short serial
controls establish correctness, not throughput, tail-latency or competitor
superiority. The initial 513-token restore was slower than cold source computation
on both engines; startup/kernel effects are not isolated. This is not evidence of
two-host, RDMA, cross-engine, multi-rank or P/D support. Permanent requester loss
still needs proven transport revocation before orphaned source pins can be freed.

The same runtime passed 240 Rust tests, the native Mooncake/CUDA round trip and
the real-etcd two-Manager integration. Fault tests cover lost grant replies after
source pinning, completion before queued authorization, old slot generations,
idle-window eviction, control outages and completion retry isolation. They also
verify that cancelling an active READ retains destination buffers until native
completion. The serving gates above use normal requests and Manager restarts;
they do not inject every Rust-level fault into model serving.

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
