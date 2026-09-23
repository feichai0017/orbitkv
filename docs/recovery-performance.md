# Ordinary recovery profile

The 2026-09-22 Qwen3-8B profile measures ordinary demand before enabling
consumer-owned preparation. It identifies where to investigate next; it does
not establish a speedup from compiled hybrid ranges or from speculation.

## Setup and retained evidence

Source revision: `90bb88255597513be9c92ef32ed3f96ebee2c0e0`.
Model revision: `b968826d9c46dd6066d109eabc6255188de91218`.
One H20 in the development container, BF16, TP=1, vLLM 0.29.0 and SGLang 0.5.20.
Every run has 16,384 GPU cache tokens, 4 GiB Manager DRAM, 16 GiB SSD capacity,
a 2 GiB query budget and transfer tracing enabled. Speculative warming is off.
The SSD file uses verified `O_DIRECT`/io_uring on the container's mounted
filesystem. Its physical NVMe device and RDMA performance are not qualified.

Each engine has 234 measured requests across 54 bursts: shared or mixed
1,024/4,096-token prefixes, concurrency 1/4/8, three repetitions, cold requests,
post-HBM-pressure requests and forced Manager-DRAM-eviction requests. The
sustained runs mix 75% reusable-prefix choices with cold requests, use a
12-prefix working set, and stop admission after 20 seconds or 128 requests per
concurrency. They contain 356 vLLM and 353 SGLang requests. Outputs have 16 tokens.

[Final results](../benches/results/20260922-recovery-baseline/summary.csv)
retain aggregate latency, throughput, transfer bytes and output-difference
counts. [Reproduction commands](../benches/results/20260922-recovery-baseline/README.md)
record the workload. Raw manifests, samples, timelines and service logs remain
in ignored `benches/results/runs/` directories on the measurement host.

## Stage observations

Times below are milliseconds, measured within one process. A Manager restore
includes dispatch/worker queueing and the synchronized H2D task. Signal time is
the notification attempt after the worker finishes; delivery time ends when
the Manager answers the engine's terminal poll. They overlap and must not be
summed to obtain client TTFT.

| Burst stage | vLLM P95 | SGLang P95 |
| --- | ---: | ---: |
| Host preparation/read | 261.82 | 256.30 |
| Manager restore | 22.60 | 27.05 |
| Completion signal | 0.136 | 0.133 |
| Completion delivery | 6.18 | 0.244 |

Dense ordinary lookup combines candidate discovery with reading; there is no
separate candidate RPC in this path. Missing discovery samples are not zero
latency. The timeline also supports separate metadata-discovery observations
for hybrid recovery, but these Qwen3 runs do not exercise that path.

Host-read tails are larger than notification time in these runs. Sustained
vLLM additionally shows a completion-delivery P99 of **447.17 ms**, while its
Manager-restore P99 is **24.12 ms** and signal P99 is **0.145 ms**. This points
to delayed completion observation in the engine under load, rather than a slow
notification write. SGLang's sustained delivery P99 is 0.327 ms. Isolate engine
polling/compute overlap before attributing those tails to H2D bandwidth.

At concurrency eight with mixed prompts after forced host eviction, vLLM's
median/P95 TTFT is 343.05/549.07 ms and SGLang's is 292.43/497.38 ms. Their
output throughput is 197.71 and 205.78 tokens/s respectively. These are scoped
within-engine workload observations, not a universal engine or cache ranking.

The `after_pressure` phase sometimes reads SSD too: pressure can evict Manager
DRAM as well as HBM. Use the recorded byte counters to describe the actual
tier mixture. Do not relabel every post-HBM-pressure request as a DRAM hit.

## Cleanup and output limits

Every measured burst/window ends with zero query-reserved bytes. The explicit
settle-plus-drain check completes within 1.42 seconds, including a fixed
1.2-second settling interval; this is not the actual last-page release latency.
Sampled peaks remain within the configured query budget. Sampling every 25 ms
is a lower bound on peaks; the ownership tests separately enforce exact limits.

Ordinary greedy performance runs are not batch-invariant output proofs. The
burst profiles retain 25 vLLM and nine SGLang output differences from their cold
references; sustained profiles retain 27 and 13 differences from serial
preparation. These observations must remain visible. Deterministic Qwen3 fault
controls and exact GPU-byte gates are separate correctness evidence, not grounds
to erase performance-run differences. See [fault qualification](fault-qualification.md).

The subsequent [preparation controls](request-preparation.md#measured-results)
compare bounded demand, owned preparation and stopping policies. Preparation
stays opt-in because SGLang throughput improves while its P95 latency regresses.
