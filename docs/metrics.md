# OrbitKV Metrics Guide

This guide explains how to collect, export, and visualize metrics from OrbitKV.

## Overview

OrbitKV supports two methods for exposing metrics:

### Method 1: Direct Prometheus (Recommended)

```
Cache Manager → Prometheus → Grafana
   (/metrics)      (scrape)    (visualize)
```

- Simpler deployment (2 components)
- OrbitKV exposes `/metrics` endpoint directly
- Use `examples/metric-prometheus/`

### Method 2: OTLP via OpenTelemetry Collector (Optional)

```
Cache Manager → OpenTelemetry Collector → Prometheus → Grafana
   (OTLP/gRPC)         (HTTP scrape)       (HTTP queries)
```

- More flexible (supports multiple backends)
- Useful if you already have OTel infrastructure
- Use `examples/metric/`

## Available Metrics

OrbitKV exposes the following metrics for monitoring KV cache operations:

### Pool Metrics (Pinned Memory)
- **orbitkv_pool_used_bytes** (Gauge)
  - Current pinned memory pool usage in bytes
  - Use case: Monitor memory pressure

- **orbitkv_pool_capacity_bytes** (Gauge)
  - Total pinned memory pool capacity in bytes
  - Use case: Derive pool utilization

- **orbitkv_pool_largest_free_bytes** (Gauge)
  - Largest contiguous free region in pinned pool (fragmentation signal)
  - Use case: Distinguish true exhaustion vs fragmentation (largest_free << free_bytes)

- **orbitkv_pool_alloc_failures_total** (Counter)
  - Total allocation failures after eviction retries
  - Use case: Detect memory exhaustion issues

### Cache Metrics (Block-level)
- **orbitkv_cache_block_hits_total** (Counter)
  - Legacy resident-cache hit counter
  - Use case: Backward-compatible dashboards only; do not use for tier
    contribution analysis

- **orbitkv_cache_block_misses_total** (Counter)
  - Legacy resident-cache miss counter
  - Use case: Backward-compatible dashboards only; do not use for tier
    contribution analysis

- **orbitkv_cache_tier_block_requests_total** (Counter)
  - Per-decision `query_prefetch` block attribution by `tier`
    (`ram`, `remote`, `ssd`, or `miss`)
  - Use case: Calculate overall hit ratio and each cache tier's contribution
    from one consistent denominator
  - Invariant: for each attributed decision,
    `ram + remote + ssd + miss == block_hashes.len()`
  - Semantics: this is decision attribution. `tier="remote"` and `tier="ssd"`
    mean the block was selected to be satisfied by that backing tier; they do
    not guarantee the later backing operation succeeded.

- **orbitkv_cache_block_insertions_total** (Counter)
  - New blocks inserted into cache
  - Use case: Track cache growth

- **orbitkv_cache_block_evictions_total** (Counter)
  - Blocks evicted from cache due to memory pressure
  - Use case: Monitor eviction frequency, tune pool size

- **orbitkv_cache_block_evictions_by_class_total** (Counter)
  - Blocks evicted from cache due to memory pressure, labelled by replacement class (`reclaimable` or `retained`)
  - Use case: Verify remote-fetched replicas are reclaimed before locally produced blocks

- **orbitkv_cache_block_evictions_still_referenced_total** (Counter)
  - Evicted blocks that still had external references (eviction did not immediately reclaim pinned memory)
  - Use case: Explain "evictions spike but pool_used_bytes doesn't drop"

- **orbitkv_cache_eviction_reclaimed_bytes_total** (Counter)
  - Estimated bytes actually reclaimed in pinned allocator after cache eviction
  - Use case: Measure effectiveness of eviction under real reference patterns

- **orbitkv_cache_resident_blocks** (Gauge)
  - Current number of sealed blocks resident in cache, labelled by replacement class (`reclaimable` or `retained`)
  - Use case: Track cache size and source-based replacement pressure in blocks

- **orbitkv_cache_resident_bytes** (Gauge)
  - Current sealed block bytes resident in cache (sum of footprints)
  - Use case: Attribute pinned pool usage to cache residency

- **orbitkv_cache_residence_duration_seconds** (Histogram)
  - RAM resident block lifetime from its first successful cache insertion to
    removal, measured in seconds
  - Labels: `reason` (`pressure` or `cleanup`)
  - `pressure`: allocator pressure removed the block through LRU reclaim
  - `cleanup`: the memory-cache cleanup endpoint removed the block
  - Buckets: `1s`, `5s`, `10s`, `30s`, `1m`, `2m`, `5m`, `10m`, `30m`,
    `1h`, `2h`, `6h`, `12h`, `24h`, and `+Inf`
  - Use case: Track typical and tail cache residence time and visualize the
    eviction-age distribution
  - Scope: each RAM residence is a separate lifetime. A block reinserted after
    eviction starts a new lifetime; cache hits, duplicate inserts, and
    replacement-class changes do not reset the original insertion time.
  - The lifetime ends when the block leaves the resident cache, even if an
    outstanding `Arc` keeps its pinned memory allocated. Correlate
    `orbitkv_cache_block_evictions_still_referenced_total` and
    `orbitkv_cache_eviction_reclaimed_bytes_total` to diagnose delayed memory
    reclamation.
  - SSD ring-cache overwrite and blocks still resident at server shutdown are
    not observed.

- **orbitkv_pinned_for_load_entries** (Gauge)
  - Current number of pinned_for_load entries (instance_id, block_key)
  - Use case: Diagnose load-path pins keeping evicted blocks alive

- **orbitkv_pinned_for_load_refs** (Gauge)
  - Current outstanding pinned_for_load consumer refcount (sum of per-entry counts)
  - Use case: Detect stuck consumers / missing release on load path

- **orbitkv_pinned_for_load_unique_blocks** (Gauge)
  - Current number of unique blocks referenced by pinned_for_load
  - Use case: Understand how many distinct blocks are being kept alive by pins

- **orbitkv_pinned_for_load_unique_bytes** (Gauge)
  - Current bytes referenced by pinned_for_load (unique blocks; sum of footprints)
  - Use case: Attribute pinned pool usage to load-path pins

### HLL Reuse Metrics
- **orbitkv_hll_cardinality** (Gauge)
  - Estimated distinct `(namespace, block hash)` objects classified as misses in a configured sliding window
  - Labels: `window` (`15m`, `1h`, `1d` by default)
  - Use case: Derive approximate prefix reuse over longer windows without
    storing every block hash

- **orbitkv_hll_total_requests** (Gauge)
  - Total queried blocks in the same configured sliding window, including ready blocks and duplicates
  - Labels: `window` (`15m`, `1h`, `1d` by default)
  - Use case: Denominator for HLL-based reference reuse rate

- **orbitkv_hll_estimated_hit_rate** (Gauge)
  - Server-computed miss-based infinite-cache reuse reference from the same
    HLL snapshot as the two gauges above
  - Labels: `window`
  - Value is clamped to `[0, 1]`

The existing cardinality and total metrics are retained. Existing PromQL
continues to work, but new dashboards should prefer the direct gauge because
it applies the same cardinality clamp as the tracker:

```promql
1 - (
  orbitkv_hll_cardinality{window="1h"}
  /
  clamp_min(orbitkv_hll_total_requests{window="1h"}, 1)
)
```

```promql
orbitkv_hll_estimated_hit_rate{window="1h"}
```

This is a metrics semantic update, not an `/metrics` protocol breaking change:
metric names, types, existing labels, and the HTTP endpoint are unchanged;
the new gauge is additive. The default HLL size changes from 16,384 registers
(`bucket_bits=14`, about 0.8% standard error) to 65,536 registers
(`bucket_bits=16`, about 0.4%). Three default windows use about 192 KiB of
register storage; sliding slots make the live tracker a few MiB per server.
The setting remains configurable with `--metric-hll-bucket-bits`.

### Save Metrics (GPU → CPU)
- **orbitkv_save_bytes_total** (Counter)
  - Total bytes saved from GPU to CPU storage
  - Use case: Monitor save throughput

- **orbitkv_save_duration_seconds** (Histogram)
  - Save operation latency distribution
  - Use case: Track save performance (p50, p99)

### Load Metrics (CPU → GPU)
- **orbitkv_load_bytes_total** (Counter)
  - Total bytes loaded from CPU storage to GPU
  - Use case: Monitor load throughput

- **orbitkv_load_duration_seconds** (Histogram)
  - Load operation latency distribution
  - Use case: Track load performance (p50, p99)

- **orbitkv_load_failures_total** (Counter)
  - Load operation failures (e.g., transfer errors)
  - Use case: Detect data transfer issues

### SSD Cache Metrics
- **orbitkv_ssd_write_bytes_total** (Counter) - Bytes written to SSD cache
- **orbitkv_ssd_write_duration_seconds** (Histogram) - Per-block write submission
  and completion latency. Concurrent block writes overlap; summing these
  durations does not measure wall-clock flush time.
- **orbitkv_ssd_prefetch_bytes_total** (Counter) - Successfully read and
  validated SSD bytes, including reads an engine may not subsequently consume.
- **orbitkv_ssd_prefetch_success_total** (Counter) - Successful SSD prefetches
- **orbitkv_ssd_prefetch_failures_total** (Counter) - Failed SSD prefetches
- **orbitkv_ssd_prefetch_duration_seconds** (Histogram) - Prefix prefetch latency
  for nonempty SSD candidates, including queueing, pinned-memory allocation,
  reads, and block reconstruction. Excludes H2D and does not prove a successful
  GPU restore; correlate with read bytes, failures, and load bytes.

### Tier Attribution Semantics

`orbitkv_cache_tier_block_requests_total{tier}` is the canonical metric for
explaining how each cache tier contributes to prefix-query hit ratio. It is
emitted once for each `query_prefetch` decision and uses exactly one label:
`tier`.

Tier values:

- `ram`: blocks already present in the resident RAM cache at the decision point
- `remote`: blocks selected to be satisfied by Mooncake remote fetch
- `ssd`: blocks selected to be satisfied by SSD prefetch
- `miss`: blocks no tier selected for that decision, including SSD prefetch
  backpressure and residual blocks after Mooncake partial availability

This metric intentionally records decisions, not completed service outcomes.
For backing failure correlation, use:

- `orbitkv_remote_fetch_total{status="error"}` for Mooncake fetch failures
- `orbitkv_ssd_prefetch_failures_total` for SSD prefetch failures

The legacy `orbitkv_cache_block_hits_total` and
`orbitkv_cache_block_misses_total` counters are retained for compatibility.
Their `Loading { hit, loading }` path only increments hits by `hit`; `loading`
does not enter the legacy denominator. New dashboards should use
`orbitkv_cache_tier_block_requests_total{tier}` instead of mixing legacy and
tier counters.

## Configuration

### Cache Manager Parameters

**Metrics Parameters:**

- `--http-addr`: HTTP server address for health check and Prometheus metrics (default: `0.0.0.0:9091`)
  - Always enabled for health check at `/health`
  - `/metrics` is enabled by default

- `--enable-prometheus`: Enable Prometheus `/metrics` endpoint (default: `true`)
  - When enabled, metrics are available at `http://<http-addr>/metrics`
  - Health check is always available at `http://<http-addr>/health`

- `--metrics-otel-endpoint`: OTLP gRPC endpoint for metrics export (optional)
  - Example: `http://127.0.0.1:4321`
  - Leave unset to disable OTLP export

- `--metrics-period-secs`: Metric export interval in seconds (default: `10`)
  - Only used when `--metrics-otel-endpoint` is set

- `--metric-hll-windows`: Comma-separated HLL sliding windows for estimated
  prefix reuse (default: `15m,1h,1d`)
  - Supported units: `s`, `m`, `h`, `d`
  - Each configured duration becomes a canonical `window` label. For example,
    the default config exports `window="15m"`, `window="1h"`, and `window="1d"`.
  - Empty entries such as `15m,,1h` and duplicate durations such as `1h,60m`
    are rejected at startup.

- `--metric-hll-bucket-bits`: HLL bucket index bits (default: `16`)
  - `2^16 = 65,536` registers per window and about 0.4% standard error.
  - Higher values use more memory and lower estimation error; `18` remains
    the supported maximum.

**Example: Prometheus Metrics**
```bash
cargo run -r --bin orbitkv-cache-manager -- \
  --addr 127.0.0.1:50055 \
  --devices 0 \
  --pool-size 30gb \
  --http-addr 0.0.0.0:9091 \
  --enable-prometheus
```

For an existing OpenTelemetry deployment, add
`--metrics-otel-endpoint http://127.0.0.1:4321` with the collector's configured
endpoint. Direct Prometheus remains available.

### Distributed inventory recovery

Manager synchronization counters are:

| Metric | Meaning |
| --- | --- |
| `orbitkv_inventory_records_sent` | Snapshot/delta records whose RPC acknowledgement was received |
| `orbitkv_inventory_snapshots_started` | Replacement inventory attempts |
| `orbitkv_inventory_snapshots_completed` | Acknowledged commits of complete inventory cuts |
| `orbitkv_inventory_history_gaps` | Retained history no longer covers directory progress |
| `orbitkv_inventory_sync_failures` | Failed inventory RPCs |
| `orbitkv_metaserver_heartbeat_failures` | Failed liveness/progress requests |
| `orbitkv_metaserver_unregister_failures` | Failed graceful owner cleanup |

These are background synchronization metrics, separate from request discovery
and Mooncake data transfer. Lost replies can undercount applied records; a
snapshot may retransmit already known entries. Repeated snapshot starts without
commits indicate failure to converge. See [directory recovery](../crates/orbitkv-metaserver/README.md).

### Environment Variables

- `RUST_LOG`: Control logging verbosity (e.g., `info,orbitkv_core=debug`)

## Quick Start: Direct Prometheus (Recommended)

The `examples/metric-prometheus/` directory provides a simple monitoring stack.

### 1. Start Cache Manager

```bash
# From repository root
cargo run -r --bin orbitkv-cache-manager -- \
  --addr 127.0.0.1:50055 \
  --devices 0 \
  --pool-size 30gb \
  --http-addr 0.0.0.0:9091 \
  --enable-prometheus
```

### 2. Start the Monitoring Stack

```bash
cd examples/metric-prometheus

docker compose up -d
# To stop: docker compose down
```

This starts two services:
- **Prometheus** (port: 9090) - Scrapes metrics from OrbitKV
- **Grafana** (port: 3000) - Visualizes metrics

### 3. Access Grafana Dashboard

1. Open browser: http://localhost:3000
2. Login: `admin` / `admin`
3. Navigate to **Dashboards** → **OrbitKV Metrics**

### 4. Test Metrics Endpoint

```bash
curl http://localhost:9091/metrics
```

## Architecture details

### Direct Prometheus

```
┌─────────────────┐
│ Cache Manager   │
│ :9091 /metrics  │
└────────┬────────┘
         │ Prometheus scrape
         ▼
┌─────────────────┐
│   Prometheus    │
│     :9090       │
└────────┬────────┘
         │ PromQL queries
         ▼
┌─────────────────┐
│    Grafana      │
│     :3000       │
└─────────────────┘
```

### Port Reference

| Service | Port or path | Protocol | Purpose |
| --- | --- | --- | --- |
| Cache Manager | `/tmp/orbitkv-<addr-port>.sock` | UDS and iceoryx2 | Inference process connection |
| Cache Manager | 50055 | gRPC | Peer authorization and lease control only when `--metaserver-addr` is set |
| Cache Manager | 9091 | HTTP | Health and Prometheus metrics |
| OTel Collector | configured endpoint | gRPC | Optional OTLP receiver |
| Prometheus | 9090 | HTTP | Query API and Web UI |
| Grafana | 3000 | HTTP | Dashboard UI |

## PromQL Query Examples

```promql
# Overall cache hit ratio from decision attribution (last 5 minutes)
sum(rate(orbitkv_cache_tier_block_requests_total{tier!="miss"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# RAM contribution to total requested blocks
sum(rate(orbitkv_cache_tier_block_requests_total{tier="ram"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# Remote contribution to total requested blocks
sum(rate(orbitkv_cache_tier_block_requests_total{tier="remote"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# SSD contribution to total requested blocks
sum(rate(orbitkv_cache_tier_block_requests_total{tier="ssd"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# Miss ratio from the same denominator
sum(rate(orbitkv_cache_tier_block_requests_total{tier="miss"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# Average save latency (p50)
histogram_quantile(0.5, rate(orbitkv_save_duration_seconds_bucket[5m]))

# Average load latency (p99)
histogram_quantile(0.99, rate(orbitkv_load_duration_seconds_bucket[5m]))

# Save throughput (MB/s)
rate(orbitkv_save_bytes_total[1m]) / 1e6

# Pool memory utilization
orbitkv_pool_used_bytes / orbitkv_pool_capacity_bytes

# RAM cache residence-time quantiles for pressure evictions
histogram_quantile(
  0.50,
  sum by (le) (
    rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

histogram_quantile(
  0.95,
  sum by (le) (
    rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

histogram_quantile(
  0.99,
  sum by (le) (
    rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

# RAM cache residence-time buckets for a Grafana heatmap
sum by (le) (
  rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
)

# HLL estimated hit rate for the 1h window (preferred)
orbitkv_hll_estimated_hit_rate{window="1h"}

# Backward-compatible derivation from the retained gauges
1 - (
  orbitkv_hll_cardinality{window="1h"}
  /
  clamp_min(orbitkv_hll_total_requests{window="1h"}, 1)
)

# HLL estimated hit rate for every configured window
1 - (
  orbitkv_hll_cardinality
  /
  clamp_min(orbitkv_hll_total_requests, 1)
)
```

## Troubleshooting

### Metrics not appearing (Direct Prometheus)

1. Check OrbitKV is exposing metrics:
   ```bash
   curl http://localhost:9091/metrics
   ```

2. Check Prometheus targets:
   - Open http://localhost:9090/targets
   - Verify `orbitkv` target is UP

3. If Docker cannot reach host, ensure `extra_hosts` is configured:
   ```yaml
   extra_hosts:
     - "host.docker.internal:host-gateway"
   ```

## Best Practices

1. **Monitor tier-attributed hit ratio** using `orbitkv_cache_tier_block_requests_total`
   - Low hit rate → consider increasing `--pool-size`

2. **Watch eviction rate**: High evictions indicate memory pressure
   - Use `rate(orbitkv_cache_block_evictions_total[5m])`

3. **Track allocation failures**: Any failures indicate critical issues
   - Alert on `orbitkv_pool_alloc_failures_total > 0`

4. **Analyze latency distributions**: Use histogram quantiles
   - p50: Typical case performance
   - p99: Worst-case user experience

## References

- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
