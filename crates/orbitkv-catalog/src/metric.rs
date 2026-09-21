use opentelemetry::metrics::{Counter, Histogram, ObservableGauge};
use opentelemetry::{KeyValue, global};
use std::sync::{Arc, LazyLock};
use std::time::Instant;
use tonic::Status;

use crate::store::{BlockHashStore, SweepStats};

// ---------------------------------------------------------------------------
// Store gauges
// ---------------------------------------------------------------------------

pub struct StoreMetrics {
    _gauges: Vec<ObservableGauge<u64>>,
}

/// Handles keep the observable callbacks alive with the Manager's service lifetime.
pub fn register_store_gauges(stores: &[Arc<BlockHashStore>]) -> StoreMetrics {
    let meter = global::meter("orbitkv_catalog");
    let mut gauges = Vec::new();
    for (name, kind, description) in [
        (
            "orbitkv_catalog_store_entries",
            0,
            "Indexed keys per catalog shard",
        ),
        (
            "orbitkv_catalog_block_owners",
            1,
            "Indexed replicas per catalog shard",
        ),
        (
            "orbitkv_catalog_metadata_bytes",
            2,
            "Accounted index and retry bytes per catalog shard",
        ),
    ] {
        let stores = stores.to_vec();
        gauges.push(
            meter
                .u64_observable_gauge(name)
                .with_description(description)
                .with_callback(move |observer| {
                    for (shard, store) in stores.iter().enumerate() {
                        let value = match kind {
                            0 => store.entry_count(),
                            1 => store.owner_count(),
                            _ => store.metadata_bytes() as u64,
                        };
                        observer.observe(value, &[KeyValue::new("shard", shard as i64)]);
                    }
                })
                .build(),
        );
    }
    StoreMetrics { _gauges: gauges }
}

// ---------------------------------------------------------------------------
// Node lifecycle sweep counters
// ---------------------------------------------------------------------------

struct SweepMetrics {
    removed_owners: Counter<u64>,
    removed_keys: Counter<u64>,
    removed_nodes: Counter<u64>,
}

static SWEEP_METRICS: LazyLock<SweepMetrics> = LazyLock::new(|| {
    let meter = global::meter("orbitkv_catalog");
    SweepMetrics {
        removed_owners: meter
            .u64_counter("orbitkv_catalog_sweep_removed_owners")
            .with_description("Total node ownership records removed by lifecycle sweep")
            .build(),
        removed_keys: meter
            .u64_counter("orbitkv_catalog_sweep_removed_keys")
            .with_description("Total block keys removed by lifecycle sweep")
            .build(),
        removed_nodes: meter
            .u64_counter("orbitkv_catalog_sweep_removed_nodes")
            .with_description("Total node records removed by lifecycle sweep")
            .build(),
    }
});

pub fn record_sweep(stats: SweepStats) {
    SWEEP_METRICS
        .removed_owners
        .add(stats.removed_owners as u64, &[]);
    SWEEP_METRICS
        .removed_keys
        .add(stats.removed_keys as u64, &[]);
    SWEEP_METRICS
        .removed_nodes
        .add(stats.removed_nodes as u64, &[]);
}

// ---------------------------------------------------------------------------
// RPC metrics
// ---------------------------------------------------------------------------

struct RpcMetrics {
    request_count: Counter<u64>,
    request_duration: Histogram<f64>,
}

impl RpcMetrics {
    fn new() -> Self {
        let meter = global::meter("orbitkv_catalog_rpc");
        let request_count = meter
            .u64_counter("orbitkv_catalog_rpc_requests")
            .with_description("Total RPC requests handled by the embedded catalog")
            .build();
        let request_duration = meter
            .f64_histogram("orbitkv_catalog_rpc_duration")
            .with_description("RPC latency in seconds")
            .with_unit("s")
            .with_boundaries(
                [
                    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
                ]
                .into(),
            )
            .build();
        Self {
            request_count,
            request_duration,
        }
    }

    fn record(&self, method: &'static str, status: &str, duration: f64) {
        let labels = [
            KeyValue::new("method", method.to_string()),
            KeyValue::new("status", status.to_string()),
        ];
        self.request_count.add(1, &labels);
        self.request_duration.record(duration, &labels);
    }
}

static RPC_METRICS: LazyLock<RpcMetrics> = LazyLock::new(RpcMetrics::new);

pub(crate) fn record_rpc_result<T>(
    method: &'static str,
    result: &Result<T, Status>,
    start: Instant,
) {
    let status = match result {
        Ok(_) => "ok".to_string(),
        Err(status) => status.code().to_string(),
    };
    let duration = start.elapsed().as_secs_f64();
    RPC_METRICS.record(method, &status, duration);
}
