//! Exercise the production instrument definitions through the Manager's exporter.
//! This separate test binary installs its provider before the metrics OnceLock.

use opentelemetry::{KeyValue, global};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use prometheus::{Registry, TextEncoder, proto::MetricType};

// Core's metrics are crate-private. Reuse their source without widening the
// production API or copying instrument names, units, labels or bucket settings.
#[allow(
    dead_code,
    reason = "Only cost instruments are exercised in this exporter test"
)]
#[path = "../../orbitkv-core/src/metrics.rs"]
mod core_metrics;

#[test]
fn cost_histograms_export_documented_seconds_names_and_microsecond_buckets() {
    let registry = Registry::new();
    let exporter = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
        .unwrap();
    let provider = SdkMeterProvider::builder().with_reader(exporter).build();
    global::set_meter_provider(provider.clone());
    let metrics = core_metrics::core_metrics();
    let cases = [
        (
            &metrics.cost_stage_seconds,
            "orbitkv_cost_stage_seconds",
            vec![
                KeyValue::new("path", "gpu_load_direct"),
                KeyValue::new("outcome", "completed"),
                KeyValue::new("stage", "service"),
            ],
        ),
        (
            &metrics.cost_prediction_absolute_error_seconds,
            "orbitkv_cost_prediction_absolute_error_seconds",
            vec![KeyValue::new("path", "gpu_load_direct")],
        ),
        (
            &metrics.cost_shadow_prediction_seconds,
            "orbitkv_cost_shadow_prediction_seconds",
            vec![
                KeyValue::new("path", "gpu_load_direct"),
                KeyValue::new("evidence", "known"),
            ],
        ),
        (
            &metrics.cost_estimate_age_seconds,
            "orbitkv_cost_estimate_age_seconds",
            vec![
                KeyValue::new("path", "gpu_load_direct"),
                KeyValue::new("evidence", "known"),
            ],
        ),
        (
            &metrics.cost_estimate_error_seconds,
            "orbitkv_cost_estimate_error_seconds",
            vec![
                KeyValue::new("path", "gpu_load_direct"),
                KeyValue::new("evidence", "known"),
            ],
        ),
    ];
    for (instrument, _, labels) in &cases {
        instrument.record(0.000_007, labels);
        instrument.record(0.000_020, labels);
    }

    let families = registry.gather();
    let names: Vec<_> = families.iter().map(|family| family.name()).collect();
    for (_, name, labels) in &cases {
        let family = families
            .iter()
            .find(|family| family.name() == *name)
            .unwrap_or_else(|| panic!("Missing {name}; exported families: {names:?}"));
        assert_eq!(family.get_field_type(), MetricType::HISTOGRAM, "{name}");
        assert_eq!(family.get_metric().len(), 1, "{name}");
        let metric = &family.get_metric()[0];
        for label in labels {
            assert!(
                metric.get_label().iter().any(|pair| {
                    pair.name() == label.key.as_str() && pair.value() == label.value.as_str()
                }),
                "Missing {label:?} on {name}"
            );
        }
        let histogram = metric.get_histogram();
        assert_eq!(histogram.get_sample_count(), 2, "{name}");
        assert!(
            (histogram.get_sample_sum() - 0.000_027).abs() < 1e-12,
            "{name}"
        );
        for (bound, count) in [(0.000_005, 0), (0.000_010, 1), (0.000_025, 2)] {
            let bucket = histogram
                .get_bucket()
                .iter()
                .find(|bucket| bucket.upper_bound() == bound)
                .unwrap_or_else(|| panic!("Missing {bound}s bucket on {name}"));
            assert_eq!(bucket.cumulative_count(), count, "{name} le={bound}");
        }
    }

    let text = TextEncoder::new().encode_to_string(&families).unwrap();
    assert!(!text.contains("_seconds_seconds"));
    for (_, name, _) in &cases {
        for suffix in ["_bucket", "_count", "_sum"] {
            assert!(
                text.lines()
                    .any(|line| line.starts_with(&format!("{name}{suffix}{{"))),
                "Missing exported {name}{suffix} series"
            );
        }
    }
    provider.shutdown().unwrap();
}
