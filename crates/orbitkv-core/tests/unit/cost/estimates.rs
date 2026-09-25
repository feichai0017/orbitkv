use super::*;
use crate::cost::bucket;
use crate::cost::{CostPath, Representation};
use std::time::Duration;

fn key(resource: u64) -> CostKey {
    CostKey::new(
        CostPath::GpuLoadDirect,
        resource,
        Representation::Raw,
        65536,
        4,
    )
}

#[test]
fn estimates_are_bounded_and_isolate_resource_representation_and_shape() {
    let start = Instant::now();
    let mut estimates = Estimates::default();
    for resource in 0..(CAPACITY as u64 + 20) {
        estimates.observe(key(resource), 0.01, start + Duration::from_millis(resource));
        assert!(estimates.entries.len() <= CAPACITY);
    }
    assert!(!estimates.entries.contains_key(&key(0)));
    let retained = key(CAPACITY as u64)
        .with_ssd_shape(131072, 8, 65536, 4)
        .with_dma_ranges(1);
    for _ in 0..MIN_SAMPLES {
        estimates.observe(retained, 0.01, start + Duration::from_secs(1));
    }
    assert!(
        estimates
            .predict(retained, start + Duration::from_secs(1))
            .is_some()
    );
    for different in [
        CostKey {
            resource: 10000,
            ..retained
        },
        CostKey {
            representation: Representation::Ans,
            ..retained
        },
        CostKey {
            size: retained.size + 1,
            ..retained
        },
        CostKey {
            fragments: retained.fragments + 1,
            ..retained
        },
        // Whole-extent reads and requested SSD ranges are independent of the
        // total restore size, especially with mixed DRAM hits and TP slots.
        retained.with_ssd_shape(262144, 8, 65536, 4),
        retained.with_ssd_shape(131072, 16, 65536, 4),
        retained.with_ssd_shape(131072, 8, 32768, 4),
        retained.with_ssd_shape(131072, 8, 65536, 2),
        retained.with_dma_ranges(4),
        CostKey {
            path: CostPath::GpuLoadKernel,
            ..retained
        },
    ] {
        assert!(estimates.predict(different, start).is_none());
    }
    assert_eq!(bucket(0), 0);
    assert_eq!(bucket(u64::MAX), 64);
}

#[test]
fn replay_requires_recent_samples_and_reports_preupdate_error() {
    let start = Instant::now();
    let mut estimates = Estimates::default();
    for i in 0..MIN_SAMPLES {
        assert!(estimates.predict(key(1), start).is_none());
        estimates.observe(key(1), 0.01, start + Duration::from_millis(i));
    }
    let predicted = estimates
        .predict(key(1), start + Duration::from_secs(1))
        .unwrap();
    assert_eq!(predicted.seconds, 0.01);
    estimates.observe(key(1), 0.02, start + Duration::from_secs(1));
    let updated = estimates
        .predict(key(1), start + Duration::from_secs(1))
        .unwrap();
    assert!((updated.seconds - 0.012).abs() < 1e-10);
    assert!((updated.absolute_error - 0.002).abs() < 1e-10);
    let stale = start + MAX_AGE + Duration::from_secs(2);
    assert!(estimates.predict(key(1), stale).is_none());
    estimates.observe(key(1), 0.5, stale);
    assert!(estimates.predict(key(1), stale).is_none());
    assert_eq!(estimates.entries[&key(1)].count, 1);
    assert_eq!(estimates.entries[&key(1)].seconds, 0.5);
}
