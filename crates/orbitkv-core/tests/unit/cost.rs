use super::*;

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
fn observations_require_explicit_process_opt_in() {
    const CHILD_EXPECTED: &str = "ORBITKV_TEST_COST_SWITCH_EXPECTED";
    if let Ok(expected) = std::env::var(CHILD_EXPECTED) {
        let expected = expected == "1";
        assert_eq!(enabled(), expected);
        assert_eq!(Observation::new(key(1), Some(4096)).0.is_some(), expected);
        return;
    }
    for setting in [None, Some("0"), Some("1"), Some("true")] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "cost::tests::observations_require_explicit_process_opt_in",
                "--nocapture",
            ])
            .env(CHILD_EXPECTED, if setting == Some("1") { "1" } else { "0" });
        if let Some(setting) = setting {
            child.env("ORBITKV_COST_OBSERVATIONS", setting);
        } else {
            child.env_remove("ORBITKV_COST_OBSERVATIONS");
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "setting={setting:?}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
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
    let retained = key(CAPACITY as u64).with_ssd_shape(131072, 8, 65536, 4);
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

#[test]
fn failed_cancelled_and_unsubmitted_operations_never_train_estimates() {
    let start = Instant::now();
    let mut running = Running {
        key: key(1),
        logical_bytes: Some(4096),
        enqueued: start,
        admitted: Some(start + Duration::from_millis(10)),
        submitted: Some(start + Duration::from_millis(30)),
        prediction: None,
    };
    let end = start + Duration::from_millis(130);
    for path in [
        CostPath::GpuLoadDirect,
        CostPath::SsdUringRestore,
        CostPath::SsdCufileRestore,
    ] {
        running.key.path = path;
        running.submitted = Some(start + Duration::from_millis(30));
        assert_eq!(running.service_sample(Outcome::Completed, end), Some(0.1));
        for outcome in [
            Outcome::Failed,
            Outcome::Cancelled,
            Outcome::TimedOut,
            Outcome::Abandoned,
        ] {
            assert_eq!(running.service_sample(outcome, end), None, "{path:?}");
            assert_eq!(running.estimate_sample(outcome, end), None, "{path:?}");
        }
        running.submitted = None;
        assert_eq!(running.service_sample(Outcome::Completed, end), None);
        assert_eq!(running.estimate_sample(Outcome::Completed, end), None);
    }
}

#[test]
fn ssd_routes_compare_enqueue_to_completion_despite_different_internal_admission() {
    let start = Instant::now();
    let end = start + Duration::from_millis(130);
    let mut running = Running {
        key: key(1),
        logical_bytes: Some(4096),
        enqueued: start,
        admitted: Some(start + Duration::from_millis(10)),
        submitted: Some(start + Duration::from_millis(30)),
        prediction: None,
    };
    assert_eq!(running.estimate_sample(Outcome::Completed, end), Some(0.1));

    for (path, submitted_ms, service) in [
        (CostPath::SsdUringRestore, 30, 0.1),
        (CostPath::SsdCufileRestore, 80, 0.05),
    ] {
        running.key.path = path;
        running.submitted = Some(start + Duration::from_millis(submitted_ms));
        assert_eq!(
            running.service_sample(Outcome::Completed, end),
            Some(service)
        );
        assert_eq!(running.estimate_sample(Outcome::Completed, end), Some(0.13));
    }
}

#[test]
fn ssd_route_submission_preserves_the_prediction_taken_at_enqueue() {
    let start = Instant::now();
    let prediction = Estimate {
        count: MIN_SAMPLES,
        seconds: 0.2,
        absolute_error: 0.03,
        updated: start,
    };
    for path in [CostPath::SsdUringRestore, CostPath::SsdCufileRestore] {
        let mut observation = Observation(Some(Running {
            key: CostKey { path, ..key(9998) },
            logical_bytes: Some(4096),
            enqueued: start,
            admitted: None,
            submitted: None,
            prediction: Some(prediction),
        }));
        observation.submitted();
        let running = observation.0.take().unwrap();
        assert!(running.submitted.is_some());
        let retained = running.prediction.expect("enqueue prediction retained");
        assert_eq!(retained.seconds, prediction.seconds);
        assert_eq!(retained.updated, prediction.updated);
    }
}

#[test]
fn repeated_submission_does_not_shorten_the_physical_operation() {
    let start = Instant::now() - Duration::from_secs(1);
    let mut observation = Observation(Some(Running {
        key: key(9999),
        logical_bytes: Some(1),
        enqueued: start,
        admitted: Some(start),
        submitted: Some(start),
        prediction: None,
    }));
    observation.admitted();
    observation.submitted();
    let running = observation.0.take().unwrap();
    assert_eq!(running.admitted, Some(start));
    assert_eq!(running.submitted, Some(start));
    let mut disabled = Observation::disabled();
    disabled.admitted();
    disabled.submitted();
    assert!(disabled.0.is_none());
}

#[test]
fn shadow_keeps_unknown_alternatives_unknown_and_never_uses_tier_rank() {
    let fast = Estimate {
        count: 10,
        seconds: 0.01,
        absolute_error: 0.0,
        updated: Instant::now(),
    };
    let slow = Estimate {
        seconds: 0.02,
        ..fast
    };
    assert_eq!(recommendation(&[Some(fast), None], 0), "unknown");
    assert_eq!(recommendation(&[None, Some(slow)], 1), "unknown");
    assert_eq!(recommendation(&[Some(fast)], 0), "unknown");
    assert_eq!(recommendation(&[Some(slow), Some(fast)], 0), "different");
    assert_eq!(recommendation(&[Some(fast), Some(slow)], 0), "agree");
}
