use super::*;
use crate::cost::{CostObservationKind, ExecutionResource, Representation};
use crate::cost::{MIN_SAMPLES, enabled};
use std::time::Duration;

fn key(resource: u64) -> CostEstimateKey {
    CostEstimateKey::new(
        CostObservationKind::GpuLoadDirect,
        ExecutionResource::Gpu(resource),
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
                "cost::observation::tests::observations_require_explicit_process_opt_in",
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
        CostObservationKind::GpuLoadDirect,
        CostObservationKind::SsdUringRestore,
        CostObservationKind::SsdCufileRestore,
    ] {
        running.key.kind = path;
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
fn completion_observations_train_only_after_admitted_completion() {
    let elapsed = Duration::from_millis(25);
    assert_eq!(
        admitted_completion_seconds(true, Outcome::Completed, elapsed),
        Some(0.025)
    );
    for admitted in [false, true] {
        for outcome in [
            Outcome::Failed,
            Outcome::Cancelled,
            Outcome::TimedOut,
            Outcome::Abandoned,
        ] {
            assert_eq!(
                admitted_completion_seconds(admitted, outcome, elapsed),
                None
            );
        }
    }
    assert_eq!(
        admitted_completion_seconds(false, Outcome::Completed, elapsed),
        None
    );
}

#[test]
fn complete_routes_compare_enqueue_to_completion_despite_different_internal_admission() {
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
        (CostObservationKind::SsdUringRestore, 30, 0.1),
        (CostObservationKind::SsdCufileRestore, 80, 0.05),
        (CostObservationKind::LocalSsdHostReady, 60, 0.07),
        (CostObservationKind::PrefillToDecodeHandoff, 20, 0.11),
    ] {
        running.key.kind = path;
        running.submitted = Some(start + Duration::from_millis(submitted_ms));
        assert_eq!(
            running.service_sample(Outcome::Completed, end),
            Some(service)
        );
        assert_eq!(running.estimate_sample(Outcome::Completed, end), Some(0.13));
    }
}

#[test]
fn complete_route_submission_preserves_the_prediction_taken_at_enqueue() {
    let start = Instant::now();
    let prediction = Estimate {
        count: MIN_SAMPLES,
        seconds: 0.2,
        absolute_error: 0.03,
        updated: start,
    };
    for path in [
        CostObservationKind::SsdUringRestore,
        CostObservationKind::SsdCufileRestore,
        CostObservationKind::LocalSsdHostReady,
    ] {
        let mut observation = Observation(Some(Running {
            key: key(9998).with_observation_kind(path),
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
fn raw_shape_refinement_preserves_timing_and_cannot_relabel_composite_or_submitted_work() {
    let start = Instant::now();
    let admitted = start + Duration::from_millis(10);
    let refined = key(9997).with_dma_ranges(1);
    for (path, raw) in [
        (CostObservationKind::GpuLoadDirect, true),
        (CostObservationKind::GpuLoadKernel, true),
        (CostObservationKind::GpuSaveDirect, true),
        (CostObservationKind::GpuSaveKernel, true),
        (CostObservationKind::GpuDecode, false),
        (CostObservationKind::GpuEncode, false),
        (CostObservationKind::GpuSsdLoad, false),
        (CostObservationKind::GpuSsdSave, false),
        (CostObservationKind::SsdUringRestore, false),
        (CostObservationKind::SsdCufileRestore, false),
    ] {
        let original = key(9997).with_observation_kind(path);
        let candidate = if raw {
            refined.with_observation_kind(path)
        } else {
            refined
        };
        let mut observation = Observation(Some(Running {
            key: original,
            logical_bytes: Some(123),
            enqueued: start,
            admitted: Some(admitted),
            submitted: None,
            prediction: None,
        }));
        assert_eq!(observation.refine_raw_copy(candidate, 65536), raw);
        let mut running = observation.0.take().unwrap();
        assert_eq!(running.enqueued, start);
        assert_eq!(running.admitted, Some(admitted));
        assert_eq!(running.submitted, None);
        assert_eq!(running.key, if raw { candidate } else { original });
        assert_eq!(running.logical_bytes, Some(if raw { 65536 } else { 123 }));
        running.submitted = Some(start + Duration::from_millis(30));
        let expected_key = running.key;
        observation.0 = Some(running);
        assert!(!observation.refine_raw_copy(candidate.with_dma_ranges(4), 32768));
        let running = observation.0.take().unwrap();
        assert_eq!(running.key, expected_key);
        assert_eq!(
            running.service_sample(Outcome::Completed, start + Duration::from_millis(130)),
            Some(0.1)
        );
    }
    assert!(!Observation::disabled().refine_raw_copy(refined, 65536));
}
