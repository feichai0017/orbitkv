use super::*;
use crate::cost::{CostPath, Representation, Resource};

fn key(path: CostPath, peer: u64) -> CostKey {
    CostKey::new(path, Resource::Peer(peer), Representation::Raw, 4096, 2)
}

fn estimate(seconds: f64, error: f64) -> Estimate {
    Estimate {
        count: 8,
        seconds,
        absolute_error: error,
        updated: Instant::now(),
    }
}

#[test]
fn selection_requires_complete_compatible_fresh_evidence() {
    let candidates = [
        key(CostPath::PeerDramHostReady, 1),
        key(CostPath::PeerDramHostReady, 2),
    ];
    assert_eq!(
        choose(&candidates, &[Some(estimate(0.02, 0.0)), None], 0),
        (0, "unknown")
    );
    assert_eq!(
        choose(
            &[
                candidates[0],
                CostKey::new(
                    CostPath::PeerSsdHostReady,
                    Resource::Peer(2),
                    Representation::Ans,
                    4096,
                    2,
                ),
            ],
            &[Some(estimate(0.02, 0.0)), Some(estimate(0.01, 0.0)),],
            0,
        ),
        (0, "incomparable")
    );
    assert_eq!(
        choose(
            &[
                candidates[0],
                CostKey::new(
                    CostPath::LocalSsdHostReady,
                    Resource::SsdStore(9),
                    Representation::Raw,
                    4096,
                    2,
                ),
            ],
            &[Some(estimate(0.02, 0.0)), Some(estimate(0.01, 0.0)),],
            0,
        ),
        (1, "selected"),
        "different resources and media remain comparable for the same HostReady demand"
    );
}

#[test]
fn selection_requires_gain_beyond_error_and_margin() {
    let candidates = [
        key(CostPath::PeerDramHostReady, 1),
        key(CostPath::PeerDramHostReady, 2),
        key(CostPath::PeerDramHostReady, 3),
    ];
    assert_eq!(
        choose(
            &candidates,
            &[
                Some(estimate(0.02, 0.001)),
                Some(estimate(0.01, 0.001)),
                Some(estimate(0.015, 0.001))
            ],
            0,
        ),
        (1, "selected")
    );
    assert_eq!(
        choose(
            &candidates[..2],
            &[Some(estimate(0.02, 0.006)), Some(estimate(0.01, 0.005)),],
            0,
        ),
        (0, "within_margin")
    );
    assert_eq!(
        choose(
            &candidates[..2],
            &[Some(estimate(0.02, 0.0)), Some(estimate(0.0195, 0.0)),],
            0,
        ),
        (0, "within_margin")
    );
}

#[test]
fn execution_selection_requires_both_explicit_switches() {
    const CHILD: &str = "ORBITKV_TEST_ROUTE_SELECTION_CHILD";
    if let Ok(expected) = std::env::var(CHILD) {
        let candidates = [
            key(CostPath::PeerDramHostReady, 101),
            key(CostPath::PeerDramHostReady, 102),
        ];
        let now = Instant::now();
        for _ in 0..super::super::MIN_SAMPLES {
            ESTIMATES.lock().observe(candidates[0], 0.02, now);
            ESTIMATES.lock().observe(candidates[1], 0.01, now);
        }
        assert_eq!(
            select_route(&candidates, 0, SelectionScope::PeerOwner),
            expected.parse::<usize>().unwrap()
        );
        return;
    }

    for (observations, selection, expected) in [
        (None, None, 0),
        (Some("1"), None, 0),
        (None, Some("1"), 0),
        (Some("1"), Some("1"), 1),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "cost::decision::tests::execution_selection_requires_both_explicit_switches",
                "--nocapture",
            ])
            .env(CHILD, expected.to_string());
        if let Some(value) = observations {
            child.env("ORBITKV_COST_OBSERVATIONS", value);
        } else {
            child.env_remove("ORBITKV_COST_OBSERVATIONS");
        }
        if let Some(value) = selection {
            child.env("ORBITKV_COST_SELECTION", value);
        } else {
            child.env_remove("ORBITKV_COST_SELECTION");
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "observations={observations:?} selection={selection:?}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

#[test]
fn cross_medium_selection_requires_three_explicit_switches() {
    const CHILD: &str = "ORBITKV_TEST_CROSS_MEDIUM_SELECTION_CHILD";
    if let Ok(expected) = std::env::var(CHILD) {
        assert_eq!(
            super::super::cross_medium_selection_enabled(),
            expected == "1"
        );
        return;
    }

    for (observations, selection, cross_medium, expected) in [
        (None, None, None, false),
        (Some("1"), Some("1"), None, false),
        (Some("1"), None, Some("1"), false),
        (None, Some("1"), Some("1"), false),
        (Some("1"), Some("1"), Some("1"), true),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "cost::decision::tests::cross_medium_selection_requires_three_explicit_switches",
                "--nocapture",
            ])
            .env(CHILD, if expected { "1" } else { "0" });
        for (name, value) in [
            ("ORBITKV_COST_OBSERVATIONS", observations),
            ("ORBITKV_COST_SELECTION", selection),
            ("ORBITKV_CROSS_MEDIUM_SELECTION", cross_medium),
        ] {
            if let Some(value) = value {
                child.env(name, value);
            } else {
                child.env_remove(name);
            }
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "observations={observations:?} selection={selection:?} cross={cross_medium:?}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
