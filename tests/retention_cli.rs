use std::path::PathBuf;
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_orbitkv"))
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn run(arguments: &[&str]) -> serde_json::Value {
    let output = Command::new(binary())
        .args(arguments)
        .output()
        .expect("run orbitkv CLI");
    assert!(
        output.status.success(),
        "orbitkv failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("valid JSON output")
}

#[test]
fn legacy_and_retention_frontends_emit_identical_artifacts() {
    let legacy = root().join("examples/full_swa.json");
    let retention = root().join("examples/full_swa_retention.json");
    for command in ["emit-layout", "emit-sglang-policy"] {
        let legacy = run(&[command, legacy.to_str().unwrap()]);
        let retention = run(&[command, retention.to_str().unwrap()]);
        assert_eq!(legacy, retention);
    }
}

#[test]
fn retention_analysis_reports_derived_window_and_address() {
    let retention = root().join("examples/full_swa_retention.json");
    let report = run(&["analyze-retention", retention.to_str().unwrap()]);
    assert_eq!(report["schema"], "orbitkv.retention-analysis.v1");
    assert_eq!(report["analyses"][0]["inferred"]["kind"], "unbounded");
    assert_eq!(report["analyses"][1]["inferred"]["window_tokens"], 1024);
    assert_eq!(
        report["analyses"][1]["proven_query_key_delta_upper_bound"],
        1023
    );
    assert_eq!(
        report["layout"]["classes"][1]["address"]["period_blocks"],
        65
    );
}
