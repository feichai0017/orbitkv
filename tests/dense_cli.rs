use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_orbitkv"))
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the test intentionally records one complete JSONL sidecar lifecycle"
)]
fn dense_sidecar_executes_backend_bound_lifecycle() {
    let directory = tempfile::tempdir().unwrap();
    let state_plan_path = directory.path().join("runtime-state-plan.json");
    let output = Command::new(binary())
        .args([
            "compile-runtime-state-plan",
            root()
                .join("examples/gpt_oss_hybrid_tiny.json")
                .to_str()
                .unwrap(),
            "--eviction-interval",
            "32",
            "--execution-mode",
            "owner",
            "--owner-transport",
            "sidecar",
            "--capsule-enabled",
            "false",
            "--capsule-chunk-tokens",
            "128",
            "--capsule-max-payload-bytes",
            "1073741824",
            "--dense-max-requests",
            "1",
            "--dense-max-inflight",
            "4",
            "--dense-max-blocks",
            "16",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::write(&state_plan_path, &output.stdout).unwrap();

    let mut child = Command::new(binary())
        .arg("serve-dense-runtime")
        .arg(&state_plan_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    write_command(&mut stdin, &serde_json::json!({"op": "acquire_request"}));
    let acquired = read_response(&mut stdout);
    assert_eq!(acquired["status"], "request_acquired");
    let request = acquired["request"].clone();

    write_command(
        &mut stdin,
        &serde_json::json!({
            "op": "prepare_binding",
            "request": request,
            "boundary": 16,
        }),
    );
    let prepared = read_response(&mut stdout);
    assert_eq!(prepared["status"], "binding_prepared");
    let intent = &prepared["intent"];
    let artifact_fingerprint: serde_json::Value = serde_json::from_slice::<serde_json::Value>(
        &output.stdout,
    )
    .unwrap()["dense_runtime"]["artifact_fingerprint"]
        .clone();
    let blocks = intent["pending_blocks"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, block)| {
            serde_json::json!({
                "logical": block["logical"],
                "physical": block["physical"],
                "backend": {
                    "domain": block["logical"]["class_id"],
                    "index": index,
                },
                "payload_ready": true,
            })
        })
        .collect::<Vec<_>>();
    write_command(
        &mut stdin,
        &serde_json::json!({
            "op": "commit_binding",
            "receipt": {
                "schema": "orbitkv.dense-physical-binding-receipt.v1",
                "artifact_fingerprint": artifact_fingerprint,
                "binding_id": intent["binding_id"],
                "backend_transaction_id": "sglang:test-sidecar",
                "blocks": blocks,
            },
        }),
    );
    let committed = read_response(&mut stdout);
    assert_eq!(committed["status"], "binding_committed");
    assert_eq!(
        committed["blocks"].as_array().unwrap().len(),
        intent["pending_blocks"].as_array().unwrap().len()
    );

    write_command(
        &mut stdin,
        &serde_json::json!({
            "op": "advance_semantic_frontier",
            "request": request,
            "boundary": 16,
        }),
    );
    assert_eq!(
        read_response(&mut stdout)["status"],
        "semantic_frontier_advanced"
    );

    write_command(
        &mut stdin,
        &serde_json::json!({"op": "submit_view", "request": request}),
    );
    let submitted = read_response(&mut stdout);
    assert_eq!(submitted["status"], "view_submitted");
    assert!(
        submitted["view"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block.get("backend").is_some())
    );
    write_command(
        &mut stdin,
        &serde_json::json!({
            "op": "complete_submission",
            "submission_id": submitted["view"]["submission_id"],
        }),
    );
    assert_eq!(read_response(&mut stdout)["status"], "submission_completed");

    write_command(
        &mut stdin,
        &serde_json::json!({"op": "release_request", "request": request}),
    );
    let released = read_response(&mut stdout);
    assert_eq!(released["status"], "request_released");
    for certificate in released["certificates"].as_array().unwrap() {
        write_command(
            &mut stdin,
            &serde_json::json!({
                "op": "commit_reclamation",
                "receipt": {
                    "schema": "orbitkv.dense-physical-reclamation-receipt.v1",
                    "artifact_fingerprint": certificate["artifact_fingerprint"],
                    "certificate_id": certificate["certificate_id"],
                    "physical": certificate["physical"],
                    "backend": certificate["backend"],
                },
            }),
        );
        assert_eq!(
            read_response(&mut stdout)["status"],
            "reclamation_committed"
        );
    }
    write_command(
        &mut stdin,
        &serde_json::json!({"op": "recycle_request", "request": request}),
    );
    assert_eq!(read_response(&mut stdout)["status"], "request_recycled");
    write_command(&mut stdin, &serde_json::json!({"op": "stats"}));
    let stats = read_response(&mut stdout);
    assert_eq!(stats["stats"]["active_requests"], 0);
    assert_eq!(stats["stats"]["resident_blocks"], 0);

    drop(stdin);
    assert!(child.wait().unwrap().success());
}

fn write_command(stdin: &mut impl Write, command: &serde_json::Value) {
    serde_json::to_writer(&mut *stdin, command).unwrap();
    stdin.write_all(b"\n").unwrap();
    stdin.flush().unwrap();
}

fn read_response(stdout: &mut impl BufRead) -> serde_json::Value {
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}
