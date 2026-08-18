use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use orbitkv::ContentDigest;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_orbitkv"))
}

fn identity() -> serde_json::Value {
    serde_json::json!({
        "namespace": [116, 101, 110, 97, 110, 116],
        "model_fingerprint": ContentDigest::sha256(b"model"),
        "tokenizer_fingerprint": ContentDigest::sha256(b"tokenizer"),
        "adapter_fingerprint": ContentDigest::sha256(b"adapter"),
        "state_plan_fingerprint": ContentDigest::sha256(b"plan"),
    })
}

#[test]
fn capsule_sidecar_publishes_and_restores_longest_prefix() {
    let directory = tempfile::tempdir().unwrap();
    let payload_path = directory.path().join("payload.bin");
    std::fs::write(&payload_path, b"kv-payload").unwrap();
    let mut child = Command::new(binary())
        .arg("serve-capsules")
        .arg(directory.path().join("catalog"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    write_command(
        &mut stdin,
        &serde_json::json!({
            "op": "publish",
            "identity": identity(),
            "chunk_tokens": 4,
            "token_ids": [1, 2, 3, 4],
            "live_token_count": 4,
            "payload_path": payload_path,
            "components": [{"state_class": "sglang-kv", "length_bytes": 10}],
            "created_unix_ms": 1,
        }),
    );
    let published = read_response(&mut stdout);
    assert_eq!(published["status"], "published");
    assert_eq!(published["prefix_token_count"], 4);
    assert_eq!(published["payload_bytes"], 10);

    write_command(
        &mut stdin,
        &serde_json::json!({
            "op": "restore",
            "identity": identity(),
            "chunk_tokens": 4,
            "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        }),
    );
    let restored = read_response(&mut stdout);
    assert_eq!(restored["status"], "restored");
    assert_eq!(restored["manifest"]["prefix_token_count"], 4);
    assert_eq!(
        std::fs::read(restored["payload_path"].as_str().unwrap()).unwrap(),
        b"kv-payload"
    );

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
