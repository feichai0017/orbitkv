#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, SystemTime};

use orbitkv_local::{CallOptions, Command, CommandCode, LocalClient, StatusCode};

#[test]
fn request_response_crosses_a_real_process_boundary() {
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let service_name = format!("orbitkv/test/{}/{nonce}", std::process::id());
    let mut child = ProcessCommand::new(env!("CARGO_BIN_EXE_orbitkv-local-echo"))
        .arg(&service_name)
        .arg("11")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut ready = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut ready)
        .unwrap();
    assert_eq!(ready.trim(), "READY");

    let client = LocalClient::connect(&service_name).unwrap();
    let mut ping = Command::ping(7, 11);
    ping.arg0 = 41;
    let response = client.call(ping, CallOptions::default()).unwrap();
    assert_eq!(response.status, StatusCode::Ok);
    assert_eq!(response.request_id, 7);
    assert_eq!(response.session_epoch, 11);
    assert_eq!(response.value0, 42);

    let stale = client
        .call(Command::ping(8, 10), CallOptions::default())
        .unwrap();
    assert_eq!(stale.status, StatusCode::StaleSession);
    assert_eq!(stale.request_id, 8);
    assert_eq!(stale.session_epoch, 11);

    let shutdown = Command {
        code: CommandCode::Shutdown,
        request_id: 9,
        session_epoch: 11,
        ..Command::ping(9, 11)
    };
    let response = client
        .call(
            shutdown,
            CallOptions {
                timeout: Duration::from_secs(2),
                ..CallOptions::default()
            },
        )
        .unwrap();
    assert_eq!(response.status, StatusCode::Ok);
    assert!(child.wait().unwrap().success());
}
