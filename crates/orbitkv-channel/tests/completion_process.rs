#![cfg(target_os = "linux")]

use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use orbitkv_channel::{
    BootstrapServer, CacheClient, CallOptions, ChannelError, CommandCode, GrantState,
    RESPONSE_FLAG_REQUEST_CONSUMED, Response, RestoreRequest, RestoreState, TransportServer,
};

struct ChildGuard(Child);

impl ChildGuard {
    fn start(directory: &Path) -> Self {
        let mut child = Self(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "shared_completion_child", "--nocapture"])
                .env("ORBITKV_COMPLETION_TEST_DIRECTORY", directory)
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while !directory.join("ready").exists() {
            assert!(Instant::now() < deadline, "child did not start");
            assert!(child.0.try_wait().unwrap().is_none(), "child exited early");
            std::thread::sleep(Duration::from_millis(1));
        }
        child
    }

    fn wait_for_success(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                assert!(status.success());
                return;
            }
            assert!(Instant::now() < deadline, "child did not exit");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn native_wait_consumes_success_and_error_across_processes_without_poll_rpc() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handles: Vec<_> = ["success", "failure"]
        .into_iter()
        .map(|instance| {
            client
                .start_restore(&RestoreRequest {
                    instance_id: instance.into(),
                    tp_rank: 0,
                    device_id: 0,
                    layer_groups: vec![],
                    loads: vec![],
                })
                .unwrap()
        })
        .collect();
    for &handle in &handles {
        assert_eq!(
            client.poll_restore(handle).unwrap().state,
            RestoreState::Pending
        );
    }
    std::fs::write(dir.path().join("complete"), b"").unwrap();
    assert_eq!(
        client
            .wait_restore(handles[0], Duration::from_secs(5))
            .unwrap()
            .state,
        RestoreState::Succeeded
    );
    let failed = client
        .wait_restore(handles[1], Duration::from_secs(5))
        .unwrap();
    assert_eq!(failed.state, RestoreState::Failed);
    assert_eq!(failed.message, "GPU copy failed after drain: 错误");
    assert!(
        client.poll_restore(handles[0]).is_err(),
        "terminal records are consumed once"
    );
    client.close();
    child.wait_for_success();
}

#[test]
fn pending_restore_reports_actual_peer_process_exit_without_fabricating_completion() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "pending-until-exit".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    assert_eq!(
        client.poll_restore(handle).unwrap().state,
        RestoreState::Pending
    );
    std::fs::write(dir.path().join("exit"), b"").unwrap();
    child.wait_for_success();
    assert!(matches!(
        client.poll_restore(handle),
        Err(ChannelError::SessionRequiresReconnect)
    ));
    assert!(matches!(
        client.wait_restore(handle, Duration::from_secs(1)),
        Err(ChannelError::SessionRequiresReconnect)
    ));
}

#[test]
fn local_grants_remain_pending_until_engine_drain_and_manager_reaping() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "local".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    assert_eq!(
        client.claim_local_restore(handle).unwrap().unwrap(),
        b"cross-process bounded plan"
    );
    assert_eq!(
        client.poll_restore(handle).unwrap().state,
        RestoreState::Pending
    );
    client.finish_local_restore(handle, Ok(()), None).unwrap();
    assert_eq!(
        client
            .wait_restore(handle, Duration::from_secs(5))
            .unwrap()
            .state,
        RestoreState::Succeeded
    );
    client.close();
    child.wait_for_success();
}

#[test]
fn manager_process_death_does_not_fence_active_engine_dma() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "local".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    client.claim_local_restore(handle).unwrap().unwrap();
    std::fs::write(dir.path().join("exit"), b"").unwrap();
    child.wait_for_success();
    assert_eq!(
        client.poll_restore(handle).unwrap().state,
        RestoreState::Pending
    );
    assert!(matches!(
        client.wait_restore(handle, Duration::from_millis(2)),
        Err(ChannelError::RestoreTimeout { .. })
    ));
    client.finish_local_restore(handle, Ok(()), None).unwrap();
    assert!(matches!(
        client.poll_restore(handle),
        Err(ChannelError::SessionRequiresReconnect)
    ));
}

#[test]
fn shared_completion_child() {
    let Some(directory) = std::env::var_os("ORBITKV_COMPLETION_TEST_DIRECTORY") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    let name = format!("orbitkv/test/completions/{}", std::process::id());
    let server = TransportServer::bind(&name).unwrap();
    let bootstrap =
        BootstrapServer::bind(directory.join("cache.sock"), &name, 91, 65536, 4096).unwrap();
    std::fs::write(directory.join("ready"), b"").unwrap();
    let mut session = bootstrap.accept().unwrap();
    let mut pending = Vec::new();
    let mut calls = 0;
    while session.is_alive().unwrap() {
        server
            .try_serve_for_epoch(91, |command| {
                assert_eq!(command.code, CommandCode::Restore);
                let slot = bootstrap
                    .descriptor_slot(command.descriptor.offset)
                    .unwrap();
                session
                    .validate_request(command.descriptor, command.arg0, slot)
                    .unwrap();
                // Any obsolete Poll command fails this decode. Only two Submits are allowed.
                let request =
                    RestoreRequest::decode(&bootstrap.arena().read(command.descriptor).unwrap())
                        .unwrap();
                calls += 1;
                assert!(calls <= 2);
                let id = command.arg1;
                session.completions().claim(id).unwrap();
                if request.instance_id == "local" {
                    session
                        .completions()
                        .publish_local(id, b"cross-process bounded plan")
                        .unwrap();
                    session.notify().unwrap();
                } else {
                    session.completions().start_managed(id).unwrap();
                    pending.push((id, request.instance_id == "failure"));
                }
                let mut response = Response::ok(command);
                response.value1 = RESPONSE_FLAG_REQUEST_CONSUMED;
                session.complete_request().unwrap();
                response
            })
            .unwrap();
        if directory.join("exit").exists() {
            assert_eq!(calls, 1);
            for (id, _) in &pending {
                assert_eq!(
                    session.completions().poll(*id).unwrap().state,
                    RestoreState::Pending
                );
            }
            return;
        }
        if directory.join("complete").exists() && !pending.is_empty() {
            for (id, failed) in pending.drain(..) {
                session
                    .completions()
                    .complete(
                        id,
                        if failed {
                            Err("GPU copy failed after drain: 错误".into())
                        } else {
                            Ok(())
                        },
                    )
                    .unwrap();
            }
            session.notify().unwrap();
        }
        for (id, state) in session.completions().manager_updates().unwrap() {
            match state {
                GrantState::Active => session.completions().release_plan(id).unwrap(),
                GrantState::Drained => session.completions().reap(id).unwrap(),
                _ => panic!("unexpected grant state: {state:?}"),
            }
        }
        std::thread::yield_now();
    }
    assert!(
        (1..=2).contains(&calls),
        "completion consumption must not issue Poll requests"
    );
}
