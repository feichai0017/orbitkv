use std::time::Duration;

use orbitkv_channel::{
    BootstrapClient, BootstrapServer, Command, CommandCode, CompletionError,
    RESPONSE_FLAG_REQUEST_CONSUMED, RestoreRequest, RestoreState, StatusCode,
};
use orbitkv_core::{EngineConfig, EngineError, OrbitKVEngine};

use super::*;

fn pair() -> (
    tempfile::TempDir,
    BootstrapServer,
    BootstrapClient,
    orbitkv_channel::BootstrapSession,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("restore.sock");
    let server = BootstrapServer::bind(&path, "orbitkv/test/outcome", 41, 16384, 1024).unwrap();
    let (client, session) = std::thread::scope(|scope| {
        let client = scope.spawn(|| BootstrapClient::connect(&path).unwrap());
        let session = server.accept().unwrap();
        (client.join().unwrap(), session)
    });
    (dir, server, client, session)
}

#[tokio::test]
async fn result_publication_waits_for_worker_outcome_and_outlives_disconnected_session() {
    let (_dir, _server, client, session) = pair();
    let records = Arc::clone(session.completions());
    let id = client.completions().reserve().unwrap();
    records.claim(id).unwrap();
    let (sender, receiver) = oneshot::channel();
    let task = tokio::spawn(publish(
        receiver,
        Arc::clone(&records),
        41,
        session.client_token(),
        id,
        Instant::now(),
    ));
    drop(session);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), async {
            while client.completions().poll(id).unwrap().state == RestoreState::Pending {
                tokio::task::yield_now().await;
            }
        })
        .await
        .is_err()
    );
    assert_eq!(
        client.completions().poll(id).unwrap().state,
        RestoreState::Pending
    );
    sender
        .send(LoadOutcome {
            result: Err(EngineError::InvalidArgument("drained GPU error".into())),
            completed_at: Instant::now(),
        })
        .unwrap();
    task.await.unwrap();
    let response = client.completions().poll(id).unwrap();
    assert_eq!(response.state, RestoreState::Failed);
    assert!(response.message.contains("drained GPU error"));
}

#[tokio::test]
async fn a_lost_worker_outcome_is_not_a_dma_completion() {
    let (_dir, _server, client, session) = pair();
    let id = client.completions().reserve().unwrap();
    session.completions().claim(id).unwrap();
    let (sender, receiver) = oneshot::channel();
    drop(sender);
    publish(
        receiver,
        Arc::clone(session.completions()),
        41,
        session.client_token(),
        id,
        Instant::now(),
    )
    .await;
    assert_eq!(
        client.completions().poll(id).unwrap().state,
        RestoreState::Pending
    );
}

#[tokio::test]
async fn malformed_and_rejected_requests_publish_claimed_results_without_response_payload() {
    let (_dir, server, client, session) = pair();
    let token = session.client_token();
    let mut sessions = std::collections::HashMap::from([(token, session)]);
    let engine = OrbitKVEngine::new_with_config(1 << 20, false, EngineConfig::default()).unwrap();
    let invalid_device = RestoreRequest {
        instance_id: "model".into(),
        tp_rank: 0,
        device_id: -1,
        layer_groups: Vec::new(),
        loads: Vec::new(),
    }
    .encode()
    .unwrap();
    for (payload, message) in [
        (Vec::new(), "truncated"),
        (invalid_device, "device_id -1 must be >= 0"),
    ] {
        let id = client.completions().reserve().unwrap();
        let descriptor = client.write_request(&payload).unwrap();
        let command = Command {
            code: CommandCode::Restore,
            request_id: id,
            session_epoch: 41,
            descriptor,
            arg0: token,
            arg1: id,
        };
        let response = super::super::dispatch_restore(
            command,
            &server,
            &mut sessions,
            &std::collections::HashMap::new(),
            &engine,
            &tokio::runtime::Handle::current(),
        );
        assert_eq!(response.status, StatusCode::Ok);
        assert_eq!(response.value1, RESPONSE_FLAG_REQUEST_CONSUMED);
        assert_eq!(response.descriptor, descriptor);
        assert_eq!(server.arena().read(descriptor).unwrap(), payload);
        client.complete_request(descriptor).unwrap();
        assert!(
            client
                .wait_for_notification(Duration::from_millis(10))
                .unwrap()
        );
        let result = client.completions().poll(id).unwrap();
        assert_eq!(result.state, RestoreState::Failed);
        assert!(result.message.contains(message), "{}", result.message);
    }
}

#[tokio::test]
async fn authentication_and_claim_guard_restore_execution_and_other_operation_results() {
    let (_dir, server, client, session) = pair();
    let token = session.client_token();
    let records = Arc::clone(session.completions());
    let mut sessions = std::collections::HashMap::from([(token, session)]);
    let engine = OrbitKVEngine::new_with_config(1 << 20, false, EngineConfig::default()).unwrap();
    let id = client.completions().reserve().unwrap();
    let descriptor = client.write_request(&[]).unwrap();
    let mut command = Command {
        code: CommandCode::Restore,
        request_id: id,
        session_epoch: 41,
        descriptor,
        arg0: token.wrapping_add(1),
        arg1: id,
    };
    let response = super::super::dispatch_restore(
        command,
        &server,
        &mut sessions,
        &std::collections::HashMap::new(),
        &engine,
        &tokio::runtime::Handle::current(),
    );
    assert_eq!(response.status, StatusCode::StaleSession);
    assert_eq!(response.value1, 0);
    assert!(client.completions().cancel(id).unwrap());

    command.arg0 = token;
    let active = client.completions().reserve().unwrap();
    records.claim(active).unwrap();
    for operation_id in [id, active] {
        command.arg1 = operation_id;
        command.descriptor = client.write_request(&[]).unwrap();
        let response = super::super::dispatch_restore(
            command,
            &server,
            &mut sessions,
            &std::collections::HashMap::new(),
            &engine,
            &tokio::runtime::Handle::current(),
        );
        assert_eq!(response.status, StatusCode::Invalid);
        assert_eq!(response.value1, RESPONSE_FLAG_REQUEST_CONSUMED);
        client.complete_request(command.descriptor).unwrap();
        assert_eq!(
            client.completions().poll(active).unwrap().state,
            RestoreState::Pending
        );
        assert!(matches!(
            client.completions().poll(id),
            Err(CompletionError::Stale(stale)) if stale == id
        ));
    }
    records.start_managed(active).unwrap();
    records.complete(active, Ok(())).unwrap();
    assert_eq!(
        client.completions().poll(active).unwrap().state,
        RestoreState::Succeeded
    );
}
