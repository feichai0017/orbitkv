use std::time::Duration;

use orbitkv_channel::{BootstrapClient, BootstrapServer, RestoreState};
use orbitkv_core::EngineError;

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
    let id = records.reserve(&mut 1).unwrap();
    let (sender, receiver) = oneshot::channel();
    let task = tokio::spawn(publish(
        receiver,
        Arc::clone(&records),
        41,
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
    let id = session.completions().reserve(&mut 1).unwrap();
    let (sender, receiver) = oneshot::channel();
    drop(sender);
    publish(
        receiver,
        Arc::clone(session.completions()),
        41,
        id,
        Instant::now(),
    )
    .await;
    assert_eq!(
        client.completions().poll(id).unwrap().state,
        RestoreState::Pending
    );
}
