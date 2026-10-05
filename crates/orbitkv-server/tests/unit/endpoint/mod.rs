use std::sync::Arc;
use std::time::Duration;

use orbitkv_channel::{Command, StatusCode};

use super::{ActiveTasks, shutdown_response};

#[tokio::test]
async fn publish_drain_waits_for_every_admitted_continuation() {
    let active = Arc::new(ActiveTasks::default());
    let first = active.admit();
    let second = active.admit();
    let drain = tokio::spawn({
        let active = Arc::clone(&active);
        async move { active.drain().await }
    });

    tokio::task::yield_now().await;
    assert!(!drain.is_finished());
    drop(first);
    tokio::task::yield_now().await;
    assert!(!drain.is_finished());
    drop(second);
    tokio::time::timeout(Duration::from_secs(1), drain)
        .await
        .expect("drain should finish after the last Publish continuation")
        .expect("drain task should not panic");
}

#[test]
fn shutdown_rejection_preserves_request_identity_without_consuming_payload() {
    let command = Command::ping(17, 19);
    let response = shutdown_response(command, 19);
    assert_eq!(response.status, StatusCode::StaleSession);
    assert_eq!(response.request_id, 17);
    assert_eq!(response.session_epoch, 19);
    assert_eq!(response.descriptor, command.descriptor);
    assert_eq!(response.value1, 0);
}
