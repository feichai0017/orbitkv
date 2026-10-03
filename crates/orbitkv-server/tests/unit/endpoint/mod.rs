use std::sync::Arc;
use std::time::Duration;

use super::ActivePublishes;

#[tokio::test]
async fn publish_drain_waits_for_every_admitted_continuation() {
    let active = Arc::new(ActivePublishes::default());
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
