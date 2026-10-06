use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use super::ActiveTasks;

#[tokio::test]
async fn publish_drain_waits_for_every_admitted_continuation() {
    let active = Arc::new(ActiveTasks::default());
    let first = active.try_admit().unwrap();
    let second = active.try_admit().unwrap();
    let drain = tokio::spawn({
        let active = Arc::clone(&active);
        async move { active.close_and_drain().await }
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
    assert!(active.try_admit().is_none());
}

#[tokio::test]
async fn publish_close_rejects_a_prechecked_late_admission() {
    let active = Arc::new(ActiveTasks::default());
    let accepted = active.try_admit().unwrap();
    let (checked_tx, checked_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let worker = std::thread::spawn({
        let active = Arc::clone(&active);
        move || {
            assert!(active.is_accepting());
            checked_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
            result_tx.send(active.try_admit().is_some()).unwrap();
        }
    });
    checked_rx.recv().unwrap();

    let drain = tokio::spawn({
        let active = Arc::clone(&active);
        async move { active.close_and_drain().await }
    });
    tokio::task::yield_now().await;
    assert!(!drain.is_finished());
    drop(accepted);
    tokio::time::timeout(Duration::from_secs(1), drain)
        .await
        .expect("shutdown must drain work accepted before close")
        .expect("drain task must not panic");

    resume_tx.send(()).unwrap();
    assert!(
        !result_rx.recv().unwrap(),
        "a request prechecked before close must not register after close returns"
    );
    worker.join().unwrap();
    assert_eq!(active.active_count(), 0);
}

#[path = "shutdown.rs"]
mod shutdown;
