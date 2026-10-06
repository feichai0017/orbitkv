use std::time::Duration;

use tokio::sync::watch;

use super::wait_for_shutdown;

#[tokio::test]
async fn lifecycle_shutdown_wait_observes_an_already_latched_close() {
    let (shutdown, mut receiver) = watch::channel(false);
    shutdown.send_replace(true);

    tokio::time::timeout(Duration::from_millis(50), wait_for_shutdown(&mut receiver))
        .await
        .expect("a lifecycle waiter created after close must observe the durable state");
}
