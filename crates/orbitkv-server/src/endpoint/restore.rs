use std::sync::Arc;
use std::time::Instant;

use orbitkv_channel::RestoreCompletions;
use orbitkv_core::LoadOutcome;
use tokio::sync::oneshot;

/// The worker owns submitted DMA. This task only publishes its drained outcome;
/// losing a client or timing out a wait never manufactures a terminal result.
pub(super) async fn publish(
    receiver: oneshot::Receiver<LoadOutcome>,
    completions: Arc<RestoreCompletions>,
    epoch: u64,
    client_token: u64,
    id: u64,
    started: Instant,
) {
    let (result, completed_at) = match receiver.await {
        Ok(outcome) => (
            outcome.result.map_err(|error| error.to_string()),
            outcome.completed_at,
        ),
        Err(_) => {
            // A missing outcome does not prove that GPU destinations are safe.
            // Keep the record Pending until the session/Manager is discarded.
            log::error!(
                "Restore outcome lost: manager={epoch} client={client_token} operation={id}"
            );
            return;
        }
    };
    #[cfg(feature = "test-hooks")]
    while orbitkv_core::test_faults::active("restore") {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    crate::metric::timeline::record("restore_complete", || {
        serde_json::json!({
            "restore_key": format!("manager:{epoch}:{client_token}:{id}"),
            "elapsed_us": completed_at.saturating_duration_since(started).as_micros() as u64,
            "success": result.is_ok(),
        })
    });
    if let Err(error) = completions.complete(id, result) {
        log::error!("Cannot publish restore completion: {error}");
        return;
    }
    #[cfg(feature = "test-hooks")]
    if orbitkv_core::test_faults::active("notification") {
        return;
    }
    if let Err(error) = completions.notify() {
        log::error!("Cannot notify restore completion: {error}");
    }
    crate::metric::timeline::record("restore_notification", || {
        serde_json::json!({
            "restore_key": format!("manager:{epoch}:{client_token}:{id}"),
            "elapsed_us": completed_at.elapsed().as_micros() as u64,
        })
    });
}

#[cfg(test)]
#[path = "../../tests/unit/endpoint/restore.rs"]
mod tests;
