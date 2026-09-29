use std::collections::{HashMap, VecDeque};
use std::io;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::time::Instant;

use orbitkv_channel::{CompletionError, GrantState, RESTORE_COMPLETION_SLOTS, RestoreCompletions};
use orbitkv_core::{LoadOutcome, RawRestoreGrant};
use tokio::io::unix::AsyncFd;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

const LOCAL_PLAN_METADATA_BYTES: usize = 64 * 1024 * 1024;

/// Retains live or quarantined source owners independently of the UDS session.
pub(super) struct LocalGrants {
    sender: mpsc::Sender<SourceGrant>,
    metadata_budget: Arc<Semaphore>,
    records: Arc<RestoreCompletions>,
}

struct SourceGrant {
    id: u64,
    source: Option<RawRestoreGrant>,
    metadata: Option<OwnedSemaphorePermit>,
    records: Arc<RestoreCompletions>,
}

impl Drop for SourceGrant {
    #[allow(
        clippy::mem_forget,
        reason = "undrained engine DMA requires retaining source and session budgets until Manager exit"
    )]
    fn drop(&mut self) {
        let Some(source) = self.source.take() else {
            return;
        };
        // Even task cancellation/panic cannot turn an active grant into free
        // pool offsets. Retaining the mapping also retains its session budget.
        let safe = match self.records.state(self.id) {
            Ok(GrantState::Granted | GrantState::GrantedMore) => {
                self.records.revoke(self.id).unwrap_or(false)
                    || matches!(
                        self.records.state(self.id),
                        Ok(GrantState::PartDrained
                            | GrantState::Drained
                            | GrantState::Reaped
                            | GrantState::Acknowledged)
                    )
            }
            Ok(GrantState::Active | GrantState::ActiveMore) | Err(_) => false,
            Ok(_) => true,
        };
        if !safe {
            log::error!("Quarantining undrained local restore {}", self.id);
            std::mem::forget(source);
            std::mem::forget(self.metadata.take());
            std::mem::forget(Arc::clone(&self.records));
        }
    }
}

impl LocalGrants {
    pub(super) fn start(
        records: Arc<RestoreCompletions>,
        runtime: &tokio::runtime::Handle,
        epoch: u64,
        client_token: u64,
    ) -> io::Result<Self> {
        let notification = records.manager_notification_fd().try_clone()?;
        let _guard = runtime.enter();
        let notification = AsyncFd::new(notification)?;
        let (sender, receiver) = mpsc::channel(RESTORE_COMPLETION_SLOTS);
        runtime.spawn(reap_sources(
            Arc::clone(&records),
            notification,
            receiver,
            epoch,
            client_token,
        ));
        Ok(Self {
            sender,
            records,
            metadata_budget: Arc::new(Semaphore::new(LOCAL_PLAN_METADATA_BYTES)),
        })
    }

    pub(super) fn install(&self, id: u64, source: RawRestoreGrant) {
        let Ok(bytes) = u32::try_from(source.plan_bytes()) else {
            source.finish(false, None);
            let _ = self
                .records
                .reject(id, "restore plan metadata size overflow".into());
            return;
        };
        let Ok(metadata) = Arc::clone(&self.metadata_budget).try_acquire_many_owned(bytes) else {
            source.finish(false, None);
            let _ = self
                .records
                .reject(id, "restore session plan metadata budget exhausted".into());
            return;
        };
        let grant = SourceGrant {
            id,
            source: Some(source),
            metadata: Some(metadata),
            records: Arc::clone(&self.records),
        };
        if let Err(error) = self.sender.try_send(grant) {
            drop(error.into_inner());
            let _ = self
                .records
                .reject(id, "restore grant admission is full or closed".into());
        }
    }
}

async fn reap_sources(
    records: Arc<RestoreCompletions>,
    notification: AsyncFd<OwnedFd>,
    mut incoming: mpsc::Receiver<SourceGrant>,
    epoch: u64,
    client_token: u64,
) {
    let mut grants: HashMap<u64, SourceGrant> = HashMap::new();
    let mut pending = VecDeque::new();
    let mut connected = true;
    let mut maintenance = tokio::time::interval(std::time::Duration::from_millis(250));
    maintenance.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            item = incoming.recv(), if connected => match item {
                Some(owner) => {
                    let id = owner.id;
                    grants.insert(id, owner);
                    pending.push_back(id);
                }
                None => {
                    connected = false;
                    for id in pending.drain(..) {
                        grants.remove(&id);
                        let _ = records.reject(id, "restore session closed during preparation".into());
                    }
                    // Connection loss revokes only grants not yet claimed.
                    for &id in grants.keys() { let _ = records.revoke(id); }
                }
            },
            ready = notification.readable() => {
                match ready {
                    Ok(mut ready) => {
                        let mut bytes = [0; 8];
                        let _ = ready.try_io(|fd| rustix::io::read(fd.get_ref(), &mut bytes).map_err(io::Error::from));
                    }
                    Err(error) => {
                        log::error!("Restore retirement notification failed: {error}");
                        // Drop applies bounded quarantine to every active owner.
                        return;
                    }
                }
            }
            _ = maintenance.tick(), if !grants.is_empty() => {}
        }
        let updates = match records.manager_updates() {
            Ok(updates) => updates,
            Err(error) => {
                log::error!("Cannot read restore retirements: {error}");
                return;
            }
        };
        for (id, state) in updates {
            if matches!(
                state,
                GrantState::Active
                    | GrantState::ActiveMore
                    | GrantState::PartDrained
                    | GrantState::Drained
                    | GrantState::Revoked
            ) && let Err(error) = records.release_plan(id)
            {
                log::error!("Cannot retire restore plan {id}: {error}");
            }
            match state {
                GrantState::PartDrained => {
                    #[cfg(feature = "test-hooks")]
                    orbitkv_core::test_faults::pause("local_restore_part_drained").await;
                    if !connected {
                        let _ = records.revoke(id);
                        continue;
                    }
                    // A leftover dirty bit may observe the same drain again.
                    // Only the acknowledgement winner advances the source plan.
                    if records.continue_local(id).is_err() {
                        continue;
                    }
                    if grants
                        .get_mut(&id)
                        .and_then(|owner| owner.source.as_mut())
                        .is_some_and(RawRestoreGrant::advance_plan)
                    {
                        pending.push_back(id);
                    } else {
                        if let Some(mut owner) = grants.remove(&id)
                            && let Some(source) = owner.source.take()
                        {
                            source.finish(false, None);
                        }
                        let _ = records.reject(id, "missing next Restore plan part".into());
                    }
                }
                GrantState::Drained | GrantState::Revoked => {
                    #[cfg(feature = "test-hooks")]
                    orbitkv_core::test_faults::pause("local_restore_reap").await;
                    if let Some(mut owner) = grants.remove(&id) {
                        if let Some(source) = owner.source.take() {
                            let success = state == GrantState::Drained
                                && records.drain_succeeded(id).unwrap_or(false);
                            let timing = records.drain_timing(id).ok().flatten();
                            source.finish(
                                success,
                                timing.map(|timing| {
                                    std::time::Duration::from_nanos(timing.drained_ns)
                                }),
                            );
                            if let Some(timing) = timing {
                                orbitkv_common::timeline::record("local_restore_complete", || {
                                    serde_json::json!({
                                        "restore_key": format!("manager:{epoch}:{client_token}:{id}"),
                                        "success": success,
                                        "readiness_ns": timing.readiness_ns,
                                        "dispatched_ns": timing.dispatched_ns,
                                        "dequeued_ns": timing.dequeued_ns,
                                        "claimed_ns": timing.claimed_ns,
                                        "submitted_ns": timing.submitted_ns,
                                        "drained_ns": timing.drained_ns,
                                    })
                                });
                            }
                        }
                        if let Err(error) = records.reap(id) {
                            log::error!("Cannot reap restore source {id}: {error}");
                        }
                    }
                }
                _ => {}
            }
        }
        while let Some(&id) = pending.front() {
            let Some(source) = grants.get(&id).and_then(|owner| owner.source.as_ref()) else {
                pending.pop_front();
                continue;
            };
            #[cfg(feature = "test-hooks")]
            while orbitkv_core::test_faults::active("restore") {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
            let (plan, more) = source.encoded_plan();
            match records.publish_local(id, plan, more) {
                Ok(true) => {
                    #[cfg(feature = "test-hooks")]
                    let notify = !orbitkv_core::test_faults::active("notification");
                    #[cfg(not(feature = "test-hooks"))]
                    let notify = true;
                    if notify {
                        let _ = records.notify();
                    }
                }
                Ok(false) => {
                    grants.remove(&id);
                    let _ = records.finish_cancelled(id);
                }
                Err(CompletionError::PlanFull) => break,
                Err(error) => {
                    grants.remove(&id);
                    let _ = records.reject(id, error.to_string());
                }
            }
            pending.pop_front();
        }
        if !connected && grants.is_empty() {
            return;
        }
    }
}

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
    if let Err(error) = completions.start_managed(id) {
        log::error!("Cannot mark managed Restore active: {error}");
        // The core may already own submitted work. Always await its drain.
        let _ = receiver.await;
        return;
    }
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
    orbitkv_common::timeline::record("restore_complete", || {
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
    orbitkv_common::timeline::record("restore_notification", || {
        serde_json::json!({
            "restore_key": format!("manager:{epoch}:{client_token}:{id}"),
            "elapsed_us": completed_at.elapsed().as_micros() as u64,
        })
    });
}

#[cfg(test)]
#[path = "../../tests/unit/endpoint/restore.rs"]
mod tests;
