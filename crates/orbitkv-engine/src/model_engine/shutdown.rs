//! Explicit worker retirement and the final resource census.

use super::{EngineShared, EngineStats, ModelEngine, ModelEngineError, ModelWorker, WorkerCommand};
use std::sync::atomic::Ordering;

/// Resource census immediately before the worker destroys its decoder.
/// A successful shutdown joins that worker, so captured device resources have
/// also been retired by the time this report reaches the caller.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize)]
pub struct EngineShutdownReport {
    pub stats: EngineStats,
    pub fixed_states: Box<[(u16, orbitkv::StatePoolStats)]>,
    /// Cumulative full graph builds; excludes surgical provider recaptures.
    pub decoder_graph_builds: usize,
    /// Graphs retained just before decoder destruction.
    pub materialized_decoder_graphs: usize,
}

impl EngineShutdownReport {
    /// Whether all request-owned state and in-flight lifecycle work is retired.
    #[must_use]
    pub fn is_drained(&self) -> bool {
        let manager = &self.stats.manager;
        [
            self.stats.queued_requests,
            self.stats.active_requests,
            manager.active_requests,
            manager.active_snapshots,
            manager.active_prefixes,
            manager.prepared_steps,
            manager.submitted_steps,
            manager.reserved_pages,
            manager.writing_pages,
            manager.active_pages,
            manager.retiring_pages,
            manager.quarantined_pages,
            manager.pending_reclamations,
            manager.total_request_page_refs,
            manager.total_prefix_page_refs,
            manager.total_reader_pins,
        ]
        .into_iter()
        .all(|count| count == 0)
            && self.fixed_states.iter().all(|(_, state)| {
                state.free_slots == u64::from(state.identity.slot_count)
                    && [
                        state.reserved_slots,
                        state.copying_slots,
                        state.live_slots,
                        state.retiring_slots,
                        state.quarantined_slots,
                        state.active_owners,
                        state.pending_transitions,
                        state.pending_retirements,
                    ]
                    .into_iter()
                    .all(|count| count == 0)
            })
    }
}

impl EngineShared {
    pub(super) fn request_shutdown(&self) -> Result<(), ModelEngineError> {
        self.shutdown.store(true, Ordering::Release);
        let closed = self
            .registry
            .lock()
            .map(|mut registry| {
                registry.accepting = false;
                for cancelled in registry.cancellations.values() {
                    cancelled.store(true, Ordering::Release);
                }
            })
            .map_err(|_| ModelEngineError::WorkerUnavailable);
        // A full channel already wakes the worker; the atomic flag handles it
        // after that command. Send even if the registry lock was poisoned.
        let _ = self.commands.try_send(WorkerCommand::Shutdown);
        closed
    }
}

impl ModelEngine {
    /// Closes admission, cancels queued/active requests, and joins the worker.
    /// Applies to every clone of this engine handle. This call blocks; async
    /// callers should use their runtime's blocking-task facility.
    ///
    /// # Errors
    /// Returns worker execution/panic errors or incomplete state retirement.
    /// Returns `WorkerUnavailable` if another caller already joined the worker.
    pub fn shutdown(&self) -> Result<EngineShutdownReport, ModelEngineError> {
        self.shared.request_shutdown()?;
        let worker = self
            .shared
            .worker
            .lock()
            .map_err(|_| ModelEngineError::WorkerUnavailable)?
            .take()
            .ok_or(ModelEngineError::WorkerUnavailable)?;
        worker
            .join()
            .map_err(|payload| ModelEngineError::WorkerPanicked(super::panic_message(&payload)))?
    }
}

impl ModelWorker {
    pub(super) fn shutdown_report(&self) -> Result<EngineShutdownReport, ModelEngineError> {
        let graphs = self.decoder.graph_cache_stats();
        let report = EngineShutdownReport {
            stats: self.stats(0, 0),
            fixed_states: self.session.fixed_state_stats(),
            decoder_graph_builds: graphs.graph_builds,
            materialized_decoder_graphs: graphs.materialized_graphs,
        };
        if report.is_drained() {
            Ok(report)
        } else {
            Err(ModelEngineError::ShutdownIncomplete(Box::new(report)))
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model_engine/shutdown/mod.rs"]
mod tests;
