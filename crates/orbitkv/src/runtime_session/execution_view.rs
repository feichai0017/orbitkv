use serde::Serialize;

use super::{EngineBatchId, EngineRequestId, RequestPhase, RuntimeSession, RuntimeSessionError};

/// Read-only manager-authored page view for one prepared request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePreparedRequestView {
    pub request_id: EngineRequestId,
    pub previous_boundary: u64,
    pub target_boundary: u64,
    pub pages: Box<[crate::kv_manager::SnapshotPage]>,
}

/// Exact physical attention view for an already-prepared batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePreparedBatchView {
    pub batch_id: EngineBatchId,
    pub requests: Box<[EnginePreparedRequestView]>,
}

impl RuntimeSession {
    /// Materializes the manager-authored page view used by device attention for
    /// one prepared batch. No manager state is changed.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, submitted, or internally inconsistent batches.
    pub fn prepared_execution_view(
        &mut self,
        batch_id: EngineBatchId,
    ) -> Result<EnginePreparedBatchView, RuntimeSessionError> {
        self.ensure_healthy()?;
        let batch = self.prepared_batch(batch_id)?.clone();
        self.preflight_request_phases(&batch.requests, RequestPhase::Prepared(batch_id))?;
        let requests = batch
            .requests
            .iter()
            .copied()
            .zip(batch.steps.iter())
            .map(|(request_id, step)| {
                Ok(EnginePreparedRequestView {
                    request_id,
                    previous_boundary: step.previous_boundary,
                    target_boundary: step.target_boundary,
                    pages: self.manager.materialize_prepared_step(step)?,
                })
            })
            .collect::<Result<Vec<_>, RuntimeSessionError>>()?;
        Ok(EnginePreparedBatchView {
            batch_id,
            requests: requests.into_boxed_slice(),
        })
    }
}
