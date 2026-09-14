use std::collections::BTreeSet;

use crate::kv_manager::KvManagerError;

use super::{
    EngineAppendIntent, EngineBatchId, EngineControlId, EnginePublicationId, EngineReleaseId,
    EngineRequestId, PendingBatch, PreparedBatch, RequestPhase, RuntimeSession,
    RuntimeSessionError, SessionRequest, SubmittedBatch, allocate_sequence, was_issued,
};

impl RuntimeSession {
    pub(super) fn ensure_healthy(&self) -> Result<(), RuntimeSessionError> {
        match self.poisoned {
            Some(reason) => Err(RuntimeSessionError::SessionPoisoned(reason)),
            None => Ok(()),
        }
    }

    pub(super) fn poison(&mut self, reason: &'static str) -> RuntimeSessionError {
        let reason = *self.poisoned.get_or_insert(reason);
        RuntimeSessionError::SessionPoisoned(reason)
    }

    pub(super) fn allocate_batch_id(&mut self) -> Result<EngineBatchId, RuntimeSessionError> {
        let sequence = allocate_sequence(&mut self.next_batch_sequence, "batch")?;
        Ok(EngineBatchId::from_parts(self.session_epoch, sequence))
    }

    pub(super) fn allocate_publication_id(
        &mut self,
    ) -> Result<EnginePublicationId, RuntimeSessionError> {
        let sequence = allocate_sequence(&mut self.next_publication_sequence, "publication")?;
        Ok(EnginePublicationId::from_parts(
            self.session_epoch,
            sequence,
        ))
    }

    pub(super) fn allocate_release_id(&mut self) -> Result<EngineReleaseId, RuntimeSessionError> {
        let sequence = allocate_sequence(&mut self.next_release_sequence, "release")?;
        Ok(EngineReleaseId::from_parts(self.session_epoch, sequence))
    }

    pub(super) fn allocate_control_id(&mut self) -> Result<EngineControlId, RuntimeSessionError> {
        let sequence = self
            .next_control_sequence
            .checked_add(1)
            .ok_or(RuntimeSessionError::IdentityExhausted("control"))?;
        while self
            .controls
            .len()
            .checked_add(self.canceled_requests.len())
            .is_some_and(|count| count >= self.maximum_controls)
        {
            let Some(oldest) = self
                .canceled_requests
                .iter()
                .find_map(|(control_id, pending)| pending.finalized.then_some(*control_id))
            else {
                break;
            };
            self.canceled_requests.remove(&oldest);
        }
        if self
            .controls
            .len()
            .checked_add(self.canceled_requests.len())
            .is_none_or(|count| count >= self.maximum_controls)
        {
            return Err(KvManagerError::ArenaExhausted("control").into());
        }
        let current = self.next_control_sequence;
        self.next_control_sequence = sequence;
        Ok(EngineControlId::from_parts(self.session_epoch, current))
    }

    pub(super) fn finish_batch(
        &mut self,
        batch_id: EngineBatchId,
        request_ids: &[EngineRequestId],
        phase: RequestPhase,
    ) {
        for request_id in request_ids {
            self.requests
                .get_mut(request_id)
                .expect("batch preflight retained request")
                .phase = phase;
        }
        self.batches
            .remove(&batch_id)
            .expect("batch preflight retained operation");
    }

    pub(super) fn preflight_new_request_ids(
        &self,
        request_ids: &[EngineRequestId],
    ) -> Result<(), RuntimeSessionError> {
        if request_ids.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        for &request_id in request_ids {
            if !seen.insert(request_id) {
                return Err(RuntimeSessionError::DuplicateRequest(request_id));
            }
            if self.requests.contains_key(&request_id) {
                return Err(RuntimeSessionError::RequestAlreadyAcquired(request_id));
            }
        }
        Ok(())
    }

    pub(super) fn preflight_append_intents(
        &self,
        intents: &[EngineAppendIntent],
    ) -> Result<Vec<SessionRequest>, RuntimeSessionError> {
        if intents.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        intents
            .iter()
            .map(|intent| {
                if !seen.insert(intent.request_id) {
                    return Err(RuntimeSessionError::DuplicateRequest(intent.request_id));
                }
                self.ready_request(intent.request_id).cloned()
            })
            .collect()
    }

    pub(super) fn preflight_ready_requests(
        &self,
        request_ids: &[EngineRequestId],
    ) -> Result<Vec<SessionRequest>, RuntimeSessionError> {
        if request_ids.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        request_ids
            .iter()
            .copied()
            .map(|request_id| {
                if !seen.insert(request_id) {
                    return Err(RuntimeSessionError::DuplicateRequest(request_id));
                }
                self.ready_request(request_id).cloned()
            })
            .collect()
    }

    pub(super) fn ready_request(
        &self,
        request_id: EngineRequestId,
    ) -> Result<&SessionRequest, RuntimeSessionError> {
        let record = self
            .requests
            .get(&request_id)
            .ok_or(RuntimeSessionError::UnknownRequest(request_id))?;
        if record.phase != RequestPhase::Ready {
            return Err(RuntimeSessionError::RequestNotReady {
                request_id,
                state: record.phase.name(),
            });
        }
        Ok(record)
    }

    pub(super) fn prepared_batch(
        &self,
        batch_id: EngineBatchId,
    ) -> Result<&PreparedBatch, RuntimeSessionError> {
        self.ensure_batch_epoch(batch_id)?;
        match self.batches.get(&batch_id) {
            Some(PendingBatch::Prepared(batch)) => Ok(batch),
            Some(PendingBatch::Submitted(_)) => {
                Err(RuntimeSessionError::BatchNotPrepared(batch_id))
            }
            None => Err(self.batch_id_error(batch_id)),
        }
    }

    pub(super) fn submitted_batch(
        &self,
        batch_id: EngineBatchId,
    ) -> Result<&SubmittedBatch, RuntimeSessionError> {
        self.ensure_batch_epoch(batch_id)?;
        match self.batches.get(&batch_id) {
            Some(PendingBatch::Submitted(batch)) => Ok(batch),
            Some(PendingBatch::Prepared(_)) => {
                Err(RuntimeSessionError::BatchNotSubmitted(batch_id))
            }
            None => Err(self.batch_id_error(batch_id)),
        }
    }

    pub(super) fn preflight_request_phases(
        &mut self,
        request_ids: &[EngineRequestId],
        expected: RequestPhase,
    ) -> Result<(), RuntimeSessionError> {
        for &request_id in request_ids {
            let Some(record) = self.requests.get(&request_id) else {
                return Err(self.poison("pending operation lost request"));
            };
            if record.phase != expected {
                return Err(self.poison("pending request phase changed"));
            }
        }
        Ok(())
    }

    pub(super) fn ensure_publication_epoch(
        &self,
        publication_id: EnginePublicationId,
    ) -> Result<(), RuntimeSessionError> {
        foreign_epoch(publication_id.session_epoch, self.session_epoch)
            .map_err(|()| RuntimeSessionError::ForeignPublication(publication_id))
    }

    pub(super) fn ensure_release_epoch(
        &self,
        release_id: EngineReleaseId,
    ) -> Result<(), RuntimeSessionError> {
        foreign_epoch(release_id.session_epoch, self.session_epoch)
            .map_err(|()| RuntimeSessionError::ForeignRelease(release_id))
    }

    fn ensure_batch_epoch(&self, batch_id: EngineBatchId) -> Result<(), RuntimeSessionError> {
        foreign_epoch(batch_id.session_epoch, self.session_epoch)
            .map_err(|()| RuntimeSessionError::ForeignBatch(batch_id))
    }

    pub(super) fn batch_id_error(&self, batch_id: EngineBatchId) -> RuntimeSessionError {
        if batch_id.session_epoch != self.session_epoch {
            return RuntimeSessionError::ForeignBatch(batch_id);
        }
        if was_issued(batch_id.sequence, self.next_batch_sequence) {
            RuntimeSessionError::StaleBatch(batch_id)
        } else {
            RuntimeSessionError::UnknownBatch(batch_id)
        }
    }

    pub(super) fn publication_id_error(
        &self,
        publication_id: EnginePublicationId,
    ) -> RuntimeSessionError {
        if publication_id.session_epoch != self.session_epoch {
            return RuntimeSessionError::ForeignPublication(publication_id);
        }
        if was_issued(publication_id.sequence, self.next_publication_sequence) {
            RuntimeSessionError::StalePublication(publication_id)
        } else {
            RuntimeSessionError::UnknownPublication(publication_id)
        }
    }

    pub(super) fn release_id_error(&self, release_id: EngineReleaseId) -> RuntimeSessionError {
        if release_id.session_epoch != self.session_epoch {
            return RuntimeSessionError::ForeignRelease(release_id);
        }
        if was_issued(release_id.sequence, self.next_release_sequence) {
            RuntimeSessionError::StaleRelease(release_id)
        } else {
            RuntimeSessionError::UnknownRelease(release_id)
        }
    }
}

const fn foreign_epoch(received: u64, expected: u64) -> Result<(), ()> {
    if received == expected {
        Ok(())
    } else {
        Err(())
    }
}
