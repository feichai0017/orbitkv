use std::collections::BTreeMap;

use serde::Serialize;

use crate::fixed_state::{
    StateCheckpointPool, StateCompletionReceipt, StateCopyIntent, StateCopyReceipt,
    StatePublication, StateSlotLease, StateTransitionLease,
};

use super::{
    EngineAppendIntent, EngineCompletionEvidence, EngineRequestId, EngineStepAbortEvidence,
    EngineStepExecutionEvidence, PreparedFixedState, RuntimeSession, RuntimeSessionError,
};

type FixedStatePools = BTreeMap<u16, StateCheckpointPool>;
type PreparedFixedStates = Box<[PreparedFixedState]>;
type PublishedFixedStates = Box<[EngineFixedStatePublication]>;

/// Fixed-state source and destination selected by the session for one model step.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineFixedStatePlan {
    pub state_id: u16,
    pub source: Option<StateSlotLease>,
    pub destination: StateSlotLease,
    pub byte_count: u64,
}

/// Device observation for one session-authored fixed-state update.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineFixedStateEvidence {
    pub state_id: u16,
    pub source: Option<StateSlotLease>,
    pub destination: StateSlotLease,
    pub byte_count: u64,
    pub observed: bool,
    pub written: bool,
}

/// Current fixed-state slot published after device completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineFixedStatePublication {
    pub state_id: u16,
    pub request_id: EngineRequestId,
    pub slot: StateSlotLease,
}

impl EngineFixedStatePlan {
    pub(super) const fn from_intent(state_id: u16, intent: StateCopyIntent) -> Self {
        Self {
            state_id,
            source: intent.source,
            destination: intent.destination,
            byte_count: intent.byte_count,
        }
    }
}

impl RuntimeSession {
    pub(super) fn prepare_fixed_states(
        &self,
        intents: &[EngineAppendIntent],
    ) -> Result<(PreparedFixedStates, FixedStatePools), RuntimeSessionError> {
        let mut pools = self.fixed_states.clone();
        let mut prepared = Vec::with_capacity(
            intents
                .len()
                .checked_mul(pools.len())
                .ok_or(RuntimeSessionError::FixedStateEvidenceMismatch)?,
        );
        for (&state_id, pool) in &mut pools {
            let items = intents
                .iter()
                .map(|intent| (intent.request_id.0, pool.current(intent.request_id.0)))
                .collect::<Vec<_>>();
            let transitions = pool.prepare_batch(&items)?;
            prepared.extend(intents.iter().zip(transitions).map(|(intent, state)| {
                PreparedFixedState {
                    state_id,
                    request_id: intent.request_id,
                    intent: state,
                }
            }));
        }
        Ok((prepared.into_boxed_slice(), pools))
    }

    pub(super) fn abort_fixed_states(
        &self,
        states: &[PreparedFixedState],
        evidence: &[EngineStepAbortEvidence],
    ) -> Result<FixedStatePools, RuntimeSessionError> {
        let mut pools = self.fixed_states.clone();
        for (&state_id, pool) in &mut pools {
            let transitions = states
                .iter()
                .filter(|state| state.state_id == state_id)
                .map(|state| {
                    let unobserved = evidence
                        .iter()
                        .find(|item| item.request_id == state.request_id)
                        .ok_or(RuntimeSessionError::FixedStateEvidenceMismatch)?
                        .backend_unobserved;
                    Ok((state.intent.transition, unobserved))
                })
                .collect::<Result<Vec<_>, RuntimeSessionError>>()?;
            pool.abort_batch(&transitions)?;
        }
        Ok(pools)
    }

    pub(super) fn quarantine_prepared_fixed_states(
        &self,
        states: &[PreparedFixedState],
    ) -> Result<FixedStatePools, RuntimeSessionError> {
        let mut pools = self.fixed_states.clone();
        for (&state_id, pool) in &mut pools {
            pool.quarantine_prepared_batch(&transitions(states, state_id))?;
        }
        Ok(pools)
    }

    pub(super) fn submit_fixed_states(
        &self,
        states: &[PreparedFixedState],
        evidence: &[EngineStepExecutionEvidence],
    ) -> Result<FixedStatePools, RuntimeSessionError> {
        let receipts = states
            .iter()
            .map(|state| Ok((state.state_id, fixed_receipt(state, evidence)?)))
            .collect::<Result<Vec<_>, RuntimeSessionError>>()?;
        let mut pools = self.fixed_states.clone();
        for (&state_id, pool) in &mut pools {
            let state_receipts = receipts
                .iter()
                .filter(|(receipt_state_id, _)| *receipt_state_id == state_id)
                .map(|(_, receipt)| *receipt)
                .collect::<Vec<_>>();
            pool.submit_batch(&state_receipts)?;
        }
        Ok(pools)
    }

    pub(super) fn complete_fixed_states(
        &self,
        states: &[PreparedFixedState],
        evidence: EngineCompletionEvidence,
    ) -> Result<(PublishedFixedStates, FixedStatePools), RuntimeSessionError> {
        let mut pools = self.fixed_states.clone();
        let mut published = Vec::with_capacity(states.len());
        for (&state_id, pool) in &mut pools {
            let publications = pool.complete_batch(
                &transitions(states, state_id),
                StateCompletionReceipt {
                    engine_epoch: self.session_epoch,
                    completion_domain: evidence.completion_domain,
                    completion_value: evidence.completion_value,
                    confirmed: evidence.confirmed,
                },
            )?;
            for publication in publications {
                acknowledge_replaced_state(pool, publication)?;
                published.push(EngineFixedStatePublication {
                    state_id,
                    request_id: EngineRequestId(publication.owner_id),
                    slot: publication.slot,
                });
            }
        }
        Ok((published.into_boxed_slice(), pools))
    }

    pub(super) fn quarantine_submitted_fixed_states(
        &self,
        states: &[PreparedFixedState],
    ) -> Result<FixedStatePools, RuntimeSessionError> {
        let mut pools = self.fixed_states.clone();
        for (&state_id, pool) in &mut pools {
            pool.quarantine_submitted_batch(&transitions(states, state_id))?;
        }
        Ok(pools)
    }

    pub(super) fn quarantine_submitted_fixed_candidates(
        &mut self,
        mut pools: FixedStatePools,
        states: &[PreparedFixedState],
    ) {
        for (&state_id, pool) in &mut pools {
            if pool
                .quarantine_submitted_batch(&transitions(states, state_id))
                .is_err()
            {
                let _ = self.poison("fixed-state quarantine after page failure");
                return;
            }
        }
        self.fixed_states = pools;
    }

    pub(super) fn release_fixed_states(
        &self,
        request_ids: &[EngineRequestId],
    ) -> Result<FixedStatePools, RuntimeSessionError> {
        let mut pools = self.fixed_states.clone();
        for pool in pools.values_mut() {
            for request_id in request_ids {
                if let Some(slot) = pool.current(request_id.0) {
                    let retired = pool.retire_owner_after_last_completion(request_id.0, slot)?;
                    pool.acknowledge(retired)?;
                }
            }
        }
        Ok(pools)
    }
}

fn fixed_receipt(
    state: &PreparedFixedState,
    evidence: &[EngineStepExecutionEvidence],
) -> Result<StateCopyReceipt, RuntimeSessionError> {
    let step = evidence
        .iter()
        .find(|step| step.request_id == state.request_id)
        .ok_or(RuntimeSessionError::FixedStateEvidenceMismatch)?;
    let item = step
        .fixed_states
        .iter()
        .find(|item| item.state_id == state.state_id)
        .ok_or(RuntimeSessionError::FixedStateEvidenceMismatch)?;
    if item.source != state.intent.source
        || item.destination != state.intent.destination
        || item.byte_count != state.intent.byte_count
        || step
            .fixed_states
            .iter()
            .filter(|other| other.state_id == state.state_id)
            .count()
            != 1
    {
        return Err(RuntimeSessionError::FixedStateEvidenceMismatch);
    }
    if !item.observed || !item.written {
        return Err(RuntimeSessionError::FixedStateObservationUnknown);
    }
    Ok(StateCopyReceipt {
        transition: state.intent.transition,
        source: item.source.unwrap_or_default(),
        destination: item.destination,
        byte_count: item.byte_count,
        source_present: u8::from(item.source.is_some()),
        observed: u8::from(item.observed),
        written: u8::from(item.written),
        reserved8: 0,
        reserved32: 0,
    })
}

fn transitions(states: &[PreparedFixedState], state_id: u16) -> Vec<StateTransitionLease> {
    states
        .iter()
        .filter(|state| state.state_id == state_id)
        .map(|state| state.intent.transition)
        .collect()
}

fn acknowledge_replaced_state(
    pool: &mut StateCheckpointPool,
    publication: StatePublication,
) -> Result<(), RuntimeSessionError> {
    if let Some(retirement) = publication.retirement {
        pool.acknowledge(retirement)?;
    }
    Ok(())
}
