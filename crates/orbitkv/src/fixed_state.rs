//! Generation-safe storage for recurrent and convolution state.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct StateSlotLease {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub generation: u64,
    pub slot_id: u32,
    pub pool_id: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct StateTransitionLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct StateRetirementLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct StateCopyIntent {
    pub transition: StateTransitionLease,
    pub owner_id: u64,
    pub source: Option<StateSlotLease>,
    pub destination: StateSlotLease,
    pub byte_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct StateCopyReceipt {
    pub transition: StateTransitionLease,
    pub source: StateSlotLease,
    pub destination: StateSlotLease,
    pub byte_count: u64,
    pub source_present: u8,
    pub observed: u8,
    pub written: u8,
    pub reserved8: u8,
    pub reserved32: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct StateCompletionReceipt {
    pub engine_epoch: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
    pub confirmed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct StateRetirementCertificate {
    pub retirement: StateRetirementLease,
    pub slot: StateSlotLease,
    pub byte_count: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct StatePublication {
    pub owner_id: u64,
    pub slot: StateSlotLease,
    pub retirement: Option<StateRetirementCertificate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotPhase {
    Free,
    Reserved(StateTransitionLease),
    Copying(StateTransitionLease),
    Live(u64),
    Retiring(StateRetirementLease),
    Quarantined,
}

#[derive(Clone, Copy, Debug)]
struct SlotState {
    generation: u64,
    phase: SlotPhase,
}

#[derive(Clone, Copy, Debug)]
struct TransitionState {
    owner_id: u64,
    source: Option<StateSlotLease>,
    destination: StateSlotLease,
    submitted: bool,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StateCheckpointError {
    #[error("state pool geometry must be positive")]
    InvalidGeometry,
    #[error("state owner already has a pending transition")]
    OwnerBusy,
    #[error("state owner is quarantined")]
    OwnerQuarantined,
    #[error("state pool is exhausted")]
    PoolExhausted,
    #[error("state lease is stale or belongs to another pool")]
    StaleLease,
    #[error("state transition is stale")]
    StaleTransition,
    #[error("state copy receipt is malformed")]
    CopyReceiptMismatch,
    #[error("state copy observation is unknown")]
    CopyObservationUnknown,
    #[error("state transition has already been submitted")]
    AlreadySubmitted,
    #[error("state transition has not been submitted")]
    NotSubmitted,
    #[error("state completion is not confirmed")]
    CompletionNotConfirmed,
    #[error("state retirement acknowledgement is malformed")]
    RetirementMismatch,
    #[error("state generation is exhausted")]
    GenerationExhausted,
}

/// Generation-checked double-buffer pool for recurrent and convolution state.
/// It deliberately has no token ids or token-move operation.
#[derive(Clone, Debug)]
pub struct StateCheckpointPool {
    engine_epoch: u64,
    pool_epoch: u64,
    pool_id: u32,
    byte_count: u64,
    slot_count: u32,
    slots: Vec<SlotState>,
    free: Vec<u32>,
    owners: BTreeMap<u64, StateSlotLease>,
    owner_operations: BTreeMap<u64, StateTransitionLease>,
    quarantined_owners: BTreeSet<u64>,
    operations: BTreeMap<StateTransitionLease, TransitionState>,
    retirements: BTreeMap<StateRetirementLease, StateRetirementCertificate>,
    next_operation: u32,
    next_retirement: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct StatePoolIdentity {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub byte_count: u64,
    pub pool_id: u32,
    pub slot_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct StatePoolStats {
    pub identity: StatePoolIdentity,
    pub free_slots: u64,
    pub reserved_slots: u64,
    pub copying_slots: u64,
    pub live_slots: u64,
    pub retiring_slots: u64,
    pub quarantined_slots: u64,
    pub active_owners: u64,
    pub pending_transitions: u64,
    pub pending_retirements: u64,
}

impl StateCheckpointPool {
    /// Creates a fixed-width state pool.
    ///
    /// # Errors
    ///
    /// Rejects zero identity, size, or slot geometry.
    pub fn new(
        engine_epoch: u64,
        pool_epoch: u64,
        pool_id: u32,
        byte_count: u64,
        slot_count: u32,
    ) -> Result<Self, StateCheckpointError> {
        if engine_epoch == 0 || pool_epoch == 0 || pool_id == 0 || byte_count == 0 || slot_count < 2
        {
            return Err(StateCheckpointError::InvalidGeometry);
        }
        Ok(Self {
            engine_epoch,
            pool_epoch,
            pool_id,
            byte_count,
            slot_count,
            slots: vec![
                SlotState {
                    generation: 0,
                    phase: SlotPhase::Free,
                };
                slot_count as usize
            ],
            free: (0..slot_count).rev().collect(),
            owners: BTreeMap::new(),
            owner_operations: BTreeMap::new(),
            quarantined_owners: BTreeSet::new(),
            operations: BTreeMap::new(),
            retirements: BTreeMap::new(),
            next_operation: 0,
            next_retirement: 0,
        })
    }

    /// Reserves a destination for initial state or one exact state replacement.
    ///
    /// # Errors
    ///
    /// Rejects busy owners, stale published state, exhaustion, or generation wrap.
    pub fn prepare(
        &mut self,
        owner_id: u64,
        expected: Option<StateSlotLease>,
    ) -> Result<StateCopyIntent, StateCheckpointError> {
        if self.owner_operations.contains_key(&owner_id) {
            return Err(StateCheckpointError::OwnerBusy);
        }
        if self.quarantined_owners.contains(&owner_id) {
            return Err(StateCheckpointError::OwnerQuarantined);
        }
        if self.owners.get(&owner_id).copied() != expected {
            return Err(StateCheckpointError::StaleLease);
        }
        if let Some(source) = expected {
            self.validate_live(source, owner_id)?;
        }
        let slot_id = *self
            .free
            .last()
            .ok_or(StateCheckpointError::PoolExhausted)?;
        let state = self.slots[slot_id as usize];
        let generation = state
            .generation
            .checked_add(1)
            .ok_or(StateCheckpointError::GenerationExhausted)?;
        let transition = StateTransitionLease {
            engine_epoch: self.engine_epoch,
            slot: self.next_operation,
            generation: 1,
        };
        self.next_operation = self
            .next_operation
            .checked_add(1)
            .ok_or(StateCheckpointError::GenerationExhausted)?;
        let destination = StateSlotLease {
            engine_epoch: self.engine_epoch,
            pool_epoch: self.pool_epoch,
            generation,
            slot_id,
            pool_id: self.pool_id,
        };
        self.free.pop();
        self.slots[slot_id as usize] = SlotState {
            generation,
            phase: SlotPhase::Reserved(transition),
        };
        self.owner_operations.insert(owner_id, transition);
        self.operations.insert(
            transition,
            TransitionState {
                owner_id,
                source: expected,
                destination,
                submitted: false,
            },
        );
        Ok(StateCopyIntent {
            transition,
            owner_id,
            source: expected,
            destination,
            byte_count: self.byte_count,
        })
    }

    /// Atomically prepares a batch of independent state transitions.
    ///
    /// # Errors
    ///
    /// Leaves the pool unchanged if any item is invalid or capacity is
    /// insufficient.
    pub fn prepare_batch(
        &mut self,
        items: &[(u64, Option<StateSlotLease>)],
    ) -> Result<Vec<StateCopyIntent>, StateCheckpointError> {
        let mut candidate = self.clone();
        let output = items
            .iter()
            .map(|(owner_id, expected)| candidate.prepare(*owner_id, *expected))
            .collect::<Result<Vec<_>, _>>()?;
        *self = candidate;
        Ok(output)
    }

    /// Validates an exact backend copy/initialization receipt.
    ///
    /// # Errors
    ///
    /// Malformed observed work quarantines the destination.
    pub fn submit(&mut self, receipt: StateCopyReceipt) -> Result<(), StateCheckpointError> {
        let operation = *self
            .operations
            .get(&receipt.transition)
            .ok_or(StateCheckpointError::StaleTransition)?;
        if operation.submitted {
            return Err(StateCheckpointError::AlreadySubmitted);
        }
        let expected_source = operation.source.unwrap_or_default();
        let valid = receipt.source == expected_source
            && receipt.source_present == u8::from(operation.source.is_some())
            && receipt.destination == operation.destination
            && receipt.byte_count == self.byte_count
            && receipt.written == 1
            && receipt.reserved8 == 0
            && receipt.reserved32 == 0;
        if receipt.observed != 1 {
            self.quarantine_transition(receipt.transition)?;
            return Err(StateCheckpointError::CopyObservationUnknown);
        }
        if !valid {
            self.quarantine_transition(receipt.transition)?;
            return Err(StateCheckpointError::CopyReceiptMismatch);
        }
        self.slots[operation.destination.slot_id as usize].phase =
            SlotPhase::Copying(receipt.transition);
        self.operations
            .get_mut(&receipt.transition)
            .ok_or(StateCheckpointError::StaleTransition)?
            .submitted = true;
        Ok(())
    }

    /// Atomically validates a batch of state-copy receipts.
    ///
    /// # Errors
    ///
    /// A malformed observed receipt returns a fail-stop error to the caller.
    /// Other failures leave the pool unchanged.
    pub fn submit_batch(
        &mut self,
        receipts: &[StateCopyReceipt],
    ) -> Result<(), StateCheckpointError> {
        let mut candidate = self.clone();
        let mut semantic_error = None;
        for receipt in receipts {
            match candidate.submit(*receipt) {
                Ok(()) => {}
                Err(
                    error @ (StateCheckpointError::CopyReceiptMismatch
                    | StateCheckpointError::CopyObservationUnknown),
                ) => {
                    semantic_error = Some(error);
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        if let Some(error) = semantic_error {
            for receipt in receipts {
                let _ = candidate.quarantine_transition(receipt.transition);
            }
            *self = candidate;
            return Err(error);
        }
        *self = candidate;
        Ok(())
    }

    /// Publishes the new state and returns an old-state retirement certificate.
    ///
    /// # Errors
    ///
    /// Rejects stale operations and unconfirmed completion without mutation.
    pub fn complete(
        &mut self,
        transition: StateTransitionLease,
        completion: StateCompletionReceipt,
    ) -> Result<StatePublication, StateCheckpointError> {
        if completion.engine_epoch != self.engine_epoch || !completion.confirmed {
            return Err(StateCheckpointError::CompletionNotConfirmed);
        }
        let operation = *self
            .operations
            .get(&transition)
            .ok_or(StateCheckpointError::StaleTransition)?;
        if !operation.submitted {
            return Err(StateCheckpointError::NotSubmitted);
        }
        if self.slots[operation.destination.slot_id as usize].phase
            != SlotPhase::Copying(transition)
        {
            return Err(StateCheckpointError::StaleLease);
        }
        let retirement_lease = operation
            .source
            .map(|_| {
                let lease = StateRetirementLease {
                    engine_epoch: self.engine_epoch,
                    slot: self.next_retirement,
                    generation: 1,
                };
                self.next_retirement = self
                    .next_retirement
                    .checked_add(1)
                    .ok_or(StateCheckpointError::GenerationExhausted)?;
                Ok(lease)
            })
            .transpose()?;
        let retirement = operation
            .source
            .zip(retirement_lease)
            .map(|(source, lease)| {
                let certificate = StateRetirementCertificate {
                    retirement: lease,
                    slot: source,
                    byte_count: self.byte_count,
                    completion_domain: completion.completion_domain,
                    completion_value: completion.completion_value,
                };
                self.slots[source.slot_id as usize].phase = SlotPhase::Retiring(lease);
                self.retirements.insert(lease, certificate);
                certificate
            });
        self.slots[operation.destination.slot_id as usize].phase =
            SlotPhase::Live(operation.owner_id);
        self.owners
            .insert(operation.owner_id, operation.destination);
        self.operations.remove(&transition);
        self.owner_operations.remove(&operation.owner_id);
        Ok(StatePublication {
            owner_id: operation.owner_id,
            slot: operation.destination,
            retirement,
        })
    }

    /// Atomically publishes a batch after one completion frontier.
    ///
    /// # Errors
    ///
    /// Leaves the pool unchanged if any transition or completion is invalid.
    pub fn complete_batch(
        &mut self,
        transitions: &[StateTransitionLease],
        completion: StateCompletionReceipt,
    ) -> Result<Vec<StatePublication>, StateCheckpointError> {
        let mut candidate = self.clone();
        let output = transitions
            .iter()
            .map(|transition| candidate.complete(*transition, completion))
            .collect::<Result<Vec<_>, _>>()?;
        *self = candidate;
        Ok(output)
    }

    /// Acknowledges backend detachment and makes old state reusable.
    ///
    /// # Errors
    ///
    /// Rejects stale or mismatched certificates.
    pub fn acknowledge(
        &mut self,
        certificate: StateRetirementCertificate,
    ) -> Result<(), StateCheckpointError> {
        if self.retirements.get(&certificate.retirement) != Some(&certificate)
            || self.slots[certificate.slot.slot_id as usize].phase
                != SlotPhase::Retiring(certificate.retirement)
        {
            return Err(StateCheckpointError::RetirementMismatch);
        }
        self.retirements.remove(&certificate.retirement);
        self.slots[certificate.slot.slot_id as usize].phase = SlotPhase::Free;
        self.free.push(certificate.slot.slot_id);
        Ok(())
    }

    /// Atomically acknowledges a batch of completed retirements.
    ///
    /// # Errors
    ///
    /// Leaves the pool unchanged if any certificate is stale or malformed.
    pub fn acknowledge_batch(
        &mut self,
        certificates: &[StateRetirementCertificate],
    ) -> Result<(), StateCheckpointError> {
        let mut candidate = self.clone();
        for certificate in certificates {
            candidate.acknowledge(*certificate)?;
        }
        *self = candidate;
        Ok(())
    }

    /// Detaches the final owner and returns reuse authority for its state slot.
    ///
    /// # Errors
    ///
    /// Rejects busy, stale, quarantined, or unconfirmed owners.
    pub fn retire_owner(
        &mut self,
        owner_id: u64,
        expected: StateSlotLease,
        completion: StateCompletionReceipt,
    ) -> Result<StateRetirementCertificate, StateCheckpointError> {
        if self.owner_operations.contains_key(&owner_id) {
            return Err(StateCheckpointError::OwnerBusy);
        }
        if self.quarantined_owners.contains(&owner_id) {
            return Err(StateCheckpointError::OwnerQuarantined);
        }
        if completion.engine_epoch != self.engine_epoch || !completion.confirmed {
            return Err(StateCheckpointError::CompletionNotConfirmed);
        }
        if self.owners.get(&owner_id).copied() != Some(expected) {
            return Err(StateCheckpointError::StaleLease);
        }
        self.validate_live(expected, owner_id)?;
        let retirement = StateRetirementLease {
            engine_epoch: self.engine_epoch,
            slot: self.next_retirement,
            generation: 1,
        };
        self.next_retirement = self
            .next_retirement
            .checked_add(1)
            .ok_or(StateCheckpointError::GenerationExhausted)?;
        let certificate = StateRetirementCertificate {
            retirement,
            slot: expected,
            byte_count: self.byte_count,
            completion_domain: completion.completion_domain,
            completion_value: completion.completion_value,
        };
        self.owners.remove(&owner_id);
        self.slots[expected.slot_id as usize].phase = SlotPhase::Retiring(retirement);
        self.retirements.insert(retirement, certificate);
        Ok(certificate)
    }

    /// Atomically detaches a batch of final owners at one completion frontier.
    ///
    /// # Errors
    ///
    /// Leaves the pool unchanged if any owner or lease is invalid.
    pub fn retire_owners_batch(
        &mut self,
        items: &[(u64, StateSlotLease)],
        completion: StateCompletionReceipt,
    ) -> Result<Vec<StateRetirementCertificate>, StateCheckpointError> {
        let mut candidate = self.clone();
        let output = items
            .iter()
            .map(|(owner_id, expected)| candidate.retire_owner(*owner_id, *expected, completion))
            .collect::<Result<Vec<_>, _>>()?;
        *self = candidate;
        Ok(output)
    }

    /// Aborts a prepared transition proven unobserved by the backend.
    ///
    /// # Errors
    ///
    /// Rejects stale/submitted operations without mutation. Missing unobserved
    /// proof quarantines the destination and owner.
    pub fn abort(
        &mut self,
        transition: StateTransitionLease,
        backend_unobserved: bool,
    ) -> Result<(), StateCheckpointError> {
        let operation = *self
            .operations
            .get(&transition)
            .ok_or(StateCheckpointError::StaleTransition)?;
        if operation.submitted {
            return Err(StateCheckpointError::AlreadySubmitted);
        }
        if !backend_unobserved {
            self.quarantine_transition(transition)?;
            return Err(StateCheckpointError::CopyObservationUnknown);
        }
        if self.slots[operation.destination.slot_id as usize].phase
            != SlotPhase::Reserved(transition)
        {
            self.quarantine_transition(transition)?;
            return Err(StateCheckpointError::CopyObservationUnknown);
        }
        self.operations.remove(&transition);
        self.owner_operations.remove(&operation.owner_id);
        self.slots[operation.destination.slot_id as usize].phase = SlotPhase::Free;
        self.free.push(operation.destination.slot_id);
        Ok(())
    }

    /// Atomically aborts a batch proven unobserved by the backend.
    ///
    /// # Errors
    ///
    /// Leaves the pool unchanged if any transition is stale or submitted. A
    /// missing unobserved proof quarantines every destination and owner in the
    /// batch.
    pub fn abort_batch(
        &mut self,
        transitions: &[(StateTransitionLease, bool)],
    ) -> Result<(), StateCheckpointError> {
        for (transition, _) in transitions {
            let operation = self
                .operations
                .get(transition)
                .ok_or(StateCheckpointError::StaleTransition)?;
            if operation.submitted {
                return Err(StateCheckpointError::AlreadySubmitted);
            }
        }
        let mut candidate = self.clone();
        if transitions.iter().any(|(transition, backend_unobserved)| {
            !backend_unobserved
                || candidate
                    .operations
                    .get(transition)
                    .is_some_and(|operation| {
                        candidate.slots[operation.destination.slot_id as usize].phase
                            != SlotPhase::Reserved(*transition)
                    })
        }) {
            for (transition, _) in transitions {
                let _ = candidate.quarantine_transition(*transition);
            }
            *self = candidate;
            return Err(StateCheckpointError::CopyObservationUnknown);
        }
        for (transition, backend_unobserved) in transitions {
            candidate.abort(*transition, *backend_unobserved)?;
        }
        *self = candidate;
        Ok(())
    }

    #[must_use]
    pub fn current(&self, owner_id: u64) -> Option<StateSlotLease> {
        self.owners.get(&owner_id).copied()
    }

    #[must_use]
    pub fn identity(&self) -> StatePoolIdentity {
        StatePoolIdentity {
            engine_epoch: self.engine_epoch,
            pool_epoch: self.pool_epoch,
            byte_count: self.byte_count,
            pool_id: self.pool_id,
            slot_count: self.slot_count,
        }
    }

    #[must_use]
    pub fn stats(&self) -> StatePoolStats {
        let mut stats = StatePoolStats {
            identity: self.identity(),
            free_slots: 0,
            reserved_slots: 0,
            copying_slots: 0,
            live_slots: 0,
            retiring_slots: 0,
            quarantined_slots: 0,
            active_owners: self.owners.len() as u64,
            pending_transitions: self.operations.len() as u64,
            pending_retirements: self.retirements.len() as u64,
        };
        for slot in &self.slots {
            match slot.phase {
                SlotPhase::Free => stats.free_slots += 1,
                SlotPhase::Reserved(_) => stats.reserved_slots += 1,
                SlotPhase::Copying(_) => stats.copying_slots += 1,
                SlotPhase::Live(_) => stats.live_slots += 1,
                SlotPhase::Retiring(_) => stats.retiring_slots += 1,
                SlotPhase::Quarantined => stats.quarantined_slots += 1,
            }
        }
        stats
    }

    fn quarantine_transition(
        &mut self,
        transition: StateTransitionLease,
    ) -> Result<(), StateCheckpointError> {
        let operation = self
            .operations
            .remove(&transition)
            .ok_or(StateCheckpointError::StaleTransition)?;
        self.slots[operation.destination.slot_id as usize].phase = SlotPhase::Quarantined;
        self.owner_operations.remove(&operation.owner_id);
        self.quarantined_owners.insert(operation.owner_id);
        Ok(())
    }

    fn validate_live(
        &self,
        lease: StateSlotLease,
        owner_id: u64,
    ) -> Result<(), StateCheckpointError> {
        if lease.engine_epoch != self.engine_epoch
            || lease.pool_epoch != self.pool_epoch
            || lease.pool_id != self.pool_id
            || self.slots.get(lease.slot_id as usize).is_none_or(|state| {
                state.generation != lease.generation || state.phase != SlotPhase::Live(owner_id)
            })
        {
            return Err(StateCheckpointError::StaleLease);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(intent: StateCopyIntent) -> StateCopyReceipt {
        StateCopyReceipt {
            transition: intent.transition,
            source: intent.source.unwrap_or_default(),
            destination: intent.destination,
            byte_count: intent.byte_count,
            source_present: u8::from(intent.source.is_some()),
            observed: 1,
            written: 1,
            reserved8: 0,
            reserved32: 0,
        }
    }

    #[test]
    fn initial_and_replacement_state_are_generation_checked() {
        let mut pool = StateCheckpointPool::new(1, 2, 3, 4096, 2).unwrap();
        let initial = pool.prepare(7, None).unwrap();
        pool.submit(receipt(initial)).unwrap();
        let first = pool
            .complete(
                initial.transition,
                StateCompletionReceipt {
                    engine_epoch: 1,
                    completion_domain: 4,
                    completion_value: 1,
                    confirmed: true,
                },
            )
            .unwrap();
        assert_eq!(pool.current(7), Some(first.slot));
        assert!(first.retirement.is_none());

        let replacement = pool.prepare(7, Some(first.slot)).unwrap();
        pool.submit(receipt(replacement)).unwrap();
        let second = pool
            .complete(
                replacement.transition,
                StateCompletionReceipt {
                    engine_epoch: 1,
                    completion_domain: 4,
                    completion_value: 2,
                    confirmed: true,
                },
            )
            .unwrap();
        let retirement = second.retirement.unwrap();
        assert_ne!(first.slot, second.slot);
        assert_eq!(
            pool.prepare(8, None),
            Err(StateCheckpointError::PoolExhausted)
        );
        pool.acknowledge(retirement).unwrap();
        let reused = pool.prepare(8, None).unwrap();
        assert_eq!(reused.destination.slot_id, first.slot.slot_id);
        assert!(reused.destination.generation > first.slot.generation);
    }

    #[test]
    fn owner_release_requires_completion_and_ack_before_reuse() {
        let mut pool = StateCheckpointPool::new(1, 2, 3, 1024, 2).unwrap();
        let prepared = pool.prepare(7, None).unwrap();
        pool.submit(receipt(prepared)).unwrap();
        let published = pool
            .complete(
                prepared.transition,
                StateCompletionReceipt {
                    engine_epoch: 1,
                    completion_domain: 4,
                    completion_value: 1,
                    confirmed: true,
                },
            )
            .unwrap();
        assert_eq!(
            pool.retire_owner(
                7,
                published.slot,
                StateCompletionReceipt {
                    engine_epoch: 1,
                    completion_domain: 4,
                    completion_value: 2,
                    confirmed: false,
                },
            ),
            Err(StateCheckpointError::CompletionNotConfirmed)
        );
        let certificate = pool
            .retire_owner(
                7,
                published.slot,
                StateCompletionReceipt {
                    engine_epoch: 1,
                    completion_domain: 4,
                    completion_value: 2,
                    confirmed: true,
                },
            )
            .unwrap();
        assert_eq!(pool.current(7), None);
        let other = pool.prepare(8, None).unwrap();
        assert_ne!(other.destination.slot_id, published.slot.slot_id);
        pool.abort(other.transition, true).unwrap();
        pool.acknowledge(certificate).unwrap();
        let reused = pool.prepare(9, None).unwrap();
        assert_eq!(reused.destination.slot_id, published.slot.slot_id);
        assert!(reused.destination.generation > published.slot.generation);
    }

    #[test]
    fn abort_is_recoverable_but_observed_mismatch_quarantines() {
        let mut pool = StateCheckpointPool::new(1, 2, 3, 16, 2).unwrap();
        let prepared = pool.prepare(7, None).unwrap();
        pool.abort(prepared.transition, true).unwrap();
        let retried = pool.prepare(7, None).unwrap();
        let mut malformed = receipt(retried);
        malformed.byte_count += 1;
        assert_eq!(
            pool.submit(malformed),
            Err(StateCheckpointError::CopyReceiptMismatch)
        );
        assert_eq!(
            pool.prepare(7, None),
            Err(StateCheckpointError::OwnerQuarantined)
        );
    }

    #[test]
    fn abort_without_unobserved_proof_quarantines_the_entire_batch() {
        let mut pool = StateCheckpointPool::new(1, 2, 3, 16, 2).unwrap();
        let prepared = pool.prepare_batch(&[(7, None), (8, None)]).unwrap();
        assert_eq!(
            pool.abort_batch(&[
                (prepared[0].transition, true),
                (prepared[1].transition, false),
            ]),
            Err(StateCheckpointError::CopyObservationUnknown)
        );
        let stats = pool.stats();
        assert_eq!(stats.free_slots, 0);
        assert_eq!(stats.reserved_slots, 0);
        assert_eq!(stats.quarantined_slots, 2);
        assert_eq!(stats.pending_transitions, 0);
        assert_eq!(
            pool.prepare(7, None),
            Err(StateCheckpointError::OwnerQuarantined)
        );
        assert_eq!(
            pool.prepare(8, None),
            Err(StateCheckpointError::OwnerQuarantined)
        );
    }

    #[test]
    fn batch_preflight_is_atomic_and_census_is_exact() {
        let mut pool = StateCheckpointPool::new(1, 2, 3, 16, 2).unwrap();
        let before = pool.stats();
        assert_eq!(
            pool.prepare_batch(&[(7, None), (7, None)]),
            Err(StateCheckpointError::OwnerBusy)
        );
        assert_eq!(pool.stats(), before);

        let prepared = pool.prepare_batch(&[(7, None), (8, None)]).unwrap();
        assert_eq!(pool.stats().reserved_slots, 2);
        let mut receipts = prepared.iter().copied().map(receipt).collect::<Vec<_>>();
        receipts[1].byte_count += 1;
        assert_eq!(
            pool.submit_batch(&receipts),
            Err(StateCheckpointError::CopyReceiptMismatch)
        );
        assert_eq!(pool.stats().reserved_slots, 0);
        assert_eq!(pool.stats().quarantined_slots, 2);
        assert_eq!(pool.stats().pending_transitions, 0);
        assert_eq!(
            pool.prepare(7, None),
            Err(StateCheckpointError::OwnerQuarantined)
        );
    }
}
