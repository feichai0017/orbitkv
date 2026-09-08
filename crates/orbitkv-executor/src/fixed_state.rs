//! Backend-neutral fixed-state execution contract.

use orbitkv::{AttentionStateBackend, EngineFixedStateEvidence, StatePoolIdentity, StateSlotLease};

use crate::{ExecutorError, ExecutorPlan};

/// Request-scoped persistent state whose storage is not addressed by token pages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedStateClass {
    pub state_id: u16,
    pub name: String,
    pub layers: Box<[u32]>,
    pub storage: FixedStateStorage,
}

/// Physical geometry required by a recurrent or convolution state arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedStateStorage {
    Recurrent {
        family: orbitkv::RecurrentFamily,
        bytes_per_layer: u64,
        slots_per_request: u32,
        bytes_per_request: u64,
    },
    Convolution {
        bytes_per_layer: u64,
        kernel_width: u32,
        slots_per_request: u32,
        bytes_per_request: u64,
    },
}

/// A generation-checked fixed-state pool bound to one executor state class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedStateArenaRegistration {
    pub state_id: u16,
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub pool_id: u32,
    pub slot_count: u32,
    pub slot_bytes: u64,
}

/// Exact stable byte range occupied by one fixed-state slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedStateSlotRange {
    pub state_id: u16,
    pub lease: StateSlotLease,
    pub byte_offset: usize,
    pub byte_count: usize,
}

/// Completed fixed-state writes for one request.
///
/// This evidence envelope is backend-neutral even though the first producer
/// is CUDA: a device backend may release it only after its own completion
/// primitive proves that every listed write finished.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedStateExecutionEvidence {
    pub request_id: u64,
    pub states: Box<[EngineFixedStateEvidence]>,
}

impl FixedStateArenaRegistration {
    /// Binds one session-owned fixed-state pool to its compiled executor class.
    ///
    /// # Errors
    ///
    /// Rejects a mismatched state id, per-slot byte size, or slot capacity.
    pub fn bind(
        class: &FixedStateClass,
        state_id: u16,
        identity: StatePoolIdentity,
    ) -> Result<Self, ExecutorError> {
        let (slots_per_request, bytes_per_request) = match class.storage {
            FixedStateStorage::Recurrent {
                slots_per_request,
                bytes_per_request,
                ..
            }
            | FixedStateStorage::Convolution {
                slots_per_request,
                bytes_per_request,
                ..
            } => (slots_per_request, bytes_per_request),
        };
        let slot_bytes = bytes_per_request
            .checked_div(u64::from(slots_per_request))
            .filter(|bytes| {
                *bytes > 0
                    && bytes.checked_mul(u64::from(slots_per_request)) == Some(bytes_per_request)
            })
            .ok_or(ExecutorError::FixedStateRegistrationMismatch)?;
        if state_id != class.state_id
            || identity.engine_epoch == 0
            || identity.pool_epoch == 0
            || identity.pool_id == 0
            || identity.byte_count != slot_bytes
            || identity.slot_count < slots_per_request
            || !identity.slot_count.is_multiple_of(slots_per_request)
        {
            return Err(ExecutorError::FixedStateRegistrationMismatch);
        }
        identity
            .byte_count
            .checked_mul(u64::from(identity.slot_count))
            .ok_or(ExecutorError::FixedStateRegistrationMismatch)?;
        Ok(Self {
            state_id,
            engine_epoch: identity.engine_epoch,
            pool_epoch: identity.pool_epoch,
            pool_id: identity.pool_id,
            slot_count: identity.slot_count,
            slot_bytes,
        })
    }

    /// Resolves one generation-checked lease into its stable arena range.
    ///
    /// # Errors
    ///
    /// Rejects a foreign or out-of-range slot identity.
    pub fn slot_range(self, lease: StateSlotLease) -> Result<FixedStateSlotRange, ExecutorError> {
        if lease.engine_epoch != self.engine_epoch
            || lease.pool_epoch != self.pool_epoch
            || lease.pool_id != self.pool_id
            || lease.generation == 0
            || lease.slot_id >= self.slot_count
        {
            return Err(ExecutorError::FixedStateRegistrationMismatch);
        }
        let byte_count = usize::try_from(self.slot_bytes)
            .map_err(|_| ExecutorError::FixedStateRegistrationMismatch)?;
        let byte_offset = usize::try_from(lease.slot_id)
            .ok()
            .and_then(|slot| slot.checked_mul(byte_count))
            .ok_or(ExecutorError::FixedStateRegistrationMismatch)?;
        Ok(FixedStateSlotRange {
            state_id: self.state_id,
            lease,
            byte_offset,
            byte_count,
        })
    }

    /// Returns the total stable allocation size for this state class.
    ///
    /// # Errors
    ///
    /// Rejects geometry that does not fit the host address space.
    pub fn arena_bytes(self) -> Result<usize, ExecutorError> {
        usize::try_from(self.slot_bytes)
            .ok()
            .and_then(|bytes| {
                usize::try_from(self.slot_count)
                    .ok()
                    .and_then(|slots| bytes.checked_mul(slots))
            })
            .ok_or(ExecutorError::FixedStateRegistrationMismatch)
    }
}

impl ExecutorPlan {
    /// Binds every compiled fixed-state class to one session pool identity.
    ///
    /// # Errors
    ///
    /// Rejects missing, duplicate, extra, reordered, or incompatible pools.
    pub fn fixed_state_registrations(
        &self,
        identities: &[(u16, StatePoolIdentity)],
    ) -> Result<Box<[FixedStateArenaRegistration]>, ExecutorError> {
        if identities.len() != self.fixed_states.len() {
            return Err(ExecutorError::FixedStateRegistrationMismatch);
        }
        self.fixed_states
            .iter()
            .zip(identities)
            .map(|(class, &(state_id, identity))| {
                FixedStateArenaRegistration::bind(class, state_id, identity)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }
}

pub(crate) fn compile_fixed_state(
    state_id: usize,
    state: &orbitkv::CompiledAttentionState,
) -> Result<Option<FixedStateClass>, ExecutorError> {
    let storage = match state.backend {
        AttentionStateBackend::TokenSlots { .. } => return Ok(None),
        AttentionStateBackend::RecurrentCheckpoints {
            family,
            state_bytes_per_layer,
            checkpoint_slots_per_request,
            checkpoint_bytes_per_request,
        } => FixedStateStorage::Recurrent {
            family,
            bytes_per_layer: state_bytes_per_layer,
            slots_per_request: checkpoint_slots_per_request,
            bytes_per_request: checkpoint_bytes_per_request,
        },
        AttentionStateBackend::ConvolutionRing {
            state_bytes_per_layer,
            kernel_width,
            checkpoint_slots_per_request,
            checkpoint_bytes_per_request,
        } => FixedStateStorage::Convolution {
            bytes_per_layer: state_bytes_per_layer,
            kernel_width,
            slots_per_request: checkpoint_slots_per_request,
            bytes_per_request: checkpoint_bytes_per_request,
        },
    };
    Ok(Some(FixedStateClass {
        state_id: u16::try_from(state_id).map_err(|_| ExecutorError::PreparedGeometryMismatch)?,
        name: state.name.clone(),
        layers: state.layers.clone().into_boxed_slice(),
        storage,
    }))
}

#[cfg(test)]
mod tests {
    use orbitkv::RecurrentFamily;

    use super::*;

    fn class() -> FixedStateClass {
        FixedStateClass {
            state_id: 2,
            name: "recurrent".into(),
            layers: vec![0, 1, 2].into_boxed_slice(),
            storage: FixedStateStorage::Recurrent {
                family: RecurrentFamily::Gdn,
                bytes_per_layer: 64,
                slots_per_request: 2,
                bytes_per_request: 384,
            },
        }
    }

    #[test]
    fn registration_and_slot_ranges_are_exact() {
        let registration = FixedStateArenaRegistration::bind(
            &class(),
            2,
            StatePoolIdentity {
                engine_epoch: 1,
                pool_epoch: 2,
                byte_count: 192,
                pool_id: 3,
                slot_count: 4,
            },
        )
        .unwrap();
        assert_eq!(registration.arena_bytes().unwrap(), 768);
        let range = registration
            .slot_range(StateSlotLease {
                engine_epoch: 1,
                pool_epoch: 2,
                generation: 7,
                slot_id: 3,
                pool_id: 3,
            })
            .unwrap();
        assert_eq!(range.byte_offset, 576);
        assert_eq!(range.byte_count, 192);
    }

    #[test]
    fn registration_rejects_mismatched_geometry() {
        let mut identity = StatePoolIdentity {
            engine_epoch: 1,
            pool_epoch: 2,
            byte_count: 192,
            pool_id: 3,
            slot_count: 4,
        };
        identity.byte_count -= 1;
        assert!(matches!(
            FixedStateArenaRegistration::bind(&class(), 2, identity),
            Err(ExecutorError::FixedStateRegistrationMismatch)
        ));
    }
}
