//! Stable CUDA storage for request-scoped recurrent and convolution state.

use std::{collections::BTreeMap, sync::Arc};

use luminal_cuda_lite::cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr, sys};
use luminal_cuda_lite::runtime::{CudaRuntime, copy_shared_device_range, zero_shared_device_range};
use orbitkv::{EngineFixedStateEvidence, EngineFixedStatePlan, StateSlotLease};
use thiserror::Error;

use crate::{
    ExecutorError, FixedStateArenaRegistration, FixedStateExecutionEvidence, FixedStateSlotRange,
};

/// Device range resolved from a session-authored state plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedStateDeviceRange {
    pub state_id: u16,
    pub lease: StateSlotLease,
    pub device_ptr: u64,
    pub byte_offset: usize,
    pub byte_count: usize,
}

/// One request's source/destination state ranges for a device invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedStateDeviceBatch {
    pub request_id: u64,
    pub sources: Box<[Option<FixedStateDeviceRange>]>,
    pub destinations: Box<[FixedStateDeviceRange]>,
}

/// Validated state ranges awaiting a real kernel launch.
pub struct PreparedFixedStateDeviceBatch {
    batch: FixedStateDeviceBatch,
    evidence: Box<[EngineFixedStateEvidence]>,
    stream: Arc<CudaStream>,
    operations: Box<[FixedStateDeviceOperation]>,
}

/// CUDA event proving that all state writes enqueued before it completed.
pub struct PendingFixedStateCompletion {
    batch: FixedStateDeviceBatch,
    evidence: Box<[EngineFixedStateEvidence]>,
    event: CudaEvent,
    _allocations: Box<[Arc<CudaSlice<u8>>]>,
}

struct FixedStateDeviceArena {
    registration: FixedStateArenaRegistration,
    allocation: Arc<CudaSlice<u8>>,
}

struct FixedStateDeviceOperation {
    allocation: Arc<CudaSlice<u8>>,
    source: Option<std::ops::Range<usize>>,
    destination: std::ops::Range<usize>,
}

/// One stable allocation per fixed-state class.
pub struct FixedStateDeviceArenas {
    stream: Arc<CudaStream>,
    arenas: BTreeMap<u16, FixedStateDeviceArena>,
}

#[derive(Debug, Error)]
pub enum FixedStateDeviceError {
    #[error(transparent)]
    Contract(#[from] ExecutorError),
    #[error(transparent)]
    Device(#[from] luminal_cuda_lite::cudarc::driver::DriverError),
    #[error("fixed-state plan is empty or has duplicate/unknown classes")]
    InvalidPlan,
    #[error("fixed-state source and destination ranges overlap")]
    OverlappingState,
}

impl FixedStateDeviceArenas {
    /// Allocates stable zeroed CUDA arenas for every fixed-state class.
    ///
    /// # Errors
    ///
    /// Rejects a registration that does not match the compiled plan and
    /// propagates CUDA allocation failures.
    pub fn allocate(
        plan: &crate::ExecutorPlan,
        identities: &[(u16, orbitkv::StatePoolIdentity)],
        stream: Arc<CudaStream>,
    ) -> Result<Self, FixedStateDeviceError> {
        let registrations = plan.fixed_state_registrations(identities)?;
        let mut arenas = BTreeMap::new();
        for registration in registrations.iter().copied() {
            let bytes = registration.arena_bytes()?;
            let allocation = Arc::new(stream.alloc_zeros::<u8>(bytes)?);
            if arenas
                .insert(
                    registration.state_id,
                    FixedStateDeviceArena {
                        registration,
                        allocation,
                    },
                )
                .is_some()
            {
                return Err(FixedStateDeviceError::InvalidPlan);
            }
        }
        Ok(Self { stream, arenas })
    }

    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    #[must_use]
    pub fn arena_count(&self) -> usize {
        self.arenas.len()
    }

    /// Resolves all fixed-state slots for one request without changing arena
    /// pointers. No device work is performed by this method.
    ///
    /// # Errors
    ///
    /// Rejects missing, duplicate, foreign, or overlapping state slots.
    pub fn prepare(
        &self,
        request_id: u64,
        states: &[EngineFixedStatePlan],
    ) -> Result<PreparedFixedStateDeviceBatch, FixedStateDeviceError> {
        if states.is_empty() || states.len() != self.arenas.len() {
            return Err(FixedStateDeviceError::InvalidPlan);
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut sources = Vec::with_capacity(states.len());
        let mut destinations = Vec::with_capacity(states.len());
        let mut evidence = Vec::with_capacity(states.len());
        let mut operations = Vec::with_capacity(states.len());
        for state in states {
            if !seen.insert(state.state_id) {
                return Err(FixedStateDeviceError::InvalidPlan);
            }
            let arena = self
                .arenas
                .get(&state.state_id)
                .ok_or(FixedStateDeviceError::InvalidPlan)?;
            let destination = arena.device_range(state.destination)?;
            if usize::try_from(state.byte_count).ok() != Some(destination.byte_count) {
                return Err(FixedStateDeviceError::InvalidPlan);
            }
            let source = state
                .source
                .map(|lease| arena.device_range(lease))
                .transpose()?;
            if source.is_some_and(|source| ranges_overlap(source, destination)) {
                return Err(FixedStateDeviceError::OverlappingState);
            }
            operations.push(FixedStateDeviceOperation {
                allocation: Arc::clone(&arena.allocation),
                source: source.map(FixedStateDeviceRange::byte_range),
                destination: destination.byte_range(),
            });
            sources.push(source);
            destinations.push(destination);
            evidence.push(EngineFixedStateEvidence {
                state_id: state.state_id,
                source: state.source,
                destination: state.destination,
                byte_count: state.byte_count,
                observed: true,
                written: true,
            });
        }
        Ok(PreparedFixedStateDeviceBatch {
            batch: FixedStateDeviceBatch {
                request_id,
                sources: sources.into_boxed_slice(),
                destinations: destinations.into_boxed_slice(),
            },
            evidence: evidence.into_boxed_slice(),
            stream: Arc::clone(&self.stream),
            operations: operations.into_boxed_slice(),
        })
    }

    /// Binds an entire stable arena as a required in-place Luminal state edge.
    /// The runtime retains shared ownership of the allocation.
    ///
    /// # Errors
    ///
    /// Rejects an unknown state id.
    pub fn bind_required_state(
        &self,
        runtime: &mut CudaRuntime,
        state_id: u16,
        input: luminal::prelude::GraphTensor,
        output: luminal::prelude::GraphTensor,
    ) -> Result<(), FixedStateDeviceError> {
        let arena = self
            .arenas
            .get(&state_id)
            .ok_or(FixedStateDeviceError::InvalidPlan)?;
        runtime.alias_shared_state_required(
            input,
            output,
            Arc::clone(&arena.allocation),
            arena.registration.arena_bytes()?,
        )?;
        Ok(())
    }
}

impl FixedStateDeviceArena {
    fn device_range(
        &self,
        lease: StateSlotLease,
    ) -> Result<FixedStateDeviceRange, FixedStateDeviceError> {
        let FixedStateSlotRange {
            state_id,
            lease,
            byte_offset,
            byte_count,
        } = self.registration.slot_range(lease)?;
        let base = self.allocation.device_ptr(self.stream()).0;
        let device_ptr = base
            .checked_add(
                u64::try_from(byte_offset).map_err(|_| FixedStateDeviceError::InvalidPlan)?,
            )
            .ok_or(FixedStateDeviceError::InvalidPlan)?;
        Ok(FixedStateDeviceRange {
            state_id,
            lease,
            device_ptr,
            byte_offset,
            byte_count,
        })
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.allocation.stream()
    }
}

impl PreparedFixedStateDeviceBatch {
    #[must_use]
    pub fn ranges(&self) -> &FixedStateDeviceBatch {
        &self.batch
    }

    /// Enqueues the reference state transition and records completion. Initial
    /// state is zeroed; replacement state is copied source-to-destination. A
    /// future GDN implementation will consume the same stable ranges but write
    /// the mathematically updated destination before recording its event.
    ///
    /// # Errors
    ///
    /// Propagates CUDA event creation or recording failures.
    pub fn enqueue_reference(self) -> Result<PendingFixedStateCompletion, FixedStateDeviceError> {
        let mut allocations = Vec::with_capacity(self.operations.len());
        for operation in &self.operations {
            if let Some(source) = operation.source.clone() {
                copy_shared_device_range(
                    &self.stream,
                    &operation.allocation,
                    source,
                    operation.destination.clone(),
                )?;
            } else {
                zero_shared_device_range(
                    &self.stream,
                    &operation.allocation,
                    operation.destination.clone(),
                )?;
            }
            allocations.push(Arc::clone(&operation.allocation));
        }
        let event = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?;
        Ok(PendingFixedStateCompletion {
            batch: self.batch,
            evidence: self.evidence,
            event,
            _allocations: allocations.into_boxed_slice(),
        })
    }
}

impl PendingFixedStateCompletion {
    #[must_use]
    pub fn ranges(&self) -> &FixedStateDeviceBatch {
        &self.batch
    }

    /// Waits for the recorded stream event before releasing success evidence.
    ///
    /// # Errors
    ///
    /// Propagates CUDA completion failures.
    pub fn wait(self) -> Result<FixedStateExecutionEvidence, FixedStateDeviceError> {
        self.event.synchronize()?;
        Ok(FixedStateExecutionEvidence {
            request_id: self.batch.request_id,
            states: self.evidence,
        })
    }
}

const fn ranges_overlap(left: FixedStateDeviceRange, right: FixedStateDeviceRange) -> bool {
    let left_end = left.device_ptr.saturating_add(left.byte_count as u64);
    let right_end = right.device_ptr.saturating_add(right.byte_count as u64);
    left.device_ptr < right_end && right.device_ptr < left_end
}

impl FixedStateDeviceRange {
    fn byte_range(self) -> std::ops::Range<usize> {
        self.byte_offset..self.byte_offset + self.byte_count
    }
}
