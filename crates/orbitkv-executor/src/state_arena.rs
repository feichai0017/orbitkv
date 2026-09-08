//! Stable CUDA storage for request-scoped recurrent and convolution state.

use std::{collections::BTreeMap, sync::Arc};

use luminal_cuda_lite::cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use luminal_cuda_lite::runtime::{
    CudaExecutionReceipt, CudaRuntime, CudaSharedStateBinding, copy_shared_device_range,
    zero_shared_device_range,
};
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
    batches: Box<[FixedStateDeviceBatch]>,
    stream: Arc<CudaStream>,
    operations: Box<[FixedStateDeviceOperation]>,
}

/// State destinations initialized on the execution stream but not yet written
/// by a model kernel. This type cannot produce success evidence on its own.
pub struct InitializedFixedStateDeviceBatch {
    batches: Box<[FixedStateDeviceBatch]>,
    stream: Arc<CudaStream>,
    allocations: Box<[Arc<CudaSlice<u8>>]>,
}

/// Initialized state plus uploaded manager-authored destination slot metadata.
pub struct ReadyFixedStateDeviceBatch {
    initialized: InitializedFixedStateDeviceBatch,
}

/// CUDA event proving that all state writes enqueued before it completed.
pub struct PendingFixedStateCompletion {
    batches: Box<[FixedStateDeviceBatch]>,
    stream: Arc<CudaStream>,
    receipt: CudaExecutionReceipt,
    bindings: Box<[FixedStateRuntimeBinding]>,
    _allocations: Box<[Arc<CudaSlice<u8>>]>,
}

/// Runtime-owned proof that one fixed-state class is bound to a particular
/// Luminal required alias.
#[derive(Clone)]
pub struct FixedStateRuntimeBinding {
    state_id: u16,
    allocation: Arc<CudaSlice<u8>>,
    binding: CudaSharedStateBinding,
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
        self.prepare_batch(std::iter::once((request_id, states)))
    }

    /// Resolves an ordered request batch to stable state ranges.
    ///
    /// # Errors
    ///
    /// Rejects empty, duplicate-request, missing-class, foreign, or
    /// overlapping state plans.
    pub fn prepare_batch<'a>(
        &self,
        requests: impl IntoIterator<Item = (u64, &'a [EngineFixedStatePlan])>,
    ) -> Result<PreparedFixedStateDeviceBatch, FixedStateDeviceError> {
        let requests = requests.into_iter().collect::<Vec<_>>();
        if requests.is_empty() {
            return Err(FixedStateDeviceError::InvalidPlan);
        }
        let mut request_ids = std::collections::BTreeSet::new();
        let mut source_slots = std::collections::BTreeSet::new();
        let mut destination_slots = std::collections::BTreeSet::new();
        let mut batches = Vec::with_capacity(requests.len());
        let mut operations = Vec::new();
        for (request_id, states) in requests {
            if !request_ids.insert(request_id) {
                return Err(FixedStateDeviceError::InvalidPlan);
            }
            let batch = self.resolve_request(request_id, states, &mut operations)?;
            for source in batch.sources.iter().flatten() {
                source_slots.insert((source.state_id, source.lease.slot_id));
            }
            for destination in &batch.destinations {
                if !destination_slots.insert((destination.state_id, destination.lease.slot_id)) {
                    return Err(FixedStateDeviceError::OverlappingState);
                }
            }
            batches.push(batch);
        }
        if source_slots
            .iter()
            .any(|slot| destination_slots.contains(slot))
        {
            return Err(FixedStateDeviceError::OverlappingState);
        }
        Ok(PreparedFixedStateDeviceBatch {
            batches: batches.into_boxed_slice(),
            stream: Arc::clone(&self.stream),
            operations: operations.into_boxed_slice(),
        })
    }

    fn resolve_request(
        &self,
        request_id: u64,
        states: &[EngineFixedStatePlan],
        operations: &mut Vec<FixedStateDeviceOperation>,
    ) -> Result<FixedStateDeviceBatch, FixedStateDeviceError> {
        if states.is_empty() || states.len() != self.arenas.len() {
            return Err(FixedStateDeviceError::InvalidPlan);
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut sources = Vec::with_capacity(states.len());
        let mut destinations = Vec::with_capacity(states.len());
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
        }
        Ok(FixedStateDeviceBatch {
            request_id,
            sources: sources.into_boxed_slice(),
            destinations: destinations.into_boxed_slice(),
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
    ) -> Result<FixedStateRuntimeBinding, FixedStateDeviceError> {
        let arena = self
            .arenas
            .get(&state_id)
            .ok_or(FixedStateDeviceError::InvalidPlan)?;
        let allocation = Arc::clone(&arena.allocation);
        let binding = runtime.alias_shared_state_required(
            input,
            output,
            Arc::clone(&allocation),
            arena.registration.arena_bytes()?,
        )?;
        Ok(FixedStateRuntimeBinding {
            state_id,
            allocation,
            binding,
        })
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
    pub fn ranges(&self) -> &[FixedStateDeviceBatch] {
        &self.batches
    }

    /// Initializes destination slots on the execution stream. Initial state is
    /// zeroed and replacement state is copied source-to-destination. This does
    /// not prove that a recurrent/convolution kernel subsequently wrote state.
    ///
    /// # Errors
    ///
    /// Propagates CUDA copy failures.
    pub fn initialize(self) -> Result<InitializedFixedStateDeviceBatch, FixedStateDeviceError> {
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
        Ok(InitializedFixedStateDeviceBatch {
            batches: self.batches,
            stream: self.stream,
            allocations: allocations.into_boxed_slice(),
        })
    }
}

impl InitializedFixedStateDeviceBatch {
    #[must_use]
    pub fn ranges(&self) -> &[FixedStateDeviceBatch] {
        &self.batches
    }

    /// Uploads manager-authored destination slots into every recurrent graph
    /// binding in canonical request order.
    ///
    /// # Errors
    ///
    /// Rejects missing, duplicate, or extraneous state-class bindings.
    pub fn upload_destination_slots(
        self,
        runtime: &mut CudaRuntime,
        bindings: &[crate::RecurrentStateGraphBinding],
    ) -> Result<ReadyFixedStateDeviceBatch, FixedStateDeviceError> {
        self.validate_state_ids(bindings.iter().map(|binding| binding.state_id))?;
        for binding in bindings {
            binding
                .upload_destination_slots(runtime, &self.batches)
                .map_err(|_| FixedStateDeviceError::InvalidPlan)?;
        }
        Ok(ReadyFixedStateDeviceBatch { initialized: self })
    }
}

impl ReadyFixedStateDeviceBatch {
    #[must_use]
    pub fn ranges(&self) -> &[FixedStateDeviceBatch] {
        &self.initialized.batches
    }

    /// Executes Luminal after initialization and records completion on the
    /// same stream. Every planned state class must be represented by a shared
    /// required alias owned by that runtime.
    ///
    /// # Errors
    ///
    /// Rejects missing, duplicate, or foreign state bindings and propagates
    /// CUDA event failures.
    pub fn complete_after(
        self,
        runtime: &mut CudaRuntime,
        graph: &luminal::prelude::Graph,
        bindings: &[FixedStateRuntimeBinding],
    ) -> Result<PendingFixedStateCompletion, FixedStateDeviceError> {
        self.initialized.validate_bindings(bindings)?;
        let receipt = runtime.execute_recorded_for(
            &graph.dyn_map,
            bindings.iter().map(|binding| &binding.binding),
        )?;
        Ok(self.initialized.into_pending(receipt, bindings))
    }

    /// Launches a captured Luminal execution after initialization and records
    /// completion on the same stream.
    ///
    /// # Errors
    ///
    /// Rejects missing, duplicate, or foreign state bindings and propagates
    /// CUDA launch or event failures.
    pub fn launch_captured(
        self,
        execution: &luminal_cuda_lite::runtime::CapturedCudaExecution,
        bindings: &[FixedStateRuntimeBinding],
    ) -> Result<PendingFixedStateCompletion, FixedStateDeviceError> {
        self.initialized.validate_bindings(bindings)?;
        let receipt =
            execution.launch_recorded_for(bindings.iter().map(|binding| &binding.binding))?;
        Ok(self.initialized.into_pending(receipt, bindings))
    }
}

impl InitializedFixedStateDeviceBatch {
    fn validate_state_ids(
        &self,
        state_ids: impl IntoIterator<Item = u16>,
    ) -> Result<(), FixedStateDeviceError> {
        let expected = self
            .batches
            .iter()
            .flat_map(|batch| &batch.destinations)
            .map(|destination| destination.state_id)
            .collect::<std::collections::BTreeSet<_>>();
        let state_ids = state_ids.into_iter().collect::<Vec<_>>();
        let actual = state_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        if expected.len() != state_ids.len() || expected != actual {
            return Err(FixedStateDeviceError::InvalidPlan);
        }
        Ok(())
    }

    fn validate_bindings(
        &self,
        bindings: &[FixedStateRuntimeBinding],
    ) -> Result<(), FixedStateDeviceError> {
        self.validate_state_ids(bindings.iter().map(|binding| binding.state_id))?;
        for destination in self.batches.iter().flat_map(|batch| &batch.destinations) {
            let binding = bindings
                .iter()
                .find(|binding| binding.state_id == destination.state_id)
                .ok_or(FixedStateDeviceError::InvalidPlan)?;
            let base = binding.allocation.device_ptr(&self.stream).0;
            let offset = u64::try_from(destination.byte_offset)
                .map_err(|_| FixedStateDeviceError::InvalidPlan)?;
            if destination.device_ptr.checked_sub(offset) != Some(base)
                || destination
                    .byte_offset
                    .checked_add(destination.byte_count)
                    .is_none_or(|end| end > binding.allocation.len())
            {
                return Err(FixedStateDeviceError::InvalidPlan);
            }
        }
        Ok(())
    }

    fn into_pending(
        self,
        receipt: CudaExecutionReceipt,
        bindings: &[FixedStateRuntimeBinding],
    ) -> PendingFixedStateCompletion {
        PendingFixedStateCompletion {
            batches: self.batches,
            stream: self.stream,
            receipt,
            bindings: bindings.to_vec().into_boxed_slice(),
            _allocations: self.allocations,
        }
    }
}

impl PendingFixedStateCompletion {
    #[must_use]
    pub fn ranges(&self) -> &[FixedStateDeviceBatch] {
        &self.batches
    }

    /// Waits for the recorded stream event before releasing success evidence.
    ///
    /// # Errors
    ///
    /// Propagates CUDA completion failures.
    pub fn wait(self) -> Result<Box<[FixedStateExecutionEvidence]>, FixedStateDeviceError> {
        self.receipt.synchronize_required_states(
            &self.stream,
            self.bindings.iter().map(|item| &item.binding),
        )?;
        Ok(self
            .batches
            .iter()
            .map(|batch| {
                let states = batch
                    .sources
                    .iter()
                    .zip(&batch.destinations)
                    .map(|(source, destination)| {
                        Ok(EngineFixedStateEvidence {
                            state_id: destination.state_id,
                            source: source.map(|range| range.lease),
                            destination: destination.lease,
                            byte_count: u64::try_from(destination.byte_count)
                                .map_err(|_| FixedStateDeviceError::InvalidPlan)?,
                            observed: true,
                            written: true,
                        })
                    })
                    .collect::<Result<Vec<_>, FixedStateDeviceError>>()?;
                Ok(FixedStateExecutionEvidence {
                    request_id: batch.request_id,
                    states: states.into_boxed_slice(),
                })
            })
            .collect::<Result<Vec<_>, FixedStateDeviceError>>()?
            .into_boxed_slice())
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
