use std::collections::BTreeMap;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, result, sys};
use orbitkv::{BlockHandle, PhysicalReclamationReceipt, RetirementCertificate};
use serde::Serialize;

use crate::{CudaVmmError, CudaVmmSlot};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CudaBlockAddress {
    pub physical: BlockHandle,
    pub address: u64,
    pub bytes: usize,
}

struct GenerationSlot {
    memory: CudaVmmSlot,
    active: Option<BlockHandle>,
    last_generation: u64,
}

pub struct CudaVmmBlockPool {
    class_name: String,
    slot_bytes: usize,
    slots: Vec<GenerationSlot>,
}

impl CudaVmmBlockPool {
    /// Reserves stable virtual-address slots for one lifetime class.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty class name, zero slot count, or failed
    /// CUDA VMM reservation.
    pub fn new(
        device_ordinal: usize,
        class_name: impl Into<String>,
        slot_count: usize,
        slot_bytes: usize,
    ) -> Result<Self, CudaVmmError> {
        let class_name = class_name.into();
        if class_name.is_empty() {
            return Err(CudaVmmError::EmptyClassName);
        }
        if slot_count == 0 {
            return Err(CudaVmmError::ZeroSlots);
        }
        let context = CudaContext::new(device_ordinal)?;
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(GenerationSlot {
                memory: CudaVmmSlot::reserve(context.clone(), slot_bytes)?,
                active: None,
                last_generation: 0,
            });
        }
        let slot_bytes = slots[0].memory.bytes();
        Ok(Self {
            class_name,
            slot_bytes,
            slots,
        })
    }

    #[must_use]
    pub fn class_name(&self) -> &str {
        &self.class_name
    }

    #[must_use]
    pub const fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Maps fresh physical backing for exactly one new block generation.
    ///
    /// # Errors
    ///
    /// Returns an error for a class/slot mismatch, an already-active slot, a
    /// non-monotonic generation, or failed CUDA mapping.
    pub fn activate(&mut self, physical: &BlockHandle) -> Result<CudaBlockAddress, CudaVmmError> {
        let slot = self.slot_mut(physical)?;
        if slot.active.is_some() {
            return Err(CudaVmmError::SlotAlreadyActive(physical.clone()));
        }
        let expected = slot
            .last_generation
            .checked_add(1)
            .ok_or_else(|| CudaVmmError::GenerationExhausted(physical.clone()))?;
        if physical.generation != expected {
            return Err(CudaVmmError::UnexpectedGeneration {
                physical: physical.clone(),
                expected,
            });
        }
        slot.memory.map_fresh()?;
        slot.last_generation = physical.generation;
        slot.active = Some(physical.clone());
        Ok(CudaBlockAddress {
            physical: physical.clone(),
            address: slot.memory.address(),
            bytes: slot.memory.bytes(),
        })
    }

    /// Returns the stable address for an active, generation-matched block.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is stale or inactive.
    pub fn address(&self, physical: &BlockHandle) -> Result<CudaBlockAddress, CudaVmmError> {
        let slot = self.slot(physical)?;
        validate_active(slot, physical)?;
        Ok(CudaBlockAddress {
            physical: physical.clone(),
            address: slot.memory.address(),
            bytes: slot.memory.bytes(),
        })
    }

    /// Copies host bytes into an active generation.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale handle or failed CUDA copy.
    pub fn write(&self, physical: &BlockHandle, source: &[u8]) -> Result<(), CudaVmmError> {
        let slot = self.slot(physical)?;
        validate_active(slot, physical)?;
        slot.memory.write(source)
    }

    /// Copies bytes from an active generation to host memory.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale handle or failed CUDA copy.
    pub fn read(&self, physical: &BlockHandle, destination: &mut [u8]) -> Result<(), CudaVmmError> {
        let slot = self.slot(physical)?;
        validate_active(slot, physical)?;
        slot.memory.read(destination)
    }

    /// Physically unmaps the certified generation and returns a receipt for
    /// the core manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the certificate targets another class/generation or
    /// the CUDA unmap/release fails. No receipt is returned on failure.
    pub fn reclaim(
        &mut self,
        certificate: &RetirementCertificate,
    ) -> Result<PhysicalReclamationReceipt, CudaVmmError> {
        let physical = &certificate.physical;
        let slot = self.slot_mut(physical)?;
        validate_active(slot, physical)?;
        slot.memory.unmap()?;
        slot.active = None;
        Ok(PhysicalReclamationReceipt {
            schema: "orbitkv.physical-reclamation-receipt.v1",
            certificate_id: certificate.certificate_id,
            physical: physical.clone(),
        })
    }

    /// Explicitly tears down every slot reservation.
    ///
    /// # Errors
    ///
    /// Returns an error if a slot is still active or CUDA cleanup fails.
    pub fn close(self) -> Result<(), CudaVmmError> {
        if let Some(active) = self.slots.iter().find_map(|slot| slot.active.clone()) {
            return Err(CudaVmmError::SlotStillActive(active));
        }
        for slot in self.slots {
            slot.memory.close()?;
        }
        Ok(())
    }

    fn slot(&self, physical: &BlockHandle) -> Result<&GenerationSlot, CudaVmmError> {
        validate_class(&self.class_name, physical)?;
        let index = usize::try_from(physical.slot)
            .map_err(|_| CudaVmmError::SlotOutOfRange(physical.clone()))?;
        self.slots
            .get(index)
            .ok_or_else(|| CudaVmmError::SlotOutOfRange(physical.clone()))
    }

    fn slot_mut(&mut self, physical: &BlockHandle) -> Result<&mut GenerationSlot, CudaVmmError> {
        validate_class(&self.class_name, physical)?;
        let index = usize::try_from(physical.slot)
            .map_err(|_| CudaVmmError::SlotOutOfRange(physical.clone()))?;
        self.slots
            .get_mut(index)
            .ok_or_else(|| CudaVmmError::SlotOutOfRange(physical.clone()))
    }
}

fn validate_class(class_name: &str, physical: &BlockHandle) -> Result<(), CudaVmmError> {
    if physical.class_name != class_name {
        return Err(CudaVmmError::ClassMismatch {
            expected: class_name.to_owned(),
            physical: physical.clone(),
        });
    }
    Ok(())
}

fn validate_active(slot: &GenerationSlot, physical: &BlockHandle) -> Result<(), CudaVmmError> {
    if slot.active.as_ref() != Some(physical) {
        return Err(CudaVmmError::StaleGeneration(physical.clone()));
    }
    Ok(())
}

pub struct CudaExecutionFrontier {
    pending: BTreeMap<u64, CudaEvent>,
}

impl CudaExecutionFrontier {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: BTreeMap::new(),
        }
    }

    /// Records a CUDA event after one submitted immutable KV view.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate submission ids or failed event creation.
    pub fn record(&mut self, submission_id: u64, stream: &CudaStream) -> Result<(), CudaVmmError> {
        if self.pending.contains_key(&submission_id) {
            return Err(CudaVmmError::DuplicateSubmission(submission_id));
        }
        self.pending
            .insert(submission_id, stream.record_event(None)?);
        Ok(())
    }

    /// Removes and returns every submission whose real CUDA event completed.
    ///
    /// # Errors
    ///
    /// Returns an error for any CUDA event-query failure other than
    /// `CUDA_ERROR_NOT_READY`.
    pub fn poll_completed(&mut self) -> Result<Vec<u64>, CudaVmmError> {
        let mut completed = Vec::new();
        for (&submission_id, event) in &self.pending {
            match unsafe { result::event::query(event.cu_event()) } {
                Ok(()) => completed.push(submission_id),
                Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => {}
                Err(error) => return Err(error.into()),
            }
        }
        for submission_id in &completed {
            self.pending.remove(submission_id);
        }
        Ok(completed)
    }

    /// Blocks until one submission's CUDA event completes, then removes it.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown submission or failed synchronization.
    pub fn wait(&mut self, submission_id: u64) -> Result<(), CudaVmmError> {
        let event = self
            .pending
            .remove(&submission_id)
            .ok_or(CudaVmmError::UnknownSubmission(submission_id))?;
        event.synchronize()?;
        Ok(())
    }

    #[must_use]
    pub fn pending(&self) -> usize {
        self.pending.len()
    }
}

impl Default for CudaExecutionFrontier {
    fn default() -> Self {
        Self::new()
    }
}
