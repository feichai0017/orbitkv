use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext,
    result::{self, DriverError},
    sys,
};
use thiserror::Error;

mod block_pool;

pub use block_pool::{CudaBlockAddress, CudaExecutionFrontier, CudaVmmBlockPool};
use orbitkv::BlockHandle;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct CudaVmmCapabilities {
    pub device_ordinal: usize,
    pub device_name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub virtual_address_management: bool,
    pub minimum_granularity_bytes: usize,
    pub recommended_granularity_bytes: usize,
}

#[derive(Debug, Error)]
pub enum CudaVmmError {
    #[error("CUDA driver operation failed")]
    Driver(#[from] DriverError),
    #[error("device {0} does not support CUDA virtual address management")]
    UnsupportedDevice(usize),
    #[error("requested byte size must be positive")]
    ZeroBytes,
    #[error("integer overflow while aligning the VMM allocation")]
    AlignmentOverflow,
    #[error("VMM slot is already mapped")]
    AlreadyMapped,
    #[error("VMM slot is not mapped")]
    NotMapped,
    #[error("host source length {actual} exceeds VMM slot length {capacity}")]
    SourceTooLarge { actual: usize, capacity: usize },
    #[error("host destination length {actual} exceeds VMM slot length {capacity}")]
    DestinationTooLarge { actual: usize, capacity: usize },
    #[error("VMM block class name must not be empty")]
    EmptyClassName,
    #[error("VMM block pool must contain at least one slot")]
    ZeroSlots,
    #[error("VMM block belongs to {physical:?}, expected class {expected:?}")]
    ClassMismatch {
        expected: String,
        physical: BlockHandle,
    },
    #[error("VMM slot is out of range: {0:?}")]
    SlotOutOfRange(BlockHandle),
    #[error("VMM slot is already active: {0:?}")]
    SlotAlreadyActive(BlockHandle),
    #[error("VMM slot generation exhausted: {0:?}")]
    GenerationExhausted(BlockHandle),
    #[error("VMM slot expected generation {expected}, got {physical:?}")]
    UnexpectedGeneration {
        physical: BlockHandle,
        expected: u64,
    },
    #[error("VMM block generation is inactive or stale: {0:?}")]
    StaleGeneration(BlockHandle),
    #[error("cannot close VMM pool with active block: {0:?}")]
    SlotStillActive(BlockHandle),
    #[error("CUDA submission {0} already has an execution event")]
    DuplicateSubmission(u64),
    #[error("unknown CUDA submission {0}")]
    UnknownSubmission(u64),
}

struct PhysicalAllocation {
    handle: sys::CUmemGenericAllocationHandle,
    mapped: bool,
}

pub struct CudaVmmSlot {
    context: Arc<CudaContext>,
    address: sys::CUdeviceptr,
    bytes: usize,
    physical: Option<PhysicalAllocation>,
}

impl CudaVmmSlot {
    /// Reserves a stable GPU virtual-address range.
    ///
    /// No physical memory is committed until [`Self::map_fresh`] is called.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization, capability detection, size
    /// alignment, or address reservation fails.
    pub fn reserve(
        context: Arc<CudaContext>,
        requested_bytes: usize,
    ) -> Result<Self, CudaVmmError> {
        if requested_bytes == 0 {
            return Err(CudaVmmError::ZeroBytes);
        }
        context.bind_to_thread()?;
        let capabilities = probe_context(&context)?;
        if !capabilities.virtual_address_management {
            return Err(CudaVmmError::UnsupportedDevice(context.ordinal()));
        }
        let bytes = align_up(requested_bytes, capabilities.minimum_granularity_bytes)?;
        let mut address = MaybeUninit::uninit();
        unsafe {
            sys::cuMemAddressReserve(
                address.as_mut_ptr(),
                bytes,
                capabilities.minimum_granularity_bytes,
                0,
                0,
            )
            .result()?;
        }
        Ok(Self {
            context,
            address: unsafe { address.assume_init() },
            bytes,
            physical: None,
        })
    }

    #[must_use]
    pub const fn address(&self) -> u64 {
        self.address
    }

    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    #[must_use]
    pub const fn is_mapped(&self) -> bool {
        self.physical.is_some()
    }

    /// Creates fresh physical memory and maps it at the reserved address.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot is already mapped or any CUDA driver
    /// operation fails.
    pub fn map_fresh(&mut self) -> Result<(), CudaVmmError> {
        if self.physical.is_some() {
            return Err(CudaVmmError::AlreadyMapped);
        }
        self.context.bind_to_thread()?;
        let property = allocation_property(self.context.cu_device());
        let mut handle = MaybeUninit::uninit();
        unsafe {
            sys::cuMemCreate(handle.as_mut_ptr(), self.bytes, &raw const property, 0).result()?;
        }
        let handle = unsafe { handle.assume_init() };
        if let Err(error) =
            unsafe { sys::cuMemMap(self.address, self.bytes, 0, handle, 0).result() }
        {
            unsafe {
                let _ = sys::cuMemRelease(handle).result();
            }
            return Err(error.into());
        }
        let access = sys::CUmemAccessDesc {
            location: property.location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        if let Err(error) =
            unsafe { sys::cuMemSetAccess(self.address, self.bytes, &raw const access, 1).result() }
        {
            unsafe {
                let _ = sys::cuMemUnmap(self.address, self.bytes).result();
                let _ = sys::cuMemRelease(handle).result();
            }
            return Err(error.into());
        }
        self.physical = Some(PhysicalAllocation {
            handle,
            mapped: true,
        });
        Ok(())
    }

    /// Removes and releases the current physical backing while preserving the
    /// virtual address reservation.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot is not mapped or unmap/release fails.
    pub fn unmap(&mut self) -> Result<(), CudaVmmError> {
        let physical = self.physical.take().ok_or(CudaVmmError::NotMapped)?;
        self.context.bind_to_thread()?;
        let mut physical = physical;
        if physical.mapped {
            if let Err(error) = unsafe { sys::cuMemUnmap(self.address, self.bytes).result() } {
                self.physical = Some(physical);
                return Err(error.into());
            }
            physical.mapped = false;
        }
        if let Err(error) = unsafe { sys::cuMemRelease(physical.handle).result() } {
            self.physical = Some(physical);
            return Err(error.into());
        }
        Ok(())
    }

    /// Copies bytes from host memory into the mapped slot.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot is unmapped, the source is too large, or
    /// the CUDA copy fails.
    pub fn write(&self, source: &[u8]) -> Result<(), CudaVmmError> {
        if self.physical.is_none() {
            return Err(CudaVmmError::NotMapped);
        }
        if source.len() > self.bytes {
            return Err(CudaVmmError::SourceTooLarge {
                actual: source.len(),
                capacity: self.bytes,
            });
        }
        self.context.bind_to_thread()?;
        unsafe {
            result::memcpy_htod_sync(self.address, source)?;
        }
        Ok(())
    }

    /// Copies bytes from the mapped slot into host memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot is unmapped, the destination is too large,
    /// or the CUDA copy fails.
    pub fn read(&self, destination: &mut [u8]) -> Result<(), CudaVmmError> {
        if self.physical.is_none() {
            return Err(CudaVmmError::NotMapped);
        }
        if destination.len() > self.bytes {
            return Err(CudaVmmError::DestinationTooLarge {
                actual: destination.len(),
                capacity: self.bytes,
            });
        }
        self.context.bind_to_thread()?;
        unsafe {
            result::memcpy_dtoh_sync(destination, self.address)?;
        }
        Ok(())
    }

    /// Explicitly tears down physical backing and the virtual reservation.
    ///
    /// # Errors
    ///
    /// Returns an error if any CUDA cleanup operation fails.
    pub fn close(mut self) -> Result<(), CudaVmmError> {
        self.context.bind_to_thread()?;
        if self.physical.is_some() {
            self.unmap()?;
        }
        unsafe {
            sys::cuMemAddressFree(self.address, self.bytes).result()?;
        }
        self.address = 0;
        self.bytes = 0;
        Ok(())
    }
}

impl Drop for CudaVmmSlot {
    fn drop(&mut self) {
        if self.address == 0 {
            return;
        }
        let _ = self.context.bind_to_thread();
        if let Some(physical) = self.physical.take() {
            unsafe {
                if physical.mapped {
                    let _ = sys::cuMemUnmap(self.address, self.bytes).result();
                }
                let _ = sys::cuMemRelease(physical.handle).result();
            }
        }
        unsafe {
            let _ = sys::cuMemAddressFree(self.address, self.bytes).result();
        }
    }
}

/// Queries the NVIDIA CUDA VMM capabilities of one device.
///
/// # Errors
///
/// Returns an error if CUDA initialization or capability queries fail.
pub fn probe(device_ordinal: usize) -> Result<CudaVmmCapabilities, CudaVmmError> {
    let context = CudaContext::new(device_ordinal)?;
    probe_context(&context)
}

fn probe_context(context: &CudaContext) -> Result<CudaVmmCapabilities, CudaVmmError> {
    context.bind_to_thread()?;
    let device = context.cu_device();
    let virtual_address_management = unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        )?
    } > 0;
    let (compute_capability_major, compute_capability_minor) = context.compute_capability()?;
    let property = allocation_property(device);
    let mut minimum = 0;
    let mut recommended = 0;
    if virtual_address_management {
        unsafe {
            sys::cuMemGetAllocationGranularity(
                &raw mut minimum,
                &raw const property,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
            .result()?;
            sys::cuMemGetAllocationGranularity(
                &raw mut recommended,
                &raw const property,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
            .result()?;
        }
    }
    Ok(CudaVmmCapabilities {
        device_ordinal: context.ordinal(),
        device_name: context.name()?,
        compute_capability_major,
        compute_capability_minor,
        virtual_address_management,
        minimum_granularity_bytes: minimum,
        recommended_granularity_bytes: recommended,
    })
}

fn allocation_property(device: sys::CUdevice) -> sys::CUmemAllocationProp {
    sys::CUmemAllocationProp {
        type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
            compressionType: 0,
            gpuDirectRDMACapable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, CudaVmmError> {
    debug_assert!(alignment > 0);
    value
        .checked_add(alignment - 1)
        .map(|sum| (sum / alignment) * alignment)
        .ok_or(CudaVmmError::AlignmentOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_is_checked() {
        assert_eq!(align_up(1, 64).unwrap(), 64);
        assert_eq!(align_up(64, 64).unwrap(), 64);
        assert_eq!(align_up(65, 64).unwrap(), 128);
        assert_eq!(
            align_up(usize::MAX, 64).unwrap_err().to_string(),
            CudaVmmError::AlignmentOverflow.to_string()
        );
    }
}
