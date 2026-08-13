//! Typed buffer ownership and opaque binding handles.

use super::CommandError;
use super::status::DeviceStatusState;
use crate::memory::{DeviceRegion, DeviceRegionOwner, ReadDeviceRegion, ReadWriteDeviceRegion};
use cuda_core::sys::CUdeviceptr;
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use half::{bf16, f16};
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;

/// A reusable set of checked, heterogeneous buffer leases.
///
/// Moving this value into a [`super::CommandScope`] retains shared read
/// allocations and transfers writable allocations until the returned
/// completion is settled. Owning every writable allocation makes the
/// asynchronous contract safe even if a scope or completion is leaked.
pub struct CheckedBindings {
    pub(super) queue_id: u64,
    pub(super) set_id: u64,
    pub(super) stream: Arc<CudaStream>,
    pub(super) leases: Vec<Lease>,
    pub(super) capacity: usize,
    pub(crate) status: DeviceStatusState,
}

/// Allocation provenance for one checked binding set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingMemorySummary {
    device_buffers: usize,
    external_regions: usize,
}

impl BindingMemorySummary {
    pub(crate) const fn from_counts(device_buffers: usize, external_regions: usize) -> Self {
        Self {
            device_buffers,
            external_regions,
        }
    }

    pub const fn device_buffers(self) -> usize {
        self.device_buffers
    }

    pub const fn external_regions(self) -> usize {
        self.external_regions
    }

    pub const fn total(self) -> usize {
        self.device_buffers + self.external_regions
    }

    /// Returns whether every bound operator buffer is externally owned.
    pub const fn all_external(self) -> bool {
        self.external_regions > 0 && self.device_buffers == 0
    }
}

impl CheckedBindings {
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.leases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leases.is_empty()
    }

    /// Summarizes the allocation owners retained by this binding set.
    pub fn memory_summary(&self) -> BindingMemorySummary {
        let mut summary = BindingMemorySummary::from_counts(0, 0);
        for lease in &self.leases {
            match lease.owner() {
                Some(DeviceRegionOwner::DeviceBuffer) => summary.device_buffers += 1,
                Some(DeviceRegionOwner::External) => summary.external_regions += 1,
                None => {}
            }
        }
        summary
    }

    pub(crate) fn prepare_for_reuse(&mut self) -> Result<(), CommandError> {
        assert!(
            self.status.is_empty(),
            "only settled bindings may be prepared for reuse"
        );
        self.leases.clear();
        self.set_id = super::submission::fresh_id()?;
        Ok(())
    }

    pub(crate) fn live_regions(&self) -> usize {
        self.leases
            .iter()
            .filter(|lease| lease.device_address().is_some())
            .count()
    }

    /// Returns exact device addresses in stable binding-slot order without
    /// allocating. Vacant slots and any count other than `N` are rejected.
    pub(crate) fn exact_device_addresses<const N: usize>(&self) -> Option<[CUdeviceptr; N]> {
        if self.leases.len() != N {
            return None;
        }
        let mut addresses = [0; N];
        for (index, lease) in self.leases.iter().enumerate() {
            addresses[index] = lease.device_address()?;
        }
        Some(addresses)
    }

    /// Adds a read-only buffer and returns its opaque handle.
    pub fn bind_read<T: BindingElement>(
        &mut self,
        buffer: Arc<DeviceBuffer<T>>,
    ) -> Result<Read<T>, BindError<Arc<DeviceBuffer<T>>>> {
        match self.bind_read_region(ReadDeviceRegion::from_buffer(buffer)) {
            Ok(handle) => Ok(handle),
            Err(error) => {
                let (error, region) = error.into_parts();
                let buffer = region
                    .into_buffer()
                    .expect("a DeviceBuffer convenience region keeps its buffer owner");
                Err(BindError {
                    error,
                    resource: buffer,
                })
            }
        }
    }

    /// Adds a retained read-only device region and returns its opaque handle.
    pub fn bind_read_region<T: BindingElement>(
        &mut self,
        region: ReadDeviceRegion<T>,
    ) -> Result<Read<T>, BindError<ReadDeviceRegion<T>>> {
        let slot = match self
            .check_region_context(region.context())
            .and_then(|()| self.check_region_overlap(region.view(), AccessMode::Read))
            .and_then(|()| self.reserve_slot())
        {
            Ok(slot) => slot,
            Err(error) => {
                return Err(BindError {
                    error,
                    resource: region,
                });
            }
        };
        let ErasedLease(lease) = T::__erase_read(region);
        self.leases.push(lease);
        Ok(Read {
            set_id: self.set_id,
            slot,
            element: PhantomData,
        })
    }

    /// Transfers one buffer that may be read and written.
    pub fn bind_read_write<T: BindingElement>(
        &mut self,
        buffer: DeviceBuffer<T>,
    ) -> Result<ReadWrite<T>, BindError<DeviceBuffer<T>>> {
        match self.bind_read_write_region(ReadWriteDeviceRegion::from_buffer(buffer)) {
            Ok(handle) => Ok(handle),
            Err(error) => {
                let (error, region) = error.into_parts();
                let buffer = region
                    .into_buffer()
                    .expect("a DeviceBuffer convenience region keeps its buffer owner");
                Err(BindError {
                    error,
                    resource: buffer,
                })
            }
        }
    }

    /// Transfers one retained device region with exclusive write authority.
    pub fn bind_read_write_region<T: BindingElement>(
        &mut self,
        region: ReadWriteDeviceRegion<T>,
    ) -> Result<ReadWrite<T>, BindError<ReadWriteDeviceRegion<T>>> {
        let slot = match self
            .check_region_context(region.context())
            .and_then(|()| self.check_region_overlap(region.view(), AccessMode::ReadWrite))
            .and_then(|()| self.reserve_slot())
        {
            Ok(slot) => slot,
            Err(error) => {
                return Err(BindError {
                    error,
                    resource: region,
                });
            }
        };
        let ErasedLease(lease) = T::__erase_read_write(region);
        self.leases.push(lease);
        Ok(ReadWrite {
            set_id: self.set_id,
            slot,
            element: PhantomData,
        })
    }

    /// Removes one completed allocation from the binding arena.
    ///
    /// This is intended after [`super::CommandCompletion::wait`] or
    /// [`crate::graph::GraphExec::into_bindings`] returns ownership.
    pub fn take_read_write<T: BindingElement>(
        &mut self,
        handle: ReadWrite<T>,
    ) -> Result<DeviceBuffer<T>, TakeDeviceBufferError<T>> {
        let region = self
            .take_read_write_region(handle)
            .map_err(TakeDeviceBufferError::Command)?;
        region
            .into_buffer()
            .map_err(TakeDeviceBufferError::ExternalRegion)
    }

    /// Removes one completed writable region from the binding arena.
    ///
    /// This is the ownership-preserving extraction API for both Oxide-owned
    /// buffers and external engine allocations.
    pub fn take_read_write_region<T: BindingElement>(
        &mut self,
        handle: ReadWrite<T>,
    ) -> Result<ReadWriteDeviceRegion<T>, CommandError> {
        if handle.set_id != self.set_id {
            return Err(CommandError::BindingSetMismatch);
        }
        if handle.slot >= self.leases.len() {
            return Err(CommandError::BindingSlotOutOfRange {
                slot: handle.slot,
                bindings: self.leases.len(),
            });
        }
        T::__take_read_write_region(self, handle.slot)
    }

    fn check_region_context(
        &self,
        region_context: &Arc<cuda_core::CudaContext>,
    ) -> Result<(), CommandError> {
        let stream_context = self.stream.context();
        if region_context.cu_ctx() == stream_context.cu_ctx() {
            Ok(())
        } else {
            Err(CommandError::RegionContextMismatch {
                region_device: region_context.ordinal(),
                stream_device: stream_context.ordinal(),
            })
        }
    }

    fn reserve_slot(&self) -> Result<usize, CommandError> {
        if self.leases.len() == self.capacity {
            Err(CommandError::BindingCapacityExceeded {
                capacity: self.capacity,
            })
        } else {
            Ok(self.leases.len())
        }
    }

    fn check_region_overlap<T: DeviceCopy>(
        &self,
        region: &DeviceRegion<T>,
        access: AccessMode,
    ) -> Result<(), CommandError> {
        let incoming = RegionSpan::new(region.cu_deviceptr(), region.num_bytes());
        for (slot, lease) in self.leases.iter().enumerate() {
            let Some((existing, existing_access)) = lease.region_span() else {
                continue;
            };
            if (access.is_write() || existing_access.is_write()) && incoming.overlaps(existing) {
                return Err(CommandError::OverlappingDeviceRegions {
                    existing_slot: slot,
                });
            }
        }
        Ok(())
    }
}

/// A failed ownership transfer into a checked binding arena.
///
/// The original allocation is returned so a recoverable capacity or context
/// error never destroys caller data.
pub struct BindError<R> {
    error: CommandError,
    resource: R,
}

impl<R> BindError<R> {
    pub const fn error(&self) -> &CommandError {
        &self.error
    }

    pub fn into_parts(self) -> (CommandError, R) {
        (self.error, self.resource)
    }
}

impl<R> fmt::Debug for BindError<R> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BindError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<R> Display for BindError<R> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.error, formatter)
    }
}

impl<R: 'static> std::error::Error for BindError<R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Failure to recover a Oxide-owned `DeviceBuffer` from a writable binding.
///
/// An external region remains owned by this error and can be recovered with
/// [`Self::into_region`]. Command errors leave the resource in the binding set.
pub enum TakeDeviceBufferError<T: BindingElement> {
    Command(CommandError),
    ExternalRegion(ReadWriteDeviceRegion<T>),
}

impl<T: BindingElement> TakeDeviceBufferError<T> {
    pub const fn command_error(&self) -> Option<&CommandError> {
        match self {
            Self::Command(error) => Some(error),
            Self::ExternalRegion(_) => None,
        }
    }

    pub fn into_region(self) -> Option<ReadWriteDeviceRegion<T>> {
        match self {
            Self::Command(_) => None,
            Self::ExternalRegion(region) => Some(region),
        }
    }
}

impl<T: BindingElement> fmt::Debug for TakeDeviceBufferError<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Command(error) => formatter.debug_tuple("Command").field(error).finish(),
            Self::ExternalRegion(_) => formatter
                .debug_tuple("ExternalRegion")
                .field(&"retained")
                .finish(),
        }
    }
}

impl<T: BindingElement> Display for TakeDeviceBufferError<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Command(error) => Display::fmt(error, formatter),
            Self::ExternalRegion(_) => formatter.write_str(
                "the writable binding is an external device region, not an owned DeviceBuffer",
            ),
        }
    }
}

impl<T: BindingElement> std::error::Error for TakeDeviceBufferError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Command(error) => Some(error),
            Self::ExternalRegion(_) => None,
        }
    }
}

pub(crate) enum Access<T: DeviceCopy> {
    Read(ReadDeviceRegion<T>),
    ReadWrite(ReadWriteDeviceRegion<T>),
}

impl<T: DeviceCopy> Access<T> {
    fn owner(&self) -> DeviceRegionOwner {
        match self {
            Self::Read(region) => region.owner(),
            Self::ReadWrite(region) => region.owner(),
        }
    }

    fn region_span(&self) -> (RegionSpan, AccessMode) {
        match self {
            Self::Read(region) => (
                RegionSpan::new(region.cu_deviceptr(), region.num_bytes()),
                AccessMode::Read,
            ),
            Self::ReadWrite(region) => (
                RegionSpan::new(region.cu_deviceptr(), region.num_bytes()),
                AccessMode::ReadWrite,
            ),
        }
    }
}

pub(crate) enum Lease {
    F32(Access<f32>),
    F16(Access<f16>),
    Bf16(Access<bf16>),
    I32(Access<i32>),
    U8(Access<u8>),
    Vacant,
}

impl Lease {
    fn device_address(&self) -> Option<CUdeviceptr> {
        self.region_span().map(|(span, _)| span.start)
    }

    fn owner(&self) -> Option<DeviceRegionOwner> {
        match self {
            Self::F32(access) => Some(access.owner()),
            Self::F16(access) => Some(access.owner()),
            Self::Bf16(access) => Some(access.owner()),
            Self::I32(access) => Some(access.owner()),
            Self::U8(access) => Some(access.owner()),
            Self::Vacant => None,
        }
    }

    fn region_span(&self) -> Option<(RegionSpan, AccessMode)> {
        match self {
            Self::F32(access) => Some(access.region_span()),
            Self::F16(access) => Some(access.region_span()),
            Self::Bf16(access) => Some(access.region_span()),
            Self::I32(access) => Some(access.region_span()),
            Self::U8(access) => Some(access.region_span()),
            Self::Vacant => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AccessMode {
    Read,
    ReadWrite,
}

impl AccessMode {
    const fn is_write(self) -> bool {
        matches!(self, Self::ReadWrite)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RegionSpan {
    start: u64,
    bytes: u64,
}

impl RegionSpan {
    fn new(start: u64, bytes: usize) -> Self {
        Self {
            start,
            bytes: u64::try_from(bytes).expect("device region byte extent fits CUdeviceptr"),
        }
    }

    fn overlaps(self, other: Self) -> bool {
        if self.bytes == 0 || other.bytes == 0 {
            return false;
        }
        let self_end = self
            .start
            .checked_add(self.bytes)
            .expect("checked device region pointer extent");
        let other_end = other
            .start
            .checked_add(other.bytes)
            .expect("checked device region pointer extent");
        self.start < other_end && other.start < self_end
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A device-buffer element type accepted by the command binding arena.
///
/// The trait is sealed so every handle can be resolved without type erasure,
/// downcasts, or unsafe pointer casts.
pub trait BindingElement: DeviceCopy + sealed::Sealed + Sized + 'static {
    #[doc(hidden)]
    fn __erase_read(region: ReadDeviceRegion<Self>) -> ErasedLease;

    #[doc(hidden)]
    fn __erase_read_write(region: ReadWriteDeviceRegion<Self>) -> ErasedLease;

    #[doc(hidden)]
    fn __take_read_write_region(
        bindings: &mut CheckedBindings,
        slot: usize,
    ) -> Result<ReadWriteDeviceRegion<Self>, CommandError>;
}

/// Opaque erased storage for one binding.
///
/// This type exists only to keep the sealed [`BindingElement`] interface
/// visibility-correct. Its payload is private and cannot be forged downstream.
#[doc(hidden)]
pub struct ErasedLease(Lease);

pub(crate) trait ResolveElement: BindingElement {
    fn read(lease: &Lease) -> Result<&DeviceRegion<Self>, LeaseError>;

    fn write(lease: &mut Lease) -> Result<&mut ReadWriteDeviceRegion<Self>, LeaseError>;
}

pub(crate) enum LeaseError {
    ElementMismatch,
    ReadOnly,
    Vacant,
}

macro_rules! impl_binding_element {
    ($ty:ty, $variant:ident) => {
        impl sealed::Sealed for $ty {}

        impl BindingElement for $ty {
            fn __erase_read(region: ReadDeviceRegion<Self>) -> ErasedLease {
                ErasedLease(Lease::$variant(Access::Read(region)))
            }

            fn __erase_read_write(region: ReadWriteDeviceRegion<Self>) -> ErasedLease {
                ErasedLease(Lease::$variant(Access::ReadWrite(region)))
            }

            fn __take_read_write_region(
                bindings: &mut CheckedBindings,
                slot: usize,
            ) -> Result<ReadWriteDeviceRegion<Self>, CommandError> {
                let lease = bindings
                    .leases
                    .get_mut(slot)
                    .expect("binding slot was validated before removal");
                let owned = std::mem::replace(lease, Lease::Vacant);
                match owned {
                    Lease::$variant(Access::ReadWrite(region)) => Ok(region),
                    Lease::$variant(Access::Read(region)) => {
                        *lease = Lease::$variant(Access::Read(region));
                        Err(CommandError::BindingIsReadOnly { slot })
                    }
                    Lease::Vacant => Err(CommandError::BindingSlotVacant { slot }),
                    other => {
                        *lease = other;
                        Err(CommandError::BindingTypeMismatch { slot })
                    }
                }
            }
        }

        impl ResolveElement for $ty {
            fn read(lease: &Lease) -> Result<&DeviceRegion<Self>, LeaseError> {
                match lease {
                    Lease::$variant(Access::Read(region)) => Ok(region.view()),
                    Lease::$variant(Access::ReadWrite(region)) => Ok(region.view()),
                    Lease::Vacant => Err(LeaseError::Vacant),
                    _ => Err(LeaseError::ElementMismatch),
                }
            }

            fn write(lease: &mut Lease) -> Result<&mut ReadWriteDeviceRegion<Self>, LeaseError> {
                match lease {
                    Lease::$variant(Access::ReadWrite(region)) => Ok(region),
                    Lease::$variant(Access::Read(_)) => Err(LeaseError::ReadOnly),
                    Lease::Vacant => Err(LeaseError::Vacant),
                    _ => Err(LeaseError::ElementMismatch),
                }
            }
        }
    };
}

impl_binding_element!(f32, F32);
impl_binding_element!(f16, F16);
impl_binding_element!(bf16, Bf16);
impl_binding_element!(i32, I32);
impl_binding_element!(u8, U8);

/// Opaque read access to one checked binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Read<T: BindingElement> {
    pub(super) set_id: u64,
    pub(super) slot: usize,
    pub(super) element: PhantomData<fn() -> T>,
}

/// Opaque read-write access to one checked binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadWrite<T: BindingElement> {
    pub(super) set_id: u64,
    pub(super) slot: usize,
    pub(super) element: PhantomData<fn() -> T>,
}

impl<T: BindingElement> ReadWrite<T> {
    pub const fn read(self) -> Read<T> {
        Read {
            set_id: self.set_id,
            slot: self.slot,
            element: PhantomData,
        }
    }

    pub const fn write(self) -> Write<T> {
        Write {
            set_id: self.set_id,
            slot: self.slot,
            element: PhantomData,
        }
    }
}

/// Opaque write access to one checked binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Write<T: BindingElement> {
    pub(super) set_id: u64,
    pub(super) slot: usize,
    pub(super) element: PhantomData<fn() -> T>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_spans_detect_only_nonempty_intersections() {
        assert!(RegionSpan::new(100, 16).overlaps(RegionSpan::new(108, 16)));
        assert!(!RegionSpan::new(100, 8).overlaps(RegionSpan::new(108, 8)));
        assert!(!RegionSpan::new(100, 0).overlaps(RegionSpan::new(100, 8)));
    }
}
