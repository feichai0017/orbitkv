//! Typed buffer ownership and opaque binding handles.

use super::CommandError;
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

    /// Adds a read-only buffer and returns its opaque handle.
    pub fn bind_read<T: BindingElement>(
        &mut self,
        buffer: Arc<DeviceBuffer<T>>,
    ) -> Result<Read<T>, BindError<Arc<DeviceBuffer<T>>>> {
        let slot = match self
            .check_buffer(&buffer)
            .and_then(|()| self.reserve_slot())
        {
            Ok(slot) => slot,
            Err(error) => {
                return Err(BindError {
                    error,
                    resource: buffer,
                });
            }
        };
        let ErasedLease(lease) = T::__erase_read(buffer);
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
        let slot = match self
            .check_buffer(&buffer)
            .and_then(|()| self.reserve_slot())
        {
            Ok(slot) => slot,
            Err(error) => {
                return Err(BindError {
                    error,
                    resource: buffer,
                });
            }
        };
        let ErasedLease(lease) = T::__erase_read_write(buffer);
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
    ) -> Result<DeviceBuffer<T>, CommandError> {
        if handle.set_id != self.set_id {
            return Err(CommandError::BindingSetMismatch);
        }
        if handle.slot >= self.leases.len() {
            return Err(CommandError::BindingSlotOutOfRange {
                slot: handle.slot,
                bindings: self.leases.len(),
            });
        }
        T::__take_read_write(self, handle.slot)
    }

    fn check_buffer<T: BindingElement>(
        &self,
        buffer: &DeviceBuffer<T>,
    ) -> Result<(), CommandError> {
        let buffer_context = buffer.context();
        let stream_context = self.stream.context();
        if buffer_context.cu_ctx() == stream_context.cu_ctx() {
            Ok(())
        } else {
            Err(CommandError::BufferContextMismatch {
                buffer_device: buffer_context.ordinal(),
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

pub(crate) enum Access<T: DeviceCopy> {
    Read(Arc<DeviceBuffer<T>>),
    ReadWrite(DeviceBuffer<T>),
}

pub(crate) enum Lease {
    F32(Access<f32>),
    F16(Access<f16>),
    Bf16(Access<bf16>),
    U8(Access<u8>),
    Vacant,
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
    fn __erase_read(buffer: Arc<DeviceBuffer<Self>>) -> ErasedLease;

    #[doc(hidden)]
    fn __erase_read_write(buffer: DeviceBuffer<Self>) -> ErasedLease;

    #[doc(hidden)]
    fn __take_read_write(
        bindings: &mut CheckedBindings,
        slot: usize,
    ) -> Result<DeviceBuffer<Self>, CommandError>;
}

/// Opaque erased storage for one binding.
///
/// This type exists only to keep the sealed [`BindingElement`] interface
/// visibility-correct. Its payload is private and cannot be forged downstream.
#[doc(hidden)]
pub struct ErasedLease(Lease);

pub(crate) trait ResolveElement: BindingElement {
    fn read(lease: &Lease) -> Result<&DeviceBuffer<Self>, LeaseError>;

    fn write(lease: &mut Lease) -> Result<&mut DeviceBuffer<Self>, LeaseError>;
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
            fn __erase_read(buffer: Arc<DeviceBuffer<Self>>) -> ErasedLease {
                ErasedLease(Lease::$variant(Access::Read(buffer)))
            }

            fn __erase_read_write(buffer: DeviceBuffer<Self>) -> ErasedLease {
                ErasedLease(Lease::$variant(Access::ReadWrite(buffer)))
            }

            fn __take_read_write(
                bindings: &mut CheckedBindings,
                slot: usize,
            ) -> Result<DeviceBuffer<Self>, CommandError> {
                let lease = bindings
                    .leases
                    .get_mut(slot)
                    .expect("binding slot was validated before removal");
                let owned = std::mem::replace(lease, Lease::Vacant);
                match owned {
                    Lease::$variant(Access::ReadWrite(buffer)) => Ok(buffer),
                    Lease::$variant(Access::Read(buffer)) => {
                        *lease = Lease::$variant(Access::Read(buffer));
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
            fn read(lease: &Lease) -> Result<&DeviceBuffer<Self>, LeaseError> {
                match lease {
                    Lease::$variant(Access::Read(buffer)) => Ok(buffer.as_ref()),
                    Lease::$variant(Access::ReadWrite(buffer)) => Ok(buffer),
                    Lease::Vacant => Err(LeaseError::Vacant),
                    _ => Err(LeaseError::ElementMismatch),
                }
            }

            fn write(lease: &mut Lease) -> Result<&mut DeviceBuffer<Self>, LeaseError> {
                match lease {
                    Lease::$variant(Access::ReadWrite(buffer)) => Ok(buffer),
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
