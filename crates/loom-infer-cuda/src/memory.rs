//! Retained, typed views over CUDA device memory.
//!
//! Command bindings use these regions for both Loom-owned allocations and
//! memory owned by an embedding inference engine. The public constructors
//! establish one pointer, extent, context, and lifetime contract before the
//! region reaches a kernel launch.

use cuda_core::sys::CUdeviceptr;
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchContractError,
    PinnedHostBuffer,
};
use cuda_host::cuda_async::device_operation::DeviceOperation;
use cuda_host::cuda_async::error::DeviceError;
use cuda_host::{KernelSliceArg, KernelSliceArgMut};
use std::any::Any;
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::Range;
use std::sync::Arc;
use thiserror::Error;

/// Identifies who owns the allocation behind a retained device region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceRegionOwner {
    /// The region retains a cuda-oxide [`DeviceBuffer`].
    DeviceBuffer,
    /// The region retains an opaque allocation lease supplied by an engine.
    External,
}

/// Submits a prepared region-backed launch to one exact command stream.
///
/// The generated cuda-oxide async launcher accepts [`KernelSliceArg`] instead
/// of requiring `DeviceBuffer`. This adapter executes that inert launch recipe
/// immediately on the caller's stream. Resource lifetime remains owned by the
/// command binding set.
pub(crate) fn enqueue_region_launch<O>(
    stream: &Arc<CudaStream>,
    operation: Result<O, LaunchContractError>,
) -> Result<(), DeviceRegionLaunchError>
where
    O: DeviceOperation<Output = ()>,
{
    let operation = operation.map_err(DeviceRegionLaunchError::Contract)?;
    // SAFETY: CommandScope retains every bound region until its completion
    // fence settles. It records the returned launch before any resource can be
    // extracted, and CommandQueue owns `stream` for the complete scope.
    unsafe { operation.async_on(stream) }.map_err(DeviceRegionLaunchError::Device)
}

/// Enqueues one fixed-size status packet into retained pinned host storage.
///
/// The command binding set owns `host` until its completion event settles.
/// The source pointer belongs to a retained read-write binding on `stream`.
pub(crate) fn enqueue_status_packet_copy(
    stream: &CudaStream,
    source: CUdeviceptr,
    host: &mut PinnedHostBuffer<i32>,
    host_offset: usize,
    words: usize,
) -> Result<(), DriverError> {
    let host_end = host_offset
        .checked_add(words)
        .expect("status packet host range is prevalidated");
    assert!(
        host_end <= host.len(),
        "status packet exceeds retained pinned host storage"
    );
    stream.context().bind_to_thread()?;
    let num_bytes = words
        .checked_mul(size_of::<i32>())
        .expect("status packet byte count is prevalidated");
    // SAFETY: the binding set retains the source region and pinned host
    // allocation through the final event. Their checked spans cover this
    // packet, and the copy is ordered on the command stream.
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            host.as_mut_ptr().add(host_offset),
            source,
            num_bytes,
            stream.cu_stream(),
        )
    }
}

#[derive(Debug, Error)]
pub enum DeviceRegionLaunchError {
    #[error(transparent)]
    Contract(LaunchContractError),
    #[error(transparent)]
    Device(DeviceError),
}

impl DeviceRegionLaunchError {
    pub(crate) const fn driver_error(&self) -> Option<DriverError> {
        match self {
            Self::Contract(LaunchContractError::Driver(error))
            | Self::Device(DeviceError::Driver(error)) => Some(*error),
            Self::Contract(_) | Self::Device(_) => None,
        }
    }
}

/// A read-only device-memory region with a retained allocation lease.
pub struct ReadDeviceRegion<T: DeviceCopy> {
    view: DeviceRegion<T>,
    lease: ReadLease<T>,
}

impl<T: DeviceCopy> Clone for ReadDeviceRegion<T> {
    fn clone(&self) -> Self {
        Self {
            view: self.view.clone(),
            lease: self.lease.clone(),
        }
    }
}

impl<T: DeviceCopy> ReadDeviceRegion<T> {
    /// Retains and exposes the full device buffer.
    pub fn from_buffer(buffer: Arc<DeviceBuffer<T>>) -> Self {
        let view = DeviceRegion::trusted_whole(
            buffer.cu_deviceptr(),
            buffer.len(),
            buffer.context().clone(),
        );
        Self {
            view,
            lease: ReadLease::Buffer(buffer),
        }
    }

    /// Retains a device buffer and exposes only `range`.
    pub fn from_buffer_range(
        buffer: Arc<DeviceBuffer<T>>,
        range: Range<usize>,
    ) -> Result<Self, DeviceRegionBuildError<Arc<DeviceBuffer<T>>>> {
        let span = match checked_span::<T>(buffer.cu_deviceptr(), buffer.len(), range) {
            Ok(span) => span,
            Err(error) => {
                return Err(DeviceRegionBuildError {
                    error,
                    resource: buffer,
                });
            }
        };
        let context = buffer.context().clone();
        Ok(Self {
            view: DeviceRegion::new(span, context),
            lease: ReadLease::Buffer(buffer),
        })
    }

    /// Retains an external allocation and exposes its full typed extent.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of the following:
    ///
    /// - `pointer` identifies `len` live, initialized `T` elements in
    ///   `context`; a non-empty region cannot use a null pointer.
    /// - The pointer is aligned for `T`, and `len * size_of::<T>()` is inside
    ///   the allocation. This constructor checks arithmetic and alignment but
    ///   cannot inspect the CUDA allocation.
    /// - `lease` keeps the allocation alive until its final clone is dropped.
    /// - Every device use of the allocation is ordered against the command
    ///   stream by the caller.
    /// - The caller owns shared-read authority for the full reported range.
    ///   No unsynchronized writer may access it while this region is bound.
    pub unsafe fn from_external_parts(
        pointer: CUdeviceptr,
        len: usize,
        context: Arc<CudaContext>,
        lease: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, DeviceRegionError> {
        // SAFETY: the caller accepts the external allocation contract above;
        // the range covers the complete reported allocation.
        unsafe { Self::from_external_range(pointer, len, 0..len, context, lease) }
    }

    /// Retains an external allocation and exposes only `range`.
    ///
    /// # Safety
    ///
    /// The requirements of [`Self::from_external_parts`] apply to the full
    /// `allocation_len` extent. In addition, `range` is expressed in `T`
    /// elements relative to `pointer`. The retained lease must provide
    /// shared-read authority for the selected range.
    pub unsafe fn from_external_range(
        pointer: CUdeviceptr,
        allocation_len: usize,
        range: Range<usize>,
        context: Arc<CudaContext>,
        lease: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, DeviceRegionError> {
        let span = checked_span::<T>(pointer, allocation_len, range)?;
        Ok(Self {
            view: DeviceRegion::new(span, context),
            lease: ReadLease::External { _lease: lease },
        })
    }

    pub const fn len(&self) -> usize {
        self.view.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.view.is_empty()
    }

    pub const fn num_bytes(&self) -> usize {
        self.view.num_bytes()
    }

    pub const fn cu_deviceptr(&self) -> CUdeviceptr {
        self.view.cu_deviceptr()
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        self.view.context()
    }

    /// Returns the allocation owner retained by this region.
    pub const fn owner(&self) -> DeviceRegionOwner {
        self.lease.owner()
    }

    pub(crate) const fn view(&self) -> &DeviceRegion<T> {
        &self.view
    }

    pub(crate) fn into_buffer(self) -> Result<Arc<DeviceBuffer<T>>, Self> {
        match self {
            Self {
                lease: ReadLease::Buffer(buffer),
                ..
            } => Ok(buffer),
            external => Err(external),
        }
    }
}

impl<T: DeviceCopy> fmt::Debug for ReadDeviceRegion<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReadDeviceRegion")
            .field("pointer", &self.cu_deviceptr())
            .field("len", &self.len())
            .field("device", &self.context().ordinal())
            .field("owner", &self.lease.kind())
            .finish()
    }
}

/// An exclusively writable device-memory region with a retained allocation
/// lease.
pub struct ReadWriteDeviceRegion<T: DeviceCopy> {
    view: DeviceRegion<T>,
    lease: ReadWriteLease<T>,
}

impl<T: DeviceCopy> ReadWriteDeviceRegion<T> {
    /// Transfers a full device buffer into one writable region.
    pub fn from_buffer(buffer: DeviceBuffer<T>) -> Self {
        let view = DeviceRegion::trusted_whole(
            buffer.cu_deviceptr(),
            buffer.len(),
            buffer.context().clone(),
        );
        Self {
            view,
            lease: ReadWriteLease::Buffer(buffer),
        }
    }

    /// Transfers a device buffer and exposes only `range` for device access.
    pub fn from_buffer_range(
        buffer: DeviceBuffer<T>,
        range: Range<usize>,
    ) -> Result<Self, DeviceRegionBuildError<DeviceBuffer<T>>> {
        let span = match checked_span::<T>(buffer.cu_deviceptr(), buffer.len(), range) {
            Ok(span) => span,
            Err(error) => {
                return Err(DeviceRegionBuildError {
                    error,
                    resource: buffer,
                });
            }
        };
        let context = buffer.context().clone();
        Ok(Self {
            view: DeviceRegion::new(span, context),
            lease: ReadWriteLease::Buffer(buffer),
        })
    }

    /// Retains an external allocation and exposes its full typed extent for
    /// reads and writes.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of the following:
    ///
    /// - `pointer` identifies `len` live `T` elements in `context`; a
    ///   non-empty region cannot use a null pointer.
    /// - The pointer is aligned for `T`, and `len * size_of::<T>()` is inside
    ///   the allocation. This constructor checks arithmetic and alignment but
    ///   cannot inspect the CUDA allocation.
    /// - `lease` keeps the allocation alive until its final clone is dropped.
    /// - Every device use of the allocation is ordered against the command
    ///   stream by the caller.
    /// - The caller transfers exclusive read-write authority for the full
    ///   reported range to this value. No alias may read or write that range
    ///   until the region is returned from the completed binding set.
    pub unsafe fn from_external_parts(
        pointer: CUdeviceptr,
        len: usize,
        context: Arc<CudaContext>,
        lease: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, DeviceRegionError> {
        // SAFETY: the caller accepts the external allocation contract above;
        // the range covers the complete reported allocation.
        unsafe { Self::from_external_range(pointer, len, 0..len, context, lease) }
    }

    /// Retains an external allocation and exposes only `range` for reads and
    /// writes.
    ///
    /// # Safety
    ///
    /// The requirements of [`Self::from_external_parts`] apply to the full
    /// `allocation_len` extent. In addition, `range` is expressed in `T`
    /// elements relative to `pointer`. The caller transfers exclusive
    /// read-write authority for the selected range to this value.
    pub unsafe fn from_external_range(
        pointer: CUdeviceptr,
        allocation_len: usize,
        range: Range<usize>,
        context: Arc<CudaContext>,
        lease: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, DeviceRegionError> {
        let span = checked_span::<T>(pointer, allocation_len, range)?;
        Ok(Self {
            view: DeviceRegion::new(span, context),
            lease: ReadWriteLease::External { _lease: lease },
        })
    }

    pub const fn len(&self) -> usize {
        self.view.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.view.is_empty()
    }

    pub const fn num_bytes(&self) -> usize {
        self.view.num_bytes()
    }

    pub const fn cu_deviceptr(&self) -> CUdeviceptr {
        self.view.cu_deviceptr()
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        self.view.context()
    }

    /// Returns the allocation owner retained by this region.
    pub const fn owner(&self) -> DeviceRegionOwner {
        self.lease.owner()
    }

    /// Recovers a Loom-owned allocation. External regions are returned
    /// unchanged because their allocation is owned by the retained lease.
    pub fn into_buffer(self) -> Result<DeviceBuffer<T>, Self> {
        match self {
            Self {
                lease: ReadWriteLease::Buffer(buffer),
                ..
            } => Ok(buffer),
            external => Err(external),
        }
    }

    pub(crate) const fn view(&self) -> &DeviceRegion<T> {
        &self.view
    }
}

impl<T: DeviceCopy> fmt::Debug for ReadWriteDeviceRegion<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReadWriteDeviceRegion")
            .field("pointer", &self.cu_deviceptr())
            .field("len", &self.len())
            .field("device", &self.context().ordinal())
            .field("owner", &self.lease.kind())
            .finish()
    }
}

/// Failure to build a checked region from an owned resource.
///
/// The resource is returned so a bad range cannot destroy an allocation.
pub struct DeviceRegionBuildError<R> {
    error: DeviceRegionError,
    resource: R,
}

impl<R> DeviceRegionBuildError<R> {
    pub const fn error(&self) -> &DeviceRegionError {
        &self.error
    }

    pub fn into_parts(self) -> (DeviceRegionError, R) {
        (self.error, self.resource)
    }
}

impl<R> fmt::Debug for DeviceRegionBuildError<R> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceRegionBuildError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<R> Display for DeviceRegionBuildError<R> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.error, formatter)
    }
}

impl<R: 'static> std::error::Error for DeviceRegionBuildError<R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum DeviceRegionError {
    #[error("device regions do not support zero-sized element types")]
    ZeroSizedElement,
    #[error("device region range start {start} is greater than end {end}")]
    ReversedRange { start: usize, end: usize },
    #[error("device region range {start}..{end} exceeds the {allocation_len}-element allocation")]
    RangeOutOfBounds {
        start: usize,
        end: usize,
        allocation_len: usize,
    },
    #[error("device region extent overflows: {elements} elements of {element_size} bytes each")]
    ExtentOverflow {
        elements: usize,
        element_size: usize,
    },
    #[error("device region pointer arithmetic overflows the CUDA address space")]
    PointerOverflow,
    #[error("non-empty device region has a null pointer")]
    NullPointer,
    #[error("device pointer {pointer:#x} is not aligned to {alignment} bytes")]
    MisalignedPointer {
        pointer: CUdeviceptr,
        alignment: usize,
    },
}

pub(crate) struct DeviceRegion<T: DeviceCopy> {
    pointer: CUdeviceptr,
    len: usize,
    num_bytes: usize,
    context: Arc<CudaContext>,
    element: PhantomData<T>,
}

impl<T: DeviceCopy> Clone for DeviceRegion<T> {
    fn clone(&self) -> Self {
        Self {
            pointer: self.pointer,
            len: self.len,
            num_bytes: self.num_bytes,
            context: self.context.clone(),
            element: PhantomData,
        }
    }
}

impl<T: DeviceCopy> DeviceRegion<T> {
    fn new(span: CheckedSpan, context: Arc<CudaContext>) -> Self {
        Self {
            pointer: span.pointer,
            len: span.len,
            num_bytes: span.num_bytes,
            context,
            element: PhantomData,
        }
    }

    fn trusted_whole(pointer: CUdeviceptr, len: usize, context: Arc<CudaContext>) -> Self {
        let num_bytes = len
            .checked_mul(size_of::<T>())
            .expect("DeviceBuffer has a checked byte extent");
        Self {
            pointer,
            len,
            num_bytes,
            context,
            element: PhantomData,
        }
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) const fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    pub(crate) const fn cu_deviceptr(&self) -> CUdeviceptr {
        self.pointer
    }

    pub(crate) const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}

enum ReadLease<T> {
    Buffer(Arc<DeviceBuffer<T>>),
    External { _lease: Arc<dyn Any + Send + Sync> },
}

impl<T> Clone for ReadLease<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Buffer(buffer) => Self::Buffer(buffer.clone()),
            Self::External { _lease: lease } => Self::External {
                _lease: lease.clone(),
            },
        }
    }
}

impl<T> ReadLease<T> {
    const fn owner(&self) -> DeviceRegionOwner {
        match self {
            Self::Buffer(_) => DeviceRegionOwner::DeviceBuffer,
            Self::External { .. } => DeviceRegionOwner::External,
        }
    }

    const fn kind(&self) -> &'static str {
        match self.owner() {
            DeviceRegionOwner::DeviceBuffer => "device-buffer",
            DeviceRegionOwner::External => "external",
        }
    }
}

enum ReadWriteLease<T> {
    Buffer(DeviceBuffer<T>),
    External { _lease: Arc<dyn Any + Send + Sync> },
}

impl<T> ReadWriteLease<T> {
    const fn owner(&self) -> DeviceRegionOwner {
        match self {
            Self::Buffer(_) => DeviceRegionOwner::DeviceBuffer,
            Self::External { .. } => DeviceRegionOwner::External,
        }
    }

    const fn kind(&self) -> &'static str {
        match self.owner() {
            DeviceRegionOwner::DeviceBuffer => "device-buffer",
            DeviceRegionOwner::External => "external",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CheckedSpan {
    pointer: CUdeviceptr,
    len: usize,
    num_bytes: usize,
}

fn checked_span<T>(
    base_pointer: CUdeviceptr,
    allocation_len: usize,
    range: Range<usize>,
) -> Result<CheckedSpan, DeviceRegionError> {
    let element_size = size_of::<T>();
    if element_size == 0 {
        return Err(DeviceRegionError::ZeroSizedElement);
    }
    if range.start > range.end {
        return Err(DeviceRegionError::ReversedRange {
            start: range.start,
            end: range.end,
        });
    }
    if range.end > allocation_len {
        return Err(DeviceRegionError::RangeOutOfBounds {
            start: range.start,
            end: range.end,
            allocation_len,
        });
    }
    if allocation_len > 0 && base_pointer == 0 {
        return Err(DeviceRegionError::NullPointer);
    }
    if !base_pointer.is_multiple_of(align_of::<T>() as CUdeviceptr) {
        return Err(DeviceRegionError::MisalignedPointer {
            pointer: base_pointer,
            alignment: align_of::<T>(),
        });
    }

    let allocation_bytes =
        allocation_len
            .checked_mul(element_size)
            .ok_or(DeviceRegionError::ExtentOverflow {
                elements: allocation_len,
                element_size,
            })?;
    let byte_offset =
        range
            .start
            .checked_mul(element_size)
            .ok_or(DeviceRegionError::ExtentOverflow {
                elements: range.start,
                element_size,
            })?;
    let len = range.end - range.start;
    let num_bytes = len
        .checked_mul(element_size)
        .ok_or(DeviceRegionError::ExtentOverflow {
            elements: len,
            element_size,
        })?;
    let allocation_bytes =
        CUdeviceptr::try_from(allocation_bytes).map_err(|_| DeviceRegionError::PointerOverflow)?;
    let byte_offset =
        CUdeviceptr::try_from(byte_offset).map_err(|_| DeviceRegionError::PointerOverflow)?;
    base_pointer
        .checked_add(allocation_bytes)
        .ok_or(DeviceRegionError::PointerOverflow)?;
    let pointer = base_pointer
        .checked_add(byte_offset)
        .ok_or(DeviceRegionError::PointerOverflow)?;
    if len > 0 && pointer == 0 {
        return Err(DeviceRegionError::NullPointer);
    }

    Ok(CheckedSpan {
        pointer,
        len,
        num_bytes,
    })
}

// SAFETY: DeviceRegion values can only be built by checked constructors in
// this module. Their wrapper retains the allocation lease for every borrow,
// while this view reports the checked pointer and element extent unchanged.
unsafe impl<T: DeviceCopy> KernelSliceArg for DeviceRegion<T> {
    type Elem = T;

    fn cu_deviceptr(&self) -> CUdeviceptr {
        self.pointer
    }

    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: the read wrapper retains either an Arc<DeviceBuffer<T>> or the
// caller-provided external lease. Safe constructors only create shared reads;
// unsafe external constructors require the same shared-read authority.
unsafe impl<T: DeviceCopy> KernelSliceArg for ReadDeviceRegion<T> {
    type Elem = T;

    fn cu_deviceptr(&self) -> CUdeviceptr {
        self.view.pointer
    }

    fn len(&self) -> usize {
        self.view.len
    }
}

// SAFETY: the writable wrapper owns a DeviceBuffer<T> or an external lease for
// which its unsafe constructor requires exclusive read-write authority. The
// wrapper cannot be cloned, and binding transfers it until GPU completion.
unsafe impl<T: DeviceCopy> KernelSliceArg for ReadWriteDeviceRegion<T> {
    type Elem = T;

    fn cu_deviceptr(&self) -> CUdeviceptr {
        self.view.pointer
    }

    fn len(&self) -> usize {
        self.view.len
    }
}

// SAFETY: ReadWriteDeviceRegion's construction and ownership rules provide
// exclusive device-write authority for the reported range.
unsafe impl<T: DeviceCopy> KernelSliceArgMut for ReadWriteDeviceRegion<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn checked_span_selects_an_element_subrange() {
        let span = checked_span::<u32>(0x1000, 8, 2..6).unwrap();
        assert_eq!(
            span,
            CheckedSpan {
                pointer: 0x1008,
                len: 4,
                num_bytes: 16,
            }
        );
    }

    #[test]
    fn checked_span_rejects_reversed_and_out_of_bounds_ranges() {
        assert!(matches!(
            checked_span::<u32>(0x1000, 8, Range { start: 6, end: 2 }),
            Err(DeviceRegionError::ReversedRange { start: 6, end: 2 })
        ));
        assert!(matches!(
            checked_span::<u32>(0x1000, 8, 2..9),
            Err(DeviceRegionError::RangeOutOfBounds {
                start: 2,
                end: 9,
                allocation_len: 8,
            })
        ));
    }

    #[test]
    fn checked_span_rejects_extent_and_pointer_overflow() {
        assert!(matches!(
            checked_span::<u16>(0x1000, usize::MAX, 0..1),
            Err(DeviceRegionError::ExtentOverflow { .. })
        ));
        assert!(matches!(
            checked_span::<u8>(CUdeviceptr::MAX, 2, 0..1),
            Err(DeviceRegionError::PointerOverflow)
        ));
    }

    #[test]
    fn checked_span_rejects_misalignment_and_nonempty_null() {
        assert!(matches!(
            checked_span::<u32>(0x1001, 4, 0..4),
            Err(DeviceRegionError::MisalignedPointer { .. })
        ));
        assert_eq!(
            checked_span::<u8>(0, 0, 0..0).unwrap(),
            CheckedSpan {
                pointer: 0,
                len: 0,
                num_bytes: 0,
            }
        );
        assert!(matches!(
            checked_span::<u8>(0, 1, 0..1),
            Err(DeviceRegionError::NullPointer)
        ));
        assert!(matches!(
            checked_span::<u8>(0, 4, 1..2),
            Err(DeviceRegionError::NullPointer)
        ));
    }

    #[test]
    fn external_lease_is_retained_by_its_owner() {
        struct DropProbe(Arc<AtomicUsize>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let lease: Arc<dyn Any + Send + Sync> = Arc::new(DropProbe(drops.clone()));
        let owner = ReadLease::<u8>::External {
            _lease: lease.clone(),
        };
        let owner_clone = owner.clone();
        drop(lease);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(owner);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(owner_clone);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn read_device_region_is_cloneable() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<ReadDeviceRegion<u8>>();
    }
}
