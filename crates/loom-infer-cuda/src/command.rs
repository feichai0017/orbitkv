//! Checked CUDA resource bindings and stream-ordered command submission.

#![forbid(unsafe_code)]

use cuda_core::{CudaEvent, CudaFunction, CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use half::{bf16, f16};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A reusable submission queue for one exact CUDA stream.
///
/// The queue preallocates its completion event and function-retention storage.
/// Rust's mutable borrow rules prevent a second scope from re-recording the
/// event while an earlier completion is still alive.
pub struct CommandQueue {
    id: u64,
    stream: Arc<CudaStream>,
    completion_event: CudaEvent,
    retained_functions: Vec<CudaFunction>,
    max_launches: usize,
    poisoned: bool,
}

impl CommandQueue {
    /// Creates a queue for `stream` with storage for at most `max_launches`
    /// commands per scope.
    pub fn new(stream: Arc<CudaStream>, max_launches: usize) -> Result<Self, CommandError> {
        if max_launches == 0 {
            return Err(CommandError::ZeroLaunchCapacity);
        }

        let id = fresh_id()?;
        let completion_event = stream.context().new_event(None)?;
        Ok(Self {
            id,
            stream,
            completion_event,
            retained_functions: Vec::with_capacity(max_launches),
            max_launches,
            poisoned: false,
        })
    }

    /// Returns the exact stream used by every scope from this queue.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub const fn max_launches(&self) -> usize {
        self.max_launches
    }

    /// Creates reusable checked binding storage outside the enqueue path.
    pub fn bindings<'buffer>(
        &self,
        capacity: usize,
    ) -> Result<CheckedBindings<'buffer>, CommandError> {
        if capacity == 0 {
            return Err(CommandError::ZeroBindingCapacity);
        }

        Ok(CheckedBindings {
            queue_id: self.id,
            set_id: fresh_id()?,
            stream: self.stream.clone(),
            leases: Vec::with_capacity(capacity),
            capacity,
        })
    }

    /// Begins one stream-ordered command scope.
    pub fn begin<'queue, 'buffer>(
        &'queue mut self,
        bindings: CheckedBindings<'buffer>,
    ) -> Result<CommandScope<'queue, 'buffer>, CommandError> {
        if self.poisoned {
            return Err(CommandError::QueuePoisoned);
        }
        if bindings.queue_id != self.id
            || bindings.stream.cu_stream() != self.stream.cu_stream()
            || bindings.stream.context().cu_ctx() != self.stream.context().cu_ctx()
        {
            return Err(CommandError::BindingsQueueMismatch);
        }
        if !self.retained_functions.is_empty() {
            self.poisoned = true;
            return Err(CommandError::QueuePoisoned);
        }

        Ok(CommandScope {
            queue: Some(self),
            bindings: Some(bindings),
            submitted: 0,
            submission_error: None,
            finished: false,
        })
    }
}

/// A reusable set of checked, heterogeneous buffer leases.
///
/// Moving this value into a [`CommandScope`] keeps every buffer borrowed until
/// the returned completion is settled.
pub struct CheckedBindings<'buffer> {
    queue_id: u64,
    set_id: u64,
    stream: Arc<CudaStream>,
    leases: Vec<Lease<'buffer>>,
    capacity: usize,
}

impl<'buffer> CheckedBindings<'buffer> {
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
        buffer: &'buffer DeviceBuffer<T>,
    ) -> Result<Read<T>, CommandError> {
        self.check_buffer(buffer)?;
        let slot = self.reserve_slot()?;
        let ErasedLease(lease) = T::__erase_read(buffer);
        self.leases.push(lease);
        Ok(Read {
            set_id: self.set_id,
            slot,
            element: PhantomData,
        })
    }

    /// Adds a uniquely borrowed buffer that may be read and written.
    pub fn bind_read_write<T: BindingElement>(
        &mut self,
        buffer: &'buffer mut DeviceBuffer<T>,
    ) -> Result<ReadWrite<T>, CommandError> {
        self.check_buffer(buffer)?;
        let slot = self.reserve_slot()?;
        let ErasedLease(lease) = T::__erase_read_write(buffer);
        self.leases.push(lease);
        Ok(ReadWrite {
            set_id: self.set_id,
            slot,
            element: PhantomData,
        })
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

pub(crate) enum Access<'buffer, T: DeviceCopy> {
    Read(&'buffer DeviceBuffer<T>),
    ReadWrite(&'buffer mut DeviceBuffer<T>),
}

pub(crate) enum Lease<'buffer> {
    F32(Access<'buffer, f32>),
    F16(Access<'buffer, f16>),
    Bf16(Access<'buffer, bf16>),
    U8(Access<'buffer, u8>),
}

mod sealed {
    pub trait Sealed {}
}

/// A device-buffer element type accepted by the command binding arena.
///
/// The trait is sealed so every handle can be resolved without type erasure,
/// downcasts, or unsafe pointer casts.
pub trait BindingElement: DeviceCopy + sealed::Sealed + Sized {
    #[doc(hidden)]
    fn __erase_read<'buffer>(buffer: &'buffer DeviceBuffer<Self>) -> ErasedLease<'buffer>;

    #[doc(hidden)]
    fn __erase_read_write<'buffer>(buffer: &'buffer mut DeviceBuffer<Self>)
    -> ErasedLease<'buffer>;
}

/// Opaque erased storage for one binding.
///
/// This type exists only to keep the sealed [`BindingElement`] interface
/// visibility-correct. Its payload is private and cannot be forged downstream.
#[doc(hidden)]
pub struct ErasedLease<'buffer>(Lease<'buffer>);

pub(crate) trait ResolveElement: BindingElement {
    fn read<'lease>(lease: &'lease Lease<'_>) -> Result<&'lease DeviceBuffer<Self>, LeaseError>;

    fn write<'lease>(
        lease: &'lease mut Lease<'_>,
    ) -> Result<&'lease mut DeviceBuffer<Self>, LeaseError>;
}

pub(crate) enum LeaseError {
    ElementMismatch,
    ReadOnly,
}

macro_rules! impl_binding_element {
    ($ty:ty, $variant:ident) => {
        impl sealed::Sealed for $ty {}

        impl BindingElement for $ty {
            fn __erase_read<'buffer>(buffer: &'buffer DeviceBuffer<Self>) -> ErasedLease<'buffer> {
                ErasedLease(Lease::$variant(Access::Read(buffer)))
            }

            fn __erase_read_write<'buffer>(
                buffer: &'buffer mut DeviceBuffer<Self>,
            ) -> ErasedLease<'buffer> {
                ErasedLease(Lease::$variant(Access::ReadWrite(buffer)))
            }
        }

        impl ResolveElement for $ty {
            fn read<'lease>(
                lease: &'lease Lease<'_>,
            ) -> Result<&'lease DeviceBuffer<Self>, LeaseError> {
                match lease {
                    Lease::$variant(Access::Read(buffer)) => Ok(*buffer),
                    Lease::$variant(Access::ReadWrite(buffer)) => Ok(&**buffer),
                    _ => Err(LeaseError::ElementMismatch),
                }
            }

            fn write<'lease>(
                lease: &'lease mut Lease<'_>,
            ) -> Result<&'lease mut DeviceBuffer<Self>, LeaseError> {
                match lease {
                    Lease::$variant(Access::ReadWrite(buffer)) => Ok(&mut **buffer),
                    Lease::$variant(Access::Read(_)) => Err(LeaseError::ReadOnly),
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
    set_id: u64,
    slot: usize,
    element: PhantomData<fn() -> T>,
}

/// Opaque read-write access to one checked binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadWrite<T: BindingElement> {
    set_id: u64,
    slot: usize,
    element: PhantomData<fn() -> T>,
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
    set_id: u64,
    slot: usize,
    element: PhantomData<fn() -> T>,
}

/// A stream-ordered sequence of commands with one final completion fence.
pub struct CommandScope<'queue, 'buffer> {
    queue: Option<&'queue mut CommandQueue>,
    bindings: Option<CheckedBindings<'buffer>>,
    submitted: usize,
    submission_error: Option<DriverError>,
    finished: bool,
}

impl<'queue, 'buffer> CommandScope<'queue, 'buffer> {
    /// Records one final fence and transfers all leases to the completion.
    pub fn finish(mut self) -> CommandCompletion<'queue, 'buffer> {
        let queue = self.queue.take().expect("live command scope has a queue");
        let bindings = self
            .bindings
            .take()
            .expect("live command scope has bindings");
        let record_error = if self.submitted == 0 || self.submission_error.is_some() {
            None
        } else {
            queue.completion_event.record(&queue.stream).err()
        };

        self.finished = true;
        CommandCompletion {
            queue: Some(queue),
            bindings: Some(bindings),
            submitted: self.submitted,
            submission_error: self.submission_error,
            record_error,
            poll_error: None,
            complete: false,
        }
    }

    pub(crate) fn ensure_launch_capacity(&self) -> Result<(), CommandError> {
        if self.submission_error.is_some() {
            return Err(CommandError::ScopePoisoned);
        }
        let queue = self.queue.as_ref().expect("live command scope has a queue");
        if self.submitted == queue.max_launches {
            Err(CommandError::LaunchCapacityExceeded {
                capacity: queue.max_launches,
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn resolve_triplet<T: ResolveElement>(
        &mut self,
        input: Read<T>,
        weight: Read<T>,
        output: Write<T>,
    ) -> Result<ResolvedTriplet<'_, T>, CommandError> {
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        for handle in [input.set_id, weight.set_id, output.set_id] {
            if handle != bindings.set_id {
                return Err(CommandError::BindingSetMismatch);
            }
        }
        if input.slot == weight.slot || input.slot == output.slot || weight.slot == output.slot {
            return Err(CommandError::AliasedOperands);
        }
        for slot in [input.slot, weight.slot, output.slot] {
            if slot >= bindings.leases.len() {
                return Err(CommandError::BindingSlotOutOfRange {
                    slot,
                    bindings: bindings.leases.len(),
                });
            }
        }

        let [input_lease, weight_lease, output_lease] = bindings
            .leases
            .get_disjoint_mut([input.slot, weight.slot, output.slot])
            .expect("validated binding slots are pairwise disjoint");
        let input_buffer =
            T::read(input_lease).map_err(|error| map_lease_error(error, input.slot))?;
        let weight_buffer =
            T::read(weight_lease).map_err(|error| map_lease_error(error, weight.slot))?;
        let output_buffer =
            T::write(output_lease).map_err(|error| map_lease_error(error, output.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedTriplet {
            stream: &queue.stream,
            input: input_buffer,
            weight: weight_buffer,
            output: output_buffer,
        })
    }

    pub(crate) fn retain_launch(&mut self, function: CudaFunction) {
        let queue = self.queue.as_mut().expect("live command scope has a queue");
        debug_assert!(queue.retained_functions.len() < queue.max_launches);
        queue.retained_functions.push(function);
        self.submitted += 1;
    }

    pub(crate) fn retain_failed_launch(&mut self, function: CudaFunction, error: DriverError) {
        let queue = self.queue.as_mut().expect("live command scope has a queue");
        debug_assert!(queue.retained_functions.len() < queue.max_launches);
        queue.retained_functions.push(function);
        self.submitted += 1;
        self.submission_error = Some(error);
    }
}

impl Drop for CommandScope<'_, '_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let Some(queue) = self.queue.as_mut() else {
            return;
        };

        if self.submitted > 0 {
            let synchronize_error = synchronize_stream_or_abort(queue);
            if self.submission_error.is_some() || synchronize_error.is_some() {
                queue.poisoned = true;
            }
            if let Some(error) = self.submission_error.or(synchronize_error) {
                queue.stream.context().record_err::<()>(Err(error));
            }
        }
        queue.retained_functions.clear();
    }
}

pub(crate) struct ResolvedTriplet<'scope, T: BindingElement> {
    pub(crate) stream: &'scope CudaStream,
    pub(crate) input: &'scope DeviceBuffer<T>,
    pub(crate) weight: &'scope DeviceBuffer<T>,
    pub(crate) output: &'scope mut DeviceBuffer<T>,
}

/// The final fence and all leases retained by a completed command scope.
#[must_use = "dropping the completion waits before releasing CUDA resources"]
pub struct CommandCompletion<'queue, 'buffer> {
    queue: Option<&'queue mut CommandQueue>,
    bindings: Option<CheckedBindings<'buffer>>,
    submitted: usize,
    submission_error: Option<DriverError>,
    record_error: Option<DriverError>,
    poll_error: Option<DriverError>,
    complete: bool,
}

impl<'queue, 'buffer> CommandCompletion<'queue, 'buffer> {
    /// Returns whether all submitted commands have completed without blocking.
    pub fn is_complete(&mut self) -> Result<bool, CommandError> {
        if self.submitted == 0 {
            return Ok(true);
        }
        if let Some(error) = self.submission_error {
            return Err(error.into());
        }
        if let Some(error) = self.record_error {
            return Err(error.into());
        }
        if let Some(error) = self.poll_error {
            return Err(error.into());
        }
        let queue = self.queue.as_ref().expect("live completion has a queue");
        match queue.completion_event.query() {
            Ok(complete) => Ok(complete),
            Err(error) => {
                self.poll_error = Some(error);
                Err(error.into())
            }
        }
    }

    /// Waits once and returns the reusable checked bindings.
    pub fn wait(mut self) -> Result<CheckedBindings<'buffer>, CommandError> {
        match self.settle() {
            None => {
                self.complete = true;
                self.queue
                    .as_mut()
                    .expect("live completion has a queue")
                    .retained_functions
                    .clear();
                Ok(self.bindings.take().expect("live completion has bindings"))
            }
            Some(error) => {
                self.complete = true;
                let queue = self.queue.as_mut().expect("live completion has a queue");
                queue.poisoned = true;
                queue.retained_functions.clear();
                queue.stream.context().record_err::<()>(Err(error));
                self.bindings.take();
                Err(error.into())
            }
        }
    }

    /// Number of commands covered by this one completion fence.
    pub const fn submitted(&self) -> usize {
        self.submitted
    }

    fn settle(&self) -> Option<DriverError> {
        if self.submitted == 0 {
            return self.submission_error;
        }
        let queue = self.queue.as_ref().expect("live completion has a queue");
        if let Some(submission_error) = self.submission_error {
            let _ = synchronize_stream_or_abort(queue);
            return Some(submission_error);
        }
        if let Some(record_error) = self.record_error {
            let _ = synchronize_stream_or_abort(queue);
            return Some(record_error);
        }
        if let Some(poll_error) = self.poll_error {
            let _ = synchronize_stream_or_abort(queue);
            return Some(poll_error);
        }
        match queue.completion_event.synchronize() {
            Ok(()) => None,
            Err(event_error) => {
                let _ = synchronize_stream_or_abort(queue);
                Some(event_error)
            }
        }
    }
}

impl Drop for CommandCompletion<'_, '_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let result = self.settle();
        let queue = self.queue.as_mut().expect("live completion has a queue");
        if let Some(error) = result {
            queue.poisoned = true;
            queue.stream.context().record_err::<()>(Err(error));
        }
        queue.retained_functions.clear();
        self.complete = true;
    }
}

fn map_lease_error(error: LeaseError, slot: usize) -> CommandError {
    match error {
        LeaseError::ElementMismatch => CommandError::BindingTypeMismatch { slot },
        LeaseError::ReadOnly => CommandError::BindingIsReadOnly { slot },
    }
}

fn fresh_id() -> Result<u64, CommandError> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| CommandError::IdentifierSpaceExhausted)
}

fn synchronize_stream_or_abort(queue: &CommandQueue) -> Option<DriverError> {
    match queue.stream.synchronize() {
        Ok(()) => None,
        Err(stream_error) => match queue.stream.context().synchronize() {
            Ok(()) => Some(stream_error),
            Err(context_error) => abort_after_sync_failure(stream_error, context_error),
        },
    }
}

fn abort_after_sync_failure(stream_error: DriverError, context_error: DriverError) -> ! {
    eprintln!(
        "loom-infer-cuda cannot confirm CUDA quiescence after stream and context synchronization \
         failed; aborting to preserve resource safety: stream={stream_error}; \
         context={context_error}"
    );
    std::process::abort()
}

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("command queues require capacity for at least one launch")]
    ZeroLaunchCapacity,
    #[error("checked bindings require capacity for at least one resource")]
    ZeroBindingCapacity,
    #[error("command queue identifier space is exhausted")]
    IdentifierSpaceExhausted,
    #[error("the checked bindings were created by a different command queue")]
    BindingsQueueMismatch,
    #[error("the command queue is poisoned by an earlier completion failure")]
    QueuePoisoned,
    #[error("the command scope is poisoned by a CUDA launch failure")]
    ScopePoisoned,
    #[error("checked binding capacity {capacity} is exhausted")]
    BindingCapacityExceeded { capacity: usize },
    #[error(
        "buffer belongs to CUDA device {buffer_device}, but the queue stream belongs to device {stream_device}"
    )]
    BufferContextMismatch {
        buffer_device: usize,
        stream_device: usize,
    },
    #[error("the resource handle belongs to a different checked binding set")]
    BindingSetMismatch,
    #[error("binding slot {slot} is out of range for {bindings} bindings")]
    BindingSlotOutOfRange { slot: usize, bindings: usize },
    #[error("binding slot {slot} is read-only")]
    BindingIsReadOnly { slot: usize },
    #[error("binding slot {slot} has a different element type than its resource handle")]
    BindingTypeMismatch { slot: usize },
    #[error("one operator cannot alias its input, weight, or output bindings")]
    AliasedOperands,
    #[error("command scope launch capacity {capacity} is exhausted")]
    LaunchCapacityExceeded { capacity: usize },
    #[error(transparent)]
    Driver(#[from] DriverError),
}
