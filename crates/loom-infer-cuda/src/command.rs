//! Checked CUDA resource bindings and stream-ordered command submission.

#![forbid(unsafe_code)]

use cuda_core::{CudaEvent, CudaFunction, CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use half::{bf16, f16};
use std::any::Any;
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A reusable submission queue for one exact CUDA stream.
///
/// The queue preallocates its completion event and resource-retention storage.
/// Rust's mutable borrow rules prevent a second scope from re-recording the
/// event while an earlier completion is still alive.
pub struct CommandQueue {
    id: u64,
    stream: Arc<CudaStream>,
    completion_event: CudaEvent,
    retained_resources: Vec<RetainedResource>,
    max_commands: usize,
    poisoned: bool,
}

impl CommandQueue {
    /// Creates a queue for `stream` with storage for at most `max_commands`
    /// commands per scope.
    pub fn new(stream: Arc<CudaStream>, max_commands: usize) -> Result<Self, CommandError> {
        if max_commands == 0 {
            return Err(CommandError::ZeroCommandCapacity);
        }

        let id = fresh_id()?;
        let completion_event = stream.context().new_event(None)?;
        Ok(Self {
            id,
            stream,
            completion_event,
            retained_resources: Vec::with_capacity(max_commands),
            max_commands,
            poisoned: false,
        })
    }

    /// Returns the exact stream used by every scope from this queue.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub const fn max_commands(&self) -> usize {
        self.max_commands
    }

    /// Creates reusable checked binding storage outside the enqueue path.
    pub fn bindings(&self, capacity: usize) -> Result<CheckedBindings, CommandError> {
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
    pub fn begin<'queue>(
        &'queue mut self,
        bindings: CheckedBindings,
    ) -> Result<CommandScope<'queue>, CommandError> {
        if self.poisoned {
            return Err(CommandError::QueuePoisoned);
        }
        if bindings.queue_id != self.id
            || bindings.stream.cu_stream() != self.stream.cu_stream()
            || bindings.stream.context().cu_ctx() != self.stream.context().cu_ctx()
        {
            return Err(CommandError::BindingsQueueMismatch);
        }
        if !self.retained_resources.is_empty() {
            self.poisoned = true;
            return Err(CommandError::QueuePoisoned);
        }

        Ok(CommandScope {
            queue: Some(self),
            bindings: Some(bindings),
            scope_id: fresh_id()?,
            submitted: 0,
            submission_error: None,
            finished: false,
        })
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        if self.retained_resources.is_empty() {
            return;
        }

        let synchronize_error = synchronize_stream_or_abort(&self.stream);
        self.retained_resources.clear();
        if let Some(error) = synchronize_error {
            eprintln!(
                "loom-infer-cuda command queue synchronized after a stream error during drop: \
                 {error}"
            );
        }
    }
}

/// A reusable set of checked, heterogeneous buffer leases.
///
/// Moving this value into a [`CommandScope`] retains shared read allocations and
/// transfers writable allocations until the returned completion is settled.
/// Owning every writable allocation makes the asynchronous contract safe even
/// if a scope or completion is leaked.
pub struct CheckedBindings {
    queue_id: u64,
    set_id: u64,
    stream: Arc<CudaStream>,
    leases: Vec<Lease>,
    capacity: usize,
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
    /// This is intended after [`CommandCompletion::wait`] or
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
pub struct CommandScope<'queue> {
    queue: Option<&'queue mut CommandQueue>,
    bindings: Option<CheckedBindings>,
    scope_id: u64,
    submitted: usize,
    submission_error: Option<SubmissionError>,
    finished: bool,
}

impl<'queue> CommandScope<'queue> {
    /// Records one final fence and transfers all bindings to the completion.
    pub fn finish(mut self) -> CommandCompletion<'queue> {
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

    pub(crate) fn prepare_command(&self) -> Result<CommandPermit, CommandError> {
        if self.submission_error.is_some() {
            return Err(CommandError::ScopePoisoned);
        }
        let queue = self.queue.as_ref().expect("live command scope has a queue");
        if self.submitted >= queue.max_commands {
            Err(CommandError::CommandCapacityExceeded {
                capacity: queue.max_commands,
            })
        } else {
            Ok(CommandPermit {
                queue_id: queue.id,
                scope_id: self.scope_id,
                submission_index: self.submitted,
            })
        }
    }

    pub(crate) const fn submitted_commands(&self) -> usize {
        self.submitted
    }

    pub(crate) fn capture_error(&self) -> Option<CommandError> {
        self.submission_error.map(Into::into)
    }

    pub(crate) fn finish_capture(mut self) -> CapturedCommandSet {
        assert!(
            self.submission_error.is_none() && self.submitted > 0,
            "only a non-empty healthy command scope may become a captured graph"
        );
        let queue = self.queue.take().expect("live command scope has a queue");
        let bindings = self
            .bindings
            .take()
            .expect("live command scope has bindings");
        let resources = std::mem::replace(
            &mut queue.retained_resources,
            Vec::with_capacity(queue.max_commands),
        );
        self.finished = true;
        CapturedCommandSet {
            stream: queue.stream.clone(),
            bindings,
            resources,
            submitted: self.submitted,
        }
    }

    pub(crate) fn resolve_rrw<A, B, C>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Write<C>,
    ) -> Result<ResolvedRrw<'_, A, B, C>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
    {
        self.validate_resolve_request(
            &[first.set_id, second.set_id, third.set_id],
            &[first.slot, second.slot, third.slot],
        )?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        let [first_lease, second_lease, third_lease] = bindings
            .leases
            .get_disjoint_mut([first.slot, second.slot, third.slot])
            .expect("validated binding slots are pairwise disjoint");
        let first_buffer =
            A::read(first_lease).map_err(|error| map_lease_error(error, first.slot))?;
        let second_buffer =
            B::read(second_lease).map_err(|error| map_lease_error(error, second.slot))?;
        let third_buffer =
            C::write(third_lease).map_err(|error| map_lease_error(error, third.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrw {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
        })
    }

    pub(crate) fn resolve_rrww<A, B, C, D>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Write<C>,
        fourth: Write<D>,
    ) -> Result<ResolvedRrww<'_, A, B, C, D>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
    {
        let slots = [first.slot, second.slot, third.slot, fourth.slot];
        self.validate_resolve_request(
            &[first.set_id, second.set_id, third.set_id, fourth.set_id],
            &slots,
        )?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        let [first_lease, second_lease, third_lease, fourth_lease] = bindings
            .leases
            .get_disjoint_mut(slots)
            .expect("validated binding slots are pairwise disjoint");
        let first_buffer =
            A::read(first_lease).map_err(|error| map_lease_error(error, first.slot))?;
        let second_buffer =
            B::read(second_lease).map_err(|error| map_lease_error(error, second.slot))?;
        let third_buffer =
            C::write(third_lease).map_err(|error| map_lease_error(error, third.slot))?;
        let fourth_buffer =
            D::write(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
        })
    }

    pub(crate) fn resolve_rrrww<A, B, C, D, E>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Write<D>,
        fifth: Write<E>,
    ) -> Result<ResolvedRrrww<'_, A, B, C, D, E>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
    {
        let slots = [first.slot, second.slot, third.slot, fourth.slot, fifth.slot];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
            ],
            &slots,
        )?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        let [
            first_lease,
            second_lease,
            third_lease,
            fourth_lease,
            fifth_lease,
        ] = bindings
            .leases
            .get_disjoint_mut(slots)
            .expect("validated binding slots are pairwise disjoint");
        let first_buffer =
            A::read(first_lease).map_err(|error| map_lease_error(error, first.slot))?;
        let second_buffer =
            B::read(second_lease).map_err(|error| map_lease_error(error, second.slot))?;
        let third_buffer =
            C::read(third_lease).map_err(|error| map_lease_error(error, third.slot))?;
        let fourth_buffer =
            D::write(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::write(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
        })
    }

    fn validate_resolve_request(
        &self,
        set_ids: &[u64],
        slots: &[usize],
    ) -> Result<(), CommandError> {
        let bindings = self
            .bindings
            .as_ref()
            .expect("live command scope has bindings");
        validate_binding_request(bindings.set_id, bindings.leases.len(), set_ids, slots)
    }

    pub(crate) fn record_cuda_submission(&mut self, permit: CommandPermit, function: CudaFunction) {
        self.record_submission(
            permit,
            RetainedResource::Kernel {
                _function: function,
            },
        );
    }

    pub(crate) fn record_failed_cuda_submission(
        &mut self,
        permit: CommandPermit,
        function: CudaFunction,
        error: DriverError,
    ) {
        self.record_submission(
            permit,
            RetainedResource::Kernel {
                _function: function,
            },
        );
        self.queue
            .as_mut()
            .expect("live command scope has a queue")
            .poisoned = true;
        self.submission_error = Some(SubmissionError::Driver(error));
    }

    pub(crate) fn record_external_submission<T>(&mut self, permit: CommandPermit, resource: Arc<T>)
    where
        T: Any + Send + Sync,
    {
        self.record_submission(
            permit,
            RetainedResource::External {
                _resource: resource,
            },
        );
    }

    pub(crate) fn record_failed_external_submission<T>(
        &mut self,
        permit: CommandPermit,
        resource: Arc<T>,
        error: ExternalCommandError,
    ) where
        T: Any + Send + Sync,
    {
        self.record_submission(
            permit,
            RetainedResource::External {
                _resource: resource,
            },
        );
        self.queue
            .as_mut()
            .expect("live command scope has a queue")
            .poisoned = true;
        self.submission_error = Some(SubmissionError::External(error));
    }

    pub(crate) fn record_preflight_driver_failure(&mut self, error: DriverError) {
        self.queue
            .as_mut()
            .expect("live command scope has a queue")
            .poisoned = true;
        self.submission_error = Some(SubmissionError::Driver(error));
    }

    fn record_submission(&mut self, permit: CommandPermit, resource: RetainedResource) {
        let queue = self.queue.as_mut().expect("live command scope has a queue");
        if permit.queue_id != queue.id
            || permit.scope_id != self.scope_id
            || permit.submission_index != self.submitted
            || queue.retained_resources.len() != self.submitted
            || self.submitted >= queue.max_commands
        {
            abort_after_bookkeeping_invariant();
        }
        queue.retained_resources.push(resource);
        self.submitted += 1;
    }
}

fn validate_binding_request(
    expected_set_id: u64,
    binding_count: usize,
    set_ids: &[u64],
    slots: &[usize],
) -> Result<(), CommandError> {
    if set_ids.iter().any(|&set_id| set_id != expected_set_id) {
        return Err(CommandError::BindingSetMismatch);
    }
    for (index, &slot) in slots.iter().enumerate() {
        if slots[..index].contains(&slot) {
            return Err(CommandError::DuplicateBindingSlot);
        }
        if slot >= binding_count {
            return Err(CommandError::BindingSlotOutOfRange {
                slot,
                bindings: binding_count,
            });
        }
    }
    Ok(())
}

/// A single-use proof that one command fits in the preallocated queue.
pub(crate) struct CommandPermit {
    queue_id: u64,
    scope_id: u64,
    submission_index: usize,
}

pub(crate) enum RetainedResource {
    Kernel {
        _function: CudaFunction,
    },
    External {
        _resource: Arc<dyn Any + Send + Sync>,
    },
}

pub(crate) struct CapturedCommandSet {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) bindings: CheckedBindings,
    pub(crate) resources: Vec<RetainedResource>,
    pub(crate) submitted: usize,
}

impl Drop for CommandScope<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let Some(queue) = self.queue.as_mut() else {
            return;
        };

        if self.submitted > 0 {
            let synchronize_error = synchronize_stream_or_abort(&queue.stream);
            if self.submission_error.is_some() || synchronize_error.is_some() {
                queue.poisoned = true;
            }
            let submission_driver_error = self
                .submission_error
                .and_then(SubmissionError::driver_error);
            queue.retained_resources.clear();
            if let Some(error) = submission_driver_error {
                queue.stream.context().record_err::<()>(Err(error));
            }
            if let Some(error) = synchronize_error {
                queue.stream.context().record_err::<()>(Err(error));
            }
        } else {
            queue.retained_resources.clear();
            if let Some(error) = self.submission_error {
                queue.poisoned = true;
                if let Some(error) = error.driver_error() {
                    queue.stream.context().record_err::<()>(Err(error));
                }
            }
        }
    }
}

pub(crate) struct ResolvedRrw<'scope, A: BindingElement, B: BindingElement, C: BindingElement> {
    pub(crate) stream: &'scope CudaStream,
    pub(crate) first: &'scope DeviceBuffer<A>,
    pub(crate) second: &'scope DeviceBuffer<B>,
    pub(crate) third: &'scope mut DeviceBuffer<C>,
}

pub(crate) struct ResolvedRrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
> {
    pub(crate) stream: &'scope CudaStream,
    pub(crate) first: &'scope DeviceBuffer<A>,
    pub(crate) second: &'scope DeviceBuffer<B>,
    pub(crate) third: &'scope mut DeviceBuffer<C>,
    pub(crate) fourth: &'scope mut DeviceBuffer<D>,
}

pub(crate) struct ResolvedRrrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
> {
    pub(crate) stream: &'scope CudaStream,
    pub(crate) first: &'scope DeviceBuffer<A>,
    pub(crate) second: &'scope DeviceBuffer<B>,
    pub(crate) third: &'scope DeviceBuffer<C>,
    pub(crate) fourth: &'scope mut DeviceBuffer<D>,
    pub(crate) fifth: &'scope mut DeviceBuffer<E>,
}

/// The final fence and all bindings retained by a completed command scope.
#[must_use = "dropping the completion waits before releasing CUDA resources"]
pub struct CommandCompletion<'queue> {
    queue: Option<&'queue mut CommandQueue>,
    bindings: Option<CheckedBindings>,
    submitted: usize,
    submission_error: Option<SubmissionError>,
    record_error: Option<DriverError>,
    poll_error: Option<DriverError>,
    complete: bool,
}

impl<'queue> CommandCompletion<'queue> {
    /// Returns whether all submitted commands have completed without blocking.
    pub fn is_complete(&mut self) -> Result<bool, CommandError> {
        if let Some(error) = self.submission_error {
            return Err(error.into());
        }
        if let Some(error) = self.record_error {
            return Err(error.into());
        }
        if let Some(error) = self.poll_error {
            return Err(error.into());
        }
        if self.submitted == 0 {
            return Ok(true);
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
    pub fn wait(mut self) -> Result<CheckedBindings, CommandError> {
        match self.settle() {
            None => {
                self.complete = true;
                self.queue
                    .as_mut()
                    .expect("live completion has a queue")
                    .retained_resources
                    .clear();
                Ok(self.bindings.take().expect("live completion has bindings"))
            }
            Some(failure) => {
                self.complete = true;
                let queue = self.queue.as_mut().expect("live completion has a queue");
                queue.poisoned = true;
                queue.retained_resources.clear();
                record_settlement_errors(queue, failure);
                self.bindings.take();
                Err(failure.command_error())
            }
        }
    }

    /// Number of commands covered by this one completion fence.
    pub const fn submitted(&self) -> usize {
        self.submitted
    }

    fn settle(&self) -> Option<SettlementFailure> {
        if self.submitted == 0 {
            return self.submission_error.map(|reported| SettlementFailure {
                reported,
                synchronize_error: None,
            });
        }
        let queue = self.queue.as_ref().expect("live completion has a queue");
        if let Some(submission_error) = self.submission_error {
            return Some(SettlementFailure {
                reported: submission_error,
                synchronize_error: synchronize_stream_or_abort(&queue.stream),
            });
        }
        if let Some(record_error) = self.record_error {
            return Some(SettlementFailure {
                reported: SubmissionError::Driver(record_error),
                synchronize_error: synchronize_stream_or_abort(&queue.stream),
            });
        }
        if let Some(poll_error) = self.poll_error {
            return Some(SettlementFailure {
                reported: SubmissionError::Driver(poll_error),
                synchronize_error: synchronize_stream_or_abort(&queue.stream),
            });
        }
        match queue.completion_event.synchronize() {
            Ok(()) => None,
            Err(event_error) => Some(SettlementFailure {
                reported: SubmissionError::Driver(event_error),
                synchronize_error: synchronize_stream_or_abort(&queue.stream),
            }),
        }
    }
}

impl Drop for CommandCompletion<'_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let result = self.settle();
        let queue = self.queue.as_mut().expect("live completion has a queue");
        if let Some(failure) = result {
            queue.poisoned = true;
            queue.retained_resources.clear();
            record_settlement_errors(queue, failure);
        } else {
            queue.retained_resources.clear();
        }
        self.complete = true;
    }
}

fn map_lease_error(error: LeaseError, slot: usize) -> CommandError {
    match error {
        LeaseError::ElementMismatch => CommandError::BindingTypeMismatch { slot },
        LeaseError::ReadOnly => CommandError::BindingIsReadOnly { slot },
        LeaseError::Vacant => CommandError::BindingSlotVacant { slot },
    }
}

fn fresh_id() -> Result<u64, CommandError> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| CommandError::IdentifierSpaceExhausted)
}

pub(crate) fn synchronize_stream_or_abort(stream: &CudaStream) -> Option<DriverError> {
    match stream.synchronize() {
        Ok(()) => None,
        Err(stream_error) => match stream.context().synchronize() {
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

fn abort_after_bookkeeping_invariant() -> ! {
    eprintln!(
        "loom-infer-cuda detected an internal command-accounting violation after GPU submission; \
         aborting to preserve resource safety"
    );
    std::process::abort()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("{provider} command submission failed with status {status}")]
pub struct ExternalCommandError {
    provider: &'static str,
    status: i32,
}

impl ExternalCommandError {
    pub const fn new(provider: &'static str, status: i32) -> Self {
        Self { provider, status }
    }

    pub const fn provider(self) -> &'static str {
        self.provider
    }

    pub const fn status(self) -> i32 {
        self.status
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmissionError {
    Driver(DriverError),
    External(ExternalCommandError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SettlementFailure {
    reported: SubmissionError,
    synchronize_error: Option<DriverError>,
}

impl SettlementFailure {
    fn command_error(self) -> CommandError {
        self.reported.into()
    }
}

fn record_settlement_errors(queue: &CommandQueue, failure: SettlementFailure) {
    if let Some(error) = failure.synchronize_error {
        queue.stream.context().record_err::<()>(Err(error));
    }
    if let Some(error) = failure.reported.driver_error() {
        queue.stream.context().record_err::<()>(Err(error));
    }
}

impl SubmissionError {
    const fn driver_error(self) -> Option<DriverError> {
        match self {
            Self::Driver(error) => Some(error),
            Self::External(_) => None,
        }
    }
}

impl From<SubmissionError> for CommandError {
    fn from(error: SubmissionError) -> Self {
        match error {
            SubmissionError::Driver(error) => Self::Driver(error),
            SubmissionError::External(error) => Self::External(error),
        }
    }
}

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("command queues require capacity for at least one command")]
    ZeroCommandCapacity,
    #[error("checked bindings require capacity for at least one resource")]
    ZeroBindingCapacity,
    #[error("command queue identifier space is exhausted")]
    IdentifierSpaceExhausted,
    #[error("the checked bindings were created by a different command queue")]
    BindingsQueueMismatch,
    #[error("the command queue is poisoned by an earlier completion failure")]
    QueuePoisoned,
    #[error("the command scope is poisoned by an earlier submission failure")]
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
    #[error("binding slot {slot} has already been removed")]
    BindingSlotVacant { slot: usize },
    #[error("binding slot {slot} has a different element type than its resource handle")]
    BindingTypeMismatch { slot: usize },
    #[error("one command cannot use the same binding slot for multiple operands")]
    DuplicateBindingSlot,
    #[error("command scope capacity {capacity} is exhausted")]
    CommandCapacityExceeded { capacity: usize },
    #[error(transparent)]
    External(#[from] ExternalCommandError),
    #[error(transparent)]
    Driver(#[from] DriverError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_request_accepts_matching_distinct_slots() {
        assert!(validate_binding_request(7, 3, &[7, 7, 7], &[0, 1, 2]).is_ok());
    }

    #[test]
    fn binding_request_rejects_a_handle_from_another_set() {
        assert!(matches!(
            validate_binding_request(7, 3, &[7, 8, 7], &[0, 1, 2]),
            Err(CommandError::BindingSetMismatch)
        ));
    }

    #[test]
    fn binding_request_rejects_duplicate_slots_before_resolution() {
        assert!(matches!(
            validate_binding_request(7, 3, &[7, 7, 7], &[0, 1, 1]),
            Err(CommandError::DuplicateBindingSlot)
        ));
    }

    #[test]
    fn binding_request_reports_the_first_out_of_range_slot() {
        assert!(matches!(
            validate_binding_request(7, 3, &[7, 7], &[0, 3]),
            Err(CommandError::BindingSlotOutOfRange {
                slot: 3,
                bindings: 3,
            })
        ));
    }
}
