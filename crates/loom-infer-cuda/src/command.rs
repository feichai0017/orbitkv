//! Checked CUDA resource bindings and stream-ordered command submission.

#![forbid(unsafe_code)]

use cuda_core::{CudaEvent, CudaFunction, CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use half::{bf16, f16};
use std::any::Any;
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

        let synchronize_error = synchronize_stream_or_abort(self);
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
    scope_id: u64,
    submitted: usize,
    submission_error: Option<SubmissionError>,
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
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        for handle in [first.set_id, second.set_id, third.set_id] {
            if handle != bindings.set_id {
                return Err(CommandError::BindingSetMismatch);
            }
        }
        if first.slot == second.slot || first.slot == third.slot || second.slot == third.slot {
            return Err(CommandError::DuplicateBindingSlot);
        }
        for slot in [first.slot, second.slot, third.slot] {
            if slot >= bindings.leases.len() {
                return Err(CommandError::BindingSlotOutOfRange {
                    slot,
                    bindings: bindings.leases.len(),
                });
            }
        }

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
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        for handle in [first.set_id, second.set_id, third.set_id, fourth.set_id] {
            if handle != bindings.set_id {
                return Err(CommandError::BindingSetMismatch);
            }
        }
        let slots = [first.slot, second.slot, third.slot, fourth.slot];
        for (index, slot) in slots.iter().enumerate() {
            if slots[..index].contains(slot) {
                return Err(CommandError::DuplicateBindingSlot);
            }
            if *slot >= bindings.leases.len() {
                return Err(CommandError::BindingSlotOutOfRange {
                    slot: *slot,
                    bindings: bindings.leases.len(),
                });
            }
        }

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
        self.submission_error = Some(SubmissionError::External(error));
    }

    pub(crate) fn record_preflight_driver_failure(&mut self, error: DriverError) {
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

/// A single-use proof that one command fits in the preallocated queue.
pub(crate) struct CommandPermit {
    queue_id: u64,
    scope_id: u64,
    submission_index: usize,
}

enum RetainedResource {
    Kernel {
        _function: CudaFunction,
    },
    External {
        _resource: Arc<dyn Any + Send + Sync>,
    },
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

/// The final fence and all leases retained by a completed command scope.
#[must_use = "dropping the completion waits before releasing CUDA resources"]
pub struct CommandCompletion<'queue, 'buffer> {
    queue: Option<&'queue mut CommandQueue>,
    bindings: Option<CheckedBindings<'buffer>>,
    submitted: usize,
    submission_error: Option<SubmissionError>,
    record_error: Option<DriverError>,
    poll_error: Option<DriverError>,
    complete: bool,
}

impl<'queue, 'buffer> CommandCompletion<'queue, 'buffer> {
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
    pub fn wait(mut self) -> Result<CheckedBindings<'buffer>, CommandError> {
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
                synchronize_error: synchronize_stream_or_abort(queue),
            });
        }
        if let Some(record_error) = self.record_error {
            return Some(SettlementFailure {
                reported: SubmissionError::Driver(record_error),
                synchronize_error: synchronize_stream_or_abort(queue),
            });
        }
        if let Some(poll_error) = self.poll_error {
            return Some(SettlementFailure {
                reported: SubmissionError::Driver(poll_error),
                synchronize_error: synchronize_stream_or_abort(queue),
            });
        }
        match queue.completion_event.synchronize() {
            Ok(()) => None,
            Err(event_error) => Some(SettlementFailure {
                reported: SubmissionError::Driver(event_error),
                synchronize_error: synchronize_stream_or_abort(queue),
            }),
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
        match self.synchronize_error {
            Some(error) => CommandError::Driver(error),
            None => self.reported.into(),
        }
    }
}

fn record_settlement_errors(queue: &CommandQueue, failure: SettlementFailure) {
    if let Some(error) = failure.reported.driver_error() {
        queue.stream.context().record_err::<()>(Err(error));
    }
    if let Some(error) = failure.synchronize_error {
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
